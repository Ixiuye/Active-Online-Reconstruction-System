"""端到端 demo：体素-高斯联合表征 + 体素几何约束 + 几何不确定度驱动的主动探索。

流程：
1. 解析式室内场景生成 GT RGB-D；
2. 深度图融合成 TSDF 体素地图；
3. 在表面体素上生成锚定高斯（体素作为表征骨架）；
4. 用体素损失（表面距离/法向一致/自由空间）约束高斯几何；
5. 用“未知 frontier + 低置信表面”的可见量作为 NBV 效用，主动补看；
6. 对比：纯光度 3DGS / +深度监督 / +体素约束，以及体素初始化未训练。

运行：
    python run_demo.py             # 完整流程
    python run_demo.py --quick     # 快速冒烟
"""
from __future__ import annotations

import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import cKDTree

from demo.active import candidate_poses, sample_utility_targets, select_nbv
from demo.gaussians import VoxelGaussians
from demo.losses import huber, voxel_constraint_losses
from demo.renderer import render_rays
from demo.rl_active import TARGET_COUNTS, select_nbv_rl
from demo.scene import ROOM_MAX, ROOM_MIN, TRUNC, VOXEL_SIZE, DemoScene
from demo.utils import look_at, pixel_rays
from demo.voxel_map import DenseTSDF

W_IMG, H_IMG = 128, 96
K = (110.0, 110.0, W_IMG / 2.0, H_IMG / 2.0)
DEPTH_NOISE = 0.0   # 由 --noise 设置：采集视角时逐像素加 σ(m) 深度噪声（0=无噪声）


# ---------------------------------------------------------------- 数据视图
class View:
    def __init__(self, depth, rgb, R, t, device):
        self.depth = depth
        self.rgb = rgb
        self.R = R
        self.t = t
        center, dirs = pixel_rays(R, t, K, H_IMG, W_IMG)
        self.center_np = center
        self.dirs_np = dirs
        self.center_t = torch.as_tensor(center, device=device)
        self.dirs_t = torch.as_tensor(dirs, device=device)
        self.rgb_t = torch.as_tensor(rgb.reshape(-1, 3), device=device)
        self.depth_t = torch.as_tensor(depth.reshape(-1), device=device)


def render_scene_view(scene, pose, device):
    R, t = pose
    depth, rgb, _ = scene.render_view(R, t, K, H_IMG, W_IMG)
    if DEPTH_NOISE > 0:
        hit = depth > 0
        noise = np.random.normal(0.0, DEPTH_NOISE, depth.shape).astype(np.float32)
        depth = np.where(hit, np.maximum(depth + noise, 0.05), 0.0).astype(np.float32)
    return View(depth, rgb, R, t, device)


# ---------------------------------------------------------------- 训练
def train_model(model, views, vmap, mode, iters, device, seed=0, lam=1.0,
                rays=512, verbose=False):
    opt = model.make_optimizer()
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    n_pix = H_IMG * W_IMG
    r = min(rays, n_pix)
    t0 = time.time()

    for it in range(iters):
        view = views[int(torch.randint(len(views), (1,)).item())]
        pix = torch.randint(0, n_pix, (r,), device=device, generator=gen)
        origins = view.center_t.unsqueeze(0).expand(r, 3)
        dirs = view.dirs_t[pix]
        gt_rgb = view.rgb_t[pix]
        gt_depth = view.depth_t[pix]

        pred_rgb, pred_depth, acc = render_rays(*model.properties(), origins, dirs)
        loss_rgb = ((pred_rgb - gt_rgb) ** 2).mean()

        if mode == "photo":
            loss = loss_rgb
        else:
            mask = (gt_depth > 0) & (acc > 0.25)
            if mask.any():
                loss_depth = huber(pred_depth - gt_depth, 0.05)[mask].mean()
            else:
                loss_depth = pred_depth.sum() * 0.0
            loss = loss_rgb + loss_depth

        if mode == "voxel":
            ls, ln, lf, lr = voxel_constraint_losses(model, vmap, TRUNC, n_sample=1024)
            loss = loss + lam * (1.2 * ls + 0.5 * ln + 1.5 * lf + 0.05 * lr)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if verbose and (it + 1) % 100 == 0:
            print(f"    it {it + 1}/{iters}  loss={loss.item():.4f}")

    return time.time() - t0


# ---------------------------------------------------------------- 评估
def project_to_surface(scene, p, iters=4):
    for _ in range(iters):
        s = scene.sdf(p)
        n = scene.gradient(p)
        p = p - s[:, None] * n
    return p


def evaluate(model, vmap, scene, gt_tree, held_view, gt_surface_mask):
    mu = model.means().detach().cpu().numpy()
    if len(mu) == 0:
        return {}
    d_nn, _ = gt_tree.query(mu)
    p_surf = project_to_surface(scene, mu)
    n_gt = scene.gradient(p_surf)
    n_g = model.normals().detach().cpu().numpy()
    cos = np.clip(np.abs((n_g * n_gt).sum(-1)), 0.0, 1.0)
    ang = np.degrees(np.arccos(cos))

    tv = vmap.query_np(mu)
    wv = vmap.query_weight_np(mu)
    viol = float(((tv > 0.5) & (wv > 0.5)).mean())

    with torch.no_grad():
        origins = held_view.center_t.unsqueeze(0).expand(held_view.dirs_t.shape[0], 3)
        pred_rgb, pred_depth, acc = render_rays(
            *model.properties(), origins, held_view.dirs_t
        )
    gt_d = held_view.depth_t
    gt_c = held_view.rgb_t
    m = (gt_d > 0) & (acc > 0.3)
    depth_rmse = torch.sqrt(((pred_depth[m] - gt_d[m]) ** 2).mean()).item() if m.any() else float("nan")
    mse = ((pred_rgb - gt_c) ** 2).mean().item()
    psnr = -10.0 * np.log10(max(mse, 1e-8))

    observed = vmap.surface_mask_np()
    coverage = float((observed & gt_surface_mask).sum() / max(gt_surface_mask.sum(), 1))

    return {
        "gaussians": int(len(mu)),
        "chamfer_mean_cm": 100.0 * float(d_nn.mean()),
        "chamfer_p90_cm": 100.0 * float(np.percentile(d_nn, 90)),
        "normal_err_deg": float(ang.mean()),
        "free_violation_pct": 100.0 * viol,
        "depth_rmse_cm": 100.0 * float(depth_rmse),
        "psnr_db": float(psnr),
        "coverage_pct": 100.0 * coverage,
    }


def render_full(model, view):
    with torch.no_grad():
        origins = view.center_t.unsqueeze(0).expand(view.dirs_t.shape[0], 3)
        rgb, depth, acc = render_rays(*model.properties(), origins, view.dirs_t)
    return (
        rgb.cpu().numpy().reshape(H_IMG, W_IMG, 3),
        depth.cpu().numpy().reshape(H_IMG, W_IMG),
        acc.cpu().numpy().reshape(H_IMG, W_IMG),
    )


def analyze_views(display_views, gt_surface_centers):
    """逐视角 GT 统计：深度范围、可见 GT 表面体素、相对前序视角的新增可见体素。"""
    fx, fy, cx, cy = K
    seen = np.zeros(len(gt_surface_centers), dtype=bool)
    rows = []
    for name, view in display_views:
        depth = view.depth
        pc = gt_surface_centers @ view.R.T + view.t
        z = pc[:, 2]
        u = fx * pc[:, 0] / np.maximum(z, 1e-6) + cx
        v = fy * pc[:, 1] / np.maximum(z, 1e-6) + cy
        ui = np.round(u).astype(np.int32)
        vi = np.round(v).astype(np.int32)
        inframe = (z > 0.05) & (ui >= 0) & (ui < W_IMG) & (vi >= 0) & (vi < H_IMG)
        depth_pix = np.zeros_like(z)
        depth_pix[inframe] = depth[vi[inframe], ui[inframe]]
        norm = np.sqrt(1.0 + ((u - cx) / fx) ** 2 + ((v - cy) / fy) ** 2)
        z_surf = depth_pix / np.maximum(norm, 1e-6)
        visible = inframe & (depth_pix > 0) & (np.abs(z - z_surf) < 0.08)
        new = visible & ~seen
        seen |= visible
        valid = depth[depth > 0]
        rows.append({
            "name": name,
            "depth_min_m": float(valid.min()) if valid.size else float("nan"),
            "depth_mean_m": float(valid.mean()) if valid.size else float("nan"),
            "depth_max_m": float(valid.max()) if valid.size else float("nan"),
            "depth_valid_pct": 100.0 * float(valid.size / depth.size),
            "visible_surface_voxels": int(visible.sum()),
            "new_surface_voxels": int(new.sum()),
        })
    return rows


def eval_models_on_views(models, display_views):
    """每个模型在每个 GT 视角上的深度 RMSE / PSNR，以及跨视角均值与标准差。"""
    per = {}
    for name, view in display_views:
        per[name] = {}
        for mname, model in models.items():
            rgb, depth, acc = render_full(model, view)
            gt_d = view.depth
            m = (gt_d > 0) & (acc > 0.3)
            rmse = (
                float(np.sqrt(np.mean((depth[m] - gt_d[m]) ** 2)) * 100.0)
                if m.any() else float("nan")
            )
            mse = float(np.mean((rgb - view.rgb) ** 2))
            psnr = -10.0 * np.log10(max(mse, 1e-8))
            per[name][mname] = {"depth_rmse_cm": rmse, "psnr_db": psnr}

    summary = {}
    for mname in models:
        rmses = np.array([per[n][mname]["depth_rmse_cm"] for n, _ in display_views])
        psnrs = np.array([per[n][mname]["psnr_db"] for n, _ in display_views])
        summary[mname] = {
            "depth_rmse_cm_mean": float(rmses.mean()),
            "depth_rmse_cm_std": float(rmses.std()),
            "psnr_db_mean": float(psnrs.mean()),
            "psnr_db_std": float(psnrs.std()),
        }
    return per, summary


# ---------------------------------------------------------------- 绘图
def setup_font():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                                       "WenQuanYi Zen Hei", "AR PL UMing CN", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def plot_gt_views(display_views, analysis, path):
    n = len(display_views)
    per_row = 3
    rows = int(np.ceil(n / per_row))
    fig, axes = plt.subplots(rows, per_row * 2, figsize=(18, 3.4 * rows))
    axes = np.atleast_2d(axes)
    im = None
    for i, (name, view) in enumerate(display_views):
        r, c = divmod(i, per_row)
        a_rgb = axes[r, 2 * c]
        a_dep = axes[r, 2 * c + 1]
        a_rgb.imshow(np.clip(view.rgb, 0, 1))
        a_rgb.axis("off")
        st = analysis[i]
        a_rgb.set_title(
            f"{name}\n深度 {st['depth_min_m']:.2f}–{st['depth_max_m']:.2f} m",
            fontsize=9,
        )
        im = a_dep.imshow(view.depth, cmap="viridis", vmin=0.2, vmax=3.0)
        a_dep.axis("off")
        a_dep.set_title(f"GT 深度（有效 {st['depth_valid_pct']:.0f}%）", fontsize=8)
    for j in range(n, rows * per_row):
        r, c = divmod(j, per_row)
        axes[r, 2 * c].axis("off")
        axes[r, 2 * c + 1].axis("off")
    cax = fig.add_axes([0.93, 0.15, 0.012, 0.7])
    fig.colorbar(im, cax=cax, label="深度 (m)")
    fig.suptitle("GT 视角集：建模目标（彩色图 + 深度图）")
    fig.tight_layout(rect=(0, 0, 0.92, 0.96))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_gt_scene(gt_pts, scene, initial_poses, active_views, held_pose, path):
    from matplotlib.lines import Line2D

    pts = gt_pts[::3]
    colors = scene.color(pts)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=colors, s=1.2, depthshade=False)

    x0, y0, z0 = ROOM_MIN
    x1, y1, z1 = ROOM_MAX
    corners = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ])
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    for i, j in edges:
        ax.plot(*zip(corners[i], corners[j]), color="0.55", lw=0.8, ls="--")

    def draw_pose(pose, color):
        R, t = pose
        pos = -(np.asarray(t) @ np.asarray(R))
        fwd = np.asarray(R)[2, :]
        ax.scatter([pos[0]], [pos[1]], [pos[2]], c=color, s=30, depthshade=False)
        ax.plot(
            [pos[0], pos[0] + 0.28 * fwd[0]],
            [pos[1], pos[1] + 0.28 * fwd[1]],
            [pos[2], pos[2] + 0.28 * fwd[2]],
            color=color, lw=1.6,
        )

    for pose in initial_poses:
        draw_pose(pose, "tab:blue")
    for view in active_views:
        draw_pose((view.R, view.t), "tab:green")
    draw_pose(held_pose, "tab:orange")

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:blue",
               markersize=6, label="初始视角"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:green",
               markersize=6, label="主动视角"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:orange",
               markersize=6, label="留出视角"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_zlim(z0, z1)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.view_init(elev=22, azim=-62)
    ax.set_title("GT 场景与相机轨迹：真实建模对象")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_render_vs_gt(display_views, model, path):
    n = len(display_views)
    fig, axes = plt.subplots(n, 4, figsize=(13, 2.9 * n))
    axes = np.atleast_2d(axes)
    im_d = None
    for i, (name, view) in enumerate(display_views):
        rgb, depth, acc = render_full(model, view)
        pred_depth = np.where((acc > 0.3) & (view.depth > 0), depth, np.nan)
        axes[i, 0].imshow(np.clip(view.rgb, 0, 1))
        axes[i, 0].set_ylabel(name, fontsize=8)
        im_d = axes[i, 1].imshow(view.depth, cmap="viridis", vmin=0.2, vmax=3.0)
        axes[i, 2].imshow(np.clip(rgb, 0, 1))
        axes[i, 3].imshow(pred_depth, cmap="viridis", vmin=0.2, vmax=3.0)
        for a in axes[i]:
            a.set_xticks([])
            a.set_yticks([])
    for j, title in enumerate(["GT 颜色", "GT 深度", "模型颜色", "模型深度（无效=NaN）"]):
        axes[0, j].set_title(title, fontsize=10)
    cax = fig.add_axes([0.93, 0.15, 0.012, 0.7])
    fig.colorbar(im_d, cax=cax, label="深度 (m)")
    fig.suptitle("逐视角对比：GT vs 体素约束模型")
    fig.tight_layout(rect=(0, 0, 0.92, 0.97))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_per_view_metrics(per_view, display_views, path):
    names = [n for n, _ in display_views]
    model_names = ["init", "photo", "depth", "voxel"]
    labels = {"init": "初始化", "photo": "纯光度", "depth": "+深度", "voxel": "+体素"}
    colors = {"init": "0.6", "photo": "tab:orange", "depth": "tab:blue", "voxel": "tab:green"}
    x = np.arange(len(names))
    width = 0.2
    fig, axes = plt.subplots(1, 2, figsize=(15, 4))
    for ax, metric, title in [
        (axes[0], "depth_rmse_cm", "逐视角深度 RMSE (cm)"),
        (axes[1], "psnr_db", "逐视角 PSNR (dB)"),
    ]:
        for j, mname in enumerate(model_names):
            vals = [per_view[n][mname][metric] for n in names]
            ax.bar(x + (j - 1.5) * width, vals, width,
                   color=colors[mname], label=labels[mname])
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=18, fontsize=8)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("逐视角指标（前 4 个为训练视角，后 2 个为分析/留出视角）")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_depth_models(display_views, models, per_view, path):
    """逐视角深度图对比：GT / 纯光度 / 纯光度误差 / 体素约束 / 体素误差。"""
    if len(display_views) >= 6:
        sel = [display_views[0], display_views[3], display_views[4], display_views[5]]
    else:
        sel = display_views
    n = len(sel)
    fig, axes = plt.subplots(n, 5, figsize=(16, 2.9 * n))
    axes = np.atleast_2d(axes)
    im_d = im_e = None
    for i, (name, view) in enumerate(sel):
        _, d_p, acc_p = render_full(models["photo"], view)
        _, d_v, acc_v = render_full(models["voxel"], view)
        valid = view.depth > 0
        m_p = valid & (acc_p > 0.3)
        m_v = valid & (acc_v > 0.3)
        d_p_show = np.where(m_p, d_p, np.nan)
        d_v_show = np.where(m_v, d_v, np.nan)
        e_p = np.where(m_p, np.abs(d_p - view.depth) * 100.0, np.nan)
        e_v = np.where(m_v, np.abs(d_v - view.depth) * 100.0, np.nan)
        rp = per_view[name]["photo"]["depth_rmse_cm"]
        rv = per_view[name]["voxel"]["depth_rmse_cm"]
        axes[i, 0].imshow(view.depth, cmap="viridis", vmin=0.2, vmax=3.0)
        axes[i, 0].set_ylabel(
            f"{name}\n纯光度 {rp:.1f} cm\n体素 {rv:.1f} cm", fontsize=7
        )
        im_d = axes[i, 1].imshow(d_p_show, cmap="viridis", vmin=0.2, vmax=3.0)
        im_e = axes[i, 2].imshow(e_p, cmap="magma", vmin=0, vmax=20)
        axes[i, 3].imshow(d_v_show, cmap="viridis", vmin=0.2, vmax=3.0)
        axes[i, 4].imshow(e_v, cmap="magma", vmin=0, vmax=20)
        for a in axes[i]:
            a.set_xticks([])
            a.set_yticks([])
    titles = ["GT 深度", "纯光度深度", "纯光度 |误差| (cm)", "体素约束深度", "体素约束 |误差| (cm)"]
    for j, t in enumerate(titles):
        axes[0, j].set_title(t, fontsize=9)
    cax_d = fig.add_axes([0.93, 0.55, 0.01, 0.38])
    fig.colorbar(im_d, cax=cax_d, label="深度 (m)")
    cax_e = fig.add_axes([0.93, 0.10, 0.01, 0.38])
    fig.colorbar(im_e, cax=cax_e, label="深度误差 (cm，截断到 0–20)")
    fig.suptitle("深度图对比：纯光度 3DGS vs 体素约束")
    fig.tight_layout(rect=(0, 0, 0.92, 0.96))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_progress(records, path):
    if not records:
        return
    steps = np.arange(len(records))
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    keys = [
        ("chamfer_mean_cm", "Chamfer 均值 (cm)", axes[0, 0]),
        ("normal_err_deg", "法向误差 (°)", axes[0, 1]),
        ("free_violation_pct", "自由空间违规 (%)", axes[1, 0]),
        ("coverage_pct", "GT 表面覆盖率 (%)", axes[1, 1]),
    ]
    for key, title, ax in keys:
        vals = [r[key] for r in records]
        ax.plot(steps, vals, marker="o", color="tab:green")
        ax.set_title(title)
        ax.set_xlabel("主动探索步")
        ax.grid(True, alpha=0.3)
    fig.suptitle("体素约束模型的逐步完善")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_comparison(comparisons, path):
    names = ["init", "photo", "depth", "voxel"]
    labels = ["未训练\n(体素初始化)", "纯光度\n3DGS", "+深度监督", "+体素约束\n(本方法)"]
    colors = ["0.6", "tab:orange", "tab:blue", "tab:green"]
    metrics = [
        ("chamfer_mean_cm", "Chamfer 均值 (cm)"),
        ("normal_err_deg", "法向误差 (°)"),
        ("free_violation_pct", "自由空间违规 (%)"),
        ("depth_rmse_cm", "留出视角深度 RMSE (cm)"),
        ("psnr_db", "留出视角 PSNR (dB)"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 3.6))
    for ax, (key, title) in zip(axes, metrics):
        vals = [comparisons[n][key] for n in names]
        bars = ax.bar(range(len(names)), vals, color=colors)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    fig.suptitle("同一体素初始化下的消融对比")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_map(gt_pts, models, path):
    fig = plt.figure(figsize=(12, 5))
    colors = {
        "init": ("tab:blue", "体素初始化"),
        "photo": ("tab:orange", "纯光度 3DGS"),
        "voxel": ("tab:green", "+体素约束"),
    }
    for k, azim in enumerate((-65, 115)):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        ax.scatter(gt_pts[::8, 0], gt_pts[::8, 1], gt_pts[::8, 2],
                   s=0.4, c="0.8", label="GT 表面")
        for name, (c, label) in colors.items():
            mu = models[name].means().detach().cpu().numpy()
            ax.scatter(mu[::2, 0], mu[::2, 1], mu[::2, 2], s=2.5, c=c, label=label)
        ax.set_xlim(0, 2.4)
        ax.set_ylim(0, 1.6)
        ax.set_zlim(0, 1.6)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        ax.view_init(elev=20, azim=azim)
    fig.legend(loc="lower center", ncol=4, fontsize=8)
    fig.suptitle("高斯中心与 GT 表面的空间分布")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_tsdf_slice(vmap, model, path):
    j = vmap.dims[1] // 2
    tsdf_slice = vmap.tsdf[:, j, :].T
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(tsdf_slice, extent=[0, 2.4, 0, 1.6], origin="lower",
                   cmap="coolwarm", vmin=-1, vmax=1)
    surf = vmap.surface_mask_np()[:, j, :]
    xs, zs = np.meshgrid(
        (np.arange(vmap.dims[0]) + 0.5) * vmap.voxel,
        (np.arange(vmap.dims[2]) + 0.5) * vmap.voxel,
        indexing="ij",
    )
    ax.scatter(xs[surf], zs[surf], s=1.0, c="k", label="表面体素")
    mu = model.means().detach().cpu().numpy()
    y_mid = (j + 0.5) * vmap.voxel
    sel = np.abs(mu[:, 1] - y_mid) < 2.0 * vmap.voxel
    ax.scatter(mu[sel, 0], mu[sel, 2], s=6, c="lime", label="高斯中心")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_title(f"TSDF 切片 (y = {y_mid:.2f} m)")
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(im, ax=ax, label="归一化 TSDF")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_renders(held_view, models, path):
    rows = [
        ("GT", held_view.rgb, held_view.depth),
    ]
    for name, label in (("photo", "纯光度 3DGS"), ("depth", "+深度监督"), ("voxel", "+体素约束")):
        rgb, depth, _ = render_full(models[name], held_view)
        rows.append((label, rgb, depth))

    fig, axes = plt.subplots(len(rows), 2, figsize=(9, 3.2 * len(rows)))
    for i, (label, rgb, depth) in enumerate(rows):
        axes[i, 0].imshow(np.clip(rgb, 0, 1))
        axes[i, 0].set_title(f"{label} · 颜色")
        axes[i, 0].axis("off")
        im = axes[i, 1].imshow(depth, cmap="viridis", vmin=0.2, vmax=3.0)
        axes[i, 1].set_title(f"{label} · 深度")
        axes[i, 1].axis("off")
    fig.colorbar(im, ax=axes[:, 1], shrink=0.8, label="深度 (m)")
    fig.suptitle("留出视角渲染对比")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 主流程
def main():
    global DEPTH_NOISE
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--iters-per-step", type=int, default=150)
    parser.add_argument("--static-iters", type=int, default=500)
    parser.add_argument("--rays", type=int, default=768)
    parser.add_argument("--max-gauss", type=int, default=6000)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--policy", default="heuristic", choices=["heuristic", "rl"],
                        help="主动探索的选点策略：手工作用函数 或 PPO 学习策略")
    parser.add_argument("--policy-ckpt", default=None,
                        help="--policy rl 时加载的策略权重，默认 outputs/rl_policy.pt")
    parser.add_argument("--noise", type=float, default=0.0,
                        help="采集视角的深度噪声 σ(m)，0=无噪声（与训练/评估一致才可比）")
    parser.add_argument("--seed", type=int, default=0,
                        help="全局随机种子（多 seed 几何统计用），默认 0 = 原基线")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.steps = 3
        args.iters_per_step = 40
        args.static_iters = 200
        args.rays = 384
        args.max_gauss = 4000

    if args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    DEPTH_NOISE = args.noise
    setup_font()

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    policy_ckpt = args.policy_ckpt or os.path.join(out_dir, "rl_policy.pt")
    if args.policy == "rl" and not os.path.exists(policy_ckpt):
        raise SystemExit(f"--policy rl 需要策略权重 {policy_ckpt}，先跑 python train_rl.py")
    print(f"主动探索策略 = {args.policy}" + (f"  ({policy_ckpt})" if args.policy == "rl" else ""))

    scene = DemoScene()
    initial_poses = [
        look_at((0.35, 0.35, 1.10), (1.20, 0.90, 0.55)),
        look_at((2.05, 0.35, 1.10), (1.20, 0.90, 0.55)),
        look_at((0.35, 1.25, 1.10), (1.20, 0.70, 0.45)),
        look_at((2.05, 1.25, 1.10), (1.20, 0.70, 0.45)),
    ]
    views = [render_scene_view(scene, p, device) for p in initial_poses]
    visited = []
    for R, t in initial_poses:
        pos = -(np.asarray(t) @ np.asarray(R))
        fwd = np.asarray(R).T @ np.array([0.0, 0.0, 1.0], np.float32)
        visited.append((pos, fwd))

    vmap = DenseTSDF(ROOM_MIN, ROOM_MAX, VOXEL_SIZE, TRUNC, device=str(device))
    for v in views:
        vmap.fuse(v.depth, v.rgb, v.R, v.t, K)
    print("初始体素统计:", vmap.stats())

    model = VoxelGaussians(vmap, device=device, max_total=args.max_gauss)
    n0 = model.spawn()
    print(f"初始化高斯数量: {n0}")

    gt_pts = scene.sample_surface()
    gt_tree = cKDTree(gt_pts)
    gt_surface_mask = scene.gt_surface_voxel_mask(vmap.gmin, vmap.dims, vmap.voxel)
    held_pose = look_at((1.15, 0.70, 1.35), (1.60, 1.10, 0.30))
    held_view = render_scene_view(scene, held_pose, device)
    print(f"GT 表面采样点: {len(gt_pts)}, GT 表面体素: {int(gt_surface_mask.sum())}")

    init_metrics = evaluate(model, vmap, scene, gt_tree, held_view, gt_surface_mask)
    print("未训练指标:", {k: round(v, 3) if isinstance(v, float) else v for k, v in init_metrics.items()})

    records = []
    for step in range(args.steps):
        if step > 0:
            if args.policy == "rl":
                # 目标点数量必须与训练一致（见 rl_active.TARGET_COUNTS）
                pts, wts = sample_utility_targets(vmap, rng, *TARGET_COUNTS)
                kinds = (wts > 0.8).astype(np.float32)   # frontier=1.0 / 低置信表面=0.6
                pose, score, feats = select_nbv_rl(
                    policy_ckpt, vmap, pts, wts, kinds,
                    current_pos=views[-1].center_np, current_fwd=visited[-1][1],
                    visited=visited, K=K, H=H_IMG, W=W_IMG, device=str(device), rng=rng,
                )
            else:
                pts, wts = sample_utility_targets(vmap, rng)
                cands = candidate_poses(vmap)
                pose, score = select_nbv(
                    cands, pts, wts, vmap, K, H_IMG, W_IMG,
                    current_pos=views[-1].center_np, visited=visited,
                )
            if pose is None:
                print("没有可用候选视点，提前结束主动探索")
                break
            R, t, pos, tgt = pose
            views.append(render_scene_view(scene, (R, t), device))
            vmap.fuse(views[-1].depth, views[-1].rgb, R, t, K)
            fwd = tgt - pos
            fwd = fwd / max(np.linalg.norm(fwd), 1e-8)
            visited.append((pos, fwd))
            n_pruned = model.prune_free_anchors()
            n_new = model.spawn()
            if args.policy == "rl":
                print(f"[step {step}] policy score={score:.2f} pos={np.round(pos, 2)} "
                      f"gain={feats[0]:.1f} path={feats[5]:.2f}m "
                      f"novelty_dir={feats[7]:.2f} new_gaussians={n_new} pruned={n_pruned}")
            else:
                print(f"[step {step}] NBV score={score:.1f} pos={np.round(pos, 2)} "
                      f"new_gaussians={n_new} pruned={n_pruned}")

        lam = min(1.0, 0.4 + 0.3 * step)
        dt = train_model(model, views, vmap, "voxel", args.iters_per_step, device,
                         seed=100 + step, lam=lam, rays=args.rays)
        m = evaluate(model, vmap, scene, gt_tree, held_view, gt_surface_mask)
        records.append(m)
        pretty = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()}
        print(f"[step {step}] train {dt:.1f}s", pretty)

    display_views = [("初始视角 1", views[0]), ("初始视角 3", views[2])]
    active_views = views[4:]
    if active_views:
        display_views.append(("主动视角 A", active_views[max(0, len(active_views) // 3 - 1)]))
        display_views.append(("主动视角 B", active_views[-1]))
    ana_pose = look_at((1.00, 0.55, 1.42), (1.65, 1.10, 0.20))
    display_views.append(("高位分析视角", render_scene_view(scene, ana_pose, device)))
    display_views.append(("留出视角", held_view))

    gt_surface_centers = vmap.centers[gt_surface_mask.reshape(-1)]
    view_analysis = analyze_views(display_views, gt_surface_centers)
    print("\nGT 视角分析:")
    for r in view_analysis:
        print(f"  {r['name']:8s} 深度 {r['depth_min_m']:.2f}-{r['depth_max_m']:.2f} m  "
              f"可见表面体素 {r['visible_surface_voxels']:4d}  "
              f"新增 {r['new_surface_voxels']:4d}")

    print("训练三个静态对比模型 ...")
    init_model = VoxelGaussians(vmap, device=device, max_total=args.max_gauss)
    init_model.spawn()
    models = {"init": init_model}
    for mode in ("photo", "depth", "voxel"):
        m = VoxelGaussians(vmap, device=device, max_total=args.max_gauss)
        m.spawn()
        dt = train_model(m, views, vmap, mode, args.static_iters, device,
                         seed=7, lam=1.0 if mode == "voxel" else 0.0, rays=args.rays)
        models[mode] = m
        cm = evaluate(m, vmap, scene, gt_tree, held_view, gt_surface_mask)
        print(f"{mode:5s} train {dt:.1f}s", {k: round(v, 3) if isinstance(v, float) else v for k, v in cm.items()})

    comparisons = {"init": evaluate(init_model, vmap, scene, gt_tree, held_view, gt_surface_mask)}
    for name in ("photo", "depth", "voxel"):
        comparisons[name] = evaluate(models[name], vmap, scene, gt_tree, held_view, gt_surface_mask)

    per_view_eval, per_view_summary = eval_models_on_views(models, display_views)
    print("\n逐视角模型指标（均值±标准差）:")
    for mname, s in per_view_summary.items():
        print(f"  {mname:6s} depth {s['depth_rmse_cm_mean']:.2f}±{s['depth_rmse_cm_std']:.2f} cm  "
              f"PSNR {s['psnr_db_mean']:.2f}±{s['psnr_db_std']:.2f} dB")

    print("\n================ 最终对比 ================")
    header = f"{'model':8s} {'chamfer':>9s} {'normal':>8s} {'free%':>7s} {'depth':>8s} {'psnr':>7s} {'cover%':>8s}"
    print(header)
    for name in ("init", "photo", "depth", "voxel"):
        c = comparisons[name]
        print(f"{name:8s} {c['chamfer_mean_cm']:9.2f} {c['normal_err_deg']:8.2f} "
              f"{c['free_violation_pct']:7.2f} {c['depth_rmse_cm']:8.2f} "
              f"{c['psnr_db']:7.2f} {c['coverage_pct']:8.2f}")

    result = {
        "config": vars(args),
        "device": str(device),
        "views": len(views),
        "voxel_stats": vmap.stats(),
        "init": init_metrics,
        "progress": records,
        "comparison": comparisons,
        "view_analysis": view_analysis,
        "per_view_eval": per_view_eval,
        "per_view_summary": per_view_summary,
    }
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    plot_progress(records, os.path.join(out_dir, "metrics_progress.png"))
    plot_comparison(comparisons, os.path.join(out_dir, "model_comparison.png"))
    plot_map(gt_pts, models, os.path.join(out_dir, "map_3d.png"))
    plot_tsdf_slice(vmap, models["voxel"], os.path.join(out_dir, "tsdf_slice.png"))
    plot_renders(held_view, models, os.path.join(out_dir, "render_compare.png"))
    plot_gt_views(display_views, view_analysis, os.path.join(out_dir, "gt_views.png"))
    plot_gt_scene(gt_pts, scene, initial_poses, active_views, held_pose,
                  os.path.join(out_dir, "gt_scene_3d.png"))
    plot_render_vs_gt(display_views, models["voxel"],
                      os.path.join(out_dir, "render_vs_gt_views.png"))
    plot_depth_models(display_views, models, per_view_eval,
                      os.path.join(out_dir, "depth_compare_models.png"))
    plot_per_view_metrics(per_view_eval, display_views,
                          os.path.join(out_dir, "per_view_metrics.png"))

    print(f"\n结果已写入: {out_dir}")
    for fn in ("metrics.json", "metrics_progress.png", "model_comparison.png",
               "map_3d.png", "tsdf_slice.png", "render_compare.png",
               "gt_views.png", "gt_scene_3d.png", "render_vs_gt_views.png",
               "depth_compare_models.png", "per_view_metrics.png"):
        print("  -", os.path.join(out_dir, fn))


if __name__ == "__main__":
    main()
