"""所有绘图函数（统一 Agg 后端 + 中文回退字体）。"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..common.camera import H_IMG, W_IMG, K
from ..common.scene import ROOM_MAX, ROOM_MIN, TRUNC, VOXEL_SIZE
from ..reconstruction.metrics import render_full


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
