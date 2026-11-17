"""点云表征 vs 体素-高斯表征：在可控深度噪声下的对比实验。

对比对象（同一组 RGB-D 视角、同一噪声水平）：
1. raw PC     ：所有视角的深度反投影点云（测量级精度，但含噪声、重叠、空洞）；
2. fused PC   ：TSDF 表面体素中心点云（融合去噪、但受体素分辨率限制）；
3. Voxel-GS   ：从同一 TSDF 生成的锚定高斯，训练体素约束损失（可渲染、稠密）。

指标：
- Chamfer 均值/P90（预测点 -> GT 表面点）；
- GT 表面点覆盖率（2 cm 精度阈值与 6 cm 分辨率感知阈值）；
- 法向误差（GT 最近表面点的法向；点云用 kNN PCA 估计法向）；
- 局部厚度（kNN 协方差最小特征值，衡量噪声/厚度）；
- 留出视角深度 RMSE 与深度覆盖率（点云用 1 像素半径 z-buffer splat，GS 用渲染器）；
- 自由空间违规、点/高斯数量与近似存储。
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import cKDTree

from demo.scene import DemoScene, ROOM_MIN, ROOM_MAX, TRUNC, VOXEL_SIZE
from demo.voxel_map import DenseTSDF
from demo.gaussians import VoxelGaussians
from demo.utils import look_at, pixel_rays
from run_demo import H_IMG, K, W_IMG, View, render_full, setup_font, train_model

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

POSES = [
    look_at((0.35, 0.35, 1.10), (1.20, 0.90, 0.55)),
    look_at((2.05, 0.35, 1.10), (1.20, 0.90, 0.55)),
    look_at((0.35, 1.25, 1.10), (1.20, 0.70, 0.45)),
    look_at((2.05, 1.25, 1.10), (1.20, 0.70, 0.45)),
    look_at((0.30, 0.80, 1.05), (1.20, 0.80, 0.55)),
    look_at((0.30, 1.30, 1.05), (1.20, 0.70, 0.45)),
    look_at((0.30, 0.30, 1.05), (1.20, 0.90, 0.55)),
    look_at((2.10, 1.30, 1.05), (1.20, 0.70, 0.45)),
    look_at((2.10, 0.80, 1.05), (1.20, 0.80, 0.55)),
]
HELD_POSE = look_at((1.15, 0.70, 1.35), (1.60, 1.10, 0.30))


def make_views(scene, rng, sigma, device):
    views = []
    for pose in POSES:
        R, t = pose
        depth, rgb, _ = scene.render_view(R, t, K, H_IMG, W_IMG)
        if sigma > 0:
            depth = depth + rng.normal(0.0, sigma, size=depth.shape).astype(np.float32)
            depth = np.clip(depth, 0.15, 6.0)
        views.append(View(depth, rgb, R, t, device))
    return views


def build_raw_cloud(views, step=2):
    pts, cols = [], []
    for v in views:
        d = v.depth.reshape(-1)
        sel = np.arange(0, H_IMG * W_IMG, step)
        valid = d[sel] > 0
        p = v.center_np[None] + d[sel, None] * v.dirs_np[sel]
        pts.append(p[valid])
        cols.append(v.rgb.reshape(-1, 3)[sel][valid])
    return np.concatenate(pts, 0), np.concatenate(cols, 0)


def build_tsdf_cloud(vmap):
    idx = np.flatnonzero(((vmap.weight > 0.5) & (np.abs(vmap.tsdf) < 0.35)).reshape(-1))
    centers = vmap.centers[idx]
    ijk = vmap.ijk.reshape(-1, 3)[idx].astype(np.int64)
    s = vmap.tsdf.reshape(-1)[idx] * TRUNC
    g = vmap.query_grad_np(centers) * TRUNC
    denom = (g * g).sum(1) + 1e-6
    step = s[:, None] * g / denom[:, None]
    step_norm = np.linalg.norm(step, axis=1, keepdims=True)
    step = step * np.minimum(1.0, 0.5 * vmap.voxel / np.maximum(step_norm, 1e-8))
    points = centers - step
    gn = vmap.query_grad_np(points)
    normals = gn / np.maximum(np.linalg.norm(gn, axis=1, keepdims=True), 1e-8)
    csum = vmap.color_sum.reshape(-1, 3)[idx]
    cw = vmap.color_w.reshape(-1)[idx]
    colors = np.where(cw[:, None] > 0, csum / np.maximum(cw[:, None], 1e-6), 0.5)
    return points, np.clip(colors, 0.0, 1.0), normals


def pca_normals_thickness(points, k=16):
    tree = cKDTree(points)
    k = min(k, len(points))
    _, idx = tree.query(points, k=k)
    neigh = points[idx]
    mean = neigh.mean(1, keepdims=True)
    diff = neigh - mean
    cov = np.einsum("nki,nkj->nij", diff, diff) / k
    w, v = np.linalg.eigh(cov)
    normals = v[:, :, 0]
    thickness = np.sqrt(np.maximum(w[:, 0], 0.0))
    return normals, thickness


def geometry_metrics(points, normals, vmap, gt_tree, gt_normals, gt_pts):
    d, idx = gt_tree.query(points)
    if normals is None:
        normals, thickness = pca_normals_thickness(points)
    else:
        _, thickness = pca_normals_thickness(points)
    n_gt = gt_normals[idx]
    cos = np.clip(np.abs((normals * n_gt).sum(-1)), 0.0, 1.0)
    normal_err = float(np.degrees(np.arccos(cos)).mean())
    tree = cKDTree(points)
    dg, _ = tree.query(gt_pts)
    cov2 = 100.0 * float((dg < 0.02).mean())
    cov6 = 100.0 * float((dg < 0.06).mean())
    tsdf_q = vmap.query_np(points)
    w_q = vmap.query_weight_np(points)
    free_bad = 100.0 * float(((tsdf_q > 0.5) & (w_q > 0.5)).mean())
    return {
        "points": int(len(points)),
        "chamfer_mean_cm": 100.0 * float(d.mean()),
        "chamfer_p90_cm": 100.0 * float(np.percentile(d, 90)),
        "gt_coverage_2cm_pct": cov2,
        "gt_coverage_6cm_pct": cov6,
        "normal_err_deg": normal_err,
        "local_thickness_cm": 100.0 * float(np.median(thickness)),
        "free_space_pct": free_bad,
    }


def splat_pointcloud(points, view, radius=1):
    """把点云 z-buffer splat 成深度图；radius 为方形 splat 半径（像素）。"""
    pc = points @ view.R.T + view.t
    z = pc[:, 2]
    fx, fy, cx, cy = K
    u = fx * pc[:, 0] / np.maximum(z, 1e-6) + cx
    v = fy * pc[:, 1] / np.maximum(z, 1e-6) + cy
    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)
    m = (z > 0.05) & (ui >= 0) & (ui < W_IMG) & (vi >= 0) & (vi < H_IMG)
    ui, vi, z = ui[m], vi[m], z[m]
    depth = np.full((H_IMG, W_IMG), np.inf, np.float32)
    for du in range(-radius, radius + 1):
        for dv in range(-radius, radius + 1):
            uu = ui + du
            vv = vi + dv
            inb = (uu >= 0) & (uu < W_IMG) & (vv >= 0) & (vv < H_IMG)
            np.minimum.at(depth, (vv[inb], uu[inb]), z[inb])
    depth[~np.isfinite(depth)] = np.nan
    yy, xx = np.mgrid[0:H_IMG, 0:W_IMG]
    norm = np.sqrt(1.0 + ((xx - cx) / fx) ** 2 + ((yy - cy) / fy) ** 2)
    return depth * norm


def depth_metrics(depth_pred, view):
    gt = view.depth
    valid = np.isfinite(depth_pred) & (gt > 0)
    if valid.any():
        rmse = 100.0 * float(np.sqrt(np.mean((depth_pred[valid] - gt[valid]) ** 2)))
    else:
        rmse = float("nan")
    return rmse, 100.0 * float(valid.mean())


def gs_depth_metrics(model, view):
    rgb, depth, acc = render_full(model, view)
    gt = view.depth
    valid = (acc > 0.3) & (gt > 0)
    rmse = 100.0 * float(np.sqrt(np.mean((depth[valid] - gt[valid]) ** 2))) if valid.any() else float("nan")
    return rmse, 100.0 * float(valid.mean()), depth, rgb


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_font()
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    np.random.seed(0)

    scene = DemoScene()
    gt_pts = scene.sample_surface()
    gt_tree = cKDTree(gt_pts)
    gt_normals = scene.gradient(gt_pts)

    held_gt_view = None
    results = []
    snapshot = None
    sigmas_cm = [0.0, 1.0, 2.0, 4.0]

    for sigma_cm in sigmas_cm:
        sigma = sigma_cm / 100.0
        views = make_views(scene, rng, sigma, device)
        vmap = DenseTSDF(ROOM_MIN, ROOM_MAX, VOXEL_SIZE, TRUNC, device=str(device))
        for v in views:
            vmap.fuse(v.depth, v.rgb, v.R, v.t, K)

        # 留出视角始终使用无噪声 GT
        R, t = HELD_POSE
        depth_h, rgb_h, _ = scene.render_view(R, t, K, H_IMG, W_IMG)
        held_view = View(depth_h, rgb_h, R, t, device)

        raw_pts, raw_cols = build_raw_cloud(views)
        fused_pts, fused_cols, fused_normals = build_tsdf_cloud(vmap)

        model = VoxelGaussians(vmap, device=device, max_total=6000)
        model.spawn()
        train_model(model, views, vmap, "voxel", 250, device,
                    seed=11, lam=1.0, rays=768)
        gs_pts = model.means().detach().cpu().numpy()
        gs_normals = model.normals().detach().cpu().numpy()

        raw_m = geometry_metrics(raw_pts, None, vmap, gt_tree, gt_normals, gt_pts)
        fused_m = geometry_metrics(fused_pts, fused_normals, vmap, gt_tree, gt_normals, gt_pts)
        gs_m = geometry_metrics(gs_pts, gs_normals, vmap, gt_tree, gt_normals, gt_pts)

        raw_depth = splat_pointcloud(raw_pts, held_view)
        fused_depth = splat_pointcloud(fused_pts, held_view)
        raw_d_rmse, raw_d_cov = depth_metrics(raw_depth, held_view)
        fused_d_rmse, fused_d_cov = depth_metrics(fused_depth, held_view)
        gs_d_rmse, gs_d_cov, gs_depth, gs_rgb = gs_depth_metrics(model, held_view)

        for name, m, d_rmse, d_cov in [
            ("raw_pc", raw_m, raw_d_rmse, raw_d_cov),
            ("fused_pc", fused_m, fused_d_rmse, fused_d_cov),
            ("voxel_gs", gs_m, gs_d_rmse, gs_d_cov),
        ]:
            m = dict(m)
            m.update({
                "noise_sigma_cm": sigma_cm,
                "depth_rmse_cm": d_rmse,
                "depth_coverage_pct": d_cov,
            })
            if name == "raw_pc":
                m["storage_kb"] = len(raw_pts) * 6 * 4 / 1024.0
            elif name == "fused_pc":
                m["storage_kb"] = len(fused_pts) * 9 * 4 / 1024.0
            else:
                n_vox = int(np.prod(vmap.dims))
                m["storage_kb"] = (len(gs_pts) * 14 + n_vox * 7) * 4 / 1024.0
            m["method"] = name
            results.append(m)

        if abs(sigma_cm - 2.0) < 1e-6:
            snapshot = {
                "gt_depth": held_view.depth,
                "raw_depth": raw_depth,
                "fused_depth": fused_depth,
                "gs_depth": gs_depth,
                "raw_err": np.abs(raw_depth - held_view.depth) * 100.0,
                "fused_err": np.abs(fused_depth - held_view.depth) * 100.0,
                "gs_err": np.abs(gs_depth - held_view.depth) * 100.0,
                "gs_acc": render_full(model, held_view)[2],
            }

        print(f"\n=== 深度噪声 sigma = {sigma_cm:.0f} cm ===")
        for r in results[-3:]:
            print(f"  {r['method']:8s} chamfer {r['chamfer_mean_cm']:5.2f} cm  "
                  f"cover2 {r['gt_coverage_2cm_pct']:5.1f}%  "
                  f"cover6 {r['gt_coverage_6cm_pct']:5.1f}%  "
                  f"normal {r['normal_err_deg']:5.1f}°  "
                  f"thick {r['local_thickness_cm']:4.2f} cm  "
                  f"depth {r['depth_rmse_cm']:5.2f} cm / {r['depth_coverage_pct']:4.1f}%")

    with open(os.path.join(OUT_DIR, "pointcloud_noise_study.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 图 1：指标随噪声变化
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    methods = [("raw_pc", "原始点云", "tab:orange"), ("fused_pc", "TSDF 点云", "tab:blue"),
               ("voxel_gs", "体素约束 GS", "tab:green")]
    panels = [
        ("chamfer_mean_cm", "Chamfer 均值 (cm)"),
        ("gt_coverage_6cm_pct", "GT 表面覆盖率 (6 cm, %)"),
        ("depth_rmse_cm", "留出视角深度 RMSE (cm)"),
        ("depth_coverage_pct", "留出视角深度覆盖率 (%)"),
    ]
    for ax, (key, title) in zip(axes.ravel(), panels):
        for name, label, color in methods:
            xs, ys = [], []
            for r in results:
                if r["method"] == name:
                    xs.append(r["noise_sigma_cm"])
                    ys.append(r[key])
            ax.plot(xs, ys, marker="o", label=label, color=color)
        ax.set_xlabel("深度噪声 σ (cm)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("点云 vs 体素-高斯：几何精度、完整性与可渲染性随噪声的变化")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "pointcloud_noise_study.png"), dpi=160)
    plt.close(fig)

    # 图 2：σ=2 cm 时的留出视角深度与误差
    if snapshot is not None:
        fig, axes = plt.subplots(2, 4, figsize=(15, 6.4))
        gt_d = snapshot["gt_depth"]
        items = [
            ("GT 深度", gt_d, "viridis", (0.2, 3.0)),
            ("原始点云深度", snapshot["raw_depth"], "viridis", (0.2, 3.0)),
            ("TSDF 点云深度", snapshot["fused_depth"], "viridis", (0.2, 3.0)),
            ("体素 GS 深度", snapshot["gs_depth"], "viridis", (0.2, 3.0)),
        ]
        for j, (title, img, cmap, lim) in enumerate(items):
            axes[0, j].imshow(img, cmap=cmap, vmin=lim[0], vmax=lim[1])
            axes[0, j].set_title(title, fontsize=9)
            axes[0, j].set_xticks([])
            axes[0, j].set_yticks([])
        axes[1, 0].axis("off")
        errs = [
            ("原始点云 |误差| (cm)", snapshot["raw_err"]),
            ("TSDF 点云 |误差| (cm)", snapshot["fused_err"]),
            ("体素 GS |误差| (cm)", snapshot["gs_err"]),
        ]
        im_e = None
        for j, (title, img) in enumerate(errs, start=1):
            im_e = axes[1, j].imshow(img, cmap="magma", vmin=0, vmax=20)
            axes[1, j].set_title(title, fontsize=9)
            axes[1, j].set_xticks([])
            axes[1, j].set_yticks([])
        fig.colorbar(im_e, ax=axes[1, 1:].ravel().tolist(), shrink=0.8, label="深度误差 (cm)")
        fig.suptitle("留出视角深度对比（训练深度噪声 σ = 2 cm）")
        fig.savefig(os.path.join(OUT_DIR, "pointcloud_depth_compare.png"), dpi=160,
                    bbox_inches="tight")
        plt.close(fig)

    print(f"\n结果已写入: {OUT_DIR}")


if __name__ == "__main__":
    main()
