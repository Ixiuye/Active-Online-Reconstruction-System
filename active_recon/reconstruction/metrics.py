"""重建评估：Chamfer / 法向 / 自由空间违规 / 深度 RMSE / PSNR / 覆盖率，逐视角分析。"""
from __future__ import annotations

import numpy as np
import torch

from ..common.camera import H_IMG, W_IMG, K
from .renderer import render_rays


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
