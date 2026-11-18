"""端到端 demo：融合 → 锚定高斯 → 体素约束优化 → 主动探索 → 评估出图。"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from scipy.spatial import cKDTree

from ..common.camera import (H_IMG, K, W_IMG, View, render_scene_view,
                             set_depth_noise)
from ..common.scene import (ROOM_MAX, ROOM_MIN, TRUNC, VOXEL_SIZE, make_scene)
from ..common.paths import OUTPUTS
from ..common.utils import look_at
from ..reconstruction.gaussians import VoxelGaussians
from ..reconstruction.metrics import (analyze_views, eval_models_on_views,
                                      evaluate, render_full)
from ..reconstruction.trainer import train_model
from ..reconstruction.voxel_map import DenseTSDF
from ..viewpoint.heuristic import candidate_poses, sample_utility_targets, select_nbv
from ..viewpoint.rl import TARGET_COUNTS, select_nbv_rl
from .plots import (plot_comparison, plot_depth_models, plot_gt_scene, plot_gt_views,
                    plot_map, plot_per_view_metrics, plot_progress, plot_render_vs_gt,
                    plot_renders, plot_tsdf_slice, setup_font)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=6,
                        help="总训练阶段数；决策数 = steps-1（可用 --explore-steps 直接指定）")
    parser.add_argument("--explore-steps", type=int, default=None,
                        help="主动探索决策数（覆盖 --steps：steps = explore_steps + 1）")
    parser.add_argument("--iters-per-step", type=int, default=150)
    parser.add_argument("--static-iters", type=int, default=500)
    parser.add_argument("--rays", type=int, default=768)
    parser.add_argument("--max-gauss", type=int, default=6000)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--policy", default="heuristic", choices=["heuristic", "rl"],
                        help="主动探索的选点策略：手工作用函数 或 PPO 学习策略")
    parser.add_argument("--policy-ckpt", default=None,
                        help="--policy rl 时加载的策略权重，默认 outputs/rl_policy.pt")
    parser.add_argument("--sample-policy", action="store_true",
                        help="RL 选点按策略分布采样（默认贪心 argmax；长 horizon 的贪心可能退化）")
    parser.add_argument("--noise", type=float, default=0.0,
                        help="采集视角的深度噪声 σ(m)，0=无噪声（与训练/评估一致才可比）")
    parser.add_argument("--seed", type=int, default=0,
                        help="全局随机种子（多 seed 几何统计用），默认 0 = 原基线")
    parser.add_argument("--scene", default="demo", choices=["demo", "primitive", "robot"],
                        help="重建场景：demo=原始房间(球/桌/柱)，primitive=单个几何体，robot=多关节机械臂")
    parser.add_argument("--primitive", default="sphere", choices=["sphere", "box", "cylinder"],
                        help="--scene primitive 时使用的几何体类型")
    parser.add_argument("--voxel", type=float, default=None,
                        help="体素边长 (m)；默认按场景选择（demo 0.08 / primitive 0.06 / robot 0.05）")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.steps = 3
        args.iters_per_step = 40
        args.static_iters = 200
        args.rays = 384
        args.max_gauss = 4000
    if args.explore_steps is not None:
        args.steps = max(1, args.explore_steps) + 1
    print(f"主动探索决策数 = {args.steps - 1}（总训练阶段 {args.steps}）")

    if args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    set_depth_noise(args.noise)
    setup_font()

    out_dir = str(OUTPUTS)
    os.makedirs(out_dir, exist_ok=True)
    policy_ckpt = args.policy_ckpt or os.path.join(out_dir, "rl_policy.pt")
    # 支持逗号分隔的多个权重（集成：logits 取平均），用于压低单策略方差
    ckpts = [c.strip() for c in str(policy_ckpt).split(",") if c.strip()]
    missing = [c for c in ckpts if not os.path.exists(c)]
    if args.policy == "rl" and missing:
        raise SystemExit(f"--policy rl 需要策略权重 {missing}，先跑 python train_rl.py")
    policy_ckpt = ckpts[0] if len(ckpts) == 1 else ckpts
    print(f"主动探索策略 = {args.policy}" + (f"  ({policy_ckpt})" if args.policy == "rl" else ""))

    scene = make_scene(args.scene, args.primitive)
    voxel = float(args.voxel) if args.voxel else float(getattr(scene, "voxel_size", VOXEL_SIZE))
    print(f"场景 = {args.scene}" + (f" ({args.primitive})" if args.scene == "primitive" else "")
          + f"  体素 = {voxel*100:.1f} cm")
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

    vmap = DenseTSDF(ROOM_MIN, ROOM_MAX, voxel, TRUNC, device=str(device))
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
                    greedy=not args.sample_policy,
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
