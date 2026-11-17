"""覆盖率上限实验：50% 是"方法不行"还是"这个协议下的可见性上限"？

做法：在同一个 GT 场景、同一个 8 cm 体素 / w>1 / |tsdf|<0.6 口径下，用不同的视角集合融合，
看覆盖率能到多少。视角集合包括：

  demo9        4 个初始视角 + 启发式选的 5 个（= 现在 run_demo 的配置，9 视角）
  grid144      候选表的全部 144 个位姿（4x3 位置 x 12 个手挑目标）
  random200    free space 里均匀采样 200 个位姿（含随机注视点）
  floorceil    200 个位姿，但注视点专门对着地板/天花板/墙根（探"几何上限"，看死角有多大）
  dense600     再翻三倍的均匀采样（看是否饱和）

输出：每个集合的覆盖率、以及"融合后仍完全没被任何视角看到"的 GT 表面体素占比
（= 不可见上限，任何策略都补不回来）。

用法：python analyze_coverage.py
"""
from __future__ import annotations

import json
import os

import numpy as np

from ..viewpoint.heuristic import candidate_poses, sample_utility_targets, select_nbv
from ..viewpoint.rl import INIT_POOL, sample_candidates
from ..common.scene import ROOM_MAX, ROOM_MIN, TRUNC, VOXEL_SIZE, DemoScene
from ..common.utils import look_at
from ..reconstruction.voxel_map import DenseTSDF
from ..common.camera import H_IMG, K, W_IMG
from ..common.paths import OUTPUTS


def new_map():
    return DenseTSDF(ROOM_MIN, ROOM_MAX, VOXEL_SIZE, TRUNC, device="cpu")


def fuse(vmap, scene, poses):
    for R, t in poses:
        depth, rgb, _ = scene.render_view(R, t, K, H_IMG, W_IMG)
        vmap.fuse(depth, rgb, R, t, K)


def coverage(vmap, gt_mask):
    obs = vmap.surface_mask_np().reshape(-1) & gt_mask
    return float(obs.sum() / max(gt_mask.sum(), 1))


def heuristic_sequence(scene, vmap, n=5, seed=0):
    """复现 run_demo 里启发式的选点，拿到那 5 个主动视角。"""
    rng = np.random.default_rng(seed)
    poses, visited = [], []
    for pos, tgt in INIT_POOL[:4]:
        R, t = look_at(pos, tgt)
        poses.append((R, t))
        depth, rgb, _ = scene.render_view(R, t, K, H_IMG, W_IMG)
        vmap.fuse(depth, rgb, R, t, K)          # 初始视角必须也融进地图
        fwd = (np.asarray(tgt, np.float32) - np.asarray(pos, np.float32))
        visited.append((np.asarray(pos, np.float32), fwd / np.linalg.norm(fwd)))
    for _ in range(n):
        pts, wts = sample_utility_targets(vmap, rng)
        cands = candidate_poses(vmap)
        pose, _ = select_nbv(cands, pts, wts, vmap, K, H_IMG, W_IMG,
                             current_pos=visited[-1][0], visited=visited)
        if pose is None:
            break
        R, t, p, tg = pose
        poses.append((R, t))
        fwd = tg - p
        visited.append((p, fwd / np.linalg.norm(fwd)))
        depth, rgb, _ = scene.render_view(R, t, K, H_IMG, W_IMG)
        vmap.fuse(depth, rgb, R, t, K)
    return poses


def main():
    """本脚本无命令行参数（直接运行即可）。"""
    scene = DemoScene()
    gt = scene.gt_surface_voxel_mask(np.array(ROOM_MIN, np.float32),
                                     np.ceil((np.array(ROOM_MAX) - np.array(ROOM_MIN)) / VOXEL_SIZE).astype(int),
                                     VOXEL_SIZE).reshape(-1)
    total = int(gt.sum())
    res = {}

    print(f"GT 表面带体素（|SDF| < 0.75×8cm = 6cm）= {total} 个")

    # --- demo9：4 初始 + 5 启发式
    vm = new_map()
    poses = heuristic_sequence(scene, vm)
    res["demo9_heuristic"] = {"n_views": len(poses), "coverage_pct": 100 * coverage(vm, gt)}

    # --- grid144
    vm = new_map()
    fuse(vm, scene, [look_at(p, t) for p, t in INIT_POOL[:4]])
    cands = candidate_poses(vm)
    vm2 = new_map()
    fuse(vm2, scene, [look_at(p, t) for p, t in INIT_POOL[:4]] + [(c[0], c[1]) for c in cands])
    res["grid144"] = {"n_views": 4 + len(cands), "coverage_pct": 100 * coverage(vm2, gt)}

    # --- random / floor-ceil / dense
    for name, n, directed in (("random200", 200, False), ("floorceil200", 200, True),
                              ("dense600", 600, False), ("dense1200", 1200, False)):
        rng = np.random.default_rng(7)
        vm = new_map()
        fuse(vm, scene, [look_at(p, t) for p, t in INIT_POOL[:4]])
        poses = []
        per_batch = 64
        while len(poses) < n:
            for R, t, p, tg in sample_candidates(vm, rng, k=per_batch):
                if directed:
                    # 注视点换到地板(z=3cm)或天花板(z=157cm)，专挑角落表面
                    z = rng.choice([0.03, 1.57])
                    tg = np.array([rng.uniform(0.05, ROOM_MAX[0] - 0.05),
                                   rng.uniform(0.05, ROOM_MAX[1] - 0.05), z], np.float32)
                    R, t = look_at(p, tg)
                poses.append((R, t))
                depth, rgb, _ = scene.render_view(R, t, K, H_IMG, W_IMG)
                vm.fuse(depth, rgb, R, t, K)
                if len(poses) >= n:
                    break
        res[name] = {"n_views": 4 + len(poses), "coverage_pct": 100 * coverage(vm, gt),
                     "unseen_voxels": int((gt & ~vm.surface_mask_np().reshape(-1)).sum())}

    print(f"\n{'视角集':16s} {'视角数':>6s} {'覆盖率':>8s} {'仍不可见体素':>12s}")
    for k, v in res.items():
        unseen = v.get("unseen_voxels", "")
        print(f"{k:16s} {v['n_views']:6d} {v['coverage_pct']:7.1f}% {unseen!s:>12s}")

    # --- 覆盖率 vs 视角数：90% 需要多少视角？
    print("\n覆盖率 vs 视角数（4 初始 + N 个均匀采样视角）:")
    rng = np.random.default_rng(11)
    vm = new_map()
    fuse(vm, scene, [look_at(p, t) for p, t in INIT_POOL[:4]])
    pool = []
    for _ in range(3):
        for R, t, p, tg in sample_candidates(vm, rng, k=64):
            pool.append((R, t))
    sweep = {}
    done = 0
    for target in (9, 20, 40, 80, 160, 320):
        while done < target - 4 and done < len(pool):
            R, t = pool[done]
            depth, rgb, _ = scene.render_view(R, t, K, H_IMG, W_IMG)
            vm.fuse(depth, rgb, R, t, K)
            done += 1
        cov = 100 * coverage(vm, gt)
        sweep[target] = cov
        print(f"  视角 4+{done:4d} = {4+done:4d} 个 -> 覆盖 {cov:5.1f}%")
    res["_sweep_vs_views"] = sweep

    out = str(OUTPUTS / "coverage_ceiling.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"gt_surface_voxels": total, "results": res}, f, ensure_ascii=False, indent=2)
    print(f"\n已写入: {out}")


if __name__ == "__main__":
    main()
