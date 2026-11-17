"""几何不确定度驱动的主动探索（NBV）。

候选视点由房间内的自由位置 × 注视目标生成；效用函数统计视锥内、未被当前
TSDF 遮挡的“未知 frontier 体素 + 低置信表面体素”，并扣除路径代价。
"""
from __future__ import annotations

import numpy as np

from ..common.scene import ROOM_MIN, ROOM_MAX
from ..common.utils import look_at


def sample_utility_targets(vmap, rng, n_unknown=2500, n_low=1200):
    """采样待观测目标点：未知 frontier + 低置信表面体素。"""
    unknown_idx = vmap.frontier_unknown_flat()
    if unknown_idx.size > n_unknown:
        unknown_idx = rng.choice(unknown_idx, size=n_unknown, replace=False)

    low_mask = (
        (vmap.weight > 0.3)
        & (vmap.weight < 1.5)
        & (np.abs(vmap.tsdf) < 0.6)
    )
    low_idx = np.flatnonzero(low_mask.reshape(-1))
    if low_idx.size > n_low:
        low_idx = rng.choice(low_idx, size=n_low, replace=False)

    pts = np.concatenate([vmap.centers[unknown_idx], vmap.centers[low_idx]], axis=0)
    w = np.concatenate(
        [np.ones(len(unknown_idx), np.float32), np.full(len(low_idx), 0.6, np.float32)]
    )
    return pts, w


def candidate_poses(vmap, eye_z=1.05, margin=0.30):
    xs = np.linspace(ROOM_MIN[0] + margin, ROOM_MAX[0] - margin, 4)
    ys = np.linspace(ROOM_MIN[1] + margin, ROOM_MAX[1] - margin, 3)
    positions = [np.array([x, y, eye_z], np.float32) for x in xs for y in ys]

    p = np.stack(positions, 0)
    ts, ws, valid = vmap.lookup_nearest_np(p)
    keep = ~(valid & (ws > 0.5) & (ts < -0.2))
    positions = [pos for pos, k in zip(positions, keep) if k]

    targets = [
        np.array([1.20, 0.80, 0.55], np.float32),
        np.array([0.75, 0.95, 0.35], np.float32),
        np.array([1.65, 1.15, 0.35], np.float32),
        np.array([1.95, 0.45, 0.80], np.float32),
        np.array([1.20, 0.15, 0.60], np.float32),
        np.array([1.20, 1.45, 0.60], np.float32),
        np.array([0.20, 0.80, 0.60], np.float32),
        np.array([2.20, 0.80, 0.60], np.float32),
        np.array([1.20, 0.80, 0.15], np.float32),
        np.array([1.20, 0.80, 1.50], np.float32),
        np.array([0.60, 0.40, 1.50], np.float32),
        np.array([1.80, 1.20, 1.50], np.float32),
    ]

    poses = []
    for pos in positions:
        for tgt in targets:
            R, t = look_at(pos, tgt)
            poses.append((R, t, pos, tgt))
    return poses


def score_pose(R, t, pos, points, weights, vmap, K, H, W):
    C = -np.asarray(t) @ np.asarray(R)
    v = points - C[None]
    dist = np.linalg.norm(v, axis=-1)
    ok = dist > 0.2
    d = v / np.maximum(dist, 1e-6)[:, None]
    dc = d @ np.asarray(R).T
    z = dc[:, 2]
    u = K[0] * dc[:, 0] / np.maximum(z, 1e-6) + K[2]
    vv = K[1] * dc[:, 1] / np.maximum(z, 1e-6) + K[3]
    in_fov = ok & (z > 0.15) & (u >= 0) & (u < W) & (vv >= 0) & (vv < H)

    blocked = np.zeros(len(points), bool)
    for frac in (0.25, 0.45, 0.65, 0.80):
        q = C[None] + d * (dist * frac)[:, None]
        ts, ws, val = vmap.lookup_nearest_np(q)
        blocked |= val & (ws > 0.5) & (ts < -0.2)

    visible = in_fov & ~blocked
    return float(weights[visible].sum()), visible


def select_nbv(poses, points, weights, vmap, K, H, W, current_pos, visited):
    best = None
    best_score = -1e9
    for R, t, pos, tgt in poses:
        forward = tgt - pos
        forward = forward / max(np.linalg.norm(forward), 1e-8)
        dup = False
        for vpos, vfwd in visited:
            if np.linalg.norm(pos - vpos) < 0.25 and float(forward @ vfwd) > 0.92:
                dup = True
                break
        if dup:
            continue
        score, _ = score_pose(R, t, pos, points, weights, vmap, K, H, W)
        score -= 30.0 * float(np.linalg.norm(pos - current_pos))
        if score > best_score:
            best_score = score
            best = (R, t, pos, tgt)
    return best, best_score
