"""TSDF 追迹渲染：从地图自身渲染深度/颜色，用来算"重建误差"（奖励或评估）。

为什么需要它：RL 的奖励若想用光度/深度误差，必须能在**不训练高斯**的前提下快速拿到
"当前地图渲染出来长什么样"。高斯渲染要训练（每步 417 ms），太贵；TSDF 追迹每帧只要几毫秒，
可以放进训练循环。它同时是可部署的（只用地图字段，不碰 GT）。

做法：球面行进（sphere tracing）。TSDF 归一化到 [-1,1]，乘回截断距离就是"到表面的近似距离"，
因此每步可以走 max(tsdf·trunc, voxel/2)，命中判断 tsdf < hit_thresh。
颜色取命中体素附近的 color_sum/color_w（已归一化的表面色）。
"""
from __future__ import annotations

import numpy as np

from ..common.utils import pixel_rays


def render_tsdf(vmap, R, t, K, height, width, stride=8, max_range=6.0,
                hit_thresh=0.12, max_steps=96):
    """从位姿 (R, t) 追迹渲染地图。

    stride: 像素下采样（8 → 每 8 像素一条光线，成本降 64 倍）
    返回 depth (H,W)（未命中为 0）、color (H,W,3)、hit (H,W)
    """
    fx, fy, cx, cy = K
    ys = np.arange(0, height, stride, dtype=np.float32)
    xs = np.arange(0, width, stride, dtype=np.float32)
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    d_cam = np.stack([(gx.reshape(-1) - cx) / fx, (gy.reshape(-1) - cy) / fy,
                      np.ones(gx.size, np.float32)], -1).astype(np.float32)
    d_cam /= np.linalg.norm(d_cam, axis=-1, keepdims=True)
    d_world = d_cam @ np.asarray(R, np.float32)          # (M,3)
    origin = -(np.asarray(t, np.float32) @ np.asarray(R, np.float32))

    tt = np.full(len(d_world), 0.05, np.float32)
    hit = np.zeros(len(d_world), bool)
    voxel = vmap.voxel
    for _ in range(max_steps):
        p = origin[None] + tt[:, None] * d_world
        tsdf = vmap.query_np(p)
        wt = vmap.query_weight_np(p)
        known = wt > 0.1                       # 未观测体素的 tsdf 也是 0，必须先判"已知"
        newly = (~hit) & known & (tsdf < hit_thresh)
        hit |= newly
        # 已知区域球面行进；未知区域只能一格子一格子推
        step = np.where(known, np.maximum(np.abs(tsdf) * vmap.trunc, 0.5 * voxel), voxel)
        tt = np.where(hit, tt, tt + step)
        if hit.all() or (tt > max_range).all():
            break
    valid = hit & (tt < max_range)
    p_hit = origin[None] + tt[:, None] * d_world

    # 颜色：命中点最近体素的累积表面色
    csum = vmap.color_sum.reshape(-1, 3)
    cw = vmap.color_w.reshape(-1)
    idx = np.floor((p_hit - vmap.gmin) / voxel).astype(np.int64)
    inside = np.all((idx >= 0) & (idx < np.array(vmap.dims)), axis=-1)
    idx = np.clip(idx, 0, np.array(vmap.dims) - 1)
    flat = np.ravel_multi_index(idx.T, vmap.dims)
    col = np.where(cw[flat][:, None] > 0, csum[flat] / np.maximum(cw[flat], 1e-6)[:, None],
                   np.array([0.5, 0.5, 0.5], np.float32))
    col = np.where((valid & inside)[:, None], col, 0.0)

    depth_g = np.zeros(gx.size, np.float32)
    depth_g[valid] = tt[valid]
    col = col * valid[:, None]
    shape = gx.shape
    return (depth_g.reshape(shape), col.reshape(shape + (3,)).astype(np.float32),
            valid.reshape(shape))


def map_error(vmap, poses, gt_depth, gt_color, K, height, width, stride=8,
              col_weight=20.0, err_cap_cm=200.0):
    """在固定检验视角上算地图的三个量（比"平均深度误差"稳得多）。

    返回 dict:
      hit_rate   地图能在该视角命中的像素比例（地图完整性）
      med_err_cm 命中像素上 |Δdepth| 的中位数（对"透过未知区打到远处"不敏感）
      photo      命中像素上的颜色 MSE × col_weight

    不用平均值的原因：地图稀疏时大量射线会穿过未观测空间打到远处表面，
    平均误差会被这些尾巴主导（实测 4 个初始视角时平均 75 cm，而中位数只有几 cm）。
    """
    errs, photo_terms, hits, tot = [], [], 0, 0
    for pose, d_gt, c_gt in zip(poses, gt_depth, gt_color):
        d_pred, c_pred, hit = render_tsdf(vmap, pose[0], pose[1], K, height, width, stride=stride)
        gt_valid = d_gt > 0
        tot += int(gt_valid.sum())
        hits += int((hit & gt_valid).sum())
        m = hit & gt_valid
        if m.any():
            errs.append(np.abs(d_pred[m] - d_gt[m]) * 100.0)
            photo_terms.append(((c_pred[m] - c_gt[m]) ** 2).mean())
    if tot == 0:
        return {"hit_rate": 0.0, "med_err_cm": 0.0, "photo": 0.0}
    e = np.concatenate(errs) if errs else np.zeros(1, np.float32)
    return {
        "hit_rate": hits / tot,
        "med_err_cm": float(np.median(np.minimum(e, err_cap_cm))),
        "photo": col_weight * float(np.mean(photo_terms)) if photo_terms else 0.0,
    }
