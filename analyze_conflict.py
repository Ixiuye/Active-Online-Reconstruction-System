"""conflict 指标的三变体对比：底噪来自哪里、哪种写法对噪声更敏感。

在固定 GT 场景上先建好"4 个初始视角"的地图，然后对同一批检验视角在不同 σ 下测 conflict：

  v1 nearest  , weight>1 ,  mean|tsdf|              （当前实现）
  v2 trilinear, weight>1 ,  mean|tsdf|              （去掉体素量化误差）
  v3 trilinear, weight>2 ,  mean max(|tsdf|-0.2, 0) （再加死区，压掉融合底噪）

对每种变体报告 σ=0/1/2/4 cm 的均值，以及"信号/底噪比" = (v(4cm)-v(0)) / v(0)。
比大者更适合当奖励项（同样的传感器噪声能拉开更大差距）。

用法：python analyze_conflict.py
"""
from __future__ import annotations

import json
import os

import numpy as np

from demo.scene import DemoScene, ROOM_MAX, ROOM_MIN, TRUNC, VOXEL_SIZE
from demo.utils import look_at
from demo.voxel_map import DenseTSDF
from run_demo import H_IMG, K, W_IMG

SIGMAS = (0.0, 0.01, 0.02, 0.04)
INIT = [((0.35, 0.35, 1.10), (1.20, 0.90, 0.55)),
        ((2.05, 0.35, 1.10), (1.20, 0.90, 0.55)),
        ((0.35, 1.25, 1.10), (1.20, 0.70, 0.45)),
        ((2.05, 1.25, 1.10), (1.20, 0.70, 0.45))]
PROBE = [((0.90, 0.80, 1.10), (1.60, 1.10, 0.45)),
         ((1.20, 0.30, 1.30), (1.20, 1.20, 0.55)),
         ((0.30, 0.80, 0.90), (1.90, 0.80, 0.50)),
         ((1.20, 1.30, 0.70), (1.20, 0.30, 0.50)),
         ((1.80, 0.80, 1.40), (0.60, 0.80, 0.40)),
         ((1.20, 0.80, 1.50), (1.20, 0.80, 0.10))]


def backproject(depth, R, t):
    fx, fy, cx, cy = K
    H, W = depth.shape
    ys, xs = np.mgrid[0:H, 0:W]
    z = depth.reshape(-1)
    dc = np.stack([(xs.reshape(-1) - cx) / fx, (ys.reshape(-1) - cy) / fy,
                   np.ones(H * W, np.float32)], -1).astype(np.float32)
    n = np.linalg.norm(dc, axis=-1, keepdims=True)
    d_world = (dc / n) @ np.asarray(R)
    p = -(np.asarray(t) @ np.asarray(R))[None] + z[:, None] * d_world
    return p, (z > 0)


def variants(vmap, depth, R, t):
    p, valid = backproject(depth, R, t)
    out = {}
    # v1 最近邻
    ts_n, ws_n, val_n = vmap.lookup_nearest_np(p)
    m1 = valid & val_n & (ws_n > 1.0)
    out["v1_nearest_w1"] = float(np.abs(ts_n[m1]).mean()) if m1.any() else 0.0
    # v2 三线性
    ts_t = vmap.query_np(p)
    wt_t = vmap.query_weight_np(p)
    m2 = valid & (wt_t > 1.0)
    out["v2_trilinear_w1"] = float(np.abs(ts_t[m2]).mean()) if m2.any() else 0.0
    # v3 三线性 + weight>2 + 死区 0.2
    m3 = valid & (wt_t > 2.0)
    out["v3_trilinear_w2_dead"] = float(np.maximum(np.abs(ts_t[m3]) - 0.2, 0).mean()) if m3.any() else 0.0
    out["_n_px"] = int(m2.sum())
    return out


def main():
    scene = DemoScene()
    rng = np.random.default_rng(0)
    vmap = DenseTSDF(ROOM_MIN, ROOM_MAX, VOXEL_SIZE, TRUNC, device="cpu")
    for pos, tgt in INIT:
        R, t = look_at(pos, tgt)
        depth, rgb, _ = scene.render_view(R, t, K, H_IMG, W_IMG)
        vmap.fuse(depth, rgb, R, t, K)

    res = {k: {s: [] for s in SIGMAS} for k in
           ("v1_nearest_w1", "v2_trilinear_w1", "v3_trilinear_w2_dead")}
    npx = 0
    for pos, tgt in PROBE:
        R, t = look_at(pos, tgt)
        depth, rgb, _ = scene.render_view(R, t, K, H_IMG, W_IMG)
        for s in SIGMAS:
            d = depth.copy()
            if s > 0:
                hit = d > 0
                noise = rng.normal(0.0, s, d.shape).astype(np.float32)
                d = np.where(hit, np.maximum(d + noise, 0.05), 0.0).astype(np.float32)
            v = variants(vmap, d, R, t)
            npx = v["_n_px"]
            for k in res:
                res[k][s].append(v[k])

    print(f"检验视角 {len(PROBE)} 个，参考地图来自 {len(INIT)} 个初始视角；"
          f"每帧参与统计像素约 {npx}")
    print(f"{'variant':22s} " + " ".join(f"σ={s*100:.0f}cm" for s in SIGMAS) + "   信号(4-0)  信号/底噪")
    table = {}
    for k, per_s in res.items():
        means = {f"{s}": float(np.mean(v)) for s, v in per_s.items()}
        floor = means["0.0"]
        sig = means["0.04"] - floor
        ratio = sig / floor if floor > 1e-9 else float("inf")
        table[k] = {"means": means, "floor": floor, "signal_4cm": sig, "signal_over_floor": ratio}
        row = " ".join(f"{means[f'{s}']:8.4f}" for s in SIGMAS)
        print(f"{k:22s} {row}   {sig:8.4f}  {ratio:8.3f}")

    best = max(table, key=lambda k: table[k]["signal_over_floor"])
    print(f"\n对噪声最敏感的变体: {best}")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "conflict_metric_study.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"sigmas": SIGMAS, "n_probe": len(PROBE), "n_px": npx,
                   "table": table, "best": best}, f, ensure_ascii=False, indent=2)
    print(f"已写入: {out}")


if __name__ == "__main__":
    main()
