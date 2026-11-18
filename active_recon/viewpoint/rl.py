"""强化学习驱动的主动探索 v2：均匀采样的候选位姿 + 带深度可分性的奖励。

与原 `active.select_nbv`（人工效用）及 v1（人工候选网格）的区别：

1. 动作空间不再人工设定。每个决策在房间自由空间里**均匀重采样 K 个位姿**
   （位置：TSDF 自由/未知区里拒绝采样；朝向：朝空间内均匀采样的注视点），
   策略对这 K 个候选打分后选一个。位姿表不再是 4×3 网格 × 12 个手挑目标，
   眼高也不再固定 1.05 m。K 是算力旋钮，不是先验。
2. 奖励有三项：新增**表面**体素（不是全部已知体素，修掉 v1 奖励与评测口径不同向的病）、
   深度可分性（近且正对的观测更准）、路程。
   深度可分性是完全地图内的量：对新视角的每个深度像素查 TSDF，
   用 exp(-(z/z0)²) 与入射角余弦加权，并对"新深度与已有地图表面冲突"的比例罚分。
   全程不需要 GT。

动作 → 位姿参数化：a = (p, t)，p 是相机位置，t 是注视点，R = look_at(p, t)。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from pathlib import Path

from .heuristic import candidate_poses, sample_utility_targets
from ..common.scene import DemoScene, ROOM_MAX, ROOM_MIN, TRUNC, VOXEL_SIZE
from ..reconstruction.tsdf_render import map_error, render_tsdf
from ..common.utils import look_at
from ..reconstruction.voxel_map import DenseTSDF

FEATURE_NAMES = [
    "gain_unknown",      # 视锥内可见 frontier 权重和
    "gain_lowconf",      # 视锥内可见低置信表面权重和
    "gain_per_dist",     # gain_unknown / (路径 + 0.3)
    "visible_frac",      # 可见目标点比例
    "visible_dist",      # 可见目标点平均距离
    "path_len",          # 到当前位置的直线距离
    "novelty_pos",       # 到最近已访问位置的距离
    "novelty_dir",       # 1 - max cos(视向, 已访问视向)
    "heading_cost",      # 1 - cos(运动方向, 视向)
    "wall_dist",         # 到最近墙面距离
    "seen_ratio",        # 视向上已观测为自由的比例
    "depth_quality",     # 该视角的深度可分性估计（近 + 正对，见 _point_quality）
    "height_dev",        # 眼高相对当前高度的偏离（鼓励平滑的高度变化）
    "bias",
]
N_FEATURES = len(FEATURE_NAMES)

TARGET_COUNTS = (400, 220)   # 训练/部署必须一致，否则 gain_* 量级不同分布
DEPTH_Z0 = 1.5               # 深度可分性的距离尺度（m）：z 越大越不准


# ---------------------------------------------------------------- 候选采样
def sample_candidates(vmap, rng, k=64, margin=0.15, z_range=(0.45, 1.45),
                      max_tries=40):
    """在自由空间里均匀采样 k 个候选位姿（位置 + 均匀注视点），无人工网格。

    位置：房间内缩 margin 后均匀撒点，保留 TSDF 里"不落在障碍内"的点
          （未知或自由都行，因为未知区正是要看的地方）。
    朝向：注视点在房间包围盒内均匀采样，因此朝向覆盖全空间而不只是手挑的 12 个方向。
    """
    lo = np.asarray(ROOM_MIN, np.float32) + margin
    hi = np.asarray(ROOM_MAX, np.float32) - margin
    lo[2] = z_range[0]
    hi[2] = z_range[1]

    pos = []
    tries = 0
    while len(pos) < k and tries < max_tries * k:
        tries += 1
        batch = rng.uniform(lo, hi, size=(k, 3)).astype(np.float32)
        ts, ws, val = vmap.lookup_nearest_np(batch)
        # 只排除"已知且在表面后方"的点（障碍内部/墙里），其余（自由、未知）都可用
        keep = ~(val & (ws > 0.5) & (ts < -0.15))
        pos.extend([p for p, ok in zip(batch, keep) if ok])
    pos = np.stack(pos[:k], 0)

    tgt = rng.uniform(np.asarray(ROOM_MIN, np.float32) + 0.05,
                      np.asarray(ROOM_MAX, np.float32) - 0.05, size=pos.shape).astype(np.float32)
    cands = []
    for p, t in zip(pos, tgt):
        if np.linalg.norm(t - p) < 0.25:            # 注视点离相机太近，跳过
            t = p + np.array([0.6, 0.0, -0.2], np.float32)
        R, tt = look_at(p, t)
        cands.append((R, tt, p.astype(np.float32), t.astype(np.float32)))
    return cands


# ---------------------------------------------------------------- 深度可分性
def _point_quality(points, dist, dirs_world, grad, trunc, z0=DEPTH_Z0):
    """每个目标点的"观测质量"估计：距离越近越好、视线越正对表面越好。

    grad: 目标点处的 TSDF 梯度（未归一化即可，内部会归一化）。
    """
    g = grad / np.maximum(np.linalg.norm(grad, axis=-1, keepdims=True), 1e-8)
    cos_i = np.abs((dirs_world * g).sum(-1))        # 视线与表面法向的 |cos|
    near = 1.0 / (1.0 + (dist / z0) ** 2)
    return np.clip(0.25 + 0.75 * cos_i, 0.0, 1.0) * near


def view_quality(vmap, depth, R, t, K, z0=DEPTH_Z0):
    """对刚采集的一帧算深度可分性 + 与已有地图的一致程度（纯地图量，不用 GT）。

    返回:
      quality:  表面像素上的平均可分性（近 + 正对），0~1
      conflict: 已建立表面体素（weight>1）上 |TSDF| 的均值（归一化后 0~1），
                即"新深度离地图零水平集有多远"；噪声越大、遮挡/透过观测越严重该值越高
      n_surf:   落在表面带上的像素数
    """
    fx, fy, cx, cy = K
    H, W = depth.shape
    ys, xs = np.mgrid[0:H, 0:W]
    z = depth.reshape(-1)
    valid = z > 0
    if valid.sum() == 0:
        return 0.0, 0.0, 0
    dc = np.stack([(xs.reshape(-1) - cx) / fx, (ys.reshape(-1) - cy) / fy,
                   np.ones(H * W, np.float32)], -1).astype(np.float32)
    n = np.linalg.norm(dc, axis=-1, keepdims=True)
    d_world = (dc / n) @ np.asarray(R)                       # 单位方向
    p = -(np.asarray(t) @ np.asarray(R))[None] + z[:, None] * d_world

    ts, ws, val = vmap.lookup_nearest_np(p)
    surf = valid & val & (np.abs(ts) < 1.0) & (ws > 0.0)
    grad = vmap.query_grad_np(p[surf]) if surf.any() else np.zeros((0, 3), np.float32)
    q = _point_quality(p[surf], z[surf], d_world[surf], grad, TRUNC, z0) if surf.any() else np.zeros(0)
    # 冲突：只在"地图已经比较确信是表面"的像素上算偏离度，避免未知区干扰
    conf = valid & val & (ws > 1.0)
    conflict = float(np.clip(np.abs(ts[conf]), 0.0, 1.0).mean()) if conf.any() else 0.0
    return float(q.mean()) if q.size else 0.0, conflict, int(surf.sum())


# ---------------------------------------------------------------- 特征
def candidate_features(vmap, cands, points, weights, kinds, current_pos, current_fwd,
                       visited, K, H, W, n_occl=4, normals=None):
    """向量化计算全部候选视点的特征（不含 identity，因此候选集可每次重采样）。

    normals: 目标点处的单位法向 (M,3)，用于聚合 candidate 的 depth_quality 特征
             （近 + 正对）；必须训练与部署一致，否则该维会退化成常量。
    """
    points = np.asarray(points, np.float32)
    weights = np.asarray(weights, np.float32)
    kinds = np.asarray(kinds, np.float32)
    N = len(cands)
    M = len(points)

    R = np.stack([c[0] for c in cands]).astype(np.float32)
    t = np.stack([c[1] for c in cands]).astype(np.float32)
    pos = np.stack([c[2] for c in cands]).astype(np.float32)
    tgt = np.stack([c[3] for c in cands]).astype(np.float32)

    C = -np.einsum("nj,nji->ni", t, R)                          # 相机中心 = -R^T t
    v = points[None] - C[:, None]
    dist = np.linalg.norm(v, axis=-1)
    d = v / np.maximum(dist, 1e-6)[..., None]
    dc = np.einsum("nij,nmj->nmi", R, d)                        # 相机系方向

    fx, fy, cx, cy = K
    z = dc[..., 2]
    u = fx * dc[..., 0] / np.maximum(z, 1e-6) + cx
    vv = fy * dc[..., 1] / np.maximum(z, 1e-6) + cy
    in_fov = (dist > 0.2) & (z > 0.15) & (u >= 0) & (u < W) & (vv >= 0) & (vv < H)

    blocked = np.zeros((N, M), bool)
    for frac in np.linspace(0.25, 0.85, n_occl):
        q = (C[:, None] + d * (dist * frac)[..., None]).reshape(-1, 3)
        ts, ws, val = vmap.lookup_nearest_np(q)
        blocked |= (val & (ws > 0.5) & (ts < -0.2)).reshape(N, M)
    visible = in_fov & ~blocked

    w_vis = visible * weights[None]
    gain_unknown = (w_vis * kinds[None]).sum(1)
    gain_low = (w_vis * (1.0 - kinds)[None]).sum(1)
    n_vis = visible.sum(1)
    vis_dist = np.where(n_vis > 0, (visible * dist).sum(1) / np.maximum(n_vis, 1), 0.0)

    view_dir = tgt - pos
    view_dir /= np.maximum(np.linalg.norm(view_dir, axis=-1, keepdims=True), 1e-8)
    if normals is not None:
        # 近 + 正对：与 view_quality 里的定义一致（距离项 × |cos 视线法向|）
        cos_i = np.abs(np.einsum("nmj,mj->nm", d, np.asarray(normals, np.float32)))
        near = 1.0 / (1.0 + (dist / DEPTH_Z0) ** 2)
        qp = np.clip(0.25 + 0.75 * cos_i, 0.0, 1.0) * near
        depth_q = (visible * qp).sum(1) / np.maximum(n_vis, 1)
    else:
        depth_q = np.zeros(N, np.float32)

    path_len = np.linalg.norm(pos - current_pos[None], axis=-1)
    move_dir = pos - current_pos[None]
    move_norm = np.linalg.norm(move_dir, axis=-1, keepdims=True)
    move_dir = np.where(move_norm > 1e-6, move_dir / np.maximum(move_norm, 1e-8), view_dir)
    heading_cost = 1.0 - (move_dir * view_dir).sum(1)

    if visited:
        vpos = np.stack([vp for vp, _ in visited]).astype(np.float32)
        vfwd = np.stack([vf for _, vf in visited]).astype(np.float32)
        novelty_pos = np.linalg.norm(pos[:, None] - vpos[None], axis=-1).min(1)
        novelty_dir = 1.0 - (view_dir @ vfwd.T).max(1)
    else:
        novelty_pos = np.full(N, 5.0, np.float32)
        novelty_dir = np.ones(N, np.float32)

    wall_dist = np.minimum(
        np.minimum(pos[:, 0] - ROOM_MIN[0], ROOM_MAX[0] - pos[:, 0]),
        np.minimum(
            np.minimum(pos[:, 1] - ROOM_MIN[1], ROOM_MAX[1] - pos[:, 1]),
            np.minimum(pos[:, 2] - ROOM_MIN[2], ROOM_MAX[2] - pos[:, 2]),
        ),
    )

    ts_ray = np.linspace(0.4, 2.0, 8)
    q = (pos[:, None, :] + view_dir[:, None, :] * ts_ray[None, :, None]).reshape(-1, 3)
    ts, ws, val = vmap.lookup_nearest_np(q)
    seen = (val & (ws > 0.5) & (ts > 0.3)).reshape(N, len(ts_ray)).mean(1)

    height_dev = np.abs(pos[:, 2] - current_pos[2])

    feats = np.stack([
        gain_unknown, gain_low, gain_unknown / (path_len + 0.3),
        n_vis / max(M, 1), vis_dist, path_len,
        novelty_pos, novelty_dir, heading_cost, wall_dist,
        seen, depth_q, height_dev, np.ones(N, np.float32),
    ], axis=1).astype(np.float32)

    # 去重 mask：位置 < 0.2 m 且朝向 cos > 0.9（可重采样，所以必须每次按已访问位姿屏蔽）
    valid = np.ones(N, bool)
    if visited:
        valid &= ~((novelty_pos < 0.20) & (novelty_dir < 1.0 - 0.90))
    return feats, valid, {"gain_unknown": gain_unknown, "n_visible": n_vis}


# ---------------------------------------------------------------- 策略
class CandidatePolicy(nn.Module):
    """集合打分策略：共享 MLP 编码每个候选 -> logit；池化 -> 状态价值。"""

    def __init__(self, n_feat=N_FEATURES, hidden=64):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(n_feat, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.score = nn.Linear(hidden, 1)
        self.value = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1),
        )

    def forward(self, feats, mask=None):
        h = self.enc(feats)
        logit = self.score(h).squeeze(-1)
        if mask is not None:
            logit = logit.masked_fill(~mask, -1e9)
        pooled = torch.cat([h.mean(-2), h.amax(-2)], dim=-1)
        return logit, self.value(pooled).squeeze(-1)


def save_policy(path, policy, meta):
    torch.save({"state_dict": policy.state_dict(), "features": FEATURE_NAMES,
                "meta": meta, "hidden": policy.score.in_features}, path)


def load_policy(path, device="cpu"):
    blob = torch.load(path, map_location=device, weights_only=False)
    policy = CandidatePolicy(len(blob["features"]), hidden=blob["hidden"]).to(device)
    policy.load_state_dict(blob["state_dict"])
    policy.eval()
    return policy


# ---------------------------------------------------------------- 环境
INIT_POOL = [
    ((0.35, 0.35, 1.10), (1.20, 0.90, 0.55)),
    ((2.05, 0.35, 1.10), (1.20, 0.90, 0.55)),
    ((0.35, 1.25, 1.10), (1.20, 0.70, 0.45)),
    ((2.05, 1.25, 1.10), (1.20, 0.70, 0.45)),
    ((1.20, 0.30, 1.15), (1.20, 1.20, 0.60)),
    ((1.20, 1.30, 1.15), (1.20, 0.40, 0.60)),
]

# 固定检验视角：重建侧奖励/误差在这些位姿上测，不参与探索，也不在训练轨迹里
PROBE_POSES = [
    ((0.90, 0.80, 1.10), (1.60, 1.10, 0.45)),
    ((1.20, 0.30, 1.30), (1.20, 1.20, 0.55)),
    ((0.30, 0.80, 0.90), (1.90, 0.80, 0.50)),
    ((1.20, 1.30, 0.70), (1.20, 0.30, 0.50)),
    ((1.80, 0.80, 1.40), (0.60, 0.80, 0.40)),
    ((1.20, 0.80, 1.50), (1.20, 0.80, 0.10)),
]


def randomize_scene(rng, jitter=0.12):
    """把物体在房间内小幅挪动/缩放：训练策略见到的场景不再只有那一个。"""
    scene = DemoScene()
    boxes = {
        "sphere_c": (np.array([0.45, 0.40, 0.22], np.float32),
                     np.array([2.05, 1.20, 0.55], np.float32)),
        "table_c": (np.array([0.55, 0.55, 0.20], np.float32),
                    np.array([1.95, 1.20, 0.30], np.float32)),
        "cyl_c": (np.array([0.45, 0.40, 0.60], np.float32),
                  np.array([2.05, 1.20, 0.60], np.float32)),
    }
    for attr, (lo, hi) in boxes.items():
        c = getattr(scene, attr) + rng.uniform(-jitter, jitter, 3).astype(np.float32)
        setattr(scene, attr, np.clip(c, lo, hi).astype(np.float32))
    scene.sphere_r = float(rng.uniform(0.18, 0.26))
    scene.table_h = (scene.table_h * float(rng.uniform(0.85, 1.15))).astype(np.float32)
    return scene


class ExplorationEnv:
    """一次主动探索 episode：随机场景 + 随机初始视角 -> T 步策略决策。

    奖励（全部地图内，不用 GT）：
        r = 1.0 * Δ表面体素/100
          + 0.8 * 视角深度可分性 quality
          - 2.0 * 新深度与地图冲突比例
          - 0.3 * 路径长度
    表面体素 = (weight > 1) & (|tsdf| < 0.6)，与 run_demo.evaluate 的 coverage 同口径，
    因此奖励和最终报告的指标方向一致（v1 用"全部已知体素"，这是它覆盖偏低的原因之一）。
    """

    def __init__(self, K, H, W, steps=6, n_init=4, seed=0, max_targets=TARGET_COUNTS,
                 randomize=True, fixed_init=True, init_jitter=0.06, n_init_range=None,
                 n_cand=64, w_surf=1.0, w_quality=0.8, w_conflict=2.0, w_path=0.3,
                 w_rec=0.0, depth_noise=0.0, n_probe=6, probe_stride=8,
                 w_fb=0.0, fb_stride=8,
                 refine_rounds=0, refine_k=64, refine_topk=8,
                 cov_target=0.0, w_cov=2.0):
        self.K, self.H, self.W = K, H, W
        self.steps = steps
        self.n_init = n_init
        self.randomize = randomize
        self.fixed_init = fixed_init
        self.init_jitter = init_jitter
        self.n_init_range = n_init_range
        self.rng = np.random.default_rng(seed)
        self.n_unknown, self.n_low = max_targets
        self.n_cand = n_cand
        self.w_surf, self.w_quality, self.w_conflict, self.w_path = (
            w_surf, w_quality, w_conflict, w_path)
        # 重建侧奖励：在固定检验视角上测"地图渲染 vs GT"的深度/光度误差，用其下降量当奖励
        self.w_rec = float(w_rec)
        self.w_fb = float(w_fb)
        self.fb_stride = int(fb_stride)
        self.fb_ema = 1.0
        self.n_probe = n_probe
        self.probe_stride = probe_stride
        self.rec_ema = 1.0
        # 采样细化：第一轮 64 个 -> 取 top-k -> 在附近再采 refine_k 个 -> 重打分
        self.refine_rounds = int(refine_rounds)
        self.refine_k = int(refine_k)
        self.refine_topk = int(refine_topk)
        # 覆盖率约束：cov < cov_target 时按缺口罚（拉格朗日式的软约束），
        # 而不是把覆盖率和路程放在同一个标量里加权。cov_target=0 表示不启用。
        self.cov_target = float(cov_target)
        self.w_cov = float(w_cov)
        # 深度噪声 σ（m，逐像素沿光线加噪）。0 = 无噪声。
        # 没有噪声时 conflict 项恒为 0、quality 项也几乎没有区分度，消融会失真。
        self.depth_noise = float(depth_noise)
        self.rng_noise = np.random.default_rng(seed + 991)

    def _render(self, R, t):
        depth, rgb, _ = self.scene.render_view(R, t, self.K, self.H, self.W)
        if self.depth_noise > 0:
            hit = depth > 0
            noise = self.rng_noise.normal(0.0, self.depth_noise, depth.shape).astype(np.float32)
            depth = np.where(hit, np.maximum(depth + noise, 0.05), 0.0).astype(np.float32)
        return depth, rgb

    # ---------- 地图统计 ----------
    def _surface_count(self):
        return int(self.vmap.surface_mask_np().sum())

    def _known(self):
        return int((self.vmap.weight > 0.5).sum())

    def coverage(self):
        """与 run_demo.evaluate 同口径：surface_mask ∩ GT 表面带。"""
        observed = self.vmap.surface_mask_np().reshape(-1)
        return float((observed & self.gt_surface).sum() / max(self.gt_surface.sum(), 1))

    # ---------- episode ----------
    def reset(self, rng=None):
        rng = self.rng if rng is None else rng
        self.scene = randomize_scene(rng) if self.randomize else DemoScene()
        n_init = int(rng.integers(*self.n_init_range, endpoint=True)) \
            if self.n_init_range else self.n_init
        idx = np.arange(n_init) if self.fixed_init \
            else rng.choice(len(INIT_POOL), size=n_init, replace=False)
        self.vmap = DenseTSDF(ROOM_MIN, ROOM_MAX, VOXEL_SIZE, TRUNC, device="cpu")
        self.visited = []
        for i in idx:
            pos, tgt = INIT_POOL[i]
            if self.init_jitter > 0:
                pos = np.asarray(pos, np.float32) + rng.uniform(
                    -self.init_jitter, self.init_jitter, 3).astype(np.float32)
            else:
                pos = np.asarray(pos, np.float32)
            R, t = look_at(pos, tgt)
            depth, rgb = self._render(R, t)
            self.vmap.fuse(depth, rgb, R, t, self.K)
            fwd = (tgt - pos) / np.linalg.norm(tgt - pos)
            self.visited.append((pos, fwd.astype(np.float32)))
        self.current_pos = self.visited[-1][0]
        self.t = 0
        self.surf0 = self._surface_count()
        self.known0 = self._known()
        self.gt_surface = self.scene.gt_surface_voxel_mask(
            self.vmap.gmin, self.vmap.dims, self.vmap.voxel).reshape(-1)
        # 预渲染检验视角的 GT（场景静态，每 episode 只算一次）
        self.probes, self.gt_depth, self.gt_color = [], [], []
        if self.w_rec > 0:
            s = self.probe_stride
            for pos, tgt in PROBE_POSES[: self.n_probe]:
                R, t = look_at(pos, tgt)
                d, c, _ = self.scene.render_view(R, t, self.K, self.H, self.W)
                self.probes.append((R, t))
                self.gt_depth.append(d[::s, ::s].copy())
                self.gt_color.append(c[::s, ::s].copy())
        return self._observe()

    def _frame_error(self, depth, rgb, R, t):
        """当前地图对刚采集这一帧的预测误差（深度 + 光度），不使用 GT。

        用 TSDF 球面追迹在当前视角渲染地图，再与实测 RGB-D 比较。
        stride 下采样以控制成本；返回 (hit_rate, 中位深度误差 cm, 光度 MSE)。
        """
        d_pred, c_pred, hit = render_tsdf(
            self.vmap, R, t, self.K, self.H, self.W, stride=self.fb_stride
        )
        s = self.fb_stride
        d = depth[::s, ::s]
        c = rgb[::s, ::s]
        valid = hit & (d > 0)
        if not valid.any():
            return 0.0, 0.0, 0.0
        err = np.minimum(np.abs(d_pred[valid] - d[valid]) * 100.0, 200.0)
        photo = float(np.mean((c_pred[valid] - c[valid]) ** 2))
        return float(hit.mean()), float(np.median(err)), photo

    def _probe_error(self):
        """地图在检验视角上的 {hit_rate, med_err_cm, photo}。纯地图侧渲染，GT 只用来对照。"""
        if not self.probes:
            return {"hit_rate": 0.0, "med_err_cm": 0.0, "photo": 0.0}
        return map_error(self.vmap, self.probes, self.gt_depth, self.gt_color,
                         self.K, self.H, self.W, stride=self.probe_stride)

    def features_for(self, cands):
        """给任意候选集算特征（细化采样要用）。"""
        return candidate_features(self.vmap, cands, self._pts, self._wts, self._kinds,
                                  self.current_pos, self.visited[-1][1], self.visited,
                                  self.K, self.H, self.W, normals=self._normals)

    def refine_candidates(self, centers, rng, k, pos_sigma=0.18, tgt_sigma=0.12):
        """在给定中心附近局部重采（薄封装，实现见模块级 refine_near）。"""
        return refine_near(self.vmap, centers, rng, k, pos_sigma, tgt_sigma)

    def _observe(self):
        pts, wts = sample_utility_targets(self.vmap, self.rng,
                                          n_unknown=self.n_unknown, n_low=self.n_low)
        kinds = (wts > 0.8).astype(np.float32)     # frontier 1.0 / 低置信表面 0.6
        grad = self.vmap.query_grad_np(pts)
        normals = grad / np.maximum(np.linalg.norm(grad, axis=-1, keepdims=True), 1e-8)
        self.cands = sample_candidates(self.vmap, self.rng, k=self.n_cand)
        feats, valid, _ = candidate_features(
            self.vmap, self.cands, pts, wts, kinds,
            self.current_pos, self.visited[-1][1], self.visited, self.K, self.H, self.W,
            normals=normals)
        if not valid.any():
            valid[:] = True
        self._pts, self._wts, self._kinds, self._normals = pts, wts, kinds, normals
        return feats, valid

    def step(self, action):
        R, t, pos, tgt = self.cands[int(action)]
        path = float(np.linalg.norm(pos - self.current_pos))
        err0 = self._probe_error() if self.w_rec > 0 else None
        depth, rgb = self._render(R, t)
        fb0 = self._frame_error(depth, rgb, R, t) if self.w_fb > 0 else None
        self.vmap.fuse(depth, rgb, R, t, self.K)
        fwd = tgt - pos
        fwd = fwd / max(np.linalg.norm(fwd), 1e-8)
        self.visited.append((pos.astype(np.float32), fwd.astype(np.float32)))
        self.current_pos = pos.astype(np.float32)

        surf_now = self._surface_count()
        d_surf = surf_now - self.surf0
        self.surf0 = surf_now
        known_now = self._known()
        d_known = known_now - self.known0
        self.known0 = known_now

        quality, conflict, n_surf = view_quality(self.vmap, depth, R, t, self.K)
        # 重建侧奖励：完整性(hit_rate)上升 + 中位深度误差下降 + 光度误差下降，EMA 归一化到 O(1)
        r_rec, err1 = 0.0, {"hit_rate": 0.0, "med_err_cm": 0.0, "photo": 0.0}
        if self.w_rec > 0 and err0 is not None:
            err1 = self._probe_error()
            raw = (100.0 * (err1["hit_rate"] - err0["hit_rate"])
                   + (err0["med_err_cm"] - err1["med_err_cm"])
                   + (err0["photo"] - err1["photo"]))
            self.rec_ema = 0.95 * self.rec_ema + 0.05 * max(abs(raw), 1e-3)
            r_rec = raw / self.rec_ema
        # 当前观测反馈：融合前后，地图对同一帧的深度/光度预测误差下降量（无 GT）
        r_fb, fb1 = 0.0, (0.0, 0.0, 0.0)
        if self.w_fb > 0 and fb0 is not None:
            fb1 = self._frame_error(depth, rgb, R, t)
            raw_fb = (100.0 * (fb1[0] - fb0[0])
                      + (fb0[1] - fb1[1])
                      + 20.0 * (fb0[2] - fb1[2]))
            self.fb_ema = 0.95 * self.fb_ema + 0.05 * max(abs(raw_fb), 1e-3)
            r_fb = raw_fb / self.fb_ema
        reward = (self.w_surf * d_surf / 100.0 + self.w_quality * quality
                  - self.w_conflict * conflict - self.w_path * path
                  + self.w_rec * r_rec + self.w_fb * r_fb)
        # 覆盖率软约束：低于目标时按缺口线性罚（越缺罚越狠），高于目标不奖励
        cov_now = self.coverage()
        cov_pen = self.w_cov * max(0.0, self.cov_target - cov_now) if self.cov_target > 0 else 0.0
        reward -= cov_pen
        self.t += 1
        info = {"new_known": d_known, "new_surface": d_surf, "path": path,
                "quality": quality, "conflict": conflict, "n_surf_px": n_surf,
                "rec": r_rec, "hit_rate": err1["hit_rate"],
                "err_depth_cm": err1["med_err_cm"], "err_photo": err1["photo"],
                "fb": r_fb, "fb_hit": fb1[0], "fb_depth_cm": fb1[1], "fb_photo": fb1[2],
                "cov_pen": cov_pen,
                "coverage": cov_now, "known_total": known_now, "reward": reward}
        if self.t >= self.steps:
            return None, None, reward, True, info
        feats, valid = self._observe()
        return feats, valid, reward, False, info


def refine_near(vmap, centers, rng, k, pos_sigma=0.18, tgt_sigma=0.12):
    """在给定候选中心附近局部重采（位置/注视点高斯扰动），跳过障碍内部的点。"""
    lo = np.asarray(ROOM_MIN, np.float32) + 0.15
    hi = np.asarray(ROOM_MAX, np.float32) - 0.15
    lo[2], hi[2] = 0.45, 1.45
    out, guard = [], 0
    while len(out) < k and guard < 20 * k:
        guard += 1
        c = centers[len(out) % len(centers)]
        pos = np.clip(c[2] + rng.normal(0.0, pos_sigma, 3).astype(np.float32), lo, hi)
        tgt = np.clip(c[3] + rng.normal(0.0, tgt_sigma, 3).astype(np.float32),
                      ROOM_MIN + 0.05, ROOM_MAX - 0.05)
        ts, ws, val = vmap.lookup_nearest_np(pos[None])
        if val[0] and ws[0] > 0.5 and ts[0] < -0.15:
            continue
        if np.linalg.norm(tgt - pos) < 0.25:
            continue
        R, t = look_at(pos, tgt)
        out.append((R, t, pos.astype(np.float32), tgt.astype(np.float32)))
    return out


def policy_action(policy, sampler, feats, valid, rng, device="cpu", greedy=False):
    """按策略选动作；sampler.refine_rounds>0 时做"采样 -> top-k -> 局部重采 -> 重打分"。

    policy 可以是单个网络，也可以是网络列表（集成：logits 与 value 取平均）。
    返回 (action, 用于该动作的特征, 对应 mask, logp, value)。
    """
    policies = list(policy) if isinstance(policy, (list, tuple)) else [policy]

    def _score(f, v):
        logits, vals = [], []
        with torch.no_grad():
            for p in policies:
                logit, val = p(torch.as_tensor(f, device=device)[None],
                               torch.as_tensor(v, device=device)[None])
                logits.append(logit[0])
                vals.append(float(val[0]))
        return torch.stack(logits).mean(0), float(np.mean(vals))

    logit, val = _score(feats, valid)
    if getattr(sampler, "refine_rounds", 0) > 0:
        order = torch.argsort(logit, descending=True)[: sampler.refine_topk].cpu().numpy()
        new_cands = refine_near(sampler.vmap, [sampler.cands[i] for i in order], rng,
                                sampler.refine_k)
        if new_cands:
            sampler.cands = new_cands
            feats, valid, _ = sampler.features_for(new_cands)
            logit, val = _score(feats, valid)
    dist = torch.distributions.Categorical(logits=logit)
    a = int(torch.argmax(logit)) if greedy else int(dist.sample())
    return a, feats, valid, float(dist.log_prob(torch.tensor(a, device=logit.device))), val


# ---------------------------------------------------------------- PPO 训练
def _gae(rewards, values, gamma=0.95, lam=0.95, last_value=0.0):
    adv = np.zeros(len(rewards), np.float32)
    last = 0.0
    for i in reversed(range(len(rewards))):
        next_v = values[i + 1] if i + 1 < len(values) else last_value
        delta = rewards[i] + gamma * next_v - values[i]
        last = delta + gamma * lam * last
        adv[i] = last
    return adv, adv + np.asarray(values, np.float32)


def train_ppo(policy, env, episodes=120, update_every=8, epochs=4, lr=3e-4,
              clip=0.2, entropy=0.01, value_coef=0.5, gamma=0.95, lam=0.95,
              seed=0, log=print):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    hist, buf = [], []
    for ep in range(1, episodes + 1):
        feats, valid = env.reset(np.random.default_rng(seed * 10000 + ep))
        refine_rng = np.random.default_rng(seed * 7919 + ep)
        traj = {"feats": [], "mask": [], "act": [], "logp": [], "val": [], "rew": []}
        stats = []
        for _ in range(env.steps):
            act, f_used, m_used, logp, val = policy_action(policy, env, feats, valid, refine_rng)
            traj["feats"].append(f_used)
            traj["mask"].append(m_used)
            traj["act"].append(int(act))
            traj["logp"].append(logp)
            traj["val"].append(val)
            feats, valid, rew, done, info = env.step(int(act))
            traj["rew"].append(rew)
            stats.append(info)
            if done:
                break

        adv, ret = _gae(traj["rew"], traj["val"], gamma, lam)
        for i in range(len(traj["rew"])):
            buf.append((traj["feats"][i], traj["mask"][i], traj["act"][i],
                        traj["logp"][i], adv[i], ret[i]))
        hist.append({"episode": ep, "return": float(np.sum(traj["rew"])),
                     "new_known": int(sum(s["new_known"] for s in stats)),
                     "new_surface": int(sum(s["new_surface"] for s in stats)),
                     "quality": float(np.mean([s["quality"] for s in stats])),
                     "path": float(sum(s["path"] for s in stats)),
                     "coverage": float(stats[-1]["coverage"]) if stats else 0.0})

        if ep % update_every == 0 and buf:
            adv_arr = np.array([b[4] for b in buf], np.float32)
            adv_arr = (adv_arr - adv_arr.mean()) / (adv_arr.std() + 1e-8)
            for _ in range(epochs):
                for b, a_norm in zip(buf, adv_arr):
                    feats_b, mask_b, act, logp_old, _, ret_b = b
                    logit, val = policy(torch.as_tensor(feats_b)[None],
                                        torch.as_tensor(mask_b)[None])
                    d = torch.distributions.Categorical(logits=logit[0])
                    logp = d.log_prob(torch.tensor(act))
                    ratio = torch.exp(logp - logp_old)
                    a = torch.as_tensor(a_norm)
                    pg = -torch.min(ratio * a, torch.clamp(ratio, 1 - clip, 1 + clip) * a)
                    v_loss = (val[0] - torch.as_tensor(ret_b)) ** 2
                    loss = pg + value_coef * v_loss - entropy * d.entropy()
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                    opt.step()
            buf = []
            tail = hist[-update_every:]
            log(f"  episode {ep:4d}  return {np.mean([h['return'] for h in tail]):7.2f}  "
                f"新增表面 {np.mean([h['new_surface'] for h in tail]):6.1f}  "
                f"质量 {np.mean([h['quality'] for h in tail]):.3f}  "
                f"路程 {np.mean([h['path'] for h in tail]):5.2f} m  "
                f"覆盖率 {np.mean([h['coverage'] for h in tail]):.3f}")
    return hist


# ---------------------------------------------------------------- 推理接口
_POLICY_CACHE = {}


class MapSampler:
    """给 select_nbv_rl 用的轻量采样器：只需"算特征 + 局部重采"，不含 episode 状态机。"""

    def __init__(self, vmap, cands, pts, wts, kinds, normals, current_pos, current_fwd,
                 visited, K, H, W, refine_rounds=0, refine_k=64, refine_topk=8):
        self.vmap, self.cands = vmap, cands
        self.pts, self.wts, self.kinds, self.normals = pts, wts, kinds, normals
        self.current_pos, self.current_fwd, self.visited = current_pos, current_fwd, visited
        self.K, self.H, self.W = K, H, W
        self.refine_rounds = refine_rounds
        self.refine_k, self.refine_topk = refine_k, refine_topk

    def features_for(self, cands):
        return candidate_features(self.vmap, cands, self.pts, self.wts, self.kinds,
                                  self.current_pos, self.current_fwd, self.visited,
                                  self.K, self.H, self.W, normals=self.normals)


def select_nbv_rl(ckpt, vmap, points, weights, kinds, current_pos, current_fwd,
                  visited, K, H, W, device="cpu", rng=None, n_cand=64,
                  refine_rounds=0, refine_k=64, refine_topk=8, greedy=True):
    """在均匀采样的候选集上按策略打分选 NBV。返回 (pose, score, feats)。

    ckpt 可以是单个路径，也可以是路径列表——多个策略时对 logits 取平均（集成），
    用于压低单策略的方差（实测单策略覆盖率 ±4 pp，主要来自训练随机性）。
    """
    ckpts = [ckpt] if isinstance(ckpt, (str, Path)) else list(ckpt)
    policies = []
    for c in ckpts:
        key = str(c)
        if key not in _POLICY_CACHE:
            _POLICY_CACHE[key] = load_policy(key, device)
        policies.append(_POLICY_CACHE[key])
    rng = np.random.default_rng() if rng is None else rng
    cands = sample_candidates(vmap, rng, k=n_cand)
    grad = vmap.query_grad_np(np.asarray(points, np.float32))
    normals = grad / np.maximum(np.linalg.norm(grad, axis=-1, keepdims=True), 1e-8)
    sampler = MapSampler(vmap, cands, points, weights, kinds, normals,
                         current_pos, current_fwd, visited, K, H, W,
                         refine_rounds, refine_k, refine_topk)
    feats, valid, _ = sampler.features_for(cands)
    a, feats, valid, logp, _ = policy_action(policies, sampler, feats, valid, rng,
                                             device=device, greedy=greedy)
    return sampler.cands[a], logp, feats[a]


def heuristic_action(cands, vmap, points, weights, K, H, W, current_pos, visited):
    """把启发式 NBV 映射回候选下标（供 eval 用，保持两策略同候选集）。"""
    from .heuristic import select_nbv
    pose, score = select_nbv(cands, points, weights, vmap, K, H, W,
                             current_pos=current_pos, visited=visited)
    if pose is None:
        return None, None
    pos, tgt = np.asarray(pose[2]), np.asarray(pose[3])
    d = [np.abs(np.asarray(c[2]) - pos).sum() + np.abs(np.asarray(c[3]) - tgt).sum()
         for c in cands]
    return int(np.argmin(d)), score
