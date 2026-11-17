"""Voxel-Anchored Gaussians：体素作为表征骨架的高斯集合。

每个高斯绑定一个表面体素（anchor），其参数化被体素几何约束：
- 中心：mu = c_v + 0.5 * voxel * tanh(delta_raw)，严格限制在锚点体素内；
- 旋转：由体素处 TSDF 法向构造的固定切向帧（法向轴 + 两个切向轴）；
- 尺度：sigmoid 参数化，限制在 [0.15, 2.5] * voxel；
- 颜色/不透明度：可学习。

生命周期由体素状态驱动：新的表面体素出现 -> spawn；体素被判定为自由 -> 可 prune。
"""
from __future__ import annotations

import numpy as np
import torch


def _frames_from_normals(normals):
    """由法向构造 [t1, t2, n] 列向量帧，输入 [N,3] -> [N,3,3]。"""
    normals = np.asarray(normals, dtype=np.float32)
    ref = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (normals.shape[0], 1))
    bad = np.abs((normals * ref).sum(-1)) > 0.9
    ref[bad] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    t1 = np.cross(normals, ref)
    t1 /= np.maximum(np.linalg.norm(t1, axis=-1, keepdims=True), 1e-8)
    t2 = np.cross(normals, t1)
    return np.stack([t1, t2, normals], axis=-1).astype(np.float32)


class VoxelGaussians:
    def __init__(self, tsdf_map, device="cpu", max_total=5000):
        self.map = tsdf_map
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.max_total = int(max_total)
        self.voxel = tsdf_map.voxel
        self.s_min = 0.15 * self.voxel
        self.s_max = 2.50 * self.voxel

        self.occupied = np.zeros(int(np.prod(tsdf_map.dims)), dtype=bool)
        self.voxel_ids = np.zeros(0, dtype=np.int64)
        self.anchor = torch.zeros(0, 3, device=self.device)
        self.frame = torch.zeros(0, 3, 3, device=self.device)

        self.delta_raw = torch.zeros(0, 3, device=self.device)
        self.scale_raw = torch.zeros(0, 3, device=self.device)
        self.rot_raw = torch.zeros(0, 3, device=self.device)
        self.color = torch.zeros(0, 3, device=self.device)
        self.opacity_logit = torch.zeros(0, device=self.device)
        self.max_rot = 0.25  # 允许相对 TSDF 法向帧的最大修正角（rad，约 14°）

    # ---------------- 生命周期 ----------------
    @torch.no_grad()
    def spawn(self, band=0.35, w_min=0.5):
        """为尚未绑定高斯的表面体素生成新高斯，返回新增数量。"""
        flat_tsdf = self.map.tsdf.reshape(-1)
        flat_w = self.map.weight.reshape(-1)
        mask = (flat_w > w_min) & (np.abs(flat_tsdf) < band) & (~self.occupied)
        new_idx = np.flatnonzero(mask)
        if new_idx.size == 0:
            return 0
        room = self.max_total - self.anchor.shape[0]
        if room <= 0:
            return 0
        if new_idx.size > room:
            order = np.argsort(-flat_w[new_idx])
            new_idx = new_idx[order[:room]]

        centers = self.map.centers[new_idx]
        ijk = self.map.ijk.reshape(-1, 3)[new_idx].astype(np.int64)
        normal = self.map.normal_grid()[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
        frames = _frames_from_normals(normal)

        csum = self.map.color_sum.reshape(-1, 3)[new_idx]
        cw = self.map.color_w.reshape(-1)[new_idx]
        colors = np.where(cw[:, None] > 0, csum / np.maximum(cw[:, None], 1e-6), 0.5)
        colors = np.clip(colors, 0.0, 1.0).astype(np.float32)

        # 尺度初始化：切向 ~0.75 voxel，法向 ~0.38 voxel
        init_s = np.tile(
            np.array([0.75 * self.voxel, 0.75 * self.voxel, 0.38 * self.voxel], np.float32),
            (len(new_idx), 1),
        )
        ratio = (init_s - self.s_min) / (self.s_max - self.s_min)
        ratio = np.clip(ratio, 1e-3, 1.0 - 1e-3)
        scale_raw = np.log(ratio / (1.0 - ratio)).astype(np.float32)

        n = len(new_idx)
        self.anchor = torch.cat(
            [self.anchor, torch.as_tensor(centers, device=self.device)], dim=0
        )
        self.frame = torch.cat(
            [self.frame, torch.as_tensor(frames, device=self.device)], dim=0
        )
        self.delta_raw = torch.cat(
            [self.delta_raw, torch.zeros(n, 3, device=self.device)], dim=0
        ).detach().clone().requires_grad_(True)
        self.scale_raw = torch.cat(
            [self.scale_raw, torch.as_tensor(scale_raw, device=self.device)], dim=0
        ).detach().clone().requires_grad_(True)
        self.rot_raw = torch.cat(
            [self.rot_raw, torch.zeros(n, 3, device=self.device)], dim=0
        ).detach().clone().requires_grad_(True)
        self.color = torch.cat(
            [self.color, torch.as_tensor(colors, device=self.device)], dim=0
        ).detach().clone().requires_grad_(True)
        self.opacity_logit = torch.cat(
            [self.opacity_logit, torch.full((n,), -0.405, device=self.device)], dim=0
        ).detach().clone().requires_grad_(True)

        self.occupied[new_idx] = True
        self.voxel_ids = np.concatenate([self.voxel_ids, new_idx.astype(np.int64)])
        return n

    @torch.no_grad()
    def prune_free_anchors(self, tau_free=0.5, w_min=0.5):
        """锚点体素被后续观测改判为自由空间时，删除对应高斯（体素驱动的生命周期）。"""
        if self.voxel_ids.size == 0:
            return 0
        flat_tsdf = self.map.tsdf.reshape(-1)
        flat_w = self.map.weight.reshape(-1)
        ids = self.voxel_ids
        bad = (flat_w[ids] > w_min) & (flat_tsdf[ids] > tau_free)
        if not bad.any():
            return 0
        keep = ~bad
        self.occupied[ids[bad]] = False
        self.voxel_ids = ids[keep]
        for name in ("anchor", "frame", "delta_raw", "scale_raw", "rot_raw",
                     "color", "opacity_logit"):
            tensor = getattr(self, name)
            setattr(self, name, tensor[torch.as_tensor(keep, device=self.device)].clone())
        self.delta_raw.requires_grad_(True)
        self.scale_raw.requires_grad_(True)
        self.rot_raw.requires_grad_(True)
        self.color.requires_grad_(True)
        self.opacity_logit.requires_grad_(True)
        return int(bad.sum())

    # ---------------- 参数化 ----------------
    def means(self):
        return self.anchor + 0.5 * self.voxel * torch.tanh(self.delta_raw)

    def scales(self):
        return self.s_min + (self.s_max - self.s_min) * torch.sigmoid(self.scale_raw)

    def frames(self):
        """TSDF 法向帧叠加一个限幅的小旋转，使法向可微调但不会偏离先验太远。"""
        v = self.max_rot * torch.tanh(self.rot_raw)
        theta = torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(1e-8)
        k = v / theta
        g = v.shape[0]
        K = torch.zeros(g, 3, 3, device=v.device, dtype=v.dtype)
        K[:, 0, 1] = -k[:, 2]
        K[:, 0, 2] = k[:, 1]
        K[:, 1, 0] = k[:, 2]
        K[:, 1, 2] = -k[:, 0]
        K[:, 2, 0] = -k[:, 1]
        K[:, 2, 1] = k[:, 0]
        eye = torch.eye(3, device=v.device, dtype=v.dtype).unsqueeze(0)
        sin_t = torch.sin(theta).unsqueeze(-1)
        cos_t = (1.0 - torch.cos(theta)).unsqueeze(-1)
        rot = eye + sin_t * K + cos_t * (K @ K)
        return self.frame @ rot

    def opacity(self):
        return torch.sigmoid(self.opacity_logit)

    def normals(self):
        return self.frames()[:, :, 2]

    def properties(self):
        return self.means(), self.frames(), self.scales(), self.color, self.opacity()

    def parameters(self):
        return [self.delta_raw, self.scale_raw, self.rot_raw, self.color, self.opacity_logit]

    def make_optimizer(self, lr_delta=0.03, lr_scale=0.01, lr_rot=0.01,
                       lr_color=0.02, lr_opacity=0.05):
        return torch.optim.Adam(
            [
                {"params": [self.delta_raw], "lr": lr_delta},
                {"params": [self.scale_raw], "lr": lr_scale},
                {"params": [self.rot_raw], "lr": lr_rot},
                {"params": [self.color], "lr": lr_color},
                {"params": [self.opacity_logit], "lr": lr_opacity},
            ]
        )

    def detach_clone(self):
        clone = VoxelGaussians(self.map, self.device, self.max_total)
        clone.anchor = self.anchor.detach().clone()
        clone.frame = self.frame.detach().clone()
        clone.delta_raw = self.delta_raw.detach().clone().requires_grad_(True)
        clone.scale_raw = self.scale_raw.detach().clone().requires_grad_(True)
        clone.rot_raw = self.rot_raw.detach().clone().requires_grad_(True)
        clone.color = self.color.detach().clone().requires_grad_(True)
        clone.opacity_logit = self.opacity_logit.detach().clone().requires_grad_(True)
        clone.occupied = self.occupied.copy()
        clone.voxel_ids = self.voxel_ids.copy()
        return clone
