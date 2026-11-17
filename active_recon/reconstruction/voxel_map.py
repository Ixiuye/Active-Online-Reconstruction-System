"""稠密 TSDF 体素层：融合、三线性查询、置信度与状态。

字段约定：
- tsdf 归一化到 [-1, 1]（除以截断距离），正 = 自由空间，负 = 表面后方；
- weight 是累积观测权重，既用作置信度也用于区分 unknown/free；
- color_sum / color_w 累积表面颜色，用于高斯初始化。
"""
from __future__ import annotations

import numpy as np
import torch


class DenseTSDF:
    def __init__(self, grid_min, grid_max, voxel, trunc, device="cpu"):
        self.gmin = np.asarray(grid_min, dtype=np.float32)
        self.voxel = float(voxel)
        self.trunc = float(trunc)
        dims = np.ceil((np.asarray(grid_max) - self.gmin) / voxel).astype(int)
        self.dims = tuple(int(v) for v in dims)
        self.device = device

        shape = self.dims
        self.tsdf = np.zeros(shape, dtype=np.float32)
        self.weight = np.zeros(shape, dtype=np.float32)
        self.color_sum = np.zeros(shape + (3,), dtype=np.float32)
        self.color_w = np.zeros(shape, dtype=np.float32)

        ijk = np.stack(
            np.meshgrid(
                np.arange(shape[0]),
                np.arange(shape[1]),
                np.arange(shape[2]),
                indexing="ij",
            ),
            axis=-1,
        ).astype(np.float32)
        self.ijk = ijk
        self.centers = (self.gmin + (ijk + 0.5) * voxel).reshape(-1, 3)

        # torch 侧缓存（融合后刷新）
        self._t = {}
        self.refresh_torch()

    # ---------------- 融合 ----------------
    def fuse(self, depth, rgb, R, t, K):
        fx, fy, cx, cy = K
        H, W = depth.shape
        P = self.centers
        pc = P @ np.asarray(R, np.float32).T + np.asarray(t, np.float32)[None]
        z = pc[:, 2]
        u = fx * pc[:, 0] / np.maximum(z, 1e-6) + cx
        v = fy * pc[:, 1] / np.maximum(z, 1e-6) + cy
        ui = np.round(u).astype(np.int32)
        vi = np.round(v).astype(np.int32)
        valid = (z > 0.05) & (z < 8.0) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)

        d = np.zeros_like(z)
        d[valid] = depth[vi[valid], ui[valid]]
        valid &= d > 0.0
        # depth 是沿光线的欧氏距离，先换算到同一度量再与体素的相机 z 比较
        norm = np.sqrt(1.0 + ((u - cx) / fx) ** 2 + ((v - cy) / fy) ** 2)
        sdf = d - z * norm
        front = valid & (sdf > -self.trunc)

        tsdf = self.tsdf.reshape(-1)
        weight = self.weight.reshape(-1)
        new_val = np.clip(sdf / self.trunc, -1.0, 1.0).astype(np.float32)
        w = weight[front]
        tsdf[front] = (tsdf[front] * w + new_val[front]) / (w + 1.0)
        weight[front] = np.minimum(w + 1.0, 64.0)

        near = front & (np.abs(sdf) < self.trunc)
        if near.any():
            cols = rgb.reshape(-1, 3)[vi[near] * W + ui[near]]
            csum = self.color_sum.reshape(-1, 3)
            cw = self.color_w.reshape(-1)
            csum[near] = (csum[near] * cw[near, None] + cols) / (cw[near, None] + 1.0)
            cw[near] = np.minimum(cw[near] + 1.0, 64.0)

        self.refresh_torch()

    # ---------------- torch 查询 ----------------
    def refresh_torch(self):
        dev = self.device
        self._t = {
            "tsdf": torch.as_tensor(self.tsdf, device=dev),
            "weight": torch.as_tensor(self.weight, device=dev),
        }

    def _trilinear_torch(self, field, pts):
        dims = torch.tensor(field.shape[:3], device=pts.device)
        g = (pts - torch.as_tensor(self.gmin, device=pts.device)) / self.voxel - 0.5
        i0 = torch.floor(g).long()
        f = g - i0.to(g.dtype)
        out = 0.0
        for dx in (0, 1):
            wx = f[:, 0] if dx else 1.0 - f[:, 0]
            ix = torch.clamp(i0[:, 0] + dx, 0, int(dims[0]) - 1)
            for dy in (0, 1):
                wy = f[:, 1] if dy else 1.0 - f[:, 1]
                iy = torch.clamp(i0[:, 1] + dy, 0, int(dims[1]) - 1)
                for dz in (0, 1):
                    wz = f[:, 2] if dz else 1.0 - f[:, 2]
                    iz = torch.clamp(i0[:, 2] + dz, 0, int(dims[2]) - 1)
                    v = field[ix, iy, iz]
                    w = wx * wy * wz
                    if v.dim() > 1:
                        w = w.unsqueeze(-1)
                    out = out + v * w
        return out

    def query_torch(self, pts):
        return self._trilinear_torch(self._t["tsdf"], pts)

    def query_weight_torch(self, pts):
        return self._trilinear_torch(self._t["weight"], pts)

    def query_grad_torch(self, pts):
        key = "grad"
        if key not in self._t:
            gx, gy, gz = np.gradient(self.tsdf, self.voxel)
            grad = np.stack([gx, gy, gz], axis=-1)
            self._t[key] = torch.as_tensor(grad, device=self.device)
        return self._trilinear_torch(self._t[key], pts)

    # ---------------- numpy 查询 ----------------
    def _trilinear_np(self, points, field):
        points = np.asarray(points, dtype=np.float32)
        g = (points - self.gmin) / self.voxel - 0.5
        i0 = np.floor(g).astype(np.int64)
        f = (g - i0).astype(np.float32)
        out = 0.0
        dims = np.array(field.shape[:3])
        for dx in (0, 1):
            wx = f[:, 0] if dx else 1.0 - f[:, 0]
            ix = np.clip(i0[:, 0] + dx, 0, dims[0] - 1)
            for dy in (0, 1):
                wy = f[:, 1] if dy else 1.0 - f[:, 1]
                iy = np.clip(i0[:, 1] + dy, 0, dims[1] - 1)
                for dz in (0, 1):
                    wz = f[:, 2] if dz else 1.0 - f[:, 2]
                    iz = np.clip(i0[:, 2] + dz, 0, dims[2] - 1)
                    v = field[ix, iy, iz]
                    w = wx * wy * wz
                    if v.ndim > 1:
                        w = w[:, None]
                    out = out + v * w
        return out

    def query_np(self, points):
        return self._trilinear_np(points, self.tsdf)

    def query_weight_np(self, points):
        return self._trilinear_np(points, self.weight)

    def query_grad_np(self, points):
        gx, gy, gz = np.gradient(self.tsdf, self.voxel)
        return self._trilinear_np(points, np.stack([gx, gy, gz], axis=-1))

    def normal_grid(self):
        gx, gy, gz = np.gradient(self.tsdf, self.voxel)
        g = np.stack([gx, gy, gz], axis=-1)
        return g / np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-8)

    def lookup_nearest_np(self, points):
        idx = np.floor((points - self.gmin) / self.voxel).astype(np.int64)
        valid = np.all((idx >= 0) & (idx < np.array(self.dims)), axis=-1)
        idx = np.clip(idx, 0, np.array(self.dims) - 1)
        tsdf = self.tsdf[idx[:, 0], idx[:, 1], idx[:, 2]]
        weight = self.weight[idx[:, 0], idx[:, 1], idx[:, 2]]
        return tsdf, weight, valid

    # ---------------- 状态 ----------------
    def surface_mask_np(self, band=0.6, w_min=1.0):
        return (self.weight > w_min) & (np.abs(self.tsdf) < band)

    def surface_flat_indices(self, band=0.6, w_min=1.0):
        return np.flatnonzero(self.surface_mask_np(band, w_min).reshape(-1))

    def unknown_mask_np(self, w_max=0.5):
        return self.weight < w_max

    def free_mask_np(self, w_min=1.0, t_min=0.3):
        return (self.weight > w_min) & (self.tsdf > t_min)

    def frontier_unknown_flat(self):
        unknown = self.unknown_mask_np()
        free = self.free_mask_np()
        free_neighbor = np.zeros_like(free)
        free_neighbor[1:, :, :] |= free[:-1, :, :]
        free_neighbor[:-1, :, :] |= free[1:, :, :]
        free_neighbor[:, 1:, :] |= free[:, :-1, :]
        free_neighbor[:, :-1, :] |= free[:, 1:, :]
        free_neighbor[:, :, 1:] |= free[:, :, :-1]
        free_neighbor[:, :, :-1] |= free[:, :, 1:]
        frontier = unknown & free_neighbor
        return np.flatnonzero(frontier.reshape(-1))

    def stats(self):
        w = self.weight
        return {
            "voxels": int(np.prod(self.dims)),
            "unknown": int((w < 0.5).sum()),
            "free": int(((w > 0.5) & (self.tsdf > 0.3)).sum()),
            "surface": int(((w > 0.5) & (np.abs(self.tsdf) < 0.35)).sum()),
        }
