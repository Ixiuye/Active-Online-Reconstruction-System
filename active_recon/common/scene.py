"""解析式室内场景：SDF、程序化颜色、GT 深度/彩色渲染、GT 表面采样。

场景是一个 2.4 x 1.6 x 1.6 m 的小房间，内含球体、桌子和一根细圆柱。
SDF 约定：自由空间为正，物体/墙体内部为负，零水平集为表面。
"""
from __future__ import annotations

import numpy as np

from .utils import pixel_rays

ROOM_MIN = np.array([0.0, 0.0, 0.0], dtype=np.float32)
ROOM_MAX = np.array([2.4, 1.6, 1.6], dtype=np.float32)
VOXEL_SIZE = 0.08
TRUNC = 0.20  # TSDF 截断距离（米）


def sdf_box(p, c, h):
    q = np.abs(p - c) - h
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    inside = np.minimum(np.max(q, axis=-1), 0.0)
    return outside + inside


def sdf_sphere(p, c, r):
    return np.linalg.norm(p - c, axis=-1) - r


def sdf_cylinder_z(p, c, r, half_h):
    dx = p[..., 0] - c[0]
    dy = p[..., 1] - c[1]
    dz = p[..., 2] - c[2]
    d0 = np.sqrt(dx * dx + dy * dy) - r
    d1 = np.abs(dz) - half_h
    outside = np.linalg.norm(
        np.stack([np.maximum(d0, 0.0), np.maximum(d1, 0.0)], axis=-1), axis=-1
    )
    return np.minimum(np.maximum(d0, d1), 0.0) + outside


class DemoScene:
    def __init__(self):
        self.room_c = ((ROOM_MIN + ROOM_MAX) / 2.0).astype(np.float32)
        self.room_h = ((ROOM_MAX - ROOM_MIN) / 2.0).astype(np.float32)
        self.sphere_c = np.array([0.75, 0.95, 0.33], dtype=np.float32)
        self.sphere_r = 0.22
        self.table_c = np.array([1.65, 1.15, 0.18], dtype=np.float32)
        self.table_h = np.array([0.28, 0.20, 0.18], dtype=np.float32)
        self.cyl_c = np.array([1.95, 0.45, 0.60], dtype=np.float32)
        self.cyl_r = 0.04
        self.cyl_hh = 0.60

    # ---------- 几何 ----------
    def sdf(self, p):
        p = np.asarray(p, dtype=np.float32)
        free = -sdf_box(p, self.room_c, self.room_h)
        obj = np.minimum(
            np.minimum(
                sdf_sphere(p, self.sphere_c, self.sphere_r),
                sdf_box(p, self.table_c, self.table_h),
            ),
            sdf_cylinder_z(p, self.cyl_c, self.cyl_r, self.cyl_hh),
        )
        return np.minimum(free, obj)

    def gradient(self, p, eps=1e-4):
        p = np.asarray(p, dtype=np.float32)
        gs = []
        for axis in range(3):
            e = np.zeros(3, dtype=np.float32)
            e[axis] = eps
            gs.append((self.sdf(p + e) - self.sdf(p - e)) / (2.0 * eps))
        g = np.stack(gs, axis=-1)
        return g / np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-8)

    # ---------- 颜色 ----------
    def color(self, p):
        p = np.asarray(p, dtype=np.float32)
        q = (p - self.room_c) / self.room_h
        axis = np.argmax(np.abs(q), axis=-1)
        sign = np.sign(np.take_along_axis(q, axis[..., None], axis=-1)[:, 0])

        checker = (
            np.floor(p[..., 0] / 0.4) + np.floor(p[..., 1] / 0.4)
        ) % 2
        floor_col = np.stack(
            [0.52 + 0.28 * checker, 0.50 + 0.24 * checker, 0.46 + 0.18 * checker],
            axis=-1,
        )

        wall_col = np.zeros_like(p)
        wall_col[axis == 0] = np.array([0.74, 0.54, 0.48], dtype=np.float32)
        wall_col[axis == 1] = np.array([0.48, 0.68, 0.54], dtype=np.float32)
        wall_col *= (0.82 + 0.18 * p[:, 2:3])
        stripe = 0.5 + 0.5 * np.sin(24.0 * p[:, 0] + 13.0 * p[:, 1])
        wall_col *= (0.86 + 0.14 * stripe[:, None])

        col = np.where((axis == 2)[:, None], floor_col, wall_col)
        ceiling_col = np.array([0.86, 0.86, 0.82], dtype=np.float32)
        col = np.where(((axis == 2) & (sign > 0))[:, None], ceiling_col, col)

        d_room = -sdf_box(p, self.room_c, self.room_h)
        d_sph = sdf_sphere(p, self.sphere_c, self.sphere_r)
        d_tab = sdf_box(p, self.table_c, self.table_h)
        d_cyl = sdf_cylinder_z(p, self.cyl_c, self.cyl_r, self.cyl_hh)
        dmin = np.minimum(np.minimum(d_room, d_sph), np.minimum(d_tab, d_cyl))

        bands = 0.5 + 0.5 * np.sin(30.0 * p[..., 2])
        sph_col = np.stack(
            [0.95 - 0.18 * bands, 0.52 + 0.12 * bands, 0.24 + 0.10 * bands], axis=-1
        )
        tab_col = np.array([0.48, 0.34, 0.22], dtype=np.float32) * (
            0.85 + 0.15 * np.sin(10.0 * p[..., 0])[:, None]
        )
        cyl_col = np.array([0.24, 0.30, 0.34], dtype=np.float32) * (
            0.8 + 0.2 * (0.5 + 0.5 * np.sin(40.0 * p[..., 2]))[:, None]
        )

        col = np.where((d_sph <= dmin + 1e-6)[..., None], sph_col, col)
        col = np.where((d_tab <= dmin + 1e-6)[..., None], tab_col, col)
        col = np.where((d_cyl <= dmin + 1e-6)[..., None], cyl_col, col)
        return np.clip(col, 0.0, 1.0).astype(np.float32)

    # ---------- 渲染 GT ----------
    def raycast(self, origins, dirs, max_t=7.0, iters=128):
        o = np.asarray(origins, dtype=np.float32)
        d = np.asarray(dirs, dtype=np.float32)
        if o.ndim == 1:
            o = np.broadcast_to(o, d.shape).copy()
        t = np.full(d.shape[0], 0.02, dtype=np.float32)
        hit = np.zeros(d.shape[0], dtype=bool)
        for _ in range(iters):
            p = o + t[:, None] * d
            s = self.sdf(p)
            hit = hit | (np.abs(s) < 1e-3)
            step = np.where(hit, 0.0, np.maximum(s, 0.002))
            t = t + step
            if (hit | (t > max_t)).all():
                break
        depth = np.where(hit & (t < max_t), t, 0.0).astype(np.float32)
        p_hit = o + depth[:, None] * d
        colors = self.color(p_hit)
        return depth, colors, hit

    def render_view(self, R, t, K, height, width):
        center, dirs = pixel_rays(R, t, K, height, width)
        depth, colors, hit = self.raycast(center, dirs)
        return (
            depth.reshape(height, width),
            colors.reshape(height, width, 3),
            hit.reshape(height, width),
        )

    # ---------- GT 表面采样（用于 Chamfer 评估） ----------
    def sample_surface(self, n_room=24000, n_sphere=3000, n_table=5000, n_cyl=3000, seed=0):
        rng = np.random.default_rng(seed)
        pts = []

        dims = (ROOM_MAX - ROOM_MIN).astype(np.float64)
        faces = [0, 1, 2]
        face_areas = []
        for axis in faces:
            other = [a for a in faces if a != axis]
            face_areas.append(2.0 * dims[other[0]] * dims[other[1]])
        total_area = float(sum(face_areas))
        for axis in faces:
            n_axis = int(round(n_room * face_areas[axis] / total_area / 2.0))
            for sign in (-1.0, 1.0):
                u = rng.random((n_axis, 2)).astype(np.float32)
                p = np.empty((n_axis, 3), dtype=np.float32)
                other = [a for a in faces if a != axis]
                p[:, other[0]] = ROOM_MIN[other[0]] + u[:, 0] * dims[other[0]]
                p[:, other[1]] = ROOM_MIN[other[1]] + u[:, 1] * dims[other[1]]
                p[:, axis] = (ROOM_MAX[axis] if sign > 0 else ROOM_MIN[axis]) - sign * 0.002
                obj = np.minimum(
                    np.minimum(
                        sdf_sphere(p, self.sphere_c, self.sphere_r),
                        sdf_box(p, self.table_c, self.table_h),
                    ),
                    sdf_cylinder_z(p, self.cyl_c, self.cyl_r, self.cyl_hh),
                )
                pts.append(p[obj > 0.005])

        # 球面：Fibonacci 采样
        i = np.arange(n_sphere, dtype=np.float32) + 0.5
        phi = np.arccos(1.0 - 2.0 * i / n_sphere)
        theta = np.pi * (1.0 + 5.0 ** 0.5) * i
        sph = np.stack(
            [
                np.cos(theta) * np.sin(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(phi),
            ],
            axis=-1,
        ) * self.sphere_r + self.sphere_c
        pts.append(sph.astype(np.float32))

        # 桌面 6 个面
        tmin = self.table_c - self.table_h
        tmax = self.table_c + self.table_h
        tdim = (tmax - tmin).astype(np.float64)
        areas = []
        for axis in faces:
            other = [a for a in faces if a != axis]
            areas.append(2.0 * tdim[other[0]] * tdim[other[1]])
        for axis in faces:
            n_axis = int(round(n_table * areas[axis] / sum(areas) / 2.0))
            for sign in (-1.0, 1.0):
                u = rng.random((n_axis, 2)).astype(np.float32)
                p = np.empty((n_axis, 3), dtype=np.float32)
                other = [a for a in faces if a != axis]
                p[:, other[0]] = tmin[other[0]] + u[:, 0] * tdim[other[0]]
                p[:, other[1]] = tmin[other[1]] + u[:, 1] * tdim[other[1]]
                p[:, axis] = (tmax[axis] if sign > 0 else tmin[axis]) + sign * 0.002
                pts.append(p)

        # 圆柱侧面 + 两个端面
        theta = rng.random(n_cyl) * 2.0 * np.pi
        z = rng.random(n_cyl) * (2.0 * self.cyl_hh) + self.cyl_c[2] - self.cyl_hh
        side = np.stack(
            [
                self.cyl_c[0] + self.cyl_r * np.cos(theta),
                self.cyl_c[1] + self.cyl_r * np.sin(theta),
                z,
            ],
            axis=-1,
        )
        pts.append(side.astype(np.float32))
        for zz in (self.cyl_c[2] - self.cyl_hh, self.cyl_c[2] + self.cyl_hh):
            n_cap = n_cyl // 4
            rr = self.cyl_r * np.sqrt(rng.random(n_cap))
            th = rng.random(n_cap) * 2.0 * np.pi
            cap = np.stack(
                [
                    self.cyl_c[0] + rr * np.cos(th),
                    self.cyl_c[1] + rr * np.sin(th),
                    np.full(n_cap, zz, dtype=np.float64),
                ],
                axis=-1,
            )
            pts.append(cap.astype(np.float32))

        return np.concatenate(pts, axis=0)

    def gt_surface_voxel_mask(self, grid_min, dims, voxel):
        ijk = np.stack(
            np.meshgrid(
                np.arange(dims[0]), np.arange(dims[1]), np.arange(dims[2]), indexing="ij"
            ),
            axis=-1,
        ).astype(np.float32)
        centers = grid_min + (ijk + 0.5) * voxel
        d = self.sdf(centers.reshape(-1, 3)).reshape(dims)
        return np.abs(d) < 0.75 * voxel
