"""解析式室内场景：SDF、程序化颜色、GT 深度/彩色渲染、GT 表面采样。

默认场景是一个 2.4 x 1.6 x 1.6 m 的小房间，内含球体、桌子和一根细圆柱。
另外提供两种扩展场景：
- PrimitiveScene：房间 + 单个简单几何体（球 / 长方体 / 圆柱），用于最基础的重建验证；
- RobotArmScene：房间 + 一台固定姿态的多关节串联机械臂（底座、关节、连杆、夹爪），
  用于检验体素-高斯在细长、多部件组合体上的表现。

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


def sdf_capsule(p, a, b, r):
    """线段 a-b 周围的胶囊体 SDF。"""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    pa = p - a
    ba = b - a
    t = np.clip((pa @ ba) / max(float(ba @ ba), 1e-8), 0.0, 1.0)
    return np.linalg.norm(pa - t[..., None] * ba, axis=-1) - r


# ---------------------------------------------------------------- 表面采样工具
def sample_sphere_surface(rng, c, r, n):
    """Fibonacci 球面采样。"""
    i = np.arange(n, dtype=np.float32) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    return (np.stack([np.cos(theta) * np.sin(phi),
                      np.sin(theta) * np.sin(phi),
                      np.cos(phi)], axis=-1) * r + np.asarray(c, np.float32))


def sample_capsule_surface(rng, a, b, r, n):
    """胶囊体侧面采样（端点由关节球覆盖）。"""
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    axis = b - a
    axis = axis / max(np.linalg.norm(axis), 1e-8)
    ref = np.array([0.0, 0.0, 1.0], np.float32)
    if abs(float(axis @ ref)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0], np.float32)
    u = np.cross(axis, ref)
    u /= max(np.linalg.norm(u), 1e-8)
    v = np.cross(axis, u)
    t = rng.random(n).astype(np.float32)
    th = (rng.random(n) * 2.0 * np.pi).astype(np.float32)
    return (a[None] + t[:, None] * (b - a)[None]
            + r * (np.cos(th)[:, None] * u[None] + np.sin(th)[:, None] * v[None]))


def sample_cylinder_surface(rng, c, r, half_h, n):
    c = np.asarray(c, np.float32)
    n_side = n
    th = rng.random(n_side) * 2.0 * np.pi
    z = rng.random(n_side) * (2.0 * half_h) + c[2] - half_h
    side = np.stack([c[0] + r * np.cos(th), c[1] + r * np.sin(th), z], axis=-1)
    n_cap = max(1, n // 4)
    caps = []
    for zz in (c[2] - half_h, c[2] + half_h):
        rr = r * np.sqrt(rng.random(n_cap))
        tt = rng.random(n_cap) * 2.0 * np.pi
        caps.append(np.stack([c[0] + rr * np.cos(tt), c[1] + rr * np.sin(tt),
                              np.full(n_cap, zz, np.float64)], axis=-1))
    return np.concatenate([side.astype(np.float32)] + [x.astype(np.float32) for x in caps], 0)


def sample_box_surface(rng, tmin, tmax, n):
    """长方体 6 个面的均匀采样（向内偏移 2 mm，与 DemoScene 的桌面采样一致）。"""
    tmin = np.asarray(tmin, np.float32)
    tmax = np.asarray(tmax, np.float32)
    dims = (tmax - tmin).astype(np.float64)
    faces = [0, 1, 2]
    areas = [2.0 * dims[a] * dims[b] for a, b in ((1, 2), (0, 2), (0, 1))]
    pts = []
    for axis in faces:
        n_axis = int(round(n * areas[axis] / sum(areas) / 2.0))
        other = [a for a in faces if a != axis]
        for sign in (-1.0, 1.0):
            u = rng.random((n_axis, 2)).astype(np.float32)
            p = np.empty((n_axis, 3), dtype=np.float32)
            p[:, other[0]] = tmin[other[0]] + u[:, 0] * (tmax[other[0]] - tmin[other[0]])
            p[:, other[1]] = tmin[other[1]] + u[:, 1] * (tmax[other[1]] - tmin[other[1]])
            p[:, axis] = (tmax[axis] if sign > 0 else tmin[axis]) + sign * 0.002
            pts.append(p)
    return np.concatenate(pts, 0)


class RoomScene:
    """房间场景基类：共享 SDF 求导、渲染、GT 表面体素掩码与房间颜色。"""

    voxel_size = VOXEL_SIZE

    def __init__(self):
        self.room_c = ((ROOM_MIN + ROOM_MAX) / 2.0).astype(np.float32)
        self.room_h = ((ROOM_MAX - ROOM_MIN) / 2.0).astype(np.float32)

    # ---------- 几何 ----------
    def room_free_sdf(self, p):
        return -sdf_box(p, self.room_c, self.room_h)

    def sdf(self, p):
        raise NotImplementedError

    def gradient(self, p, eps=1e-4):
        p = np.asarray(p, dtype=np.float32)
        gs = []
        for axis in range(3):
            e = np.zeros(3, dtype=np.float32)
            e[axis] = eps
            gs.append((self.sdf(p + e) - self.sdf(p - e)) / (2.0 * eps))
        g = np.stack(gs, axis=-1)
        return g / np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-8)

    # ---------- 房间颜色 ----------
    def room_color(self, p):
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

    # ---------- GT 表面体素掩码 ----------
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

    # ---------- 房间墙面采样 ----------
    def _sample_room_walls(self, rng, n_room, exclude_fn):
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
                pts.append(p[exclude_fn(p) > 0.005])
        return pts


class DemoScene(RoomScene):
    def __init__(self):
        super().__init__()
        self.sphere_c = np.array([0.75, 0.95, 0.33], dtype=np.float32)
        self.sphere_r = 0.22
        self.table_c = np.array([1.65, 1.15, 0.18], dtype=np.float32)
        self.table_h = np.array([0.28, 0.20, 0.18], dtype=np.float32)
        self.cyl_c = np.array([1.95, 0.45, 0.60], dtype=np.float32)
        self.cyl_r = 0.04
        self.cyl_hh = 0.60

    def _object_sdf(self, p):
        return np.minimum(
            np.minimum(
                sdf_sphere(p, self.sphere_c, self.sphere_r),
                sdf_box(p, self.table_c, self.table_h),
            ),
            sdf_cylinder_z(p, self.cyl_c, self.cyl_r, self.cyl_hh),
        )

    def sdf(self, p):
        p = np.asarray(p, dtype=np.float32)
        return np.minimum(self.room_free_sdf(p), self._object_sdf(p))

    def color(self, p):
        p = np.asarray(p, dtype=np.float32)
        col = self.room_color(p)
        d_room = self.room_free_sdf(p)
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

    def sample_surface(self, n_room=24000, n_sphere=3000, n_table=5000, n_cyl=3000, seed=0):
        rng = np.random.default_rng(seed)
        pts = self._sample_room_walls(rng, n_room, self._object_sdf)

        # 球面：Fibonacci 采样
        pts.append(sample_sphere_surface(rng, self.sphere_c, self.sphere_r, n_sphere))

        # 桌面 6 个面（原实现用 +2 mm 外偏，保持数值一致）
        tmin = self.table_c - self.table_h
        tmax = self.table_c + self.table_h
        tdim = (tmax - tmin).astype(np.float64)
        faces = [0, 1, 2]
        areas = [2.0 * tdim[a] * tdim[b] for a, b in ((1, 2), (0, 2), (0, 1))]
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


class PrimitiveScene(RoomScene):
    """房间 + 单个简单几何体（sphere / box / cylinder），用于基础重建验证。"""

    def __init__(self, kind="sphere"):
        super().__init__()
        self.kind = kind
        self.voxel_size = 0.06
        if kind == "sphere":
            self.c = np.array([1.20, 0.80, 0.32], dtype=np.float32)
            self.r = 0.32
        elif kind == "box":
            self.c = np.array([1.20, 0.80, 0.30], dtype=np.float32)
            self.h = np.array([0.32, 0.26, 0.30], dtype=np.float32)
        elif kind == "cylinder":
            self.c = np.array([1.20, 0.80, 0.45], dtype=np.float32)
            self.r = 0.22
            self.hh = 0.45
        else:
            raise ValueError(f"unknown primitive kind: {kind}")

    def _object_sdf(self, p):
        if self.kind == "sphere":
            return sdf_sphere(p, self.c, self.r)
        if self.kind == "box":
            return sdf_box(p, self.c, self.h)
        return sdf_cylinder_z(p, self.c, self.r, self.hh)

    def sdf(self, p):
        p = np.asarray(p, dtype=np.float32)
        return np.minimum(self.room_free_sdf(p), self._object_sdf(p))

    def _object_color(self, p):
        p = np.asarray(p, dtype=np.float32)
        if self.kind == "sphere":
            bands = 0.5 + 0.5 * np.sin(26.0 * p[..., 2] + 4.0 * p[..., 0])
            return np.stack([0.92 - 0.16 * bands, 0.50 + 0.16 * bands,
                             0.22 + 0.10 * bands], axis=-1)
        if self.kind == "box":
            stripe = 0.5 + 0.5 * np.sin(18.0 * p[..., 0] + 9.0 * p[..., 1])
            base = np.array([0.24, 0.52, 0.72], np.float32)
            return base[None] * (0.82 + 0.18 * stripe)[..., None]
        rings = 0.5 + 0.5 * np.sin(34.0 * p[..., 2])
        return np.stack([0.26 + 0.10 * rings, 0.62 - 0.14 * rings,
                         0.40 + 0.10 * rings], axis=-1)

    def color(self, p):
        p = np.asarray(p, dtype=np.float32)
        col = self.room_color(p)
        d_room = self.room_free_sdf(p)
        d_obj = self._object_sdf(p)
        obj = self._object_color(p)
        col = np.where((d_obj <= d_room + 1e-6)[..., None], obj, col)
        return np.clip(col, 0.0, 1.0).astype(np.float32)

    def sample_surface(self, n_room=24000, n_obj=8000, seed=0):
        rng = np.random.default_rng(seed)
        pts = self._sample_room_walls(rng, n_room, self._object_sdf)
        if self.kind == "sphere":
            pts.append(sample_sphere_surface(rng, self.c, self.r, n_obj))
        elif self.kind == "box":
            pts.append(sample_box_surface(rng, self.c - self.h, self.c + self.h, n_obj))
        else:
            pts.append(sample_cylinder_surface(rng, self.c, self.r, self.hh, n_obj))
        return np.concatenate(pts, axis=0)


class RobotArmScene(RoomScene):
    """房间 + 固定姿态的多关节串联机械臂（底座、关节、连杆、两指夹爪）。

    机械臂由 10 个部件组成（1 个底座、4 个关节球、4 段连杆/腕部、2 根手指），
    每个部件有独立颜色；SDF 取各部件的并集，用于检验体素-高斯组合体重建。
    """

    def __init__(self):
        super().__init__()
        self.voxel_size = 0.05
        base_c = (1.20, 0.80, 0.10)
        shoulder = (1.20, 0.80, 0.24)
        elbow = (1.20, 0.80, 0.64)
        wrist = (1.48, 0.98, 0.94)
        palm = (1.66, 1.10, 1.08)
        # (name, sdf, color, sampler, label)
        self.parts = [
            ("base", lambda p: sdf_cylinder_z(np.asarray(p), np.array(base_c, np.float32), 0.16, 0.10),
             lambda p: np.tile(np.array([0.22, 0.24, 0.28], np.float32), (len(p), 1))
             * (0.9 + 0.1 * np.sin(30.0 * p[..., 2]))[:, None],
             lambda rng, n: sample_cylinder_surface(rng, base_c, 0.16, 0.10, n)),
            ("shoulder", lambda p: sdf_sphere(np.asarray(p), np.array(shoulder, np.float32), 0.085),
             lambda p: np.tile(np.array([0.62, 0.64, 0.68], np.float32), (len(p), 1))
             * (0.85 + 0.15 * np.sin(40.0 * p[..., 0]))[:, None],
             lambda rng, n: sample_sphere_surface(rng, shoulder, 0.085, n)),
            ("upper_arm", lambda p: sdf_capsule(np.asarray(p), shoulder, elbow, 0.055),
             lambda p: np.tile(np.array([0.86, 0.45, 0.18], np.float32), (len(p), 1))
             * (0.86 + 0.14 * np.sin(18.0 * p[..., 2]))[:, None],
             lambda rng, n: sample_capsule_surface(rng, shoulder, elbow, 0.055, n)),
            ("elbow", lambda p: sdf_sphere(np.asarray(p), np.array(elbow, np.float32), 0.075),
             lambda p: np.tile(np.array([0.62, 0.64, 0.68], np.float32), (len(p), 1))
             * (0.85 + 0.15 * np.sin(40.0 * p[..., 1]))[:, None],
             lambda rng, n: sample_sphere_surface(rng, elbow, 0.075, n)),
            ("forearm", lambda p: sdf_capsule(np.asarray(p), elbow, wrist, 0.05),
             lambda p: np.tile(np.array([0.20, 0.56, 0.66], np.float32), (len(p), 1))
             * (0.86 + 0.14 * np.sin(16.0 * p[..., 0]))[:, None],
             lambda rng, n: sample_capsule_surface(rng, elbow, wrist, 0.05, n)),
            ("wrist", lambda p: sdf_sphere(np.asarray(p), np.array(wrist, np.float32), 0.062),
             lambda p: np.tile(np.array([0.62, 0.64, 0.68], np.float32), (len(p), 1))
             * (0.85 + 0.15 * np.sin(40.0 * p[..., 2]))[:, None],
             lambda rng, n: sample_sphere_surface(rng, wrist, 0.062, n)),
            ("wrist_link", lambda p: sdf_capsule(np.asarray(p), wrist, palm, 0.04),
             lambda p: np.tile(np.array([0.86, 0.45, 0.18], np.float32), (len(p), 1))
             * (0.86 + 0.14 * np.sin(22.0 * p[..., 2]))[:, None],
             lambda rng, n: sample_capsule_surface(rng, wrist, palm, 0.04, n)),
            ("palm", lambda p: sdf_sphere(np.asarray(p), np.array(palm, np.float32), 0.045),
             lambda p: np.tile(np.array([0.78, 0.18, 0.16], np.float32), (len(p), 1))
             * (0.88 + 0.12 * np.sin(30.0 * p[..., 0]))[:, None],
             lambda rng, n: sample_sphere_surface(rng, palm, 0.045, n)),
            ("finger_a", lambda p: sdf_capsule(np.asarray(p), (1.66, 1.05, 1.08),
                                               (1.82, 1.00, 1.12), 0.018),
             lambda p: np.tile(np.array([0.30, 0.30, 0.32], np.float32), (len(p), 1)),
             lambda rng, n: sample_capsule_surface(rng, (1.66, 1.05, 1.08),
                                                   (1.82, 1.00, 1.12), 0.018, n)),
            ("finger_b", lambda p: sdf_capsule(np.asarray(p), (1.66, 1.15, 1.08),
                                               (1.82, 1.20, 1.12), 0.018),
             lambda p: np.tile(np.array([0.30, 0.30, 0.32], np.float32), (len(p), 1)),
             lambda rng, n: sample_capsule_surface(rng, (1.66, 1.15, 1.08),
                                                   (1.82, 1.20, 1.12), 0.018, n)),
        ]

    def _object_sdf(self, p):
        P = np.asarray(p, np.float32)
        d = self.parts[0][1](P)
        for part in self.parts[1:]:
            d = np.minimum(d, part[1](P))
        return d

    def sdf(self, p):
        p = np.asarray(p, dtype=np.float32)
        return np.minimum(self.room_free_sdf(p), self._object_sdf(p))

    def color(self, p):
        p = np.asarray(p, dtype=np.float32)
        col = self.room_color(p)
        d_room = self.room_free_sdf(p)
        dmin = d_room.copy()
        best = np.full(len(p), -1, dtype=np.int32)
        part_cols = []
        for i, (_, sdf_fn, color_fn, _) in enumerate(self.parts):
            d = sdf_fn(p)
            part_cols.append(color_fn(p))
            mask = d < dmin
            dmin = np.where(mask, d, dmin)
            best = np.where(mask, i, best)
        for i, pc in enumerate(part_cols):
            col = np.where((best == i)[:, None], pc, col)
        return np.clip(col, 0.0, 1.0).astype(np.float32)

    def sample_surface(self, n_room=24000, seed=0):
        rng = np.random.default_rng(seed)
        pts = self._sample_room_walls(rng, n_room, self._object_sdf)
        counts = {"base": 800, "shoulder": 500, "upper_arm": 1400, "elbow": 500,
                  "forearm": 1400, "wrist": 500, "wrist_link": 1200, "palm": 500,
                  "finger_a": 300, "finger_b": 300}
        for name, _, _, sampler in self.parts:
            pts.append(sampler(rng, counts[name]))
        return np.concatenate(pts, axis=0)


def make_scene(name="demo", primitive="sphere"):
    """场景工厂：run_demo --scene demo|primitive|robot。"""
    if name == "demo":
        return DemoScene()
    if name == "primitive":
        return PrimitiveScene(primitive)
    if name == "robot":
        return RobotArmScene()
    raise ValueError(f"unknown scene: {name}")
