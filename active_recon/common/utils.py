"""相机与几何工具函数。"""
from __future__ import annotations

import numpy as np


def look_at(position, target, up=(0.0, 0.0, 1.0)):
    """返回 world->camera 的 (R, t)，满足 p_cam = R @ p_world + t（OpenCV 约定）。"""
    position = np.asarray(position, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    z = target - position
    z = z / np.linalg.norm(z)
    up = np.asarray(up, dtype=np.float64)
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-8:
        x = np.array([1.0, 0.0, 0.0])
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=0)
    t = -R @ position
    return R.astype(np.float32), t.astype(np.float32)


def camera_center(R, t):
    return -(np.asarray(t) @ np.asarray(R)).astype(np.float32)


def pixel_rays(R, t, K, height, width):
    """返回相机中心 [3] 与每个像素的世界射线方向 [H*W, 3]。"""
    fx, fy, cx, cy = K
    ys, xs = np.mgrid[0:height, 0:width]
    xs = xs.reshape(-1).astype(np.float32)
    ys = ys.reshape(-1).astype(np.float32)
    d_cam = np.stack([(xs - cx) / fx, (ys - cy) / fy, np.ones_like(xs)], axis=-1)
    d_cam /= np.linalg.norm(d_cam, axis=-1, keepdims=True)
    d_world = d_cam @ np.asarray(R)  # (R^T d_cam) 的行向量写法
    return camera_center(R, t), d_world.astype(np.float32)
