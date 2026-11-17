"""相机与视图：分辨率、内参、深度噪声开关、GT 视角封装。"""
from __future__ import annotations

import numpy as np
import torch

from .utils import pixel_rays


W_IMG, H_IMG = 128, 96
K = (110.0, 110.0, W_IMG / 2.0, H_IMG / 2.0)
DEPTH_NOISE = 0.0   # 采集视角时逐像素加 σ(m) 深度噪声（0=无噪声），由 set_depth_noise 设置


# ---------------------------------------------------------------- 数据视图
class View:
    def __init__(self, depth, rgb, R, t, device):
        self.depth = depth
        self.rgb = rgb
        self.R = R
        self.t = t
        center, dirs = pixel_rays(R, t, K, H_IMG, W_IMG)
        self.center_np = center
        self.dirs_np = dirs
        self.center_t = torch.as_tensor(center, device=device)
        self.dirs_t = torch.as_tensor(dirs, device=device)
        self.rgb_t = torch.as_tensor(rgb.reshape(-1, 3), device=device)
        self.depth_t = torch.as_tensor(depth.reshape(-1), device=device)


def render_scene_view(scene, pose, device):
    R, t = pose
    depth, rgb, _ = scene.render_view(R, t, K, H_IMG, W_IMG)
    if DEPTH_NOISE > 0:
        hit = depth > 0
        noise = np.random.normal(0.0, DEPTH_NOISE, depth.shape).astype(np.float32)
        depth = np.where(hit, np.maximum(depth + noise, 0.05), 0.0).astype(np.float32)
    return View(depth, rgb, R, t, device)


def set_depth_noise(sigma):
    """统一入口：main() 里调用，避免各模块各自维护一份全局。"""
    global DEPTH_NOISE
    DEPTH_NOISE = float(sigma)
