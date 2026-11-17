"""基于光线的可微 Gaussian splatting 渲染器（简化版 ray-splat）。

对每条光线，把高斯变换到其局部帧，计算光线到高斯中心的最近距离，
得到 alpha = opacity * exp(-0.5 * d^2)（d 在按尺度归一化的坐标中），
然后按最近点深度排序、做标准 front-to-back alpha 合成。

这是一个用于演示的简化 EWA/ray-splat 混合：不含 tile 并行和球谐视角相关颜色，
但保留了 3DGS 的“投影 -> 排序 -> alpha 合成”核心，梯度可回传。
"""
from __future__ import annotations

import torch


def render_rays(mu, frame, scales, colors, opacity, origins, dirs):
    """返回 (rgb [R,3], depth [R], acc [R])。

    mu/scales/colors: [G,3]；frame: [G,3,3]（列为切向1、切向2、法向）；
    opacity: [G]；origins/dirs: [R,3]（dirs 需归一化）。
    """
    rel = origins[:, None, :] - mu[None, :, :]              # [R,G,3]
    o_l = torch.einsum("gij,rgj->rgi", frame, rel)
    d_l = torch.einsum("gij,rj->rgi", frame, dirs)
    inv_s = 1.0 / scales[None]
    a = o_l * inv_s
    b = d_l * inv_s
    bb = (b * b).sum(-1) + 1e-8
    t = -(a * b).sum(-1) / bb
    valid = (t > 0.02) & (t < 8.0)
    t = torch.clamp(t, min=0.02)
    d2 = ((a + t[..., None] * b) ** 2).sum(-1)
    alpha = opacity[None] * torch.exp(-0.5 * d2)
    alpha = torch.where(valid, alpha, torch.zeros_like(alpha))

    order = torch.argsort(t, dim=1)
    alpha_s = torch.gather(alpha, 1, order)
    depth_s = torch.gather(t, 1, order)
    color_s = colors[order]

    trans = torch.cumprod(1.0 - alpha_s, dim=1)
    trans_prev = torch.cat(
        [torch.ones_like(trans[:, :1]), trans[:, :-1]], dim=1
    )
    weights = alpha_s * trans_prev
    acc = 1.0 - trans[:, -1]
    rgb = (weights[..., None] * color_s).sum(1)
    depth = (weights * depth_s).sum(1) / (weights.sum(1) + 1e-8)
    return rgb, depth, acc
