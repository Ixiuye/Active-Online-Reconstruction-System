"""体素-高斯约束损失。

全部约束按体素置信度 conf = w / (w + w0) 调制：体素观测越充分，约束越强。
这保证体素地图不可靠处（透明、超量程、弱观测）不会把 3DGS 拉偏。
"""
from __future__ import annotations

import torch


def huber(x, delta=0.05):
    ax = x.abs()
    return torch.where(ax < delta, 0.5 * x * x / delta, ax - 0.5 * delta)


def voxel_constraint_losses(model, vmap, trunc, n_sample=2048, w0=2.0,
                            free_margin=0.5):
    """返回 (surface, normal, free_space, rotation_prior) 四个标量损失。"""
    mu = model.means()
    frame = model.frames()
    scales = model.scales()
    opacity = model.opacity()

    g = mu.shape[0]
    n = min(n_sample, g)
    idx = torch.randint(0, g, (n,), device=mu.device)

    mu_s = mu[idx]
    s_mu = vmap.query_torch(mu_s)
    w_mu = vmap.query_weight_torch(mu_s)
    conf = w_mu / (w_mu + w0)

    l_surface = (huber(s_mu * trunc, delta=0.05) * conf).mean() / max(trunc, 1e-6)

    grad = vmap.query_grad_torch(mu_s)
    n_g = frame[idx][:, :, 2]
    cos = (grad * n_g).sum(-1).abs().clamp(max=1.0)
    l_normal = ((1.0 - cos) * conf).mean()

    eps = torch.randn_like(mu_s)
    x = mu_s + torch.einsum("gij,gj->gi", frame[idx], scales[idx] * eps)
    s_x = vmap.query_torch(x)
    w_x = vmap.query_weight_torch(x)
    conf_x = w_x / (w_x + w0)
    alpha = opacity[idx]
    l_free = (alpha * torch.relu(s_x - free_margin) * conf_x).mean()

    l_rot = (torch.tanh(model.rot_raw[idx]) ** 2).mean()

    return l_surface, l_normal, l_free, l_rot
