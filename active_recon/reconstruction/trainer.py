"""高斯模型的训练循环：按模式（photo / depth / voxel）采样光线做一步优化。"""
from __future__ import annotations

import time

import torch

from ..common.camera import H_IMG, W_IMG
from ..common.scene import TRUNC
from .losses import huber, voxel_constraint_losses
from .renderer import render_rays


def train_model(model, views, vmap, mode, iters, device, seed=0, lam=1.0,
                rays=512, verbose=False):
    opt = model.make_optimizer()
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    n_pix = H_IMG * W_IMG
    r = min(rays, n_pix)
    t0 = time.time()

    for it in range(iters):
        view = views[int(torch.randint(len(views), (1,)).item())]
        pix = torch.randint(0, n_pix, (r,), device=device, generator=gen)
        origins = view.center_t.unsqueeze(0).expand(r, 3)
        dirs = view.dirs_t[pix]
        gt_rgb = view.rgb_t[pix]
        gt_depth = view.depth_t[pix]

        pred_rgb, pred_depth, acc = render_rays(*model.properties(), origins, dirs)
        loss_rgb = ((pred_rgb - gt_rgb) ** 2).mean()

        if mode == "photo":
            loss = loss_rgb
        else:
            mask = (gt_depth > 0) & (acc > 0.25)
            if mask.any():
                loss_depth = huber(pred_depth - gt_depth, 0.05)[mask].mean()
            else:
                loss_depth = pred_depth.sum() * 0.0
            loss = loss_rgb + loss_depth

        if mode == "voxel":
            ls, ln, lf, lr = voxel_constraint_losses(model, vmap, TRUNC, n_sample=1024)
            loss = loss + lam * (1.2 * ls + 0.5 * ln + 1.5 * lf + 0.05 * lr)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if verbose and (it + 1) % 100 == 0:
            print(f"    it {it + 1}/{iters}  loss={loss.item():.4f}")

    return time.time() - t0
