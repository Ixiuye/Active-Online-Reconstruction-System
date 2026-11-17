"""训练主动探索策略（PPO），把手工 NBV 效用换成学出来的候选打分。

用法：
    python train_rl.py                    # 默认 120 episode，约 2 分钟（CPU 即可）
    python train_rl.py --episodes 300     # 更长训练
    python train_rl.py --steps 8          # 与 run_demo.py --steps 对齐

产物：
    outputs/rl_policy.pt      策略权重（run_demo.py --policy rl 会加载它）
    outputs/rl_training.png   训练曲线（return / 新增已知体素 / 覆盖率）
    outputs/rl_training.json  逐 episode 记录

训练细节：
- episode = 随机化场景 + 随机 3 个初始视角 + T 步策略决策；
- 奖励 r = Δ已知体素/500 - 0.5 * 路径长度（只读 TSDF 地图，不用 GT）；
- PPO：clip 0.2、GAE(γ=0.95, λ=0.95)、价值损失 0.5、熵奖励 0.01；
- 覆盖率只在日志里用 GT 计算，不进入奖励。
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from demo.rl_active import CandidatePolicy, ExplorationEnv, save_policy, train_ppo
from run_demo import H_IMG, K, W_IMG, setup_font


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--update-every", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--n-cand", type=int, default=64)
    ap.add_argument("--noise", type=float, default=0.0,
                    help="训练时逐像素深度噪声 σ(m)，建议 0.01~0.02；0=无噪声")
    ap.add_argument("--w-surf", type=float, default=1.0, help="Δ表面体素奖励权重")
    ap.add_argument("--w-quality", type=float, default=0.8, help="深度可分性奖励权重")
    ap.add_argument("--w-conflict", type=float, default=2.0, help="深度-地图冲突罚权重")
    ap.add_argument("--w-path", type=float, default=0.3, help="路程罚权重")
    ap.add_argument("--tag", default="", help="输出文件名后缀，如 wq0（消融用）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    sfx = f"_{args.tag}" if args.tag else ""
    ckpt = args.out or os.path.join(out_dir, f"rl_policy{sfx}.pt")
    curve = os.path.join(out_dir, f"rl_training{sfx}.png")
    log_json = os.path.join(out_dir, f"rl_training{sfx}.json")

    setup_font()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(f"训练 PPO 策略：{args.episodes} episode × {args.steps} 步，hidden={args.hidden}")

    # 初始视角数在 3~6 之间随机：既包含“地图很空”的局面，也包含 run_demo 那种
    # 4 个角落视角已经把房间大致看过的局面，否则策略只会在空地图上表现好
    env = ExplorationEnv(K, H_IMG, W_IMG, steps=args.steps, n_init_range=(3, 6),
                         seed=args.seed, randomize=True, init_jitter=0.08,
                         n_cand=args.n_cand, depth_noise=args.noise,
                         w_surf=args.w_surf, w_quality=args.w_quality,
                         w_conflict=args.w_conflict, w_path=args.w_path)
    print(f"奖励权重 surf={args.w_surf} quality={args.w_quality} "
          f"conflict={args.w_conflict} path={args.w_path}；深度噪声 σ={args.noise*100:.1f} cm")
    policy = CandidatePolicy(hidden=args.hidden)
    hist = train_ppo(policy, env, episodes=args.episodes,
                     update_every=args.update_every, epochs=args.epochs,
                     lr=args.lr, seed=args.seed)

    meta = {"episodes": args.episodes, "steps": args.steps, "seed": args.seed,
            "hidden": args.hidden, "lr": args.lr, "n_cand": args.n_cand,
            "depth_noise": args.noise,
            "weights": {"surf": args.w_surf, "quality": args.w_quality,
                        "conflict": args.w_conflict, "path": args.w_path},
            "reward": f"{args.w_surf}*Δsurface/100 + {args.w_quality}*quality "
                      f"- {args.w_conflict}*conflict - {args.w_path}*path",
            "action_space": f"uniform-sampled candidates (K={args.n_cand}/step)",
            "algo": "PPO"}
    save_policy(ckpt, policy, meta)
    with open(log_json, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "history": hist}, f, ensure_ascii=False, indent=2)

    ep = np.array([h["episode"] for h in hist])
    ret = np.array([h["return"] for h in hist])
    surf = np.array([h["new_surface"] for h in hist])
    qual = np.array([h["quality"] for h in hist])
    cover = np.array([h["coverage"] for h in hist])
    path = np.array([h["path"] for h in hist])
    k = max(1, len(ep) // 12)

    def smooth(x):
        return np.convolve(x, np.ones(k) / k, mode="valid")

    fig, axes = plt.subplots(2, 3, figsize=(13, 6.4))
    for ax, (y, title, ylabel) in zip(axes.ravel(), [
        (ret, "episode 回报", "return"),
        (surf, "新增表面体素 (个)", "voxels"),
        (qual, "平均深度可分性", "quality"),
        (cover, "最终覆盖率（GT 评估用）", "coverage"),
        (path, "总路程 (m)", "m"),
        (surf / np.maximum(path, 1e-6), "单位路程新增表面体素", "voxels/m"),
    ]):
        ax.plot(ep, y, alpha=0.35, lw=0.9, color="tab:gray")
        if len(y) >= k:
            ax.plot(ep[k - 1:], smooth(y), lw=1.8, color="tab:blue")
        ax.set_title(title)
        ax.set_xlabel("episode")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    fig.suptitle(f"PPO 主动探索策略训练曲线（{args.episodes} episodes）")
    fig.tight_layout()
    fig.savefig(curve, dpi=150)
    plt.close(fig)

    n = max(1, len(hist) // 4)
    print("\n分段平均:")
    for i in range(4):
        seg = hist[i * n:(i + 1) * n]
        if not seg:
            continue
        print(f"  第 {i * n + 1:4d}-{(i + 1) * n:4d} episode  "
              f"return {np.mean([s['return'] for s in seg]):7.2f}  "
              f"新增体素 {np.mean([s['new_known'] for s in seg]):7.1f}  "
              f"覆盖率 {np.mean([s['coverage'] for s in seg]):.3f}  "
              f"路程 {np.mean([s['path'] for s in seg]):5.2f} m")
    print(f"\n策略已保存: {ckpt}")
    print(f"曲线: {curve}")
    print("下一步: python eval_policies.py     # 与启发式 NBV 同题对照")
    print("        python run_demo.py --policy rl   # 在完整流程里用 RL 选视角")


if __name__ == "__main__":
    main()
