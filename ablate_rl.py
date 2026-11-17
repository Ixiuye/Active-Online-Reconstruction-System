"""奖励项消融：w_quality × w_conflict 的 2×2，全部在深度噪声下训练与评估。

问题：加了"深度准确度"两项之后，RL 到底靠哪一项变好？没有噪声时 conflict 恒为 0、
quality 也几乎无区分度，所以消融必须带噪声跑。本脚本把两件事一起做完：

  训练：随机场景 + 3~6 初始视角 + 逐像素深度噪声 σ（默认 2 cm）+ 250 episode
  评估：固定 GT 场景 + 4 初始视角（= run_demo 协议）+ 同样噪声，12 episode × 6 步，
        对照启发式在**同一批采样候选**上的表现

四格：
  ctrl      quality=0,  conflict=0    （只有 Δ表面 + 路程，等于 v1 的奖励结构）
  quality   quality=0.8, conflict=0
  conflict  quality=0,  conflict=2.0
  full      quality=0.8, conflict=2.0

用法：
    python ablate_rl.py                 # 默认 4 格 × 250 episode + 12 episode 评估
    python ablate_rl.py --episodes 400 --eval-episodes 20 --noise 0.01
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

from demo.rl_active import CandidatePolicy, ExplorationEnv, load_policy, save_policy, train_ppo
from eval_policies import run_episode
from run_demo import H_IMG, K, W_IMG, setup_font

GRID = [
    ("ctrl", 0.0, 0.0),
    ("quality", 0.8, 0.0),
    ("quality02", 0.2, 0.0),
    ("conflict", 0.0, 2.0),
    ("full", 0.8, 2.0),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=250)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--eval-episodes", type=int, default=12)
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-train", action="store_true", help="复用已有 ckpt，只做评估")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "outputs")
    ck_dir = os.path.join(out_dir, "ablate")
    os.makedirs(ck_dir, exist_ok=True)
    setup_font()

    # ---------------- 训练四格 ----------------
    for tag, wq, wc in GRID:
        ck = os.path.join(ck_dir, f"rl_{tag}.pt")
        if args.skip_train and os.path.exists(ck):
            print(f"[{tag}] 复用 {ck}")
            continue
        print(f"\n=== 训练 [{tag}] quality={wq} conflict={wc}  "
              f"σ={args.noise * 100:.1f}cm {args.episodes}ep ===")
        env = ExplorationEnv(K, H_IMG, W_IMG, steps=args.steps, n_init_range=(3, 6),
                             seed=args.seed, randomize=True, init_jitter=0.08,
                             depth_noise=args.noise, w_quality=wq, w_conflict=wc)
        policy = CandidatePolicy()
        hist = train_ppo(policy, env, episodes=args.episodes, seed=args.seed,
                         log=lambda s: print(s, flush=True))
        meta = {"grid": tag, "w_quality": wq, "w_conflict": wc,
                "depth_noise": args.noise, "episodes": args.episodes,
                "reward": f"1.0*Δsurface/100 + {wq}*quality - {wc}*conflict - 0.3*path"}
        save_policy(ck, policy, meta)
        with open(os.path.join(ck_dir, f"train_{tag}.json"), "w", encoding="utf-8") as f:
            json.dump({"meta": meta, "history": hist}, f, ensure_ascii=False, indent=2)

    # ---------------- 评估 ----------------
    class A:
        steps = args.steps
        n_init = 4
        init_jitter = 0.0
        randomize = False
        noise = args.noise

    seeds = [args.seed + i for i in range(args.eval_episodes)]
    rows = {}

    def summarise(name, res):
        cov = np.array([r["final_coverage"] for r in res]) * 100
        rows[name] = {
            "coverage_pct_mean": float(cov.mean()), "coverage_pct_std": float(cov.std()),
            "new_surface_mean": float(np.mean([r["new_surface"] for r in res])),
            "new_known_mean": float(np.mean([r["new_known"] for r in res])),
            "quality_mean": float(np.mean([r["quality_mean"] for r in res])),
            "conflict_mean": float(np.mean([r["conflict_mean"] for r in res])),
            "path_m_mean": float(np.mean([r["path"] for r in res])),
        }

    print(f"\n=== 评估：{args.eval_episodes} episodes × {args.steps} 步，"
          f"σ={args.noise * 100:.1f}cm ===")
    summarise("heuristic", [run_episode("heuristic", s, A) for s in seeds])
    for tag, _, _ in GRID:
        pol = load_policy(os.path.join(ck_dir, f"rl_{tag}.pt"))
        summarise(tag, [run_episode("rl", s, A, pol) for s in seeds])

    names = ["heuristic"] + [t for t, _, _ in GRID]
    print(f"\n{'policy':10s} {'覆盖率':>8s} {'±':>6s} {'Δ表面':>7s} {'质量':>7s} "
          f"{'冲突':>7s} {'路程':>7s}")
    for n in names:
        r = rows[n]
        print(f"{n:10s} {r['coverage_pct_mean']:7.2f}% {r['coverage_pct_std']:6.2f} "
              f"{r['new_surface_mean']:7.1f} {r['quality_mean']:7.3f} "
              f"{r['conflict_mean']:7.3f} {r['path_m_mean']:7.2f}")

    base = rows["heuristic"]
    print("\n相对启发式：")
    for n in names[1:]:
        r = rows[n]
        print(f"  {n:9s} 覆盖率 {r['coverage_pct_mean'] - base['coverage_pct_mean']:+6.2f} pp   "
              f"Δ表面 {r['new_surface_mean'] - base['new_surface_mean']:+7.1f}   "
              f"质量 {r['quality_mean'] - base['quality_mean']:+.3f}   "
              f"冲突 {r['conflict_mean'] - base['conflict_mean']:+.3f}   "
              f"路程 {r['path_m_mean'] - base['path_m_mean']:+.2f} m")

    colors = {"heuristic": "tab:orange", "ctrl": "tab:gray", "quality": "tab:green",
              "quality02": "tab:olive", "conflict": "tab:red", "full": "tab:blue"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    x = np.arange(len(names))
    for ax, key, title in zip(axes,
                              ("coverage_pct_mean", "quality_mean", "path_m_mean"),
                              ("GT 表面覆盖率 (%)", "平均深度可分性", "累计路程 (m)")):
        vals = [rows[n][key] for n in names]
        errs = [rows[n]["coverage_pct_std"] for n in names] if key == "coverage_pct_mean" else None
        ax.bar(x, vals, yerr=errs, capsize=4, color=[colors.get(n, "tab:purple") for n in names])
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20)
        ax.set_title(title)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle(f"奖励项消融（σ={args.noise * 100:.1f} cm，{args.episodes} episode 训练）")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "rl_ablation.png"), dpi=150)
    plt.close(fig)

    with open(os.path.join(out_dir, "rl_ablation.json"), "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "grid": [{"tag": t, "w_quality": q, "w_conflict": c}
                                                  for t, q, c in GRID],
                   "results": rows}, f, ensure_ascii=False, indent=2)
    print(f"\n表格: {os.path.join(out_dir, 'rl_ablation.json')}")
    print(f"图:   {os.path.join(out_dir, 'rl_ablation.png')}")


if __name__ == "__main__":
    main()
