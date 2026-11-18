"""同题对照：随机 / 启发式 NBV / 学习策略，在完全相同的 episode 上比。

每个 episode 用同一 seed 构建（同一初始视角 + 同一场景开关），三种策略各跑一遍：
- random    : 在合法候选里均匀随机（下界）
- heuristic : active_recon/viewpoint/heuristic.select_nbv，utility = 可见目标权重 - 30 * 路程（原方法）
- rl        : outputs/rl_policy.pt 的 PPO 策略

指标（逐步记录）：新增已知体素、GT 表面覆盖率、累计路程；末尾给汇总表与曲线。

用法：
    python eval_policies.py                  # 12 个 episode，固定 GT 场景（默认）
    python eval_policies.py --episodes 30 --randomize   # 随机场景泛化测试
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

from ..viewpoint.heuristic import select_nbv
from ..viewpoint.rl import ExplorationEnv, heuristic_action, load_policy
from ..common.camera import H_IMG, K, W_IMG
from ..common.paths import OUTPUTS
from .plots import setup_font


def _heuristic_action(env):
    a, _ = heuristic_action(env.cands, env.vmap, env._pts, env._wts, env.K, env.H, env.W,
                            env.current_pos, env.visited)
    return a


def run_episode(kind, seed, args, policy=None):
    env = ExplorationEnv(K, H_IMG, W_IMG, steps=args.steps, n_init=args.n_init,
                         seed=seed, randomize=args.randomize,
                         init_jitter=args.init_jitter,
                         depth_noise=getattr(args, "noise", 0.0))
    feats, valid = env.reset(np.random.default_rng(seed))
    rng = np.random.default_rng(seed + 777)
    steps = []
    for _ in range(args.steps):
        if kind == "random":
            a = int(rng.choice(np.flatnonzero(valid)))
        elif kind == "heuristic":
            a = _heuristic_action(env)
            if a is None:
                break
        else:
            with torch.no_grad():
                logit, _ = policy(torch.as_tensor(feats)[None], torch.as_tensor(valid)[None])
            if getattr(args, "sample", False):
                probs = torch.softmax(logit[0], dim=-1).numpy()
                probs = probs / max(probs.sum(), 1e-12)
                a = int(rng.choice(len(probs), p=probs))
            else:
                a = int(torch.argmax(logit[0]))
        feats, valid, rew, done, info = env.step(a)
        steps.append(info)
        if done:
            break
    curve = [s["coverage"] for s in steps]
    curve += [curve[-1] if curve else 0.0] * (args.steps - len(curve))  # 提前结束时补齐
    return {"steps": steps,
            "final_coverage": steps[-1]["coverage"] if steps else 0.0,
            "new_known": int(sum(s["new_known"] for s in steps)),
            "new_surface": int(sum(s["new_surface"] for s in steps)),
            "quality_mean": float(np.mean([s["quality"] for s in steps])) if steps else 0.0,
            "conflict_mean": float(np.mean([s["conflict"] for s in steps])) if steps else 0.0,
            "path": float(sum(s["path"] for s in steps)),
            "coverage_curve": curve}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--n-init", type=int, default=4,
                    help="初始视角数，默认 4 = run_demo.py 的设置")
    ap.add_argument("--init-jitter", type=float, default=0.0,
                    help="初始位置抖动（m）；0 = 与 run_demo.py 完全相同的初始位姿")
    ap.add_argument("--randomize", action="store_true",
                    help="随机化场景（泛化测试）；默认用固定 GT 场景")
    ap.add_argument("--noise", type=float, default=0.0,
                    help="逐像素深度噪声 σ(m)，应与训练时一致才公平")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample", action="store_true",
                    help="按策略分布采样选点（默认贪心 argmax；贪心在部分长 horizon 策略上会退化）")
    args = ap.parse_args()

    out_dir = str(OUTPUTS)
    ckpt = args.ckpt or os.path.join(out_dir, "rl_policy.pt")
    if not os.path.exists(ckpt):
        raise SystemExit(f"找不到策略权重 {ckpt}，先跑 python train_rl.py")
    setup_font()
    policy = load_policy(ckpt, "cpu")

    seeds = [args.seed + i for i in range(args.episodes)]
    kinds = ("random", "heuristic", "rl")
    res = {k: [run_episode(k, s, args, policy) for s in seeds] for k in kinds}

    print(f"对照设置：{args.episodes} episodes × {args.steps} 步，"
          f"场景 = {'随机化' if args.randomize else '固定 GT 场景'}\n")
    header = f"{'policy':10s} {'覆盖率':>8s} {'±':>6s} {'新增已知体素':>12s} {'±':>8s} {'路程(m)':>9s}"
    print(header)
    summary = {}
    for k in kinds:
        cov = np.array([r["final_coverage"] for r in res[k]]) * 100
        known = np.array([r["new_known"] for r in res[k]], float)
        path = np.array([r["path"] for r in res[k]])
        summary[k] = {"coverage_pct_mean": float(cov.mean()), "coverage_pct_std": float(cov.std()),
                      "new_known_mean": float(known.mean()), "new_known_std": float(known.std()),
                      "path_m_mean": float(path.mean()),
                      "coverage_curve_mean": np.mean(
                          [r["coverage_curve"] for r in res[k]], axis=0).tolist()}
        print(f"{k:10s} {cov.mean():7.2f}% {cov.std():6.2f} {known.mean():12.1f} "
              f"{known.std():8.1f} {path.mean():9.2f}")

    base = summary["heuristic"]["coverage_pct_mean"]
    rl = summary["rl"]["coverage_pct_mean"]
    print(f"\nRL 相对启发式：覆盖率 {rl - base:+.2f} 个百分点；"
          f"新增体素 {summary['rl']['new_known_mean'] - summary['heuristic']['new_known_mean']:+.1f}；"
          f"路程 {summary['rl']['path_m_mean'] - summary['heuristic']['path_m_mean']:+.2f} m")

    colors = {"random": "tab:gray", "heuristic": "tab:orange", "rl": "tab:blue"}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    xs = np.arange(1, args.steps + 1)
    for k in kinds:
        c = summary[k]["coverage_curve_mean"]
        axes[0].plot(xs[:len(c)], np.array(c) * 100, marker="o", label=k, color=colors[k])
    axes[0].set_title("GT 表面覆盖率 vs 探索步")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("%")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    vals = [summary[k]["coverage_pct_mean"] for k in kinds]
    errs = [summary[k]["coverage_pct_std"] for k in kinds]
    axes[1].bar(list(kinds), vals, yerr=errs, color=[colors[k] for k in kinds], capsize=4)
    axes[1].set_title("末尾覆盖率（均值±std）")
    axes[1].set_ylabel("%")
    axes[1].grid(alpha=0.3, axis="y")

    axes[2].bar(list(kinds), [summary[k]["path_m_mean"] for k in kinds],
                color=[colors[k] for k in kinds])
    axes[2].set_title("累计路程")
    axes[2].set_ylabel("m")
    axes[2].grid(alpha=0.3, axis="y")
    fig.suptitle("主动探索策略对照：随机 / 启发式 NBV / PPO 学习策略")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "policy_comparison.png"), dpi=150)
    plt.close(fig)

    with open(os.path.join(out_dir, "policy_eval.json"), "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "summary": summary,
                   "episodes": {k: [{"final_coverage": r["final_coverage"],
                                     "new_known": r["new_known"], "path": r["path"]}
                                    for r in res[k]] for k in kinds}},
                  f, ensure_ascii=False, indent=2)
    print(f"\n结果: {os.path.join(out_dir, 'policy_comparison.png')}")
    print(f"      {os.path.join(out_dir, 'policy_eval.json')}")


if __name__ == "__main__":
    main()
