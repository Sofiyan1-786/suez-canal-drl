"""
train_fcfs.py -- Baseline 1: First-Come-First-Served (current Suez rule).

Usage:
    python train_fcfs.py --n-episodes 100 --seed-start 1000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from suez_env import SuezCanalEnv


def fcfs_action(env: SuezCanalEnv) -> np.ndarray:
    """Build the FCFS action: pick up to C ships from the queue head."""
    n = min(env.max_waiting, len(env.waiting_queue))
    action = np.zeros(env.max_waiting, dtype=np.int8)
    pick = min(env.convoy_capacity, n)
    action[:pick] = 1
    return action


def run_episode(env: SuezCanalEnv, seed: int) -> dict:
    obs, info = env.reset(seed=seed)
    done = False
    while not done:
        action = fcfs_action(env)
        obs, reward, term, trunc, info = env.step(action)
        done = term or trunc
    summary = env.episode_summary()
    summary["reward"] = sum(h.get("reward", 0.0) for h in env.episode_history)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--disruption", action="store_true")
    parser.add_argument("--output", type=str, default="results/fcfs.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    env = SuezCanalEnv(seed=args.seed_start, disruption_active=args.disruption)

    print("=" * 60)
    print(f"FCFS baseline: {args.n_episodes} episodes (seed_start={args.seed_start}, "
          f"disruption={args.disruption})")
    print("=" * 60)

    results = []
    for ep in range(args.n_episodes):
        summary = run_episode(env, seed=args.seed_start + ep)
        results.append(summary)
        if (ep + 1) % 10 == 0 or ep < 3:
            print(f"  ep{ep+1:4d}: delay={summary['total_delay_h']:6.0f}h, "
                  f"capital=${summary['total_capital_cost_usd']:>12,.0f}, "
                  f"served={summary['total_ships_served']:>2d}")

    n = len(results)
    agg = {
        "method": "FCFS",
        "n_episodes": n,
        "disruption": args.disruption,
        "mean_total_delay_h": float(np.mean([r["total_delay_h"] for r in results])),
        "std_total_delay_h": float(np.std([r["total_delay_h"] for r in results])),
        "mean_total_capital_cost_usd": float(np.mean([r["total_capital_cost_usd"] for r in results])),
        "std_total_capital_cost_usd": float(np.std([r["total_capital_cost_usd"] for r in results])),
        "mean_total_ships_served": float(np.mean([r["total_ships_served"] for r in results])),
        "mean_total_cargo_value_usd": float(np.mean([r["total_cargo_value_usd"] for r in results])),
        "raw_per_episode": results,
    }
    print()
    print(f"FCFS mean: delay = {agg['mean_total_delay_h']:.0f}h, "
          f"capital = ${agg['mean_total_capital_cost_usd']:,.0f}, "
          f"served = {agg['mean_total_ships_served']:.1f}")

    with open(args.output, "w") as f:
        json.dump(agg, f, indent=2, default=str)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
