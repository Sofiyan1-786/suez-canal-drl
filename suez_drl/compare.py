"""
compare.py -- Run all 4 methods on a common test set and produce thesis figures.

Usage:
    python compare.py --n-test 50 --seed-start 5000
    python compare.py --skip-rerun
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from suez_env import SuezCanalEnv
from train_fcfs import fcfs_action, run_episode as fcfs_run_episode
from train_ga import run_one_episode_with_weights
from train_milp_pulp import preview_arrivals, solve_milp


# ============================================================================
# PPO evaluation
# ============================================================================
def evaluate_ppo(model_path: str, seeds: list[int]) -> list[dict]:
    """Evaluate a trained PPO model on a list of seeds. Returns list of summaries."""
    from sb3_contrib import MaskablePPO
    model = MaskablePPO.load(model_path)
    env = SuezCanalEnv(seed=seeds[0])
    results = []
    for seed in seeds:
        obs, info = env.reset(seed=seed)
        done = False
        while not done:
            flat_mask = env.action_masks()
            action, _ = model.predict(obs, deterministic=True, action_masks=flat_mask)
            obs, r, term, trunc, info = env.step(action)
            done = term or trunc
        results.append(env.episode_summary())
    return results


# ============================================================================
# FCFS evaluation
# ============================================================================
def evaluate_fcfs(seeds: list[int]) -> list[dict]:
    env = SuezCanalEnv(seed=seeds[0])
    return [fcfs_run_episode(env, seed=seed) for seed in seeds]


# ============================================================================
# MILP evaluation (offline perfect info)
# ============================================================================
def evaluate_milp(seeds: list[int], time_limit_s: float = 30.0) -> list[dict]:
    env = SuezCanalEnv(seed=seeds[0])
    n_convoys = env.n_steps_per_episode
    C = env.convoy_capacity
    hps = env.hours_per_step
    results = []
    for seed in seeds:
        arrivals = preview_arrivals(env, seed=seed, disruption=False)
        sol = solve_milp(arrivals, n_convoys, C, hps, time_limit_s=time_limit_s)
        results.append({
            "total_capital_cost_usd": sol["total_capital_cost_usd"],
            "total_delay_h": sol["total_delay_h"],
            "total_ships_served": sol["n_ships_served"],
            "total_cargo_value_usd": sol["total_cargo_value_usd"],
        })
    return results


# ============================================================================
# GA evaluation (using pre-tuned weights from a JSON file)
# ============================================================================
def evaluate_ga(weights: np.ndarray, seeds: list[int]) -> list[dict]:
    env = SuezCanalEnv(seed=seeds[0])
    results = []
    for seed in seeds:
        s = run_one_episode_with_weights(env, weights, seed)
        results.append(s)
    return results


# ============================================================================
# Per-method evaluation summary
# ============================================================================
def summarize_method(name: str, summaries: list[dict]) -> dict:
    return {
        "method": name,
        "n_scenarios": len(summaries),
        "mean_capital_usd": float(np.mean([s["total_capital_cost_usd"] for s in summaries])),
        "std_capital_usd": float(np.std([s["total_capital_cost_usd"] for s in summaries])),
        "mean_delay_h": float(np.mean([s["total_delay_h"] for s in summaries])),
        "std_delay_h": float(np.std([s["total_delay_h"] for s in summaries])),
        "mean_ships_served": float(np.mean([s["total_ships_served"] for s in summaries])),
        "mean_cargo_value_usd": float(np.mean([s["total_cargo_value_usd"] for s in summaries])),
    }


# ============================================================================
# Plotting
# ============================================================================
def plot_cost_comparison(summaries: list[dict], out_path: str):
    """Bar chart of mean capital cost per method (with std error bars)."""
    methods = [s["method"] for s in summaries]
    means = [s["mean_capital_usd"] for s in summaries]
    stds = [s["std_capital_usd"] for s in summaries]
    # Color: FCFS = red, others = green
    colors = ["#d62728" if m == "FCFS" else "#2ca02c" for m in methods]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(methods, means, yerr=stds, color=colors, alpha=0.85, capsize=5)
    ax.set_ylabel("Mean Total Capital Cost (USD)")
    ax.set_title("Capital Cost of Waiting Cargo — Suez Canal Scheduling Methods\n"
                 "(lower is better)")
    # Annotate each bar
    for bar, m in zip(bars, means):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h, f"${h:,.0f}",
                ha="center", va="bottom", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_savings_vs_fcfs(summaries: list[dict], out_path: str):
    """Bar chart of % savings vs FCFS."""
    fcfs = next((s for s in summaries if s["method"] == "FCFS"), None)
    if not fcfs:
        return
    fcfs_cap = fcfs["mean_capital_usd"]
    methods, savings = [], []
    for s in summaries:
        if s["method"] == "FCFS":
            continue
        pct = 100 * (fcfs_cap - s["mean_capital_usd"]) / fcfs_cap
        methods.append(s["method"])
        savings.append(pct)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(methods, savings, color="#2ca02c", alpha=0.85)
    ax.set_ylabel("Capital Cost Savings vs FCFS (%)")
    ax.set_title("Value-Based Scheduling Reduces Cargo Capital Cost\n"
                 "(positive = better than current FCFS rule)")
    ax.axhline(0, color="black", linewidth=0.8)
    for bar, v in zip(bars, savings):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h, f"{v:+.1f}%",
                ha="center", va="bottom" if h > 0 else "top", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-test", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=5000)
    parser.add_argument("--ppo-model", type=str, default="models/ppo_suez")
    parser.add_argument("--ga-weights", type=str, default="results/ga.json")
    parser.add_argument("--milp-time-limit", type=float, default=30.0)
    parser.add_argument("--out-dir", type=str, default="results")
    parser.add_argument("--skip-rerun", action="store_true",
                        help="Reuse results from prior runs (read JSON files)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    seeds = [args.seed_start + i for i in range(args.n_test)]

    print("=" * 60)
    print(f"Comparing 4 methods on {args.n_test} test scenarios "
          f"(seeds {args.seed_start}..{args.seed_start + args.n_test - 1})")
    print("=" * 60)

    per_scenario_rows = []

    # ---- 1. FCFS --------------------------------------------------------
    if args.skip_rerun and os.path.exists("results/fcfs.json"):
        with open("results/fcfs.json") as f:
            data = json.load(f)
        print(f"[1/4] FCFS    : loaded from results/fcfs.json "
              f"(mean=${data['mean_total_capital_cost_usd']:,.0f})")
    else:
        print("[1/4] FCFS    : running...")
        t0 = time.time()
        fcfs_res = evaluate_fcfs(seeds)
        print(f"        done in {time.time()-t0:.1f}s")
    # We'll just re-run for the test seeds to be fair. Always do this for the actual comparison.
    print("[1/4] FCFS    : evaluating on test seeds...")
    t0 = time.time()
    fcfs_res = evaluate_fcfs(seeds)
    print(f"        done in {time.time()-t0:.1f}s, mean=${np.mean([r['total_capital_cost_usd'] for r in fcfs_res]):,.0f}")
    for seed, r in zip(seeds, fcfs_res):
        per_scenario_rows.append({
            "scenario_seed": seed,
            "method": "FCFS",
            **r,
        })
    fcfs_summary = summarize_method("FCFS", fcfs_res)

    # ---- 2. MILP --------------------------------------------------------
    print("[2/4] MILP-CBC: solving (offline perfect info)...")
    t0 = time.time()
    milp_res = evaluate_milp(seeds, time_limit_s=args.milp_time_limit)
    print(f"        done in {time.time()-t0:.1f}s, mean=${np.mean([r['total_capital_cost_usd'] for r in milp_res]):,.0f}")
    for seed, r in zip(seeds, milp_res):
        per_scenario_rows.append({
            "scenario_seed": seed,
            "method": "MILP-CBC",
            **r,
        })
    milp_summary = summarize_method("MILP-CBC", milp_res)

    # ---- 3. GA ----------------------------------------------------------
    ga_weights = None
    if os.path.exists(args.ga_weights):
        with open(args.ga_weights) as f:
            ga_data = json.load(f)
        ga_weights = np.array(ga_data["best_weights"])
        print(f"[3/4] GA      : loaded weights {ga_weights.round(3).tolist()} from {args.ga_weights}")
    else:
        print("[3/4] GA      : WARNING: no results/ga.json found, using random weights [1,1,1,1]")
        ga_weights = np.array([1.0, 1.0, 1.0, 1.0])
    t0 = time.time()
    ga_res = evaluate_ga(ga_weights, seeds)
    print(f"        done in {time.time()-t0:.1f}s, mean=${np.mean([r['total_capital_cost_usd'] for r in ga_res]):,.0f}")
    for seed, r in zip(seeds, ga_res):
        per_scenario_rows.append({
            "scenario_seed": seed,
            "method": "GA",
            **r,
        })
    ga_summary = summarize_method("GA", ga_res)

    # ---- 4. PPO ---------------------------------------------------------
    ppo_summary = None
    if os.path.exists(f"{args.ppo_model}.zip"):
        print(f"[4/4] PPO     : loading {args.ppo_model}.zip")
        t0 = time.time()
        ppo_res = evaluate_ppo(args.ppo_model, seeds)
        print(f"        done in {time.time()-t0:.1f}s, mean=${np.mean([r['total_capital_cost_usd'] for r in ppo_res]):,.0f}")
        for seed, r in zip(seeds, ppo_res):
            per_scenario_rows.append({
                "scenario_seed": seed,
                "method": "PPO",
                **r,
            })
        ppo_summary = summarize_method("PPO", ppo_res)
    else:
        print(f"[4/4] PPO     : SKIPPED (no model at {args.ppo_model}.zip)")

    # ---- Aggregate ------------------------------------------------------
    all_summaries = [fcfs_summary, milp_summary, ga_summary]
    if ppo_summary:
        all_summaries.append(ppo_summary)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Method':<12} {'Mean Capital ($)':>20} {'Mean Delay (h)':>16} {'Ships/ep':>10}")
    for s in all_summaries:
        print(f"{s['method']:<12} {s['mean_capital_usd']:>20,.0f} {s['mean_delay_h']:>16.1f} {s['mean_ships_served']:>10.1f}")

    # Compute savings
    fcfs_cap = fcfs_summary["mean_capital_usd"]
    print(f"\nSavings vs FCFS (${fcfs_cap:,.0f}):")
    for s in all_summaries:
        if s["method"] == "FCFS":
            continue
        pct = 100 * (fcfs_cap - s["mean_capital_usd"]) / fcfs_cap
        print(f"  {s['method']:<10}: {pct:+6.1f}%")

    # ---- Save outputs ---------------------------------------------------
    per_scenario_df = pd.DataFrame(per_scenario_rows)
    per_scenario_df.to_csv(f"{args.out_dir}/per_scenario.csv", index=False)

    summary_df = pd.DataFrame([{k: v for k, v in s.items() if k != "raw_per_episode"} for s in all_summaries])
    summary_df.to_csv(f"{args.out_dir}/summary.csv", index=False)

    # Savings table
    savings_rows = []
    for s in all_summaries:
        savings_rows.append({
            "method": s["method"],
            "mean_capital_usd": s["mean_capital_usd"],
            "mean_delay_h": s["mean_delay_h"],
            "savings_vs_fcfs_pct": 100 * (fcfs_cap - s["mean_capital_usd"]) / fcfs_cap,
            "savings_vs_milp_pct": 100 * (milp_summary["mean_capital_usd"] - s["mean_capital_usd"]) / milp_summary["mean_capital_usd"],
        })
    pd.DataFrame(savings_rows).to_csv(f"{args.out_dir}/savings_table.csv", index=False)

    # IAR savings (the thesis-table-friendly version)
    iar_rows = []
    for s in all_summaries:
        iar_rows.append({
            "Method": s["method"],
            "Mean Capital Cost (USD)": f"${s['mean_capital_usd']:,.0f}",
            "Mean Total Delay (h)": f"{s['mean_delay_h']:.1f}",
            "Mean Ships Served": f"{s['mean_ships_served']:.1f}",
            "Savings vs FCFS": f"{100*(fcfs_cap - s['mean_capital_usd'])/fcfs_cap:+.1f}%",
        })
    pd.DataFrame(iar_rows).to_csv(f"{args.out_dir}/iar_savings.csv", index=False)
    print(f"\nSaved IAR savings table to {args.out_dir}/iar_savings.csv")

    # Plots
    plot_cost_comparison(all_summaries, f"{args.out_dir}/cost_comparison.png")
    plot_savings_vs_fcfs(all_summaries, f"{args.out_dir}/savings_vs_fcfs.png")

    # Full JSON
    full = {
        "test_seeds": seeds,
        "summaries": all_summaries,
        "savings_vs_fcfs": {s["method"]: 100 * (fcfs_cap - s["mean_capital_usd"]) / fcfs_cap
                            for s in all_summaries},
    }
    with open(f"{args.out_dir}/summary.json", "w") as f:
        json.dump(full, f, indent=2, default=str)
    print(f"Saved summary to {args.out_dir}/summary.json")
    print("\nDONE.")


if __name__ == "__main__":
    main()
