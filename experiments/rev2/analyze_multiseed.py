"""
Rev-2: statistical analysis of the multi-seed evaluation.

Produces:
  - results/multiseed_summary.csv  (per-method mean/std + Welch t vs ours)
  - Figures/fig_baseline_comparison_stats.pdf (error-bar comparison figure)
  - stdout: LaTeX-ready rows for Table 9 and the ablation table
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGDIR = Path(__file__).resolve().parents[2] / "Paper" / "D_MAPPO_ABC_MAIN_Rev_2" / "Figures"

# NOTE: "Pure MAPPO" and the ablation variants are intentionally excluded
# from the main comparison figure/table: the ablation study keeps its
# original training/evaluation protocol (Section 5.6), which this harness
# does not reproduce.
MAIN_ORDER = ["Round-Robin", "Greedy", "Random", "Load-Aware WRR",
              "IPPO", "QMIX-inspired", "CommNet-inspired", "MADDPG-inspired",
              "Zhao et al. (MARL-ABC)", "FADDEER (attention)",
              "Wang et al. (hybrid actions)",
              "D-MAPPO-ABC"]
ABLATION_ORDER = ["D-MAPPO-ABC", "Pure MAPPO", "w/o Scout", "w/o Onlooker",
                  "Single Policy"]


def cohens_d(a, b):
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1))
                     / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else np.inf


def main():
    df = pd.read_csv(RESULTS / "multiseed_episodes.csv")
    hyb_path = RESULTS / "hybrid_baselines.csv"
    if hyb_path.exists():
        hyb = pd.read_csv(hyb_path)
        df = pd.concat([df, hyb], ignore_index=True)

    ours_seed_means = (df[df.method == "D-MAPPO-ABC"]
                       .groupby("seed").reward.mean().values)

    rows = []
    for method, g in df.groupby("method"):
        seed_means = g.groupby("seed").reward.mean().values
        rec = {
            "method": method,
            "n_episodes": len(g),
            "reward_mean": g.reward.mean(),
            "reward_std": g.reward.std(),
            "latency_ms_mean": g.avg_latency_s.mean() * 1000,
            "latency_ms_std": g.avg_latency_s.std() * 1000,
            "energy_mean": g.energy_seu.mean(),
            "energy_std": g.energy_seu.std(),
            "dm_pct_mean": g.deadline_miss_rate.mean() * 100,
            "dm_pct_max": g.deadline_miss_rate.max() * 100,
            "decision_ms": g.decision_ms.mean() if "decision_ms" in g and g.decision_ms.notna().any() else np.nan,
        }
        if method != "D-MAPPO-ABC":
            t, p = stats.ttest_ind(ours_seed_means, seed_means,
                                   equal_var=False)
            rec["welch_t"] = t
            rec["p_value"] = p
            rec["cohens_d"] = cohens_d(ours_seed_means, seed_means)
        rows.append(rec)

    summary = pd.DataFrame(rows).set_index("method")
    summary.to_csv(RESULTS / "multiseed_summary.csv")
    pd.set_option("display.width", 250)
    print(summary.round(3).to_string())

    # ---- LaTeX rows for Table 9 ----
    print("\n==== LaTeX rows (Table 9) ====")
    for m in MAIN_ORDER:
        if m not in summary.index:
            continue
        r = summary.loc[m]
        stat = ""
        if m != "D-MAPPO-ABC":
            p = r["p_value"]
            stat = "$p<0.001$" if p < 0.001 else f"$p={p:.3f}$"
        print(f"{m} & ${r.reward_mean:.1f} \\pm {r.reward_std:.1f}$ & "
              f"${r.latency_ms_mean:.1f} \\pm {r.latency_ms_std:.1f}$ & "
              f"${r.energy_mean:,.0f} \\pm {r.energy_std:.0f}$ & "
              f"${r.dm_pct_mean:.2f}$ & {stat} \\\\")

    # ---- LaTeX rows for the ablation table ----
    print("\n==== LaTeX rows (ablation) ====")
    for m in ABLATION_ORDER:
        if m not in summary.index:
            continue
        r = summary.loc[m]
        print(f"{m} & ${r.reward_mean:.2f} \\pm {r.reward_std:.2f}$ & "
              f"${r.energy_mean:,.0f} \\pm {r.energy_std:.0f}$ & "
              f"${r.dm_pct_mean:.2f}$ \\\\")

    # ---- Error-bar comparison figure ----
    plt.rcParams.update({
        "font.size": 9, "font.family": "sans-serif",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "0.85", "grid.linewidth": 0.6,
    })
    methods = [m for m in MAIN_ORDER if m in summary.index]
    short = {"Zhao et al. (MARL-ABC)": "Zhao (MARL-ABC)",
             "FADDEER (attention)": "FADDEER",
             "Wang et al. (hybrid actions)": "Wang (hybrid)"}
    labels = [short.get(m, m) for m in methods]
    colors = ["0.35" if m == "D-MAPPO-ABC" else "0.75" for m in methods]
    hatches = ["\\\\" if m == "D-MAPPO-ABC" else "" for m in methods]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))

    ax = axes[0]
    means = summary.loc[methods, "reward_mean"]
    stds = summary.loc[methods, "reward_std"]
    bars = ax.barh(range(len(methods)), means, xerr=stds, capsize=2.5,
                   color=colors, edgecolor="black", linewidth=0.7)
    for b, h in zip(bars, hatches):
        b.set_hatch(h)
    ax.set_yticks(range(len(methods)), labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xscale("symlog", linthresh=50)
    ax.set_xlabel("Episode reward (symlog)")
    ax.set_title("(a) Reward (mean ± std, 100 episodes)", fontsize=9)

    ax = axes[1]
    means = summary.loc[methods, "latency_ms_mean"]
    stds = summary.loc[methods, "latency_ms_std"]
    bars = ax.barh(range(len(methods)), means, xerr=stds, capsize=2.5,
                   color=colors, edgecolor="black", linewidth=0.7)
    for b, h in zip(bars, hatches):
        b.set_hatch(h)
    ax.set_yticks(range(len(methods)), ["" for _ in methods])
    ax.invert_yaxis()
    ax.set_xlabel("Mean task latency (ms)")
    ax.set_title("(b) End-to-end task latency", fontsize=9)

    ax = axes[2]
    means = summary.loc[methods, "dm_pct_mean"]
    bars = ax.barh(range(len(methods)), means, color=colors,
                   edgecolor="black", linewidth=0.7)
    for b, h in zip(bars, hatches):
        b.set_hatch(h)
    ax.set_yticks(range(len(methods)), ["" for _ in methods])
    ax.invert_yaxis()
    ax.set_xlabel("Deadline miss rate (%)")
    ax.set_title("(c) Deadline misses", fontsize=9)

    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_baseline_comparison_stats.pdf",
                bbox_inches="tight")
    print("\nsaved", FIGDIR / "fig_baseline_comparison_stats.pdf")


if __name__ == "__main__":
    main()
