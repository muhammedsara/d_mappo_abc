"""Regenerate the deployment performance figure from the 10-seed data."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
FIGDIR = Path(__file__).resolve().parents[2] / "Paper" / "D_MAPPO_ABC_MAIN_Rev_2" / "Figures"

plt.rcParams.update({
    "font.size": 9, "font.family": "sans-serif",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "0.85", "grid.linewidth": 0.6,
})

df = pd.read_csv(HERE / "results" / "multiseed_episodes.csv")
d = df[df.method == "D-MAPPO-ABC"].reset_index(drop=True)
x = range(1, len(d) + 1)

fig, axes = plt.subplots(2, 2, figsize=(9, 5.4))

ax = axes[0, 0]
ax.plot(x, d.reward, color="0.2", linewidth=0.9)
m, s = d.reward.mean(), d.reward.std()
ax.axhline(m, color="0.45", linestyle="--", linewidth=1)
ax.fill_between(x, m - s, m + s, color="0.8", alpha=0.5)
ax.set_ylabel("Episode reward")
ax.set_title(f"(a) Reward: {m:.2f} ± {s:.2f}", fontsize=9)

ax = axes[0, 1]
lat = d.avg_latency_s * 1000
ax.plot(x, lat, color="0.2", linewidth=0.9)
m, s = lat.mean(), lat.std()
ax.axhline(m, color="0.45", linestyle="--", linewidth=1)
ax.fill_between(x, m - s, m + s, color="0.8", alpha=0.5)
ax.set_ylabel("Mean task latency (ms)")
ax.set_title(f"(b) Task latency: {m:.1f} ± {s:.1f} ms", fontsize=9)

ax = axes[1, 0]
ax.plot(x, d.energy_seu, color="0.2", linewidth=0.9)
m, s = d.energy_seu.mean(), d.energy_seu.std()
ax.axhline(m, color="0.45", linestyle="--", linewidth=1)
ax.set_ylabel("Energy (SEU)")
ax.set_xlabel("Episode (10 seeds × 10 episodes)")
ax.set_title(f"(c) Energy: {m:,.0f} ± {s:.1f} SEU", fontsize=9)

ax = axes[1, 1]
ax.plot(x, d.deadline_miss_rate * 100, color="0.2", linewidth=0.9)
ax.set_ylim(-0.02, 0.5)
ax.set_ylabel("Deadline miss rate (%)")
ax.set_xlabel("Episode (10 seeds × 10 episodes)")
ax.set_title("(d) Deadline misses: 0.00% in all episodes", fontsize=9)

fig.tight_layout()
fig.savefig(FIGDIR / "fig_deployment_performance.pdf", bbox_inches="tight")
print("saved", FIGDIR / "fig_deployment_performance.pdf")
