"""Generate the fault-tolerance figure (Rev-2, Reviewer #8 C2)."""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGDIR = Path(__file__).resolve().parents[2] / "Paper" / "D_MAPPO_ABC_MAIN_Rev_2" / "Figures"

plt.rcParams.update({
    "font.size": 9,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "0.85",
    "grid.linewidth": 0.6,
})

traces = json.load(open(RESULTS / "fault_tolerance_traces.json"))
df = pd.read_csv(RESULTS / "fault_tolerance.csv")

FAIL_STEP = 1800
WINDOW = 60  # smoothing window (steps)


def smooth_mean(traces_list, key):
    arr = np.array([t[key] for t in traces_list])   # (seeds, steps)
    mean = arr.mean(axis=0)
    kernel = np.ones(WINDOW) / WINDOW
    return np.convolve(mean, kernel, mode="valid")


fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))

# (a) smoothed per-step reward traces, zoomed around the failure point
ax = axes[0]
ZOOM = (1200, 2800)
for level, style, color, label in [("0.0", "-", "0.6", "no failures"),
                                   ("0.3", "--", "0.0", "30% agent loss at t=1800")]:
    y = smooth_mean(traces[level], "reward_trace")
    x = np.arange(len(y)) + WINDOW // 2
    m = (x >= ZOOM[0]) & (x <= ZOOM[1])
    ax.plot(x[m], y[m], style, color=color, linewidth=1.4, label=label)
ax.axvline(FAIL_STEP, color="0.3", linewidth=0.9, linestyle=":", zorder=0)
ax.text(FAIL_STEP + 30, ax.get_ylim()[0] + 0.0005, "failure", fontsize=8,
        color="0.3")
ax.set_xlabel("Episode step")
ax.set_ylabel("Mean step reward (alive agents)")
ax.set_title("(a) Reward around failure point", fontsize=9)
ax.legend(frameon=False, fontsize=7.5, loc="upper right")

# (b) cumulative deadline miss rate traces
ax = axes[1]
for level, style, color, label in [("0.0", "-", "0.6", "no failures"),
                                   ("0.3", "--", "0.0", "30% agent loss")]:
    arr = np.array([t["miss_trace"] for t in traces[level]])
    mean = arr.mean(axis=0) * 100
    ax.plot(mean, style, color=color, linewidth=1.4, label=label)
ax.axvline(FAIL_STEP, color="0.3", linewidth=0.9, linestyle=":", zorder=0)
ax.set_xlabel("Episode step")
ax.set_ylabel("Cumulative deadline miss rate (%)")
ax.set_ylim(-0.005, 0.1)
ax.set_title("(b) Deadline miss rate", fontsize=9)
ax.legend(frameon=False, fontsize=7.5)

# (c) summary bars: task latency and reward vs loss level
ax = axes[2]
g = df.groupby("loss_fraction")
levels = sorted(df.loss_fraction.unique())
lat_mean = [g.get_group(l)["avg_latency_s"].mean() * 1000 for l in levels]
lat_std = [g.get_group(l)["avg_latency_s"].std() * 1000 for l in levels]
xs = np.arange(len(levels))
ax.bar(xs, lat_mean, yerr=lat_std, capsize=3, color="0.75",
       edgecolor="black", linewidth=0.8, hatch="//")
ax.set_xticks(xs, [f"{int(l*100)}%" for l in levels])
ax.set_xlabel("Agents lost at mid-episode")
ax.set_ylabel("Mean task latency (ms)")
ax.set_title("(c) Task latency vs. loss level", fontsize=9)

fig.tight_layout()
fig.savefig(FIGDIR / "fig_fault_tolerance.pdf", bbox_inches="tight")
print("saved", FIGDIR / "fig_fault_tolerance.pdf")

# print summary table numbers
s = df.groupby("loss_fraction").agg(
    reward_m=("reward", "mean"), reward_s=("reward", "std"),
    lat_m=("avg_latency_s", "mean"), lat_s=("avg_latency_s", "std"),
    dm_max=("deadline_miss_rate", "max"),
    cr_m=("completion_ratio", "mean")).round(4)
print(s.to_string())
