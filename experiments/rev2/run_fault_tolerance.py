"""
Rev-2 Experiment 2 — Fault-tolerance analysis (Reviewer #8, C2).

At step 1800 (episode midpoint) a fraction of agents goes offline:
they stop making scheduling decisions and stop accepting offloaded
tasks. Fractions: 0% (control), 10%, 20%, 30%. 10 episodes per level
(seeds 0-9). Per-step reward and deadline-miss traces are stored for
the 0% and 30% cases to produce the time-series figure.
"""

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.rev2.harness import (  # noqa: E402
    RESULTS_DIR, build_env, make_controller, run_episode,
)

FAIL_STEP = 1800
LEVELS = [0.0, 0.1, 0.2, 0.3]
NUM_AGENTS = 10
EPISODES = 10  # one per seed

OUT_CSV = RESULTS_DIR / "fault_tolerance.csv"
OUT_TRACES = RESULTS_DIR / "fault_tolerance_traces.json"


def main():
    t0 = time.time()
    rows = []
    traces = {}

    for level in LEVELS:
        n_dead = int(round(level * NUM_AGENTS))
        level_traces = []
        for seed in range(EPISODES):
            rng = np.random.RandomState(seed)
            dead_agents = sorted(
                rng.choice(NUM_AGENTS, size=n_dead, replace=False).tolist()
            ) if n_dead else []
            env = build_env(task_seed=2000 + seed)
            controller = make_controller("D-MAPPO-ABC")
            dropout = ({"step": FAIL_STEP, "agents": dead_agents}
                       if n_dead else None)
            res = run_episode(env, controller, seed=seed,
                              dropout=dropout, collect_traces=True)
            rows.append({
                "loss_fraction": level,
                "seed": seed,
                "dead_agents": ";".join(map(str, dead_agents)),
                "reward": res["reward"],
                "avg_latency_s": res["avg_latency_s"],
                "energy_seu": res["energy_seu"],
                "deadline_miss_rate": res["deadline_miss_rate"],
                "completion_ratio": res["completion_ratio"],
                "completed_tasks": res["completed_tasks"],
                "total_tasks": res["total_tasks"],
            })
            if level in (0.0, 0.3):
                level_traces.append({
                    "seed": seed,
                    "reward_trace": res["reward_trace"],
                    "miss_trace": res["miss_trace"],
                })
            print(f"[{time.time()-t0:6.0f}s] loss={level:.0%} seed={seed} "
                  f"reward={res['reward']:.2f} "
                  f"miss={res['deadline_miss_rate']:.4%}", flush=True)
        if level in (0.0, 0.3):
            traces[str(level)] = level_traces

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(OUT_TRACES, "w") as f:
        json.dump(traces, f)
    print(f"DONE in {time.time()-t0:.0f}s -> {OUT_CSV}")


if __name__ == "__main__":
    main()
