"""
Rev-2 Experiment 1 — Multi-seed statistical evaluation (Reviewer #8, C1).

Evaluates D-MAPPO-ABC, all baselines, and the ablation variants across
10 independent random seeds. Per-episode records are written to
results/multiseed_episodes.csv for downstream statistics (mean ± std,
Welch t-tests, effect sizes).
"""

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.rev2.harness import (  # noqa: E402
    RESULTS_DIR, build_env, make_controller, run_episode,
)

SEEDS = list(range(10))
EPISODES_PER_SEED_MAIN = 10
EPISODES_PER_SEED_ABLATION = 5

MAIN_METHODS = [
    "D-MAPPO-ABC",
    "Pure MAPPO",
    "Round-Robin",
    "Greedy",
    "Random",
    "Load-Aware WRR",
    "IPPO",
    "QMIX-inspired",
    "CommNet-inspired",
    "MADDPG-inspired",
]
ABLATION_METHODS = ["w/o Scout", "w/o Onlooker", "Single Policy"]

OUT_PATH = RESULTS_DIR / "multiseed_episodes.csv"
FIELDS = ["method", "seed", "episode", "reward", "avg_latency_s",
          "energy_seu", "deadline_miss_rate", "completed_tasks",
          "total_tasks", "rejected_tasks", "completion_ratio",
          "decision_ms", "offload_msgs_per_step", "cloud_msgs_per_step"]


def main():
    t_start = time.time()
    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()

        for method in MAIN_METHODS + ABLATION_METHODS:
            n_eps = (EPISODES_PER_SEED_ABLATION if method in ABLATION_METHODS
                     else EPISODES_PER_SEED_MAIN)
            for seed in SEEDS:
                env = build_env(task_seed=1000 + seed)
                controller = make_controller(method)
                for ep in range(n_eps):
                    res = run_episode(env, controller,
                                      seed=seed * 1_000 + ep)
                    row = {k: res[k] for k in FIELDS
                           if k in res}
                    row.update(method=method, seed=seed, episode=ep)
                    writer.writerow(row)
                    f.flush()
                print(f"[{time.time()-t_start:7.0f}s] {method} seed={seed} "
                      f"done (last reward {res['reward']:.2f})", flush=True)

    print(f"DONE in {time.time()-t_start:.0f}s -> {OUT_PATH}")


if __name__ == "__main__":
    main()
