"""
Rev-2 Experiment 3 — Communication overhead and per-node memory
versus agent count (Reviewer #8, C3).

For N in {5, 10, 20, 50}: measures per-step message counts (state
broadcasts + executed offload transfers) empirically over 5 episodes,
and reports per-node memory: exact policy parameter sizes plus the
resident-set size of a single-agent inference process.
"""

import csv
import resource
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from experiments.rev2.harness import (  # noqa: E402
    PROJECT_ROOT, RESULTS_DIR, build_env, make_controller, run_episode,
)

AGENT_COUNTS = [5, 10, 20, 50]
EPISODES = 5

OUT_CSV = RESULTS_DIR / "scalability_overhead.csv"
OUT_MEM = RESULTS_DIR / "memory_per_node.csv"

_RSS_SNIPPET = r"""
import resource, sys
sys.path.insert(0, {root!r})
from experiments.rev2.harness import make_controller
import numpy as np, torch
c = make_controller("D-MAPPO-ABC")
obs = {{f"agent_{{i}}": np.random.rand(15).astype(np.float32) for i in range(1)}}
class E: agents = ["agent_0"]
for _ in range(100):
    c.get_actions(E(), obs)
print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
"""


def measure_policy_memory():
    """Exact per-policy parameter memory (float32) in KB."""
    ctrl = make_controller("D-MAPPO-ABC")
    rows = []
    for name, model in ctrl.policies.items():
        n_params = sum(p.numel() for p in model.parameters())
        rows.append({
            "policy": name,
            "n_params": n_params,
            "param_kb": n_params * 4 / 1024,
        })
    return rows


def measure_process_rss():
    """RSS (MB) of a single-agent inference process (policy + torch)."""
    code = _RSS_SNIPPET.format(root=str(PROJECT_ROOT))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, cwd=PROJECT_ROOT)
    rss_kb = int(out.stdout.strip().splitlines()[-1])
    return rss_kb / 1024.0


def main():
    t0 = time.time()

    mem_rows = measure_policy_memory()
    rss_mb = measure_process_rss()
    with open(OUT_MEM, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["policy", "n_params",
                                               "param_kb"])
        writer.writeheader()
        writer.writerows(mem_rows)
        f.write(f"# single-agent inference process RSS: {rss_mb:.1f} MB\n")
    print(f"policy memory: {mem_rows}; process RSS {rss_mb:.1f} MB")

    rows = []
    for n in AGENT_COUNTS:
        for seed in range(EPISODES):
            env = build_env(num_agents=n, task_seed=3000 + seed)
            controller = make_controller("D-MAPPO-ABC", num_agents=n)
            res = run_episode(env, controller, seed=seed)
            # Message accounting: every active agent broadcasts its local
            # state summary once per step (N messages) + executed
            # offload transfers.
            state_msgs = n
            total_msgs = state_msgs + res["offload_msgs_per_step"]
            rows.append({
                "num_agents": n,
                "seed": seed,
                "reward": res["reward"],
                "decision_ms": res["decision_ms"],
                "offload_msgs_per_step": res["offload_msgs_per_step"],
                "state_msgs_per_step": state_msgs,
                "total_msgs_per_step": total_msgs,
                "centralized_msgs_per_step": 2 * n + n * (n - 1),
                "deadline_miss_rate": res["deadline_miss_rate"],
            })
            print(f"[{time.time()-t0:6.0f}s] N={n} seed={seed} "
                  f"msgs/step={total_msgs:.2f} reward={res['reward']:.1f}",
                  flush=True)

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"DONE in {time.time()-t0:.0f}s -> {OUT_CSV}")


if __name__ == "__main__":
    main()
