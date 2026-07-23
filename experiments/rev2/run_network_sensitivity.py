"""
Rev-2 Experiment 4 — Network condition sensitivity (Reviewer #8, C6).

Two sweeps with the trained (frozen) D-MAPPO-ABC policies:
  1. Packet loss p in {1%, 3%, 5%, 10%}: the simulator converts loss
     into retransmission latency, so inter-device latencies scale by
     1/(1-p) and the observed packet-loss feature changes.
  2. Mean inter-device latency in {30, 60, 90, 120} ms (congestion /
     reduced-bandwidth proxy) at the nominal 1% loss.

10 episodes (seeds 0-9) per configuration.
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.rev2.harness import (  # noqa: E402
    RESULTS_DIR, build_env, make_controller, run_episode,
)

PACKET_LOSS_LEVELS = [0.01, 0.03, 0.05, 0.10]
LATENCY_SCALES = [1.0, 2.0, 3.0, 4.0]  # 30, 60, 90, 120 ms mean
EPISODES = 10

OUT_CSV = RESULTS_DIR / "network_sensitivity.csv"


def patch_cloud_tracking(env, stats):
    """Track sampled cloud round-trip latencies vs. task deadlines.

    The simulator optimistically treats every cloud offload as completing
    within its deadline; this wrapper records how often the sampled cloud
    round-trip actually exceeds the task deadline ("at-risk offloads").
    """
    def patched(task):
        cloud_latency = max(
            np.random.normal(env.network_config["cloud"]["latency_ms"],
                             env.network_config["cloud"]["latency_std_ms"]),
            1.0)
        stats["cloud_total"] += 1
        if cloud_latency > task.deadline_ms:
            stats["cloud_over_deadline"] += 1
        latency_ratio = max(cloud_latency / task.deadline_ms, 1e-6)
        latency_reward = -np.log(np.clip(latency_ratio, 1e-6, 1e6))
        energy_reward = -np.clip(
            env.network_config["cloud"]["cost_per_task"], -10, 10)
        reward = (env.reward_weights["alpha"] * latency_reward
                  + env.reward_weights["beta"] * energy_reward)
        return np.clip(reward * 0.3, -5, 5)

    env._compute_cloud_offload_reward = patched


def main():
    t0 = time.time()
    rows = []

    def run_config(sweep, value, env_kwargs, seed):
        env = build_env(task_seed=4000 + seed, **env_kwargs)
        stats = {"cloud_total": 0, "cloud_over_deadline": 0}
        patch_cloud_tracking(env, stats)
        controller = make_controller("D-MAPPO-ABC")
        res = run_episode(env, controller, seed=seed)
        at_risk = (stats["cloud_over_deadline"] / stats["cloud_total"]
                   if stats["cloud_total"] else 0.0)
        rows.append({
            "sweep": sweep, "value": value,
            "seed": seed, "reward": res["reward"],
            "avg_latency_s": res["avg_latency_s"],
            "energy_seu": res["energy_seu"],
            "deadline_miss_rate": res["deadline_miss_rate"],
            "cloud_offloads": stats["cloud_total"],
            "cloud_over_deadline_rate": at_risk,
            "offload_msgs_per_step": res["offload_msgs_per_step"],
        })
        print(f"[{time.time()-t0:6.0f}s] {sweep}={value} seed={seed} "
              f"reward={res['reward']:.2f} at_risk={at_risk:.4%}", flush=True)

    for p in PACKET_LOSS_LEVELS:
        for seed in range(EPISODES):
            run_config("packet_loss", p, {"packet_loss": p}, seed)

    for s in LATENCY_SCALES:
        if s == 1.0:
            continue  # 1x covered by the 1% packet-loss baseline rows
        for seed in range(EPISODES):
            run_config("latency_scale", s, {"latency_scale": s}, seed)

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"DONE in {time.time()-t0:.0f}s -> {OUT_CSV}")


if __name__ == "__main__":
    main()
