"""
Rev-2 Experiment 5 — Load scenarios with multi-seed statistics
(supports the mean±std revision of Table 12 / load-scenario results).

Scenarios (matching Section 4.4.3 of the manuscript):
  low     : 30 tasks/min constant
  normal  : 50 tasks/min constant (baseline)
  burst   : 150 tasks/min for 5-min windows, 10-min recovery at nominal
  ramp    : linear 30 -> 80 tasks/min over the episode
  spikes  : 200 tasks/min with 5% probability per step, else nominal

10 episodes per scenario (seeds 0-9).
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

EPISODES = 10
OUT_CSV = RESULTS_DIR / "load_scenarios.csv"


class LoadSchedule:
    """Modulates the task generator arrival rate per step."""

    def __init__(self, env, scenario, rng):
        self.gen = env.task_generator
        self.base = self.gen.total_arrival_rate  # 50/min
        self.scenario = scenario
        self.rng = rng

    def rate_at(self, step):
        if self.scenario == "low":
            return 30.0
        if self.scenario == "normal":
            return self.base
        if self.scenario == "burst":
            # 15-minute cycle: 5 min at 150/min, 10 min at nominal
            return 150.0 if (step % 900) < 300 else self.base
        if self.scenario == "ramp":
            return 30.0 + (80.0 - 30.0) * step / 3600.0
        if self.scenario == "spikes":
            return 200.0 if self.rng.rand() < 0.05 else self.base
        raise ValueError(self.scenario)

    def apply(self, step):
        self.gen.total_arrival_rate = self.rate_at(step)


def run_scenario_episode(scenario, seed):
    env = build_env(task_seed=5000 + seed)
    controller = make_controller("D-MAPPO-ABC")
    rng = np.random.RandomState(9000 + seed)
    schedule = LoadSchedule(env, scenario, rng)

    obs, _ = env.reset(seed=seed)
    controller.reset()
    total_reward = 0.0
    cloud_actions = 0
    total_actions = 0
    step = 0
    done = False
    while not done and step < 3600:
        schedule.apply(step)
        actions = controller.get_actions(env, obs)
        cloud_actions += sum(1 for a in actions.values() if a == 10)
        total_actions += len(actions)
        obs, rewards, terminations, truncations, infos = env.step(actions)
        total_reward += float(np.mean(list(rewards.values())))
        done = any(t for k, t in truncations.items() if k != "__all__")
        step += 1

    info = infos[env.agents[0]]
    return {
        "scenario": scenario,
        "seed": seed,
        "reward": total_reward,
        "avg_latency_s": info.get("avg_latency", 0.0),
        "energy_seu": info.get("total_energy", 0.0),
        "deadline_miss_rate": info.get("deadline_miss_rate", 0.0),
        "completed_tasks": info.get("completed_tasks", 0),
        "total_tasks": info.get("total_tasks", 0),
        "cloud_action_share": cloud_actions / max(total_actions, 1),
    }


def main():
    t0 = time.time()
    rows = []
    for scenario in ["low", "normal", "burst", "ramp", "spikes"]:
        for seed in range(EPISODES):
            res = run_scenario_episode(scenario, seed)
            rows.append(res)
            print(f"[{time.time()-t0:6.0f}s] {scenario} seed={seed} "
                  f"reward={res['reward']:.2f} "
                  f"miss={res['deadline_miss_rate']:.4%} "
                  f"cloud={res['cloud_action_share']:.1%}", flush=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"DONE in {time.time()-t0:.0f}s -> {OUT_CSV}")


if __name__ == "__main__":
    main()
