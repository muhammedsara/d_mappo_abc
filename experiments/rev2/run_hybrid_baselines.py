"""
Rev-2: evaluate the three hybrid-baseline adaptations under the same
10-seed protocol as the main comparison. Learning-based baselines
(Zhao, Wang) receive a 20-episode online pretraining phase per seed
before the 10 evaluation episodes (their updates stay on during
evaluation, matching their online-learning formulations).
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.rev2.harness import RESULTS_DIR, build_env  # noqa: E402
from experiments.rev2.hybrid_baselines import (  # noqa: E402
    FADDEERAttention, WangHybridAction, ZhaoCentralMARLABC,
)

SEEDS = list(range(10))
PRETRAIN_EPISODES = 20
EVAL_EPISODES = 10

OUT_CSV = RESULTS_DIR / "hybrid_baselines.csv"


def run_episode(env, ctrl, seed, learn=True):
    obs, _ = env.reset(seed=seed)
    total = 0.0
    step = 0
    done = False
    while not done and step < 3600:
        actions = ctrl.get_actions(obs)
        obs, rewards, terms, truncs, infos = env.step(actions)
        if learn:
            ctrl.update(obs, rewards)
        total += float(np.mean(list(rewards.values())))
        done = any(t for k, t in truncs.items() if k != "__all__")
        step += 1
    ctrl.end_episode()
    info = infos[env.agents[0]]
    return {
        "reward": total,
        "avg_latency_s": info.get("avg_latency", 0.0),
        "energy_seu": info.get("total_energy", 0.0),
        "deadline_miss_rate": info.get("deadline_miss_rate", 0.0),
        "completed_tasks": info.get("completed_tasks", 0),
        "total_tasks": info.get("total_tasks", 0),
    }


def main():
    t0 = time.time()
    rows = []
    for name, cls, kwargs, learn in [
        ("Zhao et al. (MARL-ABC)", ZhaoCentralMARLABC, {}, True),
        ("FADDEER (attention)", FADDEERAttention, {}, False),
        ("Wang et al. (hybrid actions)", WangHybridAction, {}, True),
    ]:
        for seed in SEEDS:
            np.random.seed(seed)
            env = build_env(task_seed=6000 + seed)
            ctrl = cls(env, **kwargs)
            if learn:
                for ep in range(PRETRAIN_EPISODES):
                    run_episode(env, ctrl, seed=seed * 777 + ep, learn=True)
            for ep in range(EVAL_EPISODES):
                res = run_episode(env, ctrl, seed=seed * 1000 + ep,
                                  learn=learn)
                res.update(method=name, seed=seed, episode=ep)
                rows.append(res)
            print(f"[{time.time()-t0:6.0f}s] {name} seed={seed} "
                  f"last reward={res['reward']:.2f} "
                  f"miss={res['deadline_miss_rate']:.4%}", flush=True)

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"DONE in {time.time()-t0:.0f}s -> {OUT_CSV}")


if __name__ == "__main__":
    main()
