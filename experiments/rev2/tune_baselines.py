"""
Rev-2: small grid searches for tunable baselines (documented in
Appendix B). Selection criterion: mean reward over 5 pilot episodes
(seeds 100-104), pretraining learners for 10 episodes first.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.rev2.harness import RESULTS_DIR, build_env  # noqa: E402
from experiments.rev2.hybrid_baselines import (  # noqa: E402
    FADDEERAttention, WangHybridAction, ZhaoCentralMARLABC,
)
from experiments.rev2.run_hybrid_baselines import run_episode  # noqa: E402

PILOT_SEEDS = [100, 101, 102, 103, 104]


def pilot_score(make_ctrl, learn, pretrain=10):
    scores = []
    for seed in PILOT_SEEDS:
        np.random.seed(seed)
        env = build_env(task_seed=7000 + seed)
        ctrl = make_ctrl(env)
        if learn:
            for ep in range(pretrain):
                run_episode(env, ctrl, seed=seed * 31 + ep, learn=True)
        res = run_episode(env, ctrl, seed=seed, learn=learn)
        scores.append(res["reward"])
    return float(np.mean(scores))


def greedy_score(low, high):
    scores = []
    for seed in PILOT_SEEDS:
        env = build_env(task_seed=7000 + seed)
        obs, _ = env.reset(seed=seed)
        total, step, done = 0.0, 0, False
        while not done and step < 3600:
            actions = {}
            for a in env.agents:
                cpu = obs[a][0]
                actions[a] = 0 if cpu < low else (1 if cpu < high else 10)
            obs, r, te, tr, infos = env.step(actions)
            total += float(np.mean(list(r.values())))
            done = any(t for k, t in tr.items() if k != "__all__")
            step += 1
        scores.append(total)
    return float(np.mean(scores))


def main():
    t0 = time.time()
    results = {}

    grid = {}
    for lr in [0.005, 0.01, 0.05]:
        s = pilot_score(lambda env, lr=lr: ZhaoCentralMARLABC(env, lr=lr),
                        learn=True)
        grid[f"lr={lr}"] = s
        print(f"[{time.time()-t0:5.0f}s] Zhao lr={lr}: {s:.2f}", flush=True)
    results["zhao"] = grid

    grid = {}
    for temp in [0.3, 0.5, 1.0]:
        for bias in [0.4, 0.6, 0.8]:
            s = pilot_score(lambda env, t=temp, b=bias:
                            FADDEERAttention(env, temperature=t, local_bias=b),
                            learn=False)
            grid[f"temp={temp},bias={bias}"] = s
            print(f"[{time.time()-t0:5.0f}s] FADDEER temp={temp} "
                  f"bias={bias}: {s:.2f}", flush=True)
    results["faddeer"] = grid

    grid = {}
    for lr in [0.005, 0.01, 0.05]:
        s = pilot_score(lambda env, lr=lr: WangHybridAction(env, lr=lr),
                        learn=True)
        grid[f"lr={lr}"] = s
        print(f"[{time.time()-t0:5.0f}s] Wang lr={lr}: {s:.2f}", flush=True)
    results["wang"] = grid

    grid = {}
    for low in [0.4, 0.5, 0.6]:
        for high in [0.7, 0.8, 0.9]:
            s = greedy_score(low, high)
            grid[f"low={low},high={high}"] = s
            print(f"[{time.time()-t0:5.0f}s] Greedy {low}/{high}: {s:.2f}",
                  flush=True)
    results["greedy"] = grid

    with open(RESULTS_DIR / "baseline_tuning.json", "w") as f:
        json.dump(results, f, indent=2)
    for method, grid in results.items():
        best = max(grid, key=grid.get)
        print(f"BEST {method}: {best} ({grid[best]:.2f})")


if __name__ == "__main__":
    main()
