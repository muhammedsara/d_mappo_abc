"""
Rev-2 experiments: lightweight policy runner.

Loads the exported RLlib policy state dicts (deployment/policies/*.pth)
into plain PyTorch MLPs and runs deployment-style evaluation episodes
without Ray. Matches RLlib FullyConnectedNetwork with
fcnet_activation=relu and separate value branch (value branch unused
for greedy action selection).
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from envs.edge_scheduling_env import EdgeSchedulingEnv  # noqa: E402

POLICY_DIR = PROJECT_ROOT / "deployment" / "policies"
CHECKPOINT_POLICY_DIR = PROJECT_ROOT / "logs" / "checkpoints" / "policies"

ARCHS = {
    "employed_policy": [128, 128],
    "onlooker_policy": [256, 128],
    "scout_policy": [64, 64],
}


class _FakeVersion:
    """Shim for packaging.version.Version objects pickled by older RLlib."""

    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)


class _ShimUnpickler(__import__("pickle").Unpickler):
    def find_class(self, module, name):
        if module.startswith("packaging.version") and name == "Version":
            return _FakeVersion
        return super().find_class(module, name)


def load_checkpoint_weights(name: str) -> dict:
    """Load policy weights from the RLlib checkpoint (policy_state.pkl)."""
    path = CHECKPOINT_POLICY_DIR / name / "policy_state.pkl"
    with open(path, "rb") as f:
        state = _ShimUnpickler(f).load()
    return state["weights"]


class PolicyMLP(nn.Module):
    """Replicates RLlib FullyConnectedNetwork policy branch."""

    def __init__(self, obs_dim: int, hiddens, num_actions: int):
        super().__init__()
        layers = []
        last = obs_dim
        for h in hiddens:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        self.hidden = nn.Sequential(*layers)
        self.logits = nn.Linear(last, num_actions)

    def forward(self, x):
        return self.logits(self.hidden(x))


def load_policy(name: str, obs_dim: int = 15, num_actions: int = 12,
                source: str = "checkpoint") -> PolicyMLP:
    if source == "checkpoint":
        w = load_checkpoint_weights(name)
        sd = {k: torch.as_tensor(np.asarray(v)) for k, v in w.items()}
    else:
        sd = torch.load(POLICY_DIR / f"{name}.pth", map_location="cpu",
                        weights_only=False)
    model = PolicyMLP(obs_dim, ARCHS[name], num_actions)
    new_sd = {}
    for i in range(len(ARCHS[name])):
        new_sd[f"hidden.{2*i}.weight"] = sd[f"_hidden_layers.{i}._model.0.weight"]
        new_sd[f"hidden.{2*i}.bias"] = sd[f"_hidden_layers.{i}._model.0.bias"]
    new_sd["logits.weight"] = sd["_logits._model.0.weight"]
    new_sd["logits.bias"] = sd["_logits._model.0.bias"]
    model.load_state_dict(new_sd)
    model.eval()
    return model


class DMappoAbcController:
    """Role-based multi-policy controller (deterministic, exploitative)."""

    def __init__(self, num_agents: int = 10):
        with open(POLICY_DIR / "policy_mapping.yaml") as f:
            mapping = yaml.safe_load(f)
        self.policies = {name: load_policy(name) for name in ARCHS}
        self.num_agents = num_agents
        # Extend the 4-4-2 role ratio proportionally for other agent counts.
        self.role_of = {}
        for i in range(num_agents):
            if num_agents == 10:
                if i in mapping["employed_agents"]:
                    self.role_of[i] = "employed_policy"
                elif i in mapping["onlooker_agents"]:
                    self.role_of[i] = "onlooker_policy"
                else:
                    self.role_of[i] = "scout_policy"
            else:
                frac = i / num_agents
                if frac < 0.4:
                    self.role_of[i] = "employed_policy"
                elif frac < 0.8:
                    self.role_of[i] = "onlooker_policy"
                else:
                    self.role_of[i] = "scout_policy"

    @torch.no_grad()
    def act(self, agent_idx: int, obs: np.ndarray) -> int:
        model = self.policies[self.role_of[agent_idx]]
        logits = model(torch.from_numpy(obs).float().unsqueeze(0))
        return int(torch.argmax(logits, dim=1).item())

    @torch.no_grad()
    def act_all(self, obs_dict: dict) -> dict:
        actions = {}
        for agent_name, obs in obs_dict.items():
            idx = int(agent_name.split("_")[1])
            actions[agent_name] = self.act(idx, obs)
        return actions


def run_episode(env: EdgeSchedulingEnv, controller: DMappoAbcController,
                seed: int | None = None, max_steps: int = 3600,
                dropout: dict | None = None):
    """Run one deployment episode.

    dropout: optional {"step": int, "agents": [indices]} — from `step`
    onward the listed agents stop acting and stop receiving offloads
    (fault-tolerance experiment).
    """
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    decision_times = []
    step = 0
    done = False
    dead: set[int] = set()
    reward_trace = []
    miss_trace = []

    while not done and step < max_steps:
        if dropout and step == dropout["step"]:
            dead = set(dropout["agents"])
            # Offline devices: no offload target, queue frozen.
            for d in dead:
                env.adjacency[:, d] = False
                env.adjacency[d, :] = False

        t0 = time.perf_counter()
        actions = {}
        for agent_name, ob in obs.items():
            idx = int(agent_name.split("_")[1])
            if idx in dead:
                continue
            actions[agent_name] = controller.act(idx, ob)
        decision_times.append((time.perf_counter() - t0) / max(len(actions), 1))

        obs, rewards, terminations, truncations, infos = env.step(actions)
        alive_rewards = [r for a, r in rewards.items()
                         if int(a.split("_")[1]) not in dead]
        step_r = float(np.mean(alive_rewards)) if alive_rewards else 0.0
        total_reward += step_r
        reward_trace.append(step_r)

        done = any(t for k, t in truncations.items() if k != "__all__") or \
               any(t for k, t in terminations.items() if k != "__all__")
        step += 1
        info = infos[env.agents[0]]
        miss_trace.append(info.get("deadline_miss_rate", 0.0))

    info = infos[env.agents[0]]
    return {
        "reward": total_reward,
        "avg_latency_s": info.get("avg_latency", 0.0),
        "total_energy_wh": info.get("total_energy", 0.0),
        "deadline_miss_rate": info.get("deadline_miss_rate", 0.0),
        "completed_tasks": info.get("completed_tasks", 0),
        "rejected_tasks": info.get("rejected_tasks", 0),
        "total_tasks": info.get("total_tasks", 0),
        "steps": step,
        "mean_decision_time_ms": 1000.0 * float(np.mean(decision_times)),
        "reward_trace": reward_trace,
        "miss_trace": miss_trace,
    }


if __name__ == "__main__":
    t0 = time.time()
    env = EdgeSchedulingEnv(str(PROJECT_ROOT / "configs" / "env_config.yaml"))
    ctrl = DMappoAbcController(num_agents=10)
    res = run_episode(env, ctrl, seed=123, max_steps=3600)
    res.pop("reward_trace"); res.pop("miss_trace")
    print(f"episode wall-clock: {time.time()-t0:.1f}s")
    for k, v in res.items():
        print(f"  {k}: {v}")
