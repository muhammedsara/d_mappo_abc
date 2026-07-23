"""
Rev-2 shared experiment harness.

Provides environment builders (with configurable agent count and network
conditions), all scheduling controllers (D-MAPPO-ABC checkpoint variants,
classical heuristics, MARL-inspired baselines), and the episode runner
used by every Rev-2 experiment.
"""

import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from envs.edge_scheduling_env import EdgeSchedulingEnv  # noqa: E402
from experiments.rev2.policy_runner import (  # noqa: E402
    ARCHS, PolicyMLP, _ShimUnpickler,
)
from experiments.evaluate_advanced_baselines import (  # noqa: E402
    IndependentPPOBaseline, QMIXInspiredBaseline, CommNetInspiredBaseline,
    MADDPGInspiredBaseline, LoadAwareWeightedRoundRobin,
)

CHECKPOINT_ROOT = PROJECT_ROOT / "logs" / "checkpoints"
ENV_CONFIG_PATH = PROJECT_ROOT / "configs" / "env_config.yaml"

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# --------------------------------------------------------------------------
# Environment construction
# --------------------------------------------------------------------------

def build_env(num_agents: int = 10,
              packet_loss: float | None = None,
              latency_scale: float = 1.0,
              arrival_rate_scale: float = 1.0,
              task_seed: int = 42) -> EdgeSchedulingEnv:
    """Build an EdgeSchedulingEnv with modified configuration.

    packet_loss: if set, packet_loss_rate is changed AND inter-device
        latencies are scaled by the expected retransmission factor
        1/(1-p) on top of latency_scale (lost packets are retransmitted,
        which converts loss into extra latency, consistent with the
        simulator's stated retransmission semantics).
    latency_scale: multiplies the base mean network latency (congestion /
        reduced-bandwidth proxy).
    """
    with open(ENV_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    cfg["environment"]["num_agents"] = num_agents

    if num_agents != 10:
        # Keep the 30/40/30 hardware mix of the baseline configuration.
        n_jetson = max(1, round(0.3 * num_agents))
        n_cam = max(1, round(0.3 * num_agents))
        n_rpi = max(1, num_agents - n_jetson - n_cam)
        for dev in cfg["devices"]:
            if dev["type"] == "jetson_nano":
                dev["count"] = n_jetson
            elif dev["type"] == "raspberry_pi":
                dev["count"] = n_rpi
            else:
                dev["count"] = n_cam

    if arrival_rate_scale != 1.0:
        for task in cfg["tasks"]:
            task["arrival_rate"] = task["arrival_rate"] * arrival_rate_scale

    eff_latency_scale = latency_scale
    if packet_loss is not None:
        cfg["network"]["packet_loss_rate"] = packet_loss
        eff_latency_scale *= 1.0 / (1.0 - packet_loss)
    if eff_latency_scale != 1.0:
        cfg["network"]["base_latency_ms"] = (
            cfg["network"]["base_latency_ms"] * eff_latency_scale)
        cfg["network"]["latency_std_ms"] = (
            cfg["network"]["latency_std_ms"] * eff_latency_scale)
        # Degradation affects the WAN path to the cloud as well.
        cfg["network"]["cloud"]["latency_ms"] = (
            cfg["network"]["cloud"]["latency_ms"] * eff_latency_scale)
        cfg["network"]["cloud"]["latency_std_ms"] = (
            cfg["network"]["cloud"]["latency_std_ms"] * eff_latency_scale)

    env = EdgeSchedulingEnv(copy.deepcopy(cfg))
    # Give the task generator an experiment-controlled stream so different
    # seeds produce genuinely different workload realizations.
    env.task_generator.rng = np.random.RandomState(task_seed)
    return env


# --------------------------------------------------------------------------
# Controllers
# --------------------------------------------------------------------------

class CheckpointController:
    """Role-based controller loading weights from an RLlib checkpoint dir.

    variant:
      "full"          -> standard 4-4-2 role mapping
      "no_scout"      -> scout agents use the employed policy
      "no_onlooker"   -> onlooker agents use the employed policy
      "single_policy" -> all agents use the employed policy
    """

    name = "D-MAPPO-ABC"

    def __init__(self, checkpoint_dir: Path | None = None,
                 num_agents: int = 10, variant: str = "full"):
        ckpt = Path(checkpoint_dir) if checkpoint_dir else (
            CHECKPOINT_ROOT / "policies")
        self.policies = {}
        for pname in ARCHS:
            with open(ckpt / pname / "policy_state.pkl", "rb") as f:
                weights = _ShimUnpickler(f).load()["weights"]
            model = PolicyMLP(15, ARCHS[pname], 12)
            sd = {}
            for i in range(len(ARCHS[pname])):
                sd[f"hidden.{2*i}.weight"] = torch.as_tensor(
                    np.asarray(weights[f"_hidden_layers.{i}._model.0.weight"]))
                sd[f"hidden.{2*i}.bias"] = torch.as_tensor(
                    np.asarray(weights[f"_hidden_layers.{i}._model.0.bias"]))
            sd["logits.weight"] = torch.as_tensor(
                np.asarray(weights["_logits._model.0.weight"]))
            sd["logits.bias"] = torch.as_tensor(
                np.asarray(weights["_logits._model.0.bias"]))
            model.load_state_dict(sd)
            model.eval()
            self.policies[pname] = model

        self.role_of = {}
        for i in range(num_agents):
            frac = i / num_agents
            if frac < 0.4:
                role = "employed_policy"
            elif frac < 0.8:
                role = "onlooker_policy"
            else:
                role = "scout_policy"
            if variant == "no_scout" and role == "scout_policy":
                role = "employed_policy"
            elif variant == "no_onlooker" and role == "onlooker_policy":
                role = "employed_policy"
            elif variant == "single_policy":
                role = "employed_policy"
            self.role_of[i] = role

    def reset(self):
        pass

    @torch.no_grad()
    def get_actions(self, env, obs):
        actions = {}
        for agent_name, ob in obs.items():
            idx = int(agent_name.split("_")[1])
            logits = self.policies[self.role_of[idx]](
                torch.from_numpy(ob).float().unsqueeze(0))
            actions[agent_name] = int(torch.argmax(logits, dim=1).item())
        return actions


class RoundRobinController:
    name = "Round-Robin"

    def __init__(self, num_agents: int = 10):
        self.num_agents = num_agents

    def reset(self):
        pass

    def get_actions(self, env, obs):
        return {a: (i % min(self.num_agents, 10)) + 1
                for i, a in enumerate(env.agents)}


class GreedyController:
    name = "Greedy"

    def reset(self):
        pass

    def get_actions(self, env, obs):
        actions = {}
        for agent in env.agents:
            cpu = obs[agent][0]
            if cpu < 0.5:
                actions[agent] = 0
            elif cpu < 0.8:
                actions[agent] = 1
            else:
                actions[agent] = 10
        return actions


class RandomController:
    name = "Random"

    def reset(self):
        pass

    def get_actions(self, env, obs):
        return {a: int(np.random.randint(0, 12)) for a in env.agents}


class AdvancedBaselineController:
    """Wraps the MARL-inspired baseline classes behind a uniform API."""

    def __init__(self, cls, name):
        self.cls = cls
        self.name = name
        self._instance = None

    def reset(self):
        self._instance = None

    def get_actions(self, env, obs):
        if self._instance is None:
            self._instance = self.cls(env)
        return self._instance.get_actions(obs)


def make_controller(method: str, num_agents: int = 10):
    if method == "D-MAPPO-ABC":
        return CheckpointController(num_agents=num_agents)
    if method == "Pure MAPPO":
        c = CheckpointController(CHECKPOINT_ROOT / "ablation_no_abc" / "policies",
                                 num_agents=num_agents)
        c.name = "Pure MAPPO"
        return c
    if method == "w/o Scout":
        c = CheckpointController(CHECKPOINT_ROOT / "ablation_no_scout" / "policies",
                                 num_agents=num_agents, variant="no_scout")
        c.name = "w/o Scout"
        return c
    if method == "w/o Onlooker":
        c = CheckpointController(CHECKPOINT_ROOT / "ablation_no_onlooker" / "policies",
                                 num_agents=num_agents, variant="no_onlooker")
        c.name = "w/o Onlooker"
        return c
    if method == "Single Policy":
        c = CheckpointController(CHECKPOINT_ROOT / "ablation_single_policy" / "policies",
                                 num_agents=num_agents, variant="single_policy")
        c.name = "Single Policy"
        return c
    if method == "Round-Robin":
        return RoundRobinController(num_agents)
    if method == "Greedy":
        return GreedyController()
    if method == "Random":
        return RandomController()
    if method == "IPPO":
        return AdvancedBaselineController(IndependentPPOBaseline, "IPPO")
    if method == "QMIX-inspired":
        return AdvancedBaselineController(QMIXInspiredBaseline, "QMIX-inspired")
    if method == "CommNet-inspired":
        return AdvancedBaselineController(CommNetInspiredBaseline,
                                          "CommNet-inspired")
    if method == "MADDPG-inspired":
        return AdvancedBaselineController(MADDPGInspiredBaseline,
                                          "MADDPG-inspired")
    if method == "Load-Aware WRR":
        return AdvancedBaselineController(LoadAwareWeightedRoundRobin,
                                          "Load-Aware WRR")
    raise ValueError(f"unknown method {method}")


# --------------------------------------------------------------------------
# Episode runner
# --------------------------------------------------------------------------

def run_episode(env, controller, seed=None, max_steps=3600,
                dropout=None, collect_traces=False):
    """Run one evaluation episode.

    dropout: {"step": int, "agents": [indices]} — from `step` onward the
        listed agents stop acting and stop receiving offloaded tasks.

    Returns a metrics dict. Messages/step counts every executed
    neighbor-offload transfer (task transmission) plus one local state
    broadcast per active agent per step (the information each agent shares
    with neighbors for the observation aggregates).
    """
    obs, _ = env.reset(seed=seed)
    controller.reset()

    total_reward = 0.0
    decision_times = []
    offload_msgs = 0
    cloud_msgs = 0
    step = 0
    done = False
    dead = set()
    reward_trace, miss_trace, throughput_trace = [], [], []

    while not done and step < max_steps:
        if dropout and step == dropout["step"]:
            dead = set(dropout["agents"])
            for d in dead:
                env.adjacency[:, d] = False
                env.adjacency[d, :] = False

        t0 = time.perf_counter()
        actions = controller.get_actions(env, obs)
        n_actions = max(len(actions), 1)
        decision_times.append((time.perf_counter() - t0) / n_actions)

        actions = {a: act for a, act in actions.items()
                   if int(a.split("_")[1]) not in dead}
        for a, act in actions.items():
            if 1 <= act <= 9:
                offload_msgs += 1
            elif act == 10:
                cloud_msgs += 1

        obs, rewards, terminations, truncations, infos = env.step(actions)

        alive = [r for a, r in rewards.items()
                 if int(a.split("_")[1]) not in dead]
        step_r = float(np.mean(alive)) if alive else 0.0
        total_reward += step_r

        if collect_traces:
            info = infos[env.agents[0]]
            reward_trace.append(step_r)
            miss_trace.append(info.get("deadline_miss_rate", 0.0))
            throughput_trace.append(info.get("completed_tasks", 0))

        done = any(t for k, t in truncations.items() if k != "__all__") or \
               any(t for k, t in terminations.items() if k != "__all__")
        step += 1

    info = infos[env.agents[0]]
    completed = info.get("completed_tasks", 0)
    total = max(info.get("total_tasks", 1), 1)
    out = {
        "reward": total_reward,
        "avg_latency_s": info.get("avg_latency", 0.0),
        "energy_seu": info.get("total_energy", 0.0),
        "deadline_miss_rate": info.get("deadline_miss_rate", 0.0),
        "completed_tasks": completed,
        "total_tasks": info.get("total_tasks", 0),
        "rejected_tasks": info.get("rejected_tasks", 0),
        "completion_ratio": completed / total,
        "steps": step,
        "decision_ms": 1000.0 * float(np.mean(decision_times)),
        "offload_msgs_per_step": offload_msgs / max(step, 1),
        "cloud_msgs_per_step": cloud_msgs / max(step, 1),
    }
    if collect_traces:
        out["reward_trace"] = reward_trace
        out["miss_trace"] = miss_trace
        out["throughput_trace"] = throughput_trace
    return out
