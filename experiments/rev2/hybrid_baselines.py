"""
Rev-2: reproducible adaptations of the three hybrid/swarm-RL baselines
compared in Table 9 (Reviewer #8, C4).

All three follow the same methodology as the other MARL-inspired
baselines in experiments/evaluate_advanced_baselines.py: they are
re-implementations adapted to our observation/action spaces, capturing
each method's characteristic coordination mechanism. Exact decision
rules and parameters are documented in Appendix B of the manuscript.

  ZhaoCentralMARLABC   -- Zhao et al. (2023): Q-learning agents whose
                          strategy parameters are managed by a CENTRAL
                          ABC controller (food sources = per-agent
                          preference vectors; fitness-proportional
                          onlooker copying; scout reset after limit).
  FADDEERAttention     -- Nagabushnam & Kim (2025): attention-weighted
                          distributed scheduling; each agent scores
                          offload targets by a compatibility attention
                          over (task demand, target capacity) features.
  WangHybridAction     -- Wang et al. (2025): independent learners with
                          a hybrid action space (discrete target +
                          continuous admission intensity), no
                          population-based coordination.
"""

import numpy as np


class ZhaoCentralMARLABC:
    """Centralized MARL-ABC: linear Q-learning + central ABC controller."""

    NAME = "Zhao et al. (MARL-ABC)"

    def __init__(self, env, lr=0.05, gamma=0.9, epsilon=0.1,
                 n_sources=10, limit=3):
        self.env = env
        self.agents = list(env.possible_agents)
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.limit = limit
        # Linear Q weights per agent: (obs_dim+1) x actions
        self.W = {a: np.zeros((16, 12)) for a in self.agents}
        # Central ABC food sources: candidate preference vectors (12,)
        self.sources = [np.random.randn(12) * 0.1 for _ in range(n_sources)]
        self.fitness = np.zeros(n_sources)
        self.trials = np.zeros(n_sources, dtype=int)
        self.assign = {a: i % n_sources for i, a in enumerate(self.agents)}
        self.prev = {}
        self.ep_reward = 0.0

    def _feat(self, obs):
        return np.concatenate([obs, [1.0]])

    def _q(self, agent, obs):
        pref = self.sources[self.assign[agent]]
        return self._feat(obs) @ self.W[agent] + pref

    def get_actions(self, obs):
        actions = {}
        for a in self.env.agents:
            q = self._q(a, obs[a])
            if np.random.rand() < self.epsilon:
                act = np.random.randint(12)
            else:
                act = int(np.argmax(q))
            actions[a] = act
            self.prev[a] = (obs[a].copy(), act)
        return actions

    def update(self, obs, rewards):
        """Online TD(0) update + central ABC bookkeeping (per step)."""
        for a, (s, act) in self.prev.items():
            if a not in rewards:
                continue
            r = rewards[a]
            self.ep_reward += r
            q_next = self._q(a, obs[a]).max() if a in obs else 0.0
            td = r + self.gamma * q_next - (self._feat(s) @ self.W[a])[act]
            self.W[a][:, act] += self.lr * td * self._feat(s)
            # central fitness bookkeeping (coordinator receives all rewards)
            i = self.assign[a]
            self.fitness[i] = 0.99 * self.fitness[i] + 0.01 * r

    def end_episode(self):
        """Central ABC phases: onlooker selection + scout replacement."""
        probs = np.exp(self.fitness - self.fitness.max())
        probs = probs / probs.sum()
        best = int(np.argmax(self.fitness))
        for i in range(len(self.sources)):
            if i == best:
                continue
            if self.fitness[i] < self.fitness[best] - 1e-9:
                self.trials[i] += 1
            # onlooker: probabilistically move source toward the best
            if np.random.rand() < probs[best]:
                phi = np.random.uniform(-1, 1, size=12)
                self.sources[i] = self.sources[i] + phi * (
                    self.sources[best] - self.sources[i])
                self.trials[i] = 0
            # scout: reset stagnant sources
            if self.trials[i] > self.limit:
                self.sources[i] = np.random.randn(12) * 0.1
                self.trials[i] = 0
                self.fitness[i] = self.fitness.mean()
        self.ep_reward = 0.0


class FADDEERAttention:
    """Attention-weighted distributed scheduler (FADDEER-style)."""

    NAME = "FADDEER (attention)"

    def __init__(self, env, temperature=0.5, local_bias=0.6,
                 cloud_penalty=0.3):
        self.env = env
        self.temperature = temperature
        self.local_bias = local_bias
        self.cloud_penalty = cloud_penalty

    def get_actions(self, obs):
        actions = {}
        agents = list(self.env.agents)
        # capacity feature per agent: low load + low queue = high capacity
        capacity = {a: 1.0 - 0.5 * (obs[a][0] + obs[a][4]) for a in agents}
        for a in agents:
            own = obs[a]
            demand = 0.5 * (own[0] + own[4])       # own congestion
            urgency = 1.0 - own[11]                # short deadline = urgent
            # Lightly loaded agents keep the task local (local_bias acts as
            # the offload threshold on own congestion).
            if demand < (1.0 - self.local_bias):
                actions[a] = 0
                continue
            # Attention over neighbors: softmax-weighted capacity match.
            cand = []
            for other in agents:
                if other == a:
                    continue
                idx = int(other.split("_")[1])
                if idx > 8:
                    continue  # actions 1-9 address neighbors 0-8
                cand.append((idx, capacity[other]))
            if cand:
                caps = np.array([c for _, c in cand])
                attn = np.exp(caps / self.temperature)
                attn = attn / attn.sum()
                j = int(np.argmax(attn))
                best_idx, best_cap = cand[j]
                if best_cap - capacity[a] > 0.05:
                    actions[a] = best_idx + 1
                    continue
            # No sufficiently better neighbor: escalate urgent overload
            # to the cloud, otherwise stay local.
            actions[a] = 10 if (demand + urgency) > (1.0 + self.cloud_penalty) \
                else 0
        return actions

    def update(self, obs, rewards):
        pass

    def end_episode(self):
        pass


class WangHybridAction:
    """Independent hybrid-action learners (Wang et al.-style)."""

    NAME = "Wang et al. (hybrid actions)"

    def __init__(self, env, lr=0.005, gamma=0.9, epsilon=0.15):
        self.env = env
        self.agents = list(env.possible_agents)
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        # discrete head: target selection Q (obs+1 x 11: local, 9 nbrs, cloud)
        self.Wd = {a: np.zeros((16, 11)) for a in self.agents}
        # continuous head: admission intensity in [0,1]; below threshold
        # the task is rejected (hybrid action semantics)
        self.Wc = {a: np.zeros(16) for a in self.agents}
        self.prev = {}

    def _feat(self, obs):
        return np.concatenate([obs, [1.0]])

    def get_actions(self, obs):
        actions = {}
        for a in self.env.agents:
            f = self._feat(obs[a])
            intensity = 1.0 / (1.0 + np.exp(-(f @ self.Wc[a])))
            if intensity < 0.2 and np.random.rand() < 0.5:
                act = 11  # reject (continuous admission head says drop)
            else:
                q = f @ self.Wd[a]
                if np.random.rand() < self.epsilon:
                    choice = np.random.randint(11)
                else:
                    choice = int(np.argmax(q))
                act = choice if choice < 10 else 10
            actions[a] = act
            self.prev[a] = (obs[a].copy(), act)
        return actions

    def update(self, obs, rewards):
        for a, (s, act) in self.prev.items():
            if a not in rewards:
                continue
            r = rewards[a]
            f = self._feat(s)
            if act == 11:
                # admission head update toward rejecting when r >= 0
                grad = (1 if r > 0 else -1) * f
                self.Wc[a] -= self.lr * grad
                continue
            head_act = min(act, 10)
            q_next = (self._feat(obs[a]) @ self.Wd[a]).max() if a in obs else 0.0
            td = r + self.gamma * q_next - (f @ self.Wd[a])[head_act]
            self.Wd[a][:, head_act] += self.lr * td * f

    def end_episode(self):
        pass
