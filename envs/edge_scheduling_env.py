"""
Edge Scheduling Environment for Multi-Agent RL
Compatible with PettingZoo ParallelEnv and RLlib MultiAgentEnv
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pettingzoo import ParallelEnv
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from typing import Dict, List, Optional, Tuple
from collections import deque
import yaml

from envs.task_generator import TaskGenerator, Task


class EdgeDevice:
    """Represents a single edge device"""
    
    def __init__(self, device_id: int, config: Dict):
        self.device_id = device_id
        self.device_type = config['type']
        
        # Hardware specs
        self.cpu_cores = config['cpu_cores']
        self.cpu_speed_ghz = config['cpu_speed_ghz']
        self.memory_mb = config['memory_mb']
        self.gpu_cores = config.get('gpu_cores', 0)
        
        # Power specs
        self.battery_capacity_mah = config['battery_capacity_mah']
        self.idle_power_w = config['idle_power_w']
        self.max_power_w = config['max_power_w']
        
        # State variables
        self.cpu_usage = 0.0
        self.memory_usage = 0.0
        self.battery_level = 100.0
        self.temperature = 40.0
        self.task_queue = deque(maxlen=50)
        self.current_tasks = []
        
        # Statistics
        self.total_tasks_processed = 0
        self.total_energy_consumed_mj = 0
        self.deadline_misses = 0
        
    def get_state_vector(self) -> np.ndarray:
        """Get device state as numpy array (6 dimensions)"""
        return np.array([
            self.cpu_usage,
            self.memory_usage,
            self.battery_level,
            self.temperature,
            len(self.task_queue),
            self._get_normalized_load()
        ], dtype=np.float32)
    
    def _get_normalized_load(self) -> float:
        """Normalized load (0-1)"""
        cpu_norm = self.cpu_usage / 100.0
        queue_norm = min(len(self.task_queue) / 50.0, 1.0)
        return 0.7 * cpu_norm + 0.3 * queue_norm
    
    def add_task_to_queue(self, task: Task):
        """Add task to queue"""
        self.task_queue.append(task)
    
    def process_tasks(self, current_time: float, step_duration: float) -> Tuple[List[Task], float]:
        """
        Process tasks for one timestep
        
        Returns:
            completed_tasks: List of completed tasks
            energy_consumed: Energy consumed in this step (mJ)
        """
        completed = []
        energy = 0.0
        
        # Start new tasks if CPU available
        while self.task_queue and self.cpu_usage < 90:
            task = self.task_queue.popleft()
            self.current_tasks.append({
                'task': task,
                'start_time': current_time,
                'remaining_ms': task.duration_ms
            })
            self.cpu_usage = min(100, self.cpu_usage + task.cpu_load)
            self.memory_usage = min(100, self.memory_usage + (task.memory_mb / self.memory_mb) * 100)
        
        # Process running tasks
        completed_tasks = []
        for running in self.current_tasks[:]:
            running['remaining_ms'] -= step_duration * 1000
            
            if running['remaining_ms'] <= 0:
                task = running['task']
                latency = current_time - task.arrival_time
                
                if latency * 1000 > task.deadline_ms:
                    self.deadline_misses += 1
                
                completed.append(task)
                completed_tasks.append(task)
                self.current_tasks.remove(running)
                
                self.cpu_usage = max(0, self.cpu_usage - task.cpu_load)
                self.memory_usage = max(0, self.memory_usage - (task.memory_mb / self.memory_mb) * 100)
                
                energy += task.energy_cost_mj
                self.total_tasks_processed += 1
        
        # Idle energy consumption
        if self.cpu_usage < 10:
            energy += self.idle_power_w * step_duration * 1000
        else:
            power = self.idle_power_w + (self.max_power_w - self.idle_power_w) * (self.cpu_usage / 100.0)
            energy += power * step_duration * 1000
        
        # Update battery
        energy_mah = energy / 3.6
        self.battery_level = max(0, self.battery_level - (energy_mah / self.battery_capacity_mah) * 100)
        
        # Update temperature (simplified model)
        target_temp = 40 + (self.cpu_usage / 100.0) * 50
        self.temperature += (target_temp - self.temperature) * 0.1
        
        self.total_energy_consumed_mj += energy
        
        return completed, energy
    
    def reset(self):
        """Reset device state"""
        self.cpu_usage = 0.0
        self.memory_usage = 0.0
        self.battery_level = 100.0
        self.temperature = 40.0
        self.task_queue.clear()
        self.current_tasks.clear()
        self.total_tasks_processed = 0
        self.total_energy_consumed_mj = 0
        self.deadline_misses = 0


class EdgeSchedulingEnv(ParallelEnv, MultiAgentEnv):
    """
    Multi-Agent Edge Scheduling Environment
    Compatible with PettingZoo ParallelEnv and RLlib MultiAgentEnv APIs
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "edge_scheduling_v0"
    }
    
    def __init__(self, config: dict = None):
        """
        Initialize environment
        
        Args:
            config: Either a dict with config data or path to YAML file
        """
        super().__init__()
        
        if config is None:
            config_path = "configs/env_config.yaml"
        elif isinstance(config, dict):
            if 'config_path' in config:
                config_path = config['config_path']
            else:
                self.config = config
                config_path = None
        else:
            config_path = config
        
        if config_path is not None:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        
        env_cfg = self.config['environment']
        self._num_agents = env_cfg['num_agents']
        self.max_episode_steps = env_cfg['max_episode_steps']
        self.step_duration = env_cfg['step_duration']
        
        # Create agent names
        self.possible_agents = [f"agent_{i}" for i in range(self._num_agents)]
        self.agents = self.possible_agents[:]
        
        # RLlib MultiAgentEnv requirement
        self._agent_ids = set(self.possible_agents)
        
        # Initialize devices
        self.devices = self._create_devices()
        
        # Initialize task generator
        self.task_generator = TaskGenerator(
            self.config['tasks'],
            seed=42
        )
        
        # Network configuration
        self.network_config = self.config['network']
        self._build_network_topology()
        
        # Reward weights
        reward_cfg = self.config['reward']
        self.reward_weights = {
            'alpha': reward_cfg['alpha'],
            'beta': reward_cfg['beta'],
            'gamma': reward_cfg['gamma'],
            'delta': reward_cfg['delta'],
            'epsilon': reward_cfg['epsilon']
        }
        self.deadline_penalty = reward_cfg['deadline_penalty']
        self.rejection_penalty = reward_cfg['rejection_penalty']
        self.communication_cost = reward_cfg['communication_cost']
        
        # Define observation and action spaces
        self._setup_spaces()
        
        # State variables
        self.current_step = 0
        self.current_time = 0.0
        self.episode_stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'deadline_misses': 0,
            'total_latency': 0,
            'total_energy': 0,
            'rejected_tasks': 0
        }
        
    def _create_devices(self) -> List[EdgeDevice]:
        """Create edge devices from configuration"""
        devices = []
        device_id = 0
        
        for device_config in self.config['devices']:
            device_type_cfg = {
                'type': device_config['type'],
                'cpu_cores': device_config['cpu_cores'],
                'cpu_speed_ghz': device_config['cpu_speed_ghz'],
                'memory_mb': device_config['memory_mb'],
                'gpu_cores': device_config.get('gpu_cores', 0),
                'battery_capacity_mah': device_config['battery_capacity_mah'],
                'idle_power_w': device_config['idle_power_w'],
                'max_power_w': device_config['max_power_w']
            }
            
            for _ in range(device_config['count']):
                device = EdgeDevice(device_id, device_type_cfg)
                devices.append(device)
                device_id += 1
                
                if device_id >= self._num_agents:
                    return devices
        
        return devices
    
    def _build_network_topology(self):
        """Build network adjacency matrix (mesh topology)"""
        self.adjacency = np.ones((self._num_agents, self._num_agents), dtype=bool)
        np.fill_diagonal(self.adjacency, False)
        
        # Network latencies (Gamma distribution)
        base_latency = self.network_config['base_latency_ms']
        latency_std = self.network_config['latency_std_ms']
        
        shape = (base_latency / latency_std) ** 2
        scale = (latency_std ** 2) / base_latency
        
        self.network_latencies = np.random.gamma(shape, scale, 
                                                  (self._num_agents, self._num_agents))
        np.fill_diagonal(self.network_latencies, 0)
    
    @property
    def num_agents(self):
        """Return number of agents (PettingZoo property)"""
        return self._num_agents
    
    def _setup_spaces(self):
        """Setup observation and action spaces"""
        # Observation space: 15 dimensions per agent (normalized to [0, 1])
        obs_low = np.zeros(15, dtype=np.float32)
        obs_high = np.ones(15, dtype=np.float32)
        
        self.observation_spaces = {
            agent: spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
            for agent in self.possible_agents
        }
        
        # Action space: 12 discrete actions
        self.action_spaces = {
            agent: spaces.Discrete(12)
            for agent in self.possible_agents
        }
    
    def observation_space(self, agent):
        """Return observation space for agent"""
        return self.observation_spaces[agent]
    
    def action_space(self, agent):
        """Return action space for agent"""
        return self.action_spaces[agent]
    
    def reset(self, seed=None, options=None):
        """Reset environment"""
        if seed is not None:
            np.random.seed(seed)
        
        self.agents = self.possible_agents[:]
        
        for device in self.devices:
            device.reset()
        
        self.task_generator.reset()
        
        self.current_step = 0
        self.current_time = 0.0
        self.episode_stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'deadline_misses': 0,
            'total_latency': 0,
            'total_energy': 0,
            'rejected_tasks': 0
        }
        
        # Generate initial tasks
        initial_tasks = self.task_generator.generate_tasks_for_timestep(0, self.step_duration)
        for task in initial_tasks:
            device_idx = np.random.randint(0, self.num_agents)
            self.devices[device_idx].add_task_to_queue(task)
        
        observations = self._get_observations()
        infos = {agent: {} for agent in self.agents}
        
        return observations, infos
    
    def step(self, actions: Dict[str, int]):
        """
        Execute one step
        
        Args:
            actions: Dict mapping agent_name -> action_id
            
        Returns:
            observations, rewards, terminations, truncations, infos
        """
        # Generate new tasks for this timestep
        new_tasks = self.task_generator.generate_tasks_for_timestep(
            self.current_time, 
            self.step_duration
        )
        self.episode_stats['total_tasks'] += len(new_tasks)
        
        # Assign new tasks to devices
        for task in new_tasks:
            device_idx = np.random.randint(0, self._num_agents)
            self.devices[device_idx].add_task_to_queue(task)
        
        # Execute actions
        rewards = {}
        
        for agent_name, action in actions.items():
            agent_idx = int(agent_name.split('_')[1])
            device = self.devices[agent_idx]
            
            if len(device.task_queue) == 0:
                rewards[agent_name] = 0.0
                continue
            
            task = device.task_queue[0]
            reward, task_moved = self._execute_action(agent_idx, action, task)
            rewards[agent_name] = reward
            
            if task_moved:
                device.task_queue.popleft()
        
        # Process tasks on all devices
        step_energy = 0.0
        
        for device in self.devices:
            completed, energy = device.process_tasks(self.current_time, self.step_duration)
            step_energy += energy
            
            for task in completed:
                latency = self.current_time - task.arrival_time
                self.episode_stats['completed_tasks'] += 1
                self.episode_stats['total_latency'] += latency
                
                if latency * 1000 > task.deadline_ms:
                    self.episode_stats['deadline_misses'] += 1
        
        self.episode_stats['total_energy'] += step_energy
        
        # Update time
        self.current_step += 1
        self.current_time += self.step_duration
        
        # Get new observations
        observations = self._get_observations()
        
        # Check termination
        terminations = {agent: False for agent in self.agents}
        terminations['__all__'] = False
        
        truncations = {agent: self.current_step >= self.max_episode_steps for agent in self.agents}
        truncations['__all__'] = self.current_step >= self.max_episode_steps
        
        infos = {agent: self._get_info() for agent in self.agents}
        
        return observations, rewards, terminations, truncations, infos
    
    def _execute_action(self, agent_idx: int, action: int, task: Task) -> Tuple[float, bool]:
        """
        Execute agent action
        
        Returns:
            reward: Immediate reward
            task_moved: Whether task was moved/processed
        """
        device = self.devices[agent_idx]
        reward = 0.0
        task_moved = False
        
        if action == 0:
            # Execute locally
            reward = self._compute_local_execution_reward(device, task)
            task_moved = False
            
        elif 1 <= action <= 9:
            # Offload to neighbor
            target_idx = action - 1
            if target_idx >= self._num_agents:
                target_idx = agent_idx
            
            if target_idx != agent_idx and self.adjacency[agent_idx, target_idx]:
                self.devices[target_idx].add_task_to_queue(task)
                reward = self._compute_offload_reward(agent_idx, target_idx, task)
                task_moved = True
            else:
                reward = -self.communication_cost
                task_moved = False
                
        elif action == 10:
            # Offload to cloud
            reward = self._compute_cloud_offload_reward(task)
            task_moved = True
            self.episode_stats['completed_tasks'] += 1
            
        elif action == 11:
            # Reject task
            reward = -self.rejection_penalty
            task_moved = True
            self.episode_stats['rejected_tasks'] += 1
        
        return reward, task_moved
    
    def _compute_local_execution_reward(self, device: EdgeDevice, task: Task) -> float:
        """Compute reward for local execution"""
        estimated_latency = max(task.duration_ms, 1.0)
        estimated_energy = task.energy_cost_mj
        current_load = device._get_normalized_load()
        
        latency_ratio = max(estimated_latency / task.deadline_ms, 1e-6)
        latency_reward = -np.log(np.clip(latency_ratio, 1e-6, 1e6))
        
        energy_reward = -np.clip(estimated_energy / max(device.battery_capacity_mah, 1.0), -10, 10)
        balance_reward = -np.clip(current_load ** 2, 0, 1)
        
        reward = (
            self.reward_weights['alpha'] * latency_reward +
            self.reward_weights['beta'] * energy_reward +
            self.reward_weights['delta'] * balance_reward
        )
        
        return np.clip(reward * 1.5, -5, 5)

    def _compute_offload_reward(self, source_idx: int, target_idx: int, task: Task) -> float:
        """Compute reward for offloading to neighbor"""
        source_device = self.devices[source_idx]
        target_device = self.devices[target_idx]
        
        network_latency = max(self.network_latencies[source_idx, target_idx], 1.0)
        total_latency = max(task.duration_ms + network_latency, 1.0)
        
        transmission_energy = self.communication_cost
        
        source_load = source_device._get_normalized_load()
        target_load = target_device._get_normalized_load()
        balance_benefit = np.clip(source_load - target_load, -1, 1)
        
        latency_ratio = max(total_latency / task.deadline_ms, 1e-6)
        latency_reward = -np.log(np.clip(latency_ratio, 1e-6, 1e6))
        
        energy_reward = -np.clip(transmission_energy, -10, 10)
        balance_reward = balance_benefit
        
        reward = (
            self.reward_weights['alpha'] * latency_reward +
            self.reward_weights['beta'] * energy_reward +
            self.reward_weights['delta'] * balance_reward
        )
        
        return np.clip(reward * 1.2, -5, 5)

    def _compute_cloud_offload_reward(self, task: Task) -> float:
        """Compute reward for cloud offloading"""
        cloud_latency = max(
            np.random.normal(
                self.network_config['cloud']['latency_ms'],
                self.network_config['cloud']['latency_std_ms']
            ),
            1.0
        )
        
        latency_ratio = max(cloud_latency / task.deadline_ms, 1e-6)
        latency_reward = -np.log(np.clip(latency_ratio, 1e-6, 1e6))
        
        energy_reward = -np.clip(self.network_config['cloud']['cost_per_task'], -10, 10)
        
        reward = (
            self.reward_weights['alpha'] * latency_reward +
            self.reward_weights['beta'] * energy_reward
        )
        
        return np.clip(reward * 0.3, -5, 5)

    def _get_observations(self) -> Dict[str, np.ndarray]:
        """Get observations for all agents"""
        observations = {}
        
        for agent_idx, agent_name in enumerate(self.agents):
            device = self.devices[agent_idx]
            
            # Local state (6) - normalized to [0, 1]
            local_state = device.get_state_vector()
            local_normalized = np.array([
                local_state[0] / 100.0,
                local_state[1] / 100.0,
                local_state[2] / 100.0,
                (local_state[3] - 30) / 60.0,
                local_state[4] / 50.0,
                local_state[5]
            ], dtype=np.float32)
            
            # Network state (3) - normalized
            neighbor_latencies = self.network_latencies[agent_idx]
            avg_latency = np.mean(neighbor_latencies[neighbor_latencies > 0])
            network_normalized = np.array([
                np.clip(avg_latency / 500.0, 0, 1),
                np.clip((self.network_config['bandwidth_mbps'] * device.cpu_usage / 100) / 100.0, 0, 1),
                self.network_config['packet_loss_rate'] * 100
            ], dtype=np.float32)
            
            # Task state (3) - normalized
            if len(device.task_queue) > 0:
                task = device.task_queue[0]
                task_normalized = np.array([
                    task.task_type_id / 3.0,
                    task.priority / 2.0,
                    np.clip(task.deadline_ms / 5000.0, 0, 1)
                ], dtype=np.float32)
            else:
                task_normalized = np.array([0, 0, 1], dtype=np.float32)
            
            # Neighbor state (3) - normalized
            neighbor_devices = [self.devices[i] for i in range(self._num_agents) if i != agent_idx]
            if neighbor_devices:
                avg_neighbor_cpu = np.mean([d.cpu_usage for d in neighbor_devices]) / 100.0
                min_neighbor_queue = min([len(d.task_queue) for d in neighbor_devices]) / 50.0
            else:
                avg_neighbor_cpu = 0.0
                min_neighbor_queue = 0.0
            
            neighbor_normalized = np.array([
                avg_neighbor_cpu,
                min_neighbor_queue,
                1.0
            ], dtype=np.float32)
            
            # Concatenate all states (15 dimensions)
            obs = np.concatenate([
                local_normalized,
                network_normalized,
                task_normalized,
                neighbor_normalized
            ])
            
            obs = np.clip(obs, 0.0, 1.0).astype(np.float32)
            observations[agent_name] = obs
        
        return observations

    def _get_info(self) -> Dict:
        """Get info dict"""
        if self.episode_stats['completed_tasks'] > 0:
            avg_latency = self.episode_stats['total_latency'] / self.episode_stats['completed_tasks']
        else:
            avg_latency = 0
        
        energy_wh = self.episode_stats['total_energy'] / 3600
        
        return {
            'episode_step': self.current_step,
            'total_tasks': self.episode_stats['total_tasks'],
            'completed_tasks': self.episode_stats['completed_tasks'],
            'deadline_miss_rate': self.episode_stats['deadline_misses'] / max(self.episode_stats['completed_tasks'], 1),
            'avg_latency': avg_latency,
            'total_energy': energy_wh,
            'rejected_tasks': self.episode_stats['rejected_tasks']
        }
    
    def render(self):
        """Render environment (optional)"""
        if self.current_step % 100 == 0:
            print(f"\n=== Step {self.current_step} ===")
            print(f"Time: {self.current_time:.1f}s")
            print(f"Tasks completed: {self.episode_stats['completed_tasks']}")
            print(f"Deadline misses: {self.episode_stats['deadline_misses']}")
            for i, device in enumerate(self.devices[:3]):
                print(f"Device {i}: CPU={device.cpu_usage:.1f}%, Queue={len(device.task_queue)}")
    
    def close(self):
        """Close environment"""
        pass
