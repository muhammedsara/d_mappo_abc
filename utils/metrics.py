"""
Metrics tracking and computation for D-MAPPO-ABC
"""

import numpy as np
from typing import Dict, List
from collections import defaultdict


class MetricsTracker:
    """Track and compute performance metrics"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.episode_rewards = []
        self.episode_lengths = []
        self.latencies = []
        self.energies = []
        self.deadline_misses = []
        self.throughputs = []
        self.cpu_usages = defaultdict(list)
        self.queue_lengths = defaultdict(list)
    
    def add_episode(
        self,
        reward: float,
        length: int,
        avg_latency: float,
        total_energy: float,
        deadline_miss_rate: float,
        throughput: float
    ):
        """Add episode metrics"""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.latencies.append(avg_latency)
        self.energies.append(total_energy)
        self.deadline_misses.append(deadline_miss_rate)
        self.throughputs.append(throughput)
    
    def add_device_metrics(self, device_id: int, cpu_usage: float, queue_length: int):
        """Add per-device metrics"""
        self.cpu_usages[device_id].append(cpu_usage)
        self.queue_lengths[device_id].append(queue_length)
    
    def compute_statistics(self) -> Dict:
        """Compute summary statistics"""
        return {
            # Reward
            'reward_mean': np.mean(self.episode_rewards),
            'reward_std': np.std(self.episode_rewards),
            'reward_min': np.min(self.episode_rewards),
            'reward_max': np.max(self.episode_rewards),
            
            # Latency
            'latency_mean': np.mean(self.latencies),
            'latency_std': np.std(self.latencies),
            'latency_p50': np.percentile(self.latencies, 50),
            'latency_p95': np.percentile(self.latencies, 95),
            'latency_p99': np.percentile(self.latencies, 99),
            
            # Energy
            'energy_mean': np.mean(self.energies),
            'energy_total': np.sum(self.energies),
            
            # Deadline
            'deadline_miss_rate': np.mean(self.deadline_misses),
            
            # Throughput
            'throughput_mean': np.mean(self.throughputs),
            'throughput_total': np.sum(self.throughputs),
            
            # Load Balance (Jain's Fairness Index)
            'load_balance_jain': self._compute_jain_index(),
            'load_balance_variance': self._compute_load_variance(),
        }
    
    def _compute_jain_index(self) -> float:
        """
        Compute Jain's Fairness Index for load balance
        JFI = (sum(x_i))^2 / (n * sum(x_i^2))
        where x_i is the load of device i
        """
        if not self.cpu_usages:
            return 1.0
        
        avg_loads = [np.mean(loads) for loads in self.cpu_usages.values()]
        
        n = len(avg_loads)
        sum_loads = sum(avg_loads)
        sum_loads_squared = sum(x**2 for x in avg_loads)
        
        if sum_loads_squared == 0:
            return 1.0
        
        jfi = (sum_loads ** 2) / (n * sum_loads_squared)
        return jfi
    
    def _compute_load_variance(self) -> float:
        """Compute variance of CPU usage across devices"""
        if not self.cpu_usages:
            return 0.0
        
        avg_loads = [np.mean(loads) for loads in self.cpu_usages.values()]
        return np.var(avg_loads)
    
    def print_summary(self):
        """Print summary statistics"""
        stats = self.compute_statistics()
        
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)
        
        print(f"\nReward:")
        print(f"  Mean: {stats['reward_mean']:.2f} +/- {stats['reward_std']:.2f}")
        print(f"  Range: [{stats['reward_min']:.2f}, {stats['reward_max']:.2f}]")
        
        print(f"\nLatency:")
        print(f"  Mean: {stats['latency_mean']:.2f} ms")
        print(f"  P50: {stats['latency_p50']:.2f} ms")
        print(f"  P95: {stats['latency_p95']:.2f} ms")
        print(f"  P99: {stats['latency_p99']:.2f} ms")
        
        print(f"\nEnergy:")
        print(f"  Mean per episode: {stats['energy_mean']:.2f} mJ")
        print(f"  Total: {stats['energy_total']:.2f} mJ")
        
        print(f"\nDeadline Miss Rate: {stats['deadline_miss_rate']:.2%}")
        
        print(f"\nThroughput:")
        print(f"  Mean: {stats['throughput_mean']:.2f} tasks/episode")
        print(f"  Total: {stats['throughput_total']:.0f} tasks")
        
        print(f"\nLoad Balance:")
        print(f"  Jain's Index: {stats['load_balance_jain']:.3f} (1.0 = perfect)")
        print(f"  Variance: {stats['load_balance_variance']:.3f}")
        
        print("="*60 + "\n")
        
        return stats
