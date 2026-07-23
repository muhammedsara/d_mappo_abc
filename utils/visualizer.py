"""
Visualization utilities for D-MAPPO-ABC
"""

import matplotlib.pyplot as plt


class Visualizer:
    """Visualization tools for training and evaluation"""
    
    def plot_rewards(self, rewards, save_path=None):
        """Plot reward progression over episodes"""
        plt.figure(figsize=(10, 6))
        plt.plot(rewards)
        plt.title("Episode Reward Progression")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_latency(self, latencies, save_path=None):
        """Plot latency over episodes"""
        plt.figure(figsize=(10, 6))
        plt.plot(latencies)
        plt.title("Average Latency per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Latency (ms)")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_comparison(self, results_dict, metric_name, save_path=None):
        """Plot comparison of multiple methods"""
        plt.figure(figsize=(12, 6))
        
        for name, values in results_dict.items():
            plt.plot(values, label=name)
        
        plt.title(f"{metric_name} Comparison")
        plt.xlabel("Episode")
        plt.ylabel(metric_name)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
