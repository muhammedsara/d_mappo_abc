"""
Task Generator for Edge Scheduling Environment
Generates tasks with Poisson arrival process
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class Task:
    """Task representation"""
    task_id: int
    task_type: str
    task_type_id: int
    cpu_load: float
    memory_mb: int
    duration_ms: float
    deadline_ms: int
    energy_cost_mj: float
    priority: int
    arrival_time: float
    
    def __repr__(self):
        return f"Task(id={self.task_id}, type={self.task_type}, deadline={self.deadline_ms}ms)"


class TaskGenerator:
    """Generates tasks with Poisson arrival process"""
    
    def __init__(self, task_configs: List[Dict], seed: Optional[int] = None):
        """
        Args:
            task_configs: List of task type configurations from YAML
            seed: Random seed for reproducibility
        """
        self.task_configs = task_configs
        self.task_types = [cfg['name'] for cfg in task_configs]
        self.rng = np.random.RandomState(seed)
        
        # Calculate total arrival rate (tasks per minute)
        self.total_arrival_rate = sum(cfg['arrival_rate'] for cfg in task_configs)
        
        # Task ID counter
        self.task_id_counter = 0
        
    def generate_tasks_for_timestep(self, timestep: float, step_duration: float = 1.0) -> List[Task]:
        """
        Generate tasks for current timestep using Poisson process
        
        Args:
            timestep: Current simulation timestep (seconds)
            step_duration: Duration of one timestep (seconds)
            
        Returns:
            List of generated tasks
        """
        tasks = []
        
        # Convert arrival rate from per-minute to per-second
        lambda_per_second = self.total_arrival_rate / 60.0
        
        # Number of tasks in this timestep (Poisson distribution)
        num_tasks = self.rng.poisson(lambda_per_second * step_duration)
        
        for _ in range(num_tasks):
            task = self._generate_single_task(timestep)
            tasks.append(task)
            
        return tasks
    
    def _generate_single_task(self, arrival_time: float) -> Task:
        """Generate a single task"""
        
        # Select task type based on arrival rates (weighted probability)
        arrival_rates = [cfg['arrival_rate'] for cfg in self.task_configs]
        probabilities = np.array(arrival_rates) / sum(arrival_rates)
        task_type_idx = self.rng.choice(len(self.task_configs), p=probabilities)
        
        config = self.task_configs[task_type_idx]
        
        # Sample task parameters
        cpu_load = self.rng.uniform(config['cpu_load'][0], config['cpu_load'][1])
        duration_ms = self.rng.uniform(config['duration_ms'][0], config['duration_ms'][1])
        
        task = Task(
            task_id=self.task_id_counter,
            task_type=config['name'],
            task_type_id=task_type_idx,
            cpu_load=cpu_load,
            memory_mb=config['memory_mb'],
            duration_ms=duration_ms,
            deadline_ms=config['deadline_ms'],
            energy_cost_mj=config['energy_cost_mj'],
            priority=config['priority'],
            arrival_time=arrival_time
        )
        
        self.task_id_counter += 1
        return task
    
    def generate_burst_scenario(
        self, 
        timestep: float, 
        burst_multiplier: float = 3.0,
        burst_duration: float = 60.0
    ) -> List[Task]:
        """
        Generate burst of tasks (e.g., morning rush)
        
        Args:
            timestep: Current timestep
            burst_multiplier: Multiply arrival rate by this factor
            burst_duration: Duration of burst in seconds
        """
        original_rate = self.total_arrival_rate
        self.total_arrival_rate *= burst_multiplier
        
        tasks = self.generate_tasks_for_timestep(timestep, burst_duration)
        
        self.total_arrival_rate = original_rate
        return tasks
    
    def reset(self):
        """Reset task generator"""
        self.task_id_counter = 0
