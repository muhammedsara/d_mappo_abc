"""
Base Agent Class for Multi-Agent RL
"""

import numpy as np
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """Abstract base class for RL agents"""
    
    def __init__(
        self,
        agent_id: int,
        observation_space,
        action_space,
        config: Optional[Dict] = None
    ):
        """
        Args:
            agent_id: Unique agent identifier
            observation_space: Gymnasium space
            action_space: Gymnasium space
            config: Agent configuration dict
        """
        self.agent_id = agent_id
        self.observation_space = observation_space
        self.action_space = action_space
        self.config = config or {}
        
        # Agent statistics
        self.total_reward = 0.0
        self.episode_steps = 0
        self.episode_reward = 0.0
        
    @abstractmethod
    def select_action(self, observation: np.ndarray, **kwargs) -> int:
        """
        Select action given observation
        
        Args:
            observation: Current state observation
            
        Returns:
            action: Selected action (int)
        """
        pass
    
    @abstractmethod
    def update(self, *args, **kwargs):
        """Update agent policy/value function"""
        pass
    
    def reset_episode(self):
        """Reset episode-specific counters"""
        self.episode_steps = 0
        self.episode_reward = 0.0
    
    def step_update(self, reward: float):
        """Update after environment step"""
        self.episode_reward += reward
        self.total_reward += reward
        self.episode_steps += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        return {
            'agent_id': self.agent_id,
            'total_reward': self.total_reward,
            'episode_reward': self.episode_reward,
            'episode_steps': self.episode_steps,
        }
