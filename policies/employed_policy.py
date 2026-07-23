"""
Employed Bee Policy Network for D-MAPPO-ABC
Optimized for local task execution decisions
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional


class EmployedPolicy(nn.Module):
    """
    Policy network for employed bee agents
    Architecture: [128, 128] - compact for rapid local decisions
    """
    
    def __init__(
        self,
        obs_dim: int = 15,
        action_dim: int = 12,
        hidden_sizes: Tuple[int, int] = (128, 128),
        activation: str = "relu"
    ):
        """
        Initialize employed policy network
        
        Args:
            obs_dim: Observation space dimension
            action_dim: Action space dimension
            hidden_sizes: Hidden layer sizes
            activation: Activation function
        """
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()
        
        # Policy network (actor)
        self.policy_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_sizes[0]),
            self.activation,
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            self.activation,
            nn.Linear(hidden_sizes[1], action_dim)
        )
        
        # Value network (critic)
        self.value_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_sizes[0]),
            self.activation,
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            self.activation,
            nn.Linear(hidden_sizes[1], 1)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            obs: Observation tensor
            
        Returns:
            action_logits: Action logits
            value: State value estimate
        """
        action_logits = self.policy_net(obs)
        value = self.value_net(obs)
        return action_logits, value
    
    def compute_action(self, obs: np.ndarray, deterministic: bool = False) -> int:
        """
        Compute action from observation
        
        Args:
            obs: Observation array
            deterministic: If True, return argmax action
            
        Returns:
            Selected action
        """
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            action_logits, _ = self.forward(obs_tensor)
            
            if deterministic:
                action = action_logits.argmax(dim=-1).item()
            else:
                probs = torch.softmax(action_logits, dim=-1)
                action = torch.multinomial(probs, 1).item()
        
        return action
    
    def get_value(self, obs: np.ndarray) -> float:
        """Get value estimate for observation"""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            _, value = self.forward(obs_tensor)
        return value.item()
    
    def get_weights(self) -> Dict[str, np.ndarray]:
        """Get policy weights as numpy arrays"""
        return {k: v.cpu().numpy() for k, v in self.state_dict().items()}
    
    def set_weights(self, weights: Dict[str, np.ndarray]):
        """Set policy weights from numpy arrays"""
        state_dict = {k: torch.FloatTensor(v) for k, v in weights.items()}
        self.load_state_dict(state_dict)
