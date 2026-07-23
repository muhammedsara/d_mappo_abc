"""
Artificial Bee Colony Coordinator
Integrates with MAPPO for hybrid learning
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class FoodSource:
    """Represents a food source (solution) in ABC"""
    position: np.ndarray
    fitness: float = -np.inf
    trial_counter: int = 0
    
    def __repr__(self):
        return f"FoodSource(fitness={self.fitness:.3f}, trials={self.trial_counter})"


class ABCCoordinator:
    """
    ABC Coordinator that works alongside MAPPO
    Manages food sources and ABC operators
    """
    
    def __init__(
        self,
        num_food_sources: int = 50,
        solution_dim: int = 10,
        phi_range: Tuple[float, float] = (-1.0, 1.0),
        limit: int = 10,
        seed: Optional[int] = None
    ):
        """
        Initialize ABC Coordinator
        
        Args:
            num_food_sources: Number of solutions to maintain
            solution_dim: Dimension of solution space
            phi_range: Range for position update parameter
            limit: Abandonment limit for scout bees
            seed: Random seed
        """
        self.num_food_sources = num_food_sources
        self.solution_dim = solution_dim
        self.phi_min, self.phi_max = phi_range
        self.limit = limit
        
        self.rng = np.random.RandomState(seed)
        
        # Initialize food sources
        self.food_sources = []
        self._initialize_food_sources()
        
        # Best solution tracking
        self.best_source: Optional[FoodSource] = None
        self.best_fitness_history = []
        
        # Statistics
        self.iteration = 0
        
    def _initialize_food_sources(self):
        """Initialize food sources with random positions"""
        self.food_sources = []
        
        for _ in range(self.num_food_sources):
            position = self.rng.uniform(-1, 1, self.solution_dim)
            source = FoodSource(position=position)
            self.food_sources.append(source)
    
    def employed_bee_phase(self) -> List[FoodSource]:
        """
        Employed bee phase: Explore neighborhood of current solutions
        
        Returns:
            Updated food sources
        """
        updated_sources = []
        
        for i, source in enumerate(self.food_sources):
            # Select random neighbor (different from self)
            neighbor_indices = [j for j in range(self.num_food_sources) if j != i]
            j = self.rng.choice(neighbor_indices)
            neighbor = self.food_sources[j]
            
            # Position update: x_new = x + phi * (x - x_neighbor)
            phi = self.rng.uniform(self.phi_min, self.phi_max, self.solution_dim)
            new_position = source.position + phi * (source.position - neighbor.position)
            
            # Clip to valid range
            new_position = np.clip(new_position, -1, 1)
            
            # Create new candidate
            new_source = FoodSource(
                position=new_position,
                fitness=source.fitness,
                trial_counter=source.trial_counter
            )
            
            updated_sources.append(new_source)
        
        return updated_sources
    
    def onlooker_bee_phase(self) -> List[int]:
        """
        Onlooker bee phase: Select best solutions probabilistically
        
        Returns:
            List of selected food source indices
        """
        # Calculate selection probabilities (roulette wheel)
        fitnesses = np.array([source.fitness for source in self.food_sources])
        
        # Handle invalid fitnesses (inf, -inf, nan)
        valid_mask = np.isfinite(fitnesses)
        if not valid_mask.any():
            probabilities = np.ones(self.num_food_sources) / self.num_food_sources
        else:
            # Shift to positive values (handle negative fitnesses)
            min_fitness = np.min(fitnesses[valid_mask])
            if min_fitness < 0:
                fitnesses = fitnesses - min_fitness + 1.0
            
            # Replace invalid values with minimum
            fitnesses[~valid_mask] = np.min(fitnesses[valid_mask]) if valid_mask.any() else 1.0
            
            # Avoid division by zero
            fitness_sum = np.sum(fitnesses)
            if fitness_sum == 0 or not np.isfinite(fitness_sum):
                probabilities = np.ones(self.num_food_sources) / self.num_food_sources
            else:
                probabilities = fitnesses / fitness_sum
        
        # Final check for NaN
        if np.any(~np.isfinite(probabilities)):
            probabilities = np.ones(self.num_food_sources) / self.num_food_sources
        
        # Select food sources based on probability
        num_onlookers = self.num_food_sources
        selected_indices = self.rng.choice(
            self.num_food_sources,
            size=num_onlookers,
            replace=True,
            p=probabilities
        )
        
        return selected_indices.tolist()
    
    def scout_bee_phase(self) -> List[Tuple[int, FoodSource]]:
        """
        Scout bee phase: Replace abandoned solutions
        
        Returns:
            List of (index, new_food_source) tuples
        """
        replacements = []
        
        for i, source in enumerate(self.food_sources):
            if source.trial_counter > self.limit:
                # Abandon this solution and generate new random position
                new_position = self.rng.uniform(-1, 1, self.solution_dim)
                new_source = FoodSource(
                    position=new_position,
                    fitness=-np.inf,
                    trial_counter=0
                )
                replacements.append((i, new_source))
        
        return replacements
    
    def update_fitness(self, index: int, fitness: float, improved: bool = False):
        """
        Update fitness of a food source
        
        Args:
            index: Food source index
            fitness: New fitness value
            improved: Whether fitness improved
        """
        self.food_sources[index].fitness = fitness
        
        if improved:
            self.food_sources[index].trial_counter = 0
        else:
            self.food_sources[index].trial_counter += 1
        
        # Update best solution
        if self.best_source is None or fitness > self.best_source.fitness:
            self.best_source = FoodSource(
                position=self.food_sources[index].position.copy(),
                fitness=fitness,
                trial_counter=0
            )
    
    def get_experience_weights(self, experience_fitnesses: np.ndarray) -> np.ndarray:
        """
        Get experience replay weights based on ABC fitness
        
        Args:
            experience_fitnesses: Array of fitness values for experiences
            
        Returns:
            Normalized weights for prioritized sampling
        """
        temperature = 1.0
        weights = np.exp(experience_fitnesses / temperature)
        weights = weights / np.sum(weights)
        
        return weights
    
    def get_guidance_for_agent(self, agent_state: np.ndarray) -> np.ndarray:
        """
        Get ABC guidance for an agent's state
        
        Args:
            agent_state: Current agent state
            
        Returns:
            Guidance vector (similar dimension to action space)
        """
        # Find most similar food source
        similarities = []
        for source in self.food_sources:
            sim = -np.linalg.norm(source.position - agent_state[:self.solution_dim])
            similarities.append(sim)
        
        best_idx = np.argmax(similarities)
        best_source = self.food_sources[best_idx]
        
        # Return guidance as direction vector
        guidance = best_source.position - agent_state[:self.solution_dim]
        guidance = np.clip(guidance, -1, 1)
        
        return guidance
    
    def step(self):
        """Perform one ABC iteration"""
        # 1. Employed bee phase
        updated_sources = self.employed_bee_phase()
        
        # 2. Onlooker bee phase
        selected_indices = self.onlooker_bee_phase()
        
        # 3. Scout bee phase
        replacements = self.scout_bee_phase()
        
        # Apply scout replacements
        for idx, new_source in replacements:
            self.food_sources[idx] = new_source
        
        # Update best fitness history
        if self.best_source:
            self.best_fitness_history.append(self.best_source.fitness)
        
        # Calculate statistics
        valid_fitnesses = [s.fitness for s in self.food_sources if np.isfinite(s.fitness)]
        improved_count = sum(1 for s in self.food_sources if s.trial_counter == 0)
        stagnant_count = sum(1 for s in self.food_sources if s.trial_counter > self.limit // 2)
        
        self.iteration += 1
        
        return {
            'iteration': self.iteration,
            'best_fitness': self.best_source.fitness if self.best_source and np.isfinite(self.best_source.fitness) else -np.inf,
            'avg_fitness': np.mean(valid_fitnesses) if valid_fitnesses else -np.inf,
            'num_scouts': len(replacements),
            'improved_ratio': improved_count / self.num_food_sources if self.num_food_sources > 0 else 0,
            'stagnation_count': stagnant_count,
            'valid_sources': len(valid_fitnesses),
            'phi_mean': np.mean([self.phi_min, self.phi_max])
        }
    
    def reset(self):
        """Reset ABC coordinator"""
        self._initialize_food_sources()
        self.best_source = None
        self.best_fitness_history = []
        self.iteration = 0
    
    def get_stats(self) -> Dict:
        """Get ABC statistics"""
        fitnesses = [s.fitness for s in self.food_sources if np.isfinite(s.fitness)]
        
        return {
            'iteration': self.iteration,
            'num_food_sources': len(self.food_sources),
            'best_fitness': self.best_source.fitness if self.best_source and np.isfinite(self.best_source.fitness) else -np.inf,
            'avg_fitness': np.mean(fitnesses) if fitnesses else 0,
            'std_fitness': np.std(fitnesses) if fitnesses else 0,
            'num_valid_sources': len(fitnesses),
            'avg_trial_counter': np.mean([s.trial_counter for s in self.food_sources])
        }
