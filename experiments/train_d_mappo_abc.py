"""
Training script for D-MAPPO-ABC
Distributed Multi-Agent PPO + Artificial Bee Colony
"""

import os
import sys
import yaml
import numpy as np
from pathlib import Path

# Add project root to path
# Ensure project root is in path
project_root = Path(__file__).parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

"""
Training script for D-MAPPO-ABC
Distributed Multi-Agent PPO + Artificial Bee Colony
"""

import os
import sys
import yaml
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.tune.logger import pretty_print

# Import custom modules
from envs.edge_scheduling_env import EdgeSchedulingEnv
from agents.abc_agent import ABCCoordinator

# Optional: Weights & Biases
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Logging to local only.")


def load_config(config_path: str) -> dict:
    """Load YAML configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_policy_mapping_fn(config: dict):
    """
    Create policy mapping function
    Maps agents to policies based on configuration
    """
    employed_agents = [f"agent_{i}" for i in config['training']['policy_mapping']['employed_agents']]
    onlooker_agents = [f"agent_{i}" for i in config['training']['policy_mapping']['onlooker_agents']]
    scout_agents = [f"agent_{i}" for i in config['training']['policy_mapping']['scout_agents']]
    
    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        if agent_id in employed_agents:
            return "employed_policy"
        elif agent_id in onlooker_agents:
            return "onlooker_policy"
        elif agent_id in scout_agents:
            return "scout_policy"
        else:
            # Default to employed
            return "employed_policy"
    
    return policy_mapping_fn


def create_multiagent_policies(env, config: dict):
    """
    Create multi-agent policy specifications
    
    Returns:
        policies: Dict of policy specs
        policy_mapping_fn: Function mapping agents to policies
    """
    obs_space = env.observation_space(env.possible_agents[0])
    act_space = env.action_space(env.possible_agents[0])
    
    # Policy configurations
    employed_cfg = config['training']['policies']['employed_policy']
    onlooker_cfg = config['training']['policies']['onlooker_policy']
    scout_cfg = config['training']['policies']['scout_policy']
    
    policies = {
        "employed_policy": PolicySpec(
            observation_space=obs_space,
            action_space=act_space,
            config={
                "model": employed_cfg['model']
            }
        ),
        "onlooker_policy": PolicySpec(
            observation_space=obs_space,
            action_space=act_space,
            config={
                "model": onlooker_cfg['model']
            }
        ),
        "scout_policy": PolicySpec(
            observation_space=obs_space,
            action_space=act_space,
            config={
                "model": scout_cfg['model']
            }
        )
    }
    
    policy_mapping_fn = create_policy_mapping_fn(config)
    
    return policies, policy_mapping_fn


def create_rllib_config(env, config: dict, abc_coordinator: ABCCoordinator):
    """Create RLlib PPO configuration"""
    
    # Get policies
    policies, policy_mapping_fn = create_multiagent_policies(env, config)
    
    # Training config
    train_cfg = config['training']
    ppo_cfg = train_cfg['ppo']
    
    # Create PPO config
    algo_config = (
        PPOConfig()
        .environment(
            env=EdgeSchedulingEnv,
            env_config={"config_path": "configs/env_config.yaml"}
        )
        .framework(train_cfg['framework'])
        .rollouts(
            num_rollout_workers=train_cfg['num_workers'],
            rollout_fragment_length=train_cfg['rollout']['rollout_fragment_length'],
            batch_mode=train_cfg['rollout']['batch_mode']
        )
        .training(
            lr=ppo_cfg['lr'],
            gamma=ppo_cfg['gamma'],
            lambda_=ppo_cfg['lambda'],
            clip_param=ppo_cfg['clip_param'],
            vf_clip_param=ppo_cfg['vf_clip_param'],
            entropy_coeff=ppo_cfg['entropy_coeff'],
            train_batch_size=ppo_cfg['train_batch_size'],
            sgd_minibatch_size=ppo_cfg['sgd_minibatch_size'],
            num_sgd_iter=ppo_cfg['num_sgd_iter'],
            use_gae=ppo_cfg['use_gae'],
            vf_loss_coeff=ppo_cfg['vf_loss_coeff'],
            kl_coeff=ppo_cfg['kl_coeff'],
            kl_target=ppo_cfg['kl_target'],
            grad_clip=ppo_cfg.get('grad_clip', 0.5)  # EKLENDİ
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=list(policies.keys())
        )
        .resources(
            num_gpus=train_cfg.get('num_gpus', 0),
            num_gpus_per_worker=train_cfg.get('num_gpus_per_worker', 0)
        )
        .debugging(
            log_level="INFO"
        )
    )
    
    return algo_config


class ABCMAPPOTrainer:
    """Custom trainer integrating ABC with MAPPO"""
    
    def __init__(self, config_path: str = "configs/training_config.yaml"):
        """Initialize trainer"""
        
        # Load configurations
        self.train_config = load_config(config_path)
        self.env_config = load_config("configs/env_config.yaml")
        self.abc_config = load_config("configs/abc_config.yaml")
        
        # Create environment
        self.env = EdgeSchedulingEnv("configs/env_config.yaml")
        
        # Initialize ABC Coordinator
        abc_cfg = self.abc_config['abc']
        self.abc_coordinator = ABCCoordinator(
            num_food_sources=abc_cfg['num_food_sources'],
            solution_dim=15,  # State dimension
            phi_range=(abc_cfg['employed']['phi_min'], abc_cfg['employed']['phi_max']),
            limit=abc_cfg['scout']['limit'],
            seed=self.train_config['training']['seed']
        )
        
        # Setup logging
        self.setup_logging()
        
        # Training state
        self.episode = 0
        self.best_reward = -np.inf
        
    def setup_logging(self):
        """Setup logging (W&B, TensorBoard, etc.)"""
        log_cfg = self.train_config['logging']
        
        # Weights & Biases
        if WANDB_AVAILABLE and log_cfg['wandb']['enabled']:
            # Create meaningful run name
            timestamp = datetime.now().strftime('%m%d_%H%M')
            lr = self.train_config['training']['ppo']['lr']
            abc_sources = self.abc_config['abc']['num_food_sources']
            
            run_name = f"D-MAPPO-ABC_lr{lr}_abc{abc_sources}_{timestamp}"
            
            # Prepare config for W&B
            wandb_config = {
                # Training params
                'learning_rate': lr,
                'train_batch_size': self.train_config['training']['ppo']['train_batch_size'],
                'gamma': self.train_config['training']['ppo']['gamma'],
                'clip_param': self.train_config['training']['ppo']['clip_param'],
                
                # ABC params
                'abc_food_sources': abc_sources,
                'abc_limit': self.abc_config['abc']['scout']['limit'],
                'abc_phi_range': [self.abc_config['abc']['employed']['phi_min'], 
                                self.abc_config['abc']['employed']['phi_max']],
                
                # Environment params
                'num_agents': self.env_config['environment']['num_agents'],
                'max_episode_steps': self.env_config['environment']['max_episode_steps'],
                
                # Reward weights
                'reward_alpha': self.env_config['reward']['alpha'],
                'reward_beta': self.env_config['reward']['beta'],
                'reward_gamma': self.env_config['reward']['gamma'],
            }
            
            # Create tags
            tags = log_cfg['wandb'].get('tags', [])
            tags.extend([
                f"lr_{lr}",
                f"abc_{abc_sources}",
                f"agents_{self.env_config['environment']['num_agents']}"
            ])
            
            wandb.init(
                project=log_cfg['wandb']['project'],
                entity=log_cfg['wandb'].get('entity'),
                name=run_name,  # Meaningful name
                config=wandb_config,  # All hyperparameters
                tags=tags,  # Searchable tags
                notes=f"D-MAPPO-ABC training with {abc_sources} ABC sources"  # Description
            )
            self.use_wandb = True
            
            print(f"✓ W&B initialized: {run_name}")
            print(f"✓ Tags: {tags}")
        else:
            self.use_wandb = False
        
        # Create checkpoint directory
        checkpoint_dir = Path(log_cfg['checkpoint']['save_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = checkpoint_dir

    '''def setup_logging(self):
        """Setup logging (W&B, TensorBoard, etc.)"""
        log_cfg = self.train_config['logging']
        
        # Weights & Biases
        if WANDB_AVAILABLE and log_cfg['wandb']['enabled']:
            wandb.init(
                project=log_cfg['wandb']['project'],
                entity=log_cfg['wandb'].get('entity'),
                config={
                    'train_config': self.train_config,
                    'env_config': self.env_config,
                    'abc_config': self.abc_config
                },
                tags=log_cfg['wandb'].get('tags', []),
                name=f"{self.train_config['training']['run_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            self.use_wandb = True
        else:
            self.use_wandb = False
        
        # Create checkpoint directory
        checkpoint_dir = Path(log_cfg['checkpoint']['save_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = checkpoint_dir'''
        
    def train(self):
        """Main training loop"""
        
        # Initialize Ray
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        
        # Create RLlib config
        algo_config = create_rllib_config(self.env, self.train_config, self.abc_coordinator)
        
        # Build algorithm
        algo = algo_config.build()
        
        print("\n" + "="*60)
        print("Starting D-MAPPO-ABC Training")
        print("="*60)
        print(f"Environment: {self.env_config['environment']['name']}")
        print(f"Num Agents: {self.env_config['environment']['num_agents']}")
        print(f"ABC Food Sources: {self.abc_config['abc']['num_food_sources']}")
        print(f"Max Episodes: {self.train_config['training']['num_episodes']}")
        print("="*60 + "\n")
        
        # Training loop
        num_episodes = self.train_config['training']['num_episodes']
        eval_interval = self.train_config['training']['evaluation_interval']
        checkpoint_interval = self.train_config['training']['checkpoint_interval']
        
        for episode in range(num_episodes):
            self.episode = episode
            
            # Train one iteration
            result = algo.train()
            
            # ABC step (update food sources based on episode rewards)
            self._abc_step(result)
            
            # Logging
            if episode % self.train_config['logging']['console']['log_interval'] == 0:
                self._log_training_stats(result)
            
            # Evaluation
            if episode % eval_interval == 0 and episode > 0:
                eval_result = self._evaluate(algo)
                self._log_evaluation_stats(eval_result)
            
            # Checkpointing
            if episode % checkpoint_interval == 0 and episode > 0:
                checkpoint_path = algo.save(str(self.checkpoint_dir))
                print(f"✓ Checkpoint saved: {checkpoint_path}")
                
                # Save ABC state
                self._save_abc_state(episode)
        
        # Final evaluation
        print("\n" + "="*60)
        print("Training Complete! Running final evaluation...")
        print("="*60)
        final_eval = self._evaluate(algo, num_episodes=20)
        self._log_evaluation_stats(final_eval, final=True)
        
        # Save final model
        final_checkpoint = algo.save(str(self.checkpoint_dir / "final"))
        print(f"✓ Final model saved: {final_checkpoint}")
        
        # Cleanup
        algo.stop()
        ray.shutdown()
        
        if self.use_wandb:
            wandb.finish()
    def _abc_step(self, result: dict):
        """Perform ABC step based on training results"""
        # Get episode rewards - USE EVAL REWARD if available
        episode_reward = result.get('evaluation', {}).get('episode_reward_mean', 
                                result.get('episode_reward_mean', 0))
        
        # NaN/Inf check
        if np.isnan(episode_reward) or np.isinf(episode_reward):
            print(f"  ⚠️ Invalid reward detected: {episode_reward}, skipping ABC update")
            # Don't return - still do ABC step to maintain iteration
            fitness = -100  # Default bad fitness
        else:
            # Normalize fitness to reasonable range
            if episode_reward > 0:
                fitness = min(episode_reward / 10, 100)  # Scale down positive
            else:
                fitness = max(episode_reward / 100, -100)  # Scale down negative
        
        # Update ABC coordinator - update MORE food sources for faster learning
        #num_updates = min(10, self.abc_coordinator.num_food_sources // 5)  # 20% of population
        # EKLE: Episode sayısına göre dinamik update
        if self.episode < 100:
            num_updates = min(15, self.abc_coordinator.num_food_sources // 3)  # İlk 100'de agresif
        else:
            num_updates = min(10, self.abc_coordinator.num_food_sources // 5)  # Sonra normal
    


        indices = np.random.choice(self.abc_coordinator.num_food_sources, 
                                size=num_updates, replace=False)
        
        updated_count = 0
        for idx in indices:
            current_fitness = self.abc_coordinator.food_sources[idx].fitness
            improved = fitness > current_fitness if np.isfinite(current_fitness) else True
            if not np.isnan(fitness):
                self.abc_coordinator.update_fitness(idx, fitness, improved)
                updated_count += 1
        
        # Perform ABC iteration
        abc_stats = self.abc_coordinator.step()
        
        # Enhanced logging
        print(f"  ABC: fitness={fitness:.2f}, best={abc_stats['best_fitness']:.2f}, "
            f"avg={abc_stats['avg_fitness']:.2f}, scouts={abc_stats['num_scouts']}, "
            f"improved={abc_stats['improved_ratio']:.2%}, stagnant={abc_stats['stagnation_count']}")
        
        if self.use_wandb:
            wandb.log({
                'abc/current_fitness': fitness,
                'abc/best_fitness': abc_stats['best_fitness'],
                'abc/avg_fitness': abc_stats['avg_fitness'],
                'abc/num_scouts': abc_stats['num_scouts'],
                'abc/improved_ratio': abc_stats['improved_ratio'],
                'abc/stagnation_count': abc_stats['stagnation_count'],
                'abc/valid_sources': abc_stats['valid_sources'],
                'abc/iteration': abc_stats['iteration'],
                'abc/updated_sources': updated_count
            }, step=self.episode)
    '''
    def _abc_step(self, result: dict):
        """
        Perform ABC step based on training results
        Update food sources with episode rewards
        """
        # Get episode rewards - USE EVAL REWARD (more stable)
        episode_reward = result.get('evaluation', {}).get('episode_reward_mean', 
                                result.get('episode_reward_mean', 0))
        
        # NaN/Inf check
        if np.isnan(episode_reward) or np.isinf(episode_reward):
            print(f"  ⚠️ Invalid reward detected: {episode_reward}, skipping ABC update")
            return
        
        # Normalize fitness to reasonable range
        # Eval rewards are around 70, so scale to [0, 100]
        if episode_reward > 0:
            fitness = min(episode_reward, 100)  # Cap at 100
        else:
            fitness = max(episode_reward / 100, -100)  # Scale down negative
        
        # Update ABC coordinator
        # Randomly select 5 food sources to update (diversify)
        num_updates = min(5, self.abc_coordinator.num_food_sources)
        indices = np.random.choice(self.abc_coordinator.num_food_sources, 
                                size=num_updates, replace=False)
        
        for idx in indices:
            current_fitness = self.abc_coordinator.food_sources[idx].fitness
            improved = fitness > current_fitness
            self.abc_coordinator.update_fitness(idx, fitness, improved)
        
        # Perform ABC iteration
        abc_stats = self.abc_coordinator.step()
        
        # Log ABC stats
        print(f"  ABC: fitness={fitness:.2f}, best={abc_stats['best_fitness']:.2f}, "
            f"avg={abc_stats['avg_fitness']:.2f}, scouts={abc_stats['num_scouts']}")
        
        if self.use_wandb:
            wandb.log({
                'abc/current_fitness': fitness,
                'abc/best_fitness': abc_stats['best_fitness'],
                'abc/avg_fitness': abc_stats['avg_fitness'],
                'abc/num_scouts': abc_stats['num_scouts'],
                'abc/iteration': abc_stats['iteration']
            }, step=self.episode)
    '''
    '''
    def _abc_step(self, result: dict):
        """
        Perform ABC step based on training results
        Update food sources with episode rewards
        """
        # Get episode rewards
        episode_reward = result.get('episode_reward_mean', 0)

        # DÜZELT: NaN / Inf kontrolü ekle — geçersizse default kötü reward kullan
        if np.isnan(episode_reward) or np.isinf(episode_reward):
            episode_reward = -100  # Default kötü reward

        # Map episode reward to ABC fitness (higher reward = higher fitness)
        # DÜZELT: Fitness'i güvenli aralığa clip et
        fitness = np.clip(episode_reward, -1000, 1000)

        # Update a random food source (simple approach)
        idx = np.random.randint(0, self.abc_coordinator.num_food_sources)

        # DÜZELT: Mevcut fitness NaN/Inf olabilir, robust karşılaştırma yap
        current_fitness = getattr(self.abc_coordinator.food_sources[idx], "fitness", np.nan)
        if np.isnan(current_fitness) or np.isinf(current_fitness):
            current_fitness = -np.inf
        improved = fitness > current_fitness

        # DÜZELT: NaN check before update
        if not np.isnan(fitness) and not np.isinf(fitness):
            self.abc_coordinator.update_fitness(idx, float(fitness), improved)

        # Perform ABC iteration
        abc_stats = self.abc_coordinator.step()

        # Log ABC stats
        if self.use_wandb:
            wandb.log({
                'abc/best_fitness': abc_stats.get('best_fitness'),
                'abc/avg_fitness' : abc_stats.get('avg_fitness'),
                'abc/num_scouts'  : abc_stats.get('num_scouts'),
                'abc/iteration'   : abc_stats.get('iteration')
            }, step=self.episode)

    
    '''        
    '''
    def _abc_step(self, result: dict):
        """Perform ABC step based on training results and update ABC coordinator."""
        #import numpy as np

        # 1) Episode reward'u al ve sayısal/sonlu yap
        episode_reward = result.get('episode_reward_mean', 0)
        try:
            episode_reward = float(episode_reward)
        except Exception:
            episode_reward = -100.0
        if not np.isfinite(episode_reward):
            episode_reward = -100.0

        # 2) Fitness'a haritala ve sınırla
        fitness = float(np.clip(episode_reward, -1000, 1000))

        # 3) Rastgele bir food source seç ve güvenli "improved" karşılaştırması yap
        if self.abc_coordinator.num_food_sources > 0:
            idx = np.random.randint(0, self.abc_coordinator.num_food_sources)
            current_fit = getattr(self.abc_coordinator.food_sources[idx], "fitness", -np.inf)
            if not np.isfinite(current_fit):
                current_fit = -np.inf
            improved = fitness > current_fit

            # Tek bir kez güncelle (NaN değil çünkü yukarıda garanti ettik)
            self.abc_coordinator.update_fitness(idx, fitness, improved)

        # 4) ABC iterasyonu
        abc_stats = self.abc_coordinator.step()

        # 5) (Opsiyonel) Logla
        if getattr(self, "use_wandb", False):
            import wandb
            wandb.log({
                "abc/best_fitness": abc_stats.get("best_fitness"),
                "abc/avg_fitness": abc_stats.get("avg_fitness"),
                "abc/num_scouts": abc_stats.get("num_scouts"),
                "abc/iteration": abc_stats.get("iteration"),
            }, step=getattr(self, "episode", None))

        return abc_stats
        
    '''
    ''''
    def _log_training_stats(self, result: dict):
        """Log training statistics"""
        episode_reward = result.get('episode_reward_mean', 0)
        episode_len = result.get('episode_len_mean', 0)
        
        print(f"\nEpisode {self.episode}")
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Length: {episode_len:.0f}")
        print(f"  ABC Best Fitness: {self.abc_coordinator.best_source.fitness if self.abc_coordinator.best_source else 0:.2f}")
        
        if self.use_wandb:
            wandb.log({
                'train/episode_reward': episode_reward,
                'train/episode_length': episode_len,
                'train/episode': self.episode
            }, step=self.episode)
    '''
    def _log_training_stats(self, result: dict):
        """Enhanced logging"""
        episode_reward = result.get('episode_reward_mean', 0)
        episode_len = result.get('episode_len_mean', 0)
        
        # Policy-specific rewards
        #policy_rewards = result.get('policy_reward_mean', {})
        # EKLE: Daha detaylı log
        policy_loss = result.get('info', {}).get('learner', {}).get('employed_policy', {}).get('learner_stats', {}).get('policy_loss', 0)
        vf_loss = result.get('info', {}).get('learner', {}).get('employed_policy', {}).get('learner_stats', {}).get('vf_loss', 0)



        print(f"\nEpisode {self.episode}")
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Length: {episode_len:.0f}")
        print(f"  Policy Loss: {policy_loss:.4f}")  # EKLE
        print(f"  Value Loss: {vf_loss:.4f}")  # EKLE
        print(f"  ABC Best Fitness: {self.abc_coordinator.best_source.fitness if self.abc_coordinator.best_source else 0:.2f}")
        
        if self.use_wandb:
            wandb.log({
                'train/episode_reward': episode_reward,
                'train/episode_length': episode_len,
                'train/policy_loss': policy_loss,  # EKLE
                'train/value_loss': vf_loss,  # EKLE
                'train/episode': self.episode
            }, step=self.episode)


        '''
        print(f"\n{'='*60}")
        print(f"Episode {self.episode}")
        print(f"{'='*60}")
        print(f"  Overall Reward: {episode_reward:.2f}")
        print(f"  Episode Length: {episode_len:.0f}")
        
        for policy_id, reward in policy_rewards.items():
            print(f"  {policy_id}: {reward:.2f}")
        
        print(f"\n  ABC Stats:")
        print(f"    Best Fitness: {self.abc_coordinator.best_source.fitness if self.abc_coordinator.best_source else 0:.2f}")
        print(f"    Avg Fitness:  {np.mean([s.fitness for s in self.abc_coordinator.food_sources if s.fitness != -np.inf]):.2f}")
        print(f"    Valid Sources: {sum(1 for s in self.abc_coordinator.food_sources if s.fitness != -np.inf)}/{len(self.abc_coordinator.food_sources)}")
        print(f"{'='*60}\n")
        '''

    def _evaluate(self, algo, num_episodes: int = 10) -> dict:
        """Evaluate trained policy"""
        eval_env = EdgeSchedulingEnv("configs/env_config.yaml")
        
        total_rewards = []
        total_latencies = []
        total_energies = []
        deadline_miss_rates = []
        
        for ep in range(num_episodes):
            obs, infos = eval_env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # Get actions from policies
                actions = {}
                for agent in eval_env.agents:
                    agent_idx = int(agent.split('_')[1])
                    
                    # Determine policy
                    if agent_idx in self.train_config['training']['policy_mapping']['employed_agents']:
                        policy_id = "employed_policy"
                    elif agent_idx in self.train_config['training']['policy_mapping']['onlooker_agents']:
                        policy_id = "onlooker_policy"
                    else:
                        policy_id = "scout_policy"
                    
                    # Get action from policy
                    action = algo.compute_single_action(
                        obs[agent],
                        policy_id=policy_id,
                        explore=False
                    )
                    actions[agent] = action
                
                # Step environment
                obs, rewards, terminations, truncations, infos = eval_env.step(actions)
                
                episode_reward += sum(rewards.values()) / len(rewards)
                done = any(truncations.values()) or any(terminations.values())
            
            # Collect metrics
            info = infos[eval_env.agents[0]]
            total_rewards.append(episode_reward)
            total_latencies.append(info.get('avg_latency', 0))
            total_energies.append(info.get('total_energy', 0))
            deadline_miss_rates.append(info.get('deadline_miss_rate', 0))
        
        return {
            'eval_reward_mean': np.mean(total_rewards),
            'eval_reward_std': np.std(total_rewards),
            'eval_latency_mean': np.mean(total_latencies),
            'eval_energy_mean': np.mean(total_energies),
            'eval_deadline_miss_rate': np.mean(deadline_miss_rates)
        }
    
    def _log_evaluation_stats(self, eval_result: dict, final: bool = False):
        """Log evaluation statistics"""
        prefix = "FINAL" if final else "EVAL"
        
        print(f"\n{prefix} Evaluation Results:")
        print(f"  Reward: {eval_result['eval_reward_mean']:.2f} ± {eval_result['eval_reward_std']:.2f}")
        print(f"  Latency: {eval_result['eval_latency_mean']:.2f} ms")
        print(f"  Energy: {eval_result['eval_energy_mean']:.2f} mJ")
        print(f"  Deadline Miss Rate: {eval_result['eval_deadline_miss_rate']:.2%}")
        
        if self.use_wandb:
            log_dict = {
                'eval/reward_mean': eval_result['eval_reward_mean'],
                'eval/reward_std': eval_result['eval_reward_std'],
                'eval/latency': eval_result['eval_latency_mean'],
                'eval/energy': eval_result['eval_energy_mean'],
                'eval/deadline_miss_rate': eval_result['eval_deadline_miss_rate']
            }
            
            if final:
                # Log as summary for final results
                for key, value in log_dict.items():
                    wandb.run.summary[f"final_{key}"] = value
            else:
                wandb.log(log_dict, step=self.episode)
        
        # Update best reward
        if eval_result['eval_reward_mean'] > self.best_reward:
            self.best_reward = eval_result['eval_reward_mean']
            print(f"  ★ New best reward: {self.best_reward:.2f}")
    
    def _save_abc_state(self, episode: int):
        """Save ABC coordinator state"""
        import pickle
        
        abc_state = {
            'food_sources': self.abc_coordinator.food_sources,
            'best_source': self.abc_coordinator.best_source,
            'best_fitness_history': self.abc_coordinator.best_fitness_history,
            'iteration': self.abc_coordinator.iteration
        }
        
        save_path = self.checkpoint_dir / f"abc_state_ep{episode}.pkl"
        with open(save_path, 'wb') as f:
            pickle.dump(abc_state, f)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Train D-MAPPO-ABC")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/training_config.yaml',
        help='Path to training configuration'
    )
    parser.add_argument(
        '--no-wandb',
        action='store_true',
        help='Disable Weights & Biases logging'
    )
    
    args = parser.parse_args()
    
    # Override wandb setting if flag is set
    if args.no_wandb:
        config = load_config(args.config)
        config['logging']['wandb']['enabled'] = False
        # Save modified config
        with open('configs/training_config_temp.yaml', 'w') as f:
            yaml.dump(config, f)
        config_path = 'configs/training_config_temp.yaml'
    else:
        config_path = args.config
    
    # Create and run trainer
    trainer = ABCMAPPOTrainer(config_path)
    trainer.train()


if __name__ == "__main__":
    main()