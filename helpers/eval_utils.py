"""
Evaluation utilities for Transformer SAC Wind Farm Control.

This module provides functions for periodic policy evaluation during training.
Evaluation runs for a fixed number of steps (not waiting for episode termination),
which gives more consistent and comparable metrics across different layouts.

Usage:
    evaluator = PolicyEvaluator(
        actor=actor,
        eval_layouts=["test_layout", "square_2x2"],
        env_factory=env_factory,
        combined_wrapper=combined_wrapper,
        num_envs=4,
        num_eval_steps=100,
        num_eval_episodes=5,
        device=device,
        rotor_diameter=rotor_diameter,
        seed=42,
    )
    
    # In training loop:
    if global_step >= next_eval_step:
        metrics = evaluator.evaluate()
        for name, value in metrics.items():
            writer.add_scalar(name, value, global_step)
        next_eval_step += eval_interval

Author: Marcus (DTU Wind Energy)
"""

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from collections import defaultdict
from .agent import WindFarmAgent
from .multi_layout_env import MultiLayoutEnv, LayoutConfig
from .helper_funcs import transform_to_wind_relative
from .layouts import get_layout_positions

from WindGym.wrappers import RecordEpisodeVals

@dataclass
class EvalMetrics:
    """Container for evaluation metrics."""
    mean_reward: float
    std_reward: float
    mean_step_reward: float
    mean_power: float
    std_power: float
    mean_baseline_power: float
    power_ratio: float  # agent_power / baseline_power
    per_layout_rewards: Dict[str, float]
    per_layout_powers: Dict[str, float]
    per_layout_power_ratios: Dict[str, float]
    num_episodes: int
    num_steps_per_episode: int
    
    def to_dict(self, prefix: str = "eval") -> Dict[str, float]:
        """Convert to flat dictionary for logging."""
        metrics = {
            f"{prefix}/mean_reward": self.mean_reward,
            f"{prefix}/std_reward": self.std_reward,
            f"{prefix}/mean_step_reward": self.mean_step_reward,
            f"{prefix}/mean_power": self.mean_power,
            f"{prefix}/std_power": self.std_power,
            f"{prefix}/mean_baseline_power": self.mean_baseline_power,
            f"{prefix}/power_ratio": self.power_ratio,
        }
        
        # Add per-layout metrics
        for layout, reward in self.per_layout_rewards.items():
            metrics[f"{prefix}/layout/{layout}/mean_reward"] = reward
        for layout, power in self.per_layout_powers.items():
            metrics[f"{prefix}/layout/{layout}/mean_power"] = power
        for layout, ratio in self.per_layout_power_ratios.items():
            metrics[f"{prefix}/layout/{layout}/power_ratio"] = ratio
            
        return metrics


class PolicyEvaluator:
    """
    Evaluator for transformer-based wind farm control policies.
    
    Creates and manages evaluation environments, runs episodes for a fixed
    number of steps, and computes metrics.
    
    Key features:
    - Supports multiple evaluation layouts (different from training)
    - Per-layout metrics for analyzing generalization
    - Fixed step count (doesn't wait for termination)
    - Deterministic action selection
    - Tracks both agent and baseline power for comparison
    - Computes receptivity/influence profiles when needed
    """
    
    def __init__(
        self,
        agent: WindFarmAgent,  # Changed from actor
        eval_layouts: List[str],
        env_factory: Callable[[np.ndarray, np.ndarray], gym.Env],
        combined_wrapper: Callable[[gym.Env], gym.Env],
        num_envs: int,
        num_eval_steps: int,
        num_eval_episodes: int,
        device: torch.device,
        rotor_diameter: float,
        wind_turbine: Any,
        seed: int = 42,
        max_turbines: Optional[int] = None,
        deterministic: bool = True,
        use_profiles: bool = False,
        n_profile_directions: int = 360,
        profile_source: str = "pywake",
    ):
        """
        Args:
            agent: The agent to evaluate
            eval_layouts: List of layout names to evaluate on
            env_factory: Function that creates base WindFarmEnv given (x_pos, y_pos)
            combined_wrapper: Wrapper function to apply to each env
            num_envs: Number of parallel environments for evaluation
            num_eval_steps: Number of steps per evaluation episode
            num_eval_episodes: Number of episode iterations to run (total episodes = num_eval_episodes * num_envs)
            device: Torch device
            rotor_diameter: Rotor diameter for position normalization
            wind_turbine: PyWake wind turbine object for layout generation
            seed: Random seed for evaluation environments
            max_turbines: Max turbines for padding (if None, computed from layouts)
            deterministic: If True, use deterministic actions (mean). If False, sample stochastically.
            use_profiles: Whether to compute receptivity/influence profiles for layouts
            n_profile_directions: Number of directions in profile (default 360)
        """
        self.agent = agent
        self.eval_layout_names = eval_layouts
        self.env_factory = env_factory
        self.combined_wrapper = combined_wrapper
        self.num_envs = num_envs
        self.num_eval_steps = num_eval_steps
        self.num_eval_episodes = num_eval_episodes
        self.device = device
        self.rotor_diameter = rotor_diameter
        self.wind_turbine = wind_turbine
        self.seed = seed
        self.max_turbines = max_turbines
        self.deterministic = deterministic
        self.use_profiles = use_profiles
        self.n_profile_directions = n_profile_directions
        self.profile_source = profile_source
        
        # Create layout configs
        self.eval_layouts = self._create_layout_configs()
        
        # Create evaluation environments (will be created lazily on first evaluate call)
        self._eval_envs: Optional[gym.vector.VectorEnv] = None
        
    def _create_layout_configs(self) -> List[LayoutConfig]:
        """Create LayoutConfig objects for evaluation layouts, including profiles if needed."""
        configs = []
        for name in self.eval_layout_names:
            x_pos, y_pos = get_layout_positions(name, self.wind_turbine)
            config = LayoutConfig(name=name, x_pos=x_pos, y_pos=y_pos)
            
            # Compute profiles if enabled
            if self.use_profiles:
                if self.profile_source.lower() == "geometric":
                    from .geometric_profiles import compute_layout_profiles_vectorized
                    D = self.wind_turbine.diameter()
                    print(f"[Evaluator] Computing GEOMETRIC profiles for layout: {name}")
                    receptivity_profiles, influence_profiles = compute_layout_profiles_vectorized(
                        x_pos, y_pos,
                        rotor_diameter=D,
                        k_wake=0.04,
                        n_directions=self.n_profile_directions,
                        sigma_smooth=10.0,
                        scale_factor=15.0,
                    )
                else:
                    from .receptivity_profiles import compute_layout_profiles
                    print(f"[Evaluator] Computing PyWake profiles for layout: {name}")
                    receptivity_profiles, influence_profiles = compute_layout_profiles(
                        x_pos, y_pos, self.wind_turbine,
                        n_directions=self.n_profile_directions,
                    )



                config.receptivity_profiles = receptivity_profiles
                config.influence_profiles = influence_profiles
            
            configs.append(config)
        return configs
    
    
    def _create_eval_envs(self) -> gym.vector.VectorEnv:
        """Create vectorized evaluation environments."""
        
        # Compute required max_turbines from eval layouts
        eval_max_turbines = max(layout.n_turbines for layout in self.eval_layouts)
        
        # Use the larger of provided max_turbines or required for eval layouts
        # This ensures eval envs can handle all eval layouts, even if they're
        # larger than training layouts
        effective_max_turbines = max(self.max_turbines or 0, eval_max_turbines)
        
        if self.max_turbines is not None and effective_max_turbines > self.max_turbines:
            print(f"Note: Eval layouts require {eval_max_turbines} turbines, "
                  f"which is larger than training max_turbines ({self.max_turbines}). "
                  f"Using {effective_max_turbines} for eval environments.")
        
        def make_eval_env_fn(seed: int):
            def _init():
                env = MultiLayoutEnv(
                    layouts=self.eval_layouts,
                    env_factory=self.env_factory,
                    per_turbine_wrapper=self.combined_wrapper,
                    seed=seed,
                    shuffle=False,  # No shuffling during evaluation for consistency
                    max_turbines=effective_max_turbines,
                    max_episode_steps=self.num_eval_steps+1000, #Just to be safe.
                )
                return env
            return _init
        
        # Use AsyncVectorEnv for parallel execution (faster for expensive sims)
        envs = gym.vector.AsyncVectorEnv(
            [make_eval_env_fn(self.seed + i) for i in range(self.num_envs)]
        )
        envs = RecordEpisodeVals(envs)
        return envs
    
    @property
    def eval_envs(self) -> gym.vector.VectorEnv:
        """Lazily create evaluation environments."""
        if self._eval_envs is None:
            self._eval_envs = self._create_eval_envs()
        return self._eval_envs
    
    def _get_current_layouts(self) -> List[str]:
        """Get current layout name for each environment."""
        names_tuple = self.eval_envs.env.get_attr('current_layout')
        return [names_tuple[x].name for x in range(len(names_tuple))]
    
    def evaluate(self) -> EvalMetrics:
        """
        Run evaluation and return metrics.
        
        Runs num_eval_episodes episode iterations. With num_envs parallel environments,
        this results in (num_eval_episodes * num_envs) total independent episodes.
        Each episode runs for num_eval_steps steps (not waiting for termination).
        
        Returns:
            EvalMetrics object containing all evaluation statistics
        """
        self.agent.eval()
        
        # Storage for metrics - each env run is treated as an independent episode
        # So with 5 iterations × 10 envs, we get 50 episode samples
        all_episode_rewards = []  # Total reward per episode (length: num_eval_episodes * num_envs)
        all_episode_powers = []   # Mean power per episode
        all_baseline_powers = []  # Mean baseline power per episode
        
        # Per-layout tracking (already correct - tracks each env independently)
        layout_rewards = defaultdict(list)
        layout_powers = defaultdict(list)
        layout_baseline_powers = defaultdict(list)
        
        n_turbines_max = self.eval_envs.single_observation_space.shape[0]
        
        for episode_idx in range(self.num_eval_episodes):
            # Reset environments
            obs, infos = self.eval_envs.reset(seed=self.seed + episode_idx)
            
            episode_reward = np.zeros(self.num_envs)  # Track reward per env
            episode_power = []      # List of (num_envs,) arrays per step
            episode_baseline_power = []
            episode_layouts = self._get_current_layouts()
            
            for step_idx in range(self.num_eval_steps):

                actions = self.agent.act(self.eval_envs, obs, 
                                         deterministic=self.deterministic)

                # Step environment
                next_obs, rewards, terminations, truncations, infos = self.eval_envs.step(actions)
                
                # Accumulate rewards per env
                episode_reward += rewards
                
                # Track power metrics per env
                if "Power agent" in infos:
                    episode_power.append(infos["Power agent"])
                if "Power baseline" in infos:
                    episode_baseline_power.append(infos["Power baseline"])
                
                # Handle auto-reset (environments that terminated/truncated)
                # The rewards are already captured, we just continue with the new obs
                obs = next_obs
            
            # Store metrics for EACH env as an independent episode
            for env_idx in range(self.num_envs):
                # Total reward for this env's episode
                all_episode_rewards.append(episode_reward[env_idx])
                
                # Mean power across steps for this env's episode
                if episode_power:
                    env_mean_power = np.mean([p[env_idx] for p in episode_power])
                    all_episode_powers.append(env_mean_power)
                
                if episode_baseline_power:
                    env_mean_baseline = np.mean([p[env_idx] for p in episode_baseline_power])
                    all_baseline_powers.append(env_mean_baseline)
                
                # Per-layout tracking
                layout_name = episode_layouts[env_idx]
                layout_rewards[layout_name].append(episode_reward[env_idx])
                if episode_power:
                    layout_powers[layout_name].append(env_mean_power)
                if episode_baseline_power:
                    layout_baseline_powers[layout_name].append(env_mean_baseline)
        
        self.agent.train()
        
        # Compute aggregate metrics over ALL episodes (num_eval_episodes * num_envs)
        total_episodes = len(all_episode_rewards)
        mean_reward = float(np.mean(all_episode_rewards))
        std_reward = float(np.std(all_episode_rewards))
        mean_step_reward = mean_reward / self.num_eval_steps
        
        mean_power = float(np.mean(all_episode_powers)) if all_episode_powers else 0.0
        std_power = float(np.std(all_episode_powers)) if all_episode_powers else 0.0
        mean_baseline = float(np.mean(all_baseline_powers)) if all_baseline_powers else 0.0
        
        # Power ratio (agent / baseline)
        power_ratio = mean_power / mean_baseline if mean_baseline > 0 else 1.0
        
        # Per-layout aggregates
        per_layout_rewards = {
            layout: float(np.mean(rewards)) 
            for layout, rewards in layout_rewards.items()
        }
        per_layout_powers = {
            layout: float(np.mean(powers)) 
            for layout, powers in layout_powers.items()
        }
        per_layout_power_ratios = {}
        for layout in per_layout_powers:
            if layout in layout_baseline_powers and layout_baseline_powers[layout]:
                baseline = float(np.mean(layout_baseline_powers[layout]))
                if baseline > 0:
                    per_layout_power_ratios[layout] = per_layout_powers[layout] / baseline
        
        return EvalMetrics(
            mean_reward=mean_reward,
            std_reward=std_reward,
            mean_step_reward=mean_step_reward,
            mean_power=mean_power,
            std_power=std_power,
            mean_baseline_power=mean_baseline,
            power_ratio=power_ratio,
            per_layout_rewards=per_layout_rewards,
            per_layout_powers=per_layout_powers,
            per_layout_power_ratios=per_layout_power_ratios,
            num_episodes=self.num_eval_episodes,
            num_steps_per_episode=self.num_eval_steps,
        )
    
    def close(self):
        """Close evaluation environments."""
        if self._eval_envs is not None:
            self._eval_envs.close()
            self._eval_envs = None


def run_evaluation(
    actor: nn.Module,
    eval_layouts: List[str],
    env_factory: Callable[[np.ndarray, np.ndarray], gym.Env],
    combined_wrapper: Callable[[gym.Env], gym.Env],
    num_envs: int,
    num_eval_steps: int,
    num_eval_episodes: int,
    device: torch.device,
    rotor_diameter: float,
    wind_turbine: Any,
    seed: int = 42,
    max_turbines: Optional[int] = None,
    writer=None,
    global_step: int = 0,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Convenience function to run evaluation and optionally log results.
    
    This creates a temporary evaluator, runs evaluation, logs results,
    and cleans up. Use PolicyEvaluator directly if you want to reuse
    environments across multiple evaluations.
    
    Args:
        actor: Actor network to evaluate
        eval_layouts: List of layout names
        env_factory: Factory for creating base environments
        combined_wrapper: Wrapper to apply
        num_envs: Number of parallel eval environments
        num_eval_steps: Steps per episode
        num_eval_episodes: Number of episodes
        device: Torch device
        rotor_diameter: For position normalization
        wind_turbine: PyWake wind turbine
        seed: Random seed
        max_turbines: Max turbines for padding
        writer: Optional tensorboard writer
        global_step: Current training step (for logging)
        verbose: Whether to print results
    
    Returns:
        Dictionary of metrics
    """
    evaluator = PolicyEvaluator(
        actor=actor,
        eval_layouts=eval_layouts,
        env_factory=env_factory,
        combined_wrapper=combined_wrapper,
        num_envs=num_envs,
        num_eval_steps=num_eval_steps,
        num_eval_episodes=num_eval_episodes,
        device=device,
        rotor_diameter=rotor_diameter,
        wind_turbine=wind_turbine,
        seed=seed,
        max_turbines=max_turbines,
    )
    
    try:
        metrics = evaluator.evaluate()
        metrics_dict = metrics.to_dict()
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"EVALUATION RESULTS (Step {global_step})")
            print(f"{'='*60}")
            print(f"Episodes: {metrics.num_episodes}, Steps/episode: {metrics.num_steps_per_episode}")
            print(f"Layouts: {eval_layouts}")
            print(f"Mean reward: {metrics.mean_reward:.4f} ± {metrics.std_reward:.4f}")
            print(f"Mean step reward: {metrics.mean_step_reward:.6f}")
            print(f"Mean power: {metrics.mean_power:.2f} W")
            print(f"Mean baseline power: {metrics.mean_baseline_power:.2f} W")
            print(f"Power ratio (agent/baseline): {metrics.power_ratio:.4f}")
            
            if metrics.per_layout_rewards:
                print(f"\nPer-layout results:")
                for layout in metrics.per_layout_rewards:
                    reward = metrics.per_layout_rewards.get(layout, 0)
                    power = metrics.per_layout_powers.get(layout, 0)
                    ratio = metrics.per_layout_power_ratios.get(layout, 0)
                    print(f"  {layout}: reward={reward:.4f}, power={power:.2f}, ratio={ratio:.4f}")
            print(f"{'='*60}\n")
        
        if writer is not None:
            for name, value in metrics_dict.items():
                writer.add_scalar(name, value, global_step)
        
        return metrics_dict
        
    finally:
        evaluator.close()


if __name__ == "__main__":
    # Quick test of the evaluation utilities
    print("Evaluation utilities module loaded successfully.")
    print("Use PolicyEvaluator class or run_evaluation function for evaluation.")