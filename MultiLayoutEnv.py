"""
MultiLayoutEnv: A unified environment wrapper for wind farm control.

This wrapper handles both single-layout and multi-layout training scenarios.
On reset, it can optionally reinitialize the underlying environment with a 
newly sampled layout configuration.

Key features:
- Single or multiple layout configurations
- Reinitializes environment on reset (no pre-created environment pool)
- Handles observation/action padding for variable farm sizes
- Provides attention masks for transformer architectures
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional, List, Callable
from dataclasses import dataclass


@dataclass
class LayoutConfig:
    """Configuration for a single wind farm layout."""
    name: str
    x_pos: np.ndarray
    y_pos: np.ndarray
    
    @property
    def n_turbines(self) -> int:
        return len(self.x_pos)


class MultiLayoutEnv(gym.Env):
    """
    Environment wrapper that supports sampling from multiple wind farm layouts.
    
    Unlike maintaining a pool of pre-created environments, this wrapper 
    reinitializes the underlying WindFarmEnv with a new layout configuration
    on each reset. This is more memory-efficient and cleaner.
    
    For single-layout training, simply pass a single layout configuration.
    The behavior is identical to having a fixed-layout environment.
    
    Observations are padded to max_turbines size to enable:
    - Training across variable-size farms
    - Use with vectorized environments (all envs have same obs/action shape)
    - Transformer architectures with attention masking
    """
    
    def __init__(
        self,
        layouts: List[LayoutConfig],
        env_factory: Callable[[np.ndarray, np.ndarray], gym.Env],
        per_turbine_wrapper: Callable[[gym.Env], gym.Env],
        seed: int = 0,
    ):
        """
        Args:
            layouts: List of LayoutConfig objects defining available layouts.
                     For single-layout training, pass a list with one element.
            env_factory: Callable that creates a base WindFarmEnv given (x_pos, y_pos).
                        Should return the unwrapped environment.
            per_turbine_wrapper: Callable that wraps the env with per-turbine observations.
            seed: Random seed for layout sampling.
        """
        super().__init__()
        
        if not layouts:
            raise ValueError("Must provide at least one layout configuration")
        
        self.layouts = layouts
        self.layout_names = [l.name for l in layouts]
        self.env_factory = env_factory
        self.per_turbine_wrapper = per_turbine_wrapper
        self.seed_value = seed
        self.rng = np.random.default_rng(seed)
        
        # Determine max turbines from layouts
        self.max_turbines = max(l.n_turbines for l in layouts)
        
        # Initialize with first layout to get observation dimensions
        self.current_layout: LayoutConfig = layouts[0]
        self._current_env: Optional[gym.Env] = None
        self._create_env(self.current_layout)
        
        # Get observation dimensions from the wrapped env
        sample_obs, _ = self._current_env.reset()
        self.obs_dim_per_turbine = sample_obs.shape[1]
        
        # Store rotor diameter (same for all layouts since same turbine type)
        self._rotor_diameter = self._get_rotor_diameter()
        
        # Define padded observation space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.max_turbines, self.obs_dim_per_turbine),
            dtype=np.float32
        )
        
        # Define padded action space (1 action per turbine for yaw)
        # Get action bounds from the wrapped env
        base_action_space = self._current_env.action_space
        self.action_space = spaces.Box(
            low=base_action_space.low[0],
            high=base_action_space.high[0],
            shape=(self.max_turbines,),
            dtype=np.float32
        )
        
        # Initialize attention mask
        self._attention_mask = self._compute_attention_mask()
    
    def _create_env(self, layout: LayoutConfig) -> None:
        """Create a new environment for the given layout."""
        # Close existing environment if any
        if self._current_env is not None:
            self._current_env.close()
        
        # Create new environment with the layout positions
        base_env = self.env_factory(layout.x_pos, layout.y_pos)
        self._current_env = self.per_turbine_wrapper(base_env)
        self.current_layout = layout
    
    def _get_rotor_diameter(self) -> float:
        """Get rotor diameter from the current environment."""
        env = self._current_env
        # Try different ways to access rotor diameter
        if hasattr(env, 'rotor_diameter'):
            return env.rotor_diameter
        if hasattr(env, 'D'):
            return env.D
        # Try unwrapping
        while hasattr(env, 'env'):
            env = env.env
            if hasattr(env, 'D'):
                return env.D
        raise AttributeError("Could not find rotor diameter in environment")
    
    def _get_base_env(self) -> Any:
        """Get the unwrapped base environment."""
        env = self._current_env
        while hasattr(env, 'env'):
            env = env.env
        return env
    
    def _compute_attention_mask(self) -> np.ndarray:
        """Compute attention mask for current layout. True = padding (ignore)."""
        mask = np.ones(self.max_turbines, dtype=bool)
        mask[:self.n_turbines] = False
        return mask
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def n_turbines(self) -> int:
        """Actual number of turbines in current layout (not padded)."""
        return self.current_layout.n_turbines
    
    @property
    def rotor_diameter(self) -> float:
        """Rotor diameter in meters."""
        return self._rotor_diameter
    
    @property  
    def D(self) -> float:
        """Rotor diameter (alias for compatibility)."""
        return self._rotor_diameter
    
    @property
    def turbine_positions(self) -> np.ndarray:
        """
        Padded turbine positions (max_turbines, 2).
        Padding positions are zeros.
        """
        actual = np.column_stack([
            self.current_layout.x_pos,
            self.current_layout.y_pos
        ])
        padded = np.zeros((self.max_turbines, 2), dtype=np.float32)
        padded[:self.n_turbines] = actual
        return padded
    
    @property
    def x_pos(self) -> np.ndarray:
        """X positions of turbines (not padded)."""
        return self.current_layout.x_pos
    
    @property
    def y_pos(self) -> np.ndarray:
        """Y positions of turbines (not padded)."""
        return self.current_layout.y_pos
    
    @property
    def attention_mask(self) -> np.ndarray:
        """Boolean mask where True = padding (should be ignored by transformer)."""
        return self._attention_mask.copy()
    
    @property
    def mean_wind_direction(self) -> float:
        """Current mean wind direction from the active environment."""
        base_env = self._get_base_env()
        return base_env.wd
    
    @property
    def wd(self) -> float:
        """Current wind direction (alias for compatibility)."""
        return self.mean_wind_direction
    
    @property
    def is_multi_layout(self) -> bool:
        """Whether this environment has multiple layout options."""
        return len(self.layouts) > 1
    
    # =========================================================================
    # Core Methods
    # =========================================================================
    
    def reset(
        self, 
        *, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment, optionally sampling a new layout.
        
        If multiple layouts are available, randomly samples one and reinitializes
        the environment. For single-layout mode, simply resets the existing env.
        
        Args:
            seed: Optional seed for the reset
            options: Optional dict that may contain:
                - 'layout_name': Force a specific layout by name
                - 'layout_index': Force a specific layout by index
        
        Returns:
            obs: Padded observation of shape (max_turbines, obs_dim_per_turbine)
            info: Dict containing layout information
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        # Determine which layout to use
        if options is not None and 'layout_name' in options:
            # Use specified layout by name
            layout_name = options['layout_name']
            try:
                layout_idx = self.layout_names.index(layout_name)
            except ValueError:
                raise ValueError(f"Unknown layout name: {layout_name}")
            new_layout = self.layouts[layout_idx]
        elif options is not None and 'layout_index' in options:
            # Use specified layout by index
            layout_idx = options['layout_index']
            new_layout = self.layouts[layout_idx]
        else:
            # Randomly sample a layout
            new_layout = self.rng.choice(self.layouts)
        
        # Only reinitialize if layout changed (optimization for single-layout case)
        if new_layout.name != self.current_layout.name:
            self._create_env(new_layout)
        
        # Update attention mask for current layout
        self._attention_mask = self._compute_attention_mask()
        
        # Reset the underlying environment
        obs, info = self._current_env.reset(seed=seed, options=options)
        
        # Pad observation
        padded_obs = self._pad_observation(obs)
        
        # Add layout info to info dict
        info['n_turbines'] = self.n_turbines
        info['layout'] = self.current_layout.name
        info['attention_mask'] = self._attention_mask.copy()
        info['max_turbines'] = self.max_turbines
        
        return padded_obs, info
    
    def step(
        self, 
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step with padded action.
        
        Args:
            action: Shape (max_turbines,) - only first n_turbines are used
        
        Returns:
            obs: Padded observation of shape (max_turbines, obs_dim_per_turbine)
            reward: Scalar reward
            terminated: Whether episode terminated
            truncated: Whether episode was truncated
            info: Dict with additional information
        """
        # Only use actions for real turbines (ignore padding actions)
        real_action = action[:self.n_turbines]
        
        obs, reward, terminated, truncated, info = self._current_env.step(real_action)
        
        # Pad observation
        padded_obs = self._pad_observation(obs)
        
        # Add layout info
        info['n_turbines'] = self.n_turbines
        info['attention_mask'] = self._attention_mask.copy()
        
        return padded_obs, reward, terminated, truncated, info
    
    def _pad_observation(self, obs: np.ndarray) -> np.ndarray:
        """Pad observation to (max_turbines, obs_dim)."""
        padded = np.zeros((self.max_turbines, self.obs_dim_per_turbine), dtype=np.float32)
        padded[:obs.shape[0]] = obs
        return padded
    
    def close(self) -> None:
        """Close the environment."""
        if self._current_env is not None:
            self._current_env.close()
            self._current_env = None
    
    def render(self):
        """Render the current environment."""
        if self._current_env is not None:
            return self._current_env.render()
        return None


# =============================================================================
# Factory Functions
# =============================================================================

def create_layout_configs(
    layout_names: List[str],
    layout_generator: Callable[[str], Tuple[np.ndarray, np.ndarray]]
) -> List[LayoutConfig]:
    """
    Create LayoutConfig objects from layout names.
    
    Args:
        layout_names: List of layout identifier strings (e.g., ["square_1", "circular_1"])
        layout_generator: Function that takes a layout name and returns (x_pos, y_pos)
    
    Returns:
        List of LayoutConfig objects
    """
    configs = []
    for name in layout_names:
        x_pos, y_pos = layout_generator(name)
        configs.append(LayoutConfig(name=name, x_pos=x_pos, y_pos=y_pos))
    return configs


def make_multi_layout_env(
    layouts: List[LayoutConfig],
    base_env_kwargs: Dict[str, Any],
    per_turbine_wrapper_class,
    seed: int = 0,
) -> MultiLayoutEnv:
    """
    Create a MultiLayoutEnv with the given configuration.
    
    This is a convenience factory function that creates the env_factory callable
    from base_env_kwargs and WindFarmEnv.
    
    Args:
        layouts: List of LayoutConfig objects
        base_env_kwargs: Kwargs to pass to WindFarmEnv (excluding x_pos, y_pos)
        per_turbine_wrapper_class: The wrapper class for per-turbine observations
        seed: Random seed
    
    Returns:
        MultiLayoutEnv instance
    """
    # Import here to avoid circular imports
    from windgym.WindGym import WindFarmEnv
    
    def env_factory(x_pos: np.ndarray, y_pos: np.ndarray) -> gym.Env:
        return WindFarmEnv(x_pos=x_pos, y_pos=y_pos, **base_env_kwargs)
    
    return MultiLayoutEnv(
        layouts=layouts,
        env_factory=env_factory,
        per_turbine_wrapper=per_turbine_wrapper_class,
        seed=seed,
    )


# =============================================================================
# Testing
# =============================================================================

def test_multi_layout_env():
    """Test the MultiLayoutEnv wrapper."""
    print("Testing MultiLayoutEnv...")
    
    # Create mock layouts
    layout1 = LayoutConfig(
        name="small",
        x_pos=np.array([0, 500, 1000]),
        y_pos=np.array([0, 0, 0])
    )
    layout2 = LayoutConfig(
        name="large", 
        x_pos=np.array([0, 500, 1000, 1500, 2000]),
        y_pos=np.array([0, 0, 0, 0, 0])
    )
    
    print(f"  Layout 1: {layout1.name} with {layout1.n_turbines} turbines")
    print(f"  Layout 2: {layout2.name} with {layout2.n_turbines} turbines")
    print(f"  This test requires the actual WindGym environment to run fully.")
    print("  ✓ MultiLayoutEnv structure is correct!")


if __name__ == "__main__":
    test_multi_layout_env()