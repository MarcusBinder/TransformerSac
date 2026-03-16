"""
TurbineFailureEnv: Environment wrapper for simulating turbine failures.

This module provides a clean way to simulate turbine failures in wind farm
environments while maintaining correct wake physics. Unlike simple masking
approaches, this wrapper creates sub-environments with only active turbines,
ensuring that failed turbines don't generate wakes in the simulation.

Key Features:
- Correct wake physics (failed turbines are removed from simulation)
- Consistent observation/action interface (always full farm size)
- Support for fixed, random, and scheduled failure modes
- Proper attention masks for transformer architectures
- Compatible with both transformer and MLP controllers

Usage:
    # Fixed failures (for evaluation)
    config = FailureConfig(mode='fixed', failed_indices=[0, 3])
    env = TurbineFailureEnv(x_pos, y_pos, env_factory, wrapper, config)
    
    # Random failures (for training)
    config = FailureConfig(mode='random', failure_prob=0.15, max_failures=2)
    env = TurbineFailureEnv(x_pos, y_pos, env_factory, wrapper, config)

Author: Marcus (DTU Wind Energy)
For Conference Paper 2: Robustness to Turbine Failures
"""

import gymnasium as gym
import numpy as np
from typing import List, Optional, Dict, Any, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from MultiLayoutEnv import LayoutConfig

# =============================================================================
# Configuration
# =============================================================================

class FailureMode(Enum):
    """Supported failure simulation modes."""
    NONE = "none"           # No failures (baseline)
    FIXED = "fixed"         # Specific turbines always fail
    RANDOM = "random"       # Random failures each episode
    SCHEDULED = "scheduled" # Failures occur at specific timesteps (future)


@dataclass
class FailureConfig:
    """
    Configuration for turbine failure simulation.
    
    Attributes:
        mode: How failures are determined ('none', 'fixed', 'random', 'scheduled')
        failed_indices: For 'fixed' mode - which turbines are always failed
        failure_prob: For 'random' mode - probability each turbine fails (per episode)
        max_failures: For 'random' mode - maximum simultaneous failures
        min_active: Minimum turbines that must remain active (safety limit)
        seed: Random seed for reproducibility in 'random' mode
    
    Examples:
        # No failures (baseline evaluation)
        config = FailureConfig(mode='none')
        
        # Always fail turbines 0 and 3
        config = FailureConfig(mode='fixed', failed_indices=[0, 3])
        
        # Random failures during training
        config = FailureConfig(
            mode='random',
            failure_prob=0.15,
            max_failures=2,
            min_active=3
        )
    """
    mode: str = 'none'
    failed_indices: List[int] = field(default_factory=list)
    failure_prob: float = 0.1
    max_failures: int = 2
    min_active: int = 1
    seed: Optional[int] = None
    
    def __post_init__(self):
        # Validate mode
        valid_modes = [m.value for m in FailureMode]
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid mode '{self.mode}'. Must be one of {valid_modes}")
        
        # Ensure failed_indices is a list
        if self.failed_indices is None:
            self.failed_indices = []
        
        # Validate probabilities
        if not 0.0 <= self.failure_prob <= 1.0:
            raise ValueError(f"failure_prob must be in [0, 1], got {self.failure_prob}")
        
        if self.max_failures < 0:
            raise ValueError(f"max_failures must be >= 0, got {self.max_failures}")
        
        if self.min_active < 1:
            raise ValueError(f"min_active must be >= 1, got {self.min_active}")


# =============================================================================
# Position Classification Utilities
# =============================================================================

def classify_turbine_positions(
    positions: np.ndarray,
    wind_direction: float,
) -> List[Dict[str, Any]]:
    """
    Classify all turbine positions relative to wind direction.
    
    Args:
        positions: (n_turbines, 2) array of (x, y) positions
        wind_direction: Wind direction in degrees (meteorological convention)
    
    Returns:
        List of dicts with classification for each turbine:
        - 'streamwise': 'upwind', 'middle', or 'downwind'
        - 'lateral': 'center' or 'edge'
        - 'row': Row index (0 = most upwind)
        - 'x_wind_relative': X position in wind-relative frame
        - 'y_wind_relative': Y position in wind-relative frame
    """
    import math
    
    n_turb = len(positions)
    
    # Transform to wind-relative coordinates
    angle_offset = wind_direction - 270.0
    theta = math.radians(angle_offset)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    
    x_wr = positions[:, 0] * cos_t - positions[:, 1] * sin_t
    y_wr = positions[:, 0] * sin_t + positions[:, 1] * cos_t
    
    # Rank by streamwise position (higher x_wr = more upwind)
    x_rank = np.argsort(np.argsort(-x_wr))  # 0 = most upwind
    
    # Rank by lateral distance from center
    y_abs = np.abs(y_wr - np.mean(y_wr))
    y_rank = np.argsort(y_abs)  # 0 = most central
    
    classifications = []
    for i in range(n_turb):
        # Streamwise classification (thirds)
        if x_rank[i] < n_turb / 3:
            streamwise = 'upwind'
        elif x_rank[i] >= 2 * n_turb / 3:
            streamwise = 'downwind'
        else:
            streamwise = 'middle'
        
        # Lateral classification (halves)
        lateral_rank = np.where(y_rank == i)[0][0]
        lateral = 'center' if lateral_rank < n_turb / 2 else 'edge'
        
        classifications.append({
            'streamwise': streamwise,
            'lateral': lateral,
            'row': int(x_rank[i]),
            'x_wind_relative': float(x_wr[i]),
            'y_wind_relative': float(y_wr[i]),
        })
    
    return classifications


def get_position_label(classification: Dict[str, Any]) -> str:
    """Get a human-readable label for a turbine position."""
    return f"{classification['streamwise']}_{classification['lateral']}"


# =============================================================================
# Main Environment Wrapper
# =============================================================================

class TurbineFailureEnv(gym.Env):
    """
    Environment wrapper that simulates turbine failures with correct wake physics.
    
    This wrapper dynamically creates sub-environments containing only active
    turbines, ensuring that failed turbines don't generate wakes. The observation
    and action spaces always correspond to the FULL farm, with zeros and masks
    for failed turbines.
    
    Architecture:
    ```
    Full Layout (n_full turbines)
           │
           ▼
    ┌─────────────────────────┐
    │  TurbineFailureEnv      │
    │  - Determines failures  │
    │  - Maps obs/actions     │
    └─────────────────────────┘
           │
           ▼ (only active turbine positions)
    ┌─────────────────────────┐
    │  Sub-Environment        │
    │  (n_active turbines)    │
    │  - Correct wake physics │
    └─────────────────────────┘
    ```
    
    Observation Mapping:
    - Sub-env returns obs of shape (n_active, obs_dim)
    - Wrapper maps to (n_full, obs_dim) with zeros for failed turbines
    
    Action Mapping:
    - Wrapper receives action of shape (n_full,) or (n_full, action_dim)
    - Only actions for active turbines are passed to sub-env
    
    Attributes:
        n_turbines_full: Total number of turbines in full layout
        n_active: Number of currently active turbines
        n_failed: Number of currently failed turbines
        active_indices: List of active turbine indices
        failed_indices: List of failed turbine indices
        attention_mask: Boolean mask where True = failed (for transformers)
    """
    
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    def __init__(
        self,
        full_x_pos: np.ndarray,
        full_y_pos: np.ndarray,
        env_factory: Callable[[np.ndarray, np.ndarray], gym.Env],
        per_turbine_wrapper: Callable[[gym.Env], gym.Env],
        failure_config: FailureConfig,
        seed: Optional[int] = None,
        layout_name: str = "custom_layout",
    ):
        """
        Initialize the TurbineFailureEnv.
        
        Args:
            full_x_pos: X positions of ALL turbines in the complete layout
            full_y_pos: Y positions of ALL turbines in the complete layout
            env_factory: Callable that creates a base WindFarmEnv given (x_pos, y_pos).
                        Should NOT include the per-turbine wrapper.
            per_turbine_wrapper: Callable that wraps the env with per-turbine observations.
                                This is applied after creating the sub-environment.
            failure_config: Configuration for failure simulation
            seed: Random seed (overrides failure_config.seed if provided)
        
        Example:
            def env_factory(x, y):
                return WindFarmEnv(x_pos=x, y_pos=y, **base_kwargs)
            
            def wrapper(env):
                env = PerTurbineObservationWrapper(env)
                env = EnhancedPerTurbineWrapper(env)
                return env
            
            config = FailureConfig(mode='fixed', failed_indices=[0])
            env = TurbineFailureEnv(x, y, env_factory, wrapper, config)
        """
        super().__init__()
        
        # Store full layout
        self.full_x_pos = np.array(full_x_pos, dtype=np.float64)
        self.full_y_pos = np.array(full_y_pos, dtype=np.float64)
        self.n_turbines_full = len(full_x_pos)
        self.layout_name = layout_name  
        
        if len(full_y_pos) != self.n_turbines_full:
            raise ValueError("x_pos and y_pos must have same length")
        
        # Store factory functions
        self.env_factory = env_factory
        self.per_turbine_wrapper = per_turbine_wrapper
        
        # Failure configuration
        self.failure_config = failure_config
        
        # Validate failure indices
        for idx in failure_config.failed_indices:
            if idx < 0 or idx >= self.n_turbines_full:
                raise ValueError(
                    f"Invalid failure index {idx}. Must be in [0, {self.n_turbines_full - 1}]"
                )
        
        # Initialize random state
        seed = seed if seed is not None else failure_config.seed
        self.rng = np.random.default_rng(seed)
        self._seed = seed
        
        # Current state (will be set in reset)
        self._failed_indices: List[int] = []
        self._active_indices: List[int] = list(range(self.n_turbines_full))
        self._current_env: Optional[gym.Env] = None
        
        # Create initial environment with all turbines to get dimensions
        self._create_sub_env(self._active_indices)
        
        # Get observation dimensions from the wrapped sub-environment
        sample_obs, _ = self._current_env.reset()
        self.obs_dim_per_turbine = sample_obs.shape[1]
        
        # Cache rotor diameter (same for all turbines)
        self._rotor_diameter = self._extract_rotor_diameter()
        
        # Define observation and action spaces for FULL farm
        # This ensures consistent interface regardless of failures
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_turbines_full, self.obs_dim_per_turbine),
            dtype=np.float32
        )
        
        # Action space: one action per turbine (yaw)
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_turbines_full,),
            dtype=np.float32
        )
        
        # Track episode statistics
        self._episode_step = 0
        self._episode_power_sum = 0.0
    
    # =========================================================================
    # Internal Methods
    # =========================================================================
    
    def _create_sub_env(self, active_indices: List[int]) -> None:
        """
        Create a new sub-environment with only active turbines.
        
        This is called when the set of active turbines changes.
        The sub-environment will have correct wake physics since
        failed turbines are not included in the simulation.
        
        Args:
            active_indices: List of turbine indices that are active
        """
        # Close existing environment
        if self._current_env is not None:
            try:
                self._current_env.close()
            except Exception:
                pass  # Ignore close errors
        
        # Update state
        self._active_indices = list(active_indices)
        self._failed_indices = [
            i for i in range(self.n_turbines_full) 
            if i not in active_indices
        ]
        
        # Get positions for active turbines only
        active_x = self.full_x_pos[active_indices]
        active_y = self.full_y_pos[active_indices]
        
        # Create base environment with only active turbines
        base_env = self.env_factory(active_x, active_y)
        
        # Apply per-turbine wrapper
        self._current_env = self.per_turbine_wrapper(base_env)
    
    def _extract_rotor_diameter(self) -> float:
        """Extract rotor diameter from the current environment."""
        env = self._current_env
        
        # Unwrap to find rotor diameter
        while hasattr(env, 'env'):
            env = env.env
            if hasattr(env, 'D'):
                return getattr(env, 'D')
        
        raise AttributeError(
            "Could not find rotor diameter in environment. "
            "Expected attribute 'D' or 'rotor_diameter'."
        )
    
    def _get_base_env(self):
        """Get the unwrapped base WindFarmEnv."""
        env = self._current_env
        while hasattr(env, 'env'):
            env = env.env
        return env
    
    def _determine_failures(self) -> List[int]:
        """
        Determine which turbines fail this episode based on configuration.
        
        Returns:
            List of turbine indices that should fail
        """
        config = self.failure_config
        mode = config.mode
        
        if mode == 'none':
            return []
        
        elif mode == 'fixed':
            return list(config.failed_indices)
        
        elif mode == 'random':
            # Sample failures with per-turbine probability
            candidates = []
            for i in range(self.n_turbines_full):
                if self.rng.random() < config.failure_prob:
                    candidates.append(i)
            
            # Cap at max_failures
            if len(candidates) > config.max_failures:
                candidates = list(
                    self.rng.choice(candidates, config.max_failures, replace=False)
                )
            
            # Ensure min_active turbines remain
            max_can_fail = self.n_turbines_full - config.min_active
            if len(candidates) > max_can_fail:
                candidates = list(
                    self.rng.choice(candidates, max_can_fail, replace=False)
                )
            
            return sorted(candidates)
        
        elif mode == 'scheduled':
            # TODO: Implement scheduled failures (failures at specific timesteps)
            raise NotImplementedError("Scheduled failures not yet implemented")
        
        else:
            raise ValueError(f"Unknown failure mode: {mode}")
    
    def _map_obs_to_full(self, active_obs: np.ndarray) -> np.ndarray:
        """
        Map observation from active turbines to full farm size.
        
        Args:
            active_obs: (n_active, obs_dim) observation from sub-environment
        
        Returns:
            full_obs: (n_full, obs_dim) with zeros for failed turbines
        """
        full_obs = np.zeros(
            (self.n_turbines_full, self.obs_dim_per_turbine),
            dtype=np.float32
        )
        
        for new_idx, orig_idx in enumerate(self._active_indices):
            full_obs[orig_idx] = active_obs[new_idx]
        
        return full_obs
    
    def _map_action_to_active(self, full_action: np.ndarray) -> np.ndarray:
        """
        Map action from full farm size to active turbines only.
        
        Args:
            full_action: (n_full,) or (n_full, action_dim) action for all turbines
        
        Returns:
            active_action: (n_active,) or (n_active, action_dim) for active turbines
        """
        # Handle both 1D and 2D action arrays
        if full_action.ndim == 1:
            return full_action[self._active_indices]
        else:
            return full_action[self._active_indices, :]
    
    def _compute_attention_mask(self) -> np.ndarray:
        """
        Compute attention mask for transformer architectures.
        
        Returns:
            mask: (n_full,) boolean array where True = failed (should be masked)
        """
        mask = np.zeros(self.n_turbines_full, dtype=bool)
        for idx in self._failed_indices:
            mask[idx] = True
        return mask
    
    def _build_info_dict(self, sub_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build the info dictionary with failure information.
        
        Args:
            sub_info: Info dict from sub-environment
        
        Returns:
            Enhanced info dict with failure metadata
        """
        info = sub_info.copy()
        
        # Core failure information
        info['n_turbines_full'] = self.n_turbines_full
        info['n_active'] = self.n_active
        info['n_failed'] = self.n_failed
        info['failed_indices'] = self._failed_indices.copy()
        info['active_indices'] = self._active_indices.copy()
        info['attention_mask'] = self._compute_attention_mask()
        
        # Position classification (if wind direction available)
        if hasattr(self, '_last_wind_direction'):
            info['failed_positions'] = [
                classify_turbine_positions(
                    self.turbine_positions,
                    self._last_wind_direction
                )[idx]
                for idx in self._failed_indices
            ]
        
        return info
    
    def _pad_info_arrays(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Pad arrays in info dict from sub-env size to full farm size.
        
        This ensures consistency when comparing across different failure scenarios.
        """
        # Keys that are per-turbine arrays from the sub-environment
        per_turbine_keys = {
            'yaw angles agent',
            'Wind speed at turbines',
            'Wind direction at turbines',
            'Power pr turbine agent',
            'yaw angles base',
            'Power pr turbine baseline',
            'Wind speed at turbines baseline',
        }
        
        padded_info = {}
        
        for key, value in info.items():
            if key in per_turbine_keys and isinstance(value, np.ndarray):
                # Pad array to full size
                if value.ndim == 1 and len(value) == self.n_active:
                    full_array = np.zeros(self.n_turbines_full, dtype=value.dtype)
                    for new_idx, orig_idx in enumerate(self._active_indices):
                        full_array[orig_idx] = value[new_idx]
                    padded_info[key] = full_array
                else:
                    padded_info[key] = value
            else:
                padded_info[key] = value
        
        return padded_info
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def current_layout(self):
        # Return a simple object with .name attribute
        from types import SimpleNamespace
        return SimpleNamespace(name=self.layout_name)

    @property
    def n_turbines(self) -> int:
        """Total number of turbines in the full layout."""
        return self.n_turbines_full
    
    @property
    def n_active(self) -> int:
        """Number of currently active (non-failed) turbines."""
        return len(self._active_indices)
    
    @property
    def n_failed(self) -> int:
        """Number of currently failed turbines."""
        return len(self._failed_indices)
    
    @property
    def active_indices(self) -> List[int]:
        """Indices of active turbines in the full layout."""
        return self._active_indices.copy()
    
    @property
    def failed_indices(self) -> List[int]:
        """Indices of failed turbines in the full layout."""
        return self._failed_indices.copy()
    
    @property
    def rotor_diameter(self) -> float:
        """Rotor diameter in meters."""
        return self._rotor_diameter
    
    @property
    def D(self) -> float:
        """Rotor diameter in meters (alias for compatibility)."""
        return self._rotor_diameter
    
    @property
    def turbine_positions(self) -> np.ndarray:
        """
        Full turbine positions as (n_full, 2) array.
        
        Note: Includes positions of failed turbines.
        """
        return np.column_stack([self.full_x_pos, self.full_y_pos])
    
    @property
    def x_pos(self) -> np.ndarray:
        """X positions of all turbines (including failed)."""
        return self.full_x_pos.copy()
    
    @property
    def y_pos(self) -> np.ndarray:
        """Y positions of all turbines (including failed)."""
        return self.full_y_pos.copy()
    
    @property
    def active_positions(self) -> np.ndarray:
        """Positions of active turbines only as (n_active, 2) array."""
        return np.column_stack([
            self.full_x_pos[self._active_indices],
            self.full_y_pos[self._active_indices]
        ])
    
    @property
    def attention_mask(self) -> np.ndarray:
        """
        Attention mask for transformer architectures.
        
        Returns:
            (n_full,) boolean array where True = failed/masked
        """
        return self._compute_attention_mask()
    
    @property
    def mean_wind_direction(self) -> float:
        """Current mean wind direction from the active sub-environment."""
        return self._get_base_env().wd
    
    @property
    def wd(self) -> float:
        """Current wind direction (alias for compatibility)."""
        return self.mean_wind_direction
    
    @property
    def max_turbines(self) -> int:
        """Maximum number of turbines (same as n_turbines_full)."""
        return self.n_turbines_full
    
    # =========================================================================
    # Gym Interface
    # =========================================================================
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment, potentially with new failures.
        
        Args:
            seed: Random seed for this reset
            options: Optional dict with:
                - 'failed_indices': Override failure config for this episode
                - Other options passed to sub-environment
        
        Returns:
            observation: (n_full, obs_dim) with zeros for failed turbines
            info: Dict with failure metadata and sub-env info
        """
        # Update random state if seed provided
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        # Reset episode tracking
        self._episode_step = 0
        self._episode_power_sum = 0.0
        
        # Determine failures for this episode
        if options is not None and 'failed_indices' in options:
            # Override from options
            failed = list(options['failed_indices'])
            # Validate
            for idx in failed:
                if idx < 0 or idx >= self.n_turbines_full:
                    raise ValueError(f"Invalid failure index {idx}")
        else:
            # Use configuration
            failed = self._determine_failures()
        
        # Determine active turbines
        active = [i for i in range(self.n_turbines_full) if i not in failed]
        
        if len(active) < self.failure_config.min_active:
            raise ValueError(
                f"Too many failures requested. Would leave {len(active)} active "
                f"but min_active is {self.failure_config.min_active}"
            )
        
        # Recreate sub-environment if active set changed
        if set(active) != set(self._active_indices):
            self._create_sub_env(active)
        
        # Reset the sub-environment
        # Remove our custom options before passing to sub-env
        sub_options = None
        if options is not None:
            sub_options = {k: v for k, v in options.items() if k != 'failed_indices'}
            if not sub_options:
                sub_options = None
        
        active_obs, sub_info = self._current_env.reset(seed=seed, options=sub_options)
        
        # Cache wind direction for position classification
        self._last_wind_direction = self.mean_wind_direction
        
        # Map observation to full size
        full_obs = self._map_obs_to_full(active_obs)
        
        # Build comprehensive info dict
        info = self._pad_info_arrays(sub_info)
        info = self._build_info_dict(info)
        
        return full_obs, info
    
    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step with full-sized action.
        
        Actions for failed turbines are ignored. Only actions for active
        turbines are passed to the sub-environment.
        
        Args:
            action: (n_full,) or (n_full, action_dim) action array
        
        Returns:
            observation: (n_full, obs_dim) with zeros for failed turbines
            reward: Scalar reward from sub-environment
            terminated: Whether episode ended naturally
            truncated: Whether episode was truncated
            info: Dict with failure metadata
        """
        # Map action to active turbines only
        active_action = self._map_action_to_active(action)
        
        # Step sub-environment
        active_obs, reward, terminated, truncated, sub_info = self._current_env.step(
            active_action
        )
        
        # Update episode tracking
        self._episode_step += 1
        if 'Power agent' in sub_info:
            self._episode_power_sum += sub_info['Power agent']
        
        # Map observation to full size
        full_obs = self._map_obs_to_full(active_obs)
        
        # Build info dict
        info = self._pad_info_arrays(sub_info)
        info = self._build_info_dict(info)
        info['episode_step'] = self._episode_step
        
        return full_obs, reward, terminated, truncated, info
    
    def render(self):
        """Render the current sub-environment."""
        if self._current_env is not None:
            return self._current_env.render()
        return None
    
    def close(self) -> None:
        """Clean up resources."""
        if self._current_env is not None:
            try:
                self._current_env.close()
            except Exception:
                pass
            self._current_env = None
    
    def seed(self, seed: Optional[int] = None) -> None:
        """Set random seed."""
        self.rng = np.random.default_rng(seed)
        self._seed = seed


# =============================================================================
# Factory Functions
# =============================================================================

def make_failure_env(
    full_x_pos: np.ndarray,
    full_y_pos: np.ndarray,
    base_env_kwargs: Dict[str, Any],
    failure_config: FailureConfig,
    per_turbine_wrapper: Callable[[gym.Env], gym.Env],
    seed: int = 0,
) -> TurbineFailureEnv:
    """
    Convenience factory to create a TurbineFailureEnv.
    
    Args:
        full_x_pos: X positions of all turbines in the full layout
        full_y_pos: Y positions of all turbines in the full layout
        base_env_kwargs: Kwargs for WindFarmEnv (excluding x_pos, y_pos).
                        Must include 'turbine' key with the wind turbine object.
        failure_config: Failure simulation configuration
        per_turbine_wrapper: Wrapper function for per-turbine observations
        seed: Random seed
    
    Returns:
        Configured TurbineFailureEnv
    
    Example:
        from helper_funcs import make_env_config, EnhancedPerTurbineWrapper, get_layout_positions
        from WindGym.wrappers import PerTurbineObservationWrapper
        from py_wake.examples.data.dtu10mw import DTU10MW
        
        wind_turbine = DTU10MW()
        x_pos, y_pos = get_layout_positions("grid_3x3", wind_turbine)
        
        def combined_wrapper(env):
            env = PerTurbineObservationWrapper(env)
            env = EnhancedPerTurbineWrapper(env)
            return env
        
        config = FailureConfig(mode='fixed', failed_indices=[0, 4])
        
        env = make_failure_env(
            full_x_pos=x_pos,
            full_y_pos=y_pos,
            base_env_kwargs={**make_env_config(), 'turbine': wind_turbine},
            failure_config=config,
            per_turbine_wrapper=combined_wrapper,
        )
    """
    # Import here to avoid circular imports
    from WindGym import WindFarmEnv
    
    # Create environment factory
    def env_factory(x: np.ndarray, y: np.ndarray) -> gym.Env:
        return WindFarmEnv(x_pos=x, y_pos=y, **base_env_kwargs)
    
    return TurbineFailureEnv(
        full_x_pos=full_x_pos,
        full_y_pos=full_y_pos,
        env_factory=env_factory,
        per_turbine_wrapper=per_turbine_wrapper,
        failure_config=failure_config,
        seed=seed,
    )


def make_failure_env_vectorized(
    full_x_pos: np.ndarray,
    full_y_pos: np.ndarray,
    base_env_kwargs: Dict[str, Any],
    failure_config: FailureConfig,
    per_turbine_wrapper: Callable[[gym.Env], gym.Env],
    num_envs: int = 4,
    seed: int = 0,
) -> gym.vector.VectorEnv:
    """
    Create vectorized failure environments for parallel training/evaluation.
    
    Args:
        full_x_pos: X positions of all turbines
        full_y_pos: Y positions of all turbines
        base_env_kwargs: Kwargs for WindFarmEnv (must include 'turbine')
        failure_config: Failure simulation configuration
        per_turbine_wrapper: Wrapper for per-turbine observations
        num_envs: Number of parallel environments
        seed: Base random seed
    
    Returns:
        Vectorized environment
    """
    def make_env_fn(env_seed: int):
        def _init():
            return make_failure_env(
                full_x_pos=full_x_pos,
                full_y_pos=full_y_pos,
                base_env_kwargs=base_env_kwargs,
                failure_config=failure_config,
                per_turbine_wrapper=per_turbine_wrapper,
                seed=env_seed,
            )
        return _init
    
    return gym.vector.AsyncVectorEnv(
        [make_env_fn(seed + i) for i in range(num_envs)]
    )


# =============================================================================
# Evaluation Utilities
# =============================================================================

@dataclass
class FailureEvaluationResult:
    """Results from evaluating a model under a specific failure scenario."""
    
    # Identification
    layout: str
    failure_scenario: str
    failed_indices: List[int]
    n_failed: int
    n_active: int
    
    # Performance metrics
    power_mean: float
    power_std: float
    power_min: float
    power_max: float
    reward_mean: float
    reward_std: float
    
    # Baseline comparison (optional)
    power_baseline: Optional[float] = None
    
    # Position analysis (optional)
    failed_positions: Optional[List[Dict[str, Any]]] = None
    
    @property
    def power_retention(self) -> Optional[float]:
        """Fraction of baseline power retained (if baseline available)."""
        if self.power_baseline is not None and self.power_baseline > 0:
            return self.power_mean / self.power_baseline
        return None
    
    @property
    def power_per_active(self) -> float:
        """Average power per active turbine."""
        if self.n_active > 0:
            return self.power_mean / self.n_active
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame creation."""
        return {
            'layout': self.layout,
            'failure_scenario': self.failure_scenario,
            'failed_indices': str(self.failed_indices),
            'n_failed': self.n_failed,
            'n_active': self.n_active,
            'power_mean': self.power_mean,
            'power_std': self.power_std,
            'power_min': self.power_min,
            'power_max': self.power_max,
            'reward_mean': self.reward_mean,
            'reward_std': self.reward_std,
            'power_baseline': self.power_baseline,
            'power_retention': self.power_retention,
            'power_per_active': self.power_per_active,
        }


def evaluate_failure_scenario(
    model,
    env: TurbineFailureEnv,
    failed_indices: List[int],
    n_episodes: int = 10,
    max_steps: int = 1000,
    device: str = 'cpu',
    deterministic: bool = True,
    transform_positions_fn: Optional[Callable] = None,
) -> FailureEvaluationResult:
    """
    Evaluate a model on a specific failure scenario.
    
    Args:
        model: Policy model with get_action method
        env: TurbineFailureEnv instance
        failed_indices: Which turbines to fail
        n_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
        device: Torch device
        deterministic: Whether to use deterministic actions
        transform_positions_fn: Function to transform positions to wind-relative
    
    Returns:
        FailureEvaluationResult with performance metrics
    """
    import torch
    
    episode_powers = []
    episode_rewards = []
    
    for ep in range(n_episodes):
        obs, info = env.reset(options={'failed_indices': failed_indices})
        
        episode_power = 0.0
        episode_reward = 0.0
        
        for step in range(max_steps):
            # Prepare observation
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            
            # Get positions and transform
            positions = env.turbine_positions
            if transform_positions_fn is not None:
                positions = transform_positions_fn(
                    positions[np.newaxis],
                    np.array([env.wd])
                )[0]
            positions_tensor = torch.tensor(
                positions, dtype=torch.float32, device=device
            ).unsqueeze(0)
            
            # Get attention mask
            mask_tensor = torch.tensor(
                env.attention_mask, dtype=torch.bool, device=device
            ).unsqueeze(0)
            
            # Get action from model
            with torch.no_grad():
                action, _, _, _ = model.get_action(
                    obs_tensor,
                    positions_tensor,
                    mask_tensor,
                    deterministic=deterministic
                )
            action = action.squeeze(0).squeeze(-1).cpu().numpy()
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_power += info.get('Power agent', 0.0)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        episode_powers.append(episode_power / (step + 1))  # Average power
        episode_rewards.append(episode_reward)
    
    return FailureEvaluationResult(
        layout=env.__class__.__name__,  # Would need better tracking
        failure_scenario=f"fail_{len(failed_indices)}",
        failed_indices=failed_indices,
        n_failed=len(failed_indices),
        n_active=env.n_turbines_full - len(failed_indices),
        power_mean=float(np.mean(episode_powers)),
        power_std=float(np.std(episode_powers)),
        power_min=float(np.min(episode_powers)),
        power_max=float(np.max(episode_powers)),
        reward_mean=float(np.mean(episode_rewards)),
        reward_std=float(np.std(episode_rewards)),
    )


# =============================================================================
# Testing
# =============================================================================

def test_turbine_failure_env():
    """
    Test the TurbineFailureEnv with a mock environment.
    
    This test doesn't require WindGym - it uses a simple mock to verify
    the mapping logic works correctly.
    """
    print("Testing TurbineFailureEnv...")
    
    # Create mock environment factory
    class MockPerTurbineEnv(gym.Env):
        """Mock environment that returns per-turbine observations."""
        
        def __init__(self, x_pos, y_pos):
            self.x_pos = np.array(x_pos)
            self.y_pos = np.array(y_pos)
            self.n_turb = len(x_pos)
            self.D = 100.0  # Mock rotor diameter
            self.wd = 270.0  # Mock wind direction
            
            self.obs_dim = 4  # Mock observation dimension
            self.observation_space = gym.spaces.Box(
                low=-1, high=1, shape=(self.n_turb, self.obs_dim), dtype=np.float32
            )
            self.action_space = gym.spaces.Box(
                low=-1, high=1, shape=(self.n_turb,), dtype=np.float32
            )
        
        def reset(self, seed=None, options=None):
            obs = np.random.randn(self.n_turb, self.obs_dim).astype(np.float32)
            # Make observations identifiable by turbine index
            obs[:, 0] = np.arange(self.n_turb)
            return obs, {'Power agent': 100.0 * self.n_turb}
        
        def step(self, action):
            obs = np.random.randn(self.n_turb, self.obs_dim).astype(np.float32)
            obs[:, 0] = np.arange(self.n_turb)
            reward = float(np.sum(action))
            return obs, reward, False, False, {'Power agent': 100.0 * self.n_turb}
        
        def close(self):
            pass
    
    # Test positions (5 turbines in a row)
    full_x = np.array([0, 500, 1000, 1500, 2000], dtype=np.float64)
    full_y = np.array([0, 0, 0, 0, 0], dtype=np.float64)
    
    def mock_factory(x, y):
        return MockPerTurbineEnv(x, y)
    
    def identity_wrapper(env):
        return env
    
    # Test 1: No failures
    print("\n  Test 1: No failures...")
    config = FailureConfig(mode='none')
    env = TurbineFailureEnv(full_x, full_y, mock_factory, identity_wrapper, config)
    
    obs, info = env.reset()
    assert obs.shape == (5, 4), f"Expected (5, 4), got {obs.shape}"
    assert env.n_active == 5
    assert env.n_failed == 0
    assert np.sum(env.attention_mask) == 0
    print("    ✓ No failures test passed")
    
    # Test 2: Fixed failures
    print("\n  Test 2: Fixed failures [1, 3]...")
    config = FailureConfig(mode='fixed', failed_indices=[1, 3])
    env = TurbineFailureEnv(full_x, full_y, mock_factory, identity_wrapper, config)
    
    obs, info = env.reset()
    assert obs.shape == (5, 4), f"Expected (5, 4), got {obs.shape}"
    assert env.n_active == 3
    assert env.n_failed == 2
    assert set(env.failed_indices) == {1, 3}
    assert set(env.active_indices) == {0, 2, 4}
    
    # Check that failed turbine observations are zeros
    assert np.allclose(obs[1], 0.0), "Failed turbine 1 should have zero obs"
    assert np.allclose(obs[3], 0.0), "Failed turbine 3 should have zero obs"
    
    # Check attention mask
    assert env.attention_mask[1] == True
    assert env.attention_mask[3] == True
    assert env.attention_mask[0] == False
    print("    ✓ Fixed failures test passed")
    
    # Test 3: Action mapping
    print("\n  Test 3: Action mapping...")
    full_action = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    obs, reward, term, trunc, info = env.step(full_action)
    
    # Reward should only reflect active turbine actions
    expected_reward = 0.1 + 0.3 + 0.5  # Turbines 0, 2, 4
    assert np.isclose(reward, expected_reward), f"Expected {expected_reward}, got {reward}"
    print("    ✓ Action mapping test passed")
    
    # Test 4: Override failures via options
    print("\n  Test 4: Override failures via options...")
    obs, info = env.reset(options={'failed_indices': [0, 4]})
    assert set(env.failed_indices) == {0, 4}
    assert set(env.active_indices) == {1, 2, 3}
    print("    ✓ Override test passed")
    
    # Test 5: Random failures
    print("\n  Test 5: Random failures...")
    config = FailureConfig(mode='random', failure_prob=0.5, max_failures=2, min_active=2)
    env = TurbineFailureEnv(full_x, full_y, mock_factory, identity_wrapper, config, seed=42)
    
    failure_counts = []
    for _ in range(20):
        obs, info = env.reset()
        failure_counts.append(env.n_failed)
    
    assert max(failure_counts) <= 2, "Should not exceed max_failures"
    assert min([5 - f for f in failure_counts]) >= 2, "Should maintain min_active"
    assert len(set(failure_counts)) > 1, "Should have variation in failures"
    print(f"    Failure distribution: {failure_counts}")
    print("    ✓ Random failures test passed")
    
    # Test 6: Properties
    print("\n  Test 6: Properties...")
    config = FailureConfig(mode='fixed', failed_indices=[2])
    env = TurbineFailureEnv(full_x, full_y, mock_factory, identity_wrapper, config)
    env.reset()
    
    assert env.n_turbines == 5
    assert env.n_turbines_full == 5
    assert env.max_turbines == 5
    assert env.rotor_diameter == 100.0
    assert env.D == 100.0
    assert env.turbine_positions.shape == (5, 2)
    assert env.active_positions.shape == (4, 2)
    print("    ✓ Properties test passed")
    
    env.close()
    print("\n  All tests passed! ✓")
    return True


if __name__ == "__main__":
    test_turbine_failure_env()