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
- Pads info dict arrays for AsyncVectorEnv compatibility
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
    receptivity_profiles: Optional[np.ndarray] = None  # Shape: (n_turbines, n_directions)
    influence_profiles: Optional[np.ndarray] = None  # Shape: (n_turbines, n_directions)
    
    @property
    def n_turbines(self) -> int:
        return len(self.x_pos)


    @property
    def has_profiles(self) -> bool: # If we have one, we have both
        return self.receptivity_profiles is not None
    
    @property
    def n_profile_directions(self) -> int:
        if self.receptivity_profiles is None:
            return 0
        return self.receptivity_profiles.shape[1]


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
        pad_value: float = 0.0,
        shuffle: bool = False,
        max_turbines: Optional[int] = None,
        max_episode_steps: Optional[int] = None,  
    ):
        """
        Args:
            layouts: List of LayoutConfig objects defining available layouts.
                     For single-layout training, pass a list with one element.
            env_factory: Callable that creates a base WindFarmEnv given (x_pos, y_pos).
                        Should return the unwrapped environment.
            per_turbine_wrapper: Callable that wraps the env with per-turbine observations.
            seed: Random seed for layout sampling.
            pad_value: Value to use for padding (default: 0.0).
            shuffle: If True, randomly permute turbine indices on each reset.
                    This tests whether the model learns spatial relationships
                    through attention rather than memorizing index-based patterns.
            max_turbines: Override max turbines for padding. If None, computed from layouts.
                         Use this to train on small farms but size network for larger farms.
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
        self.pad_value = pad_value
        self.shuffle = shuffle

        # Determine max turbines from layouts (or use override)
        layout_max = max(l.n_turbines for l in layouts)
        if max_turbines is not None:
            # print("Overriding max_turbines for padding")
            if max_turbines < layout_max:
                raise ValueError(
                    f"max_turbines ({max_turbines}) must be >= largest layout ({layout_max})"
                )
            self.max_turbines = max_turbines
        else:
            self.max_turbines = layout_max
        
        # Initialize with first layout to get observation dimensions
        self.current_layout: LayoutConfig = layouts[0]
        self._current_env: Optional[gym.Env] = None
        self._create_env(self.current_layout)
        
        # Initialize permutation arrays (identity until first reset with shuffle)
        self._perm = np.arange(self.current_layout.n_turbines)
        self._inv_perm = np.arange(self.current_layout.n_turbines)

        # Get observation dimensions
        self.obs_dim_per_turbine = self._get_obs_dim_from_env()
        # print("FROM INSIDE MULTILAYOUT ENV, obs_dim_per_turbine =", self.obs_dim_per_turbine)
        
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

        self.max_episode_steps = max_episode_steps
        self._elapsed_steps = 0
    
    def _create_env(self, layout: LayoutConfig) -> None:
        """Create a new environment for the given layout."""
        # print(f"We call _create_env with layout: {layout.name}")
        # Close existing environment if any
        if self._current_env is not None:
            self._current_env.close()
        
        # Create new environment with the layout positions
        base_env = self.env_factory(layout.x_pos, layout.y_pos)
        self._current_env = self.per_turbine_wrapper(base_env)
        self.current_layout = layout
        
        # Reset permutation to identity (will be shuffled on reset if shuffle=True)
        self._perm = np.arange(layout.n_turbines)
        self._inv_perm = np.arange(layout.n_turbines)
    
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

    def _get_obs_dim_from_env(self) -> int:
        """Get observation dimension per turbine without calling reset().

        Traverses wrapper chain to find _obs_dim_per_turbine (from
        PerTurbineObservationWrapper) or get_obs_dim_per_turbine() (from
        WindFarmEnv). This enables lazy initialization.
        """
        env = self._current_env
        # Check wrapper first (PerTurbineObservationWrapper stores it)
        while hasattr(env, 'env'):
            if hasattr(env, '_obs_dim_per_turbine'):
                return env._obs_dim_per_turbine
            env = env.env
        # Check base env
        if hasattr(env, 'get_obs_dim_per_turbine'):
            return env.get_obs_dim_per_turbine()
        raise AttributeError(
            "Cannot determine obs_dim without reset. Either provide "
            "obs_dim_per_turbine explicitly or set skip_initial_reset=False."
        )
    
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
    # Info Dict Padding (fixes AsyncVectorEnv compatibility)
    # =========================================================================
    
    # Keys that are known to be per-turbine arrays (shape n_turb,)
    # These will be padded to max_turbines for consistency across layouts
    _PER_TURBINE_KEYS = {
        "yaw angles agent",
        "Wind speed at turbines",
        "Wind direction at turbines",
        "Power pr turbine agent",
        "Turbine x positions",
        "Turbine y positions",
        "Turbulence intensity at turbines",
        "yaw angles base",
        "Power pr turbine baseline",
        "Wind speed at turbines baseline",
    }
    
    # Keys that are per-turbine but flattened with history (shape n_turb * history,)
    # These need reshaping before padding
    _PER_TURBINE_HISTORY_KEYS = {
        "yaw angles measured",
        "Wind speed at turbines measured",
        "Wind direction at turbines measured",
    }
    
    # Keys that are 2D time-series arrays (shape T, n_turb)
    _TIMESERIES_KEYS = {
        "windspeeds",
        "winddirs",
        "yaws",
        "powers",
        "baseline_powers",
        "yaws_baseline",
        "windspeeds_baseline",
    }
    
    def _pad_info(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle info dict arrays for AsyncVectorEnv compatibility.
        
        Uses KEY-BASED detection to identify per-turbine arrays for CONSISTENT
        behavior across different layouts. Shape-based detection caused issues
        when array lengths coincidentally matched n_turb for some layouts.
        
        Strategy:
        - Known per-turbine keys: pad to max_turbines
        - Known per-turbine-history keys: reshape, pad, flatten
        - Known 2D time-series keys: pad second dimension
        - Everything else: convert to list (safe fallback)
        
        Converting to list prevents AsyncVectorEnv from trying to stack arrays
        with inconsistent shapes.
        """
        padded_info = {}
        n_turb = self.n_turbines
        
        for key, value in info.items():
            if not isinstance(value, np.ndarray):
                # Non-arrays pass through unchanged
                padded_info[key] = value
            elif key in self._PER_TURBINE_KEYS:
                # Known per-turbine array: pad to max_turbines
                padded_info[key] = self._pad_1d_to_max(value)
            elif key in self._PER_TURBINE_HISTORY_KEYS:
                # Known flattened per-turbine array with history
                padded_info[key] = self._pad_flattened_per_turbine(value, n_turb)
            elif key in self._TIMESERIES_KEYS and value.ndim == 2:
                # Known 2D time-series: pad second dimension
                padded_info[key] = self._pad_2d_timeseries(value)
            else:
                # Unknown array: convert to list for safe fallback
                # This ensures consistent behavior across layouts
                padded_info[key] = value.tolist()
                
        return padded_info
    
    def _pad_1d_to_max(self, arr: np.ndarray) -> np.ndarray:
        """Pad 1D array from (n_turb,) to (max_turbines,)."""
        if arr.shape[0] >= self.max_turbines:
            return arr[:self.max_turbines]  # Truncate if somehow larger
        pad_width = self.max_turbines - arr.shape[0]
        return np.pad(arr, (0, pad_width), constant_values=self.pad_value)
    
    def _pad_flattened_per_turbine(self, arr: np.ndarray, n_turb: int) -> np.ndarray:
        """
        Pad flattened per-turbine array from (n_turb * features,) to (max_turb * features,).
        """
        if arr.ndim != 1 or len(arr) == 0:
            return arr.tolist()  # Fallback
            
        arr_len = arr.shape[0]
        if arr_len % n_turb != 0:
            return arr.tolist()  # Can't reshape, fallback
            
        features_per_turb = arr_len // n_turb
        
        # Reshape to (n_turb, features)
        reshaped = arr.reshape(n_turb, features_per_turb)
        
        # Pad in turbine dimension
        if n_turb < self.max_turbines:
            pad_width = self.max_turbines - n_turb
            padded = np.pad(reshaped, ((0, pad_width), (0, 0)), 
                           constant_values=self.pad_value)
        else:
            padded = reshaped[:self.max_turbines]  # Truncate if larger
        
        # Flatten back to 1D
        return padded.flatten()
    
    def _pad_2d_timeseries(self, arr: np.ndarray) -> np.ndarray:
        """Pad 2D time-series array from (T, n_turb) to (T, max_turbines)."""
        if arr.ndim != 2:
            return arr.tolist()  # Fallback
            
        n_turb_actual = arr.shape[1]
        if n_turb_actual >= self.max_turbines:
            return arr[:, :self.max_turbines]  # Truncate if larger
            
        pad_width = self.max_turbines - n_turb_actual
        return np.pad(arr, ((0, 0), (0, pad_width)), constant_values=self.pad_value)
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def current_layout_index(self) -> int:
        """Index of current layout in self.layouts list."""
        for i, layout in enumerate(self.layouts):
            if layout.name == self.current_layout.name:
                return i
        raise ValueError(f"Current layout '{self.current_layout.name}' not found in layouts list")

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
        Positions are shuffled if shuffle=True to match observation order.
        Padding positions are zeros.
        """
        actual = np.column_stack([
            self.current_layout.x_pos,
            self.current_layout.y_pos
        ])
        # Apply shuffle to match observation order
        actual = actual[self._perm]
        padded = np.zeros((self.max_turbines, 2), dtype=np.float32)
        padded[:self.n_turbines] = actual
        return padded
    
    @property
    def x_pos(self) -> np.ndarray:
        """X positions of turbines (shuffled if shuffle=True, not padded)."""
        return self.current_layout.x_pos[self._perm]
    
    @property
    def y_pos(self) -> np.ndarray:
        """Y positions of turbines (shuffled if shuffle=True, not padded)."""
        return self.current_layout.y_pos[self._perm]
    
    @property
    def x_pos_original(self) -> np.ndarray:
        """Original (unshuffled) X positions of turbines."""
        return self.current_layout.x_pos
    
    @property
    def y_pos_original(self) -> np.ndarray:
        """Original (unshuffled) Y positions of turbines."""
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
    def ws(self) -> float:
        """Current wind speed from the active environment."""
        return self._get_base_env().ws

    @property
    def ti(self) -> float:
        """Current turbulence intensity from the active environment."""
        return self._get_base_env().ti

    @property
    def current_yaw(self) -> np.ndarray:
        """Current yaw angles (not padded, in original order)."""
        return self._get_base_env().current_yaw

    @property
    def ActionMethod(self) -> str:
        """Action method from the active environment ('wind' or 'yaw')."""
        return self._get_base_env().ActionMethod

    @property
    def yaw_step_env(self) -> float:
        """Yaw step per environment step (for 'yaw' ActionMethod)."""
        return self._get_base_env().yaw_step_env

    @property
    def is_multi_layout(self) -> bool:
        """Whether this environment has multiple layout options."""
        return len(self.layouts) > 1
    
    @property
    def is_shuffled(self) -> bool:
        """Whether turbine indices are currently shuffled."""
        return self.shuffle and not np.array_equal(self._perm, np.arange(self.n_turbines))
    
    # @property
    # def current_permutation(self) -> np.ndarray:
    #     """
    #     Current turbine permutation array.
        
    #     Maps from shuffled index to original index:
    #     original_idx = current_permutation[shuffled_idx]
        
    #     If shuffle=False, this is identity [0, 1, 2, ...].
    #     """
    #     return self._perm.copy()
    
    @property
    def inverse_permutation(self) -> np.ndarray:
        """
        Inverse permutation array.
        
        Maps from original index to shuffled index:
        shuffled_idx = inverse_permutation[original_idx]
        """
        return self._inv_perm.copy()
    
    @property
    def receptivity_profiles(self) -> Optional[np.ndarray]:
        """
        Padded receptivity profiles for current layout.
        
        Returns:
            Shape (max_turbines, n_directions) with padding rows as zeros,
            or None if profiles not available.
            Profiles are shuffled to match observation order if shuffle=True.
        """
        if not self.current_layout.has_profiles:
            return None
            
        profiles = self.current_layout.receptivity_profiles
        
        # Apply shuffle to match observation order
        profiles = profiles[self._perm]
        
        # Pad to max_turbines
        n_dirs = profiles.shape[1]
        padded = np.zeros((self.max_turbines, n_dirs), dtype=np.float32)
        padded[:self.n_turbines] = profiles
        
        return padded

    @property
    def influence_profiles(self) -> Optional[np.ndarray]:
        """
        Padded influence profiles for current layout.
        
        Returns:
            Shape (max_turbines, n_directions) with padding rows as zeros,
            or None if profiles not available.
            Profiles are shuffled to match observation order if shuffle=True.
        """
        if not self.current_layout.has_profiles:
            return None
            
        profiles = self.current_layout.influence_profiles
        
        # Apply shuffle to match observation order
        profiles = profiles[self._perm]
        
        # Pad to max_turbines
        n_dirs = profiles.shape[1]
        padded = np.zeros((self.max_turbines, n_dirs), dtype=np.float32)
        padded[:self.n_turbines] = profiles
        
        return padded

    @property
    def current_permutation(self) -> np.ndarray:
        """Return permutation padded to max_turbines (identity for padding slots)."""
        padded = np.arange(self.max_turbines, dtype=np.int64)
        padded[:self.n_turbines] = self._perm
        return padded

    @property
    def n_profile_directions(self) -> int:
        """Number of directions in receptivity profiles (typically 360)."""
        if self.current_layout.has_profiles:
            return self.current_layout.n_profile_directions
        return 0
    
    @property
    def has_receptivity_profiles(self) -> bool:
        """Whether current layout has receptivity profiles."""
        return self.current_layout.has_profiles

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
            info: Dict containing layout information (with padded arrays)
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
            # print(f"Resetting environment with layout: {new_layout.name}")
            self._create_env(new_layout)
        
        # Generate new shuffle permutation if shuffle is enabled
        if self.shuffle:
            # print("Shuffling turbine indices on reset")
            self._perm = self.rng.permutation(self.n_turbines)
            self._inv_perm = np.argsort(self._perm)
        else:
            self._perm = np.arange(self.n_turbines)
            self._inv_perm = np.arange(self.n_turbines)
        
        # Update attention mask for current layout
        self._attention_mask = self._compute_attention_mask()
        
        # Reset the underlying environment
        obs, info = self._current_env.reset(seed=seed, options=options)
        
        # Apply shuffle to observation (permute turbine dimension)
        obs = obs[self._perm]
        
        # Pad observation
        padded_obs = self._pad_observation(obs)
        
        # Pad info dict arrays for AsyncVectorEnv compatibility
        info = self._pad_info(info)
        
        # Add layout info to info dict
        info['n_turbines'] = self.n_turbines
        info['layout'] = self.current_layout.name
        info['attention_mask'] = self._attention_mask.copy()
        info['max_turbines'] = self.max_turbines
        if self.shuffle:
            # print("Adding turbine_permutation to info dict")
            # Pad turbine_permutation to max_turbines for AsyncVectorEnv compatibility
            # Use -1 for padding values (valid indices are 0 to n_turbines-1)
            padded_perm = np.full(self.max_turbines, -1, dtype=np.int64)
            padded_perm[:self.n_turbines] = self._perm
            info['turbine_permutation'] = padded_perm
        
        self._elapsed_steps = 0

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
            info: Dict with additional information (with padded arrays)
        """
        # Only use actions for real turbines (ignore padding actions)
        real_action = action[:self.n_turbines]
        
        # Unshuffle action to match underlying env's turbine order
        # If shuffle is off, _inv_perm is identity so this is a no-op
        unshuffled_action = real_action[self._inv_perm]
        
        obs, reward, terminated, truncated, info = self._current_env.step(unshuffled_action)
        
        # Apply shuffle to observation (permute turbine dimension)
        obs = obs[self._perm]
        
        # Pad observation
        padded_obs = self._pad_observation(obs)
        
        # Pad info dict arrays for AsyncVectorEnv compatibility
        info = self._pad_info(info)
        
        # Add layout info
        info['n_turbines'] = self.n_turbines
        info['attention_mask'] = self._attention_mask.copy()

        self._elapsed_steps += 1
        if self.max_episode_steps is not None and self._elapsed_steps >= self.max_episode_steps:
            truncated = True

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
    
