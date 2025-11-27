"""
PerTurbineObservationWrapper Template

This wrapper restructures the flat observation/action spaces of WindFarmEnv
into per-turbine format for use with the transformer architecture.

REQUIRED INTERFACE (used by transformer_sac_windfarm.py):
-------------------------------------------------------
Properties:
    - n_turbines: int
    - turbine_positions: np.ndarray of shape (n_turbines, 2)
    - rotor_diameter: float
    - mean_wind_direction: float

Methods:
    - reset() -> obs of shape (n_turbines, obs_dim_per_turbine)
    - step(action) -> obs of shape (n_turbines, obs_dim_per_turbine)

The action input to step() will be shape (n_turbines, action_dim_per_turbine)
and needs to be flattened before passing to the base environment.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional


class PerTurbineObservationWrapper(gym.Wrapper):
    """
    Wrapper that restructures WindFarmEnv observations and actions 
    from flat vectors to per-turbine format.
    
    Base env observation (flat): [t0_ws, t0_wd, t0_yaw, t1_ws, t1_wd, t1_yaw, ...]
    Wrapped observation: [[t0_ws, t0_wd, t0_yaw], [t1_ws, t1_wd, t1_yaw], ...]
    
    This enables the transformer to treat each turbine as a token.
    """
    
    def __init__(self, env: gym.Env):
        """
        Args:
            env: The base WindFarmEnv (or vectorized version)
        """
        super().__init__(env)
        
        
        self._n_turbines = env.n_turb
        self._turbine_positions = np.column_stack([env.x_pos, env.y_pos])
        
        # Rotor diameter in meters (for position normalization)
        self._rotor_diameter = env.D

        # Observation dimension per turbine
        self._obs_dim_per_turbine = len(env.farm_measurements.turb_mes[0].get_measurements())
        
        # Action dimension per turbine
        self._action_dim_per_turbine = 1  # Usually just yaw
        
        # =====================================================================
        # Update observation and action spaces to reflect new shapes
        # =====================================================================
        
        # New observation space: (n_turbines, obs_dim_per_turbine)
        obs_low = -np.inf  # or get from base env
        obs_high = np.inf
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            shape=(self._n_turbines, self._obs_dim_per_turbine),
            dtype=np.float32
        )
        
        # Action space stays the same total size, but we'll handle reshaping
        # Keep the original action space for compatibility
        # The transformer will output (n_turbines, action_dim_per_turbine)
        # which gets flattened to (n_turbines * action_dim_per_turbine,)
    
    # =========================================================================
    # Required Properties
    # =========================================================================
    
    @property
    def n_turbines(self) -> int:
        """Number of turbines in the farm."""
        return self._n_turbines
    
    @property
    def turbine_positions(self) -> np.ndarray:
        """
        Turbine positions in meters.
        
        Returns:
            np.ndarray of shape (n_turbines, 2) with (x, y) coordinates
        """
        return self._turbine_positions
    
    @property
    def rotor_diameter(self) -> float:
        """Rotor diameter in meters (used for position normalization)."""
        return self._rotor_diameter
    
    @property
    def mean_wind_direction(self) -> float:
        """
        Current mean wind direction in degrees (meteorological convention).
        
        This is used for transforming positions to wind-relative coordinates.
        270° means wind comes from the West.
        
        Returns:
            float: Wind direction in degrees
        """
        return self.env.wd
        
    # =========================================================================
    # Core Methods
    # =========================================================================
    
    def _reshape_obs_to_per_turbine(self, flat_obs: np.ndarray) -> np.ndarray:
        """
        Reshape flat observation to per-turbine format.
        
        Args:
            flat_obs: Shape (n_turbines * obs_dim_per_turbine,) or 
                      (batch, n_turbines * obs_dim_per_turbine) for vectorized
        
        Returns:
            obs: Shape (n_turbines, obs_dim_per_turbine) or
                 (batch, n_turbines, obs_dim_per_turbine)
        """
        """Reshape assuming obs is ordered by turbine."""
        return flat_obs.reshape(self._n_turbines, self._obs_dim_per_turbine)
    

    def _flatten_action(self, per_turbine_action: np.ndarray) -> np.ndarray:
        """
        Flatten per-turbine action back to format expected by base env.
        
        Args:
            per_turbine_action: Shape (n_turbines, action_dim_per_turbine)
                               or already flat (n_turbines * action_dim,)
        
        Returns:
            flat_action: Shape expected by base env
        """
        # Handle case where action is already flat
        if per_turbine_action.ndim == 1:
            return per_turbine_action
        
        # Flatten: (n_turbines, action_dim) -> (n_turbines * action_dim,)
        return per_turbine_action.reshape(-1)
    
    def reset(
        self, 
        *, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment and return per-turbine observation.
        
        Returns:
            obs: Shape (n_turbines, obs_dim_per_turbine)
            info: Dict with additional information
        """
        flat_obs, info = self.env.reset(seed=seed, options=options)
        
        # Reshape to per-turbine
        obs = self._reshape_obs_to_per_turbine(flat_obs)
        
        return obs, info
    
    def step(
        self, 
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step with per-turbine action.
        
        Args:
            action: Shape (n_turbines, action_dim_per_turbine) or flat
        
        Returns:
            obs: Shape (n_turbines, obs_dim_per_turbine)
            reward: float
            terminated: bool
            truncated: bool
            info: dict
        """
        # Flatten action for base env
        flat_action = self._flatten_action(action)
        
        # Step base env
        flat_obs, reward, terminated, truncated, info = self.env.step(flat_action)
        
        # Reshape observation
        obs = self._reshape_obs_to_per_turbine(flat_obs)
        
        return obs, reward, terminated, truncated, info


# =============================================================================
# Example Implementation (fill in based on your WindFarmEnv structure)
# =============================================================================

class PerTurbineObservationWrapperExample(gym.Wrapper):
    """
    Example implementation assuming a specific WindFarmEnv structure.
    
    Assumes:
    - env.x_pos, env.y_pos: turbine positions
    - env.turbine.diameter(): rotor diameter
    - Observation ordered by turbine: [t0_features..., t1_features..., ...]
    - Each turbine has same features: ws_history, wd_history, yaw_history
    """
    
    def __init__(self, env: gym.Env):
        super().__init__(env)
        
        # Get turbine count from positions
        self._n_turbines = len(env.x_pos)
        
        # Stack positions
        self._turbine_positions = np.column_stack([env.x_pos, env.y_pos])
        
        # Get rotor diameter
        self._rotor_diameter = env.turbine.diameter()
        
        # Calculate obs_dim_per_turbine from config
        # Assuming: ws_history_N * 1 + wd_history_N * 1 + yaw_history_N * 1
        config = env.config
        n_ws = config['ws_mes']['ws_history_N'] if config['mes_level']['turb_ws'] else 0
        n_wd = config['wd_mes']['wd_history_N'] if config['mes_level']['turb_wd'] else 0
        n_yaw = config['yaw_mes']['yaw_history_N']
        n_power = config['power_mes']['power_history_N'] if config['mes_level']['turb_power'] else 0
        
        self._obs_dim_per_turbine = n_ws + n_wd + n_yaw + n_power
        
        # Action dim (yaw only)
        self._action_dim_per_turbine = 1
        
        # Update observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._n_turbines, self._obs_dim_per_turbine),
            dtype=np.float32
        )
        
        # Store wind direction for access
        self._current_wd = (config['wind']['wd_min'] + config['wind']['wd_max']) / 2
    
    @property
    def n_turbines(self) -> int:
        return self._n_turbines
    
    @property
    def turbine_positions(self) -> np.ndarray:
        return self._turbine_positions
    
    @property
    def rotor_diameter(self) -> float:
        return self._rotor_diameter
    
    @property
    def mean_wind_direction(self) -> float:
        # If your env tracks current wind direction, use that
        # Otherwise use config midpoint or extract from observations
        if hasattr(self.env, 'current_wd'):
            return self.env.current_wd
        return self._current_wd
    
    def _reshape_obs_to_per_turbine(self, flat_obs: np.ndarray) -> np.ndarray:
        """Reshape assuming obs is ordered by turbine."""
        return flat_obs.reshape(self._n_turbines, self._obs_dim_per_turbine)
    
    def _flatten_action(self, action: np.ndarray) -> np.ndarray:
        if action.ndim == 1:
            return action
        return action.reshape(-1)
    
    def reset(self, *, seed=None, options=None):
        flat_obs, info = self.env.reset(seed=seed, options=options)
        obs = self._reshape_obs_to_per_turbine(flat_obs)
        
        # Update wind direction if available
        if 'wind_direction' in info:
            self._current_wd = info['wind_direction']
        
        return obs, info
    
    def step(self, action):
        flat_action = self._flatten_action(action)
        flat_obs, reward, terminated, truncated, info = self.env.step(flat_action)
        obs = self._reshape_obs_to_per_turbine(flat_obs)
        
        # Update wind direction if available
        if 'wind_direction' in info:
            self._current_wd = info['wind_direction']
        
        return obs, reward, terminated, truncated, info


# =============================================================================
# Testing utility
# =============================================================================

def test_wrapper(env):
    """
    Test that the wrapper is correctly implemented.
    """
    print("Testing PerTurbineObservationWrapper...")
    print(f"  n_turbines: {env.n_turbines}")
    print(f"  turbine_positions shape: {env.turbine_positions.shape}")
    print(f"  rotor_diameter: {env.rotor_diameter}")
    print(f"  mean_wind_direction: {env.mean_wind_direction}")
    
    obs, info = env.reset()
    print(f"  observation shape after reset: {obs.shape}")
    
    expected_shape = (env.n_turbines, env.observation_space.shape[1])
    assert obs.shape == expected_shape, f"Expected {expected_shape}, got {obs.shape}"
    
    # Test step with per-turbine action
    action = np.zeros((env.n_turbines, 1))  # Zero yaw for all turbines
    obs, reward, term, trunc, info = env.step(action)
    print(f"  observation shape after step: {obs.shape}")
    assert obs.shape == expected_shape, f"Expected {expected_shape}, got {obs.shape}"
    
    # Test step with flat action
    action_flat = np.zeros(env.n_turbines)
    obs, reward, term, trunc, info = env.step(action_flat)
    assert obs.shape == expected_shape, f"Expected {expected_shape}, got {obs.shape}"
    
    print("  ✓ All tests passed!")
    
    return True


# =============================================================================
# Quick checklist for implementation
# =============================================================================

"""
IMPLEMENTATION CHECKLIST:

□ Set self._n_turbines
  - How do you get the number of turbines from your env?

□ Set self._turbine_positions  
  - Shape must be (n_turbines, 2)
  - Units should be meters

□ Set self._rotor_diameter
  - In meters (DTU10MW ≈ 178.3m, V80 ≈ 80m)

□ Set self._obs_dim_per_turbine
  - Count all features for ONE turbine
  - Example: 3 ws_history + 3 wd_history + 3 yaw_history = 9

□ Implement mean_wind_direction property
  - Return current wind direction in degrees
  - Meteorological convention: 270° = wind from West

□ Implement _reshape_obs_to_per_turbine()
  - Key question: is your flat obs ordered by turbine or by feature?
  - Test with a simple print to see the structure

□ Run test_wrapper() to verify everything works

DEBUGGING TIPS:

If obs shape is wrong:
  flat_obs = self.env.reset()[0]
  print(f"Flat obs shape: {flat_obs.shape}")
  print(f"Expected: n_turbines={self._n_turbines} × obs_dim={self._obs_dim_per_turbine}")
  print(f"          = {self._n_turbines * self._obs_dim_per_turbine}")

If positions look wrong:
  import matplotlib.pyplot as plt
  plt.scatter(env.turbine_positions[:, 0], env.turbine_positions[:, 1])
  plt.axis('equal')
  plt.show()
"""