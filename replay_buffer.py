"""
Pre-allocated replay buffer for Transformer-based SAC training.

Wind-relative transformation is applied at sample time. Profiles are looked up
via vectorized gather on a pre-padded registry.
"""

import numpy as np
import torch
from typing import Optional, List, Tuple, Dict

from helpers.helper_funcs import transform_to_wind_relative


class TransformerReplayBuffer:
    """
    Replay buffer with pre-allocated numpy arrays for O(1) insertion and
    vectorized batch sampling (no per-sample Python loops).

    Wind-relative transformation is applied at sample time to ensure
    correct positional encoding regardless of when the transition was collected.

    Profiles are looked up at sample time via vectorized gather + permute
    on a pre-padded profile registry, rather than storing full profiles
    per-transition.

    Storage (pre-allocated numpy arrays):
    - _obs:            (capacity, max_turbines, obs_dim)         float32
    - _next_obs:       (capacity, max_turbines, obs_dim)         float32
    - _actions:        (capacity, max_turbines, action_dim)      float32
    - _rewards:        (capacity,)                               float32
    - _dones:          (capacity,)                               float32
    - _raw_positions:  (capacity, max_turbines, 2)               float32
    - _attention_mask: (capacity, max_turbines)                  bool
    - _wind_directions:(capacity,)                               float32
    - _layout_indices: (capacity,)                               int32   (profiles only)
    - _permutations:   (capacity, max_turbines)                  int64   (profiles only)
    """

    def __init__(
        self,
        capacity: int,
        device: torch.device,
        rotor_diameter: float,
        max_turbines: int,
        obs_dim: int,
        action_dim: int,
        use_wind_relative: bool = True,
        use_profiles: bool = False,
        rotate_profiles: bool = False,
        profile_registry: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    ):
        """
        Args:
            capacity: Maximum number of transitions
            device: Torch device for sampled tensors
            rotor_diameter: For position normalization
            max_turbines: Maximum number of turbines across all layouts
            obs_dim: Observation dimension per turbine
            action_dim: Action dimension per turbine
            use_wind_relative: Whether to transform positions to wind-relative frame
            use_profiles: Whether to store and return receptivity profiles
            rotate_profiles: Whether to rotate profiles to wind-relative frame at sample time
            profile_registry: List of (recep, influence) tuples per layout, each (n_turb, n_dirs)
        """
        self.capacity = capacity
        self.device = device
        self.rotor_diameter = rotor_diameter
        self.max_turbines = max_turbines
        self.use_wind_relative = use_wind_relative
        self.use_profiles = use_profiles
        self.rotate_profiles = rotate_profiles
        self.position = 0
        self.size = 0  # Current number of stored transitions

        # --- Pre-allocate storage arrays ---
        self._obs = np.zeros((capacity, max_turbines, obs_dim), dtype=np.float32)
        self._next_obs = np.zeros((capacity, max_turbines, obs_dim), dtype=np.float32)
        self._actions = np.zeros((capacity, max_turbines, action_dim), dtype=np.float32)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=np.float32)
        self._raw_positions = np.zeros((capacity, max_turbines, 2), dtype=np.float32)
        self._attention_mask = np.zeros((capacity, max_turbines), dtype=bool)
        self._wind_directions = np.zeros(capacity, dtype=np.float32)

        # --- Profile-specific storage ---
        if self.use_profiles:
            assert profile_registry is not None, "Must provide profile_registry when use_profiles=True"

            self._layout_indices = np.zeros(capacity, dtype=np.int32)
            self._permutations = np.zeros((capacity, max_turbines), dtype=np.int64)

            # Pre-pad registry profiles to max_turbines for vectorized gather.
            # _padded_recep[layout_idx] = (max_turbines, n_dirs), zero-padded
            # _padded_infl[layout_idx]  = (max_turbines, n_dirs), zero-padded
            n_dirs = profile_registry[0][0].shape[1]
            n_layouts = len(profile_registry)
            self._padded_recep = np.zeros((n_layouts, max_turbines, n_dirs), dtype=np.float32)
            self._padded_infl = np.zeros((n_layouts, max_turbines, n_dirs), dtype=np.float32)
            for li, (recep, infl) in enumerate(profile_registry):
                nt = recep.shape[0]
                self._padded_recep[li, :nt] = recep
                self._padded_infl[li, :nt] = infl
            self._n_dirs = n_dirs
        else:
            self._layout_indices = None
            self._permutations = None

        alloc_mb = (
            self._obs.nbytes + self._next_obs.nbytes + self._actions.nbytes
            + self._rewards.nbytes + self._dones.nbytes
            + self._raw_positions.nbytes + self._attention_mask.nbytes
            + self._wind_directions.nbytes
        ) / 1e6
        print(f"[ReplayBuffer] Pre-allocated {alloc_mb:.1f} MB for {capacity} transitions "
              f"(max_turb={max_turbines}, obs_dim={obs_dim}, act_dim={action_dim})")

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        raw_positions: np.ndarray,
        attention_mask: np.ndarray,
        wind_direction: float,
        layout_index: Optional[int] = None,
        permutation: Optional[np.ndarray] = None,
    ) -> None:
        """
        Store a transition by writing directly into pre-allocated arrays.

        Args:
            obs: (max_turbines, obs_dim)
            next_obs: (max_turbines, obs_dim)
            action: (max_turbines, action_dim)
            reward: scalar
            done: bool
            raw_positions: (max_turbines, 2)
            attention_mask: (max_turbines,) - True = padding
            wind_direction: scalar
            layout_index: Index of current layout (for profile lookup)
            permutation: Turbine permutation array (for shuffled profiles)
        """
        i = self.position

        self._obs[i] = obs
        self._next_obs[i] = next_obs
        self._actions[i] = action
        self._rewards[i] = reward
        self._dones[i] = float(done)
        self._raw_positions[i] = raw_positions
        self._attention_mask[i] = attention_mask
        self._wind_directions[i] = wind_direction

        if self.use_profiles:
            assert layout_index is not None, "layout_index required when use_profiles=True"
            self._layout_indices[i] = layout_index

            if permutation is not None:
                # Sanitize permutation: padding positions get identity mapping
                # so that indexing into the zero-padded registry stays zero.
                safe_perm = permutation.copy()
                n_real = int((~attention_mask).sum())
                if n_real < self.max_turbines:
                    safe_perm[n_real:] = np.arange(n_real, self.max_turbines)
                self._permutations[i] = safe_perm
            else:
                self._permutations[i] = np.arange(self.max_turbines)

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample a batch using vectorized array indexing (no Python loop).

        Returns:
            Dict with keys:
            - observations: (batch, max_turb, obs_dim)
            - next_observations: (batch, max_turb, obs_dim)
            - actions: (batch, max_turb, action_dim)
            - positions: (batch, max_turb, 2) - transformed and normalized
            - attention_mask: (batch, max_turb)
            - rewards: (batch, 1)
            - dones: (batch, 1)
            - receptivity: (batch, max_turb, n_directions) - only if use_profiles=True
            - influence: (batch, max_turb, n_directions) - only if use_profiles=True
        """
        indices = np.random.choice(self.size, batch_size, replace=False)

        # --- Vectorized array indexing (the whole point) ---
        raw_positions = self._raw_positions[indices]            # (B, T, 2)
        wind_directions = self._wind_directions[indices]        # (B,)

        # Normalize positions by rotor diameter
        positions_norm = raw_positions / self.rotor_diameter

        # Convert to tensors
        positions_tensor = torch.tensor(positions_norm, device=self.device, dtype=torch.float32)

        # Conditionally apply wind-relative transformation
        if self.use_wind_relative:
            wind_dir_tensor = torch.tensor(wind_directions, device=self.device, dtype=torch.float32)
            positions_final = transform_to_wind_relative(positions_tensor, wind_dir_tensor)
        else:
            positions_final = positions_tensor

        result = {
            "observations": torch.tensor(self._obs[indices], device=self.device, dtype=torch.float32),
            "next_observations": torch.tensor(self._next_obs[indices], device=self.device, dtype=torch.float32),
            "actions": torch.tensor(self._actions[indices], device=self.device, dtype=torch.float32),
            "positions": positions_final,
            "attention_mask": torch.tensor(self._attention_mask[indices], device=self.device, dtype=torch.bool),
            "rewards": torch.tensor(self._rewards[indices], device=self.device, dtype=torch.float32).unsqueeze(-1),
            "dones": torch.tensor(self._dones[indices], device=self.device, dtype=torch.float32).unsqueeze(-1),
        }

        # --- Vectorized profile lookup + permutation ---
        if self.use_profiles:
            layout_idx_batch = self._layout_indices[indices]    # (B,)
            perm_batch = self._permutations[indices]            # (B, T)

            # Gather from pre-padded registry: (B, T, n_dirs)
            recep_batch = self._padded_recep[layout_idx_batch]  # (B, T, D)
            infl_batch = self._padded_infl[layout_idx_batch]    # (B, T, D)

            # Apply permutation via advanced indexing (vectorized, no loop)
            # perm_batch[:, :, None] broadcasts over the n_dirs axis
            recep_batch = np.take_along_axis(recep_batch, perm_batch[:, :, None], axis=1)
            infl_batch = np.take_along_axis(infl_batch, perm_batch[:, :, None], axis=1)

            # Optionally rotate profiles to wind-relative frame
            if self.rotate_profiles:
                recep_batch = self._rotate_profiles_batch(recep_batch, wind_directions)
                infl_batch = self._rotate_profiles_batch(infl_batch, wind_directions)

            result["receptivity"] = torch.tensor(recep_batch, device=self.device, dtype=torch.float32)
            result["influence"] = torch.tensor(infl_batch, device=self.device, dtype=torch.float32)

        return result

    def _rotate_profiles_batch(
        self,
        profiles: np.ndarray,
        wind_directions: np.ndarray
    ) -> np.ndarray:
        """
        Rotate profiles so current wind direction is at index 0 (vectorized).

        Args:
            profiles: (batch, max_turb, n_directions)
            wind_directions: (batch,) wind directions in degrees

        Returns:
            Rotated profiles with same shape
        """
        n_directions = profiles.shape[2]
        degrees_per_index = 360.0 / n_directions

        shifts = np.round(wind_directions / degrees_per_index).astype(int)
        # Build shifted index array: (batch, 1, n_directions)
        indices = (np.arange(n_directions)[None, None, :] + shifts[:, None, None]) % n_directions
        return np.take_along_axis(profiles, indices, axis=-1)

    def __len__(self) -> int:
        return self.size
