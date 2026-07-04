"""
Pre-allocated replay buffer for Transformer-based SAC training.

Wind-relative transformation is applied at sample time. Profiles are looked up
via vectorized gather on a pre-padded registry.
"""

import json
import os

import numpy as np
import torch
from typing import Optional, List, Tuple, Dict

from helpers.helper_funcs import transform_to_wind_relative, rotate_profiles_tensor


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
        profile_registry_gpu_budget_mb: int = 256,
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
            profile_registry_gpu_budget_mb: If the padded profile registry fits under this
                many MB (and device is CUDA), keep it GPU-resident so sample() transfers only
                the tiny per-sample index/permutation arrays instead of the full profiles.
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

            # Keep the registry GPU-resident if it fits the budget. Then sample()
            # transfers only the (B,) layout idx + (B, T) permutation arrays and does
            # the gather/permute/rotate on-device, avoiding the ~n_dirs-wide H2D copy
            # of the full profile batch every call. The CPU arrays are retained
            # regardless (save() reads their shape).
            reg_bytes = self._padded_recep.nbytes + self._padded_infl.nbytes
            budget_bytes = profile_registry_gpu_budget_mb * 1024 * 1024
            self._registry_on_gpu = device.type == "cuda" and reg_bytes <= budget_bytes
            if self._registry_on_gpu:
                self._padded_recep_gpu = torch.from_numpy(self._padded_recep).to(device)
                self._padded_infl_gpu = torch.from_numpy(self._padded_infl).to(device)
            print(
                f"[ReplayBuffer] profile registry: "
                f"{'GPU-resident' if self._registry_on_gpu else 'CPU (per-sample transfer)'} "
                f"({reg_bytes / 1e6:.1f} MB)"
            )
        else:
            self._layout_indices = None
            self._permutations = None
            self._registry_on_gpu = False

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
        # Sample with replacement (standard RL replay; matches SB3/CleanRL). Avoids
        # np.random.choice(replace=False), which builds a full O(self.size) permutation
        # every call -- a hidden cost that scales with buffer fill (up to 1e6).
        indices = np.random.randint(0, self.size, size=batch_size)

        # --- Vectorized array indexing (the whole point) ---
        raw_positions = self._raw_positions[indices]            # (B, T, 2)
        wind_directions = self._wind_directions[indices]        # (B,)

        # Normalize positions by rotor diameter. Force float32: float32 / python-float
        # upcasts to float64 under numpy's legacy (pre-NEP50) promotion, which then
        # crashes the float32 positional-bias MLP. The eval/agent path already casts.
        positions_norm = (raw_positions / self.rotor_diameter).astype(np.float32)

        # Move wind directions to device once (reused by the wind-relative
        # transform and GPU-side profile rotation below).
        need_wind = self.use_wind_relative or (self.use_profiles and self.rotate_profiles)
        wind_dir_tensor = self._to_device(wind_directions) if need_wind else None

        # Positions: normalize on CPU (cheap), transfer, rotate to wind-relative on GPU.
        positions_tensor = self._to_device(positions_norm)
        if self.use_wind_relative:
            positions_final = transform_to_wind_relative(positions_tensor, wind_dir_tensor)
        else:
            positions_final = positions_tensor

        result = {
            "observations": self._to_device(self._obs[indices]),
            "next_observations": self._to_device(self._next_obs[indices]),
            "actions": self._to_device(self._actions[indices]),
            "positions": positions_final,
            "attention_mask": self._to_device(self._attention_mask[indices]),
            "rewards": self._to_device(self._rewards[indices]).unsqueeze(-1),
            "dones": self._to_device(self._dones[indices]).unsqueeze(-1),
        }

        # --- Vectorized profile lookup + permutation ---
        if self.use_profiles:
            layout_idx_batch = self._layout_indices[indices]    # (B,)
            perm_batch = self._permutations[indices]            # (B, T)

            if self._registry_on_gpu:
                # Transfer only the tiny index/permutation arrays; gather + permute the
                # GPU-resident registry on-device. torch.gather over axis 1 is the exact
                # equivalent of np.take_along_axis(..., perm[:, :, None], axis=1).
                layout_idx_t = self._to_device(layout_idx_batch).long()  # (B,)
                perm_t = self._to_device(perm_batch)                     # (B, T) int64
                recep_t = self._padded_recep_gpu[layout_idx_t]           # (B, T, D)
                infl_t = self._padded_infl_gpu[layout_idx_t]
                perm_idx = perm_t[:, :, None].expand(-1, -1, self._n_dirs)
                recep_t = torch.gather(recep_t, 1, perm_idx)
                infl_t = torch.gather(infl_t, 1, perm_idx)
            else:
                # CPU path: gather from pre-padded registry, permute, then transfer.
                recep_batch = self._padded_recep[layout_idx_batch]  # (B, T, D)
                infl_batch = self._padded_infl[layout_idx_batch]    # (B, T, D)

                # Apply permutation via advanced indexing (vectorized, no loop)
                # perm_batch[:, :, None] broadcasts over the n_dirs axis
                recep_batch = np.take_along_axis(recep_batch, perm_batch[:, :, None], axis=1)
                infl_batch = np.take_along_axis(infl_batch, perm_batch[:, :, None], axis=1)

                recep_t = self._to_device(recep_batch)
                infl_t = self._to_device(infl_batch)

            # Rotate to wind-relative frame on-device (avoids CPU rotation + re-copy)
            if self.rotate_profiles:
                recep_t = rotate_profiles_tensor(recep_t, wind_dir_tensor)
                infl_t = rotate_profiles_tensor(infl_t, wind_dir_tensor)

            result["receptivity"] = recep_t
            result["influence"] = infl_t

        return result

    def _to_device(self, arr: np.ndarray) -> torch.Tensor:
        """
        Host->device transfer for a sampled batch array.

        A plain synchronous copy: the result is consumed immediately by the
        critic forward, so there is no concurrent GPU work to overlap with --
        pinning + non_blocking would only add page-lock allocation overhead
        for zero benefit (revisit only if a prefetch pipeline is added).
        Storage arrays already hold the target dtypes (float32 / bool), so no
        cast is needed here. ascontiguousarray is kept: fancy-indexed sources are
        contiguous, but normalized positions / permuted profiles may not be.
        """
        t = torch.from_numpy(np.ascontiguousarray(arr))
        return t.to(self.device)

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

    def save(self, path: str, extra_meta: Optional[dict] = None) -> None:
        """
        Save the filled portion of the buffer to an uncompressed .npz file.

        Only the first `size` transitions are written (circular order does not
        matter for uniform sampling). The write is atomic: data goes to a temp
        file which is then renamed, so a killed job never leaves a truncated
        buffer at `path`.

        Args:
            path: Destination .npz path
            extra_meta: Optional JSON-serializable dict stored alongside the
                data (e.g. layouts, seed, global_step) and returned by load()
        """
        n = self.size
        payload = {
            "obs": self._obs[:n],
            "next_obs": self._next_obs[:n],
            "actions": self._actions[:n],
            "rewards": self._rewards[:n],
            "dones": self._dones[:n],
            "raw_positions": self._raw_positions[:n],
            "attention_mask": self._attention_mask[:n],
            "wind_directions": self._wind_directions[:n],
            "size": np.int64(n),
            "max_turbines": np.int64(self.max_turbines),
            "obs_dim": np.int64(self._obs.shape[2]),
            "action_dim": np.int64(self._actions.shape[2]),
            "use_profiles": np.bool_(self.use_profiles),
            "meta_json": np.str_(json.dumps(extra_meta or {})),
        }
        if self.use_profiles:
            payload["layout_indices"] = self._layout_indices[:n]
            payload["permutations"] = self._permutations[:n]
            payload["n_layouts"] = np.int64(self._padded_recep.shape[0])
            payload["n_dirs"] = np.int64(self._n_dirs)

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp_path = path + ".tmp"
        with open(tmp_path, "wb") as f:
            np.savez(f, **payload)
        os.replace(tmp_path, path)

        size_mb = os.path.getsize(path) / 1e6
        print(f"[ReplayBuffer] Saved {n} transitions to {path} ({size_mb:.1f} MB)")

    def load(self, path: str) -> dict:
        """
        Load transitions from a .npz file (created by save()) into this buffer.

        The buffer must already be constructed with a compatible configuration
        (same max_turbines/obs_dim/action_dim/use_profiles, and for profiles
        the same registry layout count and direction resolution). Existing
        contents are overwritten.

        Args:
            path: Source .npz path

        Returns:
            The extra_meta dict that was passed to save()
        """
        with np.load(path, allow_pickle=False) as data:
            n = int(data["size"])

            def _check(name: str, expected: int) -> None:
                actual = int(data[name])
                if actual != expected:
                    raise ValueError(
                        f"Replay buffer mismatch on '{name}': saved buffer has "
                        f"{actual}, current buffer expects {expected} ({path})"
                    )

            _check("max_turbines", self.max_turbines)
            _check("obs_dim", self._obs.shape[2])
            _check("action_dim", self._actions.shape[2])
            if bool(data["use_profiles"]) != self.use_profiles:
                raise ValueError(
                    f"Replay buffer mismatch on 'use_profiles': saved buffer has "
                    f"{bool(data['use_profiles'])}, current buffer expects "
                    f"{self.use_profiles} ({path})"
                )
            if n > self.capacity:
                raise ValueError(
                    f"Saved buffer holds {n} transitions but capacity is only "
                    f"{self.capacity}. Increase --buffer_size."
                )

            self._obs[:n] = data["obs"]
            self._next_obs[:n] = data["next_obs"]
            self._actions[:n] = data["actions"]
            self._rewards[:n] = data["rewards"]
            self._dones[:n] = data["dones"]
            self._raw_positions[:n] = data["raw_positions"]
            self._attention_mask[:n] = data["attention_mask"]
            self._wind_directions[:n] = data["wind_directions"]

            if self.use_profiles:
                _check("n_layouts", self._padded_recep.shape[0])
                _check("n_dirs", self._n_dirs)
                self._layout_indices[:n] = data["layout_indices"]
                self._permutations[:n] = data["permutations"]

            meta = json.loads(str(data["meta_json"]))

        self.size = n
        self.position = n % self.capacity

        print(f"[ReplayBuffer] Loaded {n} transitions from {path}")
        return meta

    def __len__(self) -> int:
        return self.size
