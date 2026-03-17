"""
PyTorch Dataset for Wind Farm Pretraining Data

Reads HDF5 files produced by collect_pretrain_data.py and serves
(obs, positions, profiles, targets) samples with configurable history windows.

Supports two modes:
    1. Snapshot mode (history_length=1): For masked prediction on single timesteps
    2. History mode (history_length>1): Stacks past observations like the RL agent sees

Preprocessing (matching the RL training pipeline):
    - Normalization: Raw values → [-1, 1] using env-compatible scaling limits
    - Wind direction deviation: Optionally convert absolute WD to deviation from
      episode mean, matching EnhancedPerTurbineWrapper behavior
    - Wind-relative positions: Optionally rotate positions so wind comes from 270°
    - Profile rotation: Optionally rotate profiles so wind direction aligns to index 0

Global features (global_features=["ws", "wd"]):
    Features listed in global_features are replaced with episode-level scalars
    broadcast identically to all turbines (1 dim each instead of history_length).
    Each flag is independent — any subset of ["ws", "wd"] can be made global:
        global_features=["ws"]       → ws becomes broadcast mean_ws
        global_features=["ws","wd"]  → both become broadcast scalars
    This forces the transformer to learn wake interactions through attention
    rather than shortcutting via per-turbine wind speed → power curve lookup.

Author: Marcus (DTU Wind Energy)
"""

import math
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Dict, Tuple
from pathlib import Path


# =============================================================================
# NORMALIZATION UTILITIES
# =============================================================================

def normalize_to_minus1_plus1(
    values: np.ndarray,
    vmin: float,
    vmax: float,
) -> np.ndarray:
    """
    Scale values from [vmin, vmax] → [-1, 1].

    This matches the WindFarmEnv's internal observation scaling:
        scaled = 2 * (val - vmin) / (vmax - vmin) - 1

    Args:
        values: Array of raw physical values
        vmin: Minimum of the expected range
        vmax: Maximum of the expected range

    Returns:
        Scaled array in [-1, 1] (values outside [vmin, vmax] will exceed ±1)
    """
    return (2.0 * (values - vmin) / (vmax - vmin) - 1.0).astype(np.float32)


def compute_wd_deviation(
    local_wd: np.ndarray,
    mean_wd: float,
    scale_range: float = 90.0,
) -> np.ndarray:
    """
    Convert absolute wind direction to deviation from episode mean, scaled to [-1, 1].

    Matches helper_funcs.compute_wind_direction_deviation exactly:
        deviation = ((local_wd - mean_wd + 180) % 360) - 180
        scaled = clip(deviation / scale_range, -1, 1)

    Args:
        local_wd: Per-turbine local wind direction in degrees
        mean_wd: Episode-level mean wind direction in degrees
        scale_range: ±scale_range degrees maps to [-1, 1] (default 90°)

    Returns:
        Scaled deviation in [-1, 1], same shape as input
    """
    deviation = local_wd - mean_wd
    deviation = ((deviation + 180) % 360) - 180
    scaled = np.clip(deviation / scale_range, -1.0, 1.0)
    return scaled.astype(np.float32)


def rotate_positions_wind_relative(
    positions: np.ndarray,
    wind_direction: float,
) -> np.ndarray:
    """
    Rotate positions so wind effectively comes from 270° (negative x).

    Matches helper_funcs.transform_to_wind_relative (numpy single-sample version).

    Args:
        positions: (n_turbines, 2) normalized positions
        wind_direction: Wind direction in degrees (meteorological convention)

    Returns:
        Rotated positions with same shape
    """
    angle_offset = wind_direction - 270.0
    theta = angle_offset * (math.pi / 180.0)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    x = positions[:, 0]
    y = positions[:, 1]

    x_rot = cos_t * x - sin_t * y
    y_rot = sin_t * x + cos_t * y

    return np.stack([x_rot, y_rot], axis=-1).astype(np.float32)


def rotate_profiles_numpy(
    profiles: np.ndarray,
    wind_direction: float,
) -> np.ndarray:
    """
    Rotate profiles so current wind direction aligns to index 0.

    Matches helper_funcs.rotate_profiles_tensor and
    TransformerReplayBuffer._rotate_profiles_batch (single-sample).

    Args:
        profiles: (n_turbines, n_directions)
        wind_direction: Wind direction in degrees

    Returns:
        Rotated profiles with same shape
    """
    n_directions = profiles.shape[1]
    degrees_per_index = 360.0 / n_directions
    shift = int(round(wind_direction / degrees_per_index))
    indices = (np.arange(n_directions) + shift) % n_directions
    return profiles[:, indices].astype(np.float32)


# =============================================================================
# DEFAULT SCALING LIMITS (matching WindFarmEnv constructor defaults)
# =============================================================================

DEFAULT_SCALING = {
    "ws":    (0.0, 30.0),     # m/s  — WindFarmEnv: ws_scaling_min/max
    "wd":    (0.0, 360.0),    # deg  — WindFarmEnv: wd_scaling_min/max
    "yaw":   (-45.0, 45.0),   # deg  — WindFarmEnv: yaw_scaling_min/max
    "power": (0.0, 10e6),      # Fraction of max power (see note below)
}
# NOTE on power scaling:
# The env uses [0, maxturbpower] where maxturbpower = max(turbine.power(...)).
# For DTU10MW ≈ 10.64 MW, for V80 ≈ 2.0 MW.
# The raw data from make_datasets.py stores power in physical units (W or kW
# depending on PyWake config). The user should set power_range to match their
# turbine's max power output. Alternatively, if the data is already normalized
# to [0, 1] (fraction of rated), the default (0, 1) will work.


# =============================================================================
# HISTORY DATASET
# =============================================================================

class WindFarmPretrainDataset(Dataset):
    """
    Serves (obs, positions, profiles, target_power) for pretraining.

    Each sample is a single timestep with history_length context,
    drawn from a random episode across all layout files.

    Features listed in `global_features` are replaced with episode-level scalars
    (broadcast identically to all turbines), contributing 1 dim each instead of
    history_length dims. This forces the transformer to learn wake interactions
    through attention rather than shortcutting via per-turbine features.

    Examples (features=["ws","wd","yaw"], history_length=15):
        global_features=[]           → obs_dim = 15×3 = 45  (all per-turbine)
        global_features=["ws"]       → obs_dim = 1 + 15×2 = 31
        global_features=["ws","wd"]  → obs_dim = 2 + 15×1 = 17

    Preprocessing pipeline (applied per sample in __getitem__):
        1. Stack history window for each feature → (n_turb, H * n_features)
           OR build mixed obs with global scalars + history → (n_turb, n_global + H * n_hist)
        2. Normalize each feature to [-1, 1] using scaling limits
        3. Optionally convert WD to deviation from episode mean
        4. Pad to max_turbines with attention_mask
        5. Optionally transform positions to wind-relative frame
        6. Optionally rotate profiles to wind-relative frame
    """

    def __init__(
        self,
        layout_files: List[str],
        history_length: int = 15,
        max_turbines: Optional[int] = None,
        features: List[str] = ["ws", "wd", "yaw", "power"],
        # --- Global features ---
        global_features: List[str] = [],
        # --- Actions ---
        action_type: Optional[str] = "wind",
        # --- Normalization ---
        scaling_limits: Optional[Dict[str, Tuple[float, float]]] = None,
        # --- Wind direction deviation ---
        use_wd_deviation: bool = False,
        wd_scale_range: float = 90.0,
        # --- Wind-relative positions ---
        use_wind_relative_pos: bool = True,
        # --- Profile rotation ---
        rotate_profiles: bool = True,
    ):
        """
        Args:
            layout_files: List of HDF5 file paths (one per layout)
            history_length: Number of past timesteps to stack per feature
            max_turbines: Pad to this many turbines (None = auto from data)
            features: Which per-turbine features to include in obs
            global_features: Subset of `features` to replace with episode-level
                             scalars (broadcast to all turbines, 1 dim each).
                             E.g. ["ws"] replaces per-turbine ws history with mean_ws.
                             Supported: "ws" (→ mean_ws), "wd" (→ mean_wd).
            scaling_limits: Dict mapping feature name → (min, max) for [-1,1] scaling.
                            Defaults to WindFarmEnv defaults if not provided.
            use_wd_deviation: If True, convert WD to deviation from episode mean_wd
                              (matches EnhancedPerTurbineWrapper behavior).
                              When "wd" is in global_features, deviation from itself
                              is always 0, so consider using use_wd_deviation=False.
            wd_scale_range: ±degrees that map to [-1, 1] for WD deviation (default 90°)
            use_wind_relative_pos: If True, rotate positions so wind comes from 270°
            rotate_profiles: If True, rotate profiles so wind direction is at index 0
        """
        self.history_length = history_length
        self.features = features
        self.global_features = list(global_features)
        self.action_type = action_type
        self.use_wd_deviation = use_wd_deviation
        self.wd_scale_range = wd_scale_range
        self.use_wind_relative_pos = use_wind_relative_pos
        self.rotate_profiles = rotate_profiles

        # Validate global_features
        _supported_global = {"ws", "wd"}
        for gf in self.global_features:
            if gf not in _supported_global:
                raise ValueError(f"Unsupported global feature: '{gf}'. Supported: {_supported_global}")

        # Auto-merge: if a global feature isn't already in features, prepend it
        for gf in self.global_features:
            if gf not in self.features:
                self.features = list(self.features)
                self.features.insert(0, gf)

        # Merge user-provided limits with defaults
        self.scaling_limits = dict(DEFAULT_SCALING)
        if scaling_limits is not None:
            self.scaling_limits.update(scaling_limits)

        # Determine which features get per-turbine history vs global scalar
        self._global_features = [f for f in self.features if f in self.global_features]
        self._history_features = [f for f in self.features if f not in self.global_features]

        # Build index: list of (layout_idx, ep_key, timestep)
        self.index = []
        self.layouts = []

        n_global = len(self._global_features)
        n_hist = len(self._history_features)
        obs_dim_desc = []
        if n_global > 0:
            obs_dim_desc.append(f"{n_global} global ({', '.join(self._global_features)})")
        if n_hist > 0:
            obs_dim_desc.append(f"{n_hist}×{history_length} history ({', '.join(self._history_features)})")

        print(f"Loading dataset from {len(layout_files)} layout file(s)...")
        print(f"  Observation: {' + '.join(obs_dim_desc)}"
              f" → dim={n_global + history_length * n_hist}")
        print(f"  Normalization: ON (limits: { {k: v for k, v in self.scaling_limits.items() if k in features or k in ('ws', 'wd')} })")
        print(f"  WD deviation: {'ON' if use_wd_deviation else 'OFF'}"
              f"{f' (±{wd_scale_range}°→[-1,1])' if use_wd_deviation else ''}")
        print(f"  Wind-relative positions: {'ON' if use_wind_relative_pos else 'OFF'}")
        print(f"  Profile rotation: {'ON' if rotate_profiles else 'OFF'}")

        for li, path in enumerate(layout_files):
            with h5py.File(path, "r") as f:
                layout_info = {
                    "path": str(path),
                    "layout_name": str(f.attrs["layout_name"]),
                    "n_turbines": int(f.attrs["n_turbines"]),
                    "rotor_diameter": float(f.attrs["rotor_diameter"]),
                    "positions": f["positions/xy"][:].astype(np.float32),
                }

                # Load profiles if available
                if "profiles" in f:
                    layout_info["receptivity"] = f["profiles/receptivity"][:].astype(np.float32)
                    layout_info["influence"] = f["profiles/influence"][:].astype(np.float32)

                # Load episode-level metadata AND time-series into memory
                episode_meta = {}
                episode_data = {}
                for ep_key in sorted(f["episodes"].keys()):
                    ep = f[f"episodes/{ep_key}"]
                    n_steps = int(ep.attrs["n_steps"])
                    episode_meta[ep_key] = {
                        "n_steps": n_steps,
                        "mean_ws": float(ep.attrs["mean_ws"]),
                        "mean_wd": float(ep.attrs["mean_wd"]),
                        "mean_ti": float(ep.attrs["mean_ti"]),
                    }
                    # Pre-load time-series into memory
                    # Only load features that need history; global features come from episode attrs
                    ep_series = {}
                    for feat in self._history_features:
                        ep_series[feat] = ep[feat][:].astype(np.float32)
                    # Always load power for targets
                    ep_series["power"] = ep["power"][:].astype(np.float32)
                    # Conditionally load actions (BC needs them, pretrain may not)
                    if self.action_type is not None:
                        action_key = f"actions_{self.action_type}"
                        ep_series["actions"] = ep[action_key][:].astype(np.float32)
                    episode_data[ep_key] = ep_series

                    # Valid timesteps: need history_length steps of context before
                    for t in range(history_length, n_steps):
                        self.index.append((li, ep_key, t))

                layout_info["episode_meta"] = episode_meta
                layout_info["episode_data"] = episode_data
                self.layouts.append(layout_info)

                print(f"  [{li}] {layout_info['layout_name']}: "
                      f"{layout_info['n_turbines']} turbines, "
                      f"{len(episode_meta)} episodes, "
                      f"{sum(m['n_steps'] for m in episode_meta.values())} total steps")

        # Determine max turbines for padding
        self.max_turbines = max_turbines or max(l["n_turbines"] for l in self.layouts)

        # Compute obs_dim based on which features are global vs history
        self.obs_dim = len(self._global_features) + history_length * len(self._history_features)

        print(f"Dataset ready: {len(self.index)} samples, "
              f"max_turbines={self.max_turbines}, "
              f"obs_dim={self.obs_dim}")

        # === Precompute all samples into tensors (vectorized) ===
        print("Precomputing all samples (vectorized)...")
        N = len(self.index)
        self._obs = torch.zeros(N, self.max_turbines, self.obs_dim, dtype=torch.float32)
        self._target_power = torch.zeros(N, self.max_turbines, dtype=torch.float32)
        if self.action_type is not None:
            self._actions = torch.zeros(N, self.max_turbines, dtype=torch.float32)
        else:
            self._actions = None
        self._positions = torch.zeros(N, self.max_turbines, 2, dtype=torch.float32)
        self._attention_mask = torch.ones(N, self.max_turbines, dtype=torch.bool)
        self._n_turbines = torch.zeros(N, dtype=torch.long)
        self._layout_idx = torch.zeros(N, dtype=torch.long)
        self._mean_ws = torch.zeros(N, dtype=torch.float32)
        self._mean_wd = torch.zeros(N, dtype=torch.float32)
        self._mean_ti = torch.zeros(N, dtype=torch.float32)

        has_profiles = "receptivity" in self.layouts[0]
        if has_profiles:
            n_dirs = self.layouts[0]["receptivity"].shape[1]
            n_episodes = sum(len(l["episode_meta"]) for l in self.layouts)
            self._ep_receptivity = torch.zeros(n_episodes, self.max_turbines, n_dirs, dtype=torch.float32)
            self._ep_influence = torch.zeros(n_episodes, self.max_turbines, n_dirs, dtype=torch.float32)
            self._sample_ep_idx = torch.zeros(N, dtype=torch.long)
        self._has_profiles = has_profiles

        # Group sample indices by (layout_idx, episode_key) for bulk processing
        from collections import defaultdict
        groups = defaultdict(list)  # (li, ep_key) -> [(global_idx, t), ...]
        for global_idx, (li, ep_key, t) in enumerate(self.index):
            groups[(li, ep_key)].append((global_idx, t))

        ep_counter = 0
        done = 0
        for (li, ep_key), samples in groups.items():
            layout = self.layouts[li]
            n_turb = layout["n_turbines"]
            ep_meta = layout["episode_meta"][ep_key]
            ep_data = layout["episode_data"][ep_key]
            mean_ws = ep_meta["mean_ws"]
            mean_wd = ep_meta["mean_wd"]
            H = self.history_length

            global_idxs = np.array([s[0] for s in samples])
            timesteps = np.array([s[1] for s in samples])
            n_samples = len(timesteps)

            # --- Build observation: global scalars + per-turbine history ---
            obs_parts = []

            # Map from feature name → episode-level scalar value
            _global_values = {"ws": mean_ws, "wd": mean_wd}

            # Global features: broadcast scalar to (n_samples, n_turb, 1)
            for feat in self._global_features:
                raw_val = _global_values[feat]
                if feat == "wd" and self.use_wd_deviation:
                    val_norm = np.float32(0.0)  # deviation from itself
                else:
                    vmin, vmax = self.scaling_limits[feat]
                    val_norm = np.float32((2.0 * (raw_val - vmin) / (vmax - vmin)) - 1.0)
                obs_parts.append(np.full((n_samples, n_turb, 1), val_norm, dtype=np.float32))

            # History features: (n_samples, n_turb, H) each
            for feat in self._history_features:
                feat_data = ep_data[feat]  # (n_steps, n_turb)
                window_idxs = timesteps[:, None] + np.arange(-H, 0)[None, :]
                windows = feat_data[window_idxs]  # (n_samples, H, n_turb)
                windows_norm = self._normalize_feature(feat, windows, mean_wd)
                obs_parts.append(windows_norm.transpose(0, 2, 1))  # (n_samples, n_turb, H)

            obs = np.concatenate(obs_parts, axis=-1)

            self._obs[global_idxs, :n_turb] = torch.from_numpy(obs)

            # --- Target power & actions (vectorized) ---
            target_raw = ep_data["power"][timesteps - 1]  # (n_samples, n_turb)
            target_norm = self._normalize_feature("power", target_raw, mean_wd)
            self._target_power[global_idxs, :n_turb] = torch.from_numpy(target_norm)

            if self.action_type is not None:
                actions = ep_data["actions"][timesteps - 1]  # (n_samples, n_turb)
                self._actions[global_idxs, :n_turb] = torch.from_numpy(actions)

            # --- Mask & scalar metadata (bulk assign) ---
            self._attention_mask[global_idxs, :n_turb] = False
            self._n_turbines[global_idxs] = n_turb
            self._layout_idx[global_idxs] = li
            self._mean_ws[global_idxs] = mean_ws
            self._mean_wd[global_idxs] = mean_wd
            self._mean_ti[global_idxs] = ep_meta["mean_ti"]

            # --- Positions: same for all samples in this episode (same mean_wd) ---
            positions_norm = layout["positions"] / layout["rotor_diameter"]
            if self.use_wind_relative_pos:
                positions_norm = rotate_positions_wind_relative(positions_norm, mean_wd)
            pos_tensor = torch.from_numpy(positions_norm)  # (n_turb, 2)
            self._positions[global_idxs, :n_turb] = pos_tensor.unsqueeze(0).expand(n_samples, -1, -1)

            # --- Profiles: store once per episode, map samples → episode index ---
            if has_profiles:
                recep = layout["receptivity"]
                infl = layout["influence"]
                if self.rotate_profiles:
                    recep = rotate_profiles_numpy(recep, mean_wd)
                    infl = rotate_profiles_numpy(infl, mean_wd)
                self._ep_receptivity[ep_counter, :n_turb] = torch.from_numpy(recep)
                self._ep_influence[ep_counter, :n_turb] = torch.from_numpy(infl)
                self._sample_ep_idx[global_idxs] = ep_counter
                ep_counter += 1

            done += n_samples
            if done % 200000 < n_samples:
                print(f"  Precomputed {done}/{N} samples...")

        # Free raw episode data
        for layout in self.layouts:
            del layout["episode_data"]

        print(f"  Done. Precomputed {N} samples.")

    def __len__(self):
        return len(self.index)

    def _normalize_feature(self, feat_name: str, raw_values: np.ndarray,
                           mean_wd: float = 0.0) -> np.ndarray:
        """
        Normalize a single feature's raw values to [-1, 1].

        For wind direction with use_wd_deviation=True, computes deviation
        from episode mean instead of standard min-max scaling.

        Args:
            feat_name: Feature name (e.g., "ws", "wd", "yaw", "power")
            raw_values: Raw physical values, any shape
            mean_wd: Episode mean wind direction (only used if feat_name=="wd" and use_wd_deviation)

        Returns:
            Normalized values in [-1, 1], same shape
        """
        if feat_name == "wd" and self.use_wd_deviation:
            return compute_wd_deviation(raw_values, mean_wd, self.wd_scale_range)
        else:
            vmin, vmax = self.scaling_limits.get(feat_name, (0.0, 1.0))
            return normalize_to_minus1_plus1(raw_values, vmin, vmax)

    def __getitem__(self, idx):
        sample = {
            "obs": self._obs[idx],
            "positions": self._positions[idx],
            "attention_mask": self._attention_mask[idx],
            "target_power": self._target_power[idx],
            "n_turbines": self._n_turbines[idx],
            "layout_idx": self._layout_idx[idx],
            "mean_ws": self._mean_ws[idx],
            "mean_wd": self._mean_wd[idx],
            "mean_ti": self._mean_ti[idx],
        }
        if self._actions is not None:
            sample["actions"] = self._actions[idx]
        if self._has_profiles:
            ep_idx = self._sample_ep_idx[idx]
            sample["receptivity"] = self._ep_receptivity[ep_idx]
            sample["influence"] = self._ep_influence[ep_idx]
        return sample


# =============================================================================
# SNAPSHOT DATASET (no history, for pure masked prediction on steady-state)
# =============================================================================

class WindFarmSnapshotDataset(Dataset):
    """
    Simplified dataset for snapshot-based pretraining (history_length=1).

    Each sample is a single timestep with per-turbine features.
    Obs dim = n_features (not n_features * history).

    Supports global_features list (same as history dataset):
        Features in global_features are replaced with broadcast episode-level scalars.
        E.g. global_features=["ws"] → replaces per-turbine ws with mean_ws (1 dim)

    Useful for:
    - Masked turbine prediction from PyWake steady-state data
    - Quick pretraining experiments without temporal context
    """

    def __init__(
        self,
        layout_files: List[str],
        max_turbines: Optional[int] = None,
        features: List[str] = ["ws", "wd", "yaw", "power"],
        skip_steps: int = 1,
        # --- Global features ---
        global_features: List[str] = [],
        # --- Actions ---
        action_type: Optional[str] = None,  # accepted for API compat, not used in snapshot mode
        # --- Normalization ---
        scaling_limits: Optional[Dict[str, Tuple[float, float]]] = None,
        # --- Wind direction deviation ---
        use_wd_deviation: bool = False,
        wd_scale_range: float = 90.0,
        # --- Wind-relative positions ---
        use_wind_relative_pos: bool = True,
        # --- Profile rotation ---
        rotate_profiles: bool = True,
    ):
        self.features = features
        self.global_features = list(global_features)
        self.use_wd_deviation = use_wd_deviation
        self.wd_scale_range = wd_scale_range
        self.use_wind_relative_pos = use_wind_relative_pos
        self.rotate_profiles = rotate_profiles

        # Validate global_features
        _supported_global = {"ws", "wd"}
        for gf in self.global_features:
            if gf not in _supported_global:
                raise ValueError(f"Unsupported global feature: '{gf}'. Supported: {_supported_global}")

        # Auto-merge: if a global feature isn't already in features, prepend it
        for gf in self.global_features:
            if gf not in self.features:
                self.features = list(self.features)
                self.features.insert(0, gf)

        # Merge user-provided limits with defaults
        self.scaling_limits = dict(DEFAULT_SCALING)
        if scaling_limits is not None:
            self.scaling_limits.update(scaling_limits)

        # Determine which features are global vs per-turbine
        self._global_features = [f for f in self.features if f in self.global_features]
        self._snapshot_features = [f for f in self.features if f not in self.global_features]

        self.index = []
        self.layouts = []

        for li, path in enumerate(layout_files):
            with h5py.File(path, "r") as f:
                layout_info = {
                    "path": str(path),
                    "layout_name": str(f.attrs["layout_name"]),
                    "n_turbines": int(f.attrs["n_turbines"]),
                    "rotor_diameter": float(f.attrs["rotor_diameter"]),
                    "positions": f["positions/xy"][:].astype(np.float32),
                }
                if "profiles" in f:
                    layout_info["receptivity"] = f["profiles/receptivity"][:].astype(np.float32)
                    layout_info["influence"] = f["profiles/influence"][:].astype(np.float32)

                # Store episode metadata
                episode_meta = {}
                for ep_key in sorted(f["episodes"].keys()):
                    ep = f[f"episodes/{ep_key}"]
                    n_steps = int(ep.attrs["n_steps"])
                    episode_meta[ep_key] = {
                        "n_steps": n_steps,
                        "mean_ws": float(ep.attrs["mean_ws"]),
                        "mean_wd": float(ep.attrs["mean_wd"]),
                    }
                    for t in range(0, n_steps, skip_steps):
                        self.index.append((li, ep_key, t))

                layout_info["episode_meta"] = episode_meta
                self.layouts.append(layout_info)

        self.max_turbines = max_turbines or max(l["n_turbines"] for l in self.layouts)
        print(f"Snapshot dataset: {len(self.index)} samples, max_turb={self.max_turbines}"
              f"{', global: ' + ','.join(self._global_features) if self._global_features else ''}")

    def __len__(self):
        return len(self.index)

    def _normalize_feature(self, feat_name: str, raw_values: np.ndarray,
                           mean_wd: float = 0.0) -> np.ndarray:
        """Normalize a single feature (same logic as history dataset)."""
        if feat_name == "wd" and self.use_wd_deviation:
            return compute_wd_deviation(raw_values, mean_wd, self.wd_scale_range)
        else:
            vmin, vmax = self.scaling_limits.get(feat_name, (0.0, 1.0))
            return normalize_to_minus1_plus1(raw_values, vmin, vmax)

    def __getitem__(self, idx):
        li, ep_key, t = self.index[idx]
        layout = self.layouts[li]
        n_turb = layout["n_turbines"]
        ep_meta = layout["episode_meta"][ep_key]
        mean_ws = ep_meta["mean_ws"]
        mean_wd = ep_meta["mean_wd"]

        with h5py.File(layout["path"], "r") as f:
            ep = f[f"episodes/{ep_key}"]

            obs_parts = []

            # Map from feature name → episode-level scalar value
            _global_values = {"ws": mean_ws, "wd": mean_wd}

            # Global features: broadcast scalar to (n_turb, 1)
            for feat in self._global_features:
                raw_val = _global_values[feat]
                if feat == "wd" and self.use_wd_deviation:
                    val_norm = np.float32(0.0)
                else:
                    vmin, vmax = self.scaling_limits[feat]
                    val_norm = np.float32((2.0 * (raw_val - vmin) / (vmax - vmin)) - 1.0)
                obs_parts.append(np.full((n_turb, 1), val_norm, dtype=np.float32))

            # Per-turbine features: load from HDF5
            for feat in self._snapshot_features:
                raw = ep[feat][t]  # (n_turb,)
                norm = self._normalize_feature(feat, raw, mean_wd)
                obs_parts.append(norm[:, None])  # (n_turb, 1)

            obs = np.concatenate(obs_parts, axis=-1)  # (n_turb, n_global + n_snapshot)
            target_power = self._normalize_feature("power", ep["power"][t])

        # Positions
        positions_norm = layout["positions"] / layout["rotor_diameter"]
        if self.use_wind_relative_pos:
            positions_norm = rotate_positions_wind_relative(positions_norm, mean_wd)

        # Determine obs width for padding
        obs_width = obs.shape[-1]

        # Pad
        obs_padded = np.zeros((self.max_turbines, obs_width), dtype=np.float32)
        obs_padded[:n_turb] = obs

        target_padded = np.zeros(self.max_turbines, dtype=np.float32)
        target_padded[:n_turb] = target_power

        positions_padded = np.zeros((self.max_turbines, 2), dtype=np.float32)
        positions_padded[:n_turb] = positions_norm

        attention_mask = np.ones(self.max_turbines, dtype=bool)
        attention_mask[:n_turb] = False

        sample = {
            "obs": obs_padded,
            "positions": positions_padded,
            "attention_mask": attention_mask,
            "target_power": target_padded,
            "n_turbines": n_turb,
            "layout_idx": li,
        }

        if "receptivity" in layout:
            n_dirs = layout["receptivity"].shape[1]

            recep = layout["receptivity"]
            infl = layout["influence"]

            if self.rotate_profiles:
                recep = rotate_profiles_numpy(recep, mean_wd)
                infl = rotate_profiles_numpy(infl, mean_wd)

            recep_padded = np.zeros((self.max_turbines, n_dirs), dtype=np.float32)
            recep_padded[:n_turb] = recep
            sample["receptivity"] = recep_padded

            infl_padded = np.zeros((self.max_turbines, n_dirs), dtype=np.float32)
            infl_padded[:n_turb] = infl
            sample["influence"] = infl_padded

        return sample


# =============================================================================
# DATALOADER HELPER
# =============================================================================

def create_pretrain_dataloader(
    layout_files: List[str],
    history_length: int = 15,
    batch_size: int = 256,
    max_turbines: Optional[int] = None,
    snapshot_mode: bool = False,
    skip_steps: int = 1,
    # --- Global features ---
    global_features: List[str] = [],
    # --- Actions ---
    action_type: Optional[str] = "wind",
    # --- Preprocessing flags ---
    scaling_limits: Optional[Dict[str, Tuple[float, float]]] = None,
    use_wd_deviation: bool = False,
    wd_scale_range: float = 90.0,
    use_wind_relative_pos: bool = True,
    rotate_profiles: bool = True,
    **kwargs,
) -> DataLoader:
    """
    Create a DataLoader for pretraining.

    Args:
        layout_files: HDF5 files to load
        history_length: Timesteps of history (ignored if snapshot_mode)
        batch_size: Batch size
        max_turbines: Padding target (None=auto)
        snapshot_mode: If True, use single-step snapshots instead of history
        skip_steps: Subsample every N steps (snapshot mode only)
        global_features: Features to replace with episode-level scalars
        scaling_limits: Dict of feature → (min, max) for normalization
        use_wd_deviation: Convert WD to deviation from episode mean
        wd_scale_range: ±degrees for WD deviation scaling
        use_wind_relative_pos: Transform positions to wind-relative frame
        rotate_profiles: Rotate profiles to wind-relative frame
    """
    common_kwargs = dict(
        global_features=global_features,
        action_type=action_type,
        scaling_limits=scaling_limits,
        use_wd_deviation=use_wd_deviation,
        wd_scale_range=wd_scale_range,
        use_wind_relative_pos=use_wind_relative_pos,
        rotate_profiles=rotate_profiles,
    )

    if snapshot_mode:
        dataset = WindFarmSnapshotDataset(
            layout_files=layout_files,
            max_turbines=max_turbines,
            skip_steps=skip_steps,
            **common_kwargs,
        )
    else:
        dataset = WindFarmPretrainDataset(
            layout_files=layout_files,
            history_length=history_length,
            max_turbines=max_turbines,
            **common_kwargs,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        # num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        **kwargs,
    )


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    import sys
    from glob import glob

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./pretrain_data"
    files = sorted(glob(f"{data_dir}/layout_*.h5"))

    if not files:
        print(f"No layout files found in {data_dir}")
        sys.exit(1)

    # --- Test standard mode ---
    print("=" * 60)
    print("Testing STANDARD history dataset (per-turbine ws/wd/yaw)")
    print("=" * 60)
    loader = create_pretrain_dataloader(
        files,
        history_length=15,
        batch_size=32,
        num_workers=0,
        use_wd_deviation=True,
        use_wind_relative_pos=True,
        rotate_profiles=True,
    )
    batch = next(iter(loader))
    print(f"  obs:           {batch['obs'].shape}")            # (32, max_turb, 45)
    print(f"  positions:     {batch['positions'].shape}")
    print(f"  attention_mask:{batch['attention_mask'].shape}")
    print(f"  target_power:  {batch['target_power'].shape}")
    print(f"  obs range:     [{batch['obs'].min():.3f}, {batch['obs'].max():.3f}]")
    if "receptivity" in batch:
        print(f"  receptivity:   {batch['receptivity'].shape}")

    # --- Test global_ws only (obs_dim = 1 + 15*2 = 31) ---
    print("\n" + "=" * 60)
    print("Testing GLOBAL WS only (mean_ws + wd_history + yaw_history)")
    print("=" * 60)
    loader_gws = create_pretrain_dataloader(
        files,
        history_length=15,
        batch_size=32,
        num_workers=0,
        global_features=["ws"],
    )
    batch_gws = next(iter(loader_gws))
    print(f"  obs:           {batch_gws['obs'].shape}")         # (32, max_turb, 31)
    n_turb = (~batch_gws['attention_mask'][0]).sum().item()
    ws_vals = batch_gws['obs'][0, :n_turb, 0]
    print(f"  ws identical across turbines: {ws_vals.std().item() < 1e-6}")
    print(f"  wd varies across turbines:    {batch_gws['obs'][0, :n_turb, 1:16].std(dim=0).mean().item() > 0}")

    # --- Test both global (obs_dim = 2 + 15 = 17) ---
    print("\n" + "=" * 60)
    print("Testing GLOBAL WS+WD (mean_ws + mean_wd + yaw_history)")
    print("=" * 60)
    loader_both = create_pretrain_dataloader(
        files,
        history_length=15,
        batch_size=32,
        num_workers=0,
        global_features=["ws", "wd"],
    )
    batch_both = next(iter(loader_both))
    print(f"  obs:           {batch_both['obs'].shape}")        # (32, max_turb, 17)
    ws_vals = batch_both['obs'][0, :n_turb, 0]
    wd_vals = batch_both['obs'][0, :n_turb, 1]
    print(f"  ws identical across turbines: {ws_vals.std().item() < 1e-6}")
    print(f"  wd identical across turbines: {wd_vals.std().item() < 1e-6}")
    print(f"  yaw varies across turbines:   {batch_both['obs'][0, :n_turb, 2:].std(dim=0).mean().item() > 0}")

    # --- Test global snapshot mode ---
    print("\n" + "=" * 60)
    print("Testing GLOBAL WS snapshot dataset")
    print("=" * 60)
    loader_snap = create_pretrain_dataloader(
        files,
        snapshot_mode=True,
        batch_size=32,
        global_features=["ws"],
    )
    batch_snap = next(iter(loader_snap))
    print(f"  obs:           {batch_snap['obs'].shape}")
    print(f"  obs range:     [{batch_snap['obs'].min():.3f}, {batch_snap['obs'].max():.3f}]")

    print("\nAll good!")