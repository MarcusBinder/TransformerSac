"""
PyTorch Dataset for Wind Farm Pretraining Data

Reads HDF5 files produced by collect_pretrain_data.py and serves
(obs, positions, profiles, targets) samples with configurable history windows.

Supports two modes:
    1. Snapshot mode (history_length=1): For masked prediction on single timesteps
    2. History mode (history_length>1): Stacks past observations like the RL agent sees

Author: Marcus (DTU Wind Energy)
"""

import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional
from pathlib import Path


class WindFarmPretrainDataset(Dataset):
    """
    Serves (obs, positions, profiles, target_power) for pretraining.

    Each sample is a single timestep with history_length context,
    drawn from a random episode across all layout files.

    Padding: all samples are padded to max_turbines (across all layouts)
    with attention_mask indicating padded positions.
    """

    def __init__(
        self,
        layout_files: List[str],
        history_length: int = 15,
        max_turbines: Optional[int] = None,
        features: List[str] = ["ws", "wd", "yaw", "power"],
    ):
        """
        Args:
            layout_files: List of HDF5 file paths (one per layout)
            history_length: Number of past timesteps to stack per feature
            max_turbines: Pad to this many turbines (None = auto from data)
            features: Which per-turbine features to include in obs
        """
        self.history_length = history_length
        self.features = features

        # Build index: list of (layout_idx, ep_key, timestep)
        self.index = []
        self.layouts = []

        print(f"Loading dataset from {len(layout_files)} layout file(s)...")

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

                # Load episode-level metadata (for conditioning, if needed later)
                episode_meta = {}
                for ep_key in sorted(f["episodes"].keys()):
                    ep = f[f"episodes/{ep_key}"]
                    n_steps = int(ep.attrs["n_steps"])
                    episode_meta[ep_key] = {
                        "n_steps": n_steps,
                        "mean_ws": float(ep.attrs["mean_ws"]),
                        "mean_wd": float(ep.attrs["mean_wd"]),
                        "mean_ti": float(ep.attrs["mean_ti"]),
                    }

                    # Valid timesteps: need history_length steps of context before
                    for t in range(history_length, n_steps):
                        self.index.append((li, ep_key, t))

                layout_info["episode_meta"] = episode_meta
                self.layouts.append(layout_info)

                print(f"  [{li}] {layout_info['layout_name']}: "
                      f"{layout_info['n_turbines']} turbines, "
                      f"{len(episode_meta)} episodes, "
                      f"{sum(m['n_steps'] for m in episode_meta.values())} total steps")

        # Determine max turbines for padding
        self.max_turbines = max_turbines or max(l["n_turbines"] for l in self.layouts)
        self.obs_dim = history_length * len(features)

        print(f"Dataset ready: {len(self.index)} samples, "
              f"max_turbines={self.max_turbines}, "
              f"obs_dim={self.obs_dim} ({history_length} × {len(features)} features)")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        li, ep_key, t = self.index[idx]
        layout = self.layouts[li]
        n_turb = layout["n_turbines"]

        with h5py.File(layout["path"], "r") as f:
            ep = f[f"episodes/{ep_key}"]

            # Build stacked observation: (n_turb, history_length * n_features)
            obs_parts = []
            for feat in self.features:
                # Slice: (history_length, n_turb) → transpose to (n_turb, history_length)
                hist = ep[feat][t - self.history_length : t]  # (H, n_turb)
                obs_parts.append(hist.T)  # (n_turb, H)

            obs = np.concatenate(obs_parts, axis=-1)  # (n_turb, H * n_features)

            # Target: current power at this timestep
            target_power = ep["power"][t - 1]  # (n_turb,) — last step in history window

            # Current actions (for next-step prediction, if needed)
            actions = ep["actions"][t - 1]  # (n_turb,)

        # Episode metadata
        ep_meta = layout["episode_meta"][ep_key]

        # --- Pad to max_turbines ---
        obs_padded = np.zeros((self.max_turbines, obs.shape[-1]), dtype=np.float32)
        obs_padded[:n_turb] = obs

        target_padded = np.zeros(self.max_turbines, dtype=np.float32)
        target_padded[:n_turb] = target_power

        actions_padded = np.zeros(self.max_turbines, dtype=np.float32)
        actions_padded[:n_turb] = actions

        positions_padded = np.zeros((self.max_turbines, 2), dtype=np.float32)
        positions_padded[:n_turb] = layout["positions"] / layout["rotor_diameter"]

        attention_mask = np.ones(self.max_turbines, dtype=bool)  # True = padding
        attention_mask[:n_turb] = False

        sample = {
            "obs": obs_padded,                          # (max_turb, obs_dim)
            "positions": positions_padded,              # (max_turb, 2) normalized
            "attention_mask": attention_mask,            # (max_turb,) True=padding
            "target_power": target_padded,              # (max_turb,)
            "actions": actions_padded,                  # (max_turb,)
            "n_turbines": n_turb,                       # scalar
            "layout_idx": li,                           # scalar
            # Episode-level conditions (useful for conditional pretraining)
            "mean_ws": np.float32(ep_meta["mean_ws"]),
            "mean_wd": np.float32(ep_meta["mean_wd"]),
            "mean_ti": np.float32(ep_meta["mean_ti"]),
        }

        # Profiles (if available)
        if "receptivity" in layout:
            n_dirs = layout["receptivity"].shape[1]

            recep_padded = np.zeros((self.max_turbines, n_dirs), dtype=np.float32)
            recep_padded[:n_turb] = layout["receptivity"]
            sample["receptivity"] = recep_padded

            infl_padded = np.zeros((self.max_turbines, n_dirs), dtype=np.float32)
            infl_padded[:n_turb] = layout["influence"]
            sample["influence"] = infl_padded

        return sample


# =============================================================================
# SNAPSHOT DATASET (no history, for pure masked prediction on steady-state)
# =============================================================================

class WindFarmSnapshotDataset(Dataset):
    """
    Simplified dataset for snapshot-based pretraining (history_length=1).

    Each sample is a single timestep with per-turbine (ws, wd, yaw, power).
    Obs dim = n_features (4), not n_features * history.
    
    Useful for:
    - Masked turbine prediction from PyWake steady-state data
    - Quick pretraining experiments without temporal context
    """

    def __init__(
        self,
        layout_files: List[str],
        max_turbines: Optional[int] = None,
        features: List[str] = ["ws", "wd", "yaw", "power"],
        skip_steps: int = 1,  # Subsample: take every N-th step (reduces correlation)
    ):
        self.features = features
        self.index = []
        self.layouts = []

        for li, path in enumerate(layout_files):
            with h5py.File(path, "r") as f:
                layout_info = {
                    "path": str(path),
                    "n_turbines": int(f.attrs["n_turbines"]),
                    "rotor_diameter": float(f.attrs["rotor_diameter"]),
                    "positions": f["positions/xy"][:].astype(np.float32),
                }
                if "profiles" in f:
                    layout_info["receptivity"] = f["profiles/receptivity"][:].astype(np.float32)
                    layout_info["influence"] = f["profiles/influence"][:].astype(np.float32)

                self.layouts.append(layout_info)

                for ep_key in sorted(f["episodes"].keys()):
                    n_steps = int(f[f"episodes/{ep_key}"].attrs["n_steps"])
                    for t in range(0, n_steps, skip_steps):
                        self.index.append((li, ep_key, t))

        self.max_turbines = max_turbines or max(l["n_turbines"] for l in self.layouts)
        print(f"Snapshot dataset: {len(self.index)} samples, max_turb={self.max_turbines}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        li, ep_key, t = self.index[idx]
        layout = self.layouts[li]
        n_turb = layout["n_turbines"]

        with h5py.File(layout["path"], "r") as f:
            ep = f[f"episodes/{ep_key}"]
            obs_parts = [ep[feat][t] for feat in self.features]  # each (n_turb,)
            obs = np.stack(obs_parts, axis=-1)  # (n_turb, n_features)
            target_power = ep["power"][t]

        # Pad
        obs_padded = np.zeros((self.max_turbines, len(self.features)), dtype=np.float32)
        obs_padded[:n_turb] = obs

        target_padded = np.zeros(self.max_turbines, dtype=np.float32)
        target_padded[:n_turb] = target_power

        positions_padded = np.zeros((self.max_turbines, 2), dtype=np.float32)
        positions_padded[:n_turb] = layout["positions"] / layout["rotor_diameter"]

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
            recep_padded = np.zeros((self.max_turbines, n_dirs), dtype=np.float32)
            recep_padded[:n_turb] = layout["receptivity"]
            sample["receptivity"] = recep_padded

            infl_padded = np.zeros((self.max_turbines, n_dirs), dtype=np.float32)
            infl_padded[:n_turb] = layout["influence"]
            sample["influence"] = infl_padded

        return sample


# =============================================================================
# DATALOADER HELPER
# =============================================================================

def create_pretrain_dataloader(
    layout_files: List[str],
    history_length: int = 15,
    batch_size: int = 256,
    num_workers: int = 4,
    max_turbines: Optional[int] = None,
    snapshot_mode: bool = False,
    skip_steps: int = 1,
    **kwargs,
) -> DataLoader:
    """
    Create a DataLoader for pretraining.
    
    Args:
        layout_files: HDF5 files to load
        history_length: Timesteps of history (ignored if snapshot_mode)
        batch_size: Batch size
        num_workers: DataLoader workers
        max_turbines: Padding target (None=auto)
        snapshot_mode: If True, use single-step snapshots instead of history
        skip_steps: Subsample every N steps (snapshot mode only)
    """
    if snapshot_mode:
        dataset = WindFarmSnapshotDataset(
            layout_files=layout_files,
            max_turbines=max_turbines,
            skip_steps=skip_steps,
        )
    else:
        dataset = WindFarmPretrainDataset(
            layout_files=layout_files,
            history_length=history_length,
            max_turbines=max_turbines,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
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

    print("Testing history dataset (history_length=15)...")
    loader = create_pretrain_dataloader(files, history_length=15, batch_size=32, num_workers=0)
    batch = next(iter(loader))
    print(f"  obs:           {batch['obs'].shape}")           # (32, max_turb, 60)
    print(f"  positions:     {batch['positions'].shape}")     # (32, max_turb, 2)
    print(f"  attention_mask:{batch['attention_mask'].shape}") # (32, max_turb)
    print(f"  target_power:  {batch['target_power'].shape}")  # (32, max_turb)
    if "receptivity" in batch:
        print(f"  receptivity:   {batch['receptivity'].shape}")  # (32, max_turb, 360)

    print("\nTesting snapshot dataset...")
    loader_snap = create_pretrain_dataloader(files, snapshot_mode=True, batch_size=32, num_workers=0)
    batch_snap = next(iter(loader_snap))
    print(f"  obs:           {batch_snap['obs'].shape}")      # (32, max_turb, 4)
    print(f"  target_power:  {batch_snap['target_power'].shape}")

    print("\nAll good!")