"""
Behavioral Cloning Fine-Tuning for Wind Farm Transformer

Step 2 in the 3-step pretraining pipeline:
    1. pretrain_power.py  — Self-supervised pretraining (power prediction / masked turbine)
    2. finetune_bc.py     — Behavioral cloning from expert controller (THIS SCRIPT)
    3. transformer_sac_windfarm_v26.py — Online RL fine-tuning (SAC)

Trains the TransformerActor to imitate an expert controller (PyWake or greedy)
using demonstration data collected by make_datasets.py.

The encoder can be initialized from a pretrain_power checkpoint (step 1).
Differential learning rates allow the pretrained encoder to be fine-tuned
gently while the action heads learn from scratch.

Saves checkpoints compatible with the RL loading system (step 3) via both:
    - encoder_state_dict  (for --pretrain_checkpoint in RL)
    - actor_state_dict    (for direct actor loading)

Usage:
    # BC from scratch (no pretraining)
    python finetune_bc.py --data-dir ./pretrain_data --policy pywake

    # BC with pretrained encoder from step 1
    python finetune_bc.py --data-dir ./pretrain_data --policy pywake \\
        --pretrain-checkpoint ./pretrain_checkpoints/best.pt

    # With differential learning rates
    python finetune_bc.py --data-dir ./pretrain_data --policy pywake \\
        --pretrain-checkpoint ./pretrain_checkpoints/best.pt \\
        --encoder-lr 1e-5 --head-lr 3e-4

Author: Marcus (DTU Wind Energy)
"""

import os
import json
import time
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from glob import glob
from pathlib import Path

import tyro

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not installed — logging disabled. Install with: pip install wandb")

# Import real RL model architecture (ensures identical weight keys)
from networks import (
    TransformerActor,
    TransformerEncoder,
    TransformerEncoderLayer,
    create_positional_encoding,
    create_profile_encoding,
    LOG_STD_MIN,
    LOG_STD_MAX,
)


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class Args:
    """Command-line arguments for behavioral cloning fine-tuning."""

    # === Experiment Settings ===
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """Experiment name (used for run naming)."""
    seed: int = 42
    """Random seed for reproducibility."""
    track: bool = True
    """Enable wandb tracking."""
    wandb_project_name: str = "windfarm-bc"
    """Wandb project name."""
    wandb_entity: Optional[str] = None
    """Wandb entity (team/user). None = default entity."""
    wandb_tags: Optional[str] = None
    """Comma-separated wandb tags."""

    # === Data Settings ===
    data_dir: str = "./pretrain_data"
    """Directory containing layout_*.h5 files."""
    policy: str = "pywake"
    """Filter HDF5 files by policy name (e.g. 'pywake', 'greedy'). Empty = all files."""
    history_length: int = 15
    """Number of timesteps of history per feature (must match RL config)."""
    features: str = "ws,wd,yaw,power"
    """Comma-separated input features. Must match RL observation features."""

    # === Preprocessing (must match RL config) ===
    use_wd_deviation: bool = False
    """Convert wind direction to deviation from episode mean."""
    use_wind_relative_pos: bool = True
    """Transform turbine positions to wind-relative frame."""
    rotate_profiles: bool = True
    """Rotate profiles to wind-relative frame."""
    wd_scale_range: float = 90.0
    """Wind direction deviation range for scaling (±degrees → [-1,1])."""

    # === Scaling (should match RL environment bounds) ===
    ws_min: float = 0.0
    """Min wind speed for scaling."""
    ws_max: float = 30.0
    """Max wind speed for scaling."""
    wd_min: float = 0.0
    """Min wind direction for scaling (ignored if use_wd_deviation)."""
    wd_max: float = 360.0
    """Max wind direction for scaling (ignored if use_wd_deviation)."""
    yaw_min: float = -45.0
    """Min yaw angle for scaling."""
    yaw_max: float = 45.0
    """Max yaw angle for scaling."""
    power_max: float = 10e6
    """Max power for normalization (W). DTU10MW ≈ 10.64e6."""

    # === Transformer Architecture (auto-overridden if loading pretrain checkpoint) ===
    embed_dim: int = 64
    """Transformer hidden dimension."""
    pos_embed_dim: int = 16
    """Dimension for positional encoding MLP."""
    num_heads: int = 4
    """Number of attention heads."""
    num_layers: int = 2
    """Number of transformer encoder layers."""
    mlp_ratio: float = 2.0
    """FFN hidden dim = embed_dim * mlp_ratio."""
    dropout: float = 0.1
    """Dropout rate."""

    # === Positional Encoding (must match RL config) ===
    pos_encoding_type: Optional[str] = None
    """Positional encoding type. Must match RL config."""
    rel_pos_hidden_dim: int = 64
    """Hidden dim for relative position MLP."""
    rel_pos_per_head: bool = True
    """Whether relative bias is per-head."""
    pos_embedding_mode: str = "concat"
    """'add' or 'concat' positional embedding to token."""

    # === Profile Encoding (must match RL config) ===
    profile_encoding_type: Optional[str] = None
    """Profile encoding type. Must match RL config."""
    profile_encoder_hidden: int = 128
    """Hidden dim in profile encoder."""
    n_profile_directions: int = 360
    """Number of directions in profile."""
    profile_fusion_type: str = "add"
    """'add' or 'joint' fusion of receptivity and influence."""
    profile_embed_mode: str = "add"
    """'add' or 'concat' profile embedding into token."""
    profile_encoder_kwargs: str = "{}"
    """JSON string of encoder-specific kwargs."""

    # === Action Settings ===
    action_type: str = "wind"
    """Which action representation to clone: 'wind' (target setpoint) or 'yaw' (delta yaw).
    Must match the ActionMethod used in RL training (step 3)."""
    action_dim_per_turbine: int = 1
    """Action dimension per turbine (1 for yaw)."""
    action_scale: float = 1.0
    """Action scale (for tanh squashing). Must match RL config."""
    action_bias: float = 0.0
    """Action bias (for tanh squashing). Must match RL config."""

    # === Training Hyperparameters ===
    epochs: int = 100
    """Number of training epochs."""
    batch_size: int = 256
    """Batch size."""
    head_lr: float = 3e-4
    """Learning rate for action heads (fc_mean, fc_logstd)."""
    encoder_lr: float = 3e-5
    """Learning rate for encoder (obs_encoder, transformer, etc.)."""
    weight_decay: float = 1e-4
    """AdamW weight decay."""
    val_split: float = 0.1
    """Fraction of data held out for validation."""
    patience: int = 15
    """Early stopping patience (epochs without improvement)."""

    # === Loss Settings ===
    loss_type: str = "mse"
    """Loss type: 'mse' (on mean action), 'logprob' (log-likelihood), or 'combined'."""
    logprob_weight: float = 1.0
    """Weight for log-prob loss when using 'combined'."""
    mse_weight: float = 1.0
    """Weight for MSE loss when using 'combined'."""
    target_log_std: float = -1.0
    """Target log_std for regularization (pre-tanh space)."""
    std_reg_weight: float = 0.1
    """Weight for log_std regularization loss."""

    # === Pretrained Encoder Loading ===
    pretrain_checkpoint: Optional[str] = None
    """Path to pretrained encoder checkpoint from pretrain_power.py (step 1)."""
    freeze_encoder_epochs: int = 0
    """Freeze encoder for first N epochs (0 = no freeze)."""

    # === Output / Checkpointing ===
    save_dir: str = "./bc_checkpoints"
    """Directory for saving checkpoints."""
    save_every: int = 10
    """Save checkpoint every N epochs."""

    # === Device ===
    device: str = "auto"
    """Device: 'auto', 'cuda', 'cpu'."""


# =============================================================================
# DATASET
# =============================================================================

class WindFarmBCDataset(Dataset):
    """
    Behavioral cloning dataset from HDF5 files.

    All data is loaded into memory and pre-stacked into contiguous tensors
    for fast indexing without multiprocessing DataLoader workers.

    Each sample contains:
        obs:             (n_turbines, n_features × history_length) scaled to [-1, 1]
        positions:       (n_turbines, 2) turbine positions
        attention_mask:  (n_turbines,)  True = padding
        expert_actions:  (n_turbines,) expert actions in [-1, 1]
        receptivity:     (n_turbines, n_dirs) [optional]
        influence:       (n_turbines, n_dirs) [optional]
    """

    def __init__(
        self,
        layout_files: List[str],
        features: List[str] = ("ws", "wd", "yaw", "power"),
        history_length: int = 15,
        max_turbines: Optional[int] = None,
        action_type: str = "wind",
        # Scaling
        scaling_ranges: Optional[dict] = None,
        # Preprocessing
        use_wd_deviation: bool = False,
        wd_scale_range: float = 90.0,
        use_wind_relative_pos: bool = True,
        rotate_profiles: bool = True,
    ):
        super().__init__()
        self.features = list(features)
        self.history_length = history_length
        self.action_type = action_type
        self.use_wd_deviation = use_wd_deviation
        self.wd_scale_range = wd_scale_range
        self.use_wind_relative_pos = use_wind_relative_pos
        self.rotate_profiles = rotate_profiles

        # Default scaling ranges (matching WindFarmEnv defaults)
        self.scaling_ranges = scaling_ranges or {
            "ws": (0.0, 30.0),
            "wd": (0.0, 360.0),
            "yaw": (-45.0, 45.0),
            "power": (0.0, 10e6),
        }

        # Temporary list for accumulation during loading
        self._samples_list: List[dict] = []

        # Episode boundaries: list of (start_idx, end_idx, layout_file, ep_key)
        self.episode_boundaries: List[Tuple[int, int, str, str]] = []

        # Load all data into self._samples_list
        self._load_all(layout_files, max_turbines)

        # Stack into contiguous tensors and free the list
        self._consolidate_tensors()

    def _scale_feature(self, data: np.ndarray, feature: str) -> np.ndarray:
        """Scale a feature array to [-1, 1] using known ranges."""
        lo, hi = self.scaling_ranges[feature]
        if hi == lo:
            return np.zeros_like(data)
        scaled = 2.0 * (data - lo) / (hi - lo) - 1.0
        return np.clip(scaled, -1.0, 1.0)

    def _rotate_positions(self, xy: np.ndarray, mean_wd_deg: float) -> np.ndarray:
        """Rotate positions to wind-relative frame (wind from 270°)."""
        angle_rad = np.deg2rad(270.0 - mean_wd_deg)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
        return (xy @ R.T)

    def _normalize_positions(self, xy: np.ndarray) -> np.ndarray:
        """Center and normalize positions to roughly [-1, 1]."""
        xy = xy - xy.mean(axis=0, keepdims=True)
        scale = np.abs(xy).max() + 1e-8
        return xy / scale

    def _rotate_profile(self, profile: np.ndarray, shift_dirs: int) -> np.ndarray:
        """Rotate a directional profile by shift_dirs bins."""
        return np.roll(profile, shift_dirs, axis=-1)

    def _load_all(self, layout_files: List[str], max_turbines: Optional[int]):
        """Load all episodes from all layout files and create sample indices."""
        all_n_turbs = []
        for fpath in layout_files:
            with h5py.File(fpath, "r") as f:
                all_n_turbs.append(int(f.attrs["n_turbines"]))

        if max_turbines is None:
            max_turbines = max(all_n_turbs) if all_n_turbs else 0
        self.max_turbines = max_turbines

        for fpath in layout_files:
            self._load_layout(fpath)

        print(f"  Total BC samples: {len(self._samples_list)} "
              f"from {len(self.episode_boundaries)} episodes")

    def _load_layout(self, fpath: str):
        """Load all episodes from a single layout file."""
        with h5py.File(fpath, "r") as f:
            n_turb = int(f.attrs["n_turbines"])
            positions_raw = f["positions/xy"][:]  # (n_turb, 2)

            has_profiles = "profiles" in f
            recep_raw = f["profiles/receptivity"][:] if has_profiles else None
            infl_raw = f["profiles/influence"][:] if has_profiles else None
            n_dirs = recep_raw.shape[-1] if has_profiles else 0

            ep_keys = sorted(f["episodes"].keys())
            layout_name = f.attrs.get("layout_name", Path(fpath).stem)
            policy = f.attrs.get("policy", "unknown")
            print(f"  Loading {fpath}: {layout_name} ({policy}), "
                  f"{n_turb} turbines, {len(ep_keys)} episodes")

            for ep_key in ep_keys:
                ep = f[f"episodes/{ep_key}"]
                n_steps = int(ep.attrs["n_steps"])
                mean_wd = float(ep.attrs["mean_wd"])

                if n_steps < self.history_length:
                    continue

                # Load raw per-step features
                feature_data = {}
                for feat in self.features:
                    feature_data[feat] = ep[feat][:]  # (n_steps, n_turb)
                actions = ep[f"actions_{self.action_type}"][:]  # (n_steps, n_turb)

                # Handle wind direction deviation
                if self.use_wd_deviation and "wd" in feature_data:
                    deviation = feature_data["wd"] - mean_wd
                    feature_data["wd"] = ((deviation + 180) % 360) - 180  # circular wrap

                # Scale features to [-1, 1]
                scaled = {}
                for feat in self.features:
                    if feat == "wd" and self.use_wd_deviation:
                        scaled[feat] = np.clip(
                            feature_data[feat] / self.wd_scale_range, -1.0, 1.0
                        ).astype(np.float32)
                    else:
                        scaled[feat] = self._scale_feature(
                            feature_data[feat], feat
                        ).astype(np.float32)

                # Prepare positions (fixed per episode)
                positions = positions_raw.copy()
                if self.use_wind_relative_pos:
                    positions = self._rotate_positions(positions, mean_wd)
                positions = self._normalize_positions(positions)

                # Prepare profiles (fixed per layout, optionally rotated per episode)
                recep = None
                infl = None
                if has_profiles and recep_raw is not None:
                    recep = recep_raw.copy()
                    infl = infl_raw.copy()
                    if self.rotate_profiles:
                        shift = int(round(mean_wd / 360.0 * n_dirs)) % n_dirs
                        recep = self._rotate_profile(recep, -shift)
                        infl = self._rotate_profile(infl, -shift)

                # Pad to max_turbines if needed
                pad_n = self.max_turbines - n_turb
                attention_mask = np.zeros(self.max_turbines, dtype=bool)
                if pad_n > 0:
                    attention_mask[n_turb:] = True
                    positions = np.pad(positions, ((0, pad_n), (0, 0)))
                    for feat in self.features:
                        scaled[feat] = np.pad(scaled[feat], ((0, 0), (0, pad_n)))
                    actions = np.pad(actions, ((0, 0), (0, pad_n)))
                    if recep is not None:
                        recep = np.pad(recep, ((0, pad_n), (0, 0)))
                        infl = np.pad(infl, ((0, pad_n), (0, 0)))

                # Create sliding window samples
                H = self.history_length
                ep_start = len(self._samples_list)
                for t in range(H - 1, n_steps):
                    # Build obs: (max_turbines, n_features × H)
                    obs_parts = []
                    for feat in self.features:
                        feat_hist = scaled[feat][t - H + 1 : t + 1, :].T  # (max_turb, H)
                        obs_parts.append(feat_hist)
                    obs = np.concatenate(obs_parts, axis=-1)  # (max_turb, n_feat × H)

                    sample = {
                        "obs": obs.astype(np.float32),
                        "positions": positions.astype(np.float32),
                        "attention_mask": attention_mask.copy(),
                        "expert_actions": actions[t, :].astype(np.float32),
                    }
                    if recep is not None:
                        sample["receptivity"] = recep.astype(np.float32)
                        sample["influence"] = infl.astype(np.float32)

                    self._samples_list.append(sample)

                ep_end = len(self._samples_list)
                if ep_end > ep_start:
                    self.episode_boundaries.append(
                        (ep_start, ep_end, fpath, ep_key)
                    )

    def _consolidate_tensors(self):
        """Stack all samples into contiguous tensors for fast indexing."""
        n = len(self._samples_list)
        if n == 0:
            self.obs = torch.empty(0)
            self.positions = torch.empty(0)
            self.attention_mask = torch.empty(0)
            self.expert_actions = torch.empty(0)
            self.receptivity = None
            self.influence = None
            self._n_samples = 0
            del self._samples_list
            return

        has_profiles = "receptivity" in self._samples_list[0]

        print(f"  Consolidating {n} samples into contiguous tensors...")
        t0 = time.time()

        self.obs = torch.from_numpy(
            np.stack([s["obs"] for s in self._samples_list])
        )
        self.positions = torch.from_numpy(
            np.stack([s["positions"] for s in self._samples_list])
        )
        self.attention_mask = torch.from_numpy(
            np.stack([s["attention_mask"] for s in self._samples_list])
        )
        self.expert_actions = torch.from_numpy(
            np.stack([s["expert_actions"] for s in self._samples_list])
        )

        if has_profiles:
            self.receptivity = torch.from_numpy(
                np.stack([s["receptivity"] for s in self._samples_list])
            )
            self.influence = torch.from_numpy(
                np.stack([s["influence"] for s in self._samples_list])
            )
        else:
            self.receptivity = None
            self.influence = None

        self._n_samples = n

        # Free the temporary list
        del self._samples_list

        mem_mb = (
            self.obs.nbytes + self.positions.nbytes +
            self.attention_mask.nbytes + self.expert_actions.nbytes
        ) / 1e6
        if self.receptivity is not None:
            mem_mb += (self.receptivity.nbytes + self.influence.nbytes) / 1e6

        print(f"  Consolidated in {time.time() - t0:.1f}s "
              f"({mem_mb:.1f} MB in tensors)")

    def __len__(self):
        return self._n_samples

    def __getitem__(self, idx):
        sample = {
            "obs": self.obs[idx],
            "positions": self.positions[idx],
            "attention_mask": self.attention_mask[idx],
            "expert_actions": self.expert_actions[idx],
        }
        if self.receptivity is not None:
            sample["receptivity"] = self.receptivity[idx]
            sample["influence"] = self.influence[idx]
        return sample

    def episode_split(
        self,
        val_fraction: float = 0.1,
        seed: int = 42,
    ) -> Tuple["Subset", "Subset"]:
        """
        Split into train/val by entire episodes (no window overlap leakage).

        Episodes are shuffled and assigned to val until the cumulative sample
        count reaches val_fraction of the total. This guarantees that no two
        sliding windows from the same episode end up in different splits.

        Args:
            val_fraction: Target fraction of *samples* in the validation set.
            seed: Random seed for reproducible episode shuffling.

        Returns:
            (train_subset, val_subset) as torch Subset views into this dataset.
        """
        n_episodes = len(self.episode_boundaries)
        if n_episodes == 0:
            raise ValueError("No episodes loaded — cannot split.")

        rng = np.random.RandomState(seed)
        ep_indices = rng.permutation(n_episodes)

        target_val_samples = int(self._n_samples * val_fraction)
        val_sample_ids = []
        train_sample_ids = []
        val_ep_count = 0

        for ep_idx in ep_indices:
            start, end, _, _ = self.episode_boundaries[ep_idx]
            ep_sample_ids = list(range(start, end))

            if len(val_sample_ids) < target_val_samples:
                val_sample_ids.extend(ep_sample_ids)
                val_ep_count += 1
            else:
                train_sample_ids.extend(ep_sample_ids)

        # Diagnostics
        n_train_ep = n_episodes - val_ep_count
        print(f"  Episode split: {n_train_ep} train episodes ({len(train_sample_ids)} samples), "
              f"{val_ep_count} val episodes ({len(val_sample_ids)} samples)")

        # Sanity check
        assert len(train_sample_ids) + len(val_sample_ids) == self._n_samples, \
            "Episode split lost samples — boundary tracking bug"

        return Subset(self, train_sample_ids), Subset(self, val_sample_ids)


# =============================================================================
# ENCODER UTILS (from pretrain_power.py)
# =============================================================================

ACTOR_HEAD_KEYS = {"fc_mean", "fc_logstd", "action_scale", "action_bias_val"}


def get_encoder_state_dict(actor: TransformerActor) -> dict:
    """Extract only encoder weights from actor (excluding action heads)."""
    return {
        k: v for k, v in actor.state_dict().items()
        if not any(k.startswith(head) for head in ACTOR_HEAD_KEYS)
    }


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

def masked_mse_loss(
    predicted: torch.Tensor,       # (B, N, action_dim)
    target: torch.Tensor,          # (B, N, action_dim) or (B, N)
    attention_mask: torch.Tensor,  # (B, N) True = padding
) -> torch.Tensor:
    """MSE loss over real (non-padded) turbines."""
    if target.dim() == 2:
        target = target.unsqueeze(-1)
    if predicted.dim() == 2:
        predicted = predicted.unsqueeze(-1)

    real_mask = ~attention_mask  # True = real turbine
    n_real = real_mask.sum()

    if n_real == 0:
        return torch.tensor(0.0, device=predicted.device)

    diff = (predicted - target) ** 2
    mask_expanded = real_mask.unsqueeze(-1).float()  # (B, N, 1)
    loss = (diff * mask_expanded).sum() / (n_real * predicted.shape[-1])
    return loss


def masked_logprob_loss(
    mean: torch.Tensor,            # (B, N, action_dim)
    log_std: torch.Tensor,         # (B, N, action_dim)
    expert_actions: torch.Tensor,  # (B, N) or (B, N, action_dim)
    attention_mask: torch.Tensor,  # (B, N) True = padding
    action_scale: float = 1.0,
    action_bias: float = 0.0,
) -> torch.Tensor:
    """
    Negative log-likelihood of expert actions under the squashed Gaussian policy.

    Inverse of the SAC action mapping: action = tanh(x_t) * scale + bias
    """
    eps = 1e-6

    if expert_actions.dim() == 2:
        expert_actions = expert_actions.unsqueeze(-1)

    # Invert the tanh squashing: x_t = atanh((action - bias) / scale)
    normalized = (expert_actions - action_bias) / (action_scale + eps)
    normalized = normalized.clamp(-1 + eps, 1 - eps)
    x_t = torch.atanh(normalized)

    # Evaluate log-probability under the Gaussian
    std = log_std.exp()
    dist = torch.distributions.Normal(mean, std)
    log_p = dist.log_prob(x_t)  # (B, N, action_dim)

    # Tanh Jacobian correction
    y_t = torch.tanh(x_t)
    log_jac = torch.log((action_scale * (1 - y_t.pow(2))).clamp(min=eps))

    # Full log-prob with correction
    log_prob = log_p - log_jac  # (B, N, action_dim)

    # Mask padding
    real_mask = (~attention_mask).unsqueeze(-1).float()  # (B, N, 1)
    n_real = real_mask.sum()

    if n_real == 0:
        return torch.tensor(0.0, device=mean.device)

    # Average NLL over real turbines
    nll = -(log_prob * real_mask).sum() / (n_real * log_prob.shape[-1])
    return nll


def masked_std_reg_loss(
    log_std: torch.Tensor,         # (B, N, action_dim)
    target: float,                 # scalar target log_std
    attention_mask: torch.Tensor,  # (B, N) True = padding
) -> torch.Tensor:
    """Pull log_std toward a target value over real turbines."""
    real_mask = (~attention_mask).unsqueeze(-1).float()
    n_real = real_mask.sum()
    if n_real == 0:
        return torch.tensor(0.0, device=log_std.device)
    diff = (log_std - target) ** 2
    return (diff * real_mask).sum() / (n_real * log_std.shape[-1])


# =============================================================================
# TRAINING / EVALUATION
# =============================================================================

def train_one_epoch(
    actor: TransformerActor,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_type: str = "mse",
    mse_weight: float = 1.0,
    logprob_weight: float = 1.0,
    log_wandb: bool = False,
    target_log_std: float = -1.0,
    std_reg_weight: float = 0.0,
) -> dict:
    """Train for one epoch. Returns dict of average losses."""
    actor.train()
    total_mse = 0.0
    total_logprob = 0.0
    total_std_reg = 0.0
    total_loss = 0.0
    total_log_std = 0.0
    total_std = 0.0
    n_batches = 0

    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        obs = batch["obs"]
        positions = batch["positions"]
        mask = batch["attention_mask"]
        expert_actions = batch["expert_actions"]
        recep = batch.get("receptivity")
        infl = batch.get("influence")

        # Forward pass through actor
        mean, log_std, _ = actor(
            obs, positions,
            key_padding_mask=mask,
            recep_profile=recep,
            influence_profile=infl,
        )

        # Log std diagnostics (safe defaults if no real turbines)
        batch_mean_log_std = 0.0
        batch_mean_std = 0.0
        with torch.no_grad():
            real_mask_bool = ~mask
            if real_mask_bool.any():
                real_log_std = log_std[real_mask_bool.unsqueeze(-1).expand_as(log_std)]
                batch_mean_log_std = real_log_std.mean().item()
                batch_mean_std = real_log_std.exp().mean().item()
        total_log_std += batch_mean_log_std
        total_std += batch_mean_std

        # Mean action (deterministic output)
        mean_action = torch.tanh(mean) * actor.action_scale + actor.action_bias_val

        # Compute losses
        loss = torch.tensor(0.0, device=device)

        if loss_type in ("mse", "combined"):
            mse = masked_mse_loss(mean_action, expert_actions, mask)
            loss = loss + mse_weight * mse
            total_mse += mse.item()

        if loss_type in ("logprob", "combined"):
            nll = masked_logprob_loss(
                mean, log_std, expert_actions, mask,
                action_scale=actor.action_scale.item(),
                action_bias=actor.action_bias_val.item(),
            )
            loss = loss + logprob_weight * nll
            total_logprob += nll.item()

        # Std regularization (always applied when weight > 0)
        if std_reg_weight > 0:
            std_reg = masked_std_reg_loss(log_std, target_log_std, mask)
            loss = loss + std_reg_weight * std_reg
            total_std_reg += std_reg.item()

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if log_wandb:
            global_step = (epoch - 1) * len(loader) + n_batches
            log_dict = {
                "train/step_loss": loss.item(),
                "train/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                "train/mean_log_std": batch_mean_log_std,
                "train/mean_std": batch_mean_std,
                "global_step": global_step,
            }
            if loss_type in ("mse", "combined"):
                log_dict["train/step_mse"] = mse.item()
            if loss_type in ("logprob", "combined"):
                log_dict["train/step_nll"] = nll.item()
            if std_reg_weight > 0:
                log_dict["train/step_std_reg"] = std_reg.item()
            wandb.log(log_dict, step=global_step)

    n = max(n_batches, 1)
    return {
        "loss": total_loss / n,
        "mse": total_mse / n if loss_type in ("mse", "combined") else 0.0,
        "nll": total_logprob / n if loss_type in ("logprob", "combined") else 0.0,
        "std_reg": total_std_reg / n if std_reg_weight > 0 else 0.0,
        "mean_log_std": total_log_std / n,
        "mean_std": total_std / n,
    }


@torch.no_grad()
def evaluate(
    actor: TransformerActor,
    loader: DataLoader,
    device: torch.device,
    loss_type: str = "mse",
    mse_weight: float = 1.0,
    logprob_weight: float = 1.0,
    target_log_std: float = -1.0,
    std_reg_weight: float = 0.0,
) -> dict:
    """Evaluate on validation set. Returns dict of average losses + metrics."""
    actor.eval()
    total_mse = 0.0
    total_logprob = 0.0
    total_std_reg = 0.0
    total_loss = 0.0
    total_mae = 0.0
    total_log_std = 0.0
    total_std = 0.0
    n_batches = 0

    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        obs = batch["obs"]
        positions = batch["positions"]
        mask = batch["attention_mask"]
        expert_actions = batch["expert_actions"]
        recep = batch.get("receptivity")
        infl = batch.get("influence")

        mean, log_std, _ = actor(
            obs, positions,
            key_padding_mask=mask,
            recep_profile=recep,
            influence_profile=infl,
        )

        # Std diagnostics
        real_mask_bool = ~mask
        if real_mask_bool.any():
            real_log_std = log_std[real_mask_bool.unsqueeze(-1).expand_as(log_std)]
            total_log_std += real_log_std.mean().item()
            total_std += real_log_std.exp().mean().item()

        mean_action = torch.tanh(mean) * actor.action_scale + actor.action_bias_val

        # Losses
        loss = torch.tensor(0.0, device=device)

        mse = masked_mse_loss(mean_action, expert_actions, mask)
        total_mse += mse.item()
        if loss_type in ("mse", "combined"):
            loss = loss + mse_weight * mse

        if loss_type in ("logprob", "combined"):
            nll = masked_logprob_loss(
                mean, log_std, expert_actions, mask,
                action_scale=actor.action_scale.item(),
                action_bias=actor.action_bias_val.item(),
            )
            loss = loss + logprob_weight * nll
            total_logprob += nll.item()

        # Std regularization (include in val loss for consistent early stopping)
        if std_reg_weight > 0:
            std_reg = masked_std_reg_loss(log_std, target_log_std, mask)
            loss = loss + std_reg_weight * std_reg
            total_std_reg += std_reg.item()

        total_loss += loss.item()

        # MAE metric (always computed for interpretability)
        if expert_actions.dim() == 2:
            expert_3d = expert_actions.unsqueeze(-1)
        else:
            expert_3d = expert_actions
        real_mask = (~mask).unsqueeze(-1).float()
        n_real = real_mask.sum()
        if n_real > 0:
            mae = ((mean_action - expert_3d).abs() * real_mask).sum() / (n_real * mean_action.shape[-1])
            total_mae += mae.item()

        n_batches += 1

    n = max(n_batches, 1)
    return {
        "loss": total_loss / n,
        "mse": total_mse / n,
        "nll": total_logprob / n if loss_type in ("logprob", "combined") else 0.0,
        "mae": total_mae / n,
        "std_reg": total_std_reg / n if std_reg_weight > 0 else 0.0,
        "mean_log_std": total_log_std / n,
        "mean_std": total_std / n,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = tyro.cli(Args)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # =========================================================================
    # Load pretrain checkpoint config (if provided)
    # =========================================================================
    _pretrain_encoder_sd = None

    if args.pretrain_checkpoint is not None:
        print(f"\n{'='*60}")
        print(f"PRETRAIN CHECKPOINT: loading architecture config")
        print(f"{'='*60}")
        print(f"Checkpoint: {args.pretrain_checkpoint}")

        if not os.path.exists(args.pretrain_checkpoint):
            raise FileNotFoundError(
                f"Pretrain checkpoint not found: {args.pretrain_checkpoint}"
            )

        ckpt = torch.load(
            args.pretrain_checkpoint, map_location="cpu", weights_only=False
        )

        if "args" in ckpt:
            pt_args = ckpt["args"]

            # Override architecture args to match pretrained model
            ARCH_KEYS = [
                "embed_dim", "num_heads", "num_layers", "mlp_ratio",
                "pos_embed_dim", "dropout",
                "pos_encoding_type", "rel_pos_hidden_dim", "rel_pos_per_head",
                "pos_embedding_mode",
                "profile_encoding_type", "profile_encoder_hidden",
                "profile_fusion_type", "profile_embed_mode",
                "profile_encoder_kwargs",
                "n_profile_directions",
            ]

            overrides = []
            for key in ARCH_KEYS:
                if key in pt_args:
                    old_val = getattr(args, key, None)
                    new_val = pt_args[key]
                    if old_val != new_val:
                        overrides.append((key, old_val, new_val))
                        setattr(args, key, new_val)

            if overrides:
                print(f"\n  Overrode {len(overrides)} args from pretrain config:")
                for key, old, new in overrides:
                    print(f"    {key}: {old} → {new}")
            else:
                print(f"\n  All architecture args already match pretrain config ✓")

        # Store encoder state dict for later loading
        if "encoder_state_dict" in ckpt:
            _pretrain_encoder_sd = ckpt["encoder_state_dict"]
            print(f"  Encoder state dict: {len(_pretrain_encoder_sd)} parameter tensors")
        elif "model_state_dict" in ckpt:
            # Fall back: extract actor encoder weights from PowerPredictionModel
            _pretrain_encoder_sd = {
                k.replace("actor.", ""): v
                for k, v in ckpt["model_state_dict"].items()
                if k.startswith("actor.") and not any(
                    k.replace("actor.", "").startswith(h) for h in ACTOR_HEAD_KEYS
                )
            }
            print(f"  Extracted encoder from model_state_dict: {len(_pretrain_encoder_sd)} tensors")

        print(f"{'='*60}\n")
        del ckpt

    # =========================================================================
    # Wandb
    # =========================================================================
    use_wandb = WANDB_AVAILABLE and args.track

    if use_wandb:
        mode = f"hist{args.history_length}"
        pretrain_tag = "pretrained" if args.pretrain_checkpoint else "scratch"
        run_name = (
            f"{args.exp_name}_{args.policy}_{args.action_type}_{mode}"
            f"_e{args.embed_dim}_L{args.num_layers}_{pretrain_tag}"
        )
        tags = [t.strip() for t in args.wandb_tags.split(",")] if args.wandb_tags else None
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            name=run_name,
            tags=tags,
            config=vars(args),
        )
        print(f"Wandb: logging to {args.wandb_project_name}/{run_name}")
    else:
        print("Wandb: disabled")

    # =========================================================================
    # Data
    # =========================================================================
    if args.policy:
        files = sorted(glob(f"{args.data_dir}/layout_*_{args.policy}*.h5"))
        if not files:
            files = sorted(glob(f"{args.data_dir}/layout_*.h5"))
            print(f"No files matching policy '{args.policy}', using all {len(files)} files")
    else:
        files = sorted(glob(f"{args.data_dir}/layout_*.h5"))

    if not files:
        print(f"No layout files found in {args.data_dir}")
        return
    print(f"Found {len(files)} layout file(s)")

    features = [f.strip() for f in args.features.split(",")]
    print(f"Features: {features}")

    scaling_ranges = {
        "ws": (args.ws_min, args.ws_max),
        "wd": (args.wd_min, args.wd_max),
        "yaw": (args.yaw_min, args.yaw_max),
        "power": (0.0, args.power_max),
    }

    dataset = WindFarmBCDataset(
        layout_files=files,
        features=features,
        history_length=args.history_length,
        max_turbines=None,
        action_type=args.action_type,
        scaling_ranges=scaling_ranges,
        use_wd_deviation=args.use_wd_deviation,
        wd_scale_range=args.wd_scale_range,
        use_wind_relative_pos=args.use_wind_relative_pos,
        rotate_profiles=args.rotate_profiles,
    )

    if len(dataset) == 0:
        print("ERROR: No valid samples found. Check data and history_length.")
        return

    # Train/val split (by episode to avoid window-overlap leakage)
    train_set, val_set = dataset.episode_split(
        val_fraction=args.val_split,
        seed=args.seed,
    )
    n_train = len(train_set)
    n_val = len(val_set)
    print(f"Train: {n_train} samples, Val: {n_val} samples")

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    # --- Dimensions from first sample ---
    sample = dataset[0]
    obs_dim = sample["obs"].shape[-1]
    max_turbines = sample["obs"].shape[0]
    has_profiles = "receptivity" in sample
    n_profile_dirs = sample["receptivity"].shape[-1] if has_profiles else 0

    print(f"\nModel config:")
    print(f"  obs_dim (features × history): {obs_dim}")
    print(f"  max_turbines: {max_turbines}")
    print(f"  features: {features} ({len(features)} × {args.history_length} = {obs_dim})")
    print(f"  action_type: {args.action_type}")
    print(f"  profiles: {'yes, ' + str(n_profile_dirs) + ' dirs' if has_profiles else 'no'}")
    print(f"  embed_dim: {args.embed_dim}")
    print(f"  num_heads: {args.num_heads}")
    print(f"  num_layers: {args.num_layers}")
    print(f"  loss_type: {args.loss_type}")
    if args.std_reg_weight > 0:
        print(f"  std_reg: weight={args.std_reg_weight}, target_log_std={args.target_log_std}")

    # =========================================================================
    # Build actor
    # =========================================================================
    # NOTE: obs_dim mismatch is expected when pretraining used 3 features (ws, wd, yaw)
    # but BC uses 4 features (ws, wd, yaw, power). The obs_encoder input layer will
    # differ and the pretrained obs_encoder weights will be skipped gracefully.
    actor = TransformerActor(
        obs_dim_per_turbine=obs_dim,
        action_dim_per_turbine=args.action_dim_per_turbine,
        embed_dim=args.embed_dim,
        pos_embed_dim=args.pos_embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        action_scale=args.action_scale,
        action_bias=args.action_bias,
        # Positional encoding
        pos_encoding_type=args.pos_encoding_type,
        rel_pos_hidden_dim=args.rel_pos_hidden_dim,
        rel_pos_per_head=args.rel_pos_per_head,
        pos_embedding_mode=args.pos_embedding_mode,
        # Profile encoding
        profile_encoding=args.profile_encoding_type,
        profile_encoder_hidden=args.profile_encoder_hidden,
        n_profile_directions=n_profile_dirs if has_profiles else args.n_profile_directions,
        profile_fusion_type=args.profile_fusion_type,
        profile_embed_mode=args.profile_embed_mode,
        args=args,
    ).to(device)

    n_params = sum(p.numel() for p in actor.parameters() if p.requires_grad)
    print(f"  Total parameters: {n_params:,}")

    # =========================================================================
    # Load pretrained encoder weights
    # =========================================================================
    if _pretrain_encoder_sd is not None:
        print(f"\nLoading pretrained encoder weights...")
        net_sd = actor.state_dict()
        matched, skipped = [], []

        for key, value in _pretrain_encoder_sd.items():
            if key in net_sd:
                if net_sd[key].shape == value.shape:
                    net_sd[key] = value
                    matched.append(key)
                else:
                    skipped.append(
                        f"{key} (shape: {list(value.shape)} vs {list(net_sd[key].shape)})"
                    )
            else:
                skipped.append(f"{key} (not in actor)")

        actor.load_state_dict(net_sd)
        print(f"  Loaded {len(matched)}/{len(_pretrain_encoder_sd)} encoder params")
        if matched:
            print(f"  Matched: {matched[:5]}{'...' if len(matched) > 5 else ''}")
        if skipped:
            print(f"  Skipped: {skipped[:5]}{'...' if len(skipped) > 5 else ''}")

        del _pretrain_encoder_sd

    # =========================================================================
    # Optimizer with differential learning rates
    # =========================================================================
    head_params = []
    encoder_params = []

    for name, param in actor.named_parameters():
        if name.startswith("fc_mean") or name.startswith("fc_logstd"):
            head_params.append(param)
        else:
            encoder_params.append(param)

    n_enc_params = sum(p.numel() for p in encoder_params)
    n_head_params = sum(p.numel() for p in head_params)
    print(f"\n  Encoder params: {n_enc_params:,} (lr={args.encoder_lr})")
    print(f"  Head params:    {n_head_params:,} (lr={args.head_lr})")

    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": args.encoder_lr},
        {"params": head_params, "lr": args.head_lr},
    ], weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=min(args.encoder_lr, args.head_lr) * 0.01
    )

    # Wandb config
    if use_wandb:
        wandb.config.update({
            "obs_dim": obs_dim,
            "max_turbines": max_turbines,
            "has_profiles": has_profiles,
            "n_profile_dirs": n_profile_dirs,
            "n_train": n_train,
            "n_val": n_val,
            "n_params": n_params,
            "n_encoder_params": n_enc_params,
            "n_head_params": n_head_params,
            "n_layout_files": len(files),
            "pretrained": args.pretrain_checkpoint is not None,
            "action_type": args.action_type,
        }, allow_val_change=True)
        wandb.watch(actor, log="gradients", log_freq=100)

    # =========================================================================
    # Optional: freeze encoder for initial epochs
    # =========================================================================
    frozen_params = []
    if args.freeze_encoder_epochs > 0 and args.pretrain_checkpoint is not None:
        for name, param in actor.named_parameters():
            if not (name.startswith("fc_mean") or name.startswith("fc_logstd")):
                param.requires_grad = False
                frozen_params.append(name)
        print(f"\n  Froze {len(frozen_params)} encoder params for "
              f"{args.freeze_encoder_epochs} epochs")

    # Output dir
    os.makedirs(args.save_dir, exist_ok=True)

    # =========================================================================
    # Training loop
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"Starting behavioral cloning: {args.epochs} epochs")
    print(f"  Loss: {args.loss_type} | Early stopping patience: {args.patience}")
    if args.std_reg_weight > 0:
        print(f"  Std reg: weight={args.std_reg_weight}, target={args.target_log_std}")
    if args.pretrain_checkpoint:
        print(f"  Encoder: pretrained (freeze {args.freeze_encoder_epochs} epochs)")
    else:
        print(f"  Encoder: from scratch")
    print(f"{'='*60}")

    best_val_loss = float("inf")
    best_val_mse = float("inf")
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Unfreeze encoder after warm-up period
        if (args.freeze_encoder_epochs > 0
                and epoch == args.freeze_encoder_epochs + 1
                and frozen_params):
            for name, param in actor.named_parameters():
                param.requires_grad = True
            # Rebuild optimizer with all params unfrozen
            optimizer = torch.optim.AdamW([
                {"params": encoder_params, "lr": args.encoder_lr},
                {"params": head_params, "lr": args.head_lr},
            ], weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=args.epochs - epoch + 1,
                eta_min=min(args.encoder_lr, args.head_lr) * 0.01,
            )
            print(f"\n  [Epoch {epoch}] Unfroze encoder parameters ✓")
            frozen_params = []

        # Train
        train_metrics = train_one_epoch(
            actor, train_loader, optimizer, device, epoch,
            loss_type=args.loss_type,
            mse_weight=args.mse_weight,
            logprob_weight=args.logprob_weight,
            log_wandb=use_wandb,
            target_log_std=args.target_log_std,
            std_reg_weight=args.std_reg_weight,
        )

        # Evaluate
        val_metrics = evaluate(
            actor, val_loader, device,
            loss_type=args.loss_type,
            mse_weight=args.mse_weight,
            logprob_weight=args.logprob_weight,
            target_log_std=args.target_log_std,
            std_reg_weight=args.std_reg_weight,
        )

        scheduler.step()

        dt = time.time() - t0
        lr_enc = optimizer.param_groups[0]["lr"]
        lr_head = optimizer.param_groups[1]["lr"]

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss: {train_metrics['loss']:.6f} | "
            f"val_loss: {val_metrics['loss']:.6f} | "
            f"val_mse: {val_metrics['mse']:.6f} | "
            f"val_mae: {val_metrics['mae']:.4f} | "
            f"log_std: {val_metrics['mean_log_std']:.2f} | "
            f"lr: {lr_enc:.2e}/{lr_head:.2e} | "
            f"{dt:.1f}s"
        )

        # Wandb
        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_metrics["loss"],
                "train/mse": train_metrics["mse"],
                "train/nll": train_metrics["nll"],
                "train/std_reg": train_metrics["std_reg"],
                "train/mean_log_std": train_metrics["mean_log_std"],
                "train/mean_std": train_metrics["mean_std"],
                "val/loss": val_metrics["loss"],
                "val/mse": val_metrics["mse"],
                "val/nll": val_metrics["nll"],
                "val/mae": val_metrics["mae"],
                "val/std_reg": val_metrics["std_reg"],
                "val/mean_log_std": val_metrics["mean_log_std"],
                "val/mean_std": val_metrics["mean_std"],
                "lr_encoder": lr_enc,
                "lr_head": lr_head,
                "epoch_time_s": dt,
            })

        # ─── Early stopping / checkpointing ─────────────────────────
        val_key = val_metrics["loss"]
        is_best = val_key < best_val_loss - 1e-6

        if is_best:
            best_val_loss = val_key
            best_val_mse = val_metrics["mse"]
            best_epoch = epoch
            patience_counter = 0

            torch.save({
                "epoch": epoch,
                "encoder_state_dict": get_encoder_state_dict(actor),
                "actor_state_dict": actor.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_metrics["loss"],
                "val_mse": val_metrics["mse"],
                "val_mae": val_metrics["mae"],
                "args": vars(args),
                "obs_dim": obs_dim,
                "max_turbines": max_turbines,
                "n_profile_dirs": n_profile_dirs,
                "features": features,
            }, f"{args.save_dir}/best.pt")
            print(f"  → Saved best model (val_loss={val_key:.6f})")

            if use_wandb:
                wandb.run.summary["best_val_loss"] = best_val_loss
                wandb.run.summary["best_val_mse"] = best_val_mse
                wandb.run.summary["best_val_mae"] = val_metrics["mae"]
                wandb.run.summary["best_epoch"] = epoch
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}. "
                      f"Best val_loss: {best_val_loss:.6f} at epoch {best_epoch}")
                break

        # Periodic save
        if epoch % args.save_every == 0:
            torch.save({
                "epoch": epoch,
                "encoder_state_dict": get_encoder_state_dict(actor),
                "actor_state_dict": actor.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_metrics["loss"],
                "args": vars(args),
            }, f"{args.save_dir}/epoch_{epoch:04d}.pt")

    # =========================================================================
    # Final save
    # =========================================================================
    final_val = evaluate(
        actor, val_loader, device,
        loss_type=args.loss_type,
        mse_weight=args.mse_weight,
        logprob_weight=args.logprob_weight,
        target_log_std=args.target_log_std,
        std_reg_weight=args.std_reg_weight,
    )

    torch.save({
        "epoch": epoch,
        "encoder_state_dict": get_encoder_state_dict(actor),
        "actor_state_dict": actor.state_dict(),
        "val_loss": final_val["loss"],
        "val_mse": final_val["mse"],
        "val_mae": final_val["mae"],
        "args": vars(args),
        "obs_dim": obs_dim,
        "max_turbines": max_turbines,
        "n_profile_dirs": n_profile_dirs,
        "features": features,
    }, f"{args.save_dir}/final.pt")

    print(f"\n{'='*60}")
    print(f"Behavioral Cloning complete!")
    print(f"  Best val loss:  {best_val_loss:.6f} (epoch {best_epoch})")
    print(f"  Best val MSE:   {best_val_mse:.6f}")
    print(f"  Final val MAE:  {final_val['mae']:.4f}")
    print(f"  Final log_std:  {final_val['mean_log_std']:.3f}")
    print(f"  Checkpoints:    {args.save_dir}/")
    print(f"{'='*60}")
    print(f"\nTo use in RL (step 3):")
    print(f"  python transformer_sac_windfarm_v26.py \\")
    print(f"    --pretrain-checkpoint {args.save_dir}/best.pt")
    print(f"{'='*60}")

    # Wandb artifact
    if use_wandb:
        wandb.run.summary["final_val_loss"] = final_val["loss"]
        wandb.run.summary["final_val_mse"] = final_val["mse"]
        wandb.run.summary["final_val_mae"] = final_val["mae"]
        wandb.run.summary["final_mean_log_std"] = final_val["mean_log_std"]

        best_path = f"{args.save_dir}/best.pt"
        if os.path.exists(best_path):
            artifact = wandb.Artifact(
                name="bc-finetuned-actor",
                type="model",
                description=f"BC fine-tuned actor (val_mse={best_val_mse:.6f})",
                metadata={
                    "best_epoch": best_epoch,
                    "obs_dim": obs_dim,
                    "embed_dim": args.embed_dim,
                    "num_layers": args.num_layers,
                    "loss_type": args.loss_type,
                    "pretrained": args.pretrain_checkpoint is not None,
                },
            )
            artifact.add_file(best_path)
            wandb.log_artifact(artifact)

        wandb.finish()


if __name__ == "__main__":
    main()