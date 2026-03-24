"""
Unified Pretraining for Wind Farm Transformer

Supports multiple pretraining objectives:
  1. "power"  – Predict per-turbine power from (ws, wd, yaw) + positions + profiles
  2. "masked" – BERT-style: mask random turbines and reconstruct their features

Both modes use the REAL TransformerActor backbone so that saved encoder weights
have identical keys to the RL agent and can be loaded directly.

Includes periodic diagnostic visualizations logged to wandb:
  - Attention maps overlaid on farm layout
  - Predicted vs. actual power scatter (power mode)
  - Per-turbine error heatmap (power mode)
  - Attention entropy per layer/head
  - t-SNE of token embeddings
  - Profile embedding diagnostics

Usage:
    # Power prediction (default)
    python pretraining.py --pretrain-mode power --data-dir ./pretrain_data

    # Power with global wind features
    python pretraining.py --pretrain-mode power --global-features ws wd

    # BERT-style masked turbine prediction
    python pretraining.py --pretrain-mode masked --data-dir ./pretrain_data

    # Masked mode with custom mask ratio and snapshot data
    python pretraining.py --pretrain-mode masked --mask-ratio 0.25 --snapshot

    # Disable wandb
    python pretraining.py --no-track

    # Diagnostic plots every 5 epochs
    python pretraining.py --plot-every 5

Author: Marcus (DTU Wind Energy)
"""

import os
import time
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
from glob import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for HPC / headless
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import tyro

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not installed — logging disabled. Install with: pip install wandb")

try:
    from sklearn.manifold import TSNE
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from helpers.data_loader import create_pretrain_dataloader, WindFarmPretrainDataset, WindFarmSnapshotDataset

# Import real RL model architecture (ensures identical weight keys)
from networks import (
    TransformerActor,
    TransformerEncoder,
    TransformerEncoderLayer,
    create_positional_encoding,
    create_profile_encoding,
)

# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class Args:
    """Command-line arguments for wind farm pretraining."""

    # === Pretraining Mode ===
    pretrain_mode: str = "power"
    """Pretraining objective: 'power' or 'masked'."""

    # === Experiment Settings ===
    exp_name: str = "pretrain"
    """Experiment name (used for run naming)."""
    seed: int = 42
    """Random seed for reproducibility."""
    track: bool = True
    """Enable wandb tracking."""
    wandb_project_name: str = "windfarm-pretrain"
    """Wandb project name."""
    wandb_entity: Optional[str] = None
    """Wandb entity (team/user). None = default entity."""
    wandb_tags: Optional[str] = None
    """Comma-separated wandb tags (e.g. 'baseline,snapshot')."""

    # === Data Settings ===
    data_dir: str = "./pretrain_data"
    """Directory containing layout_*.h5 files."""
    snapshot: bool = False
    """Use snapshot mode (no history)."""
    history_length: int = 15
    """Number of timesteps of history per feature (sequence mode)."""
    skip_steps: int = 1
    """Subsample every N steps (snapshot mode only)."""

    # === Preprocessing ===
    features: List[str] = field(default_factory=lambda: ["ws", "wd", "yaw"])
    """Input features for pretraining. In power mode, power is always the target
    (not an input). In masked mode, 'power' is automatically added if missing."""
    global_features: List[str] = field(default_factory=list)
    """Features to replace with episode-level scalars (broadcast to all turbines).
    E.g. ["ws", "wd"] uses global mean_ws/mean_wd instead of per-turbine history."""
    use_wd_deviation: bool = False
    """Convert wind direction to deviation from mean."""
    use_wind_relative_pos: bool = True
    """Transform turbine positions to wind-relative frame."""
    rotate_profiles: bool = True
    """Rotate profiles to wind-relative frame."""
    wd_scale_range: float = 90.0
    """Wind direction deviation range for scaling (±degrees → [-1,1])."""
    power_max: float = 10e6
    """Max power for normalization (W). DTU10MW ≈ 10.64e6."""

    # === Transformer Architecture ===
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

    # === Positional Encoding (must match RL training) ===
    pos_encoding_type: Optional[str] = None
    """Positional encoding type. Must match RL config. None = no pos encoding."""
    rel_pos_hidden_dim: int = 64
    """Hidden dim for relative position MLP."""
    rel_pos_per_head: bool = True
    """Whether relative bias is per-head."""
    pos_embedding_mode: str = "concat"
    """'add' or 'concat' positional embedding to token."""

    # === Profile Encoding (must match RL training) ===
    profile_encoding_type: Optional[str] = None
    """Profile encoding type. Must match RL config. None = no profiles."""
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

    # === BERT-style Masking (masked mode only) ===
    mask_ratio: float = 0.20
    """Fraction of real turbines to mask per sample."""
    mask_replace_prob: float = 0.80
    """Probability of replacing masked token with [MASK] embedding."""
    mask_random_prob: float = 0.10
    """Probability of replacing masked token with random noise."""
    # Remaining (1 - replace - random) = keep original but still predict.
    predict_power_weight: float = 1.0
    """Weight for power features in masked reconstruction loss (vs other features)."""

    # === Training Hyperparameters ===
    epochs: int = 50
    """Number of training epochs."""
    batch_size: int = 1024
    """Batch size."""
    lr: float = 1.2e-3
    """Learning rate."""
    weight_decay: float = 1e-4
    """AdamW weight decay."""
    val_split: float = 0.1
    """Fraction of data held out for validation."""
    val_split_by_layout: bool = False
    """Split validation by layout instead of random. All samples from selected layouts go to val."""
    val_layout_names: Optional[List[str]] = None
    """Explicit layout names for validation (e.g. ['HornsRev']). All files sharing the name go to val."""
    # Data is loaded into RAM — no num_workers needed.

    # === Output / Checkpointing ===
    save_dir: str = "./pretrain_checkpoints"
    """Directory for saving checkpoints."""
    save_every: int = 10
    """Save checkpoint every N epochs."""

    # === Diagnostic Plots ===
    plot_every: int = 10
    """Generate diagnostic plots every N epochs (0 = disabled). Logged to wandb."""
    plot_n_samples: int = 256
    """Number of validation samples used for diagnostic plots."""
    plot_attn_top_k: int = 3
    """Top-K attention edges per turbine in attention layout plot."""
    plot_n_sample_indices: int = 3
    """Number of different samples to plot attention maps for."""

    # === Device ===
    device: str = "auto"
    """Device: 'auto', 'cuda', 'cpu'."""


# =============================================================================
# SHARED UTILITIES
# =============================================================================

# Keys that belong to the actor head, NOT the encoder
ACTOR_HEAD_KEYS = {"fc_mean", "fc_logstd", "action_scale", "action_bias_val"}


def get_encoder_state_dict(actor: TransformerActor) -> dict:
    """Extract only encoder weights from actor (excluding action heads)."""
    return {
        k: v for k, v in actor.state_dict().items()
        if not any(k.startswith(head) for head in ACTOR_HEAD_KEYS)
    }


def split_by_layout(
    dataset,
    val_fraction: float,
    val_layout_names: Optional[List[str]],
    seed: int,
) -> Tuple[Subset, Subset, List[int], List[int]]:
    """Split dataset into train/val Subsets by layout name.

    All samples from held-out layout *names* go to val; the rest go to train.
    Multiple file indices that share the same layout_name are grouped together.

    Returns:
        (train_subset, val_subset, train_layout_idxs, val_layout_idxs)
    """
    import random as _random

    # Discover unique layout indices
    if isinstance(dataset, WindFarmPretrainDataset):
        all_layout_idxs = dataset._layout_idx.unique().tolist()
    elif isinstance(dataset, WindFarmSnapshotDataset):
        all_layout_idxs = sorted(set(entry[0] for entry in dataset.index))
    else:
        raise TypeError(f"Unsupported dataset type: {type(dataset)}")

    # Build name → [layout_idx, ...] mapping
    name_to_idxs = {}
    for li in all_layout_idxs:
        name = dataset.layouts[li].get("layout_name", f"layout_{li}")
        name_to_idxs.setdefault(name, []).append(li)

    all_names = sorted(name_to_idxs.keys())

    # Determine which layout *names* go to validation
    if val_layout_names is not None:
        for vn in val_layout_names:
            if vn not in name_to_idxs:
                raise ValueError(f"Layout name '{vn}' not found. Available: {all_names}")
        val_name_set = set(val_layout_names)
    else:
        rng = _random.Random(seed)
        shuffled_names = list(all_names)
        rng.shuffle(shuffled_names)
        n_val_names = max(1, round(len(all_names) * val_fraction))
        val_name_set = set(shuffled_names[:n_val_names])

    # Expand names → integer indices
    val_layouts = set()
    for name in val_name_set:
        val_layouts.update(name_to_idxs[name])

    train_layouts = sorted(set(all_layout_idxs) - val_layouts)
    val_layouts_sorted = sorted(val_layouts)

    # Partition sample indices
    if isinstance(dataset, WindFarmPretrainDataset):
        val_mask = torch.isin(
            dataset._layout_idx,
            torch.tensor(val_layouts_sorted, dtype=dataset._layout_idx.dtype),
        )
        val_indices = torch.where(val_mask)[0].tolist()
        train_indices = torch.where(~val_mask)[0].tolist()
    else:  # WindFarmSnapshotDataset
        train_indices = []
        val_indices = []
        for i, (li, _, _) in enumerate(dataset.index):
            if li in val_layouts:
                val_indices.append(i)
            else:
                train_indices.append(i)

    return (
        Subset(dataset, train_indices),
        Subset(dataset, val_indices),
        train_layouts,
        val_layouts_sorted,
    )


def _encode_with_actor(
    actor: TransformerActor,
    obs: torch.Tensor,
    positions: torch.Tensor,
    attention_mask: torch.Tensor,
    receptivity: torch.Tensor = None,
    influence: torch.Tensor = None,
    need_weights: bool = False,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Run the actor's encoder path (everything before the action heads).
    Replicates lines 958-1004 of the RL script.

    Args:
        need_weights: If True, return per-layer attention weights for
                      visualization. Slightly slower due to attention
                      weight computation.

    Returns:
        h: (B, N, embed_dim) contextualised token embeddings
        attn_weights: list of (B, H, N, N) per layer (empty if need_weights=False)
    """
    # 1. Encode observations
    h = actor.obs_encoder(obs)  # (B, N, embed_dim)

    # 2. Positional encoding
    if actor.embedding_mode == "concat" and actor.pos_encoder is not None:
        pos_embed = actor.pos_encoder(positions)
        h = torch.cat([h, pos_embed], dim=-1)
    elif actor.embedding_mode == "add" and actor.pos_encoder is not None:
        pos_embed = actor.pos_encoder(positions)
        h = h + pos_embed

    # 3. Project to embed_dim
    h = actor.input_proj(h)

    # 4. Profile encoding
    if actor.recep_encoder and receptivity is not None and influence is not None:
        recep_embed = actor.recep_encoder(receptivity)
        influence_embed = actor.influence_encoder(influence)

        if actor.profile_fusion_type == "joint":
            profile_embed = actor.profile_fusion(
                torch.cat([recep_embed, influence_embed], dim=-1)
            )
        else:
            profile_embed = recep_embed + influence_embed

        if actor.profile_embed_mode == "concat":
            h = actor.profile_proj(torch.cat([h, profile_embed], dim=-1))
        else:
            h = h + profile_embed

    # 5. Relative position bias
    attn_bias = None
    if actor.rel_pos_bias is not None:
        attn_bias = actor.rel_pos_bias(positions, attention_mask)

    # 6. Transformer
    h, attn_weights = actor.transformer(
        h, attention_mask, attn_bias, need_weights=need_weights
    )

    return h, attn_weights


# =============================================================================
# MODE 1: POWER PREDICTION MODEL
# =============================================================================

class PowerPredictionModel(nn.Module):
    """
    Wraps the REAL TransformerActor backbone + a power prediction head.

    Uses the actor's encoder path so that saved weights have identical keys
    to the RL agent. The power_head is discarded after pretraining.
    """

    def __init__(self, actor: TransformerActor):
        super().__init__()
        self.actor = actor
        self.power_head = nn.Sequential(
            nn.Linear(actor.embed_dim, actor.embed_dim // 2),
            nn.GELU(),
            nn.Linear(actor.embed_dim // 2, 1),
        )

    def forward(
        self,
        batch: dict,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            batch: Dict with keys: obs, positions, attention_mask,
                   [receptivity, influence]
            need_weights: If True, also return (embeddings, attn_weights).

        Returns:
            predicted_power: (B, N) predicted normalized power per turbine
            h: (B, N, embed_dim) encoder output
            attn_weights: list of (B, H, N, N) per layer (empty if need_weights=False)
        """
        h, attn_weights = _encode_with_actor(
            self.actor,
            obs=batch["obs"],
            positions=batch["positions"],
            attention_mask=batch["attention_mask"],
            receptivity=batch.get("receptivity"),
            influence=batch.get("influence"),
            need_weights=need_weights,
        )
        power = self.power_head(h).squeeze(-1)
        return power, h, attn_weights


# =============================================================================
# MODE 2: BERT-STYLE MASKED TURBINE MODEL
# =============================================================================

class MaskedTurbineModel(nn.Module):
    """
    BERT-style pretraining: mask random turbine tokens and reconstruct
    their original observations.

    Masking strategy (following BERT):
      - Select `mask_ratio` fraction of real (non-padded) turbines
      - Of those: 80% → replace obs with learnable [MASK] embedding
                  10% → replace obs with random noise (uniform [-1, 1])
                  10% → keep original obs (but still predict)
      - Predict original obs for ALL selected turbines

    The encoder still sees positions and profiles for masked turbines —
    only the observation features are masked. This forces the transformer
    to learn spatial wake interactions: "given the surrounding turbines'
    states and this turbine's position, what should its conditions be?"

    Architecture:
        obs → [masking] → actor encoder → reconstruction_head → predicted obs
    """

    def __init__(
        self,
        actor: TransformerActor,
        obs_dim: int,
        mask_ratio: float = 0.20,
        mask_replace_prob: float = 0.80,
        mask_random_prob: float = 0.10,
    ):
        super().__init__()
        self.actor = actor
        self.obs_dim = obs_dim
        self.mask_ratio = mask_ratio
        self.mask_replace_prob = mask_replace_prob
        self.mask_random_prob = mask_random_prob

        assert mask_replace_prob + mask_random_prob <= 1.0, \
            "mask_replace_prob + mask_random_prob must be <= 1.0"

        # Learnable [MASK] embedding (replaces obs for masked turbines)
        # Same dimension as raw obs input (before obs_encoder)
        self.mask_token = nn.Parameter(torch.zeros(obs_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # Reconstruction head: predict original obs from contextualised embedding
        self.reconstruction_head = nn.Sequential(
            nn.Linear(actor.embed_dim, actor.embed_dim),
            nn.GELU(),
            nn.Linear(actor.embed_dim, obs_dim),
        )

    def _create_mask(
        self,
        attention_mask: torch.Tensor,   # (B, N) True = padding
    ) -> torch.Tensor:
        """
        Create a boolean mask indicating which turbines to predict.

        Returns:
            predict_mask: (B, N) True = this turbine is selected for prediction
        """
        B, N = attention_mask.shape
        device = attention_mask.device

        real_mask = ~attention_mask  # True = real turbine
        n_real = real_mask.sum(dim=1)  # (B,)

        # Number to mask per sample (at least 1)
        n_mask = (n_real.float() * self.mask_ratio).clamp(min=1).long()  # (B,)

        # Generate random scores, set padded turbines to -inf so they're never selected
        scores = torch.rand(B, N, device=device)
        scores[attention_mask] = -1.0

        # Select top-k per sample (variable k via loop — fast enough for typical N<100)
        predict_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        for i in range(B):
            k = n_mask[i].item()
            _, topk_idx = scores[i].topk(k)
            predict_mask[i, topk_idx] = True

        return predict_mask

    def _apply_masking(
        self,
        obs: torch.Tensor,          # (B, N, obs_dim)
        predict_mask: torch.Tensor,  # (B, N) True = selected for prediction
    ) -> torch.Tensor:
        """
        Apply BERT-style corruption to selected turbines' observations.

        Of the selected turbines:
          - mask_replace_prob → replace with [MASK] token
          - mask_random_prob  → replace with uniform noise in [-1, 1]
          - remainder         → keep original (model must still predict)
        """
        B, N, D = obs.shape
        device = obs.device

        # Start with a copy
        obs_masked = obs.clone()

        # Indices of selected turbines
        selected = predict_mask.nonzero(as_tuple=False)  # (M, 2) where M = total masked
        M = selected.shape[0]

        if M == 0:
            return obs_masked

        # Randomly assign each selected turbine to replace / random / keep
        rand_vals = torch.rand(M, device=device)
        replace_idx = rand_vals < self.mask_replace_prob
        random_idx = (rand_vals >= self.mask_replace_prob) & \
                     (rand_vals < self.mask_replace_prob + self.mask_random_prob)
        # keep_idx = everything else (no action needed)

        # Apply [MASK] token
        if replace_idx.any():
            rows = selected[replace_idx, 0]
            cols = selected[replace_idx, 1]
            obs_masked[rows, cols] = self.mask_token

        # Apply random noise
        if random_idx.any():
            rows = selected[random_idx, 0]
            cols = selected[random_idx, 1]
            obs_masked[rows, cols] = torch.rand(rows.shape[0], D, device=device) * 2 - 1

        return obs_masked

    def forward(
        self,
        batch: dict,
        need_weights: bool = False,
        return_details: bool = False,
    ) -> dict:
        """
        Forward pass with masking.

        Returns dict with:
            predicted_obs: (B, N, obs_dim) reconstructed obs (only meaningful at masked positions)
            predict_mask:  (B, N)           which turbines were masked
            original_obs:  (B, N, obs_dim)  original unmasked obs (targets)
            h:             (B, N, embed_dim) encoder output (if need_weights or return_details)
            attn_weights:  list of (B, H, N, N) (if need_weights)
        """
        obs = batch["obs"]                         # (B, N, obs_dim)
        positions = batch["positions"]             # (B, N, 2)
        attention_mask = batch["attention_mask"]    # (B, N)

        # 1. Create mask (which turbines to predict)
        predict_mask = self._create_mask(attention_mask)

        # 2. Apply corruption to obs
        obs_masked = self._apply_masking(obs, predict_mask)

        # 3. Encode (masked obs but real positions + profiles)
        h, attn_weights = _encode_with_actor(
            self.actor,
            obs=obs_masked,
            positions=positions,
            attention_mask=attention_mask,
            receptivity=batch.get("receptivity"),
            influence=batch.get("influence"),
            need_weights=need_weights,
        )

        # 4. Reconstruct obs at all positions (loss only at masked ones)
        predicted_obs = self.reconstruction_head(h)  # (B, N, obs_dim)

        result = {
            "predicted_obs": predicted_obs,
            "predict_mask": predict_mask,
            "original_obs": obs,
        }

        if need_weights or return_details:
            result["h"] = h
            result["attn_weights"] = attn_weights
        if return_details:
            result["obs_masked"] = obs_masked

        return result


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

def masked_mse_loss(
    predicted: torch.Tensor,       # (B, N)
    target: torch.Tensor,          # (B, N)
    attention_mask: torch.Tensor,  # (B, N) True = padding
) -> torch.Tensor:
    """MSE loss computed only over real (non-padded) turbines. For power mode."""
    real_mask = ~attention_mask  # True = real turbine
    n_real = real_mask.sum()

    if n_real == 0:
        return torch.tensor(0.0, device=predicted.device)

    diff = (predicted - target) ** 2
    loss = (diff * real_mask.float()).sum() / n_real
    return loss


def bert_reconstruction_loss(
    predicted_obs: torch.Tensor,    # (B, N, obs_dim)
    original_obs: torch.Tensor,     # (B, N, obs_dim)
    predict_mask: torch.Tensor,     # (B, N) True = predict this turbine
    feature_weights: torch.Tensor = None,  # (obs_dim,) optional per-feature weights
) -> dict:
    """
    MSE reconstruction loss over masked turbines only.

    Returns dict with:
        loss:        scalar total loss
        per_feature: (obs_dim,) per-feature MSE for logging
    """
    # Expand mask to feature dim
    mask = predict_mask.unsqueeze(-1).float()  # (B, N, 1)
    n_masked = predict_mask.sum()

    if n_masked == 0:
        obs_dim = predicted_obs.shape[-1]
        return {
            "loss": torch.tensor(0.0, device=predicted_obs.device),
            "per_feature": torch.zeros(obs_dim, device=predicted_obs.device),
        }

    # Per-feature squared error at masked positions
    sq_err = (predicted_obs - original_obs) ** 2  # (B, N, obs_dim)
    masked_sq_err = sq_err * mask                  # zero out non-masked

    # Per-feature MSE (for logging)
    per_feature_mse = masked_sq_err.sum(dim=(0, 1)) / n_masked  # (obs_dim,)

    # Weighted total loss
    if feature_weights is not None:
        loss = (per_feature_mse * feature_weights).mean()
    else:
        loss = per_feature_mse.mean()

    return {
        "loss": loss,
        "per_feature": per_feature_mse.detach(),
    }


# =============================================================================
# DIAGNOSTIC PLOT COLLECTION (inference with attention weights)
# =============================================================================

@torch.no_grad()
def collect_plot_data(model, dataset, device, n_samples=256, batch_size=64,
                      pretrain_mode="power"):
    """
    Run inference on a subset of the dataset, collecting predictions,
    embeddings, and attention weights for diagnostic plots.

    Returns dict with numpy arrays. Contents depend on pretrain_mode:
      Common:
        positions:       (S, N, 2)
        attention_mask:  (S, N)  bool, True=padding
        embeddings:      (S, N, embed_dim)
        obs:             (S, N, obs_dim)
        attn_weights:    list[layer] of (S, H, N, N)
      Power mode:
        predicted_power: (S, N)
        target_power:    (S, N)
      Masked mode:
        predicted_obs:   (S, N, obs_dim)
        original_obs:    (S, N, obs_dim)
        predict_mask:    (S, N) bool
    """
    model.eval()

    # Deterministic subset
    n = min(n_samples, len(dataset))
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(42))[:n]
    subset = torch.utils.data.Subset(dataset, indices.tolist())
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, pin_memory=True)

    all_pos, all_mask, all_embed, all_obs = [], [], [], []
    all_attn = None
    all_recep_embed, all_influence_embed, all_profile_embed = [], [], []
    all_mean_wd, all_layout_idx = [], []

    # Mode-specific collectors
    all_pred_power, all_target_power = [], []
    all_pred_obs, all_orig_obs, all_predict_mask = [], [], []

    actor = model.actor
    has_profiles = actor.recep_encoder is not None

    for batch in loader:
        batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            if pretrain_mode == "power":
                pred, embed, attn = model(batch_dev, need_weights=True)
                all_pred_power.append(pred.cpu().float())
                all_target_power.append(batch["target_power"])
                all_embed.append(embed.cpu().float())
            else:
                result = model(batch_dev, need_weights=True)
                all_pred_obs.append(result["predicted_obs"].cpu().float())
                all_orig_obs.append(result["original_obs"].cpu().float())
                all_predict_mask.append(result["predict_mask"].cpu())
                embed = result["h"]
                attn = result["attn_weights"]
                all_embed.append(embed.cpu().float())

            # Collect profile embeddings if available
            if has_profiles and "receptivity" in batch_dev and "influence" in batch_dev:
                recep_emb = actor.recep_encoder(batch_dev["receptivity"])
                infl_emb = actor.influence_encoder(batch_dev["influence"])
                if actor.profile_fusion_type == "joint":
                    prof_emb = actor.profile_fusion(
                        torch.cat([recep_emb, infl_emb], dim=-1)
                    )
                else:
                    prof_emb = recep_emb + infl_emb
                all_recep_embed.append(recep_emb.cpu().float())
                all_influence_embed.append(infl_emb.cpu().float())
                all_profile_embed.append(prof_emb.cpu().float())

        all_pos.append(batch["positions"])
        all_mask.append(batch["attention_mask"])
        all_obs.append(batch["obs"])
        all_mean_wd.append(batch["mean_wd"])
        all_layout_idx.append(batch["layout_idx"])

        if all_attn is None:
            all_attn = [[] for _ in range(len(attn))]
        for layer_idx, aw in enumerate(attn):
            all_attn[layer_idx].append(aw.cpu().float())

    out = {
        "positions": torch.cat(all_pos).numpy(),
        "attention_mask": torch.cat(all_mask).numpy(),
        "embeddings": torch.cat(all_embed).numpy(),
        "obs": torch.cat(all_obs).numpy(),
        "attn_weights": [torch.cat(layer_list).numpy() for layer_list in all_attn],
        "mean_wd": torch.cat(all_mean_wd).numpy(),
        "layout_idx": torch.cat(all_layout_idx).numpy(),
    }

    if pretrain_mode == "power":
        out["predicted_power"] = torch.cat(all_pred_power).numpy()
        out["target_power"] = torch.cat(all_target_power).numpy()
    else:
        out["predicted_obs"] = torch.cat(all_pred_obs).numpy()
        out["original_obs"] = torch.cat(all_orig_obs).numpy()
        out["predict_mask"] = torch.cat(all_predict_mask).numpy()

    if all_recep_embed:
        out["recep_embed"] = torch.cat(all_recep_embed).numpy()
        out["influence_embed"] = torch.cat(all_influence_embed).numpy()
        out["profile_embed"] = torch.cat(all_profile_embed).numpy()

    return out


# =============================================================================
# DIAGNOSTIC PLOTS
# =============================================================================

def _largest_layout(dataset, layout_idxs):
    """Return (layout_idx, layout_name, n_turbines) for the layout with the most turbines."""
    best = max(layout_idxs, key=lambda i: dataset.layouts[i]["n_turbines"])
    return best, dataset.layouts[best].get("layout_name", f"layout_{best}"), dataset.layouts[best]["n_turbines"]


def fig_attention_on_layout(results, sample_idx=0, top_k=3):
    """
    Attention maps overlaid on the physical farm layout, per head.

    Returns a dict of {layer_idx: fig} where each figure has one subplot
    per attention head, showing what each head has specialized on.
    """
    n_layers = len(results["attn_weights"])
    n_heads = results["attn_weights"][0].shape[1]
    pos = results["positions"][sample_idx]          # (N, 2)
    mask = results["attention_mask"][sample_idx]     # (N,)
    real = ~mask
    n_real = int(real.sum())
    pos_real = pos[:n_real]

    # Color turbines by power if available, otherwise by position
    if "target_power" in results:
        color_vals = results["target_power"][sample_idx][:n_real]
        color_label = "Target Power (norm)"
    else:
        color_vals = pos_real[:, 0]
        color_label = "x-position (norm)"

    cmap_edge = plt.cm.Reds

    def _draw(ax, attn_matrix, title):
        """Draw attention edges + turbine scatter on axis."""
        n = len(pos_real)
        for i in range(n):
            w = attn_matrix[i].copy()
            w[i] = 0  # skip self-attention
            if top_k > 0 and top_k < n:
                thresh = np.sort(w)[::-1][min(top_k, n - 1)]
                w[w < thresh] = 0
            for j in range(n):
                if w[j] > 0.01:
                    ax.annotate(
                        "", xy=pos_real[j], xytext=pos_real[i],
                        arrowprops=dict(
                            arrowstyle="->,head_width=0.15,head_length=0.1",
                            color=cmap_edge(float(w[j])),
                            alpha=float(np.clip(w[j], 0.15, 1.0)),
                            lw=1.0 + 2.0 * float(w[j]),
                            connectionstyle="arc3,rad=0.05",
                        ),
                    )
        sc = ax.scatter(
            pos_real[:, 0], pos_real[:, 1],
            c=color_vals, cmap="viridis", s=120,
            edgecolors="black", linewidth=1.0, zorder=5,
        )
        for t in range(n):
            ax.annotate(str(t), pos_real[t], fontsize=7, ha="center",
                        va="center", fontweight="bold", color="white", zorder=6)
        ax.set_title(title, fontsize=9)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        # Wind direction arrow (positions are wind-relative: wind from left)
        x_range = pos_real[:, 0].max() - pos_real[:, 0].min()
        y_mid = pos_real[:, 1].mean()
        x_start = pos_real[:, 0].min() - 0.15 * max(x_range, 0.1)
        arrow_len = 0.12 * max(x_range, 0.1)
        ax.annotate(
            "", xy=(x_start + arrow_len, y_mid), xytext=(x_start, y_mid),
            arrowprops=dict(arrowstyle="->,head_width=0.3,head_length=0.15",
                            color="dodgerblue", lw=2.0),
            zorder=7,
        )
        ax.text(x_start + arrow_len / 2, y_mid + 0.04 * max(x_range, 0.1),
                "Wind", fontsize=7, ha="center", color="dodgerblue",
                fontweight="bold", zorder=7)

        return sc

    def _normalize_rows(a):
        row_max = a.max(axis=1, keepdims=True)
        row_max = np.where(row_max > 0, row_max, 1.0)
        return a / row_max

    # One figure per layer, with one subplot per head
    layer_figs = {}
    for l in range(n_layers):
        attn_l = results["attn_weights"][l][sample_idx]  # (H, N, N)

        fig, axes = plt.subplots(1, n_heads, figsize=(4.5 * n_heads, 4.5), squeeze=False)
        axes = axes[0]
        for h in range(n_heads):
            attn_h = _normalize_rows(attn_l[h, :n_real, :n_real])
            sc = _draw(axes[h], attn_h, f"Head {h}")
        fig.colorbar(sc, ax=axes[-1], shrink=0.8, label=color_label)
        fig.suptitle(f"Layer {l} — Per-Head Attention (sample {sample_idx})",
                     fontsize=12, y=1.02)
        fig.tight_layout()
        layer_figs[l] = fig

    return layer_figs


def fig_pred_vs_actual(results, input_features=None):
    """Scatter of predicted vs. actual power, colored by wind direction.
    Only applicable to power mode."""
    if "predicted_power" not in results:
        return None

    pred = results["predicted_power"]
    target = results["target_power"]
    mask = results["attention_mask"]
    obs = results["obs"]
    real = ~mask

    pred_flat = pred[real]
    target_flat = target[real]

    # Extract wind direction for coloring (best-effort, depends on obs layout)
    obs_dim = obs.shape[-1]
    try:
        if input_features is not None:
            n_features = len(input_features)
            wd_feature_idx = input_features.index("wd") if "wd" in input_features else 1
        else:
            n_features = 3  # ws, wd, yaw fallback
            wd_feature_idx = 1
        history_len = obs_dim // n_features
        if obs_dim == n_features * history_len:
            wd_idx = wd_feature_idx * history_len + (history_len - 1)
            wd_flat = obs[:, :, wd_idx][real]
        else:
            wd_flat = np.zeros_like(pred_flat)
    except Exception:
        wd_flat = np.zeros_like(pred_flat)

    # Metrics
    mse = np.mean((pred_flat - target_flat) ** 2)
    mae = np.mean(np.abs(pred_flat - target_flat))
    ss_res = np.sum((pred_flat - target_flat) ** 2)
    ss_tot = np.sum((target_flat - target_flat.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    fig, ax = plt.subplots(figsize=(6, 5.5))
    sc = ax.scatter(target_flat, pred_flat, c=wd_flat, cmap="twilight", s=4, alpha=0.4)
    lims = [min(target_flat.min(), pred_flat.min()), max(target_flat.max(), pred_flat.max())]
    ax.plot(lims, lims, "k--", alpha=0.5, lw=1, label="Perfect")
    fig.colorbar(sc, label="Wind Dir (norm)")
    ax.set_xlabel("Target Power (norm)")
    ax.set_ylabel("Predicted Power (norm)")
    ax.set_title(f"Pred vs. Actual — MSE={mse:.6f}  MAE={mae:.4f}  R²={r2:.4f}")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def fig_error_heatmap(results):
    """
    Per-turbine error plotted on the farm grid (power mode only):
      - Absolute MAE
      - Mean target power
      - Relative error (NMAE = MAE / mean_power)

    Returns (fig, scalar_stats).
    """
    if "predicted_power" not in results:
        return None, {}

    pred = results["predicted_power"]
    target = results["target_power"]
    pos = results["positions"]
    mask = results["attention_mask"]
    real = ~mask

    n_samples, n_turbines = pred.shape
    abs_errors = np.abs(pred - target)

    # Per-turbine averages
    turbine_mae = np.zeros(n_turbines)
    avg_power = np.zeros(n_turbines)
    turbine_nmae = np.zeros(n_turbines)
    avg_pos = np.zeros((n_turbines, 2))
    active = np.zeros(n_turbines, dtype=bool)

    for t in range(n_turbines):
        valid = real[:, t]
        if valid.sum() > 0:
            turbine_mae[t] = abs_errors[valid, t].mean()
            avg_power[t] = target[valid, t].mean()
            avg_pos[t] = pos[valid, t].mean(axis=0)
            active[t] = True
            if avg_power[t] > 1e-4:
                turbine_nmae[t] = turbine_mae[t] / avg_power[t]
            else:
                turbine_nmae[t] = 0.0

    # Farm-wide scalar stats
    stats = {}
    if active.any():
        stats["val/farm_mae"] = float(turbine_mae[active].mean())
        stats["val/farm_nmae"] = float(turbine_nmae[active].mean())
        stats["val/farm_nmae_max"] = float(turbine_nmae[active].max())

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 5))

    # Left: Absolute MAE
    sc1 = ax1.scatter(avg_pos[active, 0], avg_pos[active, 1], c=turbine_mae[active],
                       cmap="hot_r", s=250, edgecolors="black", linewidth=1.0, zorder=5)
    for t in np.where(active)[0]:
        ax1.annotate(f"{turbine_mae[t]:.4f}", avg_pos[t], fontsize=6,
                     ha="center", va="center", zorder=6)
    fig.colorbar(sc1, ax=ax1, label="MAE")
    ax1.set_title("Per-Turbine MAE (absolute)")
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)

    # Center: Mean power
    sc2 = ax2.scatter(avg_pos[active, 0], avg_pos[active, 1], c=avg_power[active],
                       cmap="viridis", s=250, edgecolors="black", linewidth=1.0, zorder=5)
    for t in np.where(active)[0]:
        ax2.annotate(f"{avg_power[t]:.3f}", avg_pos[t], fontsize=6, ha="center",
                     va="center", color="white", fontweight="bold", zorder=6)
    fig.colorbar(sc2, ax=ax2, label="Mean Power (norm)")
    ax2.set_title("Mean Target Power")
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)

    # Right: Relative error (NMAE = MAE / mean_power)
    sc3 = ax3.scatter(avg_pos[active, 0], avg_pos[active, 1], c=turbine_nmae[active],
                       cmap="hot_r", s=250, edgecolors="black", linewidth=1.0, zorder=5)
    for t in np.where(active)[0]:
        ax3.annotate(f"{turbine_nmae[t]:.1%}", avg_pos[t], fontsize=6,
                     ha="center", va="center", zorder=6)
    fig.colorbar(sc3, ax=ax3, label="NMAE (MAE / Power)")
    ax3.set_title("Per-Turbine NMAE (relative)")
    ax3.set_aspect("equal")
    ax3.grid(True, alpha=0.3)

    fig.suptitle("Error & Power Distribution Across Farm", fontsize=12)
    fig.tight_layout()
    return fig, stats


def fig_masked_reconstruction(results):
    """
    Masked mode diagnostics: scatter of predicted vs original obs
    at masked positions, plus per-feature MSE bars.

    Returns (fig, scalar_stats).
    """
    if "predicted_obs" not in results:
        return None, {}

    pred_obs = results["predicted_obs"]     # (S, N, obs_dim)
    orig_obs = results["original_obs"]      # (S, N, obs_dim)
    predict_mask = results["predict_mask"]  # (S, N) bool
    mask = results["attention_mask"]        # (S, N) bool

    stats = {}

    # Flatten to masked positions only
    pred_flat = pred_obs[predict_mask]   # (M, obs_dim)
    orig_flat = orig_obs[predict_mask]   # (M, obs_dim)

    if len(pred_flat) == 0:
        return None, stats

    # Per-feature MSE
    per_feat_mse = np.mean((pred_flat - orig_flat) ** 2, axis=0)  # (obs_dim,)
    overall_mse = per_feat_mse.mean()
    stats["val/masked_mse"] = float(overall_mse)

    # Cosine similarity
    pred_norms = np.linalg.norm(pred_flat, axis=-1, keepdims=True)
    orig_norms = np.linalg.norm(orig_flat, axis=-1, keepdims=True)
    cos_sim = (pred_flat * orig_flat).sum(axis=-1) / (
        np.clip(pred_norms.squeeze() * orig_norms.squeeze(), 1e-8, None)
    )
    stats["val/masked_cosine_sim"] = float(cos_sim.mean())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: scatter of predicted vs original (first few obs dims averaged)
    pred_mean = pred_flat.mean(axis=-1)
    orig_mean = orig_flat.mean(axis=-1)
    n_plot = min(5000, len(pred_mean))
    idx = np.random.RandomState(42).choice(len(pred_mean), n_plot, replace=False)
    axes[0].scatter(orig_mean[idx], pred_mean[idx], s=4, alpha=0.3, color="steelblue")
    lims = [min(orig_mean.min(), pred_mean.min()), max(orig_mean.max(), pred_mean.max())]
    axes[0].plot(lims, lims, "k--", alpha=0.5, lw=1, label="Perfect")
    axes[0].set_xlabel("Original (mean across features)")
    axes[0].set_ylabel("Predicted (mean across features)")
    axes[0].set_title(f"Masked Reconstruction — MSE={overall_mse:.6f}  cos_sim={cos_sim.mean():.4f}")
    axes[0].legend(fontsize=8)
    axes[0].set_aspect("equal")
    axes[0].grid(True, alpha=0.3)

    # Right: per-feature MSE bars
    n_feats = len(per_feat_mse)
    axes[1].bar(range(n_feats), per_feat_mse, color="coral", edgecolor="black", linewidth=0.5)
    axes[1].set_xlabel("Feature Index")
    axes[1].set_ylabel("MSE")
    axes[1].set_title("Per-Feature Reconstruction MSE")
    axes[1].grid(True, alpha=0.3, axis="y")

    fig.suptitle("Masked Turbine Reconstruction", fontsize=12)
    fig.tight_layout()
    return fig, stats


def fig_reconstruction_vs_streamwise(results):
    """
    Per-turbine masked reconstruction error vs. streamwise position.

    If the model has learned wake structure, upstream turbines (free-stream,
    simpler flow) should have lower error than downstream turbines.

    Returns (fig, scalar_stats) where scalar_stats contains:
      masked/streamwise_slope — positive = wake-aware reconstruction
    """
    if "predicted_obs" not in results:
        return None, {}

    pred_obs = results["predicted_obs"]     # (S, N, obs_dim)
    orig_obs = results["original_obs"]      # (S, N, obs_dim)
    predict_mask = results["predict_mask"]  # (S, N) bool
    pos = results["positions"]              # (S, N, 2)
    mask = results["attention_mask"]        # (S, N) bool
    real = ~mask

    n_samples, n_turbines, obs_dim = pred_obs.shape

    # Per-turbine MSE at masked positions
    sq_error = np.sum((pred_obs - orig_obs) ** 2, axis=-1) / obs_dim  # (S, N)
    # Only count positions that are both masked AND real
    valid = predict_mask & real  # (S, N)

    turbine_mse = np.full(n_turbines, np.nan)
    turbine_x = np.full(n_turbines, np.nan)
    for t in range(n_turbines):
        t_valid = valid[:, t]
        if t_valid.sum() > 0:
            turbine_mse[t] = sq_error[t_valid, t].mean()
            turbine_x[t] = pos[t_valid, t, 0].mean()  # mean streamwise position

    active = ~np.isnan(turbine_mse)
    if active.sum() < 2:
        return None, {}

    x_active = turbine_x[active]
    mse_active = turbine_mse[active]

    # Linear fit: MSE = slope * x + intercept
    slope, intercept = np.polyfit(x_active, mse_active, 1)
    stats = {"masked/streamwise_slope": float(slope)}

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x_active, mse_active, s=80, c=mse_active, cmap="hot_r",
               edgecolors="black", linewidth=0.8, zorder=5)
    x_line = np.linspace(x_active.min(), x_active.max(), 50)
    ax.plot(x_line, slope * x_line + intercept, "b--", alpha=0.7,
            label=f"slope = {slope:.4f}")
    ax.set_xlabel("Streamwise Position (x, wind-relative)")
    ax.set_ylabel("Mean Reconstruction MSE")
    ax.set_title("Masked Reconstruction Error vs. Streamwise Position")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig, stats


def compute_attention_entropy(results):
    """
    Compute attention entropy per layer/head as scalar metrics.

    Returns a dict of floats suitable for direct wandb.log(), producing
    native interactive line charts over epochs.
    """
    mask = results["attention_mask"]
    n_layers = len(results["attn_weights"])
    real = ~mask  # (S, N)

    eps = 1e-8
    stats = {}
    all_means = []

    n_real_avg = real.sum(axis=1).mean()
    stats["attn_entropy/uniform"] = float(np.log(n_real_avg))

    for l in range(n_layers):
        attn = results["attn_weights"][l]  # (S, H, N, N)
        entropy = -(attn * np.log(attn + eps)).sum(axis=-1)  # (S, H, N)
        real_exp = real[:, np.newaxis, :]  # (S, 1, N)
        per_head = (entropy * real_exp).sum(axis=(0, 2)) / real_exp.sum(axis=(0, 2))  # (H,)

        for h in range(len(per_head)):
            stats[f"attn_entropy/L{l}_H{h}"] = float(per_head[h])
        layer_mean = float(per_head.mean())
        stats[f"attn_entropy/L{l}_mean"] = layer_mean
        all_means.append(layer_mean)

    stats["attn_entropy/mean"] = float(np.mean(all_means))
    return stats


def compute_wake_alignment(results):
    """
    Compute wake alignment score per attention head.

    For each head, measures the fraction of attention weight flowing from
    downstream queries to upstream keys (the physically-correct direction
    for wake modeling). Positions are wind-relative (wind from 270° = negative x),
    so upstream = lower x-coordinate.

    Returns a dict of scalars for wandb.log():
      wake/L{l}_H{h} — per-head score ∈ [0, 1]
      wake/L{l}_mean — per-layer mean
      wake/mean      — overall mean (primary sweep ranking metric)
    Score >0.5 = model attends preferentially upstream. ~0.5 = random.
    """
    pos = results["positions"]       # (S, N, 2)
    mask = results["attention_mask"]  # (S, N) bool
    real = ~mask                      # (S, N)
    n_layers = len(results["attn_weights"])

    stats = {}
    all_layer_means = []

    for l in range(n_layers):
        attn = results["attn_weights"][l]  # (S, H, N, N)
        n_heads = attn.shape[1]
        head_scores = []

        for h in range(n_heads):
            attn_h = attn[:, h, :, :]  # (S, N, N) — attn_h[s, i, j] = query i attends to key j

            # Streamwise positions: x-coordinate (column 0)
            x = pos[:, :, 0]  # (S, N)

            # upstream_mask[s, i, j] = True if key j is upstream of query i
            # i.e. x[j] < x[i]  (lower x = upstream in wind-relative coords)
            x_query = x[:, :, np.newaxis]  # (S, N, 1)
            x_key = x[:, np.newaxis, :]    # (S, 1, N)
            upstream_mask = x_key < x_query  # (S, N, N)

            # Self-attention mask: exclude i==j
            N = attn_h.shape[-1]
            self_mask = np.eye(N, dtype=bool)[np.newaxis, :, :]  # (1, N, N)

            # Real turbine mask: both query and key must be real
            real_query = real[:, :, np.newaxis]  # (S, N, 1)
            real_key = real[:, np.newaxis, :]    # (S, 1, N)
            valid = real_query & real_key & ~self_mask  # (S, N, N)

            # Compute score: sum of attention at upstream positions / total attention
            attn_valid = attn_h * valid
            attn_upstream = attn_h * valid * upstream_mask

            total = attn_valid.sum()
            if total > 1e-10:
                score = float(attn_upstream.sum() / total)
            else:
                score = 0.5  # undefined → neutral

            stats[f"wake/L{l}_H{h}"] = score
            head_scores.append(score)

        layer_mean = float(np.mean(head_scores))
        stats[f"wake/L{l}_mean"] = layer_mean
        all_layer_means.append(layer_mean)

    stats["wake/mean"] = float(np.mean(all_layer_means))
    return stats


def fig_embedding_tsne(results, perplexity=30, max_points=2000):
    """t-SNE of transformer output embeddings colored by power and position."""
    if not HAS_SKLEARN:
        return None

    embeddings = results["embeddings"]
    positions = results["positions"]
    mask = results["attention_mask"]
    real = ~mask

    emb_flat = embeddings[real]
    pos_flat = positions[real]

    # Color by power if available, otherwise by x-position
    if "target_power" in results:
        pow_flat = results["target_power"][real]
        pow_label = "Power (norm)"
    else:
        pow_flat = pos_flat[:, 0]
        pow_label = "x-pos (norm)"

    if len(emb_flat) > max_points:
        idx = np.random.RandomState(42).choice(len(emb_flat), max_points, replace=False)
        emb_flat, pow_flat, pos_flat = emb_flat[idx], pow_flat[idx], pos_flat[idx]

    tsne = TSNE(n_components=2, perplexity=min(perplexity, len(emb_flat) // 4),
                random_state=42, max_iter=800)
    emb_2d = tsne.fit_transform(emb_flat)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    sc0 = axes[0].scatter(emb_2d[:, 0], emb_2d[:, 1], c=pow_flat, cmap="viridis", s=6, alpha=0.5)
    fig.colorbar(sc0, ax=axes[0]).set_label(pow_label)
    axes[0].set_title(f"By {pow_label}")

    sc1 = axes[1].scatter(emb_2d[:, 0], emb_2d[:, 1], c=pos_flat[:, 0], cmap="coolwarm", s=6, alpha=0.5)
    fig.colorbar(sc1, ax=axes[1]).set_label("x-pos (norm)")
    axes[1].set_title("By Streamwise Position")

    sc2 = axes[2].scatter(emb_2d[:, 0], emb_2d[:, 1], c=pos_flat[:, 1], cmap="PiYG", s=6, alpha=0.5)
    fig.colorbar(sc2, ax=axes[2]).set_label("y-pos (norm)")
    axes[2].set_title("By Lateral Position")

    for ax in axes:
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.grid(True, alpha=0.2)

    fig.suptitle("t-SNE of Token Embeddings", fontsize=12)
    fig.tight_layout()
    return fig


def fig_profile_embeddings(results, sample_indices=None):
    """
    Profile encoding diagnostics:
      - Per-turbine embedding norms (receptivity, influence, fused)
      - Pairwise cosine similarity of fused profile embeddings
      - Comparison: do nearby turbines get similar profiles?

    Returns (fig, scalar_stats).
    """
    if "profile_embed" not in results:
        return None, {}

    profile_embed = results["profile_embed"]       # (S, N, D)
    recep_embed = results["recep_embed"]           # (S, N, D)
    influence_embed = results["influence_embed"]   # (S, N, D)
    mask = results["attention_mask"]               # (S, N) bool
    positions = results["positions"]               # (S, N, 2)
    real = ~mask

    # --- Scalar stats (averaged over all real turbines) ---
    stats = {}

    def _masked_norm_stats(embed, name):
        norms = np.linalg.norm(embed, axis=-1)  # (S, N)
        real_norms = norms[real]
        stats[f"profile/{name}_norm_mean"] = float(real_norms.mean())
        stats[f"profile/{name}_norm_std"] = float(real_norms.std())
        return norms

    fused_norms = _masked_norm_stats(profile_embed, "fused")
    recep_norms = _masked_norm_stats(recep_embed, "receptivity")
    infl_norms = _masked_norm_stats(influence_embed, "influence")

    # --- Pick sample indices for visualization ---
    n_samples = profile_embed.shape[0]
    if sample_indices is None:
        sample_indices = np.linspace(0, n_samples - 1, min(3, n_samples), dtype=int)

    si = sample_indices[0]
    n_real_i = int(real[si].sum())
    prof_i = profile_embed[si, :n_real_i]   # (n_real, D)
    pos_i = positions[si, :n_real_i]         # (n_real, 2)
    recep_i = recep_embed[si, :n_real_i]
    infl_i = influence_embed[si, :n_real_i]

    # Cosine similarity matrix
    def _cosine_sim(a):
        norms = np.linalg.norm(a, axis=-1, keepdims=True)
        norms = np.clip(norms, 1e-8, None)
        a_normed = a / norms
        return a_normed @ a_normed.T

    cos_sim_fused = _cosine_sim(prof_i)
    cos_sim_recep = _cosine_sim(recep_i)
    cos_sim_infl = _cosine_sim(infl_i)

    # Distance matrix for correlation check
    dx = pos_i[:, 0:1] - pos_i[:, 0:1].T
    dy = pos_i[:, 1:2] - pos_i[:, 1:2].T
    dist_matrix = np.sqrt(dx**2 + dy**2)

    # Correlation between spatial distance and embedding similarity
    triu_idx = np.triu_indices(n_real_i, k=1)
    if len(triu_idx[0]) > 1:
        dists_flat = dist_matrix[triu_idx]
        sims_flat = cos_sim_fused[triu_idx]
        corr = np.corrcoef(dists_flat, sims_flat)[0, 1]
        stats["profile/dist_sim_correlation"] = float(corr)

    # Off-diagonal mean similarity
    off_diag_fused = cos_sim_fused[triu_idx].mean() if len(triu_idx[0]) > 0 else 0.0
    stats["profile/mean_off_diag_cosine_sim"] = float(off_diag_fused)

    # --- Figure: 2×2 grid ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    # Top-left: fused profile cosine similarity heatmap
    im0 = axes[0, 0].imshow(cos_sim_fused, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    axes[0, 0].set_title("Fused Profile Cosine Similarity")
    axes[0, 0].set_xlabel("Turbine")
    axes[0, 0].set_ylabel("Turbine")
    fig.colorbar(im0, ax=axes[0, 0], shrink=0.8)

    # Top-right: receptivity vs influence cosine sim
    combined = np.zeros_like(cos_sim_fused)
    combined[np.triu_indices(n_real_i, k=0)] = cos_sim_recep[np.triu_indices(n_real_i, k=0)]
    combined[np.tril_indices(n_real_i, k=-1)] = cos_sim_infl[np.tril_indices(n_real_i, k=-1)]
    im1 = axes[0, 1].imshow(combined, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    axes[0, 1].set_title("Recep (upper △) vs Influence (lower △)")
    axes[0, 1].set_xlabel("Turbine")
    axes[0, 1].set_ylabel("Turbine")
    fig.colorbar(im1, ax=axes[0, 1], shrink=0.8)

    # Bottom-left: embedding norms on the farm layout
    norm_i = np.linalg.norm(prof_i, axis=-1)
    sc2 = axes[1, 0].scatter(
        pos_i[:, 0], pos_i[:, 1], c=norm_i, cmap="plasma",
        s=200, edgecolors="black", linewidth=1.0, zorder=5,
    )
    for t in range(n_real_i):
        axes[1, 0].annotate(f"{norm_i[t]:.2f}", pos_i[t], fontsize=6,
                            ha="center", va="center", color="white",
                            fontweight="bold", zorder=6)
    fig.colorbar(sc2, ax=axes[1, 0], shrink=0.8, label="L2 Norm")
    axes[1, 0].set_title("Fused Profile Embedding Norms")
    axes[1, 0].set_aspect("equal")
    axes[1, 0].grid(True, alpha=0.3)

    # Bottom-right: spatial distance vs embedding similarity scatter
    if len(triu_idx[0]) > 1:
        axes[1, 1].scatter(dists_flat, sims_flat, s=12, alpha=0.5, color="steelblue")
        z = np.polyfit(dists_flat, sims_flat, 1)
        x_line = np.linspace(dists_flat.min(), dists_flat.max(), 50)
        axes[1, 1].plot(x_line, np.polyval(z, x_line), "r--", alpha=0.7,
                        label=f"r = {stats.get('profile/dist_sim_correlation', 0):.3f}")
        axes[1, 1].legend(fontsize=8)
    axes[1, 1].set_xlabel("Spatial Distance (norm)")
    axes[1, 1].set_ylabel("Cosine Similarity")
    axes[1, 1].set_title("Distance vs. Profile Similarity")
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(f"Profile Encoding Diagnostics (sample {si})", fontsize=13)
    fig.tight_layout()
    return fig, stats


# =============================================================================
# ORCHESTRATOR: generate all plots and log to wandb
# =============================================================================

def generate_diagnostic_plots(
    model, val_dataset, device, epoch, args,
    train_set=None, dataset=None,
    train_layout_idxs=None, val_layout_idxs=None,
):
    """
    Collect inference results and generate all diagnostic plots.
    Returns a dict of {wandb_key: wandb.Image} ready for wandb.log().

    When train_set/dataset/layout_idxs are provided, attention and profile
    plots are drawn for the largest training and validation layouts (by
    turbine count) instead of arbitrary evenly-spaced samples.
    """
    print(f"  Generating diagnostic plots (n_samples={args.plot_n_samples})...")
    t0 = time.time()

    val_results = collect_plot_data(
        model, val_dataset, device,
        n_samples=args.plot_n_samples,
        batch_size=args.batch_size,
        pretrain_mode=args.pretrain_mode,
    )

    # Collect train data for layout-aware plots (only when layout info is available)
    train_results = None
    if train_set is not None and train_layout_idxs is not None and val_layout_idxs is not None:
        train_results = collect_plot_data(
            model, train_set, device,
            n_samples=args.plot_n_samples,
            batch_size=args.batch_size,
            pretrain_mode=args.pretrain_mode,
        )

    # Use val_results as the primary results for aggregate metrics
    results = val_results

    images = {}

    # 1. Attention on layout — largest layout from each split
    layout_aware = (dataset is not None and train_layout_idxs is not None
                    and val_layout_idxs is not None and train_results is not None)

    val_match = np.array([], dtype=int)
    if layout_aware:
        # Find largest layouts by turbine count
        val_li, val_name, val_nt = _largest_layout(dataset, val_layout_idxs)
        train_li, train_name, train_nt = _largest_layout(dataset, train_layout_idxs)
        print(f"    Attention plots: val layout={val_name} ({val_nt}T), "
              f"train layout={train_name} ({train_nt}T)")

        # Find first sample matching each layout
        val_match = np.where(val_results["layout_idx"] == val_li)[0]
        train_match = np.where(train_results["layout_idx"] == train_li)[0]

        plot_specs = []
        if len(val_match) > 0:
            plot_specs.append((val_results, int(val_match[0]), f"val_{val_name}"))
        if len(train_match) > 0:
            plot_specs.append((train_results, int(train_match[0]), f"train_{train_name}"))

        for res, idx, label in plot_specs:
            try:
                layer_figs = fig_attention_on_layout(
                    res, sample_idx=idx, top_k=args.plot_attn_top_k
                )
                for layer_idx, fig in layer_figs.items():
                    images[f"plots/attention_L{layer_idx}_{label}"] = wandb.Image(fig)
                    plt.close(fig)
            except Exception as e:
                print(f"    Warning: attention plot failed for {label}: {e}")
    else:
        # Fallback: evenly-spaced samples from val set
        n_samples_available = results["positions"].shape[0]
        sample_indices = np.linspace(0, n_samples_available - 1,
                                      args.plot_n_sample_indices, dtype=int)
        for i, idx in enumerate(sample_indices):
            try:
                layer_figs = fig_attention_on_layout(
                    results, sample_idx=int(idx), top_k=args.plot_attn_top_k
                )
                for layer_idx, fig in layer_figs.items():
                    images[f"plots/attention_L{layer_idx}_s{idx}"] = wandb.Image(fig)
                    plt.close(fig)
            except Exception as e:
                print(f"    Warning: attention plot failed for sample {idx}: {e}")

    # 2. Predicted vs. actual scatter (power mode only)
    if args.pretrain_mode == "power":
        try:
            fig = fig_pred_vs_actual(results, input_features=list(args.features))
            if fig is not None:
                images["plots/pred_vs_actual"] = wandb.Image(fig)
                plt.close(fig)
        except Exception as e:
            print(f"    Warning: scatter plot failed: {e}")

    # 3. Error heatmap (power mode only)
    if args.pretrain_mode == "power":
        try:
            fig, error_stats = fig_error_heatmap(results)
            if fig is not None:
                images["plots/error_heatmap"] = wandb.Image(fig)
                plt.close(fig)
                images.update(error_stats)
        except Exception as e:
            print(f"    Warning: error heatmap failed: {e}")

    # 4. Masked reconstruction diagnostics (masked mode only)
    if args.pretrain_mode == "masked":
        try:
            fig, masked_stats = fig_masked_reconstruction(results)
            if fig is not None:
                images["plots/masked_reconstruction"] = wandb.Image(fig)
                plt.close(fig)
                images.update(masked_stats)
        except Exception as e:
            print(f"    Warning: masked reconstruction plot failed: {e}")

    # 4b. Reconstruction error vs. streamwise position (masked mode only)
    if args.pretrain_mode == "masked":
        try:
            fig, streamwise_stats = fig_reconstruction_vs_streamwise(results)
            if fig is not None:
                images["plots/reconstruction_vs_streamwise"] = wandb.Image(fig)
                plt.close(fig)
            images.update(streamwise_stats)
        except Exception as e:
            print(f"    Warning: reconstruction vs streamwise plot failed: {e}")

    # 5. Attention entropy (as scalar metrics for native wandb line charts)
    try:
        entropy_stats = compute_attention_entropy(results)
        images.update(entropy_stats)
    except Exception as e:
        print(f"    Warning: entropy computation failed: {e}")

    # 5b. Wake alignment — scalar metric for sweep ranking
    try:
        wake_stats = compute_wake_alignment(results)
        images.update(wake_stats)
    except Exception as e:
        print(f"    Warning: wake alignment computation failed: {e}")

    # 6. Embedding t-SNE (skip if sklearn missing or too slow)
    try:
        fig = fig_embedding_tsne(results, max_points=min(2000, args.plot_n_samples * 9))
        if fig is not None:
            images["plots/embedding_tsne"] = wandb.Image(fig)
            plt.close(fig)
    except Exception as e:
        print(f"    Warning: t-SNE plot failed: {e}")

    # 7. Profile embedding diagnostics (if profiles are enabled)
    #    Use layout-selected sample indices when available
    try:
        profile_sample_indices = None
        if layout_aware and len(val_match) > 0:
            profile_sample_indices = np.array([int(val_match[0])])
        fig, profile_stats = fig_profile_embeddings(
            results, sample_indices=profile_sample_indices,
        )
        if fig is not None:
            images["plots/profile_embeddings"] = wandb.Image(fig)
            plt.close(fig)
        images.update(profile_stats)
    except Exception as e:
        print(f"    Warning: profile embedding plot failed: {e}")

    # 7b. Profile contribution ratio — how much signal profiles add
    try:
        if "profile_embed" in results:
            prof_emb = results["profile_embed"]       # (S, N, D)
            final_emb = results["embeddings"]         # (S, N, D_embed)
            real = ~results["attention_mask"]          # (S, N)
            prof_norms = np.linalg.norm(prof_emb, axis=-1)   # (S, N)
            final_norms = np.linalg.norm(final_emb, axis=-1)  # (S, N)
            mean_prof = prof_norms[real].mean()
            mean_final = final_norms[real].mean()
            if mean_final > 1e-10:
                images["profile/contribution_ratio"] = float(mean_prof / mean_final)
            else:
                images["profile/contribution_ratio"] = 0.0
    except Exception as e:
        print(f"    Warning: profile contribution ratio failed: {e}")

    dt = time.time() - t0
    print(f"  Generated {len(images)} plot(s) in {dt:.1f}s")
    return images


# =============================================================================
# TRAINING LOOPS
# =============================================================================

# --- Power mode ---

def train_one_epoch_power(model, loader, optimizer, device, epoch,
                          log_wandb=False, scaler=None):
    model.train()
    total_loss = 0.0
    grad_norm_sum = 0.0
    n_batches = 0

    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            predicted_power, _, _ = model(batch, need_weights=False)
            loss = masked_mse_loss(
                predicted_power, batch["target_power"], batch["attention_mask"],
            )

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        grad_norm_sum += grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    avg_grad_norm = grad_norm_sum / max(n_batches, 1)
    return avg_loss, avg_grad_norm


@torch.no_grad()
def evaluate_power(model, loader, device, layout_names=None):
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    n_batches = 0

    # Per-layout accumulators: {layout_idx: {"mse_sum": ..., "mae_sum": ..., "count": ...}}
    per_layout_acc = {}

    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            predicted_power, _, _ = model(batch, need_weights=False)

        real_mask = ~batch["attention_mask"]

        mse = masked_mse_loss(predicted_power, batch["target_power"], batch["attention_mask"])
        total_loss += mse.item()

        mae = ((predicted_power - batch["target_power"]).abs() * real_mask.float()).sum()
        mae = mae / real_mask.sum()
        total_mae += mae.item()

        # Per-layout metrics
        if layout_names is not None and "layout_idx" in batch:
            diff_sq = (predicted_power - batch["target_power"]) ** 2  # (B, N)
            diff_abs = (predicted_power - batch["target_power"]).abs()  # (B, N)
            real_f = real_mask.float()  # (B, N)
            n_real_per_sample = real_f.sum(dim=1)  # (B,)
            per_sample_mse = (diff_sq * real_f).sum(dim=1) / n_real_per_sample.clamp(min=1)
            per_sample_mae = (diff_abs * real_f).sum(dim=1) / n_real_per_sample.clamp(min=1)

            layout_idxs = batch["layout_idx"]  # (B,)
            for li in layout_idxs.unique().tolist():
                mask_li = layout_idxs == li
                if li not in per_layout_acc:
                    per_layout_acc[li] = {"mse_sum": 0.0, "mae_sum": 0.0, "count": 0}
                per_layout_acc[li]["mse_sum"] += per_sample_mse[mask_li].sum().item()
                per_layout_acc[li]["mae_sum"] += per_sample_mae[mask_li].sum().item()
                per_layout_acc[li]["count"] += mask_li.sum().item()

        n_batches += 1

    avg_mse = total_loss / max(n_batches, 1)
    avg_mae = total_mae / max(n_batches, 1)

    # Build per-layout metrics dict (aggregate indices sharing the same name)
    per_layout_metrics = {}
    if layout_names is not None:
        name_acc = {}
        for li, acc in per_layout_acc.items():
            name = layout_names.get(li, f"layout_{li}")
            if name not in name_acc:
                name_acc[name] = {"mse_sum": 0.0, "mae_sum": 0.0, "count": 0}
            name_acc[name]["mse_sum"] += acc["mse_sum"]
            name_acc[name]["mae_sum"] += acc["mae_sum"]
            name_acc[name]["count"] += acc["count"]
        for name, agg in name_acc.items():
            count = max(agg["count"], 1)
            per_layout_metrics[name] = {
                "mse": agg["mse_sum"] / count,
                "mae": agg["mae_sum"] / count,
            }

    return avg_mse, avg_mae, per_layout_metrics


# --- Masked mode ---

def train_one_epoch_masked(model, loader, optimizer, device, epoch,
                           feature_weights=None, log_wandb=False, scaler=None):
    model.train()
    total_loss = 0.0
    grad_norm_sum = 0.0
    n_batches = 0

    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            result = model(batch)
            loss_dict = bert_reconstruction_loss(
                result["predicted_obs"],
                result["original_obs"],
                result["predict_mask"],
                feature_weights=feature_weights,
            )
            loss = loss_dict["loss"]

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        grad_norm_sum += grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    avg_grad_norm = grad_norm_sum / max(n_batches, 1)
    return avg_loss, avg_grad_norm


@torch.no_grad()
def evaluate_masked(model, loader, device, feature_names, feature_weights=None,
                    layout_names=None):
    """
    Evaluate masked reconstruction.

    Returns:
        avg_loss: scalar
        per_feature_mse: dict mapping feature_name → MSE
        avg_cosine_sim: mean cosine similarity at masked positions
        per_layout_metrics: dict mapping layout_name → {"loss": x}
    """
    model.eval()
    total_loss = 0.0
    total_per_feature = None
    total_cosine_sim = 0.0
    n_batches = 0

    # Per-layout accumulators: {layout_idx: {"loss_sum": ..., "count": ...}}
    per_layout_acc = {}

    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            result = model(batch)
            loss_dict = bert_reconstruction_loss(
                result["predicted_obs"],
                result["original_obs"],
                result["predict_mask"],
                feature_weights=feature_weights,
            )

        total_loss += loss_dict["loss"].item()

        pf = loss_dict["per_feature"]
        if total_per_feature is None:
            total_per_feature = pf
        else:
            total_per_feature = total_per_feature + pf

        # Cosine similarity at masked positions
        mask = result["predict_mask"]
        if mask.sum() > 0:
            pred_flat = result["predicted_obs"][mask]
            orig_flat = result["original_obs"][mask]
            cos_sim = F.cosine_similarity(pred_flat, orig_flat, dim=-1).mean()
            total_cosine_sim += cos_sim.item()

        # Per-layout loss
        if layout_names is not None and "layout_idx" in batch:
            predict_mask = result["predict_mask"]  # (B, N)
            sq_err = (result["predicted_obs"] - result["original_obs"]) ** 2  # (B, N, D)
            # Per-sample mean reconstruction error over masked positions
            mask_expanded = predict_mask.unsqueeze(-1).float()  # (B, N, 1)
            n_masked_per_sample = predict_mask.float().sum(dim=1)  # (B,)
            obs_dim = sq_err.shape[-1]
            per_sample_loss = (sq_err * mask_expanded).sum(dim=(1, 2)) / (
                n_masked_per_sample * obs_dim
            ).clamp(min=1)  # (B,)

            layout_idxs = batch["layout_idx"]  # (B,)
            for li in layout_idxs.unique().tolist():
                mask_li = layout_idxs == li
                if li not in per_layout_acc:
                    per_layout_acc[li] = {"loss_sum": 0.0, "count": 0}
                per_layout_acc[li]["loss_sum"] += per_sample_loss[mask_li].sum().item()
                per_layout_acc[li]["count"] += mask_li.sum().item()

        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    avg_per_feature = total_per_feature / max(n_batches, 1) if total_per_feature is not None else {}
    avg_cosine_sim = total_cosine_sim / max(n_batches, 1)

    # Build per-feature dict
    per_feature_dict = {}
    if total_per_feature is not None:
        for i, name in enumerate(feature_names):
            if i < avg_per_feature.shape[0]:
                per_feature_dict[name] = avg_per_feature[i].item()

    # Build per-layout metrics dict (aggregate indices sharing the same name)
    per_layout_metrics = {}
    if layout_names is not None:
        name_acc = {}
        for li, acc in per_layout_acc.items():
            name = layout_names.get(li, f"layout_{li}")
            if name not in name_acc:
                name_acc[name] = {"loss_sum": 0.0, "count": 0}
            name_acc[name]["loss_sum"] += acc["loss_sum"]
            name_acc[name]["count"] += acc["count"]
        for name, agg in name_acc.items():
            count = max(agg["count"], 1)
            per_layout_metrics[name] = {"loss": agg["loss_sum"] / count}

    return avg_loss, per_feature_dict, avg_cosine_sim, per_layout_metrics


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = tyro.cli(Args)

    assert args.pretrain_mode in ("power", "masked"), \
        f"Invalid pretrain_mode: {args.pretrain_mode}. Must be 'power' or 'masked'."

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Pretraining mode: {args.pretrain_mode}")

    # =========================================================================
    # Wandb init
    # =========================================================================
    use_wandb = WANDB_AVAILABLE and args.track

    if use_wandb:
        mode = "snap" if args.snapshot else f"hist{args.history_length}"
        gf_tag = "_g" + "".join(args.global_features) if args.global_features else ""
        run_name = (f"{args.exp_name}_{args.pretrain_mode}_{mode}{gf_tag}"
                    f"_e{args.embed_dim}_L{args.num_layers}_H{args.num_heads}")
        if args.pretrain_mode == "masked":
            run_name += f"_m{args.mask_ratio:.0%}"

        tags = [t.strip() for t in args.wandb_tags.split(",")] if args.wandb_tags else None
        if tags is None:
            tags = []
        tags.append(args.pretrain_mode)

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            name=run_name,
            tags=tags,
            config=vars(args),
        )
        print(f"Wandb: logging to {args.wandb_project_name}/{run_name}")
    elif not WANDB_AVAILABLE and args.track:
        print("Wandb: not available (install with pip install wandb)")
    else:
        print("Wandb: disabled (--no-track)")

    # =========================================================================
    # Data
    # =========================================================================
    files = sorted(glob(f"{args.data_dir}/layout_*.h5"))
    if not files:
        print(f"No layout files found in {args.data_dir}")
        return
    print(f"Found {len(files)} layout file(s)")

    # --- Choose input features based on mode ---
    if args.pretrain_mode == "power":
        # Power mode: use configured features, power is always the target (not an input)
        input_features = list(args.features)
    else:
        # Masked mode: all features are both input AND target.
        # Auto-add 'power' if not already present, since the model
        # should learn to predict it from spatial context.
        input_features = list(args.features)
        if "power" not in input_features:
            input_features.append("power")

    # Log global feature configuration
    if args.global_features:
        all_features = list(input_features)
        for gf in args.global_features:
            if gf not in all_features:
                all_features.insert(0, gf)
        n_global = len(args.global_features)
        n_hist = len(all_features) - n_global
        obs_dim_expected = n_global + args.history_length * n_hist
        print(f"Features: {all_features} (global: {args.global_features}) | "
              f"obs_dim = {n_global} + {args.history_length}×{n_hist} = {obs_dim_expected}")

    scaling_limits = {"power": (0.0, args.power_max)}

    # --- Create dataset ---
    common_kwargs = dict(
        layout_files=files,
        max_turbines=None,  # auto
        features=input_features,
        global_features=args.global_features,
        action_type=None,  # pretrain doesn't need actions
        scaling_limits=scaling_limits,
        use_wd_deviation=args.use_wd_deviation,
        wd_scale_range=args.wd_scale_range,
        use_wind_relative_pos=args.use_wind_relative_pos,
        rotate_profiles=args.rotate_profiles,
    )

    if args.snapshot:
        dataset = WindFarmSnapshotDataset(skip_steps=args.skip_steps, **common_kwargs)
    else:
        dataset = WindFarmPretrainDataset(history_length=args.history_length, **common_kwargs)

    # Train/val split
    n_layouts = len(dataset.layouts)
    unique_layout_names = sorted(set(
        l.get("layout_name", f"layout_{i}") for i, l in enumerate(dataset.layouts)
    ))
    n_unique_layouts = len(unique_layout_names)
    train_layout_idxs = None
    val_layout_idxs = None

    if args.val_split_by_layout and n_unique_layouts > 1:
        train_set, val_set, train_layout_idxs, val_layout_idxs = split_by_layout(
            dataset, args.val_split, args.val_layout_names, args.seed,
        )
        n_train, n_val = len(train_set), len(val_set)
        # Log layout assignments with names
        def _layout_names(idxs):
            return [dataset.layouts[i].get("layout_name", f"layout_{i}") for i in idxs]
        print(f"Layout-based split:")
        print(f"  Train layouts ({len(train_layout_idxs)}): {_layout_names(train_layout_idxs)}  ({n_train} samples)")
        print(f"  Val layouts   ({len(val_layout_idxs)}):   {_layout_names(val_layout_idxs)}  ({n_val} samples)")

    else:
        if args.val_split_by_layout and n_unique_layouts <= 1:
            print("Warning: --val-split-by-layout requested but only 1 unique layout found. "
                  "Falling back to random split.")
        n_val = int(len(dataset) * args.val_split)
        n_train = len(dataset) - n_val
        train_set, val_set = random_split(
            dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(args.seed),
        )
        print(f"Train: {n_train} samples, Val: {n_val} samples")

    # Build layout name mapping for wandb and checkpoints
    layout_name_map = {
        i: dataset.layouts[i].get("layout_name", f"layout_{i}")
        for i in range(n_layouts)
    }

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        pin_memory=True,
    )

    # --- Determine dimensions from first sample ---
    sample = dataset[0]
    obs_dim = sample["obs"].shape[-1]
    max_turbines = sample["obs"].shape[0]
    has_profiles = "receptivity" in sample
    n_profile_dirs = sample["receptivity"].shape[-1] if has_profiles else 0

    print(f"\nModel config:")
    print(f"  pretrain_mode: {args.pretrain_mode}")
    print(f"  input features: {input_features}")
    print(f"  obs_dim (input features × history): {obs_dim}")
    print(f"  max_turbines: {max_turbines}")
    print(f"  profiles: {'yes, ' + str(n_profile_dirs) + ' dirs' if has_profiles else 'no'}")
    print(f"  embed_dim: {args.embed_dim}")
    print(f"  num_heads: {args.num_heads}")
    print(f"  num_layers: {args.num_layers}")
    if args.pretrain_mode == "masked":
        print(f"  mask_ratio: {args.mask_ratio}")
        print(f"  mask_replace_prob: {args.mask_replace_prob}")
        print(f"  mask_random_prob: {args.mask_random_prob}")

    # --- Build model (using REAL TransformerActor backbone) ---
    actor = TransformerActor(
        obs_dim_per_turbine=obs_dim,
        action_dim_per_turbine=1,        # dummy, head is discarded
        embed_dim=args.embed_dim,
        pos_embed_dim=args.pos_embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        action_scale=1.0,                # dummy
        action_bias=0.0,                 # dummy
        # Positional encoding (must match RL config)
        pos_encoding_type=args.pos_encoding_type,
        rel_pos_hidden_dim=args.rel_pos_hidden_dim,
        rel_pos_per_head=args.rel_pos_per_head,
        pos_embedding_mode=args.pos_embedding_mode,
        # Profile encoding (must match RL config)
        profile_encoding=args.profile_encoding_type,
        profile_encoder_hidden=args.profile_encoder_hidden,
        n_profile_directions=n_profile_dirs if has_profiles else args.n_profile_directions,
        profile_fusion_type=args.profile_fusion_type,
        profile_embed_mode=args.profile_embed_mode,
        args=args,  # for profile_encoder_kwargs
    )

    if args.pretrain_mode == "power":
        model = PowerPredictionModel(actor).to(device)
    else:
        model = MaskedTurbineModel(
            actor=actor,
            obs_dim=obs_dim,
            mask_ratio=args.mask_ratio,
            mask_replace_prob=args.mask_replace_prob,
            mask_random_prob=args.mask_random_prob,
        ).to(device)
    # model = torch.compile(model)  # requires GCC/12.3.0 on some HPC systems

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {n_params:,}")

    # Build feature weights for masked mode (optionally upweight power prediction)
    feature_weights = None
    if args.pretrain_mode == "masked" and args.predict_power_weight != 1.0:
        n_raw = len(input_features)
        w = torch.ones(obs_dim, device=device)

        power_idx = input_features.index("power") if "power" in input_features else -1
        if power_idx >= 0:
            if args.snapshot:
                w[power_idx] = args.predict_power_weight
            else:
                H = obs_dim // n_raw
                w[power_idx * H : (power_idx + 1) * H] = args.predict_power_weight
        feature_weights = w
        print(f"  Feature weights: {feature_weights.tolist()}")

    # Construct per-obs-dim feature names for logging (masked mode)
    feature_names = []
    if args.pretrain_mode == "masked":
        if args.snapshot or obs_dim == len(input_features):
            feature_names = list(input_features)
        else:
            H = obs_dim // len(input_features)
            for feat in input_features:
                for t in range(H):
                    feature_names.append(f"{feat}_t{t}")

    # Log dataset and model info to wandb
    if use_wandb:
        wandb.config.update({
            "obs_dim": obs_dim,
            "max_turbines": max_turbines,
            "has_profiles": has_profiles,
            "n_profile_dirs": n_profile_dirs,
            "n_train": n_train,
            "n_val": n_val,
            "n_params": n_params,
            "n_layout_files": len(files),
            "input_features": input_features,
            "val_split_by_layout": args.val_split_by_layout,
            "val_layout_indices": val_layout_idxs,
            "train_layout_indices": train_layout_idxs,
            "layout_names": layout_name_map,
        }, allow_val_change=True)
        wandb.watch(model, log="all", log_freq=500)

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    warmup_epochs = 5
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - warmup_epochs, eta_min=args.lr * 0.01,
    )
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0 / warmup_epochs, end_factor=1.0,
        total_iters=warmup_epochs,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    # --- Output dir ---
    os.makedirs(args.save_dir, exist_ok=True)

    # Determine if plots should be generated
    do_plots = use_wandb and args.plot_every > 0

    # =========================================================================
    # Training
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"Starting {args.pretrain_mode} pretraining: {args.epochs} epochs")
    if do_plots:
        print(f"Diagnostic plots every {args.plot_every} epochs ({args.plot_n_samples} samples)")
    print(f"{'='*60}")

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # ---- Train ----
        if args.pretrain_mode == "power":
            train_loss, train_grad_norm = train_one_epoch_power(
                model, train_loader, optimizer, device, epoch,
                log_wandb=use_wandb, scaler=scaler,
            )
        else:
            train_loss, train_grad_norm = train_one_epoch_masked(
                model, train_loader, optimizer, device, epoch,
                feature_weights=feature_weights, log_wandb=use_wandb,
                scaler=scaler,
            )

        # ---- Validate ----
        if args.pretrain_mode == "power":
            val_mse, val_mae, per_layout = evaluate_power(
                model, val_loader, device, layout_name_map,
            )
            val_loss = val_mse
        else:
            val_loss, per_feature_mse, cosine_sim, per_layout = evaluate_masked(
                model, val_loader, device, feature_names, feature_weights,
                layout_name_map,
            )

        scheduler.step()
        dt = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        # ---- Print ----
        if args.pretrain_mode == "power":
            print(f"Epoch {epoch:3d}/{args.epochs} | "
                  f"train_mse: {train_loss:.6f} | "
                  f"val_mse: {val_mse:.6f} | "
                  f"val_mae: {val_mae:.4f} | "
                  f"lr: {lr_now:.2e} | {dt:.1f}s")
        else:
            feat_str = "  ".join(f"{k}: {v:.5f}" for k, v in per_feature_mse.items())
            print(f"Epoch {epoch:3d}/{args.epochs} | "
                  f"train: {train_loss:.6f} | "
                  f"val: {val_loss:.6f} | "
                  f"cos_sim: {cosine_sim:.4f} | "
                  f"lr: {lr_now:.2e} | {dt:.1f}s")
            if feat_str:
                print(f"  per-feature MSE: {feat_str}")

        # ---- Wandb logging ----
        if use_wandb:
            log_dict = {
                "epoch": epoch,
                "train/loss": train_loss,
                "train/grad_norm": train_grad_norm,
                "val/loss": val_loss,
                "lr": lr_now,
                "epoch_time_s": dt,
                "throughput/samples_per_sec": len(train_loader.dataset) / dt,
            }
            if device.type == "cuda":
                log_dict["gpu/mem_allocated_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
                log_dict["gpu/mem_reserved_gb"] = torch.cuda.max_memory_reserved(device) / 1e9
                torch.cuda.reset_peak_memory_stats(device)

            if args.pretrain_mode == "power":
                log_dict["val/mse"] = val_mse
                log_dict["val/mae"] = val_mae
                log_dict["val/mae_watts"] = val_mae * args.power_max
                for name, metrics in per_layout.items():
                    log_dict[f"val/mse_{name}"] = metrics["mse"]
                    log_dict[f"val/mae_{name}"] = metrics["mae"]
            else:
                log_dict["val/cosine_sim"] = cosine_sim
                for k, v in per_feature_mse.items():
                    log_dict[f"val/mse_{k}"] = v
                for name, metrics in per_layout.items():
                    log_dict[f"val/loss_{name}"] = metrics["loss"]

            # --- Diagnostic plots (every N epochs + first + last) ---
            if do_plots and (epoch % args.plot_every == 0
                            or epoch == 1
                            or epoch == args.epochs):
                plot_images = generate_diagnostic_plots(
                    model, val_set, device, epoch, args,
                    train_set=train_set,
                    dataset=dataset,
                    train_layout_idxs=train_layout_idxs,
                    val_layout_idxs=val_layout_idxs,
                )
                log_dict.update(plot_images)

            wandb.log(log_dict)

        # ---- Checkpointing ----
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            checkpoint = {
                "epoch": epoch,
                "pretrain_mode": args.pretrain_mode,
                "encoder_state_dict": get_encoder_state_dict(model.actor),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "args": vars(args),
                "obs_dim": obs_dim,
                "max_turbines": max_turbines,
                "n_profile_dirs": n_profile_dirs,
                "input_features": input_features,
                "val_split_by_layout": args.val_split_by_layout,
                "val_layout_indices": val_layout_idxs,
                "train_layout_indices": train_layout_idxs,
                "layout_names": layout_name_map,
            }
            if args.pretrain_mode == "power":
                checkpoint["val_mse"] = val_mse
                checkpoint["val_mae"] = val_mae
            torch.save(checkpoint, f"{args.save_dir}/best.pt")
            print(f"  → Saved best model (val_loss={val_loss:.6f})")

            if use_wandb:
                wandb.run.summary["best_val_loss"] = best_val_loss
                wandb.run.summary["best_epoch"] = epoch
                if args.pretrain_mode == "power":
                    wandb.run.summary["best_val_mae"] = val_mae

        # Periodic save
        if epoch % args.save_every == 0:
            torch.save({
                "epoch": epoch,
                "pretrain_mode": args.pretrain_mode,
                "encoder_state_dict": get_encoder_state_dict(model.actor),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "args": vars(args),
                "layout_names": layout_name_map,
            }, f"{args.save_dir}/epoch_{epoch:04d}.pt")

    # ---- Final save ----
    final_checkpoint = {
        "epoch": args.epochs,
        "pretrain_mode": args.pretrain_mode,
        "encoder_state_dict": get_encoder_state_dict(model.actor),
        "model_state_dict": model.state_dict(),
        "val_loss": val_loss,
        "args": vars(args),
        "obs_dim": obs_dim,
        "max_turbines": max_turbines,
        "n_profile_dirs": n_profile_dirs,
        "input_features": input_features,
        "val_split_by_layout": args.val_split_by_layout,
        "val_layout_indices": val_layout_idxs,
        "train_layout_indices": train_layout_idxs,
        "layout_names": layout_name_map,
    }
    if args.pretrain_mode == "power":
        final_checkpoint["val_mae"] = val_mae
    torch.save(final_checkpoint, f"{args.save_dir}/final.pt")

    print(f"\n{'='*60}")
    print(f"Pretraining complete! ({args.pretrain_mode} mode)")
    print(f"  Best val loss: {best_val_loss:.6f}")
    print(f"  Checkpoints saved to: {args.save_dir}/")
    print(f"{'='*60}")

    # ---- Wandb artifact ----
    if use_wandb:
        wandb.run.summary["final_val_loss"] = val_loss
        if args.pretrain_mode == "power":
            wandb.run.summary["final_val_mae"] = val_mae

        best_path = f"{args.save_dir}/best.pt"
        if os.path.exists(best_path):
            artifact = wandb.Artifact(
                name=f"pretrained-encoder-{args.pretrain_mode}",
                type="model",
                description=f"Best {args.pretrain_mode} pretrained encoder "
                            f"(val_loss={best_val_loss:.6f})",
                metadata={
                    "pretrain_mode": args.pretrain_mode,
                    "best_epoch": int(wandb.run.summary.get("best_epoch", -1)),
                    "obs_dim": obs_dim,
                    "embed_dim": args.embed_dim,
                    "num_layers": args.num_layers,
                },
            )
            artifact.add_file(best_path)
            wandb.log_artifact(artifact)

        wandb.finish()


if __name__ == "__main__":
    main()