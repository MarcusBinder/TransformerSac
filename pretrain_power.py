"""
Power Prediction Pretraining for Wind Farm Transformer

Pretrains a transformer encoder to predict per-turbine power output
from (ws, wd, yaw) + positions + profiles. The learned encoder weights
can then be transferred to the SAC actor/critic for RL fine-tuning.

Includes periodic diagnostic visualizations logged to wandb:
  - Attention maps overlaid on farm layout
  - Predicted vs. actual power scatter
  - Per-turbine error heatmap
  - Attention entropy per layer/head
  - t-SNE of token embeddings

Usage:
    python pretrain_power.py --data-dir ./pretrain_data --epochs 50
    python pretrain_power.py --data-dir ./pretrain_data --snapshot  # no history
    python pretrain_power.py --data-dir ./pretrain_data --no-track  # disable wandb
    python pretrain_power.py --data-dir ./pretrain_data --plot-every 5  # plots every 5 epochs

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
from torch.utils.data import DataLoader, random_split
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

from data_loader import create_pretrain_dataloader, WindFarmPretrainDataset, WindFarmSnapshotDataset

# Import real RL model architecture (ensures identical weight keys)
from transformer_sac_windfarm_v26 import (
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
    """Command-line arguments for power prediction pretraining."""

    # === Experiment Settings ===
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
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

    # === Training Hyperparameters ===
    epochs: int = 50
    """Number of training epochs."""
    batch_size: int = 256
    """Batch size."""
    lr: float = 3e-4
    """Learning rate."""
    weight_decay: float = 1e-4
    """AdamW weight decay."""
    val_split: float = 0.1
    """Fraction of data held out for validation."""
    # num_workers: int = 4  # Removed. We just load it all into ram now.
    # """DataLoader workers."""

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
# PRETRAINING MODEL (real TransformerActor backbone + power head)
# =============================================================================

# Keys that belong to the actor head, NOT the encoder
ACTOR_HEAD_KEYS = {"fc_mean", "fc_logstd", "action_scale", "action_bias_val"}


def get_encoder_state_dict(actor: TransformerActor) -> dict:
    """Extract only encoder weights from actor (excluding action heads)."""
    return {
        k: v for k, v in actor.state_dict().items()
        if not any(k.startswith(head) for head in ACTOR_HEAD_KEYS)
    }


class PowerPredictionModel(nn.Module):
    """
    Wraps the REAL TransformerActor backbone + power prediction head.

    Uses the actor's encoder path (obs_encoder → pos_encoder → input_proj
    → profiles → transformer) so that saved weights have identical keys
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

    def _encode(
        self,
        obs: torch.Tensor,
        positions: torch.Tensor,
        attention_mask: torch.Tensor,
        receptivity: torch.Tensor = None,
        influence: torch.Tensor = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Replicate the actor's encoder path (forward up to transformer output).
        This is lines 958-1004 of the RL script, minus the action heads. (mostly)

        Args:
            need_weights: If True, return per-layer attention weights for
                          visualization. Slightly slower due to attention
                          weight computation.

        Returns:
            h: (B, N, embed_dim) transformer output
            attn_weights: list of (B, H, N, N) per layer (empty if need_weights=False)
        """
        actor = self.actor

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
            h: (B, N, embed_dim) encoder output (only if need_weights)
            attn_weights: list of (B, H, N, N) per layer (only if need_weights)
        """
        h, attn_weights = self._encode(
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
# LOSS FUNCTION
# =============================================================================

def masked_mse_loss(
    predicted: torch.Tensor,    # (B, N)
    target: torch.Tensor,       # (B, N)
    attention_mask: torch.Tensor,  # (B, N) True=padding
) -> torch.Tensor:
    """MSE loss computed only over real (non-padded) turbines."""
    real_mask = ~attention_mask  # True = real turbine
    n_real = real_mask.sum()

    if n_real == 0:
        return torch.tensor(0.0, device=predicted.device)

    diff = (predicted - target) ** 2
    loss = (diff * real_mask.float()).sum() / n_real
    return loss


# =============================================================================
# DIAGNOSTIC PLOT COLLECTION (inference with attention weights)
# =============================================================================

@torch.no_grad()
def collect_plot_data(model, dataset, device, n_samples=256, batch_size=64):
    """
    Run inference on a subset of the dataset, collecting predictions,
    embeddings, and attention weights for diagnostic plots.

    Returns dict with numpy arrays:
        predicted_power: (S, N)
        target_power:    (S, N)
        positions:       (S, N, 2)
        attention_mask:  (S, N)  bool, True=padding
        embeddings:      (S, N, embed_dim)
        obs:             (S, N, obs_dim)
        attn_weights:    list[layer] of (S, H, N, N)
    """
    model.eval()

    # Deterministic subset
    n = min(n_samples, len(dataset))
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(42))[:n]
    subset = torch.utils.data.Subset(dataset, indices.tolist())
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, pin_memory=True)

    all_pred, all_target, all_pos, all_mask, all_embed, all_obs = [], [], [], [], [], []
    all_attn = None

    for batch in loader:
        batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            pred, embed, attn = model(batch_dev, need_weights=True)

        all_pred.append(pred.cpu().float())
        all_target.append(batch["target_power"])
        all_pos.append(batch["positions"])
        all_mask.append(batch["attention_mask"])
        all_embed.append(embed.cpu().float())
        all_obs.append(batch["obs"])

        if all_attn is None:
            all_attn = [[] for _ in range(len(attn))]
        for layer_idx, aw in enumerate(attn):
            all_attn[layer_idx].append(aw.cpu().float())

    return {
        "predicted_power": torch.cat(all_pred).numpy(),
        "target_power": torch.cat(all_target).numpy(),
        "positions": torch.cat(all_pos).numpy(),
        "attention_mask": torch.cat(all_mask).numpy(),
        "embeddings": torch.cat(all_embed).numpy(),
        "obs": torch.cat(all_obs).numpy(),
        "attn_weights": [torch.cat(layer_list).numpy() for layer_list in all_attn],
    }


# =============================================================================
# DIAGNOSTIC PLOTS (return matplotlib figure objects)
# =============================================================================

def fig_attention_on_layout(results, sample_idx=0, top_k=3):
    """
    Attention maps overlaid on the physical farm layout.
    Turbine nodes colored by target power, directed edges by attention weight.

    Returns one figure per layer (head-averaged) + a per-head figure for the
    last layer.
    """
    n_layers = len(results["attn_weights"])
    pos = results["positions"][sample_idx]          # (N, 2)
    mask = results["attention_mask"][sample_idx]     # (N,)
    target = results["target_power"][sample_idx]     # (N,)
    real = ~mask
    n_real = int(real.sum())
    pos_real = pos[:n_real]
    target_real = target[:n_real]

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
            c=target_real, cmap="viridis", s=120,
            edgecolors="black", linewidth=1.0, zorder=5,
        )
        for t in range(n):
            ax.annotate(str(t), pos_real[t], fontsize=7, ha="center",
                        va="center", fontweight="bold", color="white", zorder=6)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x (norm)")
        ax.set_ylabel("y (norm)")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        return sc

    def _normalize_rows(a):
        row_max = a.max(axis=1, keepdims=True)
        row_max = np.where(row_max > 0, row_max, 1.0)
        return a / row_max

    # --- Per-layer figure (head-averaged) ---
    n_cols = min(n_layers, 4)
    fig_layers, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5), squeeze=False)
    axes = axes[0]
    for l in range(n_cols):
        attn_l = results["attn_weights"][l][sample_idx]  # (H, N, N)
        attn_mean = _normalize_rows(attn_l.mean(axis=0)[:n_real, :n_real])
        sc = _draw(axes[l], attn_mean, f"Layer {l} (head avg)")
    fig_layers.colorbar(sc, ax=axes[-1], shrink=0.8, label="Target Power (norm)")
    fig_layers.suptitle(f"Attention on Layout — sample {sample_idx}", fontsize=12, y=1.02)
    fig_layers.tight_layout()

    # --- Per-head figure for last layer ---
    n_heads = results["attn_weights"][-1].shape[1]
    fig_heads = None
    if n_heads <= 8:
        fig_heads, axes_h = plt.subplots(1, n_heads, figsize=(4 * n_heads, 4), squeeze=False)
        axes_h = axes_h[0]
        for h in range(n_heads):
            attn_h = results["attn_weights"][-1][sample_idx][h, :n_real, :n_real]
            attn_h = _normalize_rows(attn_h)
            _draw(axes_h[h], attn_h, f"Head {h}")
        fig_heads.suptitle(f"Per-Head Attention (last layer, sample {sample_idx})",
                           fontsize=12, y=1.02)
        fig_heads.tight_layout()

    return fig_layers, fig_heads


def fig_pred_vs_actual(results):
    """Scatter of predicted vs. actual power, colored by wind direction."""
    pred = results["predicted_power"]
    target = results["target_power"]
    mask = results["attention_mask"]
    obs = results["obs"]
    real = ~mask

    pred_flat = pred[real]
    target_flat = target[real]

    # Extract wind direction for coloring (feature idx 1, last timestep)
    n_features = 3  # ws, wd, yaw
    obs_dim = obs.shape[-1]
    history_len = obs_dim // n_features
    wd_idx = 1 * history_len + (history_len - 1)
    wd_flat = obs[:, :, wd_idx][real] if wd_idx < obs_dim else np.zeros_like(pred_flat)

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
    """Per-turbine MAE plotted on the farm grid, side-by-side with mean power."""
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
    avg_pos = np.zeros((n_turbines, 2))
    active = np.zeros(n_turbines, dtype=bool)

    for t in range(n_turbines):
        valid = real[:, t]
        if valid.sum() > 0:
            turbine_mae[t] = abs_errors[valid, t].mean()
            avg_power[t] = target[valid, t].mean()
            avg_pos[t] = pos[valid, t].mean(axis=0)
            active[t] = True

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: MAE
    sc1 = ax1.scatter(avg_pos[active, 0], avg_pos[active, 1], c=turbine_mae[active],
                       cmap="hot_r", s=250, edgecolors="black", linewidth=1.0, zorder=5)
    for t in np.where(active)[0]:
        ax1.annotate(f"{turbine_mae[t]:.4f}", avg_pos[t], fontsize=6,
                     ha="center", va="center", zorder=6)
    fig.colorbar(sc1, ax=ax1, label="MAE")
    ax1.set_title("Per-Turbine MAE")
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)

    # Right: Mean power
    sc2 = ax2.scatter(avg_pos[active, 0], avg_pos[active, 1], c=avg_power[active],
                       cmap="viridis", s=250, edgecolors="black", linewidth=1.0, zorder=5)
    for t in np.where(active)[0]:
        ax2.annotate(f"{avg_power[t]:.3f}", avg_pos[t], fontsize=6, ha="center",
                     va="center", color="white", fontweight="bold", zorder=6)
    fig.colorbar(sc2, ax=ax2, label="Mean Power (norm)")
    ax2.set_title("Mean Target Power")
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Error & Power Distribution Across Farm", fontsize=12)
    fig.tight_layout()
    return fig


def fig_attention_entropy(results):
    """Attention entropy per layer/head — lower = more structured attention."""
    mask = results["attention_mask"]
    n_layers = len(results["attn_weights"])
    real = ~mask  # (S, N)

    eps = 1e-8
    layer_entropies = []
    for l in range(n_layers):
        attn = results["attn_weights"][l]  # (S, H, N, N)
        entropy = -(attn * np.log(attn + eps)).sum(axis=-1)  # (S, H, N)
        real_exp = real[:, np.newaxis, :]  # (S, 1, N)
        per_head = (entropy * real_exp).sum(axis=(0, 2)) / real_exp.sum(axis=(0, 2))  # (H,)
        layer_entropies.append(per_head)

    layer_entropies = np.array(layer_entropies)  # (L, H)
    n_heads = layer_entropies.shape[1]
    n_real_avg = real.sum(axis=1).mean()
    uniform_entropy = np.log(n_real_avg)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for h in range(n_heads):
        ax.plot(range(n_layers), layer_entropies[:, h], "o-", alpha=0.4,
                markersize=4, label=f"Head {h}")
    ax.plot(range(n_layers), layer_entropies.mean(axis=1), "k^-",
            linewidth=2.5, markersize=8, label="Mean", zorder=10)
    ax.axhline(uniform_entropy, color="red", linestyle="--", alpha=0.5,
               label=f"Uniform ({uniform_entropy:.2f})")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Entropy (nats)")
    ax.set_title("Attention Entropy (lower = more focused)")
    ax.set_xticks(range(n_layers))
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def fig_embedding_tsne(results, perplexity=30, max_points=2000):
    """t-SNE of transformer output embeddings colored by power and position."""
    if not HAS_SKLEARN:
        return None

    embeddings = results["embeddings"]
    target = results["target_power"]
    positions = results["positions"]
    mask = results["attention_mask"]
    real = ~mask

    emb_flat = embeddings[real]
    pow_flat = target[real]
    pos_flat = positions[real]

    if len(emb_flat) > max_points:
        idx = np.random.RandomState(42).choice(len(emb_flat), max_points, replace=False)
        emb_flat, pow_flat, pos_flat = emb_flat[idx], pow_flat[idx], pos_flat[idx]

    tsne = TSNE(n_components=2, perplexity=min(perplexity, len(emb_flat) // 4),
                random_state=42, max_iter=800)
    emb_2d = tsne.fit_transform(emb_flat)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    sc0 = axes[0].scatter(emb_2d[:, 0], emb_2d[:, 1], c=pow_flat, cmap="viridis", s=6, alpha=0.5)
    fig.colorbar(sc0, ax=axes[0]).set_label("Power (norm)")
    axes[0].set_title("By Power Output")

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


# =============================================================================
# ORCHESTRATOR: generate all plots and log to wandb
# =============================================================================

def generate_diagnostic_plots(
    model, val_dataset, device, epoch, args,
):
    """
    Collect inference results and generate all diagnostic plots.
    Returns a dict of {wandb_key: wandb.Image} ready for wandb.log().
    """
    print(f"  Generating diagnostic plots (n_samples={args.plot_n_samples})...")
    t0 = time.time()

    results = collect_plot_data(
        model, val_dataset, device,
        n_samples=args.plot_n_samples,
        batch_size=args.batch_size,
    )

    images = {}

    # 1. Attention on layout — multiple sample indices for diversity
    n_samples_available = results["predicted_power"].shape[0]
    sample_indices = np.linspace(0, n_samples_available - 1,
                                  args.plot_n_sample_indices, dtype=int)
    for i, idx in enumerate(sample_indices):
        try:
            fig_layers, fig_heads = fig_attention_on_layout(
                results, sample_idx=int(idx), top_k=args.plot_attn_top_k
            )
            images[f"plots/attention_layers_s{idx}"] = wandb.Image(fig_layers)
            plt.close(fig_layers)
            if fig_heads is not None:
                images[f"plots/attention_heads_s{idx}"] = wandb.Image(fig_heads)
                plt.close(fig_heads)
        except Exception as e:
            print(f"    Warning: attention plot failed for sample {idx}: {e}")

    # 2. Predicted vs. actual scatter
    try:
        fig = fig_pred_vs_actual(results)
        images["plots/pred_vs_actual"] = wandb.Image(fig)
        plt.close(fig)
    except Exception as e:
        print(f"    Warning: scatter plot failed: {e}")

    # 3. Error heatmap
    try:
        fig = fig_error_heatmap(results)
        images["plots/error_heatmap"] = wandb.Image(fig)
        plt.close(fig)
    except Exception as e:
        print(f"    Warning: error heatmap failed: {e}")

    # 4. Attention entropy
    try:
        fig = fig_attention_entropy(results)
        images["plots/attention_entropy"] = wandb.Image(fig)
        plt.close(fig)
    except Exception as e:
        print(f"    Warning: entropy plot failed: {e}")

    # 5. Embedding t-SNE (skip if sklearn missing or too slow)
    try:
        fig = fig_embedding_tsne(results, max_points=min(2000, args.plot_n_samples * 9))
        if fig is not None:
            images["plots/embedding_tsne"] = wandb.Image(fig)
            plt.close(fig)
    except Exception as e:
        print(f"    Warning: t-SNE plot failed: {e}")

    dt = time.time() - t0
    print(f"  Generated {len(images)} plot(s) in {dt:.1f}s")
    return images


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_one_epoch(model, loader, optimizer, device, epoch, log_wandb=False, scaler=None):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        # Move to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        with torch.amp.autocast("cuda"):
            predicted_power, _, _ = model(batch, need_weights=False)
            loss = masked_mse_loss(
                predicted_power,
                batch["target_power"],
                batch["attention_mask"],
            )

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches += 1

        # Log per-step metrics for richer training curves
        if log_wandb:
            global_step = (epoch - 1) * len(loader) + n_batches
            wandb.log({
                "train/step_loss": loss.item(),
                "train/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                "global_step": global_step,
            }, step=global_step)

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    n_batches = 0

    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        with torch.amp.autocast("cuda"):
            predicted_power, _, _ = model(batch, need_weights=False)

        real_mask = ~batch["attention_mask"]

        # MSE
        mse = masked_mse_loss(predicted_power, batch["target_power"], batch["attention_mask"])
        total_loss += mse.item()

        # MAE (in normalized units)
        mae = ((predicted_power - batch["target_power"]).abs() * real_mask.float()).sum()
        mae = mae / real_mask.sum()
        total_mae += mae.item()

        n_batches += 1

    avg_mse = total_loss / max(n_batches, 1)
    avg_mae = total_mae / max(n_batches, 1)
    return avg_mse, avg_mae


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

    # =========================================================================
    # Wandb init
    # =========================================================================
    use_wandb = WANDB_AVAILABLE and args.track

    if use_wandb:
        # Auto-generate a descriptive run name
        mode = "snap" if args.snapshot else f"hist{args.history_length}"
        run_name = f"{args.exp_name}_{mode}_e{args.embed_dim}_L{args.num_layers}_H{args.num_heads}"

        # Parse comma-separated tags
        tags = [t.strip() for t in args.wandb_tags.split(",")] if args.wandb_tags else None

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

    # Find data files
    files = sorted(glob(f"{args.data_dir}/layout_*.h5"))
    if not files:
        print(f"No layout files found in {args.data_dir}")
        return
    print(f"Found {len(files)} layout file(s)")

    # =========================================================================
    # Key: use features WITHOUT power for obs, power is the target
    # =========================================================================
    input_features = ["ws", "wd", "yaw"]

    scaling_limits = {"power": (0.0, args.power_max)}

    # --- Create dataset (not dataloader, so we can split) ---
    common_kwargs = dict(
        layout_files=files,
        max_turbines=None,  # auto
        features=input_features,
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
    n_val = int(len(dataset) * args.val_split)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(f"Train: {n_train} samples, Val: {n_val} samples")

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
    print(f"  obs_dim (input features × history): {obs_dim}")
    print(f"  max_turbines: {max_turbines}")
    print(f"  profiles: {'yes, ' + str(n_profile_dirs) + ' dirs' if has_profiles else 'no'}")
    print(f"  embed_dim: {args.embed_dim}")
    print(f"  num_heads: {args.num_heads}")
    print(f"  num_layers: {args.num_layers}")

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
    model = PowerPredictionModel(actor).to(device)
    # model = torch.compile(model)  # add this line   ### Did not work directly on sophia. We need to module load GCC/12.3.0 first I think

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {n_params:,}")

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
        }, allow_val_change=True)
        wandb.watch(model, log="gradients", log_freq=100)

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    scaler = torch.amp.GradScaler("cuda")  # Add scaler for mixed precision (if using CUDA)

    # --- Output dir ---
    os.makedirs(args.save_dir, exist_ok=True)

    # Determine if plots should be generated
    do_plots = use_wandb and args.plot_every > 0

    # =========================================================================
    # Training
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"Starting pretraining: {args.epochs} epochs")
    if do_plots:
        print(f"Diagnostic plots every {args.plot_every} epochs ({args.plot_n_samples} samples)")
    print(f"{'='*60}")

    best_val_mse = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_mse = train_one_epoch(model, train_loader, optimizer, device, epoch,
                                    log_wandb=use_wandb, scaler=scaler)
        val_mse, val_mae = evaluate(model, val_loader, device)
        scheduler.step()

        dt = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train_mse: {train_mse:.6f} | "
              f"val_mse: {val_mse:.6f} | "
              f"val_mae: {val_mae:.4f} | "
              f"lr: {lr_now:.2e} | "
              f"{dt:.1f}s")

        # Log epoch-level metrics to wandb
        if use_wandb:
            log_dict = {
                "epoch": epoch,
                "train/mse": train_mse,
                "val/mse": val_mse,
                "val/mae": val_mae,
                "lr": lr_now,
                "epoch_time_s": dt,
            }
            # Denormalized MAE in watts for interpretability
            log_dict["val/mae_watts"] = val_mae * args.power_max

            # --- Diagnostic plots (every N epochs + first + last) ---
            if do_plots and (epoch % args.plot_every == 0
                            or epoch == 1
                            or epoch == args.epochs):
                plot_images = generate_diagnostic_plots(
                    model, val_set, device, epoch, args,
                )
                log_dict.update(plot_images)

            wandb.log(log_dict)

        # Save best
        is_best = val_mse < best_val_mse
        if is_best:
            best_val_mse = val_mse
            torch.save({
                "epoch": epoch,
                "encoder_state_dict": get_encoder_state_dict(model.actor),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_mse": val_mse,
                "val_mae": val_mae,
                "args": vars(args),
                "obs_dim": obs_dim,
                "max_turbines": max_turbines,
                "n_profile_dirs": n_profile_dirs,
                "input_features": input_features,
            }, f"{args.save_dir}/best.pt")
            print(f"  → Saved best model (val_mse={val_mse:.6f})")

            if use_wandb:
                wandb.run.summary["best_val_mse"] = best_val_mse
                wandb.run.summary["best_val_mae"] = val_mae
                wandb.run.summary["best_epoch"] = epoch

        # Periodic save
        if epoch % args.save_every == 0:
            torch.save({
                "epoch": epoch,
                "encoder_state_dict": get_encoder_state_dict(model.actor),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_mse": val_mse,
                "args": vars(args),
            }, f"{args.save_dir}/epoch_{epoch:04d}.pt")

    # Final save
    torch.save({
        "epoch": args.epochs,
        "encoder_state_dict": get_encoder_state_dict(model.actor),
        "model_state_dict": model.state_dict(),
        "val_mse": val_mse,
        "val_mae": val_mae,
        "args": vars(args),
        "obs_dim": obs_dim,
        "max_turbines": max_turbines,
        "n_profile_dirs": n_profile_dirs,
        "input_features": input_features,
    }, f"{args.save_dir}/final.pt")

    print(f"\n{'='*60}")
    print(f"Pretraining complete!")
    print(f"  Best val MSE: {best_val_mse:.6f}")
    print(f"  Checkpoints saved to: {args.save_dir}/")
    print(f"{'='*60}")

    # Save best model as wandb artifact for easy retrieval
    if use_wandb:
        wandb.run.summary["final_val_mse"] = val_mse
        wandb.run.summary["final_val_mae"] = val_mae

        best_path = f"{args.save_dir}/best.pt"
        if os.path.exists(best_path):
            artifact = wandb.Artifact(
                name="pretrained-encoder",
                type="model",
                description=f"Best pretrained encoder (val_mse={best_val_mse:.6f})",
                metadata={
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