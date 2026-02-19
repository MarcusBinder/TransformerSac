"""
Power Prediction Pretraining for Wind Farm Transformer

Pretrains a transformer encoder to predict per-turbine power output
from (ws, wd, yaw) + positions + profiles. The learned encoder weights
can then be transferred to the SAC actor/critic for RL fine-tuning.

This script uses a placeholder encoder to validate the full pipeline.
Once confirmed working, swap in the real TransformerActor backbone.

Usage:
    python pretrain_power.py --data-dir ./pretrain_data --epochs 50
    python pretrain_power.py --data-dir ./pretrain_data --snapshot  # no history
    python pretrain_power.py --data-dir ./pretrain_data --no-track  # disable wandb

Author: Marcus (DTU Wind Energy)
"""

import os
import time
from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from glob import glob
from pathlib import Path

import tyro

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not installed — logging disabled. Install with: pip install wandb")

from data_loader import create_pretrain_dataloader, WindFarmPretrainDataset, WindFarmSnapshotDataset


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
    num_workers: int = 4
    """DataLoader workers."""

    # === Output / Checkpointing ===
    save_dir: str = "./pretrain_checkpoints"
    """Directory for saving checkpoints."""
    save_every: int = 10
    """Save checkpoint every N epochs."""

    # === Device ===
    device: str = "auto"
    """Device: 'auto', 'cuda', 'cpu'."""


# =============================================================================
# PLACEHOLDER ENCODER (mirrors real TransformerActor backbone)
# =============================================================================

class PlaceholderTransformerEncoder(nn.Module):
    """
    Simplified transformer encoder for pipeline validation.

    Architecture mirrors the real TransformerActor:
        obs → obs_encoder → h
        positions → pos_encoder → added to h
        profiles → profile_encoder → added to h
        h → input_proj → transformer layers → output embeddings

    But with smaller defaults and simpler components.
    """

    def __init__(
        self,
        obs_dim: int,
        embed_dim: int = 64,
        pos_embed_dim: int = 16,
        num_heads: int = 4,
        num_layers: int = 2,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        n_profile_dirs: int = 0,  # 0 = no profiles
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_profiles = n_profile_dirs > 0

        # --- Obs encoder: raw features → embed_dim ---
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # --- Position encoder: (x, y) → embed_dim (added) ---
        self.pos_encoder = nn.Sequential(
            nn.Linear(2, pos_embed_dim),
            nn.GELU(),
            nn.Linear(pos_embed_dim, embed_dim),
        )

        # --- Profile encoders (optional) ---
        if self.use_profiles:
            # Simple 1D conv encoder for profiles
            self.recep_encoder = nn.Sequential(
                nn.Linear(n_profile_dirs, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            )
            self.influence_encoder = nn.Sequential(
                nn.Linear(n_profile_dirs, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            )
            self.profile_fusion = nn.Linear(2 * embed_dim, embed_dim)

        # --- Transformer ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embed_dim),
        )

    def forward(
        self,
        obs: torch.Tensor,             # (B, N, obs_dim)
        positions: torch.Tensor,        # (B, N, 2)
        attention_mask: torch.Tensor,   # (B, N) True=padding
        receptivity: torch.Tensor = None,   # (B, N, n_dirs)
        influence: torch.Tensor = None,     # (B, N, n_dirs)
    ) -> torch.Tensor:
        """
        Returns:
            h: (B, N, embed_dim) per-turbine embeddings
        """
        # Encode observations
        h = self.obs_encoder(obs)                    # (B, N, embed_dim)

        # Add positional encoding
        h = h + self.pos_encoder(positions)          # (B, N, embed_dim)

        # Add profile encoding
        if self.use_profiles and receptivity is not None and influence is not None:
            recep_embed = self.recep_encoder(receptivity)
            infl_embed = self.influence_encoder(influence)
            profile_embed = self.profile_fusion(
                torch.cat([recep_embed, infl_embed], dim=-1)
            )
            h = h + profile_embed

        # Transformer (attend across turbines)
        h = self.transformer(h, src_key_padding_mask=attention_mask)

        return h


# =============================================================================
# PRETRAINING MODEL (encoder + power prediction head)
# =============================================================================

class PowerPredictionModel(nn.Module):
    """
    Wraps encoder + prediction head for pretraining.

    After pretraining, transfer encoder weights to the RL agent.
    The power_head is discarded.
    """

    def __init__(self, encoder: PlaceholderTransformerEncoder):
        super().__init__()
        self.encoder = encoder
        self.power_head = nn.Sequential(
            nn.Linear(encoder.embed_dim, encoder.embed_dim // 2),
            nn.GELU(),
            nn.Linear(encoder.embed_dim // 2, 1),
        )

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Args:
            batch: Dict from dataloader with keys:
                obs, positions, attention_mask, [receptivity, influence]

        Returns:
            predicted_power: (B, N) predicted normalized power per turbine
        """
        h = self.encoder(
            obs=batch["obs"],
            positions=batch["positions"],
            attention_mask=batch["attention_mask"],
            receptivity=batch.get("receptivity"),
            influence=batch.get("influence"),
        )

        power = self.power_head(h).squeeze(-1)  # (B, N)
        return power


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
# TRAINING LOOP
# =============================================================================

def train_one_epoch(model, loader, optimizer, device, epoch, log_wandb=False):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        # Move to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        predicted_power = model(batch)

        loss = masked_mse_loss(
            predicted_power,
            batch["target_power"],
            batch["attention_mask"],
        )

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

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

        predicted_power = model(batch)
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
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
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

    # --- Build model ---
    encoder = PlaceholderTransformerEncoder(
        obs_dim=obs_dim,
        embed_dim=args.embed_dim,
        pos_embed_dim=args.pos_embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        n_profile_dirs=n_profile_dirs,
    )
    model = PowerPredictionModel(encoder).to(device)

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

    # --- Output dir ---
    os.makedirs(args.save_dir, exist_ok=True)

    # =========================================================================
    # Training
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"Starting pretraining: {args.epochs} epochs")
    print(f"{'='*60}")

    best_val_mse = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_mse = train_one_epoch(model, train_loader, optimizer, device, epoch,
                                    log_wandb=use_wandb)
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
            wandb.log(log_dict)

        # Save best
        is_best = val_mse < best_val_mse
        if is_best:
            best_val_mse = val_mse
            torch.save({
                "epoch": epoch,
                "encoder_state_dict": model.encoder.state_dict(),
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
                "encoder_state_dict": model.encoder.state_dict(),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_mse": val_mse,
                "args": vars(args),
            }, f"{args.save_dir}/epoch_{epoch:04d}.pt")

    # Final save
    torch.save({
        "epoch": args.epochs,
        "encoder_state_dict": model.encoder.state_dict(),
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