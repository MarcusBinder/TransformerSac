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
    ) -> torch.Tensor:
        """
        Replicate the actor's encoder path (forward up to transformer output).
        This is lines 958-1004 of the RL script, minus the action heads.
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
        h, _ = actor.transformer(h, attention_mask, attn_bias, need_weights=False)

        return h

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Args:
            batch: Dict with keys: obs, positions, attention_mask,
                   [receptivity, influence]
        Returns:
            predicted_power: (B, N) predicted normalized power per turbine
        """
        h = self._encode(
            obs=batch["obs"],
            positions=batch["positions"],
            attention_mask=batch["attention_mask"],
            receptivity=batch.get("receptivity"),
            influence=batch.get("influence"),
        )
        power = self.power_head(h).squeeze(-1)
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

def train_one_epoch(model, loader, optimizer, device, epoch, log_wandb=False, scaler=None):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        # Move to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        with torch.amp.autocast("cuda"):
            predicted_power = model(batch)
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
            predicted_power = model(batch)

        # predicted_power = model(batch)
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
        # num_workers=args.num_workers, 
        pin_memory=True, drop_last=True,
        persistent_workers=True,    # avoids respawning workers each epoch
        prefetch_factor=4,          # pre-loads more batches
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        # num_workers=args.num_workers, 
        pin_memory=True,
        persistent_workers=True,    # avoids respawning workers each epoch
        prefetch_factor=4,          # pre-loads more batches
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
            wandb.log(log_dict)

        # Save best
        is_best = val_mse < best_val_mse
        if is_best:
            best_val_mse = val_mse
            torch.save({
                "epoch": epoch,
                "encoder_state_dict":get_encoder_state_dict(model.actor),
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