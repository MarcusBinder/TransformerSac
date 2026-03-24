"""
Training utility functions for Transformer-SAC wind farm training.

Includes GPU memory management, adaptive entropy computation,
environment helpers, and fine-tuning diagnostics.
"""

import gc
from typing import Optional, List

import numpy as np
import torch


def clear_gpu_memory():
    """Clear GPU memory - works on both NVIDIA and AMD."""
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        # Report
        device_name = torch.cuda.get_device_name(0)
        allocated = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {device_name}")
        print(f"Memory: {allocated:.2f}GB used / {total:.2f}GB total")


def compute_adaptive_target_entropy(
    attention_mask: torch.Tensor,
    action_dim_per_turbine: int = 1
) -> torch.Tensor:
    """
    Compute target entropy adapted to actual turbine count per sample.

    This fixes a bug where using max_turbines for all samples causes
    incorrect entropy targeting when training on variable-size farms.

    Args:
        attention_mask: (batch, max_turbines) where True = padding
        action_dim_per_turbine: Actions per turbine (typically 1 for yaw)

    Returns:
        target_entropy: (batch, 1) tensor of per-sample target entropies
    """
    # Count real turbines per sample
    n_real_turbines = (~attention_mask).sum(dim=1, keepdim=True).float()

    # Target entropy scales with turbine count
    # Convention: -1 per action dimension
    target_entropy = -action_dim_per_turbine * n_real_turbines

    return target_entropy


def get_env_current_layout(envs) -> List[str]:
    '''Get the current layout name for each environment.'''
    names_tuple = envs.env.get_attr('current_layout')
    names_list = [names_tuple[x].name for x in range(len(names_tuple))]
    return names_list


# =============================================================================
# FINE-TUNING DIAGNOSTICS
# =============================================================================

def log_optimizer_effective_lr(optimizer, name: str, nominal_lr: float):
    """Print effective learning rate statistics for an optimizer."""
    effective_lrs = []
    momentum_norms = []

    for group in optimizer.param_groups:
        for p in group['params']:
            if p in optimizer.state:
                state = optimizer.state[p]
                if 'exp_avg_sq' in state:
                    # Effective LR ≈ nominal_lr / sqrt(exp_avg_sq + eps)
                    second_moment = state['exp_avg_sq']
                    eff_lr = nominal_lr / (second_moment.sqrt().mean().item() + 1e-8)
                    effective_lrs.append(eff_lr)
                if 'exp_avg' in state:
                    momentum_norms.append(state['exp_avg'].norm().item())

    if effective_lrs:
        mean_eff_lr = np.mean(effective_lrs)
        ratio = mean_eff_lr / nominal_lr
        print(f"  {name}: nominal_lr={nominal_lr:.2e}, effective_lr={mean_eff_lr:.2e}, ratio={ratio:.4f}")
    else:
        print(f"  {name}: Fresh optimizer (no accumulated state)")


def compute_optimizer_diagnostics(optimizer, name: str, nominal_lr: float) -> dict:
    """Compute optimizer diagnostics for logging to wandb/tensorboard."""
    effective_lrs = []
    momentum_norms = []
    second_moments = []

    for group in optimizer.param_groups:
        for p in group['params']:
            if p in optimizer.state:
                state = optimizer.state[p]
                if 'exp_avg_sq' in state:
                    second_moment = state['exp_avg_sq'].mean().item()
                    second_moments.append(second_moment)
                    eff_lr = nominal_lr / (second_moment ** 0.5 + 1e-8)
                    effective_lrs.append(eff_lr)
                if 'exp_avg' in state:
                    momentum_norms.append(state['exp_avg'].norm().item())

    diagnostics = {}
    if effective_lrs:
        diagnostics[f"finetune/{name}_effective_lr"] = np.mean(effective_lrs)
        diagnostics[f"finetune/{name}_effective_lr_ratio"] = np.mean(effective_lrs) / nominal_lr
    if momentum_norms:
        diagnostics[f"finetune/{name}_momentum_norm"] = np.mean(momentum_norms)
    if second_moments:
        diagnostics[f"finetune/{name}_second_moment"] = np.mean(second_moments)

    return diagnostics


def log_finetune_diagnostics(
    writer,
    global_step: int,
    actor_optimizer,
    q_optimizer,
    policy_lr: float,
    q_lr: float,
    qf1_values: Optional[torch.Tensor] = None,
    qf2_values: Optional[torch.Tensor] = None,
    episode_returns: Optional[list] = None,
    alpha: Optional[float] = None,
    policy_entropy: Optional[float] = None,
):
    """Log fine-tuning specific diagnostics to tensorboard/wandb."""

    # Optimizer state diagnostics
    actor_diag = compute_optimizer_diagnostics(actor_optimizer, "actor", policy_lr)
    critic_diag = compute_optimizer_diagnostics(q_optimizer, "critic", q_lr)

    for key, value in {**actor_diag, **critic_diag}.items():
        writer.add_scalar(key, value, global_step)

    # Q-value diagnostics (detect overestimation)
    if qf1_values is not None and qf2_values is not None:
        q_mean = (qf1_values.mean().item() + qf2_values.mean().item()) / 2
        q_std = (qf1_values.std().item() + qf2_values.std().item()) / 2
        writer.add_scalar("finetune/q_mean", q_mean, global_step)
        writer.add_scalar("finetune/q_std", q_std, global_step)

        # Q-value vs actual returns (if we have episode data)
        if episode_returns and len(episode_returns) > 0:
            mean_return = np.mean(episode_returns)
            q_overestimation = q_mean - mean_return
            writer.add_scalar("finetune/q_overestimation", q_overestimation, global_step)
            writer.add_scalar("finetune/q_to_return_ratio", q_mean / (mean_return + 1e-8), global_step)

    # Entropy tracking
    if alpha is not None:
        writer.add_scalar("finetune/alpha", alpha, global_step)

    if policy_entropy is not None:
        writer.add_scalar("finetune/policy_entropy", policy_entropy, global_step)
