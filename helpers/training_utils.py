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
    action_dim_per_turbine: int = 1,
    agg: str = "sum",
    scale: float = 1.0,
) -> torch.Tensor:
    """
    Compute the SAC target entropy, matched to how the actor aggregates log-prob.

    With agg="sum" (standard SAC) the actor's log_pi is summed over turbines, so the
    target scales with turbine count: -1 per action dimension -> -action_dim * N. This
    fixes a bug where using max_turbines for all samples mis-targets variable-size farms.

    With agg="mean" the actor's log_pi is the per-(turbine,action) MEAN, so the target is
    a size-invariant -1 per action dimension regardless of N. This keeps the entropy
    regularization from scaling with farm size (the large-farm "stay diffuse" pathology).

    Args:
        attention_mask: (batch, max_turbines) where True = padding
        action_dim_per_turbine: Actions per turbine (typically 1 for yaw)
        agg: "sum" or "mean" — must match TransformerActor.entropy_agg
        scale: --target_entropy_scale (optuna sweep); 1.0 = the standard target.
            < 1 asks for a sharper policy, > 1 keeps it more stochastic.

    Returns:
        target_entropy: (batch, 1) tensor of per-sample target entropies
    """
    batch = attention_mask.shape[0]
    if agg == "mean":
        # Size-invariant: -1 per action dimension.
        target_entropy = attention_mask.new_full(
            (batch, 1), -float(action_dim_per_turbine), dtype=torch.float32
        )
    else:
        # agg == "sum": scales with the actual turbine count per sample.
        n_real_turbines = (~attention_mask).sum(dim=1, keepdim=True).float()
        target_entropy = -action_dim_per_turbine * n_real_turbines

    if scale != 1.0:
        target_entropy = target_entropy * float(scale)
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


# ---------------------------------------------------------------------------
# Stage 12 (exp_12) learning-signal helpers. Pure functions, unit-tested in
# tests/test_stage12_flags.py; every default reproduces the Stage-8..11 trainer.
# ---------------------------------------------------------------------------

def buffer_done(terminated, truncated, bootstrap_truncated: bool) -> bool:
    """The ``done`` flag stored in the replay buffer for one transition.

    The TD target is ``r + (1 - done) * gamma * Q'(s')``. windgym episodes never
    terminate: every ``done`` of Stages 8-11 was the time-limit truncation
    (``max_episode_steps``), stored as terminal, so ~0.1 % of transitions got a
    target ~100x too small in states the critic cannot distinguish from their
    neighbours. With ``bootstrap_truncated`` only a true termination zeroes the
    bootstrap; a pure truncation keeps it (``next_obs`` is already the real
    ``final_obs``, substituted by the trainer before ``rb.add``).
    """
    if bootstrap_truncated:
        return bool(terminated)
    return bool(terminated) or bool(truncated)


def env_seed_base(seed: int, env_seed_stride: int) -> int:
    """Base of the per-env seed range: sub-env ``i`` is seeded ``base + i``.

    Stride 1 (default) is the Stage-8..11 layout ``seed + i``, where seeds s and
    s + 1 share 29 of their 30 env streams (DR draws, cycle phase, Mann seeds).
    Stride 1000 gives seeds 1..6 the disjoint ranges 1000.., 2000.., so "seed"
    varies the training conditions as well as the learner.
    """
    stride = int(env_seed_stride)
    if stride < 1:
        raise ValueError(f"--env_seed_stride must be >= 1, got {env_seed_stride}")
    return int(seed) * stride


def initial_log_alpha(alpha_init, autotune: bool) -> float:
    """Starting value of ``log_alpha`` under entropy autotuning.

    ``None`` (default) is the Stage-8..11 start ``log_alpha = 0`` (alpha 1.0,
    against per-step rewards of 0.02-0.1: the first 40-60k env steps were
    entropy-dominated, every Stage-11 seed ended near alpha 0.002). With
    ``--alpha_init 0.1`` the search starts at log(0.1). Only meaningful with
    ``--autotune``: a fixed alpha is ``--alpha``.
    """
    if alpha_init is None:
        return 0.0
    if not autotune:
        raise ValueError("--alpha_init only applies with autotune (use --alpha for a fixed coefficient)")
    if not float(alpha_init) > 0.0:
        raise ValueError(f"--alpha_init must be > 0, got {alpha_init}")
    return float(np.log(float(alpha_init)))


def tqc_n_keep(n_critics: int, n_quantiles: int, top_quantiles_to_drop: int, drop_per_net) -> int:
    """Number of target atoms kept after TQC truncation (computed once at startup).

    ``drop_per_net=None`` (default) drops ``top_quantiles_to_drop`` atoms IN
    TOTAL (Stage-8..11: 2 of 125, i.e. almost no pessimism); ``drop_per_net=d``
    drops ``d * n_critics`` atoms, the TQC paper's form (5 x 25 with d 2: 115).
    """
    total = int(n_critics) * int(n_quantiles)
    if drop_per_net is None:
        dropped = int(top_quantiles_to_drop)
        if not (0 <= dropped < total):
            raise ValueError(f"--tqc_top_quantiles_to_drop must be in [0, {total}), got {top_quantiles_to_drop}")
    else:
        d = int(drop_per_net)
        if not (0 <= d < int(n_quantiles)):
            raise ValueError(f"--tqc_drop_per_net must be in [0, {n_quantiles}), got {drop_per_net}")
        dropped = d * int(n_critics)
    return total - dropped


def tqc_td_target(kept: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor, gamma: float,
                  alpha, next_log_pi: torch.Tensor, atom_targets: bool) -> torch.Tensor:
    """TQC critic target from the ``(batch, n_keep)`` kept (sorted) next-state atoms.

    ``atom_targets=False``: the Stage-8..11 target, ``r + (1 - d) gamma (mean_kept
    - alpha log pi')`` of shape ``(batch, 1)`` -- every quantile head regressed
    against one scalar. ``True``: the paper's per-atom target of shape
    ``(batch, n_keep)``; ``quantile_huber_loss`` broadcasts over the target atoms.
    """
    if atom_targets:
        return rewards + (1 - dones) * gamma * (kept - alpha * next_log_pi)
    truncated_mean = kept.mean(dim=1, keepdim=True)
    return rewards + (1 - dones) * gamma * (truncated_mean - alpha * next_log_pi)


LR_SCHEDULES = ("none", "cosine", "linear")


def actor_update_allowed(global_step: int, args) -> bool:
    """False inside the fine-tune actor-freeze window
    ``[learning_starts, learning_starts + finetune_freeze_actor_steps)``:
    the critics keep training on the resumed policy's data while the actor
    (and alpha) stay put. 0 (the default) = always allowed."""
    freeze = int(getattr(args, "finetune_freeze_actor_steps", 0) or 0)
    if freeze <= 0:
        return True
    return int(global_step) >= int(args.learning_starts) + freeze


def lr_progress(global_step: int, learning_starts: int, total_timesteps: int) -> float:
    """Fraction of the update budget spent, clamped to [0, 1]: 0 at learning_starts,
    1 at total_timesteps (the trainer's two extra iterations past it clamp)."""
    span = max(int(total_timesteps) - int(learning_starts), 1)
    return float(min(max((int(global_step) - int(learning_starts)) / span, 0.0), 1.0))


def lr_multiplier(schedule: str, progress: float, final_frac: float) -> float:
    """Learning-rate multiplier at ``progress`` in [0, 1] (clamped).

    ``none`` = 1 (the Stage-8..11 constant lr); ``cosine`` decays from 1 to
    ``final_frac`` as 0.5 (1 + cos(pi p)); ``linear`` decays straight to it.
    Returns a python float (fused Adam takes a float ``lr`` per param group).
    """
    if schedule not in LR_SCHEDULES:
        raise ValueError(f"--lr_schedule must be one of {LR_SCHEDULES}, got {schedule!r}")
    if schedule == "none":
        return 1.0
    p = float(min(max(float(progress), 0.0), 1.0))
    f = float(final_frac)
    if schedule == "cosine":
        return float(f + (1.0 - f) * 0.5 * (1.0 + np.cos(np.pi * p)))
    return float(1.0 - (1.0 - f) * p)


def alpha_learning_rate(alpha_lr, q_lr: float) -> float:
    """Learning rate of the entropy-coefficient optimizer: ``--alpha_lr`` if given,
    else ``--q_lr`` (the Stage-8..11 coupling). The Stage-12 ``lrsched`` arm raises
    q_lr to 6e-4 and keeps alpha at 3e-4 through this flag, so the schedule is
    the only change (review finding, 2026-10-12)."""
    if alpha_lr is None:
        return float(q_lr)
    if not float(alpha_lr) > 0.0:
        raise ValueError(f"--alpha_lr must be > 0, got {alpha_lr}")
    return float(alpha_lr)
