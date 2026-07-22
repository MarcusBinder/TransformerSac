"""
Transformer-based PPO for Wind Farm Control.

On-policy sibling of transformer_sac_windfarm.py: the same environment
construction and tanh-squashed Gaussian TransformerActor, with the off-policy
machinery (twin Q-critics, targets, alpha, replay buffer) replaced by a single
value network + PPO rollout/update.

Design notes vs the SAC trainer:
    - TransformerActor is UNCHANGED (checkpoint-compatible with eval_checkpoint,
      evaluate, interactive_eval, interp/ — they only read actor_state_dict+args).
    - V(s) reuses the tested TransformerCritic trunk fed zero action tokens.
    - Rollouts store the PRE-tanh sample x_t (never atanh(action), which
      saturates near the bounds) plus the already-prepared batch tensors
      (wind-relative positions, rotated profiles) so per-episode layout
      shuffles and per-step profile rotation are handled for free.
    - log-prob for the PPO ratio is the masked SUM over turbines/action dims
      (the joint log-prob; args.entropy_agg is deliberately ignored — "mean"
      would rescale the ratio and change clipping with farm size). The entropy
      BONUS uses the per-dim estimator -log_prob / n_real_dims so ent_coef is
      farm-size invariant.

Author: Marcus Binder Nilsen (DTU Wind Energy)
"""

import os
import random
import sys
import time
from typing import Optional, Tuple, List, Dict, Any, Union
from collections import deque
import json

from config import Args

# Set memory allocation config BEFORE importing torch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

# WindGym imports (adjust path as needed for your setup)
from WindGym import WindFarmEnv
from WindGym.wrappers import RecordEpisodeVals, PerTurbineObservationWrapper
from helpers.agent import WindFarmAgent

from helpers.training_utils import clear_gpu_memory
from helpers.helper_funcs import (
    get_env_wind_directions,
    get_env_raw_positions,
    get_env_attention_masks,
    EnhancedPerTurbineWrapper,
)
from helpers.layouts import get_layout_positions
from helpers.env_configs import make_env_config

# Receptivity profile computation
from helpers.receptivity_profiles import compute_layout_profiles

# Evaluation import
from helpers.eval_utils import PolicyEvaluator

# Repo root (parent of TransformerSac/): `del_surrogate` lives there and is
# not installed into the pixi env. The path insert + import happen INSIDE
# combined_wrapper so they also run in AsyncVectorEnv worker processes,
# where module-level state of __main__ may not be replayed.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from networks import (
    TransformerActor,
    TransformerCritic,
    create_profile_encoding,
)


# =============================================================================
# VALUE NETWORK
# =============================================================================

class TransformerValue(TransformerCritic):
    """V(s) via the tested TransformerCritic trunk fed zero action tokens.

    Constructor kwargs are identical to the SAC critic_kwargs, so the trunk
    (obs_action_encoder, positional/profile encoders, transformer, q_head) is
    byte-for-byte the architecture already validated on this env; the action
    columns of the first linear layer simply see zeros.
    """

    def __init__(self, *, action_dim_per_turbine: int = 1, **kwargs):
        super().__init__(action_dim_per_turbine=action_dim_per_turbine, **kwargs)
        self._action_dim = action_dim_per_turbine

    def forward(
        self,
        obs: torch.Tensor,
        positions: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        recep_profile: Optional[torch.Tensor] = None,
        influence_profile: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        zeros = obs.new_zeros(*obs.shape[:2], self._action_dim)
        return super().forward(obs, zeros, positions, key_padding_mask,
                               recep_profile, influence_profile)  # (batch, 1)


class ValueHead(nn.Module):
    """V(s) head over the actor's trunk embeddings (--ppo_share_trunk).

    Same MLP as TransformerCritic's q_head with DroQ off (Linear-ReLU-Linear),
    and the same two aggregations as --critic_agg: "pool" = masked-mean of
    turbine embeddings -> one farm value; "vdn" = per-turbine value -> masked
    sum.
    """

    def __init__(self, embed_dim: int, agg: str = "pool", detach_trunk: bool = False):
        super().__init__()
        assert agg in ("pool", "vdn"), f"Unknown value aggregation: {agg!r}"
        self.agg = agg
        self.detach_trunk = detach_trunk
        self.v_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(
        self,
        h: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """h: (batch, n_turb, embed_dim) trunk output. Returns (batch, 1)."""
        if self.detach_trunk:
            h = h.detach()

        if self.agg == "vdn":
            # Per-turbine value head, then masked-SUM over turbines. Padded
            # turbines' embeddings are non-zero, so zero their value first.
            v_per = self.v_head(h)  # (batch, n_turb, 1)
            if key_padding_mask is not None:
                valid = (~key_padding_mask).unsqueeze(-1).float()  # (batch, n_turb, 1)
                v_per = v_per * valid
            return v_per.sum(dim=1)  # (batch, 1)

        # "pool" (standard): masked-mean of turbine embeddings -> single farm-V.
        if key_padding_mask is not None:
            mask = ~key_padding_mask.unsqueeze(-1)  # (batch, n_turb, 1), True = real
            mask_f = mask.float()
            h = h * mask_f
            h_sum = h.sum(dim=1)  # (batch, embed_dim)
            n_real = mask_f.sum(dim=1).clamp(min=1)  # (batch, 1)
            h_pooled = h_sum / n_real
        else:
            h_pooled = h.mean(dim=1)  # (batch, embed_dim)
        return self.v_head(h_pooled)  # (batch, 1)


class SharedTrunkValue(nn.Module):
    """V(s) = ValueHead(actor.forward_trunk(...)) — the --ppo_share_trunk path.

    Drop-in for TransformerValue: same forward(obs, positions, mask, recep,
    infl) signature, so no vf() call site changes. The actor is held in a
    plain list, NOT registered as a submodule: state_dict()/parameters()
    contain ONLY the head (the trainer owns/checkpoints the actor itself),
    and .to()/.train() don't touch the actor — the trainer moves the actor
    first and never toggles train/eval modes.

    In the update loop the trunk runs twice per minibatch (once inside
    evaluate_actions, once here). With dropout == 0 (the RL default; dropout
    > 0 already trips the first-minibatch ratio assert) both forwards are
    identical, so the summed gradients equal a fused forward's.
    """

    def __init__(self, actor: TransformerActor, agg: str = "pool",
                 detach_trunk: bool = False):
        super().__init__()
        self._actor = [actor]  # list => NOT a registered submodule
        self.head = ValueHead(actor.fc_mean.in_features, agg=agg,
                              detach_trunk=detach_trunk)

    @property
    def actor(self) -> TransformerActor:
        return self._actor[0]

    def forward(
        self,
        obs: torch.Tensor,
        positions: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        recep_profile: Optional[torch.Tensor] = None,
        influence_profile: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        actor = self.actor
        if not actor.use_influence:  # same guard as get_action/evaluate_actions
            influence_profile = None
        h, _ = actor.forward_trunk(obs, positions, key_padding_mask,
                                   recep_profile, influence_profile,
                                   need_weights=False)
        return self.head(h, key_padding_mask)  # (batch, 1)


# =============================================================================
# DISTRIBUTION MATH (mirrors TransformerActor.get_action; networks.py untouched)
# =============================================================================

def _masked_logprob_sum(
    log_prob: torch.Tensor,
    key_padding_mask: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Zero padded-turbine log-prob elements, then SUM over turbines/action dims.

    Args:
        log_prob: (batch, n_turbines, action_dim) per-dim log-probs
        key_padding_mask: (batch, n_turbines), True = padding

    Returns:
        logp_sum: (batch,) joint log-prob (masked sum — used for the PPO ratio)
        n_real_dims: (batch,) number of real action dims (for the per-dim
            entropy estimator)
    """
    if key_padding_mask is not None:
        mask_f = (~key_padding_mask).unsqueeze(-1).float()  # (B, K, 1), 1 = real
        log_prob = log_prob * mask_f
        n_real_dims = mask_f.sum(dim=(-2, -1)) * log_prob.shape[-1]  # (B,)
    else:
        n_real_dims = log_prob.new_full(
            (log_prob.shape[0],), float(log_prob.shape[-2] * log_prob.shape[-1])
        )
    return log_prob.sum(dim=(-2, -1)), n_real_dims


def _tanh_gaussian_logprob(
    actor: TransformerActor,
    mean: torch.Tensor,
    log_std: torch.Tensor,
    x_t: torch.Tensor,
) -> torch.Tensor:
    """Per-dim log-prob of the tanh-squashed Gaussian at PRE-tanh point x_t."""
    normal = torch.distributions.Normal(mean, log_std.exp())
    y_t = torch.tanh(x_t)
    log_prob = normal.log_prob(x_t)
    # tanh + affine change-of-variables correction (same 1e-6 as get_action)
    log_prob = log_prob - torch.log(actor.action_scale * (1 - y_t.pow(2)) + 1e-6)
    return log_prob


def sample_action(
    actor: TransformerActor,
    batch,
    deterministic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rollout-time sampling (call under torch.no_grad()).

    Returns:
        env_action: (B, K, Ad) bounded action = tanh(x_t)*scale + bias
        x_t: (B, K, Ad) PRE-tanh sample — this is what the rollout buffer
            stores; never recover it via atanh(action) (saturates near ±1)
        log_prob: (B,) masked-SUM joint log-prob
    """
    influence = batch.influence if actor.use_influence else None
    mean, log_std, _ = actor(batch.obs, batch.positions, batch.mask,
                             batch.receptivity, influence)
    if deterministic:
        x_t = mean
    else:
        x_t = torch.distributions.Normal(mean, log_std.exp()).sample()
    env_action = torch.tanh(x_t) * actor.action_scale + actor.action_bias_val
    log_prob = _tanh_gaussian_logprob(actor, mean, log_std, x_t)
    log_prob, _ = _masked_logprob_sum(log_prob, batch.mask)
    return env_action, x_t, log_prob


def evaluate_actions(
    actor: TransformerActor,
    obs: torch.Tensor,
    positions: torch.Tensor,
    mask: torch.Tensor,
    x_t: torch.Tensor,
    recep_profile: Optional[torch.Tensor] = None,
    influence_profile: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Update-time re-evaluation (with grads) of stored pre-tanh actions.

    Returns:
        log_prob: (B,) masked-SUM joint log-prob (for the PPO ratio)
        entropy: (B,) per-dim entropy estimator -log_prob / n_real_dims
            (SB3-style; per-dim so ent_coef is farm-size invariant — used ONLY
            for the entropy bonus, never for the ratio)
    """
    if not actor.use_influence:
        influence_profile = None
    mean, log_std, _ = actor(obs, positions, mask, recep_profile, influence_profile)
    log_prob = _tanh_gaussian_logprob(actor, mean, log_std, x_t)
    log_prob, n_real_dims = _masked_logprob_sum(log_prob, mask)
    entropy = -log_prob / n_real_dims.clamp(min=1.0)
    return log_prob, entropy


# =============================================================================
# CHECKPOINTING
# =============================================================================

def save_ppo_checkpoint(actor, vf, optimizer, step, run_name, args) -> str:
    """Save a PPO checkpoint.

    actor_state_dict + args are identical in meaning to the SAC checkpoint's,
    so eval_checkpoint.py / evaluate.py / interactive_eval.py / interp/ load
    PPO checkpoints unchanged (they only read those two keys).
    """
    checkpoint_dir = f"runs/{run_name}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = f"{checkpoint_dir}/step_{step}.pt"
    ckpt = {
        "step": step,
        "actor_state_dict": actor.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),  # embeds ppo_share_trunk as the mode marker
    }
    if args.ppo_share_trunk:
        # Head-only (the shared trunk is already in actor_state_dict).
        ckpt["value_head_state_dict"] = vf.state_dict()
    else:
        ckpt["vf_state_dict"] = vf.state_dict()
    torch.save(ckpt, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")
    return checkpoint_path


# =============================================================================
# MAIN TRAINING SCRIPT
# =============================================================================

def main():
    """Main training function."""

    # Parse arguments
    args = tyro.cli(Args)
    args.algorithm = "ppo"  # checkpoints self-describe

    assert args.pretrain_checkpoint is None, \
        "--pretrain_checkpoint is not supported by the PPO trainer (use --resume_checkpoint to warm-start the actor)"

    if args.ppo_share_trunk and args.dropout > 0:
        print("WARNING: --ppo_share_trunk with --dropout > 0: train-mode noise "
              "makes rollout and update log-probs disagree, so the "
              "first-minibatch ratio assert will trip (same as the separate-net "
              "path today — dropout is unsupported in this trainer).")

    # Derived PPO sizes
    batch_size = args.num_envs * args.num_steps
    assert batch_size % args.num_minibatches == 0, \
        f"batch_size ({batch_size} = num_envs*num_steps) must be divisible by num_minibatches ({args.num_minibatches})"
    minibatch_size = batch_size // args.num_minibatches
    # Treat total_timesteps as a MINIMUM: round the iteration count UP (ceil, not
    # floor) so we always run at least the requested budget, overshooting by < one
    # batch. num_iterations drives both the LR anneal and the training loop, so no
    # other change is needed (effective >= requested, so any `< total_timesteps`
    # check still holds).
    num_iterations = -(-args.total_timesteps // batch_size)  # ceil division, no math import
    effective_timesteps = num_iterations * batch_size
    if effective_timesteps != args.total_timesteps:
        print(f"[ppo] total_timesteps {args.total_timesteps} is not a multiple of "
              f"batch_size {batch_size}; rounding UP to {effective_timesteps} "
              f"({num_iterations} iterations).")

    # Parse layouts
    layout_names = [l.strip() for l in args.layouts.split(",")]
    dr_enabled = args.dr_n_hi is not None
    is_multi_layout = len(layout_names) > 1 or dr_enabled

    # Parse evaluation layouts
    if args.eval_layouts.strip():
        eval_layout_names = [l.strip() for l in args.eval_layouts.split(",")]
    else:
        eval_layout_names = layout_names  # Use training layouts for evaluation

    print(f"Training layouts: {layout_names}")
    print(f"Evaluation layouts: {eval_layout_names}")

    # Create run name
    run_name = f"{args.exp_name}"

    print("=" * 60)
    print(f"Transformer PPO for Wind Farm Control")
    print("=" * 60)
    if is_multi_layout:
        print(f"Mode: Multi-layout training with layouts: {layout_names}")
    else:
        print(f"Mode: Single-layout training: {layout_names[0]}")
    print(f"Run name: {run_name}")
    print(f"Rollout: {args.num_steps} steps x {args.num_envs} envs = {batch_size} "
          f"transitions/iter, {num_iterations} iterations")
    print("=" * 60)

    # Create directories
    os.makedirs(f"runs/{run_name}", exist_ok=True)
    os.makedirs(f"runs/{run_name}/checkpoints", exist_ok=True)

    clear_gpu_memory()

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")

    # =========================================================================
    # ENVIRONMENT SETUP (copied from transformer_sac_windfarm.py)
    # =========================================================================

    from helpers.multi_layout_env import MultiLayoutEnv, LayoutConfig

    # Wind turbine. Derate-enabled configs (e.g. power_max_derate) need a
    # turbine whose powerCtFunction accepts a `derate` input; plain turbine
    # classes fail WindFarmEnv's check_turbine_supports_derating, so dispatch
    # on --turbtype to the derate-capable surrogate turbines (IEA34 default;
    # DTU10MW for reproducing old checkpoints).
    if make_env_config(args.config).get("derate_action", False):
        from helpers.derating_turbine import make_derating_turbine
        wind_turbine = make_derating_turbine(
            args.turbtype, iea34_variant=args.iea34_variant
        )
    else:
        if args.turbtype == "DTU10MW":
            from py_wake.examples.data.dtu10mw import DTU10MW as WT
        elif args.turbtype == "V80":
            from py_wake.examples.data.hornsrev1 import V80 as WT
        else:
            raise ValueError(f"Unknown turbine type: {args.turbtype}")
        wind_turbine = WT()

    # Create layout configurations
    print("Setting up layouts...")

    def build_layout_config(name, x_pos, y_pos, verbose=True):
        """Build a LayoutConfig and attach receptivity/influence profiles (if enabled)."""
        layout = LayoutConfig(name=name, x_pos=x_pos, y_pos=y_pos)
        if args.profile_encoding_type is not None:
            if args.profile_source.lower() == "geometric":
                from helpers.geometric_profiles import compute_layout_profiles_vectorized

                D = wind_turbine.diameter()

                if verbose:
                    print(f"Computing GEOMETRIC profiles for layout: {name}")
                receptivity_profiles, influence_profiles = compute_layout_profiles_vectorized(
                    x_pos, y_pos,
                    rotor_diameter=D,
                    k_wake=0.04,
                    n_directions=args.n_profile_directions,
                    sigma_smooth=args.profile_sigma_smooth,
                    scale_factor=15.0,
                    mode=args.profile_geom_mode,
                )
            elif args.profile_source.lower() == "pywake":
                if verbose:
                    print(f"Computing PyWake profiles for layout: {name}")
                receptivity_profiles, influence_profiles = compute_layout_profiles(
                    x_pos, y_pos, wind_turbine,
                    n_directions=args.n_profile_directions,
                )
            else:
                raise ValueError(
                    f"Unknown profile_source: {args.profile_source}. "
                    f"Use 'pywake' or 'geometric'."
                )

            layout.receptivity_profiles = receptivity_profiles  # (n_turbines, n_directions)
            layout.influence_profiles = influence_profiles      # (n_turbines, n_directions)
        return layout

    layouts = []
    if dr_enabled:
        # Domain randomization (v8): training layouts are a large procedurally
        # generated pool instead of the fixed named set.
        from helpers.layout_gen import generate_layout_pool
        print(f"Domain randomization: generating {args.dr_pool_size} layouts "
              f"with n in [{args.dr_n_lo}, {args.dr_n_hi}] (seed={args.seed}, "
              f"generator={args.dr_generator})...")
        pool = generate_layout_pool(
            pool_size=args.dr_pool_size,
            n_lo=args.dr_n_lo,
            n_hi=args.dr_n_hi,
            D=wind_turbine.diameter(),
            seed=args.seed,
            min_dist_D=args.dr_min_dist_D,
            screen_headroom=args.dr_screen_headroom,
            min_involved_frac=args.dr_min_involved_frac,
            generator=args.dr_generator,
        )
        for name, x_pos, y_pos in pool:
            layouts.append(build_layout_config(name, x_pos, y_pos, verbose=False))
        layout_names = [l.name for l in layouts]
        print(f"  generated {len(layouts)} layouts; "
              f"turbine counts {min(l.n_turbines for l in layouts)}..{max(l.n_turbines for l in layouts)}")
    else:
        for name in layout_names:
            x_pos, y_pos = get_layout_positions(name, wind_turbine)
            layouts.append(build_layout_config(name, x_pos, y_pos))

    use_profiles = args.profile_encoding_type is not None

    # Environment configuration
    print(f"using the config: {args.config}")
    config = make_env_config(args.config)

    # Override ActionMethod from args
    config["ActionMethod"] = args.action_type
    print(f"ActionMethod set to: {config['ActionMethod']}")

    # Optional derate slew limit (fraction per sim substep, like yaw_step_sim).
    if args.derate_step_sim is not None:
        config["derate_step_sim"] = args.derate_step_sim
        print(f"derate_step_sim set to: {config['derate_step_sim']}")

    mes_prefixes = {
        "ws_mes": "ws",
        "wd_mes": "wd",
        "yaw_mes": "yaw",
        "power_mes": "power",
        "derate_mes": "derate",
    }

    for mes_type, prefix in mes_prefixes.items():
        if mes_type not in config:
            continue  # e.g. derate_mes is absent outside the derate presets
        config[mes_type][f"{prefix}_history_N"] = args.history_length
        config[mes_type][f"{prefix}_history_length"] = args.history_length

    base_env_kwargs = {
        "turbine": wind_turbine,
        "n_passthrough": args.max_eps,
        "TurbBox": "./boxes/",  # Adjust path as needed
        "config": config,
        "turbtype": args.TI_type,
        "backend": args.backend,
        "dt_sim": args.dt_sim,
        "dt_env": args.dt_env,
        "yaw_step_sim": args.yaw_step,
    }

    # DEL-constrained reward: see transformer_sac_windfarm.py for the full
    # rationale (Baseline_comp requirement, --del_log info-only mode, Global
    # BaseController pinning the greedy reference).
    _del_active = args.del_penalty_scale > 0 or args.del_log
    assert not args.del_limit_random or _del_active, (
        "--del_limit_random conditions the policy on the DEL limit, which "
        "only exists when the DEL wrapper is attached: also pass "
        "--del_penalty_scale > 0 (or --del_log)."
    )
    if _del_active:
        base_env_kwargs["Baseline_comp"] = True
        config["BaseController"] = "Global"

    def env_factory(x_pos: np.ndarray, y_pos: np.ndarray) -> gym.Env:
        """Create a base WindFarmEnv with given positions."""
        env = WindFarmEnv(x_pos=x_pos,
                          y_pos=y_pos,
                          reset_init=False,  # Defer reset to training loop
                          **base_env_kwargs)
        env.action_space.seed(args.seed)
        return env

    def combined_wrapper(env: gym.Env) -> gym.Env:
        """
        Combined wrapper that:
        1. Applies PerTurbineObservationWrapper (reshapes obs to per-turbine)
        2. Optionally applies EnhancedPerTurbineWrapper (converts WD to deviation)
        3. Optionally applies DELRewardWrapper (baseline-relative DEL hinge penalty)
        4. Optionally applies TransformReward (reward_scale)
        """
        env = PerTurbineObservationWrapper(env)
        if args.use_wd_deviation:
            env = EnhancedPerTurbineWrapper(env, wd_scale_range=args.wd_scale_range)
        # DEL hinge penalty BEFORE TransformReward (see the SAC trainer for the
        # ordering rationale — penalty is in pre-reward_scale units).
        if args.del_penalty_scale > 0 or args.del_log:
            if _REPO_ROOT not in sys.path:
                sys.path.insert(0, _REPO_ROOT)
            from del_surrogate import DELRewardWrapper

            # DEL surrogate set follows the turbine unless overridden.
            del_turbine = args.del_artifact_set or {
                "IEA34": "iea34", "DTU10MW": "dtu10mw",
            }.get(args.turbtype)
            if del_turbine is None:
                raise ValueError(
                    f"No DEL surrogate set for turbtype {args.turbtype!r}; "
                    "pass --del_artifact_set explicitly."
                )
            del_channels = [
                c.strip() for c in args.del_channels.split(",") if c.strip()
            ]
            env = DELRewardWrapper(
                env,
                turbine=del_turbine,
                channels=del_channels,
                reward_channels=del_channels,
                penalty_scale=args.del_penalty_scale,
                allowed_increase=args.del_allowed_increase,
                limit_range=(
                    (args.del_limit_lo, args.del_limit_hi)
                    if args.del_limit_random else None
                ),
                limit_obs_ref=args.del_limit_obs_ref,
                ti_window=args.del_ti_window,
                n_r=3,
                n_theta=12,
            )
        if args.reward_scale != 1.0:
            _scale = float(args.reward_scale)
            env = gym.wrappers.TransformReward(env, lambda r: r * _scale)
        return env

    def make_env_fn(seed, warmup_steps=None):
        """Factory function for vectorized environments."""
        def _init():
            env = MultiLayoutEnv(
                layouts=layouts,
                env_factory=env_factory,
                per_turbine_wrapper=combined_wrapper,
                seed=seed,
                shuffle=args.shuffle_turbs,
                max_turbines=args.max_turbines,
                max_episode_steps=args.max_episode_steps,
                warmup_episode_steps=warmup_steps,
            )
            return env
        return _init

    # Compute per-env one-time warm-up episode lengths (staggered resets).
    warmup_lengths = [None] * args.num_envs
    if args.stagger_warmup:
        assert args.max_episode_steps is not None, \
            "--stagger_warmup requires --max_episode_steps to be set"
        assert args.warmup_min_episode_steps is not None, \
            "--stagger_warmup requires --warmup_min_episode_steps"
        assert args.warmup_min_episode_steps <= args.max_episode_steps, \
            "--warmup_min_episode_steps must be <= --max_episode_steps"

        num_groups = -(-args.num_envs // args.warmup_group_size)  # integer ceil
        if num_groups == 1:
            group_lengths = [args.max_episode_steps]
            print("NOTE: stagger_warmup with a single group is a no-op "
                  "(all envs use the standard episode length).")
        else:
            lo = max(1, args.warmup_min_episode_steps)
            group_lengths = [int(round(v)) for v in
                             np.linspace(lo, args.max_episode_steps, num_groups)]
        warmup_lengths = [
            group_lengths[min(i // args.warmup_group_size, len(group_lengths) - 1)]
            for i in range(args.num_envs)
        ]
        print(f"Staggered warm-up lengths per env: {warmup_lengths}")

    # Create vectorized environments
    print(f"Creating {args.num_envs} parallel environment(s)...")
    envs = gym.vector.AsyncVectorEnv(
        [make_env_fn(args.seed + i, warmup_lengths[i]) for i in range(args.num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    envs = RecordEpisodeVals(envs)

    n_turbines_max = envs.env.get_attr('max_turbines')[0]
    obs_dim_per_turbine = envs.single_observation_space.shape[-1]
    # 1 for yaw-only, 2 for yaw+derate (block action [yaw..., derate...]);
    # MultiLayoutEnv derives it from the wrapped env's action space.
    action_dim_per_turbine = envs.env.get_attr('action_dim_per_turbine')[0]
    rotor_diameter = envs.env.get_attr('rotor_diameter')[0]

    print(f"Max turbines: {n_turbines_max}")
    print(f"Obs dim per turbine: {obs_dim_per_turbine}")
    print(f"Action dim per turbine: {action_dim_per_turbine}")
    print(f"Rotor diameter: {rotor_diameter:.1f} m")

    # Create policy evaluator
    evaluator = PolicyEvaluator(
        agent=None,  # Will be set after actor is created
        eval_layouts=eval_layout_names,
        env_factory=env_factory,
        combined_wrapper=combined_wrapper,
        num_envs=args.num_envs,
        num_eval_steps=args.num_eval_steps,
        num_eval_episodes=args.num_eval_episodes,
        device=device,
        rotor_diameter=rotor_diameter,
        wind_turbine=wind_turbine,
        seed=args.eval_seed,
        max_turbines=n_turbines_max,
        deterministic=args.eval_deterministic,
        use_profiles=use_profiles,
        n_profile_directions=args.n_profile_directions,
        profile_source=args.profile_source,
        profile_sigma_smooth=args.profile_sigma_smooth,
        profile_geom_mode=args.profile_geom_mode,
    )

    # Action scaling
    action_high = envs.single_action_space.high[0]
    action_low = envs.single_action_space.low[0]
    action_scale = (action_high - action_low) / 2.0
    action_bias = (action_high + action_low) / 2.0

    # =========================================================================
    # TRACKING SETUP
    # =========================================================================

    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args) | {
                "debug/n_layouts": len(layout_names),
                "debug/layout_names": (f"dr_pool[{args.dr_n_lo}-{args.dr_n_hi}]x{len(layout_names)}"
                                       if dr_enabled else layout_names),
                "debug/is_multi_layout": is_multi_layout,
                "debug/max_turbines": n_turbines_max,
                "ppo/batch_size": batch_size,
                "ppo/minibatch_size": minibatch_size,
                "ppo/num_iterations": num_iterations,
            },
            name=run_name,
            group=args.exp_group,
            monitor_gym=True,
            save_code=True,
        )

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n" + "\n".join([f"|{k}|{v}|" for k, v in vars(args).items()])
    )

    # =========================================================================
    # NETWORK SETUP (kwargs built exactly like the SAC trainer)
    # =========================================================================

    print("\nCreating networks...")
    print(f"Positional encoding type: {args.pos_encoding_type}")

    # Create SHARED profile encoders (if using profiles)
    if args.profile_encoding_type is not None and args.share_profile_encoder:
        encoder_kwargs = json.loads(args.profile_encoder_kwargs)
        print(f"Creating shared profile encoders: {args.profile_encoding_type}")
        shared_recep_encoder, shared_influence_encoder = create_profile_encoding(
            profile_type=args.profile_encoding_type,
            embed_dim=args.embed_dim,
            hidden_channels=args.profile_encoder_hidden,
            **encoder_kwargs,
        )
        shared_recep_encoder = shared_recep_encoder.to(device)
        shared_influence_encoder = shared_influence_encoder.to(device)

        recep_params = sum(p.numel() for p in shared_recep_encoder.parameters())
        influence_params = sum(p.numel() for p in shared_influence_encoder.parameters())
        print(f"Shared receptivity encoder parameters: {recep_params:,}")
        print(f"Shared influence encoder parameters: {influence_params:,}")
    else:
        shared_recep_encoder = None
        shared_influence_encoder = None

    # Common profile args (to avoid repetition)
    common_kwargs = {
        # Architecture
        "obs_dim_per_turbine": obs_dim_per_turbine,
        "action_dim_per_turbine": action_dim_per_turbine,
        "embed_dim": args.embed_dim,
        "pos_embed_dim": args.pos_embed_dim,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "mlp_ratio": args.mlp_ratio,
        "dropout": args.dropout,
        # Positional encoding
        "pos_encoding_type": args.pos_encoding_type,
        "rel_pos_hidden_dim": args.rel_pos_hidden_dim,
        "rel_pos_per_head": args.rel_pos_per_head,
        "pos_embedding_mode": args.pos_embedding_mode,
        # PyWake profiles
        "profile_encoding": args.profile_encoding_type,
        "profile_encoder_hidden": args.profile_encoder_hidden,
        "n_profile_directions": args.n_profile_directions,
        "profile_fusion_type": args.profile_fusion_type,
        "profile_embed_mode": args.profile_embed_mode,
        # SHARED encoders
        "shared_recep_encoder": shared_recep_encoder,
        "shared_influence_encoder": shared_influence_encoder,
        "args": args,  # Pass full args for any additional config needs
    }

    # Actor has additional action scaling params (UNCHANGED from SAC — the
    # checkpointed actor stays loadable by all eval/interp tooling)
    actor = TransformerActor(
        action_scale=action_scale,
        action_bias=action_bias,
        **common_kwargs,
    ).to(device)

    agent = WindFarmAgent(
        actor=actor,
        device=device,
        rotor_diameter=rotor_diameter,
        use_wind_relative=args.use_wind_relative_pos,
        use_profiles=use_profiles,
        rotate_profiles=args.rotate_profiles,
    )

    # Update evaluator with actor reference
    evaluator.agent = agent

    # Value network: shared trunk (head over actor.forward_trunk) or the
    # default separate TransformerValue (critic kwargs identical to SAC's).
    if args.ppo_share_trunk:
        vf = SharedTrunkValue(actor, agg=args.critic_agg,
                              detach_trunk=args.ppo_value_detach_trunk).to(device)
    else:
        critic_kwargs = {**common_kwargs}
        vf = TransformerValue(**critic_kwargs).to(device)

    # Single optimizer over actor + value params, deduped by id() — with
    # share_profile_encoder the shared encoders appear in BOTH modules'
    # .parameters() and must enter Adam exactly once. eps=1e-5 (PPO standard).
    seen_ids = set()
    trainable_params: List[torch.nn.Parameter] = []
    for p in list(actor.parameters()) + list(vf.parameters()):
        if id(p) not in seen_ids:
            seen_ids.add(id(p))
            trainable_params.append(p)
    optimizer = optim.Adam(trainable_params, lr=args.policy_lr, eps=1e-5,
                           fused=device.type == "cuda")

    actor_params = sum(p.numel() for p in actor.parameters())
    vf_params = sum(p.numel() for p in vf.parameters())
    if args.ppo_share_trunk:
        print(f"Actor parameters (shared trunk): {actor_params:,}")
        print(f"Value HEAD parameters (trunk not duplicated): {vf_params:,}")
        print("Value gradients into shared trunk: "
              + ("DETACHED (head-only)" if args.ppo_value_detach_trunk
                 else f"flowing (weighted by vf_coef={args.vf_coef})"))
    else:
        print(f"Actor parameters: {actor_params:,}")
        print(f"Value parameters: {vf_params:,}")
    print(f"Algorithm: PPO")

    # `trainable_params` (above) is already deduped for the optimizer, so this is
    # the honest count even with --ppo_share_trunk (no double-counted trunk) — the
    # right x-axis for the model-size scaling curve.
    total_params = sum(p.numel() for p in trainable_params)
    # Mirror the counts into TensorBoard at step 0 so the OFFLINE scaling analysis
    # (scaling_curve.py) can read model/total_params straight from the events.*
    # files: wandb.run.summary below is NOT synced back into TB by sync_tensorboard.
    writer.add_scalar("model/actor_params", actor_params, 0)
    writer.add_scalar("model/vf_params", vf_params, 0)
    writer.add_scalar("model/total_params", total_params, 0)

    if args.track:
        import wandb  # already imported + cached in the init block above
        # Post-hoc summary columns (wandb.init ran before the model was built), so
        # these are queryable per-run for the scaling x-axis; works offline too.
        wandb.run.summary["model/actor_params"] = actor_params
        wandb.run.summary["model/vf_params"] = vf_params
        wandb.run.summary["model/total_params"] = total_params
        # The rounded-up honest budget (Deliverable 0); queryable per run because
        # different batch_size settings round to different totals.
        wandb.run.summary["ppo/effective_timesteps"] = effective_timesteps

    # =========================================================================
    # LOAD CHECKPOINT (for fine-tuning or resuming)
    # =========================================================================

    if args.resume_checkpoint is not None:
        print(f"\n{'='*60}")
        print(f"LOADING CHECKPOINT FOR FINE-TUNING")
        print(f"{'='*60}")
        print(f"Checkpoint path: {args.resume_checkpoint}")

        if not os.path.exists(args.resume_checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {args.resume_checkpoint}")

        checkpoint = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)

        actor.load_state_dict(checkpoint["actor_state_dict"])
        print(f"✓ Loaded actor weights from step {checkpoint['step']}")

        # Value-net resume matrix: the checkpoint's ppo_share_trunk flag (from
        # its saved args) vs this run's. On mismatch the optimizer param list
        # doesn't line up with the checkpointed one either, so force it fresh
        # below (explicit gate — don't rely on load_state_dict raising).
        ckpt_shared = checkpoint.get("args", {}).get("ppo_share_trunk", False)
        _mode_mismatch = False
        if args.ppo_share_trunk:
            if "value_head_state_dict" in checkpoint:
                vf.load_state_dict(checkpoint["value_head_state_dict"])
                print(f"✓ Loaded shared-trunk value head weights")
            elif "vf_state_dict" in checkpoint:
                _mode_mismatch = True
                print(f"  Separate value net in checkpoint — shared value head starts fresh")
            else:
                print(f"  No value weights in checkpoint (SAC checkpoint?) — shared value head starts fresh")
        else:
            if "vf_state_dict" in checkpoint:
                vf.load_state_dict(checkpoint["vf_state_dict"])
                print(f"✓ Loaded value network weights")
            elif ckpt_shared:
                _mode_mismatch = True
                print(f"  Shared-trunk checkpoint (head-only value weights) — separate value net starts fresh")
            else:
                print(f"  No vf_state_dict in checkpoint (SAC checkpoint?) — value net starts fresh")

        _reset_opt = args.finetune_reset_actor_optimizer or args.finetune_reset_critic_optimizer
        if _mode_mismatch:
            print(f"✓ Optimizer starts fresh (shared-trunk mode differs from checkpoint)")
        elif "optimizer_state_dict" in checkpoint and not _reset_opt:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print(f"✓ Loaded optimizer state")
        else:
            print(f"✓ Optimizer starts fresh")

        if "args" in checkpoint:
            ckpt_args = checkpoint["args"]
            print(f"\nOriginal training config:")
            print(f"  - Algorithm: {ckpt_args.get('algorithm', 'unknown')}")
            print(f"  - Layouts: {ckpt_args.get('layouts', 'unknown')}")
            print(f"  - Total timesteps: {ckpt_args.get('total_timesteps', 'unknown')}")
            print(f"  - Pos encoding: {ckpt_args.get('pos_encoding_type', 'unknown')}")
        print(f"{'='*60}\n")

    # =========================================================================
    # ROLLOUT STORAGE (preallocated on device)
    # =========================================================================

    T, N, K = args.num_steps, args.num_envs, n_turbines_max
    Od, Ad = obs_dim_per_turbine, action_dim_per_turbine

    # Stores the PREPARED tensors from BatchPreparer.from_envs (wind-relative
    # positions, rotated profiles) — this handles MultiLayoutEnv per-episode
    # layout shuffles and per-step profile rotation for free. Update-time
    # forwards must NOT rotate/transform again. Never cache env geometry
    # across steps.
    obs_buf = torch.zeros((T, N, K, Od), device=device)
    pos_buf = torch.zeros((T, N, K, 2), device=device)
    mask_buf = torch.zeros((T, N, K), dtype=torch.bool, device=device)
    if use_profiles:
        recep_buf = torch.zeros((T, N, K, args.n_profile_directions), device=device)
        infl_buf = torch.zeros((T, N, K, args.n_profile_directions), device=device)
    else:
        recep_buf = infl_buf = None
    xt_buf = torch.zeros((T, N, K, Ad), device=device)  # PRE-tanh samples
    logp_buf = torch.zeros((T, N), device=device)
    val_buf = torch.zeros((T, N), device=device)
    rew_buf = torch.zeros((T, N), device=device)
    done_buf = torch.zeros((T, N), device=device)

    # =========================================================================
    # TRAINING LOOP
    # =========================================================================

    print(f"\nStarting training for {args.total_timesteps} timesteps...")
    print(f"{num_iterations} iterations x {batch_size} transitions "
          f"({args.ppo_epochs} epochs x {args.num_minibatches} minibatches of {minibatch_size})")
    print("=" * 60)

    save_ppo_checkpoint(actor, vf, optimizer, 0, run_name, args)

    next_eval_step = args.eval_interval
    next_save_step = args.save_interval

    # Initial evaluation
    if args.eval_initial:
        print("\nRunning initial evaluation before training...")
        eval_metrics = evaluator.evaluate()
        eval_dict = eval_metrics.to_dict()

        for name, value in eval_dict.items():
            writer.add_scalar(name, value, 0)

        print(f"Initial eval - Mean reward: {eval_metrics.mean_reward:.4f}, "
              f"Power ratio: {eval_metrics.power_ratio:.4f}")

    start_time = time.time()
    global_step = 0

    # Reset environments
    obs, infos = envs.reset(seed=args.seed)

    # Tracking
    step_reward_window = deque(maxlen=1000)
    # DEL-penalty diagnostics (filled only when the DELRewardWrapper is
    # attached: --del_penalty_scale > 0 or --del_log). See the SAC trainer.
    del_penalty_window = deque(maxlen=1000)
    del_ratio_window = deque(maxlen=1000)
    del_agent_max_window = deque(maxlen=1000)
    del_base_max_window = deque(maxlen=1000)
    reward_unpen_window = deque(maxlen=1000)
    del_ood_window = deque(maxlen=1000)
    del_limit_window = deque(maxlen=1000)
    del_margin_window = deque(maxlen=1000)
    # Multi-channel penalty: which reward channel realized the binding (max)
    # ratio each step -> charts/del_binding_frac/<channel>. Only logged when
    # more than one channel is configured (single-channel: trivially 1.0).
    # Canonicalized ("wtow_H0FAMnt" -> "H0FAMnt") to match the names the
    # wrapper reports in info["del_binding_channel"].
    del_binding_window = deque(maxlen=1000)
    _del_reward_channels = [
        c.strip() for c in args.del_channels.split(",") if c.strip()
    ]
    if _del_active:
        if _REPO_ROOT not in sys.path:
            sys.path.insert(0, _REPO_ROOT)
        from del_surrogate import get_set as _del_get_set
        _del_tset = _del_get_set(args.del_artifact_set or {
            "IEA34": "iea34", "DTU10MW": "dtu10mw",
        }[args.turbtype])
        _del_reward_channels = [
            _del_tset.canonical_channel(c) for c in _del_reward_channels
        ]

    # One-time sanity check: on the very first minibatch (before any gradient
    # step) the recomputed log-prob must equal the rollout one, i.e. ratio ≈ 1.
    # Proves sample_action/evaluate_actions math matches.
    ratio_check_done = False

    for iteration in range(1, num_iterations + 1):
        # Linear LR anneal
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / num_iterations
            lrnow = frac * args.policy_lr
            for pg in optimizer.param_groups:
                pg["lr"] = lrnow

        # =====================================================================
        # COLLECT ROLLOUT
        # =====================================================================
        for t in range(args.num_steps):
            global_step += args.num_envs

            # Fetch env state fresh each step (never cache across steps — the
            # SAME_STEP autoreset can swap layout/permutation mid-rollout)
            wind_dirs = get_env_wind_directions(envs)
            raw_positions = get_env_raw_positions(envs)
            current_masks = get_env_attention_masks(envs)

            batch = agent.batch_preparer.from_envs(
                envs, obs,
                wind_dirs=wind_dirs, raw_positions=raw_positions, masks=current_masks,
            )

            with torch.no_grad():
                env_action, x_t, log_prob = sample_action(actor, batch)
                value = vf(
                    batch.obs, batch.positions, batch.mask,
                    batch.receptivity,
                    batch.influence if actor.use_influence else None,
                ).squeeze(-1)

            obs_buf[t] = batch.obs
            pos_buf[t] = batch.positions
            mask_buf[t] = batch.mask
            if use_profiles:
                recep_buf[t] = batch.receptivity
                infl_buf[t] = batch.influence
            xt_buf[t] = x_t
            logp_buf[t] = log_prob
            val_buf[t] = value

            # Same layout agent.act() produces: squeeze the action_dim axis
            actions_np = env_action.squeeze(-1).cpu().numpy()  # (N, K)

            next_obs, rewards, terminations, truncations, infos = envs.step(actions_np)

            rew_buf[t] = torch.as_tensor(np.asarray(rewards, dtype=np.float32), device=device)
            done_buf[t] = torch.as_tensor(
                np.asarray(terminations | truncations, dtype=np.float32), device=device)

            # Track rewards
            step_reward_window.extend(np.array(rewards).flatten().tolist())

            # SAME_STEP autoreset: next_obs is already the NEW episode's first
            # obs for finished envs. Time-limit (truncation) bootstrap, CleanRL
            # style: fold gamma*V(final_obs) into the reward so the GAE below
            # needs no special truncation path. final_obs is evaluated with the
            # PRE-step positions/mask/profiles of that env (the episode that
            # just ended); final_obs itself is never stored as a next obs.
            trunc_only = np.asarray(truncations) & ~np.asarray(terminations)
            if trunc_only.any():
                idxs = np.where(trunc_only)[0]
                final_obs_np = np.stack([
                    np.asarray(infos["final_obs"][i], dtype=np.float32) for i in idxs
                ])
                with torch.no_grad():
                    f_obs = torch.as_tensor(final_obs_np, device=device)
                    v_final = vf(
                        f_obs,
                        pos_buf[t, idxs],
                        mask_buf[t, idxs],
                        recep_buf[t, idxs] if use_profiles else None,
                        infl_buf[t, idxs] if (use_profiles and actor.use_influence) else None,
                    ).squeeze(-1)
                rew_buf[t, idxs] += args.gamma * v_final

            # DEL-penalty per-step diagnostics (vector env exposes wrapper info
            # keys as per-env arrays).
            if _del_active and "del_penalty" in infos:
                del_penalty_window.extend(
                    np.asarray(infos["del_penalty"], dtype=float).flatten().tolist())
                reward_unpen_window.extend(
                    np.asarray(infos["reward_unpenalized"], dtype=float).flatten().tolist())
                if "del_limit" in infos:
                    del_limit_window.extend(
                        np.asarray(infos["del_limit"], dtype=float).flatten().tolist())
                for _key, _win in (("del_ratio", del_ratio_window),
                                   ("del_agent_max", del_agent_max_window),
                                   ("del_baseline_max", del_base_max_window),
                                   ("del_margin", del_margin_window)):
                    if _key not in infos:
                        continue
                    _vals = np.asarray(infos[_key], dtype=float).flatten()
                    _win.extend(_vals[np.isfinite(_vals)].tolist())
                try:
                    _ood = infos.get("loads_ood")
                    if _ood is not None:
                        del_ood_window.extend(
                            float(np.mean(np.asarray(o, dtype=float))) for o in _ood)
                except (TypeError, ValueError):
                    pass
                if len(_del_reward_channels) > 1 and "del_binding_channel" in infos:
                    del_binding_window.extend(
                        ch for ch in np.asarray(
                            infos["del_binding_channel"], dtype=object
                        ).flatten() if ch is not None
                    )

            # Log episode stats
            if "final_info" in infos:
                ep_return = np.mean(envs.return_queue)
                ep_length = np.mean(envs.length_queue)
                ep_power = np.mean(envs.mean_power_queue)

                print(f"Step {global_step}: Episode return={ep_return:.2f}, power={ep_power:.2f}")
                writer.add_scalar("charts/episodic_return", ep_return, global_step)
                writer.add_scalar("charts/episodic_length", ep_length, global_step)
                writer.add_scalar("charts/episodic_power", ep_power, global_step)

                if getattr(envs, "mean_power_queue_baseline", None) and len(envs.mean_power_queue_baseline) > 0:
                    ep_power_base = float(np.mean(envs.mean_power_queue_baseline))
                    writer.add_scalar("charts/episodic_power_baseline", ep_power_base, global_step)
                    if ep_power_base > 0:
                        writer.add_scalar("charts/episodic_power_ratio",
                                          ep_power / ep_power_base, global_step)

                if _del_active and len(del_penalty_window) > 0:
                    writer.add_scalar("charts/episodic_del_penalty",
                                      float(np.mean(del_penalty_window)), global_step)
                    writer.add_scalar("charts/episodic_reward_unpenalized",
                                      float(np.mean(reward_unpen_window)), global_step)
                    if len(del_ratio_window) > 0:
                        writer.add_scalar("charts/episodic_del_ratio",
                                          float(np.mean(del_ratio_window)), global_step)
                    if len(del_agent_max_window) > 0:
                        writer.add_scalar("charts/del_agent_max",
                                          float(np.mean(del_agent_max_window)), global_step)
                    if len(del_base_max_window) > 0:
                        writer.add_scalar("charts/del_baseline_max",
                                          float(np.mean(del_base_max_window)), global_step)
                    if len(del_ood_window) > 0:
                        writer.add_scalar("charts/del_ood_frac",
                                          float(np.mean(del_ood_window)), global_step)
                    if len(del_limit_window) > 0:
                        writer.add_scalar("charts/del_limit",
                                          float(np.mean(del_limit_window)), global_step)
                    if len(del_margin_window) > 0:
                        writer.add_scalar("charts/del_margin",
                                          float(np.mean(del_margin_window)), global_step)
                    if len(del_binding_window) > 0:
                        _bind = list(del_binding_window)
                        for _ch in _del_reward_channels:
                            writer.add_scalar(
                                f"charts/del_binding_frac/{_ch}",
                                _bind.count(_ch) / len(_bind), global_step)

            obs = next_obs

        # =====================================================================
        # GAE
        # =====================================================================
        with torch.no_grad():
            # Bootstrap value for the state after the last step. SAME_STEP
            # keeps get_attr consistent with `obs` even right after a reset.
            fbatch = agent.batch_preparer.from_envs(envs, obs)
            next_value = vf(
                fbatch.obs, fbatch.positions, fbatch.mask,
                fbatch.receptivity,
                fbatch.influence if actor.use_influence else None,
            ).squeeze(-1)

            # Standard CleanRL GAE, with done_buf[t] gating the t -> t+1 carry.
            # Truncated steps have done=1 AND their tail value already folded
            # into rew_buf (bootstrap above), so one code path handles both
            # termination and truncation.
            advantages = torch.zeros_like(rew_buf)
            lastgaelam = torch.zeros(N, device=device)
            for t in reversed(range(args.num_steps)):
                nextvalues = next_value if t == args.num_steps - 1 else val_buf[t + 1]
                nextnonterminal = 1.0 - done_buf[t]
                delta = rew_buf[t] + args.gamma * nextvalues * nextnonterminal - val_buf[t]
                lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                advantages[t] = lastgaelam
            returns = advantages + val_buf

        # Flatten the rollout: each row carries its own positions/mask/profiles,
        # so mixed layouts within a minibatch are fine (same property the SAC
        # replay batch relies on).
        b_obs = obs_buf.reshape(T * N, K, Od)
        b_pos = pos_buf.reshape(T * N, K, 2)
        b_mask = mask_buf.reshape(T * N, K)
        b_recep = recep_buf.reshape(T * N, K, -1) if use_profiles else None
        b_infl = infl_buf.reshape(T * N, K, -1) if use_profiles else None
        b_xt = xt_buf.reshape(T * N, K, Ad)
        b_logp = logp_buf.reshape(T * N)
        b_val = val_buf.reshape(T * N)
        b_adv = advantages.reshape(T * N)
        b_ret = returns.reshape(T * N)

        # =====================================================================
        # PPO UPDATE
        # =====================================================================
        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in range(args.ppo_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                mb = b_inds[start:start + minibatch_size]

                mb_recep = b_recep[mb] if use_profiles else None
                # use_influence=False => influence_profile=None everywhere
                mb_infl = b_infl[mb] if (use_profiles and actor.use_influence) else None

                newlogprob, entropy = evaluate_actions(
                    actor, b_obs[mb], b_pos[mb], b_mask[mb], b_xt[mb],
                    mb_recep, mb_infl,
                )
                newvalue = vf(b_obs[mb], b_pos[mb], b_mask[mb],
                              mb_recep, mb_infl).squeeze(-1)

                logratio = newlogprob - b_logp[mb]
                ratio = logratio.exp()

                if not ratio_check_done:
                    # Before any gradient step the policy is unchanged, so the
                    # recomputed log-prob of the stored pre-tanh x_t must match
                    # the rollout log-prob exactly. Fails if the sample/evaluate
                    # math ever diverges (or if dropout > 0 injects noise).
                    _max_dev = (ratio - 1).abs().max().item()
                    assert _max_dev < 1e-4, (
                        f"First-minibatch ratio deviates from 1 by {_max_dev:.2e}: "
                        f"sample_action/evaluate_actions log-prob math is inconsistent "
                        f"(or --dropout > 0 is injecting train-mode noise)."
                    )
                    print(f"✓ First-minibatch ratio check passed (max |ratio-1| = {_max_dev:.2e})")
                    ratio_check_done = True

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                mb_adv = b_adv[mb]
                if args.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # Clipped surrogate policy loss
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_ret[mb]) ** 2
                    v_clipped = b_val[mb] + torch.clamp(
                        newvalue - b_val[mb], -args.clip_coef, args.clip_coef)
                    v_loss_clipped = (v_clipped - b_ret[mb]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_ret[mb]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                        trainable_params, max_norm=args.grad_clip_max_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # Explained variance of the value predictions over this rollout
        y_pred = b_val.cpu().numpy()
        y_true = b_ret.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # =====================================================================
        # LOGGING
        # =====================================================================
        sps = int(global_step / (time.time() - start_time))
        mean_reward = float(np.mean(step_reward_window)) if step_reward_window else 0.0

        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", float(np.mean(clipfracs)), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/SPS", sps, global_step)
        writer.add_scalar("charts/step_reward_mean_1000", mean_reward, global_step)
        writer.add_scalar("debug/mean_wind_direction", float(np.mean(wind_dirs)), global_step)

        print(f"Iter {iteration}/{num_iterations} (step {global_step}): SPS={sps}, "
              f"pg_loss={pg_loss.item():.4f}, v_loss={v_loss.item():.4f}, "
              f"kl={approx_kl.item():.4f}, clipfrac={float(np.mean(clipfracs)):.3f}, "
              f"ev={explained_var:.3f}, reward_mean={mean_reward:.4f}")

        # =====================================================================
        # CHECKPOINTING
        # =====================================================================
        if args.save_model and global_step >= next_save_step:
            save_ppo_checkpoint(actor, vf, optimizer, global_step, run_name, args)
            while next_save_step <= global_step:
                next_save_step += args.save_interval

        # =====================================================================
        # PERIODIC EVALUATION
        # =====================================================================
        if global_step >= next_eval_step:
            print(f"\nRunning evaluation at step {global_step}...")
            eval_metrics = evaluator.evaluate()
            eval_dict = eval_metrics.to_dict()

            for name, value in eval_dict.items():
                writer.add_scalar(name, value, global_step)

            print(f"Eval step {global_step} - Mean reward: {eval_metrics.mean_reward:.4f}, "
                  f"Power ratio: {eval_metrics.power_ratio:.4f}")

            if len(eval_metrics.per_layout_rewards) > 1:
                print("  Per-layout power ratios:")
                for layout, ratio_v in eval_metrics.per_layout_power_ratios.items():
                    print(f"    {layout}: {ratio_v:.4f}")

            while next_eval_step <= global_step:
                next_eval_step += args.eval_interval

    # =========================================================================
    # FINAL SAVE AND CLEANUP
    # =========================================================================

    if args.save_model:
        save_ppo_checkpoint(actor, vf, optimizer, global_step, run_name, args)

    print("\n" + "=" * 60)
    print("Training finished!")
    print(f"Total time: {(time.time() - start_time) / 3600:.2f} hours")
    print("=" * 60)

    evaluator.close()
    envs.close()
    writer.close()


if __name__ == "__main__":
    main()
