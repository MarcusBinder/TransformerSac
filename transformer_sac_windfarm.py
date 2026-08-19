"""
Transformer-based SAC for Wind Farm Control.

Trains a Soft Actor-Critic agent with a transformer backbone to learn
yaw control policies that generalize across wind farm layouts.

Design principles:
    1. Per-turbine tokenization: Each turbine is a token with local observations
    2. Wind-relative positional encoding: Positions rotated so wind comes from 270°
    3. Wind direction as deviation from mean (rotation invariant)
    4. Shared actor/critic heads across turbines (permutation equivariant)
    5. Adaptive target entropy based on actual turbine count
    6. Modular positional encoding with absolute and relative options

Positional encoding options (--pos_encoding_type):
    absolute_mlp, sinusoidal_2d, polar_mlp, relative_mlp, relative_mlp_shared,
    relative_polar, alibi, alibi_directional, absolute_plus_relative,
    RelativePositionalBiasAdvanced, RelativePositionalBiasFactorized,
    RelativePositionalBiasWithWind

Author: Marcus Binder Nilsen (DTU Wind Energy)
"""

import copy
import os
import random
import sys
import time
from typing import Optional, Tuple, List, Dict, Any, Union
from collections import deque
import json

from config import Args
from replay_buffer import TransformerReplayBuffer
from helpers.training_utils import (
    clear_gpu_memory, compute_adaptive_target_entropy,
    get_env_current_layout, log_optimizer_effective_lr,
    compute_optimizer_diagnostics, log_finetune_diagnostics,
)

# Set memory allocation config BEFORE importing torch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

# WindGym imports (adjust path as needed for your setup)
from WindGym import WindFarmEnv
from WindGym.wrappers import RecordEpisodeVals, PerTurbineObservationWrapper
from helpers.agent import WindFarmAgent

# Logging utilities for multi-layout training
from helpers.multi_layout_debug import (
    MultiLayoutDebugLogger,
    create_debug_logger,
)

from helpers.helper_funcs import (
    get_env_wind_directions,
    get_env_raw_positions,
    get_env_attention_masks,
    save_checkpoint,
    load_checkpoint,
    compute_wind_direction_deviation,
    EnhancedPerTurbineWrapper,
    get_env_receptivity_profiles,
    get_env_influence_profiles,
    rotate_profiles_tensor,
    get_env_layout_indices,
    get_env_permutations,
    soft_update,
)
from helpers.layouts import get_layout_positions
from helpers.env_configs import (
    make_env_config,
    apply_config_overrides,
    make_eval_wind_config,
)

# Receptivity profile computation
from helpers.receptivity_profiles import compute_layout_profiles

# Evaluation import
from helpers.eval_utils import PolicyEvaluator, run_evaluation

# Repo root (parent of TransformerSac/): `del_surrogate` lives there and is
# not installed into the pixi env. The path insert + import happen INSIDE
# combined_wrapper so they also run in AsyncVectorEnv worker processes,
# where module-level state of __main__ may not be replayed.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from networks import (
    TransformerActor,
    TransformerCritic,
    TransformerTQCCritic,
    TransformerTQCSharedCritic,
    create_profile_encoding,
    quantile_huber_loss,
)



# =============================================================================
# MAIN TRAINING SCRIPT
# =============================================================================

def main():
    """Main training function."""
    
    # Parse arguments
    args = tyro.cli(Args)
    
    # Validate initial_exploration
    assert args.initial_exploration in ("random", "policy"), \
        f"--initial_exploration must be 'random' or 'policy', got '{args.initial_exploration}'"
    if args.initial_exploration == "policy" and args.resume_checkpoint is None:
        print("WARNING: --initial_exploration=policy without --resume_checkpoint. "
              "The actor is untrained, so 'policy' exploration will just be random Gaussian noise.")
    if args.initial_exploration == "policy":
        print(f"Initial exploration: using actor network for first {args.learning_starts} steps")

    # Validate replay buffer save/load flags
    assert not (args.buffer_only and args.load_buffer is not None), \
        "--buffer_only generates a warmup buffer; combining it with --load_buffer is pointless"
    if args.buffer_only and not args.save_buffer_at_learning_starts:
        print("NOTE: --buffer_only implies --save_buffer_at_learning_starts, enabling it.")
        args.save_buffer_at_learning_starts = True
    if args.save_buffer_at_learning_starts and args.initial_exploration == "policy":
        print("WARNING: Saving the warmup buffer with --initial_exploration=policy. "
              "The saved transitions depend on the actor weights, not just the seed.")
    
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
    print(f"Transformer SAC for Wind Farm Control")
    print("=" * 60)
    if is_multi_layout:
        print(f"Mode: Multi-layout training with layouts: {layout_names}")
    else:
        print(f"Mode: Single-layout training: {layout_names[0]}")
    print(f"Run name: {run_name}")
    print("=" * 60)
    
    # Create directories
    os.makedirs(f"runs/{run_name}", exist_ok=True)
    os.makedirs(f"runs/{run_name}/checkpoints", exist_ok=True)
    os.makedirs(f"runs/{run_name}/attention_plots", exist_ok=True)
    
    clear_gpu_memory()
    
    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    
    assert args.algorithm in ("sac", "tqc"), \
        f"--algorithm must be 'sac' or 'tqc', got '{args.algorithm}'"

    if args.use_droq and args.utd_ratio < 10:
        print(f"WARNING: DroQ is enabled but utd_ratio={args.utd_ratio}. "
              f"DroQ typically benefits from high UTD ratios (>=10, often 20).")

    if args.use_droq and args.policy_frequency > 1:
        print(f"WARNING: DroQ is enabled but policy_frequency={args.policy_frequency}. "
              f"DroQ typically uses policy_frequency=1 to update the actor every gradient step.")

    if args.use_droq and args.algorithm == "tqc":
        print("NOTE: DroQ dropout is active during TQC actor updates. "
              "Dropout noise affects which quantiles are truncated, potentially "
              "weakening TQC's pessimism. Monitor Q-value overestimation carefully.")

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")

    # Fused Adam: collapses Adam's per-parameter foreach kernels into a single
    # fused launch. On this small/short-sequence model the update loop is
    # launch-bound (esp. utd>1, which runs num_envs*utd updates/iter), so this is
    # a direct, math-equivalent speedup. CUDA-only; falls back to default on CPU.
    _adam_fused = device.type == "cuda"
    def make_adam(params, lr):
        return optim.Adam(params, lr=lr, fused=_adam_fused)
    if _adam_fused:
        print("Optimizers: fused Adam enabled (CUDA)")

    # Force the math SDPA backend on ROCm, which has Flash/MemEfficient kernel
    # bugs for our attention shapes. torch.version.hip is a version string on
    # ROCm builds and None on CUDA builds, so this self-activates on LUMI's
    # MI250X and stays dormant on Sophia -- no hostname sniffing, and nothing
    # changes for existing CUDA runs.
    if device.type == "cuda" and torch.version.hip is not None:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        print(f"ROCm detected (HIP {torch.version.hip}): forced math SDPA backend")

    # =========================================================================
    # ENVIRONMENT SETUP
    # =========================================================================
    
    # Import WindGym components
    from WindGym import WindFarmEnv
    from WindGym.wrappers import RecordEpisodeVals, PerTurbineObservationWrapper
    from helpers.multi_layout_env import MultiLayoutEnv, LayoutConfig
    
    # Wind turbine. Derate-enabled configs (e.g. power_max_derate) need a
    # turbine whose powerCtFunction accepts a `derate` input; plain turbine
    # classes fail WindFarmEnv's check_turbine_supports_derating, so dispatch
    # on --turbtype to the derate-capable surrogate turbines (IEA34 default;
    # DTU10MW for reproducing old checkpoints). make_env_config is pure/cheap;
    # the full env config is (re)built later for the env itself.
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
        """Build a LayoutConfig and attach receptivity/influence profiles (if enabled).

        Shared by the fixed named-layout path and the domain-randomization pool so
        generated farms get profiles computed exactly like the hand-picked ones.
        """
        layout = LayoutConfig(name=name, x_pos=x_pos, y_pos=y_pos)
        if args.profile_encoding_type is not None:
            if args.profile_source.lower() == "geometric":
                from helpers.geometric_profiles import compute_layout_profiles_vectorized

                # Get rotor diameter as a float (geometric version doesn't need the full WT object)
                D = wind_turbine.diameter()  # or however DTU10MW exposes this

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

            layout.receptivity_profiles = receptivity_profiles  # (n_turbines, n_directions
            layout.influence_profiles = influence_profiles      # (n_turbines, n_directions
        return layout

    layouts = []
    if dr_enabled:
        # Domain randomization (v8): training layouts are a large procedurally
        # generated pool instead of the fixed named set. Pool seeded from --seed so
        # different seeds draw different layout sets. See helpers/layout_gen.py.
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

    if args.profile_encoding_type is not None:
        use_profiles = True
    else:
        use_profiles = False



    # Build profile registry from layouts
    if use_profiles:
        profile_registry = [
            (layout.receptivity_profiles, layout.influence_profiles)
            for layout in layouts
        ]
    else:
        profile_registry = None


    # =========================================================================
    # PRE-SCAN CHECKPOINT FOR ENV-AFFECTING ARGS (before env creation)
    # =========================================================================
    # If a pretrain/BC checkpoint is provided, we need action_type and
    # history_length BEFORE creating the environment, since they affect
    # config["ActionMethod"] and observation shape.
    if args.pretrain_checkpoint is not None and os.path.exists(args.pretrain_checkpoint):
        _prescan = torch.load(args.pretrain_checkpoint, map_location="cpu", weights_only=False)
        _prescan_args = _prescan.get("args", {})

        # --- action_type ---
        if "action_type" in _prescan_args:
            ckpt_action_type = _prescan_args["action_type"]
            if ckpt_action_type != args.action_type:
                print(f"  [pre-scan] Overriding action_type: {args.action_type} → {ckpt_action_type} (from checkpoint)")
                args.action_type = ckpt_action_type
            else:
                print(f"  [pre-scan] action_type already matches checkpoint: {args.action_type}")

        # --- history_length ---
        if "history_length" in _prescan_args:
            ckpt_history = _prescan_args["history_length"]
            if ckpt_history != args.history_length:
                print(f"  [pre-scan] Overriding history_length: {args.history_length} → {ckpt_history} (from checkpoint)")
                args.history_length = ckpt_history
            else:
                print(f"  [pre-scan] history_length already matches checkpoint: {args.history_length}")

        del _prescan  # free memory; full load happens later

    # Environment configuration
    print(f"using the config: {args.config}")
    config = make_env_config(args.config)

    # Override ActionMethod from args (default "wind", or overridden by checkpoint above)
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

    # LES-3x3 Stage 4 (--obs_agg): the ObsAggWrapper reduces the env's raw
    # measurement deques itself, so only the DEQUE length grows to obs_agg_len
    # (history_N, i.e. the base obs width the wrapper replaces, is unchanged).
    # The env's reset burn-in follows max_hist(), so all L samples are filled
    # before the first observation (+L-power_avg sim steps per reset).
    mes_history_length = args.history_length
    if args.obs_agg:
        if args.obs_encoding:
            raise ValueError("--obs_agg cannot be combined with --obs_encoding")
        if args.use_wd_deviation:
            raise ValueError("--obs_agg cannot be combined with --use_wd_deviation")
        if args.obs_encoder_mode == "per_sensor":
            raise ValueError("--obs_agg needs --obs_encoder_mode shared "
                             "(per_sensor asserts obs_dim == 4*history_length)")
        if args.del_penalty_scale > 0 or args.del_log:
            raise ValueError("--obs_agg cannot be combined with the DEL wrapper "
                             "(it appends a per-turbine column)")
        from helpers.obs_agg import AGG_MODES
        if args.obs_agg not in AGG_MODES:
            raise ValueError(f"Unknown --obs_agg {args.obs_agg!r}; valid: "
                             f"{sorted(AGG_MODES)}")
        mes_history_length = max(args.history_length, args.obs_agg_len)
        print(f"obs_agg: mode={args.obs_agg} K={AGG_MODES[args.obs_agg].K} "
              f"L={args.obs_agg_len} ({AGG_MODES[args.obs_agg].doc}); "
              f"measurement deques lengthened to {mes_history_length}, "
              f"history_N stays {args.history_length}")

    for mes_type, prefix in mes_prefixes.items():
        if mes_type not in config:
            continue  # e.g. derate_mes is absent outside the derate presets
        config[mes_type][f"{prefix}_history_N"] = args.history_length
        config[mes_type][f"{prefix}_history_length"] = mes_history_length

    # change_wd_3 reward / training-wind overrides (all no-ops when unset).
    # Applied HERE, before make_eval_env_factory deep-copies `config`, so the
    # reward definition is shared by training and eval envs; the training-only
    # ws range is then overwritten per eval spec below.
    apply_config_overrides(config, args)

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

    # LES-3x3 campaign: relax dynamiks' apparent-turbine-motion cap so the
    # wd_slow frame can track a moving wd schedule (its ONLY consumer is the
    # frame slew limit max_wd_step = max_turb_move*360/(2*pi*max_dist) per sim
    # step). Large enough that the clip never binds => wd_slow == schedule,
    # wd_small == 0, and env.wd is exact AND causal. Spread into all three env
    # factories (train / in-training eval / train-wd) via base_env_kwargs, so
    # train and eval physics stay consistent. None = env default (2 m).
    if args.max_turb_move is not None:
        base_env_kwargs["max_turb_move"] = float(args.max_turb_move)
        print(f"max_turb_move set to: {base_env_kwargs['max_turb_move']} m "
              f"(wd_slow frame slew limit scales with it)")

    # WD-estimation ladder: swap the privileged env.wd for the sensor-derived
    # estimate. The env computes it from measurements it already takes
    # (core/wd_estimator.py); rollout fetch + agent fetch + replay buffer all
    # read through wd_source_attr, so gradients never see the true wd.
    assert args.wd_source in ("true", "est"), (
        f"--wd_source must be 'true' or 'est', got {args.wd_source!r}")
    if args.wd_source == "est":
        if args.backend == "pywake":
            raise ValueError(
                "--wd_source est requires --backend dynamiks: pywake's "
                "adapter hard-codes v=w=0, so a measured local wind "
                "direction does not exist there (atan2(v,u) == 0).")
        if args.wd_est_tau is None:
            raise ValueError("--wd_source est requires --wd_est_tau")
        base_env_kwargs["wd_est_tau"] = float(args.wd_est_tau)
        base_env_kwargs["wd_est_consensus"] = args.wd_est_consensus
        print(f"wd source: ESTIMATED (tau={args.wd_est_tau}s, "
              f"consensus={args.wd_est_consensus}) — the policy never "
              f"sees the privileged env.wd")
    wd_source_attr = "wd" if args.wd_source == "true" else "wd_est"

    # DEL-constrained reward: DELRewardWrapper compares agent DELs against a
    # greedy baseline farm, which only exists when the env is built with
    # Baseline_comp=True. With Power_reward="Baseline" (the power-max presets)
    # windgym forces Baseline_comp anyway, so the explicit kwarg is free; it
    # matters only for other reward types. --del_log attaches the wrapper with
    # penalty_scale=0 (penalty exactly 0, reward untouched) so an unpenalized
    # A/B arm still logs the same charts/del_* metrics. BaseController
    # "Global" pins the baseline yaw offset to 0 (deterministic greedy
    # reference); "Local" would chase the fluctuating local wind direction and
    # add noise to both the Baseline reward denominator and the DEL reference.
    _del_active = args.del_penalty_scale > 0 or args.del_log
    assert not args.del_limit_random or _del_active, (
        "--del_limit_random conditions the policy on the DEL limit, which "
        "only exists when the DEL wrapper is attached: also pass "
        "--del_penalty_scale > 0 (or --del_log)."
    )
    if _del_active:
        base_env_kwargs["Baseline_comp"] = True
        config["BaseController"] = "Global"

    # change_wd_4: override the env's hardcoded 0-30 m/s obs-affine range
    # (OBS_SCALING.md). These are WindFarmEnv CTOR kwargs, deliberately NOT
    # config-dict keys — None means "omit", so the env default stays untouched.
    # The eval factories spread {**base_env_kwargs, ...} below, so train and
    # eval envs get the same scaling automatically.
    if args.ws_scaling_min is not None:
        base_env_kwargs["ws_scaling_min"] = float(args.ws_scaling_min)
    if args.ws_scaling_max is not None:
        base_env_kwargs["ws_scaling_max"] = float(args.ws_scaling_max)
    if args.ws_scaling_min is not None or args.ws_scaling_max is not None:
        print(f"ws obs scaling override: "
              f"[{base_env_kwargs.get('ws_scaling_min', 0.0)}, "
              f"{base_env_kwargs.get('ws_scaling_max', 30.0)}] m/s")


    def make_env_factory(env_kwargs: dict):
        """Build a WindFarmEnv factory bound to one set of env kwargs."""
        def _factory(x_pos: np.ndarray, y_pos: np.ndarray) -> gym.Env:
            env = WindFarmEnv(x_pos=x_pos,
                              y_pos=y_pos,
                              reset_init=False,  # Defer reset to training loop
                              **env_kwargs)
            env.action_space.seed(args.seed)
            return env
        return _factory

    # Static-wd factory: the default for training, and the eval fallback when no
    # --eval_wd_function is given. Deliberately carries NO wd_function, so a
    # --train_wd_function can never leak into the eval envs.
    env_factory = make_env_factory(base_env_kwargs)

    # Eval-only env factories: when --eval_wd_function is set (comma-separated for
    # several schedules), eval envs get a time-varying wd schedule (training envs are
    # untouched by this path). The burn-in holds the per-reset base_wd, so wd is pinned
    # to wd_function(0) — and ws to the spec's speed so all episodes in a cell share
    # one condition — mirroring make_flow_gif.py.
    #
    # The ladder is the CROSS PRODUCT (schedule x --eval_ws), because the reward
    # conditioning under test is largely a wind-SPEED story: DTU10MW rates near
    # 11.4 m/s, so an eval pinned only at 12 m/s sits above rated and cannot show
    # whether an arm gained anything below it. eval_specs entries are
    # (wd_name, ws, spec_name); spec_name carries the /ws<speed> segment only when
    # more than one speed is requested, keeping the single-speed namespace
    # byte-identical to change_wd_2.
    from helpers.wd_functions import build_eval_specs
    eval_specs = build_eval_specs(args.eval_wd_function, args.eval_ws)
    eval_spec_names = [name for _, _, name in eval_specs]

    def make_eval_env_factory(wd_name: str, ws: float):
        from helpers.wd_functions import get_wd_function
        _wd_fn = get_wd_function(wd_name)
        _wd0 = float(_wd_fn(0.0))
        # Pins wd to the schedule's t=0 value and ws UNCONDITIONALLY — the latter
        # is what keeps a --train_ws_min/max override out of the eval condition.
        eval_config = make_eval_wind_config(config, _wd0, ws)
        print(f"Eval wd_function: {wd_name} "
              f"(eval wind pinned to wd={_wd0}, ws={float(ws)})")
        return make_env_factory({**base_env_kwargs,
                                 "config": eval_config,
                                 "wd_function": _wd_fn})

    eval_env_factories = ([make_eval_env_factory(wd, ws) for wd, ws, _ in eval_specs]
                          if eval_specs else [env_factory])

    # Training wd schedule (--train_wd_function). Unlike the eval path these are
    # RELATIVE — wd(t) = base_wd + delta(t) — so wd_min/wd_max are deliberately NOT
    # pinned and the config's per-episode wd randomization survives. The schedules are
    # stateful (they re-draw every episode), so each vector-env slot builds its own
    # instance seeded off that slot's seed; a single shared instance would make all
    # num_envs envs walk one identical wd trajectory.
    def make_train_env_factory(env_seed: int):
        if args.train_wd_function is None:
            return env_factory
        from helpers.wd_functions import get_train_wd_factory
        return make_env_factory({
            **base_env_kwargs,
            "wd_function": get_train_wd_factory(args.train_wd_function, seed=env_seed),
        })

    if args.train_wd_function is not None:
        from helpers.wd_functions import ABSOLUTE_TRAIN_NAMES, get_train_wd_factory
        if args.train_wd_function in ABSOLUTE_TRAIN_NAMES:
            # ABSOLUTE training schedule (LES-3x3 Stage 5 cycle): the burn-in
            # holds the drawn base_wd and wd_list[0] = base_wd, so the preset's
            # wd band MUST be pinned to f(0) or every episode starts with a wd
            # jump that the backward wd_slow pass smears non-causally into the
            # burn-in. Fail loudly instead of training on that silently.
            _probe = get_train_wd_factory(args.train_wd_function, seed=args.seed)
            _f0 = float(_probe(0.0))
            _wd_lo, _wd_hi = float(config["wind"]["wd_min"]), float(config["wind"]["wd_max"])
            if not (_wd_lo == _wd_hi == _f0):
                raise ValueError(
                    f"--train_wd_function {args.train_wd_function} is ABSOLUTE "
                    f"(f(0)={_f0}); --config {args.config} must pin wd_min=wd_max={_f0} "
                    f"(got [{_wd_lo}, {_wd_hi}]; use e.g. les_recipe_pin270)"
                )
            print(f"Train wd_function: {args.train_wd_function} "
                  f"(ABSOLUTE; phase-randomised per episode; wd pinned by preset "
                  f"to {_f0}; per-env seeds {args.seed}..{args.seed + args.num_envs - 1})")
        else:
            print(f"Train wd_function: {args.train_wd_function} "
                  f"(relative schedule; wd domain randomization preserved, "
                  f"per-env seeds {args.seed}..{args.seed + args.num_envs - 1})")

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
        # DEL hinge penalty BEFORE TransformReward, so the scaler multiplies
        # the ALREADY-penalized reward: effective reward is
        # reward_scale * (power_reward - del_penalty). del_penalty_scale is
        # therefore in pre-scale units and the penalty/reward ratio is
        # invariant to --reward_scale. Placed after the obs wrappers (not
        # innermost) because gymnasium 1.x wrappers don't forward attributes:
        # PerTurbineObservationWrapper reads env.n_turb at construction, while
        # DELRewardWrapper only touches env.unwrapped, so it can sit anywhere
        # in the chain. Reward-wise the two orders are identical (obs wrappers
        # pass reward through untouched).
        if args.del_penalty_scale > 0 or args.del_log:
            # NN DEL surrogate (default) or proxy zoo (--load_proxies); the
            # comparison rule / penalty kind come from --load_compare /
            # --load_penalty. Goal-conditioned limit: per-episode uniform
            # sample + one obs column per turbine (limit / del_limit_obs_ref).
            # None keeps the fixed allowed_increase behavior bit-identical.
            from helpers.load_reward import build_load_reward_wrapper
            env = build_load_reward_wrapper(
                env, args,
                limit_range=(
                    (args.del_limit_lo, args.del_limit_hi)
                    if args.del_limit_random else None
                ),
                limit_obs_ref=args.del_limit_obs_ref,
            )
        # change_wd_4: re-encode the ws columns (rbf/pyramid/cdf/fourier/reldef/
        # pcurve). Must sit INSIDE TransformReward so MultiLayoutEnv's
        # outermost-first _obs_dim_per_turbine probe finds the expanded dim.
        if args.obs_encoding:
            from helpers.obs_encoding import ObsEncodingWrapper
            env = ObsEncodingWrapper(env, mode=args.obs_encoding,
                                     turbine=wind_turbine,
                                     **json.loads(args.obs_encoding_kwargs))
        # LES-3x3 Stage 4: aggregate scheme over the L-long deques (same slot
        # as obs_encoding: inside TransformReward, before MultiLayoutEnv's
        # shuffle/pad, so spatial_rel sees the true layout order).
        if args.obs_agg:
            from helpers.obs_agg import ObsAggWrapper
            env = ObsAggWrapper(env, args.obs_agg, args.obs_agg_len,
                                dt_env=args.dt_env)
        # v9.1: scale the (tiny) Wake_recovery reward to probe optimization signal-to-noise.
        if args.reward_scale != 1.0:
            _scale = float(args.reward_scale)
            env = gym.wrappers.TransformReward(env, lambda r: r * _scale)
        return env
    
    def make_env_fn(seed, warmup_steps=None):
        """Factory function for vectorized environments."""
        def _init():
            # Built inside _init so the (stateful) training wd schedule is
            # constructed in this worker process, one instance per env slot.
            env = MultiLayoutEnv(
                layouts=layouts,
                env_factory=make_train_env_factory(seed),
                per_turbine_wrapper=combined_wrapper,  # Use combined wrapper
                seed=seed,
                shuffle=args.shuffle_turbs,  # Shuffle turbines within each layout
                max_turbines=args.max_turbines,  # fixed padding/network size (DR: eval up to 25)
                max_episode_steps=args.max_episode_steps,
                warmup_episode_steps=warmup_steps,
            )
            return env
        return _init

    # Compute per-env one-time warm-up episode lengths. Staggering only the first
    # episode permanently phase-offsets each group's resets/shuffles while keeping
    # every subsequent episode at the standard max_episode_steps.
    warmup_lengths = [None] * args.num_envs  # default: no stagger (current behavior)
    if args.stagger_warmup:
        assert args.max_episode_steps is not None, \
            "--stagger_warmup requires --max_episode_steps to be set"
        assert args.warmup_min_episode_steps is not None, \
            "--stagger_warmup requires --warmup_min_episode_steps"
        assert args.warmup_min_episode_steps <= args.max_episode_steps, \
            "--warmup_min_episode_steps must be <= --max_episode_steps"

        num_groups = -(-args.num_envs // args.warmup_group_size)  # integer ceil
        if num_groups == 1:
            # A single group has nothing to desync; warm-up is a no-op.
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

    # Which column of a per-turbine action row is yaw and which is derate, for
    # the actions/* diagnostics. PerTurbineObservationWrapper emits ONE ROW PER
    # TURBINE -- [yaw_i, derate_i] with both enabled, or the single enabled
    # channel -- and transposes to the base env's variable-grouped
    # [yaw_0..yaw_n | derate_0..derate_n] layout internally. So the column index
    # here is NOT the base env's block offset; mixing the two up would silently
    # mislabel yaw as derate.
    _env_flags = make_env_config(args.config)
    _yaw_on = bool(_env_flags.get("yaw_action", True))
    _derate_on = bool(_env_flags.get("derate_action", False))
    if action_dim_per_turbine >= 2:
        _act_cols = {"yaw": 0, "derate": 1}
    elif _derate_on and not _yaw_on:
        _act_cols = {"derate": 0}
    else:
        _act_cols = {"yaw": 0}
    print(f"Action columns (for actions/* logging): {_act_cols}")

    # change_wd_4: agent-side running obs normalization. Constructed from the
    # WRAPPED obs dim so it composes with --obs_encoding; applied at act() time
    # (via the agent's BatchPreparer) and on replay batches after rb.sample.
    # See helpers/obs_norm.py for why this is not a per-env wrapper.
    obs_normalizer = None
    if args.obs_norm:
        from helpers.obs_norm import ObsRunningNorm
        obs_normalizer = ObsRunningNorm(obs_dim_per_turbine, device)
        print(f"ObsRunningNorm enabled ({obs_dim_per_turbine} features, "
              f"updates from step 0, clip ±{obs_normalizer.clip})")


    # Create policy evaluators — one per eval wd schedule. Every evaluator shares the
    # same eval_seed, so the schedules are compared on PAIRED episodes.
    evaluators = [
        PolicyEvaluator(
            agent=None,  # Will be set after actor is created
            eval_layouts=eval_layout_names,
            env_factory=_factory,
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
            use_profiles=use_profiles,  # NEW: Pass profile setting
            n_profile_directions=args.n_profile_directions,  # NEW: Pass profile resolution
            profile_source=args.profile_source,
            profile_sigma_smooth=args.profile_sigma_smooth,
            profile_geom_mode=args.profile_geom_mode,
        )
        for _factory in eval_env_factories
    ]

    # FORK SAFETY: create every evaluator's AsyncVectorEnv NOW — before the
    # networks initialize the GPU (HIP/CUDA context) and before any gradient
    # burst spins up torch/OMP threads — and keep them RESIDENT for the whole
    # run. Lazily forking eval workers at the first mid-training eval is a
    # fork-after-threads deadlock: the child inherits a lock some torch thread
    # holds and hangs in futex_do_wait while the parent blocks on the worker
    # pipe (observed on local CPU and on LUMI/ROCm, where it froze all six T3
    # runs at their first 50k eval for 10+ hours). The training envs never
    # deadlock for exactly this reason — they fork at startup. Resident cost
    # is num_specs x num_envs idle workers; the eval layouts are capped small
    # (square_2x2), so this is noise next to the 30 training envs.
    for _ev in evaluators:
        _ = _ev.eval_envs  # property; triggers AsyncVectorEnv creation

    def run_all_evaluations():
        """Evaluate on every eval wd schedule.

        Returns (metrics_dict, primary_metrics). The FIRST spec additionally
        writes the historical unprefixed eval/... keys so existing W&B panels and
        paper_figures.ipynb readers keep working; every spec also writes
        eval/wd/<spec>/... . Evaluators are deliberately NOT closed between
        passes — their envs must stay resident (see the fork-safety note at
        creation above).
        """
        metrics = {}
        primary = None
        for i, ev in enumerate(evaluators):
            m = ev.evaluate()
            if i == 0:
                primary = m
                metrics.update(m.to_dict())
            if eval_spec_names:
                metrics.update(m.to_dict(prefix=f"eval/wd/{eval_spec_names[i]}"))
                print(f"  [{eval_spec_names[i]}] power ratio: {m.power_ratio:.4f}")
        return metrics, primary

    def close_all_evaluators():
        for ev in evaluators:
            ev.close()


    # Action scaling
    action_high = envs.single_action_space.high[0]
    action_low = envs.single_action_space.low[0]
    action_scale = (action_high - action_low) / 2.0
    action_bias = (action_high + action_low) / 2.0
    
    # =========================================================================
    # DEBUG LOGGER AND TRACKING SETUP
    # =========================================================================

    # Initialize debug logger with configurable frequencies.
    # Under domain randomization the training pool is huge (e.g. 2048 layouts), so
    # per-layout debug stats are meaningless and would bloat the logger / W&B; bucket
    # all DR training steps under a single "dr_pool" name. Per-layout EVAL metrics
    # (the ones we analyse) come from the evaluator on the fixed eval ladder.
    debug_layout_names = ["dr_pool"] if dr_enabled else layout_names
    debug_logger = create_debug_logger(
        layout_names=debug_layout_names,
        log_every=250000,  # Base frequency - others are multiples of this
    )
    # Frequencies will be:
    #   - summary metrics: every 100 steps
    #   - attention analysis: every 500 steps  
    #   - gradient norms: every 100 steps
    #   - q-value stats: every 50 steps
    #   - diagnostic print: every 2000 steps

    print(f"Debug logger initialized for layouts: {debug_layout_names}")
    print(f"  Attention logging every {debug_logger.attention_log_frequency} steps")
    print(f"  Gradient logging every {debug_logger.gradient_log_frequency} steps")
    
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args) | {
                # Debug/multi-layout config
                "debug/n_layouts": len(layout_names),
                # Under DR the pool is huge; log a compact summary instead of 2048 names.
                "debug/layout_names": (f"dr_pool[{args.dr_n_lo}-{args.dr_n_hi}]x{len(layout_names)}"
                                       if dr_enabled else layout_names),
                "debug/is_multi_layout": is_multi_layout,
                "debug/max_turbines": n_turbines_max,
                "debug/log_frequency": debug_logger.log_frequency,
                "debug/attention_log_frequency": debug_logger.attention_log_frequency,
                "debug/gradient_log_frequency": debug_logger.gradient_log_frequency,
            },
            name=run_name,
            group=args.exp_group,
            monitor_gym=True,
            save_code=True,
        )

        # Everything reaches W&B through sync_tensorboard, so the TB tag is the
        # W&B key and the TB step arrives as "global_step". Declaring the groups
        # explicitly makes each one chart against global_step instead of
        # W&B's own monotonically-increasing _step, which otherwise spreads
        # metrics logged at different frequencies (per-iteration perf/* vs
        # per-episode del/*) across mismatched x-axes.
        wandb.define_metric("global_step")
        for _group in ("perf/*", "actions/*", "del/*", "losses/*",
                       "charts/*", "entropy/*", "timing/*"):
            wandb.define_metric(_group, step_metric="global_step")

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n" + "\n".join([f"|{k}|{v}|" for k, v in vars(args).items()])
    )

    # =========================================================================
    # NETWORK SETUP
    # =========================================================================
    
    # =========================================================================
    # OVERRIDE ARCHITECTURE ARGS FROM PRETRAIN CHECKPOINT (if provided)
    # =========================================================================

    if args.pretrain_checkpoint is not None:
        print(f"\n{'='*60}")
        print(f"PRETRAIN CHECKPOINT: loading architecture config")
        print(f"{'='*60}")
        print(f"Checkpoint: {args.pretrain_checkpoint}")

        if not os.path.exists(args.pretrain_checkpoint):
            raise FileNotFoundError(f"Pretrain checkpoint not found: {args.pretrain_checkpoint}")

        _pt_ckpt = torch.load(args.pretrain_checkpoint, map_location="cpu", weights_only=False)

        if "args" not in _pt_ckpt:
            raise ValueError("Pretrain checkpoint missing 'args' key — cannot load architecture config")

        pt_args = _pt_ckpt["args"]

        # Keys that MUST match between pretrain and RL for weight loading to work
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

        # Store for phase 2 (weight loading after network construction)
        _pretrain_encoder_sd = _pt_ckpt["encoder_state_dict"]
        print(f"  Encoder state dict: {len(_pretrain_encoder_sd)} parameter tensors")

        # BC checkpoints also contain the full actor (including action heads)
        _pretrain_actor_sd = _pt_ckpt.get("actor_state_dict", None)
        if _pretrain_actor_sd is not None:
            print(f"  Actor state dict:   {len(_pretrain_actor_sd)} parameter tensors (BC checkpoint detected)")
        else:
            print(f"  No actor_state_dict found (self-supervised pretrain checkpoint)")
        print(f"{'='*60}\n")

        del _pt_ckpt  # free memory, keep only what we need
    
    
    
    print("\nCreating networks...")
    print(f"Positional encoding type: {args.pos_encoding_type}")

    # ==========================================================================
    # Create SHARED profile encoders (if using profiles)
    # ==========================================================================
    if args.profile_encoding_type is not None:
        if args.share_profile_encoder:
            encoder_kwargs = json.loads(args.profile_encoder_kwargs)
            print(f"Creating shared profile encoders: {args.profile_encoding_type}")
            shared_recep_encoder, shared_influence_encoder = create_profile_encoding(
                profile_type=args.profile_encoding_type,
                embed_dim=args.embed_dim,
                hidden_channels=args.profile_encoder_hidden,
                **encoder_kwargs,
            )
            # Move to device
            shared_recep_encoder = shared_recep_encoder.to(device)
            shared_influence_encoder = shared_influence_encoder.to(device)
        
            # Count shared encoder parameters
            recep_params = sum(p.numel() for p in shared_recep_encoder.parameters())
            influence_params = sum(p.numel() for p in shared_influence_encoder.parameters())
            print(f"Shared receptivity encoder parameters: {recep_params:,}")
            print(f"Shared influence encoder parameters: {influence_params:,}")
        else:
            print(f"Using separate profile encoders for each network, handled internally in the critic and actor classes")
            shared_recep_encoder = None  # 
            shared_influence_encoder = None  # 
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

    # Actor has additional action scaling params
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
        # The evaluators share this agent, so eval act() calls are normalized
        # with the same (live) statistics as training — eval envs never go cold.
        obs_normalizer=obs_normalizer,
        # PolicyEvaluator lets the agent fetch wd itself, so the wd source
        # follows the agent into every eval pass too.
        wd_attr=wd_source_attr,
    )

    # Update evaluator with actor reference
    for _ev in evaluators:
        _ev.agent = agent

    # Build critic-specific kwargs (DroQ params only go to critics, not actor)
    critic_kwargs = {**common_kwargs}
    if args.use_droq:
        critic_kwargs["droq_dropout"] = args.droq_dropout
        critic_kwargs["droq_layer_norm"] = args.droq_layer_norm

    # Get critic parameters, excluding shared profile encoders
    def get_critic_params_excluding_shared(critic, shared_recep, shared_influence):
        '''Get critic parameters, excluding shared modules.'''
        shared_param_ids = set()
        if shared_recep is not None:
            shared_param_ids.update(id(p) for p in shared_recep.parameters())
        if shared_influence is not None:
            shared_param_ids.update(id(p) for p in shared_influence.parameters())
        return [p for p in critic.parameters() if id(p) not in shared_param_ids]

    # Collect shared encoder params so they receive gradients from critic loss.
    # These are excluded from actor_optimizer to avoid double updates with
    # conflicting Adam states — q_optimizer is the sole owner.
    shared_encoder_params = []
    if shared_recep_encoder is not None:
        shared_encoder_params += list(shared_recep_encoder.parameters())
    if shared_influence_encoder is not None:
        shared_encoder_params += list(shared_influence_encoder.parameters())
    shared_param_ids = {id(p) for p in shared_encoder_params}

    # Initialize critic variables (some will be None depending on algorithm)
    qf1 = qf2 = qf1_target = qf2_target = None
    tqc_critic = tqc_critic_target = None
    taus = None

    if args.algorithm == "tqc":
        # NOTE (pre-existing, both TQC variants + SAC): with --share_profile_encoder,
        # critic_kwargs holds the LIVE encoder modules, so the target's profile
        # encoders alias the live ones and soft_update lerps them with themselves.
        tqc_cls = TransformerTQCSharedCritic if args.tqc_share_trunk else TransformerTQCCritic
        tqc_critic = tqc_cls(
            n_critics=args.tqc_n_critics,
            n_quantiles=args.tqc_n_quantiles,
            **critic_kwargs,
        ).to(device)
        tqc_critic_target = tqc_cls(
            n_critics=args.tqc_n_critics,
            n_quantiles=args.tqc_n_quantiles,
            **critic_kwargs,
        ).to(device)
        tqc_critic_target.load_state_dict(tqc_critic.state_dict())

        # Precompute quantile midpoints: tau_i = (i + 0.5) / N
        taus = (torch.arange(args.tqc_n_quantiles, device=device).float() + 0.5) / args.tqc_n_quantiles

        tqc_params = get_critic_params_excluding_shared(tqc_critic, shared_recep_encoder, shared_influence_encoder)
        q_optimizer = make_adam(tqc_params + shared_encoder_params, lr=args.q_lr)

        actor_params = sum(p.numel() for p in actor.parameters())
        critic_params = sum(p.numel() for p in tqc_critic.parameters())
        print(f"Actor parameters: {actor_params:,}")
        shared_note = (f" [shared trunk: 1 trunk x {args.tqc_n_critics} heads]"
                       if args.tqc_share_trunk else "")
        print(f"TQC Critic parameters: {critic_params:,} ({args.tqc_n_critics} critics x {args.tqc_n_quantiles} quantiles){shared_note}")
    else:
        # SAC: standard dual-critic setup (DroQ regularization applied via critic_kwargs if enabled)
        qf1 = TransformerCritic(**critic_kwargs).to(device)
        qf2 = TransformerCritic(**critic_kwargs).to(device)
        qf1_target = TransformerCritic(**critic_kwargs).to(device)
        qf2_target = TransformerCritic(**critic_kwargs).to(device)

        qf1_target.load_state_dict(qf1.state_dict())
        qf2_target.load_state_dict(qf2.state_dict())

        qf1_params = get_critic_params_excluding_shared(qf1, shared_recep_encoder, shared_influence_encoder)
        qf2_params = get_critic_params_excluding_shared(qf2, shared_recep_encoder, shared_influence_encoder)

        q_optimizer = make_adam(
            qf1_params + qf2_params + shared_encoder_params,
            lr=args.q_lr,
        )

        actor_params = sum(p.numel() for p in actor.parameters())
        critic_params = sum(p.numel() for p in qf1.parameters())
        print(f"Actor parameters: {actor_params:,}")
        print(f"Critic parameters: {critic_params:,} (x2)")

    # Optimizers (exclude shared encoder params — handled by q_optimizer only)
    actor_optimizer = make_adam(
        [p for p in actor.parameters() if id(p) not in shared_param_ids],
        lr=args.policy_lr,
    )

    # Verify parameter counts
    if shared_recep_encoder is not None:
        actor_unique = sum(p.numel() for p in actor.parameters())
        if args.algorithm == "tqc":
            critic_unique = sum(p.numel() for p in tqc_params)
        else:
            critic_unique = sum(p.numel() for p in qf1_params)
        shared_total = sum(p.numel() for p in shared_encoder_params)
        print(f"Actor parameters (includes shared): {actor_unique:,}")
        print(f"Critic parameters (excluding shared): {critic_unique:,}")
        print(f"Shared encoder parameters (in both optimizers): {shared_total:,}")

    algo_str = args.algorithm.upper()
    if args.use_droq:
        algo_str += " + DroQ"
    print(f"Algorithm: {algo_str}")


    # Entropy tuning
    if args.autotune:
        # Initial target entropy (will be adapted per-batch)
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        # Keep alpha a detached GPU tensor (not a Python float) so the in-graph
        # loss math never forces a per-step .item() sync; materialized only at logging.
        alpha = log_alpha.exp().detach()
        alpha_optimizer = make_adam([log_alpha], lr=args.q_lr)
    else:
        alpha = torch.tensor(float(args.alpha), device=device)
        log_alpha = None
        alpha_optimizer = None
    
    # =========================================================================
    # LOAD CHECKPOINT (for fine-tuning or resuming)
    # =========================================================================
    
    start_step = 0
    if args.resume_checkpoint is not None:
        print(f"\n{'='*60}")
        print(f"LOADING CHECKPOINT FOR FINE-TUNING")
        print(f"{'='*60}")
        print(f"Checkpoint path: {args.resume_checkpoint}")
        
        if not os.path.exists(args.resume_checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {args.resume_checkpoint}")
        
        checkpoint = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)

        # Validate checkpoint matches current algorithm
        ckpt_is_tqc = "tqc_critic_state_dict" in checkpoint
        if args.algorithm == "tqc" and not ckpt_is_tqc:
            raise ValueError(
                f"--algorithm=tqc but checkpoint has no TQC critic weights. "
                f"Checkpoint was saved with algorithm={checkpoint.get('args', {}).get('algorithm', 'sac')}."
            )
        if args.algorithm != "tqc" and ckpt_is_tqc:
            raise ValueError(
                f"--algorithm={args.algorithm} but checkpoint contains TQC critic weights. "
                f"Use --algorithm=tqc to resume this checkpoint."
            )
        if ckpt_is_tqc:
            # Key probe (not saved args) so pre-tqc_share_trunk checkpoints validate:
            # shared-trunk critics have "trunk.*" keys, independent ones "critics.*".
            ckpt_shared = any(k.startswith("trunk.")
                              for k in checkpoint["tqc_critic_state_dict"])
            if ckpt_shared != args.tqc_share_trunk:
                raise ValueError(
                    f"TQC critic layout mismatch: checkpoint was saved with "
                    f"tqc_share_trunk={ckpt_shared} but current run has "
                    f"tqc_share_trunk={args.tqc_share_trunk}. Shared-trunk and "
                    f"independent TQC checkpoints are not interchangeable."
                )

        # Load network weights
        actor.load_state_dict(checkpoint["actor_state_dict"])
        if args.algorithm == "tqc":
            tqc_critic.load_state_dict(checkpoint["tqc_critic_state_dict"])
            tqc_critic_target.load_state_dict(checkpoint["tqc_critic_state_dict"])
        else:
            qf1.load_state_dict(checkpoint["qf1_state_dict"])
            qf2.load_state_dict(checkpoint["qf2_state_dict"])
            qf1_target.load_state_dict(checkpoint["qf1_state_dict"])
            qf2_target.load_state_dict(checkpoint["qf2_state_dict"])
        
        print(f"✓ Loaded network weights from step {checkpoint['step']}")
        
        # === Actor optimizer ===
        if not args.finetune_reset_actor_optimizer:
            actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
            print(f"✓ Loaded actor optimizer state")
        else:
            print(f"✓ Reset actor optimizer (fresh)")

        # === Critic optimizer ===
        if not args.finetune_reset_critic_optimizer:
            q_optimizer.load_state_dict(checkpoint["q_optimizer_state_dict"])
            print(f"✓ Loaded critic optimizer state")
        else:
            print(f"✓ Reset critic optimizer (fresh)")
        
        # === Alpha (entropy coefficient) ===
        if args.autotune:
            if not args.finetune_reset_alpha:
                if "log_alpha" in checkpoint:
                    log_alpha.data = checkpoint["log_alpha"].to(device)
                    alpha = log_alpha.exp().detach()
                    print(f"✓ Loaded entropy coefficient: alpha={float(alpha):.4f}")
                if "alpha_optimizer_state_dict" in checkpoint:
                    alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer_state_dict"])
                    print(f"✓ Loaded alpha optimizer state")
            else:
                print(f"✓ Reset entropy coefficient (alpha={float(alpha):.4f})")

        # === Obs normalizer statistics (change_wd_4, --obs_norm) ===
        if obs_normalizer is not None:
            if "obs_norm_state" in checkpoint:
                obs_normalizer.load_state_dict(checkpoint["obs_norm_state"])
                print(f"✓ Loaded obs normalizer statistics "
                      f"(count={float(obs_normalizer.count):.0f})")
            else:
                print("WARNING: --obs_norm set but checkpoint has no "
                      "obs_norm_state — statistics start cold.")

        # === Resume step logic ===
        ## REMOVED FOR SIMPLICITY
        # Only resume from checkpoint step if keeping ALL optimizer states
        # if (not args.finetune_reset_actor_optimizer and 
        #     not args.finetune_reset_critic_optimizer and
        #     not args.finetune_reset_alpha):
        #     start_step = checkpoint["step"]
        #     print(f"✓ Resuming from step {start_step}")
        # else:
        #     print(f"✓ Starting from step 0 (fine-tuning mode)")

        # === Diagnostic: Check effective learning rates ===
        print(f"\n--- Optimizer State Diagnostics ---")
        log_optimizer_effective_lr(actor_optimizer, "Actor", args.policy_lr)
        log_optimizer_effective_lr(q_optimizer, "Critic", args.q_lr)
        
        # Log checkpoint info
        if "args" in checkpoint:
            ckpt_args = checkpoint["args"]
            print(f"\nOriginal training config:")
            print(f"  - Layouts: {ckpt_args.get('layouts', 'unknown')}")
            print(f"  - Total timesteps: {ckpt_args.get('total_timesteps', 'unknown')}")
            print(f"  - Pos encoding: {ckpt_args.get('pos_encoding_type', 'unknown')}")
        
        print(f"\nFine-tuning config:")
        print(f"  - Target layouts: {args.layouts}")
        print(f"  - Reset actor optimizer: {args.finetune_reset_actor_optimizer}")
        print(f"  - Reset critic optimizer: {args.finetune_reset_critic_optimizer}")
        print(f"  - Reset alpha: {args.finetune_reset_alpha}")
        print(f"{'='*60}\n")
    
    # =========================================================================
    # LOAD PRETRAINED ENCODER (from pretrain_power.py)
    # =========================================================================

    if args.pretrain_checkpoint is not None and args.resume_checkpoint is None:
        print(f"\n{'='*60}")
        print(f"LOADING PRETRAINED ENCODER")
        print(f"{'='*60}")

        def load_pretrained_into(network, network_name, encoder_sd):
            """Load matching encoder weights into an actor or critic."""
            net_sd = network.state_dict()
            matched_keys = []
            skipped_keys = []

            for key, value in encoder_sd.items():
                if key in net_sd:
                    if net_sd[key].shape == value.shape:
                        net_sd[key] = value
                        matched_keys.append(key)
                    else:
                        skipped_keys.append(
                            f"{key} (shape: {list(value.shape)} vs {list(net_sd[key].shape)})"
                        )
                else:
                    skipped_keys.append(f"{key} (not in {network_name})")

            network.load_state_dict(net_sd)
            print(f"\n  {network_name}: loaded {len(matched_keys)}/{len(encoder_sd)} params")
            if matched_keys:
                print(f"    Matched: {matched_keys[:5]}{'...' if len(matched_keys) > 5 else ''}")
            if skipped_keys:
                print(f"    Skipped: {skipped_keys}")
            return len(matched_keys)


        # =================================================================
        # Actor loading: full state dict (BC) or encoder-only (pretrain)
        # =================================================================
        if _pretrain_actor_sd is not None:
            # BC checkpoint → load full actor including fc_mean/fc_logstd
            # BUT preserve action_scale and action_bias_val from the env
            # (they should match, but this is defensive)
            env_action_scale = actor.action_scale.clone()
            env_action_bias = actor.action_bias_val.clone()

            # Flexible load: match what we can, skip shape mismatches
            net_sd = actor.state_dict()
            matched_keys = []
            skipped_keys = []
            for key, value in _pretrain_actor_sd.items():
                if key in net_sd:
                    if net_sd[key].shape == value.shape:
                        net_sd[key] = value
                        matched_keys.append(key)
                    else:
                        skipped_keys.append(
                            f"{key} (shape: {list(value.shape)} vs {list(net_sd[key].shape)})"
                        )
                else:
                    skipped_keys.append(f"{key} (not in Actor)")
            actor.load_state_dict(net_sd)

            # Restore env-derived action scaling (in case BC used different defaults)
            actor.action_scale.copy_(env_action_scale)
            actor.action_bias_val.copy_(env_action_bias)

            print(f"\n  Actor (BC full load): loaded {len(matched_keys)}/{len(_pretrain_actor_sd)} params")
            if matched_keys:
                print(f"    Matched: {matched_keys[:8]}{'...' if len(matched_keys) > 8 else ''}")
            if skipped_keys:
                print(f"    Skipped: {skipped_keys}")
            n_actor = len(matched_keys)
        else:
            # Self-supervised pretrain → encoder-only loading
            n_actor = load_pretrained_into(actor, "Actor", _pretrain_encoder_sd)

        # Critics always get encoder-only loading (obs_action_encoder input dim differs)
        if args.algorithm == "tqc":
            if args.tqc_share_trunk:
                load_pretrained_into(tqc_critic.trunk, "TQC shared trunk", _pretrain_encoder_sd)
            else:
                for i, critic in enumerate(tqc_critic.critics):
                    load_pretrained_into(critic, f"TQC Critic {i}", _pretrain_encoder_sd)
            tqc_critic_target.load_state_dict(tqc_critic.state_dict())
        else:
            n_qf1 = load_pretrained_into(qf1, "Critic qf1", _pretrain_encoder_sd)
            n_qf2 = load_pretrained_into(qf2, "Critic qf2", _pretrain_encoder_sd)
            qf1_target.load_state_dict(qf1.state_dict())
            qf2_target.load_state_dict(qf2.state_dict())
        print(f"\n  Target networks synced ✓")

        if n_actor == 0:
            print(f"\n  ⚠ WARNING: No weights matched! Something is wrong.")

        # Optional: freeze encoder initially
        if args.pretrain_freeze_steps > 0:
            frozen = []
            for name, param in actor.named_parameters():
                if "fc_mean" not in name and "fc_logstd" not in name:
                    param.requires_grad = False
                    frozen.append(name)
            actor_optimizer = make_adam(
                [p for p in actor.parameters() if p.requires_grad and id(p) not in shared_param_ids],
                lr=args.policy_lr,
            )
            print(f"\n  Froze {len(frozen)} encoder params for {args.pretrain_freeze_steps} steps")

        del _pretrain_encoder_sd  # clean up
        if _pretrain_actor_sd is not None:
            del _pretrain_actor_sd
        print(f"{'='*60}\n")
    
    
    
    # =========================================================================
    # REPLAY BUFFER
    # =========================================================================

    rb = TransformerReplayBuffer(
        capacity=args.buffer_size,
        device=device,
        rotor_diameter=rotor_diameter,
        max_turbines=n_turbines_max,
        obs_dim=obs_dim_per_turbine,
        action_dim=action_dim_per_turbine,
        use_wind_relative=args.use_wind_relative_pos,
        use_profiles=use_profiles,
        rotate_profiles=args.rotate_profiles,
        profile_registry=profile_registry,
        profile_registry_gpu_budget_mb=args.profile_registry_gpu_budget_mb,
    )

    # Shared metadata stored alongside every buffer save (also validated on load)
    def buffer_meta(step: int) -> dict:
        return {
            "layouts": args.layouts,
            "seed": args.seed,
            "global_step": step,
            "history_length": args.history_length,
        }

    if args.load_buffer is not None:
        print(f"\n{'='*60}")
        print(f"LOADING REPLAY BUFFER")
        print(f"{'='*60}")
        buffer_meta_loaded = rb.load(args.load_buffer)

        if buffer_meta_loaded.get("layouts") != args.layouts:
            raise ValueError(
                f"Replay buffer was generated with layouts='{buffer_meta_loaded.get('layouts')}' "
                f"but this run uses layouts='{args.layouts}'. Stored positions and "
                f"layout indices would be inconsistent."
            )
        if buffer_meta_loaded.get("history_length") != args.history_length:
            raise ValueError(
                f"Replay buffer was generated with history_length="
                f"{buffer_meta_loaded.get('history_length')} but this run uses "
                f"{args.history_length}. Observation contents would be inconsistent."
            )
        if buffer_meta_loaded.get("seed") != args.seed:
            print(f"NOTE: buffer was generated with seed={buffer_meta_loaded.get('seed')}, "
                  f"this run uses seed={args.seed}.")

        if args.learning_starts > 0:
            print(f"Loaded {len(rb)} transitions; skipping exploration phase "
                  f"(learning_starts: {args.learning_starts} -> 0)")
            args.learning_starts = 0
        print(f"{'='*60}\n")


    # =========================================================================
    # TRAINING LOOP
    # =========================================================================
    
    # =========================================================================
    # torch.compile (optional) — compile the network forward passes.
    # We replace each module's .forward (not the module object) so state_dict
    # keys are unchanged (checkpoints stay compatible) and the actor's
    # get_action(self.forward(...)) hot path also hits the compiled graph.
    # Shapes are static (padded max_turbines, fixed batch_size); the rare
    # logging/eval calls with other batch sizes just trigger a one-time recompile.
    # =========================================================================
    if args.compile:
        print("Compiling network forward passes with torch.compile (first steps are slow)...")

        def _compile_forward(module):
            if module is not None:
                # reduce-overhead (CUDA graphs) collapses the many tiny kernel launches
                # of this small/short-sequence model into one replay -- the dominant
                # cost for a launch-bound update loop. Requires the hot loop to be
                # sync-free (see GPU-side loss accumulation) and the held critic
                # outputs to be .clone()'d (shared CUDA-graph buffer pool).
                # NOTE: must be consistent across ALL compiled forwards -- mixing
                # reduce-overhead with a plain-compiled module corrupts the
                # CUDA-graph-trees allocator. (A vmap-ensembled critic was tried and
                # was ~2x SLOWER here, so we keep separate critics.)
                # mode is args.compile_mode for ALL forwards (consistent within a run);
                # "default" disables cudagraphs (single-rose arms; see config.compile_mode).
                module.forward = torch.compile(module.forward, mode=args.compile_mode)

        _compile_forward(actor)
        if args.algorithm == "tqc":
            _compile_forward(tqc_critic)
            _compile_forward(tqc_critic_target)
        else:
            _compile_forward(qf1)
            _compile_forward(qf2)
            _compile_forward(qf1_target)
            _compile_forward(qf2_target)

    print(f"\nStarting training for {args.total_timesteps} timesteps...")
    print(f"UTD ratio: {args.utd_ratio} (gradient updates per env step)")
    print(f"With {args.num_envs} envs: {int(args.num_envs * args.utd_ratio)} gradient updates per iteration")
    print("=" * 60)
    

    save_checkpoint(
        actor, qf1, qf2, actor_optimizer, q_optimizer,
        0, run_name, args, log_alpha, alpha_optimizer,
        tqc_critic=tqc_critic,
        obs_norm_state=obs_normalizer.state_dict() if obs_normalizer is not None else None,
    )


    # Track evaluation timing
    next_eval_step = args.eval_interval
    
    # Initial evaluation
    if args.eval_initial:
        print("\nRunning initial evaluation before training...")
        eval_dict, eval_metrics = run_all_evaluations()

        for name, value in eval_dict.items():
            writer.add_scalar(name, value, 0)

        print(f"Initial eval - Mean reward: {eval_metrics.mean_reward:.4f}, "
              f"Power ratio: {eval_metrics.power_ratio:.4f}")


    # DroQ: target networks must be in eval mode to disable dropout
    if args.use_droq:
        if tqc_critic_target is not None:
            tqc_critic_target.eval()
        if qf1_target is not None:
            qf1_target.eval()
        if qf2_target is not None:
            qf2_target.eval()

    start_time = time.time()
    global_step = start_step  # Start from checkpoint step if resuming, else 0
    total_gradient_steps = 0  # Track total gradient updates for logging

    # Wall-clock breakdown (only populated when --log_timing). Accumulated over a
    # logging window and reset on each flush. Syncs are gated so they add no
    # overhead when timing is off.
    # "env" = time BLOCKED in step_wait (name kept for dashboard continuity);
    # "env_span" = wall-clock from step_async to step_wait return, i.e. the
    # full env-step duration whether or not it was hidden behind the gradient
    # burst (--async_overlap). hidden time = env_span - env.
    timing = {"env": 0.0, "env_span": 0.0, "sample": 0.0, "critic": 0.0, "actor": 0.0}

    # -------------------------------------------------------------------------
    # perf/* : cheap, always-on resource accounting.
    #
    # On a cluster billed by GPU-hour the question that matters is whether the
    # GPU sits idle while the DWM env workers churn -- which is what sets both
    # the wall clock and the right --cpus-per-task. torch's own counters cover
    # the GPU; for CPU and memory we read the job's cgroup rather than this
    # process, because the AsyncVectorEnv workers are separate processes and
    # os.times()/RSS of the parent would miss all of their work.
    #
    # Preferred source is the job's cgroup, but /sys/fs/cgroup is NOT populated
    # inside LUMI's Singularity container (measured: cpu.stat and
    # memory.current are both absent), so fall back to walking /proc and
    # summing over our PROCESS GROUP -- multiprocessing does not setpgrp, so
    # the AsyncVectorEnv workers share our pgrp and are counted. Everything is
    # best-effort: if both sources fail the metric is simply not logged.
    _CLK_TCK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100

    def _proc_stat_fields(pid="self"):
        """Fields 3.. of /proc/<pid>/stat, 0-indexed so field N is at N-3.

        Splits on the LAST ')': the comm field is parenthesised and may itself
        contain spaces and parentheses (e.g. "(python3.12 (deleted))"), which
        would misalign a naive .split().
        """
        with open(f"/proc/{pid}/stat") as fh:
            return fh.read().rsplit(")", 1)[1].split()

    def _read_cgroup_cpu_usec():
        try:
            with open("/sys/fs/cgroup/cpu.stat") as fh:
                for line in fh:
                    if line.startswith("usage_usec"):
                        return int(line.split()[1])
        except (OSError, ValueError, IndexError):
            pass
        # Fallback: utime+stime (fields 14,15) over our process group (field 5).
        try:
            pgrp = _proc_stat_fields()[2]
        except (OSError, IndexError):
            return None
        ticks = 0
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            try:
                f = _proc_stat_fields(entry)
                if f[2] != pgrp:
                    continue
                ticks += int(f[11]) + int(f[12])
            except (OSError, IndexError, ValueError):
                continue          # process exited mid-scan, or unreadable
        return int(ticks / _CLK_TCK * 1e6)

    def _read_cgroup_mem_bytes():
        out = {}
        for key, path in (("cur", "/sys/fs/cgroup/memory.current"),
                          ("peak", "/sys/fs/cgroup/memory.peak")):
            try:
                with open(path) as fh:
                    out[key] = int(fh.read().strip())
            except (OSError, ValueError):
                pass
        if "cur" not in out:
            # Fallback: summed RSS (field 24) over the process group. Shared
            # pages are counted once per process, so this OVERSTATES real usage
            # -- it is a trend line for --mem sizing, not an accounting figure.
            try:
                pgrp = _proc_stat_fields()[2]
            except (OSError, IndexError):
                return out
            pages = 0
            for entry in os.listdir("/proc"):
                if not entry.isdigit():
                    continue
                try:
                    f = _proc_stat_fields(entry)
                    if f[2] != pgrp:
                        continue
                    pages += int(f[21])
                except (OSError, IndexError, ValueError):
                    continue
            out["rss_sum"] = pages * os.sysconf("SC_PAGE_SIZE")
        return out

    # Number of CPUs this job may actually use, so cpu_util is a percentage of
    # the allocation rather than of the whole 128-core node.
    try:
        _n_cpu_alloc = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        _n_cpu_alloc = os.cpu_count() or 1

    _perf_prev = {"t": time.time(), "cpu_usec": _read_cgroup_cpu_usec(),
                  "step": global_step}

    # Always-on env-vs-update wall-clock split, in HOST time with NO cuda sync.
    #
    # timing/* already reports a sync-accurate breakdown, but it costs a
    # torch.cuda.synchronize() per bucket per gradient step, which is why it
    # sits behind --log_timing -- and the production run scripts do not pass
    # that flag. These two counters are four perf_counter() calls per iteration,
    # so they can stay on always, and for this launch-bound update loop host
    # time is the number that actually answers "is the GPU waiting on 30 DWM
    # env workers?". Use --log_timing when you need GPU-inclusive attribution.
    _perf_split = {"env": 0.0, "update": 0.0}

    # -------------------------------------------------------------------------
    # actions/* : exact mean/std/saturation per channel over each logging window.
    #
    # Running sums (not a deque of raw arrays) so the window statistics are
    # exact rather than an average-of-per-step-averages, and so memory does not
    # scale with num_envs * n_turbines.
    #
    # Actions live in the normalized [-1, 1] space of the wrapper's Box. Under
    # the "yaw"/"step" action methods that value IS the fraction of the per-step
    # slew budget, so |a| ~ 1 means the turbine is moving at its rate limit --
    # a persistently high sat_frac says the slew limit, not the policy, is what
    # is shaping behaviour.
    _act_acc = {f"{ch}_{stat}": 0.0
                for ch in _act_cols for stat in ("sum", "sq", "sat")}
    _act_acc["n"] = 0.0

    def _accumulate_actions(act, masks):
        """Fold one step's actions into _act_acc. Cheap: ~num_envs*n_turb floats."""
        a = np.asarray(act, dtype=np.float64)
        if a.ndim == 2:                       # (num_envs, n_turb) -> single channel
            a = a[..., None]
        if a.ndim != 3:
            return
        # masks mark PADDED turbines (True = padding), matching the convention
        # in the actor/critic (n_real = (~attention_mask).sum()). Excluding them
        # keeps padded slots from dragging every statistic toward zero.
        keep = None
        if masks is not None:
            m = np.asarray(masks)
            if m.shape == a.shape[:2]:
                keep = ~m.astype(bool)
        for ch, col in _act_cols.items():
            if col >= a.shape[2]:
                continue
            v = a[:, :, col]
            v = v[keep] if keep is not None else v.reshape(-1)
            if v.size == 0:
                continue
            _act_acc[f"{ch}_sum"] += float(v.sum())
            _act_acc[f"{ch}_sq"] += float(np.square(v).sum())
            _act_acc[f"{ch}_sat"] += float((np.abs(v) >= 0.99).sum())
            if ch == next(iter(_act_cols)):
                _act_acc["n"] += float(v.size)

    def _sync_timer():
        """Return perf_counter, syncing CUDA first so GPU work is included."""
        if args.log_timing and device.type == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter()

    # AMP autocast context (bf16). Reused across all update forward passes; a
    # no-op when --amp is off. bf16 keeps fp32 range so no GradScaler is needed.
    amp_ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.amp)
    if args.amp:
        print(f"AMP enabled: bfloat16 autocast on {device.type}")
        print("  NOTE: on Sophia (Quadro RTX 4000 / Turing) the --amp flag is "
              "strictly slower -- Turing has no bf16 tensor cores. Disable it here.")
    # Reset environments
    obs, infos = envs.reset(seed=args.seed)
    
    # Tracking
    step_reward_window = deque(maxlen=1000)
    # DEL-penalty diagnostics (filled only when the DELRewardWrapper is
    # attached: --del_penalty_scale > 0 or --del_log). ratio/max keys are NaN
    # during each episode's ti_window warm-up; only finite values are
    # collected so the logged means reflect evaluated steps.
    del_penalty_window = deque(maxlen=1000)
    del_ratio_window = deque(maxlen=1000)
    del_agent_max_window = deque(maxlen=1000)
    del_base_max_window = deque(maxlen=1000)
    reward_unpen_window = deque(maxlen=1000)
    del_ood_window = deque(maxlen=1000)
    # Goal-conditioned limit diagnostics: the limit in effect (varies across
    # episodes under --del_limit_random) and the budget margin
    # (1 + limit) - ratio; margin > 0 <=> inside the DEL budget, so its mean
    # is the constraint-satisfaction signal across the sampled-limit
    # distribution. Margin is NaN wherever ratio is (finite-filtered below).
    del_limit_window = deque(maxlen=1000)
    del_margin_window = deque(maxlen=1000)
    # Multi-channel penalty: which reward channel realized the binding (max)
    # ratio each step -> charts/del_binding_frac/<channel>. Only logged when
    # more than one channel is configured (single-channel: trivially 1.0).
    # Canonicalized ("wtow_H0FAMnt" -> "H0FAMnt") to match the names the
    # wrapper reports in info["del_binding_channel"].
    del_binding_window = deque(maxlen=1000)
    _del_reward_channels = []
    if _del_active:
        from helpers.load_reward import load_reward_channels
        _del_reward_channels = load_reward_channels(args)

    # del/*: per-channel load ratios. charts/del_agent_max collapses every
    # channel into a single worst-case number, which tells you a limit is being
    # approached but not WHICH load is doing it -- so the load story is
    # unreadable from the charts alone. Keyed by channel name, created lazily
    # from whatever info["del_ratio_by_channel"] actually reports.
    del_channel_windows = {}
    # next_save_step = ((start_step // args.save_interval) + 1) * args.save_interval  # Account for resumed step
    next_save_step = start_step + args.save_interval
    # Replay buffer saving
    warmup_buffer_saved = False
    next_buffer_save_step = start_step + args.buffer_save_interval
    # For logging losses: accumulate GPU-side running sums and materialize once per
    # logging drain. Avoids a per-grad-step .item() sync that would serialize this
    # launch-bound update loop. Counts track how many updates contributed each metric.
    _qf_keys = ['qf_loss'] if args.algorithm == "tqc" else ['qf1_loss', 'qf2_loss']
    # q_mean / td_* / *_gnorm / target_entropy feed the losses/* panel. They are
    # accumulated on-GPU like the rest and materialized once per logging window,
    # so they cost no extra host syncs in the launch-bound update loop.
    # td_abs/td_sq are SAC-only: TQC regresses a quantile distribution, so there
    # is no single scalar TD error to spread.
    _acc_keys = _qf_keys + ['actor_loss', 'alpha_loss',
                            'logpi_per_turbine', 'ent_term', 'q_term', 'n_real_mean',
                            'q_mean', 'td_abs', 'td_sq',
                            'critic_gnorm', 'actor_gnorm', 'target_entropy']

    def _zero_losses():
        return {k: torch.zeros((), device=device) for k in _acc_keys}

    loss_accumulator = _zero_losses()
    n_critic_updates = 0
    n_actor_updates = 0

    # Calculate remaining updates if resuming
    remaining_timesteps = args.total_timesteps - start_step
    num_updates = max(0, remaining_timesteps // args.num_envs)
    
    if start_step > 0:
        print(f"Resuming from step {start_step}, {remaining_timesteps} timesteps remaining")
        print(f"Will run {num_updates} more updates")
    
    for update in range(num_updates + 2):
        global_step += args.num_envs
        
        # Unfreeze pretrained encoder after warmup
        if (args.pretrain_checkpoint is not None 
            and args.pretrain_freeze_steps > 0 
            and global_step >= args.pretrain_freeze_steps
            and global_step - args.num_envs < args.pretrain_freeze_steps):
            for name, param in actor.named_parameters():
                param.requires_grad = True
            actor_optimizer = make_adam(
                [p for p in actor.parameters() if id(p) not in shared_param_ids],
                lr=args.policy_lr,
            )
            print(f"\n[Step {global_step}] Unfroze pretrained encoder parameters")
        
        # Get environment info (needed for replay buffer). Under
        # --wd_source est this fetches wd_est, so the buffer stores (and the
        # policy trains on) the estimate — buffer honesty is automatic.
        wind_dirs = get_env_wind_directions(envs, attr=wd_source_attr)
        raw_positions = get_env_raw_positions(envs)
        current_masks = get_env_attention_masks(envs)

        # Get layout identifiers for replay buffer (lightweight)
        if args.profile_encoding_type is not None:
            current_layout_indices = get_env_layout_indices(envs)
            current_permutations = get_env_permutations(envs)
            # Prefetch the profiles too so agent.act does no get_attr of its
            # own: ALL env IPC must happen before step_async (--async_overlap
            # forbids IPC while a step is in flight).
            current_receptivity = get_env_receptivity_profiles(envs)
            current_influence = get_env_influence_profiles(envs)
        else:
            current_layout_indices = None
            current_permutations = None
            current_receptivity = None
            current_influence = None


        # Select action
        # Reuse the env state already fetched above to avoid duplicate get_attr IPC
        act_state = dict(wind_dirs=wind_dirs, raw_positions=raw_positions, masks=current_masks,
                        receptivity=current_receptivity, influence=current_influence)
        if global_step < args.learning_starts:
            if args.initial_exploration == "policy":
                # Use the actor network (useful when resuming from checkpoint)
                with torch.no_grad():
                    actions = agent.act(envs, obs, **act_state)
            else:
                # Random exploration (default for training from scratch)
                actions = envs.action_space.sample()
        else:
            with torch.no_grad():
                actions = agent.act(envs, obs, **act_state)

        # actions/* accounting. Before step_async, but it touches no env IPC --
        # current_masks was prefetched above -- so it is safe under
        # --async_overlap.
        _accumulate_actions(actions, current_masks)

        # Step environment. The step is always dispatched async; the flag only
        # picks the blocking point. --async_overlap collects the result AFTER
        # the gradient burst (env workers simulate while the GPU trains);
        # otherwise we block right here, giving the same compute order as the
        # old blocking envs.step().
        _t_async = time.perf_counter()
        envs.step_async(actions)
        step_result = None
        if not args.async_overlap:
            _t0 = _sync_timer()
            _t_env0 = time.perf_counter()
            step_result = envs.step_wait()
            _perf_split["env"] += time.perf_counter() - _t_env0
            if args.log_timing:
                _t_now = time.perf_counter()
                timing["env"] += _t_now - _t0
                timing["env_span"] += _t_now - _t_async

        # change_wd_4: fold the fresh observations into the running obs stats.
        # Masked so 0.0-pad rows don't drag the means; runs from step 0 (the
        # random-exploration warmup is already on-distribution).
        if obs_normalizer is not None:
            obs_normalizer.update(
                torch.as_tensor(next_obs, dtype=torch.float32, device=device),
                torch.as_tensor(current_masks, device=device),
            )


        # =====================================================================
        # TRAINING
        # =====================================================================
        # NOTE (--async_overlap): an env step may be IN FLIGHT throughout this
        # block -- do not add env IPC (get_attr / call / step) anywhere in the
        # training region or AsyncVectorEnv raises AlreadyPendingCallError.
        # envs.return_queue reads below are wrapper-local deques: safe, but one
        # iteration stale (cosmetic only). The replay buffer excludes the
        # in-flight transitions (rb.add runs after step_wait, below); at the
        # first gated iteration len(rb) ~ learning_starts >> batch_size, so at
        # worst the first burst happens one iteration later.

        if global_step > args.learning_starts and len(rb) >= args.batch_size:
            _t_upd0 = time.perf_counter()

            # Calculate number of gradient updates for this iteration
            # This scales with num_envs to maintain consistent sample efficiency
            num_gradient_updates = max(1, int(args.num_envs * args.utd_ratio))
            
            # Clear loss accumulators for this iteration
            loss_accumulator = _zero_losses()
            n_critic_updates = 0
            n_actor_updates = 0


            for grad_step in range(num_gradient_updates):
                # Sample a fresh batch for each gradient update
                _t0 = _sync_timer()
                data = rb.sample(args.batch_size)
                if args.log_timing:
                    timing["sample"] += _sync_timer() - _t0

                # change_wd_4: the buffer stores RAW obs; normalizing at the
                # single sample point covers critic, actor and every diagnostic
                # that reuses `data`, always with the freshest statistics.
                if obs_normalizer is not None:
                    data["observations"] = obs_normalizer.normalize(data["observations"])
                    data["next_observations"] = obs_normalizer.normalize(data["next_observations"])

                _t_critic = _sync_timer()

                batch_mask = data["attention_mask"]
                
                # Get profiles from batch (will be None if not using profiles)
                batch_receptivity = data.get("receptivity", None)
                batch_influence = data.get("influence", None)
                # Single-rose mode: drop the unused influence tensor so it is never a live
                # input to the compiled critic/actor forwards (phantom cudagraph input).
                if not actor.use_influence:
                    batch_influence = None

                # -----------------------------------------------------------------
                # Update Critics
                # -----------------------------------------------------------------
                with torch.no_grad(), amp_ctx:
                    # Get next actions from current policy
                    next_actions, next_log_pi, _, _ = actor.get_action(
                        data["next_observations"],
                        data["positions"],
                        batch_mask,
                        recep_profile=batch_receptivity,
                        influence_profile=batch_influence,
                    )

                if args.algorithm == "tqc":
                    # --- TQC critic update ---
                    with torch.no_grad(), amp_ctx:
                        # Target quantiles: (n_critics, batch, n_quantiles)
                        target_quantiles = tqc_critic_target(
                            data["next_observations"], next_actions,
                            data["positions"], batch_mask,
                            recep_profile=batch_receptivity,
                            influence_profile=batch_influence,
                        )
                        batch_size_cur = data["rewards"].shape[0]
                        # Flatten across critics, sort, truncate top-d (fp32 for the target math)
                        all_target_q = target_quantiles.float().permute(1, 0, 2).reshape(batch_size_cur, -1)
                        sorted_q, _ = all_target_q.sort(dim=1)
                        n_keep = args.tqc_n_critics * args.tqc_n_quantiles - args.tqc_top_quantiles_to_drop
                        truncated_mean = sorted_q[:, :n_keep].mean(dim=1, keepdim=True)
                        target_q = data["rewards"] + (1 - data["dones"]) * args.gamma * (truncated_mean - alpha * next_log_pi)

                    # Current quantiles: (n_critics, batch, n_quantiles)
                    with amp_ctx:
                        current_q = tqc_critic(
                            data["observations"], data["actions"],
                            data["positions"], batch_mask,
                            recep_profile=batch_receptivity,
                            influence_profile=batch_influence,
                        )
                        qf_loss = sum(
                            quantile_huber_loss(current_q[i].float(), target_q, taus)
                            for i in range(args.tqc_n_critics)
                        )

                    q_optimizer.zero_grad(set_to_none=True)
                    qf_loss.backward()
                    # max_norm=inf makes this a pure measurement when clipping
                    # is off (the clip coefficient is clamped to 1.0), so
                    # losses/critic_grad_norm is populated on this branch too
                    # rather than silently logging 0.
                    _critic_gnorm = torch.nn.utils.clip_grad_norm_(
                        tqc_critic.parameters(),
                        max_norm=args.grad_clip_max_norm if args.grad_clip else float("inf"),
                    )
                    loss_accumulator['critic_gnorm'] += _critic_gnorm.detach()
                    # Q level for TQC: mean over the predicted quantiles. No
                    # td_* here -- quantile regression has no single scalar TD
                    # error to take a spread of.
                    with torch.no_grad():
                        loss_accumulator['q_mean'] += current_q.detach().float().mean()
                    q_optimizer.step()

                    if debug_logger.should_log_gradients(total_gradient_steps):
                        if args.tqc_share_trunk:
                            grad_norm = sum(
                                p.grad.norm().item() ** 2
                                for p in tqc_critic.trunk.parameters() if p.grad is not None
                            ) ** 0.5
                            writer.add_scalar("debug/grad_norm/tqc_trunk", grad_norm, global_step)
                            for i, head in enumerate(tqc_critic.heads):
                                grad_norm = sum(
                                    p.grad.norm().item() ** 2
                                    for p in head.parameters() if p.grad is not None
                                ) ** 0.5
                                writer.add_scalar(f"debug/grad_norm/tqc_head_{i}", grad_norm, global_step)
                        else:
                            for i, critic in enumerate(tqc_critic.critics):
                                grad_norm = sum(
                                    p.grad.norm().item() ** 2
                                    for p in critic.parameters() if p.grad is not None
                                ) ** 0.5
                                writer.add_scalar(f"debug/grad_norm/tqc_critic_{i}", grad_norm, global_step)

                    loss_accumulator['qf_loss'] += qf_loss.detach()
                    n_critic_updates += 1
                else:
                    # --- SAC critic update ---
                    with torch.no_grad(), amp_ctx:
                        # .clone() each critic output: under reduce-overhead the compiled
                        # critics share one CUDA-graph buffer pool, so a held output is
                        # overwritten by the next critic call before it is consumed.
                        qf1_next = qf1_target(
                            data["next_observations"], next_actions,
                            data["positions"], batch_mask,
                            recep_profile=batch_receptivity,
                            influence_profile=batch_influence,
                        ).clone()
                        qf2_next = qf2_target(
                            data["next_observations"], next_actions,
                            data["positions"], batch_mask,
                            recep_profile=batch_receptivity,
                            influence_profile=batch_influence,
                        ).clone()
                        min_qf_next = torch.min(qf1_next, qf2_next) - alpha * next_log_pi
                        target_q = data["rewards"] + (1 - data["dones"]) * args.gamma * min_qf_next

                    with amp_ctx:
                        # .clone(): see target-critic note above (shared CUDA-graph pool).
                        qf1_value = qf1(data["observations"], data["actions"],
                                        data["positions"], batch_mask,
                                        recep_profile=batch_receptivity,
                                        influence_profile=batch_influence).clone()
                        qf2_value = qf2(data["observations"], data["actions"],
                                        data["positions"], batch_mask,
                                        recep_profile=batch_receptivity,
                                        influence_profile=batch_influence).clone()

                        if debug_logger.should_log_q_values(total_gradient_steps):
                            debug_logger.log_q_value_stats(
                                qf1_values=qf1_value,
                                qf2_values=qf2_value,
                                target_q=target_q,
                                writer=writer,
                                global_step=global_step,
                            )

                        # Losses in fp32 for stability
                        qf1_loss = F.mse_loss(qf1_value.float(), target_q)
                        qf2_loss = F.mse_loss(qf2_value.float(), target_q)
                        qf_loss = qf1_loss + qf2_loss

                    q_optimizer.zero_grad(set_to_none=True)
                    qf_loss.backward()
                    # clip_grad_norm_ RETURNS the pre-clip total norm, so this
                    # is free when clipping is on; with max_norm=inf the clip
                    # coefficient is clamped to 1.0, making it a pure
                    # measurement. A rising critic grad norm alongside a rising
                    # qf_loss is divergence rather than a hard task.
                    _critic_gnorm = torch.nn.utils.clip_grad_norm_(
                        qf1_params + qf2_params + shared_encoder_params,
                        max_norm=args.grad_clip_max_norm if args.grad_clip else float("inf"),
                    )
                    q_optimizer.step()

                    if debug_logger.should_log_gradients(total_gradient_steps):
                        debug_logger.log_critic_gradient_norms(qf1, qf2, writer, global_step)

                    loss_accumulator['qf1_loss'] += qf1_loss.detach()
                    loss_accumulator['qf2_loss'] += qf2_loss.detach()
                    loss_accumulator['critic_gnorm'] += _critic_gnorm.detach()
                    with torch.no_grad():
                        # Q level and TD-error spread. A td_std that grows while
                        # td_abs stays flat means a few transitions dominate the
                        # loss -- the usual precursor to Q-value blow-up.
                        _td = (qf1_value.detach().float() - target_q.float())
                        loss_accumulator['q_mean'] += qf1_value.detach().float().mean()
                        loss_accumulator['td_abs'] += _td.abs().mean()
                        loss_accumulator['td_sq'] += _td.pow(2).mean()
                    n_critic_updates += 1

                if args.log_timing:
                    timing["critic"] += _sync_timer() - _t_critic

                # -----------------------------------------------------------------
                # Update Actor (delayed based on total gradient steps)
                # -----------------------------------------------------------------
                if total_gradient_steps % args.policy_frequency == 0:
                    _t_actor = _sync_timer()
                    # Get actions from current policy + Q-values (under AMP autocast)
                    with amp_ctx:
                        actions_pi, log_pi, _, _ = actor.get_action(
                            data["observations"], data["positions"], batch_mask,
                            recep_profile=batch_receptivity,
                            influence_profile=batch_influence,
                        )

                        # Q-values for policy actions
                        if args.algorithm == "tqc":
                            all_q = tqc_critic(
                                data["observations"], actions_pi,
                                data["positions"], batch_mask,
                                recep_profile=batch_receptivity,
                                influence_profile=batch_influence,
                            )  # (n_critics, batch, n_quantiles)
                            batch_size_cur = data["rewards"].shape[0]
                            all_q_flat = all_q.permute(1, 0, 2).reshape(batch_size_cur, -1)
                            sorted_q, _ = all_q_flat.sort(dim=1)
                            n_keep = args.tqc_n_critics * args.tqc_n_quantiles - args.tqc_top_quantiles_to_drop
                            min_qf_pi = sorted_q[:, :n_keep].mean(dim=1, keepdim=True)
                        else:
                            # .clone(): see target-critic note above (shared CUDA-graph pool).
                            qf1_pi = qf1(data["observations"], actions_pi, data["positions"],
                                         batch_mask,
                                         recep_profile=batch_receptivity,
                                         influence_profile=batch_influence).clone()
                            qf2_pi = qf2(data["observations"], actions_pi, data["positions"],
                                         batch_mask,
                                         recep_profile=batch_receptivity,
                                         influence_profile=batch_influence).clone()
                            min_qf_pi = torch.min(qf1_pi, qf2_pi)

                        # Policy loss (maximize Q - alpha * entropy), fp32 for stability
                        actor_loss = (alpha * log_pi.float() - min_qf_pi.float()).mean()

                    # Update actor
                    actor_optimizer.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    # See the critic note: max_norm=inf makes this a pure
                    # measurement when --grad_clip is off.
                    _actor_gnorm = torch.nn.utils.clip_grad_norm_(
                        actor.parameters(),
                        max_norm=args.grad_clip_max_norm if args.grad_clip else float("inf"),
                    )
                    loss_accumulator['actor_gnorm'] += _actor_gnorm.detach()
                    actor_optimizer.step()

                    if debug_logger.should_log_gradients(total_gradient_steps):
                        debug_logger.log_actor_gradient_norms(actor, writer, global_step)

                    loss_accumulator['actor_loss'] += actor_loss.detach()
                    n_actor_updates += 1

                    # Entropy-vs-Q diagnostics (the large-farm "stay diffuse" pathology):
                    # per-turbine log-prob is size-invariant so it is comparable across N,
                    # and entropy_to_q_ratio shows whether the entropy term swamps the Q
                    # term in the actor loss as farm size grows.
                    with torch.no_grad():
                        n_real_b = (~data["attention_mask"]).sum(dim=1).float().clamp(min=1.0)
                        logpi_agg = log_pi.detach().float().squeeze(-1)
                        logpi_per_turb = (logpi_agg if args.entropy_agg == "mean"
                                          else logpi_agg / n_real_b)
                        loss_accumulator['logpi_per_turbine'] += logpi_per_turb.mean()
                        loss_accumulator['ent_term'] += (alpha * logpi_agg).mean()
                        loss_accumulator['q_term'] += min_qf_pi.detach().float().mean()
                        loss_accumulator['n_real_mean'] += n_real_b.mean()

                    # -------------------------------------------------------------
                    # Update Alpha (entropy coefficient)
                    # -------------------------------------------------------------
                    if args.autotune:
                        log_pi_detached = log_pi.detach()
                        
                        # Adaptive target entropy per sample (matched to actor.entropy_agg)
                        target_entropy_batch = compute_adaptive_target_entropy(
                            data["attention_mask"],
                            action_dim_per_turbine,
                            agg=args.entropy_agg,
                        )
                        
                        # Alpha loss
                        alpha_loss = (-log_alpha.exp() * (log_pi_detached + target_entropy_batch)).mean()
                        # Target entropy is adaptive (it depends on how many
                        # real turbines are in each batch element), so logging
                        # it alongside the achieved entropy is the only way to
                        # read whether alpha is tracking or saturated.
                        loss_accumulator['target_entropy'] += target_entropy_batch.detach().float().mean()
                        
                        alpha_optimizer.zero_grad(set_to_none=True)
                        alpha_loss.backward()
                        alpha_optimizer.step()
                        alpha = log_alpha.exp().detach()
                        
                        loss_accumulator['alpha_loss'] += alpha_loss.detach()

                    if args.log_timing:
                        timing["actor"] += _sync_timer() - _t_actor

                # -----------------------------------------------------------------
                # Update Target Networks
                # -----------------------------------------------------------------
                if total_gradient_steps % args.target_network_frequency == 0:
                    if args.algorithm == "tqc":
                        soft_update(tqc_critic, tqc_critic_target, args.tau)
                    else:
                        soft_update(qf1, qf1_target, args.tau)
                        soft_update(qf2, qf2_target, args.tau)
                
                # Attention physics analysis (frequency controlled by logger)
                if debug_logger.should_log_attention(total_gradient_steps):
                    with torch.no_grad():
                        # Get fresh attention weights from a small batch
                        sample_size = min(8, args.batch_size)
                        _, _, _, attn_weights = actor.get_action(
                            data["observations"][:sample_size],
                            data["positions"][:sample_size],
                            batch_mask[:sample_size] if batch_mask is not None else None,
                            recep_profile=batch_receptivity[:sample_size] if batch_receptivity is not None else None,
                            influence_profile=batch_influence[:sample_size] if batch_influence is not None else None,
                            need_weights=True, # Need this if we actually want attention
                        )
                        
                        # This logs both scalar metrics AND a visualization image!
                        debug_logger.log_attention_metrics(
                            attention_weights=attn_weights,
                            positions=data["positions"][:sample_size],
                            attention_mask=batch_mask[:sample_size] if batch_mask is not None else None,
                            writer=writer,
                            global_step=global_step,
                            log_image=args.log_image,  # Set False to disable image (faster)
                        )
                        
                        # Optional: Log per-head attention figure (more expensive)
                        if args.log_image:
                            # Useful for understanding what each head specializes in
                            if debug_logger.should_log_histograms(total_gradient_steps):  # Less frequent
                                fig = debug_logger.create_multi_head_attention_figure(
                                    attention_weights=attn_weights,
                                    positions=data["positions"][:1],  # Single sample
                                    attention_mask=batch_mask[:1] if batch_mask is not None else None,
                                    title=f"Step {global_step}",
                                )
                                if fig is not None:
                                    writer.add_figure("debug/attention/per_head", fig, global_step)
                                    import matplotlib.pyplot as plt
                                    plt.close(fig)


                total_gradient_steps += 1

            # Host time in the whole gradient burst (sampling + critic + actor).
            # Closed before the logging block so the logging cost is excluded.
            _perf_split["update"] += time.perf_counter() - _t_upd0

            # -----------------------------------------------------------------
            # Logging
            # -----------------------------------------------------------------
            if update % 20 == 0:
                sps = int(global_step / (time.time() - start_time))
                mean_reward = float(np.mean(step_reward_window)) if step_reward_window else 0.0
                
                # Average losses over the UTD updates: divide the GPU running-sums by
                # their update counts and materialize once (a handful of syncs per 20
                # iterations instead of thousands per step).
                _na = max(n_actor_updates, 1)
                _nc = max(n_critic_updates, 1)
                alpha_val = float(alpha)
                mean_actor_loss = (loss_accumulator['actor_loss'] / _na).item()

                if args.algorithm == "tqc":
                    mean_qf_loss = (loss_accumulator['qf_loss'] / _nc).item()
                    writer.add_scalar("losses/qf_loss", mean_qf_loss, global_step)
                else:
                    mean_qf1_loss = (loss_accumulator['qf1_loss'] / _nc).item()
                    mean_qf2_loss = (loss_accumulator['qf2_loss'] / _nc).item()
                    mean_qf_loss = mean_qf1_loss + mean_qf2_loss
                    writer.add_scalar("losses/qf1_loss", mean_qf1_loss, global_step)
                    writer.add_scalar("losses/qf2_loss", mean_qf2_loss, global_step)

                writer.add_scalar("losses/actor_loss", mean_actor_loss, global_step)
                writer.add_scalar("losses/alpha", alpha_val, global_step)

                # --- losses/* : health of the value function and the updates ---
                writer.add_scalar("losses/critic_grad_norm",
                                  (loss_accumulator['critic_gnorm'] / _nc).item(), global_step)
                writer.add_scalar("losses/actor_grad_norm",
                                  (loss_accumulator['actor_gnorm'] / _na).item(), global_step)
                writer.add_scalar("losses/q_mean",
                                  (loss_accumulator['q_mean'] / _nc).item(), global_step)
                if args.algorithm != "tqc":
                    _td_abs = (loss_accumulator['td_abs'] / _nc).item()
                    _td_ms = (loss_accumulator['td_sq'] / _nc).item()
                    writer.add_scalar("losses/td_error_abs_mean", _td_abs, global_step)
                    # sqrt(E[td^2] - E[|td|]^2) is a lower bound on the true
                    # spread (it uses E|td| rather than E[td]); good enough as a
                    # relative "are a few transitions dominating" signal.
                    writer.add_scalar("losses/td_error_spread",
                                      max(_td_ms - _td_abs * _td_abs, 0.0) ** 0.5, global_step)
                if args.autotune:
                    # entropy_gap > 0 => policy is MORE random than the target,
                    # so alpha should be falling. Persistently large and positive
                    # is the "stay diffuse" pathology.
                    _tgt_ent = (loss_accumulator['target_entropy'] / _na).item()
                    _ach_ent = -(loss_accumulator['logpi_per_turbine'] / _na).item()
                    writer.add_scalar("losses/target_entropy", _tgt_ent, global_step)
                    writer.add_scalar("losses/policy_entropy", _ach_ent, global_step)
                    writer.add_scalar("losses/entropy_gap", _ach_ent - _tgt_ent, global_step)

                # Entropy-scaling diagnostics (see actor-update block)
                def _acc_mean(key):
                    return (loss_accumulator[key] / _na).item()
                _ent_term = _acc_mean('ent_term')
                _q_term = _acc_mean('q_term')
                writer.add_scalar("entropy/logpi_per_turbine", _acc_mean('logpi_per_turbine'), global_step)
                writer.add_scalar("entropy/actor_entropy_term", _ent_term, global_step)
                writer.add_scalar("entropy/actor_q_term", _q_term, global_step)
                writer.add_scalar("entropy/entropy_to_q_abs_ratio", abs(_ent_term) / (abs(_q_term) + 1e-8), global_step)
                writer.add_scalar("entropy/n_real_mean", _acc_mean('n_real_mean'), global_step)
                writer.add_scalar("charts/SPS", sps, global_step)
                writer.add_scalar("charts/step_reward_mean_1000", mean_reward, global_step)
                writer.add_scalar("debug/mean_wind_direction", float(np.mean(wind_dirs)), global_step)
                writer.add_scalar("debug/total_gradient_steps", total_gradient_steps, global_step)
                writer.add_scalar("debug/gradient_updates_per_iter", num_gradient_updates, global_step)

                if args.log_timing:
                    # env_span overlaps the other buckets (the burst runs inside
                    # it under --async_overlap), so it is excluded from the
                    # fraction denominator. overlap_hidden = un-blocked portion
                    # of the dispatch->collect span (~ the burst duration when
                    # overlapping; ~0 sequential). The direct evidence of
                    # hiding is env_sec collapsing vs a sequential run.
                    total_t = sum(v for k, v in timing.items() if k != "env_span") or 1.0
                    for k, v in timing.items():
                        writer.add_scalar(f"timing/{k}_sec", v, global_step)
                        if k != "env_span":
                            writer.add_scalar(f"timing/{k}_frac", v / total_t, global_step)
                    _hidden = timing["env_span"] - timing["env"]
                    writer.add_scalar("timing/overlap_hidden_sec", _hidden, global_step)
                    print(f"  timing(s): " + ", ".join(f"{k}={v:.2f}" for k, v in timing.items()))
                    if args.async_overlap:
                        print(f"  overlap: hidden={_hidden:.2f}s, wait={timing['env']:.2f}s")
                    # With --log_timing on, also publish the sync-accurate
                    # per-bucket fractions under perf/ so the panel gains detail
                    # rather than changing meaning.
                    for k, v in timing.items():
                        if k != "env_span":
                            writer.add_scalar(f"perf/sync_{k}_frac", v / total_t, global_step)
                    timing = {k: 0.0 for k in timing}

                # -------------------------------------------------------------
                # perf/*: is the GPU idle while the DWM envs churn?
                # -------------------------------------------------------------
                _t_now_w = time.time()
                _dt = max(_t_now_w - _perf_prev["t"], 1e-9)
                _dsteps = global_step - _perf_prev["step"]
                writer.add_scalar("perf/wall_sec_per_1k_steps",
                                  1000.0 * _dt / max(_dsteps, 1), global_step)

                # Always-on env-vs-update split (host time, no cuda sync). If
                # env_frac sits near 1 the GPU is starved and more
                # --cpus-per-task buys throughput; if update_frac dominates,
                # more CPUs buy nothing.
                _split_tot = _perf_split["env"] + _perf_split["update"]
                if _split_tot > 0:
                    writer.add_scalar("perf/env_frac",
                                      _perf_split["env"] / _split_tot, global_step)
                    writer.add_scalar("perf/update_frac",
                                      _perf_split["update"] / _split_tot, global_step)
                    writer.add_scalar("perf/env_sec", _perf_split["env"], global_step)
                    writer.add_scalar("perf/update_sec", _perf_split["update"], global_step)
                _perf_split = {"env": 0.0, "update": 0.0}

                # CPU: cgroup usage covers the main process AND every env
                # worker, expressed as a percentage of the cores this job was
                # actually allocated. ~100% means the CPU side is saturated and
                # more --cpus-per-task would buy throughput.
                _cpu_now = _read_cgroup_cpu_usec()
                if _cpu_now is not None and _perf_prev["cpu_usec"] is not None:
                    _busy_cores = (_cpu_now - _perf_prev["cpu_usec"]) / 1e6 / _dt
                    writer.add_scalar("perf/cpu_busy_cores", _busy_cores, global_step)
                    writer.add_scalar("perf/cpu_util_pct",
                                      100.0 * _busy_cores / max(_n_cpu_alloc, 1), global_step)
                writer.add_scalar("perf/cpu_allocated", _n_cpu_alloc, global_step)

                _mem = _read_cgroup_mem_bytes()
                if "cur" in _mem:
                    writer.add_scalar("perf/host_mem_gb", _mem["cur"] / 2**30, global_step)
                if "peak" in _mem:
                    writer.add_scalar("perf/host_mem_peak_gb", _mem["peak"] / 2**30, global_step)
                if "rss_sum" in _mem:      # /proc fallback; see _read_cgroup_mem_bytes
                    writer.add_scalar("perf/host_rss_sum_gb",
                                      _mem["rss_sum"] / 2**30, global_step)

                if device.type == "cuda":
                    writer.add_scalar("perf/gpu_mem_alloc_gb",
                                      torch.cuda.memory_allocated() / 2**30, global_step)
                    writer.add_scalar("perf/gpu_mem_reserved_gb",
                                      torch.cuda.memory_reserved() / 2**30, global_step)
                    writer.add_scalar("perf/gpu_mem_peak_gb",
                                      torch.cuda.max_memory_allocated() / 2**30, global_step)
                    # Not implemented by every ROCm/CUDA build; never fatal.
                    try:
                        writer.add_scalar("perf/gpu_util_pct",
                                          float(torch.cuda.utilization()), global_step)
                    except Exception:  # noqa: BLE001 - diagnostic only
                        pass

                _perf_prev = {"t": _t_now_w, "cpu_usec": _cpu_now, "step": global_step}

                # -------------------------------------------------------------
                # actions/*: what is the policy actually commanding?
                # A derate mean pinned near a constant with a small std is the
                # "stuck at one derate level" failure; a sat_frac near 1 says
                # the slew limit is binding rather than the policy choosing.
                # -------------------------------------------------------------
                if _act_acc["n"] > 0:
                    _an = _act_acc["n"]
                    for _ch in _act_cols:
                        _mean = _act_acc[f"{_ch}_sum"] / _an
                        _var = max(_act_acc[f"{_ch}_sq"] / _an - _mean * _mean, 0.0)
                        writer.add_scalar(f"actions/{_ch}_mean", _mean, global_step)
                        writer.add_scalar(f"actions/{_ch}_std", _var ** 0.5, global_step)
                        writer.add_scalar(f"actions/{_ch}_sat_frac",
                                          _act_acc[f"{_ch}_sat"] / _an, global_step)
                    for _k in _act_acc:
                        _act_acc[_k] = 0.0

                print(f"Step {global_step}: SPS={sps}, qf_loss={mean_qf_loss:.4f}, "
                      f"actor_loss={mean_actor_loss:.4f}, alpha={alpha_val:.4f}, "
                      f"reward_mean={mean_reward:.4f}, grad_steps={total_gradient_steps}")
        

                # === Fine-tuning diagnostics (when resuming from checkpoint) ===
                if args.resume_checkpoint is not None and update % 100 == 0:
                    if args.algorithm == "tqc":
                        # TQC fine-tuning diagnostics (optimizer state only — no qf1/qf2 available)
                        log_finetune_diagnostics(
                            writer=writer,
                            global_step=global_step,
                            actor_optimizer=actor_optimizer,
                            q_optimizer=q_optimizer,
                            policy_lr=args.policy_lr,
                            q_lr=args.q_lr,
                            alpha=float(alpha),
                        )
                    else:
                        # SAC fine-tuning diagnostics (includes Q-value stats)
                        recent_returns = list(envs.return_queue)[-10:] if hasattr(envs, 'return_queue') else []

                        with torch.no_grad():
                            _, log_pi_diag, _, _ = actor.get_action(
                                data["observations"][:32],
                                data["positions"][:32],
                                data["attention_mask"][:32],
                                recep_profile=batch_receptivity[:32] if batch_receptivity is not None else None,
                                influence_profile=batch_influence[:32] if batch_influence is not None else None,
                            )
                            policy_entropy = -log_pi_diag.mean().item()
                            # Recompute Q-values fresh: under reduce-overhead the cached
                            # qf1_value/qf2_value buffers were overwritten by the actor update.
                            qf1_values_diag = qf1(
                                data["observations"], data["actions"], data["positions"],
                                data["attention_mask"],
                                recep_profile=batch_receptivity,
                                influence_profile=batch_influence,
                            ).clone()
                            qf2_values_diag = qf2(
                                data["observations"], data["actions"], data["positions"],
                                data["attention_mask"],
                                recep_profile=batch_receptivity,
                                influence_profile=batch_influence,
                            ).clone()

                        log_finetune_diagnostics(
                            writer=writer,
                            global_step=global_step,
                            actor_optimizer=actor_optimizer,
                            q_optimizer=q_optimizer,
                            policy_lr=args.policy_lr,
                            q_lr=args.q_lr,
                            qf1_values=qf1_values_diag,
                            qf2_values=qf2_values_diag,
                            episode_returns=recent_returns,
                            alpha=float(alpha),
                            policy_entropy=policy_entropy,
                        )


            # Log summary metrics (frequency controlled by logger)
            if debug_logger.should_log(global_step):
                debug_logger.log_summary_metrics(
                    writer=writer,
                    global_step=global_step,
                )

                # Print diagnostic summary to console (frequency controlled by logger)
                if debug_logger.should_print_diagnostics(global_step):
                    debug_logger.print_diagnostics(global_step)


        # =====================================================================
        # COLLECT ENV STEP (dispatched before the training block)
        # =====================================================================
        if step_result is None:  # overlap mode: workers simulated during the burst
            _t0 = _sync_timer()
            _t_env0 = time.perf_counter()
            step_result = envs.step_wait()
            # Under --async_overlap this is only the UNHIDDEN remainder of the
            # env step (the workers ran during the burst), which is exactly what
            # perf/env_frac should report.
            _perf_split["env"] += time.perf_counter() - _t_env0
            if args.log_timing:
                _t_now = time.perf_counter()
                timing["env"] += _t_now - _t0
                timing["env_span"] += _t_now - _t_async
        next_obs, rewards, terminations, truncations, infos = step_result

        # Get current layout names for each env. Under domain randomization the pool
        # is huge, so bucket every training layout under "dr_pool" for debug stats.
        if dr_enabled:
            current_layouts = ["dr_pool"] * args.num_envs
        else:
            current_layouts = get_env_current_layout(envs)

        # Log per-step data to debug tracker (always - internal deques handle storage)
        for i in range(args.num_envs):
            debug_logger.log_layout_step(
                layout_name=current_layouts[i],
                reward=float(rewards[i]),
                power=float(infos.get("Power agent", [0.0] * args.num_envs)[i]) if "Power agent" in infos else None,
                actions=actions[i] if isinstance(actions, np.ndarray) else np.array(actions[i]),
            )
            debug_logger.log_wind_direction(float(wind_dirs[i]))


        # Track rewards
        step_reward_window.extend(np.array(rewards).flatten().tolist())

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
            # loads_ood is per-env (T,) bool arrays; collation across envs can
            # be ragged/objecty depending on gymnasium version -- diagnostic
            # only, so never let it kill training.
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
            # Per-channel ratios. The wrapper puts ONE DICT PER ENV in
            # info["del_ratio_by_channel"], but gymnasium's VectorEnv._add_info
            # RECURSES into dict-valued infos, so what arrives here is
            # {channel: array-over-envs} -- plus a "_channel" boolean mask entry
            # per key, which must be skipped. Older/other collations hand back an
            # object array of dicts instead, so accept both shapes. Diagnostic
            # only: never let it kill training.
            try:
                _by_ch = infos.get("del_ratio_by_channel")

                def _push_del_ratio(_ch, _val):
                    _fv = float(_val)
                    if not np.isfinite(_fv):
                        return
                    if _ch not in del_channel_windows:
                        del_channel_windows[_ch] = deque(maxlen=1000)
                    del_channel_windows[_ch].append(_fv)

                if isinstance(_by_ch, dict):
                    for _ch, _arr in _by_ch.items():
                        if _ch.startswith("_"):      # gymnasium's presence mask
                            continue
                        for _v in np.asarray(_arr, dtype=float).flatten():
                            _push_del_ratio(_ch, _v)
                elif _by_ch is not None:
                    for _d in np.asarray(_by_ch, dtype=object).flatten():
                        if isinstance(_d, dict):
                            for _ch, _v in _d.items():
                                _push_del_ratio(_ch, _v)
            except (TypeError, ValueError):
                pass

        # Log episode stats
        if "final_info" in infos:
            ep_return = np.mean(envs.return_queue)
            ep_length = np.mean(envs.length_queue)
            ep_power = np.mean(envs.mean_power_queue)

            print(f"Step {global_step}: Episode return={ep_return:.2f}, power={ep_power:.2f}")
            writer.add_scalar("charts/episodic_return", ep_return, global_step)
            writer.add_scalar("charts/episodic_length", ep_length, global_step)
            writer.add_scalar("charts/episodic_power", ep_power, global_step)

            # Greedy-baseline comparison (filled by RecordEpisodeVals only when
            # the env is built with Baseline_comp, e.g. Power_reward="Baseline").
            # power_ratio > 1 means the agent beats the zero-offset greedy farm.
            if getattr(envs, "mean_power_queue_baseline", None) and len(envs.mean_power_queue_baseline) > 0:
                ep_power_base = float(np.mean(envs.mean_power_queue_baseline))
                writer.add_scalar("charts/episodic_power_baseline", ep_power_base, global_step)
                if ep_power_base > 0:
                    writer.add_scalar("charts/episodic_power_ratio",
                                      ep_power / ep_power_base, global_step)

            # DEL signs of life (rolling-window means; window length ~ recent
            # episode(s)). del_penalty stays 0 through each warm-up, and is
            # identically 0 in --del_log-only (unpenalized) runs.
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

                # --- del/* : the per-channel load story ---
                # Same numbers, but split by channel instead of collapsed into
                # del_agent_max, plus the argmax identity mirrored into the same
                # namespace so one panel shows which load binds and how hard.
                # charts/* is left untouched so existing runs stay comparable.
                for _ch, _win in del_channel_windows.items():
                    if len(_win) > 0:
                        writer.add_scalar(f"del/ratio/{_ch}",
                                          float(np.mean(_win)), global_step)
                if len(del_binding_window) > 0:
                    _bind = list(del_binding_window)
                    for _ch in _del_reward_channels:
                        writer.add_scalar(f"del/binding_frac/{_ch}",
                                          _bind.count(_ch) / len(_bind), global_step)
                if len(del_agent_max_window) > 0:
                    writer.add_scalar("del/agent_max",
                                      float(np.mean(del_agent_max_window)), global_step)
                if len(del_margin_window) > 0:
                    writer.add_scalar("del/margin",
                                      float(np.mean(del_margin_window)), global_step)


        # Handle final observations
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_obs"][idx]

        # Store in replay buffer
        for i in range(args.num_envs):
            done = terminations[i] or truncations[i]
            action_reshaped = actions[i].reshape(-1, action_dim_per_turbine)

            layout_idx_i = current_layout_indices[i] if current_layout_indices is not None else None
            perm_i = current_permutations[i] if current_permutations is not None else None

            rb.add(
                obs[i],
                real_next_obs[i],
                action_reshaped,
                rewards[i],
                done,
                raw_positions[i],
                current_masks[i],
                wind_dirs[i],
                layout_index=layout_idx_i,
                permutation=perm_i,
            )

        # One-shot warmup buffer save (buffer pre-generation for ablation runs)
        if (args.save_buffer_at_learning_starts
                and not warmup_buffer_saved
                and global_step >= args.learning_starts):
            rb.save(
                f"runs/{run_name}/replay_buffer_warmup_{args.learning_starts}.npz",
                extra_meta=buffer_meta(global_step),
            )
            warmup_buffer_saved = True
            if args.buffer_only:
                print("\n--buffer_only set: warmup buffer saved, exiting before training.")
                close_all_evaluators()
                envs.close()
                writer.close()
                return

        obs = next_obs

        # =====================================================================
        # CHECKPOINTING
        # =====================================================================
        
        if args.save_model and global_step >= next_save_step:
            save_checkpoint(
                actor, qf1, qf2, actor_optimizer, q_optimizer,
                global_step, run_name, args, log_alpha, alpha_optimizer,
                tqc_critic=tqc_critic,
                obs_norm_state=obs_normalizer.state_dict() if obs_normalizer is not None else None,
            )
            next_save_step += args.save_interval

        # Periodic replay buffer save (overwrites a single file; atomic rename
        # keeps the previous copy intact if the job is killed mid-write)
        if args.buffer_save_interval > 0 and global_step >= next_buffer_save_step:
            rb.save(
                f"runs/{run_name}/replay_buffer.npz",
                extra_meta=buffer_meta(global_step),
            )
            next_buffer_save_step += args.buffer_save_interval

        # =====================================================================
        # PERIODIC EVALUATION
        # =====================================================================
        
        if global_step >= next_eval_step:
            print(f"\nRunning evaluation at step {global_step}...")
            eval_dict, eval_metrics = run_all_evaluations()

            # Log to tensorboard/wandb
            for name, value in eval_dict.items():
                writer.add_scalar(name, value, global_step)
            
            print(f"Eval step {global_step} - Mean reward: {eval_metrics.mean_reward:.4f}, "
                  f"Power ratio: {eval_metrics.power_ratio:.4f}")
            
            # Per-layout summary
            if len(eval_metrics.per_layout_rewards) > 1:
                print("  Per-layout power ratios:")
                for layout, ratio in eval_metrics.per_layout_power_ratios.items():
                    print(f"    {layout}: {ratio:.4f}")
            
            next_eval_step += args.eval_interval
        
    # =========================================================================
    
    # FINAL SAVE AND CLEANUP
    # =========================================================================
    
    if args.save_model:
        save_checkpoint(
            actor, qf1, qf2, actor_optimizer, q_optimizer,
            global_step, run_name, args, log_alpha, alpha_optimizer,
            tqc_critic=tqc_critic,
            obs_norm_state=obs_normalizer.state_dict() if obs_normalizer is not None else None,
        )

    if args.save_buffer_final or args.buffer_save_interval > 0:
        rb.save(
            f"runs/{run_name}/replay_buffer.npz",
            extra_meta=buffer_meta(global_step),
        )

    print("\n" + "=" * 60)
    print("Training finished!")
    print(f"Total time: {(time.time() - start_time) / 3600:.2f} hours")
    print("=" * 60)
    

    # Close evaluators
    close_all_evaluators()

    envs.close()
    writer.close()


if __name__ == "__main__":
    main()