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

import os
import random
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
from helpers.env_configs import make_env_config

# Receptivity profile computation
from helpers.receptivity_profiles import compute_layout_profiles

# Evaluation import
from helpers.eval_utils import PolicyEvaluator, run_evaluation

from networks import (
    TransformerActor,
    TransformerCritic,
    TransformerTQCCritic,
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
    
    # Parse layouts
    layout_names = [l.strip() for l in args.layouts.split(",")]
    is_multi_layout = len(layout_names) > 1
    
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

    # Force math SDPA backend (avoids ROCm Flash/MemEfficient kernel bugs)
    # ONLY RELEVANT FOR LUMI. TODO make it such this only works on lumi
    if device.type == "cuda":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        print("Forced math SDPA backend")

    # =========================================================================
    # ENVIRONMENT SETUP
    # =========================================================================
    
    # Import WindGym components
    from WindGym import WindFarmEnv
    from WindGym.wrappers import RecordEpisodeVals, PerTurbineObservationWrapper
    from helpers.multi_layout_env import MultiLayoutEnv, LayoutConfig
    
    # Wind turbine
    if args.turbtype == "DTU10MW":
        from py_wake.examples.data.dtu10mw import DTU10MW as WT
    elif args.turbtype == "V80":
        from py_wake.examples.data.hornsrev1 import V80 as WT
    else:
        raise ValueError(f"Unknown turbine type: {args.turbtype}")
    
    wind_turbine = WT()
    
    # Create layout configurations
    print("Setting up layouts...")
    layouts = []
    for name in layout_names:
        x_pos, y_pos = get_layout_positions(name, wind_turbine)
        layout = LayoutConfig(name=name, x_pos=x_pos, y_pos=y_pos)
        

        if args.profile_encoding_type is not None:
            if args.profile_source.lower() == "geometric":
                from helpers.geometric_profiles import compute_layout_profiles_vectorized
                
                # Get rotor diameter as a float (geometric version doesn't need the full WT object)
                D = wind_turbine.diameter()  # or however DTU10MW exposes this
                
                print(f"Computing GEOMETRIC profiles for layout: {name}")
                receptivity_profiles, influence_profiles = compute_layout_profiles_vectorized(
                    x_pos, y_pos,
                    rotor_diameter=D,
                    k_wake=0.04,
                    n_directions=args.n_profile_directions,
                    sigma_smooth=10.0,
                    scale_factor=15.0,
                )
            elif args.profile_source.lower() == "pywake":
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
            
        layouts.append(layout)

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
    
    mes_prefixes = {
        "ws_mes": "ws",
        "wd_mes": "wd",
        "yaw_mes": "yaw",
        "power_mes": "power",
    }

    for mes_type, prefix in mes_prefixes.items():
        config[mes_type][f"{prefix}_history_N"] = args.history_length
        config[mes_type][f"{prefix}_history_length"] = args.history_length

    
    base_env_kwargs = {
        "turbine": wind_turbine,
        "n_passthrough": args.max_eps,
        "TurbBox": "/work/users/manils/rl_timestep/Boxes/V80env/",  # Adjust path as needed
        "config": config,
        "turbtype": args.TI_type,
        "dt_sim": args.dt_sim,
        "dt_env": args.dt_env,
        "yaw_step_sim": args.yaw_step,
    }
    
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
        """
        env = PerTurbineObservationWrapper(env)
        if args.use_wd_deviation:
            env = EnhancedPerTurbineWrapper(env, wd_scale_range=args.wd_scale_range)
        return env
    
    def make_env_fn(seed):
        """Factory function for vectorized environments."""
        def _init():
            env = MultiLayoutEnv(
                layouts=layouts,
                env_factory=env_factory,
                per_turbine_wrapper=combined_wrapper,  # Use combined wrapper
                seed=seed,
                shuffle=args.shuffle_turbs,  # Shuffle turbines within each layout
                max_episode_steps=args.max_episode_steps,
            )
            return env
        return _init

    # Create vectorized environments
    print(f"Creating {args.num_envs} parallel environment(s)...")
    envs = gym.vector.AsyncVectorEnv(
        [make_env_fn(args.seed + i) for i in range(args.num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    envs = RecordEpisodeVals(envs)
       

    n_turbines_max = envs.env.get_attr('max_turbines')[0]
    obs_dim_per_turbine = envs.single_observation_space.shape[-1]
    action_dim_per_turbine = 1
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
        deterministic=False,
        use_profiles=use_profiles,  # NEW: Pass profile setting
        n_profile_directions=args.n_profile_directions,  # NEW: Pass profile resolution
        profile_source=args.profile_source,
    )


    # Action scaling
    action_high = envs.single_action_space.high[0]
    action_low = envs.single_action_space.low[0]
    action_scale = (action_high - action_low) / 2.0
    action_bias = (action_high + action_low) / 2.0
    
    # =========================================================================
    # DEBUG LOGGER AND TRACKING SETUP
    # =========================================================================

    # Initialize debug logger with configurable frequencies
    debug_logger = create_debug_logger(
        layout_names=layout_names,
        log_every=250000,  # Base frequency - others are multiples of this
    )
    # Frequencies will be:
    #   - summary metrics: every 100 steps
    #   - attention analysis: every 500 steps  
    #   - gradient norms: every 100 steps
    #   - q-value stats: every 50 steps
    #   - diagnostic print: every 2000 steps

    print(f"Debug logger initialized for layouts: {layout_names}")
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
                "debug/layout_names": layout_names,
                "debug/is_multi_layout": is_multi_layout,
                "debug/max_turbines": n_turbines_max,
                "debug/log_frequency": debug_logger.log_frequency,
                "debug/attention_log_frequency": debug_logger.attention_log_frequency,
                "debug/gradient_log_frequency": debug_logger.gradient_log_frequency,
            },
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

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
    )

    # Update evaluator with actor reference
    evaluator.agent = agent

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
        tqc_critic = TransformerTQCCritic(
            n_critics=args.tqc_n_critics,
            n_quantiles=args.tqc_n_quantiles,
            **critic_kwargs,
        ).to(device)
        tqc_critic_target = TransformerTQCCritic(
            n_critics=args.tqc_n_critics,
            n_quantiles=args.tqc_n_quantiles,
            **critic_kwargs,
        ).to(device)
        tqc_critic_target.load_state_dict(tqc_critic.state_dict())

        # Precompute quantile midpoints: tau_i = (i + 0.5) / N
        taus = (torch.arange(args.tqc_n_quantiles, device=device).float() + 0.5) / args.tqc_n_quantiles

        tqc_params = get_critic_params_excluding_shared(tqc_critic, shared_recep_encoder, shared_influence_encoder)
        q_optimizer = optim.Adam(tqc_params + shared_encoder_params, lr=args.q_lr)

        actor_params = sum(p.numel() for p in actor.parameters())
        critic_params = sum(p.numel() for p in tqc_critic.parameters())
        print(f"Actor parameters: {actor_params:,}")
        print(f"TQC Critic parameters: {critic_params:,} ({args.tqc_n_critics} critics x {args.tqc_n_quantiles} quantiles)")
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

        q_optimizer = optim.Adam(
            qf1_params + qf2_params + shared_encoder_params,
            lr=args.q_lr,
        )

        actor_params = sum(p.numel() for p in actor.parameters())
        critic_params = sum(p.numel() for p in qf1.parameters())
        print(f"Actor parameters: {actor_params:,}")
        print(f"Critic parameters: {critic_params:,} (x2)")

    # Optimizers (exclude shared encoder params — handled by q_optimizer only)
    actor_optimizer = optim.Adam(
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
        alpha = log_alpha.exp().item()
        alpha_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha
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
                    alpha = log_alpha.exp().item()
                    print(f"✓ Loaded entropy coefficient: alpha={alpha:.4f}")
                if "alpha_optimizer_state_dict" in checkpoint:
                    alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer_state_dict"])
                    print(f"✓ Loaded alpha optimizer state")
            else:
                print(f"✓ Reset entropy coefficient (alpha={alpha:.4f})")
       
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
            actor_optimizer = optim.Adam(
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
    )


    # =========================================================================
    # TRAINING LOOP
    # =========================================================================
    
    print(f"\nStarting training for {args.total_timesteps} timesteps...")
    print(f"UTD ratio: {args.utd_ratio} (gradient updates per env step)")
    print(f"With {args.num_envs} envs: {int(args.num_envs * args.utd_ratio)} gradient updates per iteration")
    print("=" * 60)
    

    save_checkpoint(
        actor, qf1, qf2, actor_optimizer, q_optimizer,
        0, run_name, args, log_alpha, alpha_optimizer,
        tqc_critic=tqc_critic,
    )


    # Track evaluation timing
    next_eval_step = args.eval_interval
    
    # Initial evaluation
    if args.eval_initial:
        print("\nRunning initial evaluation before training...")
        eval_metrics = evaluator.evaluate()
        eval_dict = eval_metrics.to_dict()
        
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
    # Reset environments
    obs, infos = envs.reset(seed=args.seed)
    
    # Tracking
    step_reward_window = deque(maxlen=1000)
    # next_save_step = ((start_step // args.save_interval) + 1) * args.save_interval  # Account for resumed step
    next_save_step = start_step + args.save_interval 
    # For logging losses (we'll average over the UTD updates)
    if args.algorithm == "tqc":
        loss_accumulator = {
            'qf_loss': [], 'actor_loss': [], 'alpha_loss': []
        }
    else:
        loss_accumulator = {
            'qf1_loss': [], 'qf2_loss': [], 'actor_loss': [], 'alpha_loss': []
        }

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
            actor_optimizer = optim.Adam(
                [p for p in actor.parameters() if id(p) not in shared_param_ids],
                lr=args.policy_lr,
            )
            print(f"\n[Step {global_step}] Unfroze pretrained encoder parameters")
        
        # Get environment info (needed for replay buffer)
        wind_dirs = get_env_wind_directions(envs)
        raw_positions = get_env_raw_positions(envs)
        current_masks = get_env_attention_masks(envs)

        # Get layout identifiers for replay buffer (lightweight)
        if args.profile_encoding_type is not None:
            current_layout_indices = get_env_layout_indices(envs)
            current_permutations = get_env_permutations(envs)
        else:
            current_layout_indices = None
            current_permutations = None


        # Select action
        if global_step < args.learning_starts:
            if args.initial_exploration == "policy":
                # Use the actor network (useful when resuming from checkpoint)
                with torch.no_grad():
                    actions = agent.act(envs, obs)
            else:
                # Random exploration (default for training from scratch)
                actions = envs.action_space.sample()
        else:
            with torch.no_grad():
                actions = agent.act(envs, obs)
        
        # Step environment
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)


        # Get current layout names for each env
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
        
        # Log episode stats
        if "final_info" in infos:
            ep_return = np.mean(envs.return_queue)
            ep_length = np.mean(envs.length_queue)
            ep_power = np.mean(envs.mean_power_queue)
            
            print(f"Step {global_step}: Episode return={ep_return:.2f}, power={ep_power:.2f}")
            writer.add_scalar("charts/episodic_return", ep_return, global_step)
            writer.add_scalar("charts/episodic_length", ep_length, global_step)
            writer.add_scalar("charts/episodic_power", ep_power, global_step)


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



        obs = next_obs
        
        # =====================================================================
        # TRAINING
        # =====================================================================
        
        if global_step > args.learning_starts and len(rb) >= args.batch_size:

            # Calculate number of gradient updates for this iteration
            # This scales with num_envs to maintain consistent sample efficiency
            num_gradient_updates = max(1, int(args.num_envs * args.utd_ratio))
            
            # Clear loss accumulator for this iteration
            loss_accumulator = {k: [] for k in loss_accumulator}


            for grad_step in range(num_gradient_updates):
                # Sample a fresh batch for each gradient update
                data = rb.sample(args.batch_size)
                
                batch_mask = data["attention_mask"]
                
                # Get profiles from batch (will be None if not using profiles)
                batch_receptivity = data.get("receptivity", None)
                batch_influence = data.get("influence", None)

                # -----------------------------------------------------------------
                # Update Critics
                # -----------------------------------------------------------------
                with torch.no_grad():
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
                    with torch.no_grad():
                        # Target quantiles: (n_critics, batch, n_quantiles)
                        target_quantiles = tqc_critic_target(
                            data["next_observations"], next_actions,
                            data["positions"], batch_mask,
                            recep_profile=batch_receptivity,
                            influence_profile=batch_influence,
                        )
                        batch_size_cur = data["rewards"].shape[0]
                        # Flatten across critics, sort, truncate top-d
                        all_target_q = target_quantiles.permute(1, 0, 2).reshape(batch_size_cur, -1)
                        sorted_q, _ = all_target_q.sort(dim=1)
                        n_keep = args.tqc_n_critics * args.tqc_n_quantiles - args.tqc_top_quantiles_to_drop
                        truncated_mean = sorted_q[:, :n_keep].mean(dim=1, keepdim=True)
                        target_q = data["rewards"] + (1 - data["dones"]) * args.gamma * (truncated_mean - alpha * next_log_pi)

                    # Current quantiles: (n_critics, batch, n_quantiles)
                    current_q = tqc_critic(
                        data["observations"], data["actions"],
                        data["positions"], batch_mask,
                        recep_profile=batch_receptivity,
                        influence_profile=batch_influence,
                    )
                    qf_loss = sum(
                        quantile_huber_loss(current_q[i], target_q, taus)
                        for i in range(args.tqc_n_critics)
                    )

                    q_optimizer.zero_grad(set_to_none=True)
                    qf_loss.backward()
                    if args.grad_clip:
                        torch.nn.utils.clip_grad_norm_(
                            tqc_critic.parameters(),
                            max_norm=args.grad_clip_max_norm,
                        )
                    q_optimizer.step()

                    if debug_logger.should_log_gradients(total_gradient_steps):
                        for i, critic in enumerate(tqc_critic.critics):
                            grad_norm = sum(
                                p.grad.norm().item() ** 2
                                for p in critic.parameters() if p.grad is not None
                            ) ** 0.5
                            writer.add_scalar(f"debug/grad_norm/tqc_critic_{i}", grad_norm, global_step)

                    loss_accumulator['qf_loss'].append(qf_loss.item())
                else:
                    # --- SAC critic update ---
                    with torch.no_grad():
                        qf1_next = qf1_target(
                            data["next_observations"], next_actions,
                            data["positions"], batch_mask,
                            recep_profile=batch_receptivity,
                            influence_profile=batch_influence,
                        )
                        qf2_next = qf2_target(
                            data["next_observations"], next_actions,
                            data["positions"], batch_mask,
                            recep_profile=batch_receptivity,
                            influence_profile=batch_influence,
                        )
                        min_qf_next = torch.min(qf1_next, qf2_next) - alpha * next_log_pi
                        target_q = data["rewards"] + (1 - data["dones"]) * args.gamma * min_qf_next

                    qf1_value = qf1(data["observations"], data["actions"],
                                    data["positions"], batch_mask,
                                    recep_profile=batch_receptivity,
                                    influence_profile=batch_influence)
                    qf2_value = qf2(data["observations"], data["actions"],
                                    data["positions"], batch_mask,
                                    recep_profile=batch_receptivity,
                                    influence_profile=batch_influence)

                    if debug_logger.should_log_q_values(total_gradient_steps):
                        debug_logger.log_q_value_stats(
                            qf1_values=qf1_value,
                            qf2_values=qf2_value,
                            target_q=target_q,
                            writer=writer,
                            global_step=global_step,
                        )

                    qf1_loss = F.mse_loss(qf1_value, target_q)
                    qf2_loss = F.mse_loss(qf2_value, target_q)
                    qf_loss = qf1_loss + qf2_loss

                    q_optimizer.zero_grad(set_to_none=True)
                    qf_loss.backward()
                    if args.grad_clip:
                        torch.nn.utils.clip_grad_norm_(
                            qf1_params + qf2_params + shared_encoder_params,
                            max_norm=args.grad_clip_max_norm,
                        )
                    q_optimizer.step()

                    if debug_logger.should_log_gradients(total_gradient_steps):
                        debug_logger.log_critic_gradient_norms(qf1, qf2, writer, global_step)

                    loss_accumulator['qf1_loss'].append(qf1_loss.item())
                    loss_accumulator['qf2_loss'].append(qf2_loss.item())

                # -----------------------------------------------------------------
                # Update Actor (delayed based on total gradient steps)
                # -----------------------------------------------------------------
                if total_gradient_steps % args.policy_frequency == 0:
                    # Get actions from current policy
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
                        qf1_pi = qf1(data["observations"], actions_pi, data["positions"],
                                     batch_mask,
                                     recep_profile=batch_receptivity,
                                     influence_profile=batch_influence)
                        qf2_pi = qf2(data["observations"], actions_pi, data["positions"],
                                     batch_mask,
                                     recep_profile=batch_receptivity,
                                     influence_profile=batch_influence)
                        min_qf_pi = torch.min(qf1_pi, qf2_pi)

                    # Policy loss (maximize Q - alpha * entropy)
                    actor_loss = (alpha * log_pi - min_qf_pi).mean()
                    
                    # Update actor
                    actor_optimizer.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    if args.grad_clip:
                        torch.nn.utils.clip_grad_norm_(
                            actor.parameters(),
                            max_norm=args.grad_clip_max_norm
                        )
                    actor_optimizer.step()

                    if debug_logger.should_log_gradients(total_gradient_steps):
                        debug_logger.log_actor_gradient_norms(actor, writer, global_step)

                    loss_accumulator['actor_loss'].append(actor_loss.item())
                    
                    # -------------------------------------------------------------
                    # Update Alpha (entropy coefficient)
                    # -------------------------------------------------------------
                    if args.autotune:
                        log_pi_detached = log_pi.detach()
                        
                        # Adaptive target entropy per sample
                        target_entropy_batch = compute_adaptive_target_entropy(
                            data["attention_mask"],
                            action_dim_per_turbine
                        )
                        
                        # Alpha loss
                        alpha_loss = (-log_alpha.exp() * (log_pi_detached + target_entropy_batch)).mean()
                        
                        alpha_optimizer.zero_grad(set_to_none=True)
                        alpha_loss.backward()
                        alpha_optimizer.step()
                        alpha = log_alpha.exp().item()
                        
                        loss_accumulator['alpha_loss'].append(alpha_loss.item())
                
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

            # -----------------------------------------------------------------
            # Logging
            # -----------------------------------------------------------------
            if update % 20 == 0:
                sps = int(global_step / (time.time() - start_time))
                mean_reward = float(np.mean(step_reward_window)) if step_reward_window else 0.0
                
                # Average losses over the UTD updates
                mean_actor_loss = np.mean(loss_accumulator['actor_loss']) if loss_accumulator['actor_loss'] else 0

                if args.algorithm == "tqc":
                    mean_qf_loss = np.mean(loss_accumulator['qf_loss']) if loss_accumulator['qf_loss'] else 0
                    writer.add_scalar("losses/qf_loss", mean_qf_loss, global_step)
                else:
                    mean_qf1_loss = np.mean(loss_accumulator['qf1_loss']) if loss_accumulator['qf1_loss'] else 0
                    mean_qf2_loss = np.mean(loss_accumulator['qf2_loss']) if loss_accumulator['qf2_loss'] else 0
                    mean_qf_loss = mean_qf1_loss + mean_qf2_loss
                    writer.add_scalar("losses/qf1_loss", mean_qf1_loss, global_step)
                    writer.add_scalar("losses/qf2_loss", mean_qf2_loss, global_step)

                writer.add_scalar("losses/actor_loss", mean_actor_loss, global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                writer.add_scalar("charts/SPS", sps, global_step)
                writer.add_scalar("charts/step_reward_mean_1000", mean_reward, global_step)
                writer.add_scalar("debug/mean_wind_direction", float(np.mean(wind_dirs)), global_step)
                writer.add_scalar("debug/total_gradient_steps", total_gradient_steps, global_step)
                writer.add_scalar("debug/gradient_updates_per_iter", num_gradient_updates, global_step)

                print(f"Step {global_step}: SPS={sps}, qf_loss={mean_qf_loss:.4f}, "
                      f"actor_loss={mean_actor_loss:.4f}, alpha={alpha:.4f}, "
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
                            alpha=alpha,
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

                        log_finetune_diagnostics(
                            writer=writer,
                            global_step=global_step,
                            actor_optimizer=actor_optimizer,
                            q_optimizer=q_optimizer,
                            policy_lr=args.policy_lr,
                            q_lr=args.q_lr,
                            qf1_values=qf1_value,
                            qf2_values=qf2_value,
                            episode_returns=recent_returns,
                            alpha=alpha,
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
        # CHECKPOINTING
        # =====================================================================
        
        if args.save_model and global_step >= next_save_step:
            save_checkpoint(
                actor, qf1, qf2, actor_optimizer, q_optimizer,
                global_step, run_name, args, log_alpha, alpha_optimizer,
                tqc_critic=tqc_critic,
            )
            next_save_step += args.save_interval

        # =====================================================================
        # PERIODIC EVALUATION
        # =====================================================================
        
        if global_step >= next_eval_step:
            print(f"\nRunning evaluation at step {global_step}...")
            eval_metrics = evaluator.evaluate()
            eval_dict = eval_metrics.to_dict()
            
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
        )
    
    print("\n" + "=" * 60)
    print("Training finished!")
    print(f"Total time: {(time.time() - start_time) / 3600:.2f} hours")
    print("=" * 60)
    

    # Close evaluator
    evaluator.close()

    envs.close()
    writer.close()


if __name__ == "__main__":
    main()