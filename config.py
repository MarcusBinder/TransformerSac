"""
Configuration dataclass for Transformer-SAC wind farm training.

All CLI arguments are defined here via a tyro-compatible dataclass.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Args:
    """Command-line arguments for training."""

    # === Experiment Settings ===
    config: str = "default"  # Environment config preset
    power_schedule: str = "default"  # "default" (80/60/70/100) or "boost" (80/115/70/100, needs yaw)
    exp_name: str = "transformer_sac_windfarm"
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = True  # Enable wandb tracking
    wandb_project_name: str = "transformer_windfarm"
    wandb_entity: Optional[str] = None
    exp_group: Optional[str] = None  # W&B run group (groups seeds of one config); None = ungrouped
    save_model: bool = True
    save_interval: int = 10000
    log_image: bool = False  # Log attention images to TensorBoard

    shuffle_turbs: bool = True  # Shuffle turbine order in obs/action
    max_episode_steps: Optional[int] = None # Max steps per episode (None = use env default)

    # === Staggered warm-up episode lengths ===
    # Phase-offsets each env's resets/shuffles so they don't all happen on the
    # same global step. Only the FIRST episode of each env is staggered; every
    # episode after uses max_episode_steps. A one-time offset is enough to keep
    # the reset phases permanently desynchronized.
    stagger_warmup: bool = False                   # Enable one-time staggered warm-up
    warmup_group_size: int = 5                      # Envs per group (1 = every env distinct)
    warmup_min_episode_steps: Optional[int] = None  # Shortest warm-up length (e.g. 800)

    # === Receptivity Profile Settings ===
    # Stage-1 baseline profile pathway: geometric dual-rose, FourierProfileEncoder,
    # s1/h48 @ 360 dirs (see archive/stage_1.sh). These were the frozen A00 winners.
    profile_encoder_kwargs: str = '{"use_phase": false, "learnable_weights": true, "n_harmonics": 48}'  # JSON string of encoder-specific kwargs
    profile_source: str = "geometric"  # "PyWake" or "geometric"
    profile_encoding_type: Optional[str] = "FourierProfileEncoder"  # None for no profile encoding
    profile_encoder_hidden: int = 128       # Hidden dim in profile encoder MLP
    rotate_profiles: bool = True            # Rotate profiles to wind-relative frame
    n_profile_directions: int = 360         # Number of directions in profile
    profile_sigma_smooth: float = 1.0       # Gaussian smoothing sigma (bins) for geometric profile computation (stage-1 baseline)
    profile_use_influence: bool = True      # False => single receptivity rose + one encoder (drop redundant influence)
    profile_geom_mode: str = "wake"         # geometric rose construction: "wake" (up/downstream wake sum) or "distance" (bearing-keyed inverse-distance)
    profile_fusion_type: str = "add"       # "add" or "joint" fusion of receptivity and influence profiles
    profile_embed_mode: str = "concat"     # "add" or "concat" — how fused profile is integrated into token embedding (stage-1 baseline)
    share_profile_encoder: bool = False         # Whether to share weights between actor and critic for profile encoder

    # === Environment Settings ===
    backend: str = "dynamiks"  # Flow solver backend: "dynamiks" (default) or "pywake" (steady-state)
    # Wind turbine type. Default IEA34 (the paper turbine as of 2026-07); old
    # checkpoints saved turbtype="DTU10MW" in their args, so checkpoint-driven
    # env rebuilds stay DTU automatically.
    turbtype: str = "IEA34"
    # IEA34 derating-table variant (helpers.derating_turbine): "annrpm"
    # (constant-Omega derating, rotor speed from the DLC12 RotSpd ANN — ct
    # consistent with the HF controller the load surrogates saw) or "minct"
    # (pure min-Ct, rotor speed free). Ignored for other turbtypes.
    iea34_variant: str = "annrpm"
    TI_type: str = "Random"   # Turbulence intensity sampling
    dt_sim: int = 5           # Simulation timestep (seconds)
    dt_env: int = 10          # Environment timestep (seconds)
    yaw_step: float = 5.0     # Max yaw change per sim step (degrees)
    # Derate slew limit toward the setpoint, in derate FRACTION per sim
    # substep (mirrors yaw_step_sim; windgym config key "derate_step_sim").
    # None = setpoint applies instantly (windgym default).
    derate_step_sim: Optional[float] = None
    max_eps: int = 20         # Number of flow passthroughs per episode
    num_envs: int = 1         # Number of parallel environments

    # === Evaluation Settings ===
    eval_interval: int = 50000        # How often to evaluate (in env steps)
    eval_initial: bool = False        # Run evaluation before training starts
    num_eval_steps: int = 200         # Number of steps per evaluation episode
    num_eval_episodes: int = 1        # Number of episodes per evaluation
    eval_layouts: str = ""            # Comma-separated eval layouts (empty = use training layouts)
    eval_seed: int = 42               # Seed for evaluation environments
    eval_deterministic: bool = True   # Use the deterministic (mean) policy action during evaluation
    # Named time-varying wd schedule(s) from the EVAL registry (helpers/wd_functions.py
    # WD_FUNCTIONS), applied to eval envs only. Comma-separated for multiple schedules
    # (e.g. "static_270,step_ramp_270_315"): each gets its own evaluator, and its
    # metrics are namespaced eval/wd/<schedule>/... . The FIRST schedule additionally
    # keeps the plain eval/... keys so existing W&B panels and readers still work.
    # When set, eval wind is pinned to wd_min=wd_max=wd_function(0) and ws=12 so the
    # burn-in matches the schedule's start. None = static eval wd (unchanged behavior).
    eval_wd_function: Optional[str] = None
    # Comma-separated eval wind speeds (m/s). Every --eval_wd_function schedule is
    # evaluated at EVERY listed speed, i.e. the eval ladder is the cross product
    # (schedule x ws) and each cell gets its own evaluator namespaced
    # eval/wd/<schedule>/ws<speed>/... . With a SINGLE speed (the default "12")
    # the /ws<speed> segment is omitted entirely, so the key namespace is
    # byte-identical to the pre-flag behaviour and change_wd_2 readers keep
    # working. Only consulted when --eval_wd_function is set (the fallback
    # static-wd evaluator does not pin wind at all).
    eval_ws: str = "12"

    # === Reward conditioning overrides (change_wd_3) ===
    # All None = "don't override", so the value from --config's power_def wins and
    # every pre-existing script keeps its exact behaviour. These exist because the
    # change_wd_3 arms need arbitrary COMBINATIONS of (tau x Power_avg x
    # Power_scaling x Power_reward), which as named ENV_CONFIGS presets would be a
    # dozen near-duplicate dicts.
    #
    # reward_tau floors the Wake_recovery denominator:
    #   r = (P_agent - P_greedy) / max(P_freestream - P_greedy, tau * P_freestream)
    # so it only binds in states with little wake-steering headroom (below rated,
    # or wd already aligned). Raising it DOWN-WEIGHTS those states relative to the
    # productive ones. Contrast power_scaling, which multiplies the reward
    # UNIFORMLY -- that difference is what makes power_scaling the magnitude
    # control that renders a tau effect attributable.
    reward_tau: Optional[float] = None      # power_def["tau"]; env default 0.02
    power_reward: Optional[str] = None      # power_def["Power_reward"]: "Baseline" | "Wake_recovery" | ...
    power_avg: Optional[int] = None         # power_def["Power_avg"]: reward power-averaging window
    power_scaling: Optional[float] = None   # power_def["Power_scaling"]: uniform reward gain

    # === Training wind-speed range override (change_wd_3) ===
    # Override wind.ws_min / wind.ws_max for the TRAINING envs only; eval envs are
    # always re-pinned to their own spec's ws afterwards, so these cannot leak into
    # the eval condition. Motivation: DTU10MW rates near 11.4 m/s while hard_2 draws
    # ws ~ U[10,14], leaving over half of training states with no wake-steering
    # headroom at all. Narrowing the range is the "stop sampling dead states" arm,
    # the alternative to reweighting them via reward_tau. None = use the config's range.
    train_ws_min: Optional[float] = None
    train_ws_max: Optional[float] = None

    # === Observation encoding (change_wd_4) ===
    # The OBS_SCALING.md finding: every ws feature is affine-scaled from a HARDCODED
    # 0-30 m/s (wind_farm_env.py:71-72 ctor defaults, never overridden anywhere),
    # while training data lives in ~6-14 m/s — the signal uses ~13% of the [-1,1]
    # axis. These flags are the change_wd_4 bake-off levers. All default to "off",
    # so every pre-existing script keeps its exact behaviour.
    #
    # WARNING: every one of these changes the observation contract, so they
    # invalidate all existing checkpoints, and the STANDALONE eval scripts
    # (evaluate.py / eval_checkpoint.py / ...) do NOT apply them — checkpoints from
    # runs using these flags are only comparable through the in-training eval until
    # those scripts learn to read the flags back from checkpoint["args"].
    #
    # ws_scaling_min/max are WindFarmEnv CTOR KWARGS (not config-dict keys — see
    # OBS_SCALING.md "Already ruled out"), forwarded via base_env_kwargs so train
    # and eval envs stay consistent. None = leave the env default (0/30) untouched.
    ws_scaling_min: Optional[float] = None
    ws_scaling_max: Optional[float] = None
    # Feature-map re-encodings of the ws columns, applied by ObsEncodingWrapper
    # (helpers/obs_encoding.py) on the PER-TURBINE obs. Modes: rbf | pyramid | cdf
    # | fourier | reldef | pcurve. Appending modes add features at the END of the
    # per-turbine vector so indices 0..11 keep their meaning; cdf warps the ws
    # columns in place. The wrapper reads the env's actual ws scaling range, so
    # combining with --ws_scaling_* stays correct (but don't — one lever per arm).
    obs_encoding: Optional[str] = None
    obs_encoding_kwargs: str = "{}"   # JSON overrides of a mode's defaults (precedent: profile_encoder_kwargs)
    # Agent-side running mean/std normalization (helpers/obs_norm.py), applied at
    # act() time and on replay batches. Agent-side (not a per-env wrapper) because
    # per-env statistics cannot sync across the 30 async workers and eval envs
    # would start cold. State rides in the checkpoint (obs_norm_state).
    obs_norm: bool = False
    # "shared" = the usual single obs-encoder MLP over the full per-turbine vector.
    # "per_sensor" = one small MLP per sensor group (ws/wd/yaw/power histories),
    # concatenated — requires obs_dim_per_turbine == 4*history_length, so it
    # hard-fails if combined with an expanding --obs_encoding (intended).
    obs_encoder_mode: str = "shared"

    # === Training wind-direction schedule ===
    # Named randomized wd schedule from the TRAIN registry (helpers/wd_functions.py
    # TRAIN_WD_FACTORIES, e.g. "dr_ramp") applied to TRAINING envs. These schedules are
    # RELATIVE -- wd(t) = base_wd + delta(t) with delta(0) = 0 -- so unlike the eval
    # path they do NOT pin wd_min/wd_max: the config's per-episode wd randomization is
    # preserved and the schedule composes on top of it. Each vector env is seeded
    # independently so the 30 envs do not share one wd trajectory. The train and eval
    # registries are disjoint, so an eval schedule name is rejected here (and vice
    # versa). None = static per-episode wd (unchanged behavior).
    train_wd_function: Optional[str] = None

    # === Wind-direction source (WD-estimation ladder, T3) ===
    # What feeds the agent's rotation machinery (wind-relative position
    # transform + profile rotation) AND the replay buffer's wind_directions:
    # "true" = the privileged env.wd scalar (historical behavior); "est" = the
    # sensor-derived env.wd_est (per-turbine circular EWMA + consensus,
    # WindGym/core/wd_estimator.py). "est" requires backend=dynamiks —
    # pywake's adapter hard-codes v=w=0, so no measured local wd exists there.
    wd_source: str = "true"
    # Estimator EWMA time constant (s); forwarded to the env when
    # wd_source="est". Pick from the T1 probe against the T0 error budget.
    wd_est_tau: Optional[float] = None
    # Cross-turbine consensus: median (robust default) / mean / front.
    wd_est_consensus: str = "median"

    # === Layout Settings ===
    # Comma-separated list of layouts. Single = single-layout, Multiple = multi-layout
    layouts: str = "test_layout"  # e.g., "square_1,square_2,circular_1"
    # Override padding / network size (max turbines). None = derive from layout pool.
    # Required for domain randomization so every config's network is sized for the
    # largest farm it must EVALUATE on (e.g. 25), regardless of training-pool size.
    max_turbines: Optional[int] = None

    # === Domain-Randomization (v8) ===
    # When dr_n_hi is set, training layouts are a procedurally-generated pool of
    # dr_pool_size irregular farms (min-spacing rejection sampling, like v4_irreg),
    # each episode sampling turbine count n ~ Uniform[dr_n_lo, dr_n_hi]. Replaces the
    # frozen named-layout pool to test whether layout DIVERSITY (not architecture or
    # entropy) unlocks large-farm learning. Pool is seeded from --seed so seeds differ.
    dr_n_lo: Optional[int] = None    # lower turbine-count bound (inclusive)
    dr_n_hi: Optional[int] = None    # upper turbine-count bound (inclusive); None = DR off
    dr_pool_size: int = 2048         # number of distinct layouts generated per run
    dr_min_dist_D: float = 3.0       # minimum turbine spacing in rotor diameters
    dr_screen_headroom: bool = True  # reject generated layouts with no wake-steering headroom
    dr_min_involved_frac: float = 0.5  # min fraction of turbines in a wake interaction to keep a layout
    dr_generator: str = "irregular"  # {"irregular","cluster","grid"}: procedural pool generator (cluster = PLayGen Poisson-disc; grid = rotated regular grids, dr_n_lo/hi bound nx*ny)

    # === Observation Settings ===
    history_length: int = 15            # Number of timesteps of history per feature
    use_wd_deviation: bool = False      # If True, convert WD to deviation from mean
    use_wind_relative_pos: bool = True  # Transform positions to wind-relative frame
    wd_scale_range: float = 45.0        # Only used if use_wd_deviation=True. Wind direction deviation range for scaling (±degrees → [-1,1]) (stage-1 baseline)

    # === Transformer Architecture ===
    embed_dim: int = 128          # Transformer hidden dimension
    num_heads: int = 4            # Number of attention heads
    num_layers: int = 2           # Number of transformer layers
    mlp_ratio: float = 2.0        # FFN hidden dim = embed_dim * mlp_ratio
    dropout: float = 0.0          # Dropout rate (0 for RL typically)
    pos_embed_dim: int = 32       # Dimension for positional encoding

    # === v5 attention-dilution / size-generalization knobs ===
    # Counteract softmax flattening as turbine count N grows (train small -> test large).
    attn_logit_scale: str = "none"   # "none" | "logn" (Scalable-Softmax: scores *= softplus(s_h)*log(N))
    attn_local: str = "none"         # "none" | "radius" | "knn" | "downwind" | "downwind_knn"
                                     #   radius/knn: undirected locality (v5).
                                     #   downwind[_knn]: v6 directed "causal wake graph" — attend only to
                                     #   UPWIND sources inside a cone of half-angle attn_local_cone_deg.
    attn_local_radius_D: float = 10.0  # neighbour radius in rotor diameters (radius / downwind streamwise cap)
    attn_local_k: int = 5            # number of nearest neighbours (knn / downwind_knn)
    attn_local_cone_deg: float = 40.0  # upwind cone half-angle in degrees (downwind / downwind_knn)
    attn_softmax: str = "softmax"    # "softmax" | "entmax15" (sparse; needs `entmax` pkg)


    # === Positional Encoding Settings ===
    # Options: "absolute_mlp", "relative_mlp", "relative_mlp_shared",
    #          "sinusoidal_2d",
    pos_encoding_type: Optional[str] = "relative_mlp"  # None for no pos encoding (stage-1 baseline: relative_mlp)
    # For relative encoding: number of hidden units in the bias MLP
    rel_pos_hidden_dim: int = 64
    # For relative encoding: whether to use separate bias per head
    rel_pos_per_head: bool = True
    pos_embedding_mode: str = "concat"  # "add" or "concat" positional embedding to token (only for absolute types)

    # === Algorithm Selection ===
    algorithm: str = "sac"  # "sac" or "tqc"
    use_droq: bool = False  # Enable DroQ regularization (dropout + LayerNorm in critic MLPs)

    # === TQC Hyperparameters (only used when algorithm="tqc") ===
    tqc_n_critics: int = 5               # Number of critic networks
    tqc_n_quantiles: int = 25            # Quantiles per critic
    tqc_top_quantiles_to_drop: int = 2   # Truncation: drop top-d per-sample quantiles
    tqc_share_trunk: bool = False        # ONE TransformerCritic trunk + tqc_n_critics small
    # quantile heads (TransformerTQCSharedCritic) instead of n_critics independent trunks.
    # ~4x fewer critic params / 2 trunk passes per grad-step instead of 2*n_critics; relies
    # on TQC's quantile truncation (not ensemble independence) to control overestimation.
    # Checkpoints are NOT interchangeable with the independent TQC critic.

    # === DroQ Hyperparameters (only used when use_droq=True) ===
    droq_dropout: float = 0.01           # Dropout rate for DroQ critic MLPs
    droq_layer_norm: bool = True         # LayerNorm in DroQ critic MLPs

    # === SAC Hyperparameters ===
    utd_ratio: float = 1.0           # Update-to-data ratio
    total_timesteps: int = 100_000
    buffer_size: int = int(1e6)
    gamma: float = 0.99           # Discount factor
    tau: float = 0.005            # Target network update rate
    batch_size: int = 256
    profile_registry_gpu_budget_mb: int = 256  # Keep profile registry GPU-resident if it fits
    learning_starts: int = 5000   # Steps before training starts
    policy_lr: float = 3e-4
    q_lr: float = 3e-4
    policy_frequency: int = 2     # Policy update frequency
    target_network_frequency: int = 1
    alpha: float = 0.2            # Initial entropy coefficient
    autotune: bool = True         # Auto-tune entropy coefficient
    entropy_agg: str = "sum"      # Per-farm entropy aggregation over turbines: "sum"
    # (standard SAC: log_pi summed -> O(N), target entropy -N) or "mean" (per-turbine
    # MEAN -> O(1), target entropy -1/dim). "mean" makes the entropy regularization
    # size-invariant so large farms are not pushed diffuse relative to the pooled farm-Q.
    critic_agg: str = "pool"      # Critic aggregation over turbines (v9): "pool" (standard:
    # masked-MEAN of turbine embeddings -> single farm-Q; per-turbine policy gradient ~1/N)
    # or "vdn" (value decomposition: per-turbine q_head -> masked-SUM -> farm-Q; removes the
    # structural 1/N so each turbine gets an un-diluted gradient). Pairs with entropy_agg="sum".
    reward_scale: float = 1.0    # Multiply the env reward by this (v9.1 probe). The Wake_recovery
    # reward is tiny (~0.02-0.10/step) -> small Q -> small gradients; scaling tests signal-to-noise.
    # Applied via a gymnasium reward wrapper in combined_wrapper; 1.0 = no change.

    # === DEL-constrained reward (baseline-relative max-DEL hinge penalty) ===
    # penalty = del_penalty_scale * max(0, DEL_agent_max/DEL_baseline_max
    #                                      - (1 + del_allowed_increase))
    # subtracted from the tracking reward BEFORE reward_scale (see
    # combined_wrapper). 0.0 disables the DELRewardWrapper entirely -- the env
    # is then built WITHOUT Baseline_comp, avoiding the doubled DWM cost.
    del_penalty_scale: float = 0.0     # lambda, in pre-reward_scale units
    del_allowed_increase: float = 0.10  # allowed fractional DEL increase over greedy baseline
    del_ti_window: float = 60.0        # trailing window (s) for sector statistics
    # DEL channel(s) the hinge penalty is computed on (CSV). One channel keeps
    # today's behavior; several make the penalty bind on the WORST channel:
    # ratio_c = farm-max agent / farm-max baseline per channel, penalty =
    # hinge(max_c ratio_c) with the shared episode limit. Channel names must
    # exist in the active turbine set (del_surrogate.SETS; e.g. wtow_H0FAMnt
    # is spelled H0FAMnt).
    del_channels: str = "Bl1Rad0FlpMnt"
    # Which del_surrogate artifact set to load. None (default) derives it from
    # turbtype (IEA34 -> "iea34", DTU10MW -> "dtu10mw"); set explicitly only
    # to cross-evaluate (e.g. DTU loads on an IEA34 farm — not meaningful for
    # training).
    del_artifact_set: Optional[str] = None
    # Attach the DEL wrapper even when del_penalty_scale == 0 (info-only:
    # penalty is exactly 0, reward untouched) so case-A runs log the same
    # charts/del_* metrics as penalized runs. Free with Power_reward="Baseline"
    # (the baseline farm already exists).
    del_log: bool = False
    # Goal-conditioned DEL limit: sample del_allowed_increase per episode
    # (uniform in [del_limit_lo, del_limit_hi]) and expose it as one extra
    # observation column per turbine (limit / del_limit_obs_ref). One policy
    # then covers the whole limit sweep; at eval a limit is pinned via
    # DELRewardWrapper(fixed_limit=...) / reset(options={"del_limit": x}).
    # Requires the DEL wrapper to be attached (del_penalty_scale > 0 or
    # del_log). del_allowed_increase is ignored while this is on.
    del_limit_random: bool = False   # sample the limit per episode; adds 1 obs column/turbine
    del_limit_lo: float = 0.0
    del_limit_hi: float = 0.3
    del_limit_obs_ref: float = 0.3   # obs normalization denominator; keep fixed across checkpoints

    # === PPO Hyperparameters (transformer_ppo_windfarm.py only) ===
    # Reused existing fields: gamma, total_timesteps, num_envs, policy_lr,
    # grad_clip/grad_clip_max_norm, plus all env/arch/profile/layout/eval/DEL
    # fields. Harmless defaults for the SAC trainer, which never reads these.
    num_steps: int = 256          # rollout length per env (batch = num_steps * num_envs)
    ppo_epochs: int = 10          # optimization epochs per rollout
    num_minibatches: int = 8      # minibatches per epoch
    clip_coef: float = 0.2        # PPO surrogate clipping epsilon
    ent_coef: float = 0.0         # entropy bonus coefficient (per-dim normalized)
    vf_coef: float = 0.5          # value loss coefficient
    gae_lambda: float = 0.95      # GAE lambda
    norm_adv: bool = True         # normalize advantages per minibatch
    clip_vloss: bool = True       # clipped value loss (CleanRL style)
    target_kl: Optional[float] = None  # early-stop epoch when approx_kl exceeds this
    anneal_lr: bool = True        # linear LR anneal over num_iterations
    # Shared actor-critic trunk (opt-in A/B vs the default separate value
    # net): the value head reads the ACTOR's trunk output (forward_trunk)
    # instead of owning its own transformer. actor_state_dict is unchanged
    # either way (eval/interp tooling and SAC warm-starts unaffected). With
    # sharing on, vf_coef * v_loss gradients flow into the policy trunk —
    # if the policy destabilizes, lower vf_coef or set
    # ppo_value_detach_trunk so the value loss only trains the head.
    ppo_share_trunk: bool = False        # value head reads the ACTOR's trunk output
    ppo_value_detach_trunk: bool = False # stop-grad: value loss doesn't shape trunk

    # === Gradient Clipping ===
    grad_clip: bool = True
    grad_clip_max_norm: float = 1.0

    # === Performance / Speed ===
    amp: bool = False        # Enable bfloat16 autocast (AMP) around the gradient updates
    compile: bool = False    # torch.compile the network forward passes (static shapes)
    compile_mode: str = "reduce-overhead"  # torch.compile mode; "default" disables cudagraphs (needed for single-rose arms)
    log_timing: bool = False  # Log a wall-clock breakdown (env step / sample / critic / actor) to TensorBoard
    # Overlap the SAC gradient burst with the (async) env step: step_async ->
    # gradient updates -> step_wait, so the AsyncVectorEnv workers simulate the
    # next step while the GPU trains. Iteration time ~ max(env, grad) instead of
    # the sum. Both modes run the burst on the same (one-iteration-lagged)
    # buffer contents, so loss traces are comparable; the flag only moves the
    # blocking point. SAC-only (PPO has its own loop).
    async_overlap: bool = False

    # === Fine-tuning / Resume Settings ===
    resume_checkpoint: Optional[str] = None  # Path to checkpoint .pt file for fine-tuning or resuming
    finetune_reset_actor_optimizer: int = 0     # If True, reset optimizers for fresh fine-tuning. If False, resume optimizer states too.
    finetune_reset_critic_optimizer: int = 0    # If True, reset optimizers for fresh fine-tuning. If False, resume optimizer states too.
    finetune_reset_alpha: int = 0               # If True, reset entropy coefficient. If False, keep from checkpoint.

    # === Initial Exploration Mode ===
    initial_exploration: str = "random"  # "random" = sample from action space, "policy" = use actor network (useful when resuming from checkpoint)

    # === Replay Buffer Save/Load ===
    load_buffer: Optional[str] = None           # Path to a saved replay buffer (.npz). Loading skips the exploration phase (learning_starts -> 0).
    save_buffer_at_learning_starts: bool = False  # Save the replay buffer once global_step reaches learning_starts (buffer pre-generation)
    buffer_only: bool = False                   # With save_buffer_at_learning_starts: exit right after saving (generation-only run)
    save_buffer_final: bool = False             # Save the replay buffer at the end of training (for splitting runs across cluster jobs)
    buffer_save_interval: int = 0               # If > 0, periodically save the buffer every N steps (overwrites runs/{run_name}/replay_buffer.npz). 0 = disabled.

    # === Pretrained Encoder Loading ===
    pretrain_checkpoint: Optional[str] = None   # Path to pretrained encoder .pt from pretrain_power.py
    pretrain_freeze_steps: int = 0             # Freeze encoder for this many env steps (0 = no freeze)

    # === Action Settings ===
    action_type: str = "wind"   # "wind" (target setpoint) or "yaw" (delta). Overridden by BC checkpoint if provided.
