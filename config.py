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

    shuffle_turbs: bool = False  # Shuffle turbine order in obs/action
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
    profile_encoder_kwargs: str = "{}"  # JSON string of encoder-specific kwargs
    profile_source: str = "PyWake"  # "pywake" or "geometric"
    profile_encoding_type: Optional[str] = None  # Now Optional, use None for no pos encoding
    profile_encoder_hidden: int = 128       # Hidden dim in profile encoder MLP
    rotate_profiles: bool = True            # Rotate profiles to wind-relative frame
    n_profile_directions: int = 360         # Number of directions in profile
    profile_fusion_type: str = "add"       # "add" or "joint" fusion of receptivity and influence profiles
    profile_embed_mode: str = "add"        # "add" or "concat" — how fused profile is integrated into token embedding
    share_profile_encoder: bool = False         # Whether to share weights between actor and critic for profile encoder

    # === Environment Settings ===
    backend: str = "dynamiks"  # Flow solver backend: "dynamiks" (default) or "pywake" (steady-state)
    turbtype: str = "DTU10MW"  # Wind turbine type
    TI_type: str = "Random"   # Turbulence intensity sampling
    dt_sim: int = 5           # Simulation timestep (seconds)
    dt_env: int = 10          # Environment timestep (seconds)
    yaw_step: float = 5.0     # Max yaw change per sim step (degrees)
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
    dr_generator: str = "irregular"  # {"irregular","cluster"}: procedural pool generator (cluster = PLayGen Poisson-disc)

    # === Observation Settings ===
    history_length: int = 15            # Number of timesteps of history per feature
    use_wd_deviation: bool = False      # If True, convert WD to deviation from mean
    use_wind_relative_pos: bool = True  # Transform positions to wind-relative frame
    wd_scale_range: float = 90.0        # Only used if use_wd_deviation=True. Wind direction deviation range for scaling (±degrees → [-1,1])

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
    pos_encoding_type: Optional[str] = None  # Now Optional, use None for no pos encoding
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

    # === Gradient Clipping ===
    grad_clip: bool = True
    grad_clip_max_norm: float = 1.0

    # === Performance / Speed ===
    amp: bool = False        # Enable bfloat16 autocast (AMP) around the gradient updates
    compile: bool = False    # torch.compile the network forward passes (static shapes)
    compile_update: bool = False  # (SAC only, requires --compile) compile the WHOLE update step
                                  # (forward+loss) so AOTAutograd also graphs the backward pass --
                                  # the dominant eager cost on the launch-bound update loop. Replaces
                                  # per-forward compile for SAC; assumes --amp off.
    log_timing: bool = False  # Log a wall-clock breakdown (env step / sample / critic / actor) to TensorBoard

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
