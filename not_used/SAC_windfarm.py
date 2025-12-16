"""
MLP-based SAC for Wind Farm Control with Multi-Layout Support

This is a baseline implementation to compare against the transformer-based SAC.
The key difference is that this uses standard MLPs instead of transformers,
so it cannot explicitly model spatial relationships between turbines.

Since observations are padded to max_turbines, the MLP receives a fixed-size
input regardless of the actual farm layout. The MLP must learn to:
1. Ignore padded positions (marked by zeros or specific values)
2. Learn layout-agnostic policies (if that's even possible without spatial reasoning)

This serves as a baseline to measure how much benefit (if any) the transformer
architecture provides for generalization across wind farm layouts.

Author: Marcus (DTU Wind Energy)
Based on CleanRL's SAC implementation and transformer_sac_windfarm_v11.py
"""

import os
import random
import time
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

# WindGym imports
from WindGym import WindFarmEnv
from WindGym.wrappers import RecordEpisodeVals, PerTurbineObservationWrapper
from WindGym.utils.generate_layouts import (
    generate_square_grid, 
    generate_cirular_farm, 
    generate_right_triangle_grid,
    generate_line_dots_multiple_thetas
)
from MultiLayoutEnv import MultiLayoutEnv, LayoutConfig


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Args:
    """Command-line arguments for training."""
    
    # === Experiment Settings ===
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = True
    wandb_project_name: str = "transformer_windfarm"
    wandb_entity: Optional[str] = None
    save_model: bool = True
    save_interval: int = 25000
    log_image: bool = False  # Log attention images to TensorBoard
    shuffle_turbs: bool = False  # Shuffle turbine order in obs/action
    
    # === Environment Settings ===
    turbtype: str = "DTU10MW"
    TI_type: str = "Random"
    dt_sim: int = 5
    dt_env: int = 10
    yaw_step: float = 5.0
    max_eps: int = 20
    num_envs: int = 1
    
    # === Layout Settings ===
    # Same format as transformer version for direct comparison
    layouts: str = "test_layout"
    
    # === Observation Settings ===
    history_length: int = 15
    
    use_farm_token: bool = False  # Add learnable farm-level token
    wd_scale_range: float = 90.0  # Wind direction deviation range for scaling (±degrees → [-1,1])
    # === Transformer Architecture ===
    embed_dim: int = 128          # Transformer hidden dimension
    num_heads: int = 4            # Number of attention heads
    num_layers: int = 2           # Number of transformer layers
    mlp_ratio: float = 2.0        # FFN hidden dim = embed_dim * mlp_ratio
    dropout: float = 0.0          # Dropout rate (0 for RL typically)
    pos_embed_dim: int = 32       # Dimension for positional encoding
    # === Positional Encoding Settings ===
    # Options: "absolute_mlp", "relative_mlp", "relative_mlp_shared", 
    #          "sinusoidal_2d", "rope_2d"
    pos_encoding_type: str = "absolute_mlp"
    # For relative encoding: number of hidden units in the bias MLP
    rel_pos_hidden_dim: int = 64
    # For relative encoding: whether to use separate bias per head
    rel_pos_per_head: bool = True


    # === MLP Architecture ===
    hidden_dim: int = 256
    num_hidden_layers: int = 2
    
    # === SAC Hyperparameters ===
    utd_ratio: float = 1.0
    total_timesteps: int = 100_000
    buffer_size: int = int(1e6)
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    learning_starts: int = 5000
    policy_lr: float = 3e-4
    q_lr: float = 3e-4
    policy_frequency: int = 2
    target_network_frequency: int = 1
    alpha: float = 0.2
    autotune: bool = True
    
    # === Gradient Clipping ===
    grad_clip: bool = True
    grad_clip_max_norm: float = 1.0


# =============================================================================
# OBSERVATION PROCESSING
# =============================================================================

# NOTE: Unlike the transformer version, we do NOT use EnhancedPerTurbineWrapper here.
# 
# The transformer converts wind direction to DEVIATION because:
# - It encodes absolute wind direction in the positional encoding (wind-relative rotation)
# - Deviation just captures local wake/turbulence perturbations
#
# For the MLP baseline:
# - There's NO positional encoding, NO wind-relative rotation
# - The MLP needs ABSOLUTE wind direction to know where the wind comes from
# - Without it, the MLP would be blind to wind direction entirely!
#
# So we just use PerTurbineObservationWrapper directly, keeping raw wind direction values.


# =============================================================================
# MLP NETWORKS
# =============================================================================

LOG_STD_MAX = 2
LOG_STD_MIN = -5


class MLPActor(nn.Module):
    """
    Standard MLP actor for SAC.
    
    Takes flattened observations (n_turbines * obs_dim_per_turbine) and outputs
    flattened actions (n_turbines * action_dim_per_turbine).
    
    Unlike the transformer, this has no explicit spatial structure - it must
    learn any spatial relationships from the data alone.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_hidden_layers: int = 2,
        action_scale: float = 1.0,
        action_bias: float = 0.0,
    ):
        super().__init__()
        
        # Build MLP
        layers = [nn.Linear(obs_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        
        self.net = nn.Sequential(*layers)
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.fc_logstd = nn.Linear(hidden_dim, action_dim)
        
        # Action scaling
        self.register_buffer("action_scale", torch.tensor(action_scale, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor(action_bias, dtype=torch.float32))
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: (batch, obs_dim) flattened observations
        
        Returns:
            mean: (batch, action_dim)
            log_std: (batch, action_dim)
        """
        h = self.net(obs)
        mean = self.fc_mean(h)
        log_std = self.fc_logstd(h)
        
        # Constrain log_std
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        
        return mean, log_std
    
    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.
        
        Args:
            obs: (batch, obs_dim)
            deterministic: If True, return mean action
        
        Returns:
            action: (batch, action_dim)
            log_prob: (batch, 1)
            mean_action: (batch, action_dim)
        """
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        
        normal = torch.distributions.Normal(mean, std)
        if deterministic:
            x_t = mean
        else:
            x_t = normal.rsample()
        
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        
        # Log probability with tanh correction
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        
        return action, log_prob, mean_action


class MLPCritic(nn.Module):
    """
    Standard MLP critic (Q-function) for SAC.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_hidden_layers: int = 2,
    ):
        super().__init__()
        
        # Build MLP
        layers = [nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (batch, obs_dim)
            action: (batch, action_dim)
        
        Returns:
            q_value: (batch, 1)
        """
        x = torch.cat([obs, action], dim=-1)
        return self.net(x)


# =============================================================================
# REPLAY BUFFER
# =============================================================================

class SimpleReplayBuffer:
    """
    Simple replay buffer for flattened observations/actions.
    
    Unlike the transformer version, we don't need to store positions or
    attention masks separately - everything is flattened.
    """
    
    def __init__(self, capacity: int, obs_dim: int, action_dim: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Pre-allocate storage
        self.observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        
        self.position = 0
        self.size = 0
    
    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
    ) -> None:
        """Store a transition."""
        self.observations[self.position] = obs.flatten()
        self.next_observations[self.position] = next_obs.flatten()
        self.actions[self.position] = action.flatten()
        self.rewards[self.position] = reward
        self.dones[self.position] = done
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a batch of transitions."""
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        return {
            "observations": torch.tensor(self.observations[indices], device=self.device),
            "next_observations": torch.tensor(self.next_observations[indices], device=self.device),
            "actions": torch.tensor(self.actions[indices], device=self.device),
            "rewards": torch.tensor(self.rewards[indices], device=self.device),
            "dones": torch.tensor(self.dones[indices], device=self.device),
        }
    
    def __len__(self) -> int:
        return self.size


# =============================================================================
# ENVIRONMENT FACTORY
# =============================================================================

def make_env_config() -> Dict[str, Any]:
    """Create base environment configuration (same as transformer version)."""
    return {
        "yaw_init": "Random",
        "BaseController": "Local",
        "ActionMethod": "yaw",
        "Track_power": False,
        "farm": {"yaw_min": -30, "yaw_max": 30},
        "wind": {
            "ws_min": 10, "ws_max": 10,
            "TI_min": 0.07, "TI_max": 0.07,
            "wd_min": 260, "wd_max": 280,
        },
        "act_pen": {"action_penalty": 0.0, "action_penalty_type": "Change"},
        "power_def": {"Power_reward": "Baseline", "Power_avg": 1, "Power_scaling": 1.0},
        "mes_level": {
            "turb_ws": True,
            "turb_wd": True,
            "turb_TI": False,
            "turb_power": True,
            "farm_ws": False,
            "farm_wd": False,
            "farm_TI": False,
            "farm_power": False,
        },
        "ws_mes": {
            "ws_current": False,
            "ws_rolling_mean": True,
            "ws_history_N": 15,
            "ws_history_length": 15,
            "ws_window_length": 1,
        },
        "wd_mes": {
            "wd_current": False,
            "wd_rolling_mean": True,
            "wd_history_N": 15,
            "wd_history_length": 15,
            "wd_window_length": 1,
        },
        "yaw_mes": {
            "yaw_current": False,
            "yaw_rolling_mean": True,
            "yaw_history_N": 15,
            "yaw_history_length": 15,
            "yaw_window_length": 1,
        },
        "power_mes": {
            "power_current": False,
            "power_rolling_mean": True,
            "power_history_N": 15,
            "power_history_length": 15,
            "power_window_length": 1,
        },
    }


def get_layout_positions(layout_type: str, wind_turbine) -> Tuple[np.ndarray, np.ndarray]:
    """Get turbine positions for a given layout type (same as transformer version)."""
    layouts = {
        "test_layout": lambda: generate_square_grid(turbine=wind_turbine, nx=2, ny=1, xDist=5, yDist=5),
        "square_2x2": lambda: generate_square_grid(turbine=wind_turbine, nx=2, ny=2, xDist=5, yDist=5),
        "small_triangle": lambda: generate_right_triangle_grid(turbine=wind_turbine, nx=2, ny=3, xDist=5, yDist=5),
        "square_3x3": lambda: generate_square_grid(turbine=wind_turbine, nx=3, ny=3, xDist=5, yDist=5),
        "circular_6": lambda: generate_cirular_farm(n_list=[1, 5], turbine=wind_turbine, r_dist=5),
        "circular_10": lambda: generate_cirular_farm(n_list=[3, 7], turbine=wind_turbine, r_dist=5),
        "tri1": lambda: generate_right_triangle_grid(turbine=wind_turbine, nx=2, ny=2, xDist=5, yDist=5, orientation='lower_left'),
        "tri2": lambda: generate_right_triangle_grid(turbine=wind_turbine, nx=2, ny=2, xDist=5, yDist=5, orientation='lower_right'),
        "tri3": lambda: generate_right_triangle_grid(turbine=wind_turbine, nx=2, ny=2, xDist=5, yDist=5, orientation='upper_left'),
        "tri4": lambda: generate_right_triangle_grid(turbine=wind_turbine, nx=2, ny=2, xDist=5, yDist=5, orientation='upper_right'),
        "5turb1": lambda: generate_line_dots_multiple_thetas(X=3, spacing=5, thetas=[0, 30], turbine=wind_turbine),
        "5turb2": lambda: generate_line_dots_multiple_thetas(X=3, spacing=5, thetas=[0, -30], turbine=wind_turbine),
        "5turb3": lambda: generate_line_dots_multiple_thetas(X=3, spacing=5, thetas=[-30, 30], turbine=wind_turbine),
    }
    
    if layout_type not in layouts:
        raise ValueError(f"Unknown layout type: {layout_type}. Available: {list(layouts.keys())}")
    
    return layouts[layout_type]()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def save_checkpoint(
    actor: nn.Module,
    qf1: nn.Module,
    qf2: nn.Module,
    actor_optimizer: optim.Optimizer,
    q_optimizer: optim.Optimizer,
    step: int,
    run_name: str,
    args: Args,
    log_alpha: Optional[torch.Tensor] = None,
    alpha_optimizer: Optional[optim.Optimizer] = None,
) -> str:
    """Save training checkpoint."""
    checkpoint_dir = f"runs/{run_name}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_path = f"{checkpoint_dir}/step_{step}.pt"
    
    checkpoint = {
        "step": step,
        "actor_state_dict": actor.state_dict(),
        "qf1_state_dict": qf1.state_dict(),
        "qf2_state_dict": qf2.state_dict(),
        "actor_optimizer_state_dict": actor_optimizer.state_dict(),
        "q_optimizer_state_dict": q_optimizer.state_dict(),
        "args": vars(args),
    }
    
    if log_alpha is not None:
        checkpoint["log_alpha"] = log_alpha.detach().cpu()
    if alpha_optimizer is not None:
        checkpoint["alpha_optimizer_state_dict"] = alpha_optimizer.state_dict()
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")
    
    return checkpoint_path


# =============================================================================
# MAIN TRAINING SCRIPT
# =============================================================================

def main():
    """Main training function."""
    
    args = tyro.cli(Args)
    
    # Parse layouts
    layout_names = [l.strip() for l in args.layouts.split(",")]
    is_multi_layout = len(layout_names) > 1
    
    # Create run name
    run_name = f"{args.exp_name}_{'_'.join(layout_names)}_{args.seed}"
    
    print("=" * 60)
    print("MLP SAC Baseline for Wind Farm Control")
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
    
    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")
    
    # =========================================================================
    # ENVIRONMENT SETUP
    # =========================================================================
    
    # Wind turbine
    if args.turbtype == "DTU10MW":
        from py_wake.examples.data.dtu10mw import DTU10MW as WT
    elif args.turbtype == "V80":
        from py_wake.examples.data.hornsrev1 import V80 as WT
    else:
        raise ValueError(f"Unknown turbine type: {args.turbtype}")
    
    wind_turbine = WT()
    
    # Create layout configurations
    layouts = []
    for name in layout_names:
        x_pos, y_pos = get_layout_positions(name, wind_turbine)
        layouts.append(LayoutConfig(name=name, x_pos=x_pos, y_pos=y_pos))
    
    # Environment configuration
    config = make_env_config()
    
    # Update history length
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
        "TurbBox": "/work/users/manils/rl_timestep/Boxes/V80env/",
        "config": config,
        "turbtype": args.TI_type,
        "dt_sim": args.dt_sim,
        "dt_env": args.dt_env,
        "yaw_step_sim": args.yaw_step,
    }
    
    def env_factory(x_pos: np.ndarray, y_pos: np.ndarray) -> gym.Env:
        env = WindFarmEnv(x_pos=x_pos, y_pos=y_pos, **base_env_kwargs)
        env.action_space.seed(args.seed)
        return env
    
    def per_turbine_wrapper(env: gym.Env) -> gym.Env:
        """
        Wrapper for per-turbine observations.
        
        Unlike transformer version, we do NOT transform wind direction to deviation.
        The MLP needs absolute wind direction since it has no positional encoding.
        """
        return PerTurbineObservationWrapper(env)
    
    def make_env_fn(seed):
        def _init():
            env = MultiLayoutEnv(
                layouts=layouts,
                env_factory=env_factory,
                per_turbine_wrapper=per_turbine_wrapper,
                seed=seed,
                shuffle=args.shuffle_turbs, # Shuffle turbines within each layout
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
    
    # Get dimensions from sample env
    sample_env = make_env_fn(args.seed)()
    sample_obs, sample_info = sample_env.reset()
    
    n_turbines_max = sample_env.max_turbines
    obs_dim_per_turbine = sample_obs.shape[1]
    action_dim_per_turbine = 1
    
    # Flattened dimensions for MLP
    obs_dim = n_turbines_max * obs_dim_per_turbine
    action_dim = n_turbines_max * action_dim_per_turbine
    
    sample_env.close()
    
    print(f"Max turbines: {n_turbines_max}")
    print(f"Obs dim per turbine: {obs_dim_per_turbine}")
    print(f"Total obs dim (flattened): {obs_dim}")
    print(f"Total action dim (flattened): {action_dim}")
    
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
                "n_layouts": len(layout_names),
                "layout_names": layout_names,
                "is_multi_layout": is_multi_layout,
                "max_turbines": n_turbines_max,
                "architecture": "MLP",
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
    
    print("\nCreating MLP networks...")
    
    actor = MLPActor(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.num_hidden_layers,
        action_scale=action_scale,
        action_bias=action_bias,
    ).to(device)
    
    qf1 = MLPCritic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.num_hidden_layers,
    ).to(device)
    
    qf2 = MLPCritic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.num_hidden_layers,
    ).to(device)
    
    # Target networks
    qf1_target = MLPCritic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.num_hidden_layers,
    ).to(device)
    
    qf2_target = MLPCritic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.num_hidden_layers,
    ).to(device)
    
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    
    # Count parameters
    actor_params = sum(p.numel() for p in actor.parameters())
    critic_params = sum(p.numel() for p in qf1.parameters())
    print(f"Actor parameters: {actor_params:,}")
    print(f"Critic parameters: {critic_params:,} (x2)")
    
    # Optimizers
    actor_optimizer = optim.Adam(actor.parameters(), lr=args.policy_lr)
    q_optimizer = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()),
        lr=args.q_lr
    )
    
    # Entropy tuning
    if args.autotune:
        # Target entropy based on action dimension
        target_entropy = -action_dim
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        alpha_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha
    
    # =========================================================================
    # REPLAY BUFFER
    # =========================================================================
    
    rb = SimpleReplayBuffer(
        capacity=args.buffer_size,
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device,
    )
    
    # =========================================================================
    # TRAINING LOOP
    # =========================================================================
    
    print("\nStarting training...")
    start_time = time.time()
    
    obs, _ = envs.reset(seed=args.seed)
    # Flatten observation: (num_envs, n_turbines, obs_dim_per_turb) -> (num_envs, obs_dim)
    obs_flat = obs.reshape(args.num_envs, -1)
    
    global_step = 0
    total_gradient_steps = 0
    next_save_step = args.save_interval
    
    num_updates = args.total_timesteps // args.num_envs
    
    # For logging
    step_reward_window = deque(maxlen=1000)
    episode_rewards = []
    
    for update in range(1, num_updates + 1):
        global_step += args.num_envs
        
        # =====================================================================
        # COLLECT EXPERIENCE
        # =====================================================================
        
        if global_step < args.learning_starts:
            # Random actions during warmup
            actions = np.array([envs.single_action_space.sample() for _ in range(args.num_envs)])
            actions_flat = actions.reshape(args.num_envs, -1)
        else:
            # Sample from policy
            with torch.no_grad():
                obs_tensor = torch.tensor(obs_flat, device=device, dtype=torch.float32)
                actions_flat, _, _ = actor.get_action(obs_tensor)
                actions_flat = actions_flat.cpu().numpy()
        
        # Reshape actions for environment: (num_envs, action_dim) -> (num_envs, n_turbines, 1)
        actions = actions_flat.reshape(args.num_envs, n_turbines_max, action_dim_per_turbine)
        
        # Environment step
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        next_obs_flat = next_obs.reshape(args.num_envs, -1)
        
        # Log episode info (using RecordEpisodeVals wrapper queues)
        if "final_info" in infos:
            # Use the wrapper's queues which are more reliable
            if hasattr(envs, 'return_queue') and len(envs.return_queue) > 0:
                mean_return = np.mean(envs.return_queue)
                episode_rewards.append(mean_return)
                writer.add_scalar("charts/episodic_return", mean_return, global_step)
            if hasattr(envs, 'length_queue') and len(envs.length_queue) > 0:
                writer.add_scalar("charts/episodic_length", np.mean(envs.length_queue), global_step)
            if hasattr(envs, 'mean_power_queue') and len(envs.mean_power_queue) > 0:
                writer.add_scalar("charts/episodic_power", np.mean(envs.mean_power_queue), global_step)
        
        # Track step rewards
        for r in rewards:
            step_reward_window.append(r)
        
        # Handle truncated episodes
        real_next_obs_flat = next_obs_flat.copy()
        for idx, trunc in enumerate(truncations):
            if trunc and "final_obs" in infos:
                final_obs = infos["final_obs"][idx]
                real_next_obs_flat[idx] = final_obs.flatten()
        
        # Add to replay buffer
        for i in range(args.num_envs):
            rb.add(
                obs_flat[i],
                real_next_obs_flat[i],
                actions_flat[i],
                rewards[i],
                terminations[i],
            )
        
        obs_flat = next_obs_flat
        
        # =====================================================================
        # TRAINING
        # =====================================================================
        
        if global_step > args.learning_starts:
            # Calculate number of gradient updates based on UTD ratio
            num_gradient_updates = max(1, int(args.num_envs * args.utd_ratio))
            
            for _ in range(num_gradient_updates):
                data = rb.sample(args.batch_size)
                
                # ---------------------------------------------------------
                # Update Critics
                # ---------------------------------------------------------
                with torch.no_grad():
                    next_actions, next_log_pi, _ = actor.get_action(data["next_observations"])
                    qf1_next = qf1_target(data["next_observations"], next_actions)
                    qf2_next = qf2_target(data["next_observations"], next_actions)
                    min_qf_next = torch.min(qf1_next, qf2_next) - alpha * next_log_pi
                    next_q_value = data["rewards"] + (1 - data["dones"]) * args.gamma * min_qf_next
                
                qf1_value = qf1(data["observations"], data["actions"])
                qf2_value = qf2(data["observations"], data["actions"])
                
                qf1_loss = F.mse_loss(qf1_value, next_q_value)
                qf2_loss = F.mse_loss(qf2_value, next_q_value)
                qf_loss = qf1_loss + qf2_loss
                
                q_optimizer.zero_grad()
                qf_loss.backward()
                if args.grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                        list(qf1.parameters()) + list(qf2.parameters()),
                        max_norm=args.grad_clip_max_norm
                    )
                q_optimizer.step()
                
                # ---------------------------------------------------------
                # Update Actor (delayed)
                # ---------------------------------------------------------
                if total_gradient_steps % args.policy_frequency == 0:
                    actions_pi, log_pi, _ = actor.get_action(data["observations"])
                    qf1_pi = qf1(data["observations"], actions_pi)
                    qf2_pi = qf2(data["observations"], actions_pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    
                    actor_loss = (alpha * log_pi - min_qf_pi).mean()
                    
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    if args.grad_clip:
                        torch.nn.utils.clip_grad_norm_(
                            actor.parameters(),
                            max_norm=args.grad_clip_max_norm
                        )
                    actor_optimizer.step()
                    
                    # Update Alpha
                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data["observations"])
                        
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()
                        
                        alpha_optimizer.zero_grad()
                        alpha_loss.backward()
                        alpha_optimizer.step()
                        alpha = log_alpha.exp().item()
                
                # ---------------------------------------------------------
                # Update Target Networks
                # ---------------------------------------------------------
                if total_gradient_steps % args.target_network_frequency == 0:
                    for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                        target_param.data.copy_(
                            args.tau * param.data + (1 - args.tau) * target_param.data
                        )
                    for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                        target_param.data.copy_(
                            args.tau * param.data + (1 - args.tau) * target_param.data
                        )
                
                total_gradient_steps += 1
            
            # -----------------------------------------------------------------
            # Logging
            # -----------------------------------------------------------------
            if update % 20 == 0:
                sps = int(global_step / (time.time() - start_time))
                mean_reward = float(np.mean(step_reward_window)) if step_reward_window else 0.0
                
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                writer.add_scalar("charts/SPS", sps, global_step)
                writer.add_scalar("charts/step_reward_mean_1000", mean_reward, global_step)
                writer.add_scalar("debug/total_gradient_steps", total_gradient_steps, global_step)
                
                print(f"Step {global_step}: SPS={sps}, qf_loss={qf_loss.item()/2:.4f}, "
                      f"actor_loss={actor_loss.item():.4f}, alpha={alpha:.4f}, "
                      f"reward_mean={mean_reward:.4f}")
        
        # =====================================================================
        # CHECKPOINTING
        # =====================================================================
        
        if args.save_model and global_step >= next_save_step:
            save_checkpoint(
                actor, qf1, qf2, actor_optimizer, q_optimizer,
                global_step, run_name, args, log_alpha if args.autotune else None,
                alpha_optimizer if args.autotune else None
            )
            next_save_step += args.save_interval
    
    # =========================================================================
    # FINAL SAVE AND CLEANUP
    # =========================================================================
    
    if args.save_model:
        save_checkpoint(
            actor, qf1, qf2, actor_optimizer, q_optimizer,
            global_step, run_name, args, log_alpha if args.autotune else None,
            alpha_optimizer if args.autotune else None
        )
    
    print("\n" + "=" * 60)
    print("Training finished!")
    print(f"Total time: {(time.time() - start_time) / 3600:.2f} hours")
    print("=" * 60)
    
    envs.close()
    writer.close()


if __name__ == "__main__":
    main()