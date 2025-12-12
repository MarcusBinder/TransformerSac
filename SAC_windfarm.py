"""
MLP-based SAC for Wind Farm Control with Position Features

This is a modified baseline that adds turbine position features to observations,
enabling training across different wind farm layouts with variable numbers of turbines.

Key differences from the transformer version:
1. Positions are concatenated to observations (not used in attention)
2. Everything is flattened for MLP processing
3. No explicit spatial reasoning - MLP must learn any spatial patterns from data

Key features:
- Wind-relative position transformation (same canonical frame as transformer)
- Padding for variable-size farms
- Position normalization by rotor diameter
- Supports multi-layout training

This allows direct comparison of:
- MLP with positions vs Transformer with positions
- Whether explicit spatial reasoning (transformer) helps generalization

Author: Marcus (DTU Wind Energy)
Based on transformer_sac_windfarm_v12.py position handling
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
from MultiLayoutEnv import MultiLayoutEnv, LayoutConfig
from helper_funcs import (
    get_layout_positions,
    get_env_wind_directions,
    get_env_raw_positions,
    get_env_attention_masks,
    save_checkpoint,
    make_env_config,
    transform_to_wind_relative_numpy,
    prepare_observation_with_positions,
)

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
    save_interval: int = 10000
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
    layouts: str = "test_layout"
    
    # === Observation Settings ===
    history_length: int = 15
    wd_scale_range: float = 90.0  # Wind direction deviation range for scaling
    include_positions: bool = False  # NEW: Include position features in observations
    max_turb: Optional[int] = None  # Override max turbines (for training on small farms, evaluating on larger)
    
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
# MLP NETWORKS
# =============================================================================

LOG_STD_MAX = 2
LOG_STD_MIN = -5


class MLPActor(nn.Module):
    """
    MLP actor with position features concatenated to observations.
    
    Input: [obs_features, position_features] flattened
    Output: flattened actions for all turbines (including padded)
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
            obs: (batch, obs_dim) flattened observations with positions
        
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
        action_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.

        Args:
            obs: (batch, obs_dim)
            action_mask: (batch, max_turbines) boolean mask where True = padding (ignore).
                           If None, all turbines are considered valid.
            deterministic: If True, return mean action

        Returns:
            action: (batch, action_dim)
            log_prob: (batch, 1) - sum of log probs for VALID turbines only
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

        # Mask out padded turbines before summing
        if action_mask is not None:
            # action_mask: True = padding (ignore), False = valid
            valid_mask = ~action_mask  # True = valid turbine
            log_prob = log_prob * valid_mask.float()  # Zero out padded turbine log probs

        log_prob = log_prob.sum(dim=-1, keepdim=True)

        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob, mean_action


class MLPCritic(nn.Module):
    """MLP critic (Q-function) with position features."""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_hidden_layers: int = 2,
    ):
        super().__init__()
        
        # Build MLP: [obs + action] -> Q-value
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
# REPLAY BUFFER WITH POSITIONS
# =============================================================================

class PositionAwareReplayBuffer:
    """
    Replay buffer that stores raw positions and wind direction.
    
    Wind-relative transformation is applied at sample time to ensure
    correct positional encoding regardless of when the transition was collected.
    
    This matches the transformer version's replay buffer behavior.
    """
    
    def __init__(
        self,
        capacity: int,
        device: torch.device,
        rotor_diameter: float,
        max_turbines: int,
        obs_dim_per_turbine: int,
        include_positions: bool = True,
    ):
        """
        Args:
            capacity: Maximum number of transitions
            device: Torch device for sampled tensors
            rotor_diameter: For position normalization
            max_turbines: Maximum turbines across all layouts
            obs_dim_per_turbine: Observation dim per turbine (without positions)
            include_positions: Whether to include position features
        """
        self.capacity = capacity
        self.device = device
        self.rotor_diameter = rotor_diameter
        self.max_turbines = max_turbines
        self.obs_dim_per_turbine = obs_dim_per_turbine
        self.include_positions = include_positions
        
        # Position features: 2 (x, y) per turbine
        self.pos_dim_per_turbine = 2 if include_positions else 0
        
        self.buffer = []
        self.position = 0
    
    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        raw_positions: np.ndarray,
        action_mask: np.ndarray,
        wind_direction: float
    ) -> None:
        """Store a transition."""
        data = (obs, next_obs, action, reward, done, raw_positions, action_mask, wind_direction)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.position] = data
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample batch and apply wind-relative transformation.
        
        Returns observations with position features concatenated.
        """
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        # Unpack batch
        obs_list, next_obs_list, action_list = [], [], []
        raw_positions_list, mask_list, wind_dirs = [], [], []
        rewards, dones = [], []
        
        for obs, next_obs, action, reward, done, raw_pos, mask, wind_dir in batch:
            obs_list.append(obs)
            next_obs_list.append(next_obs)
            action_list.append(action)
            raw_positions_list.append(raw_pos)
            mask_list.append(mask)
            wind_dirs.append(wind_dir)
            rewards.append(reward)
            dones.append(done)
        
        # Stack to arrays
        obs_array = np.stack(obs_list)  # (batch, max_turb, obs_dim)
        next_obs_array = np.stack(next_obs_list)
        raw_positions = np.stack(raw_positions_list)  # (batch, max_turb, 2)
        wind_directions = np.array(wind_dirs)  # (batch,)
        
        if self.include_positions:
            # Normalize positions by rotor diameter
            positions_norm = raw_positions / self.rotor_diameter
            
            # Apply wind-relative transformation
            positions_transformed = transform_to_wind_relative_numpy(
                positions_norm, wind_directions
            )  # (batch, max_turb, 2)
            
            # Concatenate positions to observations
            # obs: (batch, max_turb, obs_dim) -> (batch, max_turb, obs_dim + 2)
            obs_with_pos = np.concatenate([obs_array, positions_transformed], axis=-1)
            next_obs_with_pos = np.concatenate([next_obs_array, positions_transformed], axis=-1)
            
            # Flatten for MLP
            obs_flat = obs_with_pos.reshape(batch_size, -1)
            next_obs_flat = next_obs_with_pos.reshape(batch_size, -1)
        else:
            # Just flatten without positions
            obs_flat = obs_array.reshape(batch_size, -1)
            next_obs_flat = next_obs_array.reshape(batch_size, -1)
        
        # Flatten actions
        actions_flat = np.stack(action_list).reshape(batch_size, -1)
        
        return {
            "observations": torch.tensor(obs_flat, device=self.device, dtype=torch.float32),
            "next_observations": torch.tensor(next_obs_flat, device=self.device, dtype=torch.float32),
            "actions": torch.tensor(actions_flat, device=self.device, dtype=torch.float32),
            "action_mask": torch.tensor(np.stack(mask_list), device=self.device, dtype=torch.bool),
            "rewards": torch.tensor(rewards, device=self.device, dtype=torch.float32).unsqueeze(-1),
            "dones": torch.tensor(dones, device=self.device, dtype=torch.float32).unsqueeze(-1),
        }
    
    def __len__(self) -> int:
        return len(self.buffer)


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
    pos_suffix = "_withpos" if args.include_positions else "_nopos"
    run_name = f"{args.exp_name}{'_'.join(layout_names)}{pos_suffix}_{args.seed}"
    
    print("=" * 60)
    print("MLP SAC Baseline with Position Features")
    print("=" * 60)
    print(f"Include positions: {args.include_positions}")
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
        """Wrapper for per-turbine observations."""
        return PerTurbineObservationWrapper(env)
    
    def make_env_fn(args, seed):
        def _init():
            env = MultiLayoutEnv(
                layouts=layouts,
                env_factory=env_factory,
                per_turbine_wrapper=per_turbine_wrapper,
                seed=seed,
                shuffle=args.shuffle_turbs,
                max_turbines=args.max_turb,  # Override max turbines if specified
            )
            return env
        return _init
    
    # Create vectorized environments
    print(f"Creating {args.num_envs} parallel environment(s)...")
    envs = gym.vector.AsyncVectorEnv(
        [make_env_fn(args, args.seed + i) for i in range(args.num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    envs = RecordEpisodeVals(envs)
    
    # Get dimensions from sample env (wrapper handles max_turb override)
    sample_env = make_env_fn(args, args.seed)()
    sample_obs, sample_info = sample_env.reset()

    n_turbines_max = sample_env.max_turbines  # Already includes max_turb override if specified
    obs_dim_per_turbine = sample_obs.shape[1]
    action_dim_per_turbine = 1
    rotor_diameter = sample_env.rotor_diameter
    
    # Position features: 2 per turbine (x, y)
    pos_dim_per_turbine = 2 if args.include_positions else 0
    
    # Total flattened dimensions
    obs_dim = n_turbines_max * (obs_dim_per_turbine + pos_dim_per_turbine)
    action_dim = n_turbines_max * action_dim_per_turbine
    
    sample_env.close()

    print(f"Max turbines: {n_turbines_max}")
    print(f"Obs dim per turbine: {obs_dim_per_turbine}")
    print(f"Position dim per turbine: {pos_dim_per_turbine}")
    print(f"Total obs dim (flattened): {obs_dim}")
    print(f"Total action dim (flattened): {action_dim}")
    print(f"Rotor diameter: {rotor_diameter:.1f} m")
    
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
                "architecture": "MLP_with_positions" if args.include_positions else "MLP",
                "rotor_diameter": rotor_diameter,
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
    # Note: target_entropy is computed per-sample based on actual turbine count
    # to handle variable-size layouts correctly
    if args.autotune:
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        alpha_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha
    
    # =========================================================================
    # REPLAY BUFFER
    # =========================================================================
    
    rb = PositionAwareReplayBuffer(
        capacity=args.buffer_size,
        device=device,
        rotor_diameter=rotor_diameter,
        max_turbines=n_turbines_max,
        obs_dim_per_turbine=obs_dim_per_turbine,
        include_positions=args.include_positions,
    )
    
    # =========================================================================
    # TRAINING LOOP
    # =========================================================================
    
    print("\nStarting training...")
    start_time = time.time()
    
    obs, _ = envs.reset(seed=args.seed)

    global_step = 0
    total_gradient_steps = 0
    next_save_step = args.save_interval
    
    num_updates = args.total_timesteps // args.num_envs
    
    # For logging
    step_reward_window = deque(maxlen=1000)
    episode_rewards = []
    

    save_checkpoint(
        actor, qf1, qf2, actor_optimizer, q_optimizer,
        0, run_name, args, log_alpha if args.autotune else None,
        alpha_optimizer if args.autotune else None
    )

    for update in range(1, num_updates + 2):
        global_step += args.num_envs
        
        # Get environment info for position processing
        wind_dirs = get_env_wind_directions(envs, args.num_envs)
        raw_positions = get_env_raw_positions(envs, args.num_envs, n_turbines_max)
        # Note that for the SAC without attention, this mask is more an action mask, but we reuse the function
        current_masks = get_env_attention_masks(envs, args.num_envs, n_turbines_max)
        
        # Prepare observation with positions
        obs_flat = prepare_observation_with_positions(
            obs, raw_positions, wind_dirs, rotor_diameter, args.include_positions
        )
        
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
                mask_tensor = torch.tensor(current_masks, device=device, dtype=torch.bool)
                actions_flat, _, _ = actor.get_action(obs_tensor, action_mask=mask_tensor)
                actions_flat = actions_flat.cpu().numpy()
        
        # Reshape actions for environment (wrapper handles slicing for actual turbines)
        actions = actions_flat.reshape(args.num_envs, n_turbines_max, action_dim_per_turbine)

        # Environment step
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # Log episode info
        if "final_info" in infos:
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
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc and "final_obs" in infos:
                real_next_obs[idx] = infos["final_obs"][idx]
        
        # Add to replay buffer (store raw positions and wind direction)
        for i in range(args.num_envs):
            done = terminations[i] or truncations[i]
            rb.add(
                obs[i],
                real_next_obs[i],
                actions_flat[i].reshape(-1),
                rewards[i],
                done,
                raw_positions[i],
                current_masks[i],
                wind_dirs[i]
            )
        
        obs = next_obs
        
        # =====================================================================
        # TRAINING
        # =====================================================================
        
        if global_step > args.learning_starts:
            num_gradient_updates = max(1, int(args.num_envs * args.utd_ratio))
            
            for _ in range(num_gradient_updates):
                data = rb.sample(args.batch_size)
                
                # ---------------------------------------------------------
                # Update Critics
                # ---------------------------------------------------------
                with torch.no_grad():
                    next_actions, next_log_pi, _ = actor.get_action(
                        data["next_observations"], action_mask=data["action_mask"]
                    )
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
                    actions_pi, log_pi, _ = actor.get_action(
                        data["observations"], action_mask=data["action_mask"]
                    )
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
                            _, log_pi, _ = actor.get_action(
                                data["observations"], action_mask=data["action_mask"]
                            )

                        # Compute per-sample target entropy based on actual turbine count
                        # action_mask: True = padding, False = valid turbine
                        actual_turbines = (~data["action_mask"]).sum(dim=-1).float()  # (batch,)
                        target_entropy = -actual_turbines  # (batch,)

                        # log_pi: (batch, 1), target_entropy: (batch,)
                        alpha_loss = (-log_alpha.exp() * (log_pi.squeeze(-1) + target_entropy)).mean()

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
                writer.add_scalar("debug/mean_wind_direction", float(np.mean(wind_dirs)), global_step)

                # Log masking info for multi-layout debugging
                mean_actual_turbs = float((~data["action_mask"]).sum(dim=-1).float().mean())
                writer.add_scalar("debug/mean_actual_turbines", mean_actual_turbs, global_step)
                if args.autotune:
                    writer.add_scalar("debug/mean_target_entropy", -mean_actual_turbs, global_step)

                print(f"Step {global_step}: SPS={sps}, qf_loss={qf_loss.item()/2:.4f}, "
                      f"actor_loss={actor_loss.item():.4f}, alpha={alpha:.4f}, "
                      f"reward_mean={mean_reward:.4f}, turbs={mean_actual_turbs:.1f}")
        
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