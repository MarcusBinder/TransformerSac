"""
Transformer-based SAC for Wind Farm Control

This implementation uses a transformer architecture to enable:
1. Training across variable-size wind farms
2. Zero-shot transfer to unseen farm configurations
3. Interpretable attention patterns that may correlate with wake interactions

Key design choices:
- Per-turbine tokenization: each turbine is a token with local observations
- Wind-relative positional encoding: positions rotated so wind comes from fixed direction
- Positions normalized by rotor diameter (physics-meaningful scale)
- Shared actor/critic heads across turbines (permutation equivariant)
"""

import os
import random
import time
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

# WindGym imports (adjust as needed)
from windgym.WindGym import WindFarmEnv
from windgym.WindGym.wrappers import RecordEpisodeVals, PerTurbineObservationWrapper
from windgym.WindGym.utils.generate_layouts import generate_square_grid, generate_cirular_farm


@dataclass
class Args:
    # Experiment settings
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = True
    wandb_project_name: str = "transformer_windfarm"
    wandb_entity: str = None
    save_model: bool = True
    save_interval: int = 25000

    # Environment settings
    max_eps: int = 20
    turbtype: str = "DTU10MW"
    TI_type: str = "Random"
    dt_sim: int = 5
    dt_env: int = 10
    yaw_step: float = 5.0
    
    # Multi-farm training: comma-separated list of layouts
    # e.g., "square_1,square_2,circular_1" for multi-task training
    train_layouts: str = "test_layout"  # Use "test_layout" for single layout testing
    eval_layout: str = "test_layout"  # For periodic evaluation
    
    # Transformer architecture
    obs_dim_per_turbine: int = 9  # Will be set by env, but provide default
    action_dim_per_turbine: int = 1  # Yaw control
    embed_dim: int = 128  # Transformer hidden dimension
    num_heads: int = 4
    num_layers: int = 2
    mlp_ratio: float = 2.0  # FFN hidden dim = embed_dim * mlp_ratio
    dropout: float = 0.0
    
    # Positional encoding
    pos_embed_dim: int = 32  # Dimension for position encoding
    use_wind_relative_pos: bool = True  # Transform positions to wind-relative coords
    
    # SAC hyperparameters
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


# =============================================================================
# Positional Encoding
# =============================================================================

class PositionalEncoding(nn.Module):
    """
    Encodes turbine (x, y) positions into embeddings.
    
    Positions are expected to be normalized by rotor diameter and optionally
    transformed to wind-relative coordinates.
    """
    def __init__(self, pos_dim: int = 2, embed_dim: int = 32):
        super().__init__()
        self.pos_encoder = nn.Sequential(
            nn.Linear(pos_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
    
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions: (batch, n_turbines, 2) normalized positions
        Returns:
            (batch, n_turbines, embed_dim) position embeddings
        """
        return self.pos_encoder(positions)


def transform_to_wind_relative(
    positions: torch.Tensor, 
    wind_direction: torch.Tensor
) -> torch.Tensor:
    """
    Transform positions to wind-relative coordinates.
    
    After transformation, wind comes from the negative x direction (270° met convention).
    This makes "upwind" = negative x, "downwind" = positive x.
    
    Args:
        positions: (batch, n_turbines, 2) absolute positions
        wind_direction: (batch,) or (batch, 1) wind direction in degrees (met convention)
    
    Returns:
        (batch, n_turbines, 2) wind-relative positions
    """
    # Meteorological convention: wind_direction is where wind comes FROM
    # 270° means wind comes from West, blows toward East
    # We want to rotate so wind always comes from 270° (negative x)
    
    # Rotation angle (how much to rotate the farm)
    # If wind is from 280°, we rotate farm by -10° so wind appears from 270°
    angle_offset = wind_direction - 270.0  # degrees
    theta = angle_offset * (math.pi / 180.0)  # to radians
    
    if theta.dim() == 1:
        theta = theta.unsqueeze(-1).unsqueeze(-1)  # (batch, 1, 1)
    
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    
    # Rotation matrix (applied to each position)
    x = positions[..., 0:1]
    y = positions[..., 1:2]
    
    x_rot = cos_theta * x - sin_theta * y
    y_rot = sin_theta * x + cos_theta * y
    
    return torch.cat([x_rot, y_rot], dim=-1)


# =============================================================================
# Transformer Blocks
# =============================================================================

class TransformerEncoderLayer(nn.Module):
    """Standard transformer encoder layer with pre-norm."""
    
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 2.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, n_turbines, embed_dim)
            attn_mask: optional attention mask
        Returns:
            x: (batch, n_turbines, embed_dim)
            attn_weights: (batch, n_turbines, n_turbines)
        """
        # Self-attention with residual
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.attn(x_norm, x_norm, x_norm, attn_mask=attn_mask)
        x = x + attn_out
        
        # FFN with residual
        x = x + self.mlp(self.norm2(x))
        
        return x, attn_weights


class TransformerEncoder(nn.Module):
    """Stack of transformer encoder layers."""
    
    def __init__(self, embed_dim: int, num_heads: int, num_layers: int, mlp_ratio: float = 2.0, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Returns:
            x: (batch, n_turbines, embed_dim)
            all_attn_weights: list of attention weights from each layer
        """
        all_attn_weights = []
        for layer in self.layers:
            x, attn_weights = layer(x, attn_mask)
            all_attn_weights.append(attn_weights)
        x = self.norm(x)
        return x, all_attn_weights


# =============================================================================
# Actor (Policy) Network
# =============================================================================

LOG_STD_MAX = 2
LOG_STD_MIN = -5


class TransformerActor(nn.Module):
    """
    Transformer-based actor for wind farm control.
    
    Each turbine is a token. The transformer learns turbine-turbine interactions,
    and a shared action head outputs per-turbine actions.
    """
    
    def __init__(
        self,
        obs_dim_per_turbine: int,
        action_dim_per_turbine: int,
        embed_dim: int = 128,
        pos_embed_dim: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        action_scale: float = 1.0,
        action_bias: float = 0.0,
    ):
        super().__init__()
        
        self.obs_dim_per_turbine = obs_dim_per_turbine
        self.action_dim_per_turbine = action_dim_per_turbine
        self.embed_dim = embed_dim
        
        # Per-turbine observation encoder (shared across turbines)
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim_per_turbine, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(pos_dim=2, embed_dim=pos_embed_dim)
        
        # Project concatenated obs + pos encoding to embed_dim
        self.input_proj = nn.Linear(embed_dim + pos_embed_dim, embed_dim)
        
        # Transformer encoder
        self.transformer = TransformerEncoder(embed_dim, num_heads, num_layers, mlp_ratio, dropout)
        
        # Action head (shared across turbines)
        self.fc_mean = nn.Linear(embed_dim, action_dim_per_turbine)
        self.fc_logstd = nn.Linear(embed_dim, action_dim_per_turbine)
        
        # Action scaling
        self.register_buffer("action_scale", torch.tensor(action_scale, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor(action_bias, dtype=torch.float32))
    
    def forward(
        self, 
        obs: torch.Tensor, 
        positions: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            obs: (batch, n_turbines, obs_dim_per_turbine)
            positions: (batch, n_turbines, 2) normalized wind-relative positions
            attn_mask: optional attention mask
        
        Returns:
            mean: (batch, n_turbines, action_dim_per_turbine)
            log_std: (batch, n_turbines, action_dim_per_turbine)
            attn_weights: list of attention weights from each layer
        """
        batch_size, n_turbines, _ = obs.shape
        
        # Encode observations
        h = self.obs_encoder(obs)  # (batch, n_turbines, embed_dim)
        
        # Encode positions
        pos_embed = self.pos_encoder(positions)  # (batch, n_turbines, pos_embed_dim)
        
        # Concatenate and project
        h = torch.cat([h, pos_embed], dim=-1)
        h = self.input_proj(h)  # (batch, n_turbines, embed_dim)
        
        # Transformer
        h, attn_weights = self.transformer(h, attn_mask)
        
        # Action head (applied per-turbine)
        mean = self.fc_mean(h)
        log_std = self.fc_logstd(h)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        
        return mean, log_std, attn_weights
    
    def get_action(
        self,
        obs: torch.Tensor,
        positions: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Sample action from policy.
        
        Returns:
            action: (batch, n_turbines, action_dim) or flattened (batch, n_turbines * action_dim)
            log_prob: (batch, 1)
            mean: (batch, n_turbines, action_dim)
            attn_weights: attention weights for visualization
        """
        mean, log_std, attn_weights = self.forward(obs, positions, attn_mask)
        std = log_std.exp()
        
        # Sample action
        normal = torch.distributions.Normal(mean, std)
        if deterministic:
            x_t = mean
        else:
            x_t = normal.rsample()
        
        # Squash with tanh
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        
        # Log probability (with tanh correction)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=(-2, -1), keepdim=True)  # Sum over turbines and action dims
        
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        
        return action, log_prob, mean_action, attn_weights


# =============================================================================
# Critic (Q) Network
# =============================================================================

class TransformerCritic(nn.Module):
    """
    Transformer-based critic for wind farm control.
    
    Processes (observation, action) pairs per turbine, then pools to get a single Q-value.
    """
    
    def __init__(
        self,
        obs_dim_per_turbine: int,
        action_dim_per_turbine: int,
        embed_dim: int = 128,
        pos_embed_dim: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        # Per-turbine (obs, action) encoder
        self.obs_action_encoder = nn.Sequential(
            nn.Linear(obs_dim_per_turbine + action_dim_per_turbine, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(pos_dim=2, embed_dim=pos_embed_dim)
        
        # Project concatenated input to embed_dim
        self.input_proj = nn.Linear(embed_dim + pos_embed_dim, embed_dim)
        
        # Transformer encoder
        self.transformer = TransformerEncoder(embed_dim, num_heads, num_layers, mlp_ratio, dropout)
        
        # Q-value head: pool over turbines, then predict Q
        self.q_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )
    
    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        positions: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            obs: (batch, n_turbines, obs_dim_per_turbine)
            action: (batch, n_turbines, action_dim_per_turbine)
            positions: (batch, n_turbines, 2)
            attn_mask: optional
        
        Returns:
            q_value: (batch, 1)
        """
        # Concatenate obs and action per turbine
        x = torch.cat([obs, action], dim=-1)
        
        # Encode
        h = self.obs_action_encoder(x)
        
        # Position encoding
        pos_embed = self.pos_encoder(positions)
        
        # Concatenate and project
        h = torch.cat([h, pos_embed], dim=-1)
        h = self.input_proj(h)
        
        # Transformer
        h, _ = self.transformer(h, attn_mask)
        
        # Pool over turbines (mean pooling)
        h_pooled = h.mean(dim=1)  # (batch, embed_dim)
        
        # Q-value
        q = self.q_head(h_pooled)
        
        return q


# =============================================================================
# Environment Factory
# =============================================================================

def make_env_config():
    """Base environment configuration."""
    return {
        "yaw_init": "Random",
        "BaseController": "Local",
        "ActionMethod": "yaw",
        "Track_power": False,
        "farm": {"yaw_min": -30, "yaw_max": 30},
        "wind": {
            "ws_min": 9, "ws_max": 9,
            "TI_min": 0.05, "TI_max": 0.05,
            "wd_min": 265, "wd_max": 275,
        },
        "act_pen": {"action_penalty": 0.0, "action_penalty_type": "Change"},
        "power_def": {"Power_reward": "Power_avg", "Power_avg": 1, "Power_scaling": 1.0},
        "mes_level": {
            "turb_ws": True, "turb_wd": True, "turb_TI": False, "turb_power": False,
            "farm_ws": False, "farm_wd": False, "farm_TI": False, "farm_power": False,
        },
        "ws_mes": {"ws_current": False, "ws_rolling_mean": True, "ws_history_N": 3, "ws_history_length": 3, "ws_window_length": 1},
        "wd_mes": {"wd_current": False, "wd_rolling_mean": True, "wd_history_N": 3, "wd_history_length": 3, "wd_window_length": 1},
        "yaw_mes": {"yaw_current": False, "yaw_rolling_mean": True, "yaw_history_N": 3, "yaw_history_length": 3, "yaw_window_length": 1},
        "power_mes": {"power_current": False, "power_rolling_mean": False, "power_history_N": 1, "power_history_length": 1, "power_window_length": 1},
    }


def get_layout_positions(layout_type: str, wind_turbine):
    """Get turbine positions for a given layout type."""
    if layout_type == "square_1":
        return generate_square_grid(turbine=wind_turbine, nx=2, ny=2, xDist=5, yDist=5)
    elif layout_type == "square_2":
        return generate_square_grid(turbine=wind_turbine, nx=3, ny=3, xDist=5, yDist=5)
    elif layout_type == "circular_1":
        return generate_cirular_farm(n_list=[1, 5], turbine=wind_turbine, r_dist=5)
    elif layout_type == "circular_2":
        return generate_cirular_farm(n_list=[3, 7], turbine=wind_turbine, r_dist=5)
    elif layout_type == "test_layout":
        return generate_square_grid(turbine=wind_turbine, nx=2, ny=1, xDist=5, yDist=5)
    else:
        raise ValueError(f"Unknown layout type: {layout_type}")


def make_env(layout_type: str, args, wind_turbine, seed: int = 0):
    """Create a single environment with the specified layout."""
    x_pos, y_pos = get_layout_positions(layout_type, wind_turbine)
    config = make_env_config()
    
    env = WindFarmEnv(
        turbine=wind_turbine,
        n_passthrough=args.max_eps,
        x_pos=x_pos,
        y_pos=y_pos,
        TurbBox="/work/users/manils/rl_timestep/Boxes/V80env/",
        config=config,
        turbtype=args.TI_type,
        dt_sim=args.dt_sim,
        dt_env=args.dt_env,
        yaw_step_sim=args.yaw_step,
    )
    env.action_space.seed(seed)
    
    # Wrap with per-turbine observation wrapper
    env = PerTurbineObservationWrapper(env)
    
    return env


class MultiLayoutEnv(gym.Env):
    """
    Meta-environment that randomly samples from multiple farm layouts each episode.
    
    This enables multi-task training across different farm configurations.
    """
    
    def __init__(self, layout_types: List[str], args, wind_turbine, seed: int = 0):
        self.layout_types = layout_types
        self.args = args
        self.wind_turbine = wind_turbine
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        # Create all environments
        self.envs = {
            layout: make_env(layout, args, wind_turbine, seed)
            for layout in layout_types
        }
        
        # Current active environment
        self.current_layout = self.rng.choice(layout_types)
        self.current_env = self.envs[self.current_layout]
        
        # Set spaces based on current env (will vary!)
        self._update_spaces()
    
    def _update_spaces(self):
        self.observation_space = self.current_env.observation_space
        self.action_space = self.current_env.action_space
    
    @property
    def n_turbines(self):
        return self.current_env.n_turbines
    
    @property
    def turbine_positions(self):
        """Returns (n_turbines, 2) array of positions."""
        return self.current_env.turbine_positions
    
    @property
    def rotor_diameter(self):
        return self.current_env.rotor_diameter
    
    @property
    def mean_wind_direction(self):
        """Returns current mean wind direction for position transformation."""
        return self.current_env.mean_wind_direction
    
    def reset(self, **kwargs):
        # Sample new layout for this episode
        self.current_layout = self.rng.choice(self.layout_types)
        self.current_env = self.envs[self.current_layout]
        self._update_spaces()
        return self.current_env.reset(**kwargs)
    
    def step(self, action):
        return self.current_env.step(action)
    
    def close(self):
        for env in self.envs.values():
            env.close()


# =============================================================================
# Replay Buffer for Variable-Size Observations
# =============================================================================

class TransformerReplayBuffer:
    """
    Replay buffer that handles variable-size observations and positions.
    
    Stores data per-transition with the turbine count, then pads when sampling.
    """
    
    def __init__(self, capacity: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.buffer = []
        self.position = 0
    
    def add(
        self,
        obs: np.ndarray,  # (n_turbines, obs_dim)
        next_obs: np.ndarray,
        action: np.ndarray,  # (n_turbines, action_dim)
        reward: float,
        done: bool,
        positions: np.ndarray,  # (n_turbines, 2)
    ):
        data = (obs, next_obs, action, reward, done, positions)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.position] = data
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int):
        """
        Sample a batch and pad to the maximum turbine count in the batch.
        
        Returns dict with padded tensors and attention mask.
        """
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        # Find max turbines in this batch
        max_turbines = max(b[0].shape[0] for b in batch)
        
        obs_list, next_obs_list, action_list, positions_list = [], [], [], []
        rewards, dones, masks = [], [], []
        
        for obs, next_obs, action, reward, done, positions in batch:
            n_turb = obs.shape[0]
            obs_dim = obs.shape[1]
            action_dim = action.shape[1]
            
            # Pad observations
            obs_padded = np.zeros((max_turbines, obs_dim), dtype=np.float32)
            obs_padded[:n_turb] = obs
            obs_list.append(obs_padded)
            
            next_obs_padded = np.zeros((max_turbines, obs_dim), dtype=np.float32)
            next_obs_padded[:n_turb] = next_obs
            next_obs_list.append(next_obs_padded)
            
            # Pad actions
            action_padded = np.zeros((max_turbines, action_dim), dtype=np.float32)
            action_padded[:n_turb] = action
            action_list.append(action_padded)
            
            # Pad positions
            pos_padded = np.zeros((max_turbines, 2), dtype=np.float32)
            pos_padded[:n_turb] = positions
            positions_list.append(pos_padded)
            
            # Create attention mask (True = masked/ignored)
            mask = np.ones(max_turbines, dtype=bool)
            mask[:n_turb] = False
            masks.append(mask)
            
            rewards.append(reward)
            dones.append(done)
        
        return {
            "observations": torch.tensor(np.stack(obs_list), device=self.device),
            "next_observations": torch.tensor(np.stack(next_obs_list), device=self.device),
            "actions": torch.tensor(np.stack(action_list), device=self.device),
            "positions": torch.tensor(np.stack(positions_list), device=self.device),
            "rewards": torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(-1),
            "dones": torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(-1),
            "attn_mask": torch.tensor(np.stack(masks), device=self.device),  # (batch, max_turbines)
        }
    
    def __len__(self):
        return len(self.buffer)


# =============================================================================
# Training Script
# =============================================================================

def save_model(actor, qf1, qf2, step_number, run_name, exp_name):
    model_path = f"runs/{run_name}/{exp_name}_{step_number}.pt"
    torch.save({
        "actor": actor.state_dict(),
        "qf1": qf1.state_dict(),
        "qf2": qf2.state_dict(),
    }, model_path)
    print(f"Model saved to {model_path}")


def get_positions_tensor(env, device, wind_relative: bool = True):
    """
    Get normalized turbine positions as a tensor.
    
    Positions are normalized by rotor diameter and optionally transformed
    to wind-relative coordinates.
    """
    positions = env.turbine_positions  # (n_turbines, 2)
    rotor_d = env.rotor_diameter
    
    # Normalize by rotor diameter
    positions_norm = positions / rotor_d
    
    # Convert to tensor
    positions_tensor = torch.tensor(positions_norm, dtype=torch.float32, device=device)
    positions_tensor = positions_tensor.unsqueeze(0)  # (1, n_turbines, 2)
    
    if wind_relative:
        wind_dir = env.mean_wind_direction  # scalar or array
        wind_dir_tensor = torch.tensor([wind_dir], dtype=torch.float32, device=device)
        positions_tensor = transform_to_wind_relative(positions_tensor, wind_dir_tensor)
    
    return positions_tensor


if __name__ == "__main__":
    import stable_baselines3 as sb3
    
    if sb3.__version__ < "2.0":
        raise ValueError("Need stable_baselines3 >= 2.0")
    
    args = tyro.cli(Args)
    
    # Parse layout list
    train_layouts = [l.strip() for l in args.train_layouts.split(",")]
    print(f"Training on layouts: {train_layouts}")
    
    run_name = f"{args.exp_name}_{'_'.join(train_layouts)}"
    os.makedirs(f"runs/{run_name}", exist_ok=True)
    
    # Tracking
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            save_code=True,
        )
    
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text("hyperparameters", "|param|value|\n|-|-|\n" + "\n".join([f"|{k}|{v}|" for k, v in vars(args).items()]))
    
    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")
    
    # Wind turbine
    if args.turbtype == "DTU10MW":
        from py_wake.examples.data.dtu10mw import DTU10MW as WT
    elif args.turbtype == "V80":
        from py_wake.examples.data.hornsrev1 import V80 as WT

    wind_turbine = WT()
    
    # Create environment(s)
    if len(train_layouts) == 1:
        env = make_env(train_layouts[0], args, wind_turbine, args.seed)
    else:
        env = MultiLayoutEnv(train_layouts, args, wind_turbine, args.seed)
    
    env = RecordEpisodeVals(env)
    
    # Get dimensions from env
    sample_obs, _ = env.reset()
    n_turbines_sample = sample_obs.shape[0]
    obs_dim_per_turbine = sample_obs.shape[1]
    action_dim_per_turbine = env.action_space.shape[0] // n_turbines_sample  # Assuming flat action space
    
    print(f"Sample farm: {n_turbines_sample} turbines, obs_dim={obs_dim_per_turbine}, action_dim={action_dim_per_turbine}")
    
    # Action scaling (assuming symmetric action space for yaw)
    action_high = env.action_space.high[0]
    action_low = env.action_space.low[0]
    action_scale = (action_high - action_low) / 2.0
    action_bias = (action_high + action_low) / 2.0
    
    # Create networks
    actor = TransformerActor(
        obs_dim_per_turbine=obs_dim_per_turbine,
        action_dim_per_turbine=action_dim_per_turbine,
        embed_dim=args.embed_dim,
        pos_embed_dim=args.pos_embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        action_scale=action_scale,
        action_bias=action_bias,
    ).to(device)
    
    qf1 = TransformerCritic(
        obs_dim_per_turbine=obs_dim_per_turbine,
        action_dim_per_turbine=action_dim_per_turbine,
        embed_dim=args.embed_dim,
        pos_embed_dim=args.pos_embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
    ).to(device)
    
    qf2 = TransformerCritic(
        obs_dim_per_turbine=obs_dim_per_turbine,
        action_dim_per_turbine=action_dim_per_turbine,
        embed_dim=args.embed_dim,
        pos_embed_dim=args.pos_embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
    ).to(device)
    
    qf1_target = TransformerCritic(
        obs_dim_per_turbine=obs_dim_per_turbine,
        action_dim_per_turbine=action_dim_per_turbine,
        embed_dim=args.embed_dim,
        pos_embed_dim=args.pos_embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
    ).to(device)
    
    qf2_target = TransformerCritic(
        obs_dim_per_turbine=obs_dim_per_turbine,
        action_dim_per_turbine=action_dim_per_turbine,
        embed_dim=args.embed_dim,
        pos_embed_dim=args.pos_embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
    ).to(device)
    
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    
    # Optimizers
    actor_optimizer = optim.Adam(actor.parameters(), lr=args.policy_lr)
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    
    # Entropy tuning
    if args.autotune:
        # Target entropy: -dim(A) per turbine, scaled by expected number of turbines
        # For variable sizes, use a reasonable estimate
        target_entropy = -action_dim_per_turbine * n_turbines_sample
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        alpha_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha
    
    # Replay buffer
    rb = TransformerReplayBuffer(args.buffer_size, device)
    
    # Training loop
    start_time = time.time()
    global_step = 0
    
    obs, _ = env.reset(seed=args.seed)
    positions = get_positions_tensor(env, device, args.use_wind_relative_pos)
    
    print(f"\nStarting training for {args.total_timesteps} timesteps...")
    
    for step in range(args.total_timesteps):
        global_step += 1
        
        # Get action
        if global_step < args.learning_starts:
            # Random action
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action_tensor, _, _, _ = actor.get_action(obs_tensor, positions)
                action = action_tensor.squeeze(0).cpu().numpy()
                # Flatten if needed for env
                action = action.reshape(-1)
        
        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Store transition
        positions_np = positions.squeeze(0).cpu().numpy()
        action_reshaped = action.reshape(-1, action_dim_per_turbine)
        rb.add(obs, next_obs, action_reshaped, reward, done, positions_np)
        
        obs = next_obs
        
        # Handle episode end
        if done:
            if hasattr(env, 'return_queue') and len(env.return_queue) > 0:
                writer.add_scalar("charts/episodic_return", np.mean(env.return_queue), global_step)
                writer.add_scalar("charts/episodic_power", np.mean(env.mean_power_queue), global_step)
                print(f"Step {global_step}: return={np.mean(env.return_queue):.2f}, power={np.mean(env.mean_power_queue):.2f}")
            
            obs, _ = env.reset()
            positions = get_positions_tensor(env, device, args.use_wind_relative_pos)
        
        # Training
        if global_step > args.learning_starts and len(rb) >= args.batch_size:
            data = rb.sample(args.batch_size)
            
            # Compute target Q-values
            with torch.no_grad():
                next_actions, next_log_pi, _, _ = actor.get_action(
                    data["next_observations"], 
                    data["positions"]
                )
                qf1_next = qf1_target(data["next_observations"], next_actions, data["positions"])
                qf2_next = qf2_target(data["next_observations"], next_actions, data["positions"])
                min_qf_next = torch.min(qf1_next, qf2_next) - alpha * next_log_pi
                next_q_value = data["rewards"] + (1 - data["dones"]) * args.gamma * min_qf_next
            
            # Update critics
            qf1_value = qf1(data["observations"], data["actions"], data["positions"])
            qf2_value = qf2(data["observations"], data["actions"], data["positions"])
            qf1_loss = F.mse_loss(qf1_value, next_q_value)
            qf2_loss = F.mse_loss(qf2_value, next_q_value)
            qf_loss = qf1_loss + qf2_loss
            
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()
            
            # Update actor (delayed)
            if global_step % args.policy_frequency == 0:
                for _ in range(args.policy_frequency):
                    actions_pi, log_pi, _, _ = actor.get_action(data["observations"], data["positions"])
                    qf1_pi = qf1(data["observations"], actions_pi, data["positions"])
                    qf2_pi = qf2(data["observations"], actions_pi, data["positions"])
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = (alpha * log_pi - min_qf_pi).mean()
                    
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()
                    
                    # Update alpha
                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _, _ = actor.get_action(data["observations"], data["positions"])
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()
                        
                        alpha_optimizer.zero_grad()
                        alpha_loss.backward()
                        alpha_optimizer.step()
                        alpha = log_alpha.exp().item()
            
            # Update target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
            
            # Logging
            if global_step % 1000 == 0:
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                sps = int(global_step / (time.time() - start_time))
                writer.add_scalar("charts/SPS", sps, global_step)
                print(f"Step {global_step}: SPS={sps}, qf_loss={qf_loss.item():.4f}, actor_loss={actor_loss.item():.4f}")
        
        # Save model
        if args.save_model and global_step % args.save_interval == 0:
            save_model(actor, qf1, qf2, global_step, run_name, args.exp_name)
    
    # Final save
    if args.save_model:
        save_model(actor, qf1, qf2, global_step, run_name, args.exp_name)
    
    print("Training finished!")
    env.close()
    writer.close()