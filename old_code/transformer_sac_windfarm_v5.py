"""
LIKE V4, but uses changable wind farms. 
"""

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
from typing import Optional, Tuple, List, Dict

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
from collections import deque

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
    num_envs: int = 1  # Number of parallel environments
    
    # Layout settings
    # For single layout: use layout_type
    # For random layouts: use random_layouts (comma-separated list)
    layout_type: str = "test_layout"  # Used when random_layouts is empty
    random_layouts: str = ""  # e.g., "square_1,square_2,circular_1" for multi-layout training
    max_turbines: int = 16  # Maximum turbines for padding (only used with random_layouts)
    
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
    """Encodes turbine (x, y) positions into embeddings."""
    def __init__(self, pos_dim: int = 2, embed_dim: int = 32):
        super().__init__()
        self.pos_encoder = nn.Sequential(
            nn.Linear(pos_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
    
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        return self.pos_encoder(positions)


def transform_to_wind_relative(
    positions: torch.Tensor, 
    wind_direction: torch.Tensor
) -> torch.Tensor:
    """Transform positions to wind-relative coordinates."""
    angle_offset = wind_direction - 270.0
    theta = angle_offset * (math.pi / 180.0)
    
    if theta.dim() == 1:
        theta = theta.unsqueeze(-1).unsqueeze(-1)
    
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    
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
    
    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, n_turbines, embed_dim)
            key_padding_mask: (batch, n_turbines) where True = ignore this position
        """
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.attn(
            x_norm, x_norm, x_norm, 
            key_padding_mask=key_padding_mask
        )
        x = x + attn_out
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
    
    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        all_attn_weights = []
        for layer in self.layers:
            x, attn_weights = layer(x, key_padding_mask)
            all_attn_weights.append(attn_weights)
        x = self.norm(x)
        return x, all_attn_weights


# =============================================================================
# Actor (Policy) Network
# =============================================================================

LOG_STD_MAX = 2
LOG_STD_MIN = -5


class TransformerActor(nn.Module):
    """Transformer-based actor for wind farm control."""
    
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
        
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim_per_turbine, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        
        self.pos_encoder = PositionalEncoding(pos_dim=2, embed_dim=pos_embed_dim)
        self.input_proj = nn.Linear(embed_dim + pos_embed_dim, embed_dim)
        self.transformer = TransformerEncoder(embed_dim, num_heads, num_layers, mlp_ratio, dropout)
        
        self.fc_mean = nn.Linear(embed_dim, action_dim_per_turbine)
        self.fc_logstd = nn.Linear(embed_dim, action_dim_per_turbine)
        
        self.register_buffer("action_scale", torch.tensor(action_scale, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor(action_bias, dtype=torch.float32))
    
    def forward(
        self, 
        obs: torch.Tensor, 
        positions: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            obs: (batch, n_turbines, obs_dim_per_turbine)
            positions: (batch, n_turbines, 2) normalized wind-relative positions
            key_padding_mask: (batch, n_turbines) where True = padding (ignore)
        """
        h = self.obs_encoder(obs)
        pos_embed = self.pos_encoder(positions)
        h = torch.cat([h, pos_embed], dim=-1)
        h = self.input_proj(h)
        h, attn_weights = self.transformer(h, key_padding_mask)
        
        mean = self.fc_mean(h)
        log_std = self.fc_logstd(h)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        
        return mean, log_std, attn_weights
    
    def get_action(
        self,
        obs: torch.Tensor,
        positions: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Sample action from policy.
        
        Args:
            key_padding_mask: (batch, n_turbines) where True = padding position
        """
        mean, log_std, attn_weights = self.forward(obs, positions, key_padding_mask)
        std = log_std.exp()
        
        normal = torch.distributions.Normal(mean, std)
        if deterministic:
            x_t = mean
        else:
            x_t = normal.rsample()
        
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        
        # Mask out padded positions before summing log prob
        if key_padding_mask is not None:
            # key_padding_mask: (batch, n_turbines), True = padding
            # Expand to match log_prob shape: (batch, n_turbines, action_dim)
            mask = ~key_padding_mask.unsqueeze(-1)  # True = real turbine
            log_prob = log_prob * mask.float()
        
        # Sum over turbines and action dims -> (batch, 1)
        log_prob = log_prob.sum(dim=(-2, -1)).unsqueeze(-1)
        
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        
        return action, log_prob, mean_action, attn_weights


# =============================================================================
# Critic (Q) Network
# =============================================================================

class TransformerCritic(nn.Module):
    """Transformer-based critic for wind farm control."""
    
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
        
        self.obs_action_encoder = nn.Sequential(
            nn.Linear(obs_dim_per_turbine + action_dim_per_turbine, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        
        self.pos_encoder = PositionalEncoding(pos_dim=2, embed_dim=pos_embed_dim)
        self.input_proj = nn.Linear(embed_dim + pos_embed_dim, embed_dim)
        self.transformer = TransformerEncoder(embed_dim, num_heads, num_layers, mlp_ratio, dropout)
        
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
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            key_padding_mask: (batch, n_turbines) where True = padding
        """
        x = torch.cat([obs, action], dim=-1)
        h = self.obs_action_encoder(x)
        pos_embed = self.pos_encoder(positions)
        h = torch.cat([h, pos_embed], dim=-1)
        h = self.input_proj(h)
        h, _ = self.transformer(h, key_padding_mask)
        
        # Masked mean pooling over turbines
        if key_padding_mask is not None:
            mask = ~key_padding_mask.unsqueeze(-1)  # (batch, n_turb, 1), True = real
            h = h * mask.float()
            h_sum = h.sum(dim=1)  # (batch, embed_dim)
            n_real = mask.float().sum(dim=1)  # (batch, 1)
            h_pooled = h_sum / n_real.clamp(min=1)
        else:
            h_pooled = h.mean(dim=1)
        
        q = self.q_head(h_pooled)
        return q


# =============================================================================
# Random Layout Environment Wrapper
# =============================================================================

class RandomLayoutEnv(gym.Env):
    """
    Meta-environment that randomly samples from multiple farm layouts each episode.
    
    Observations are padded to a fixed maximum size, enabling use with transformers
    and vectorized environments (all envs have same obs/action shape).
    
    The attention mask indicates which turbines are real vs padding.
    """
    
    def __init__(
        self, 
        layouts_config: Dict[str, Tuple[np.ndarray, np.ndarray]],
        base_env_kwargs: dict,
        max_turbines: Optional[int] = None,
        seed: int = 0
    ):
        """
        Args:
            layouts_config: Dict mapping layout_name -> (x_pos, y_pos) arrays
            base_env_kwargs: Kwargs to pass to WindFarmEnv (excluding x_pos, y_pos)
            max_turbines: Maximum turbines for padding. If None, uses max across layouts.
            seed: Random seed
        """
        super().__init__()
        
        self.layouts_config = layouts_config
        self.base_env_kwargs = base_env_kwargs
        self.seed_value = seed
        self.rng = np.random.default_rng(seed)
        
        # Determine max turbines for padding
        if max_turbines is None:
            max_turbines = max(len(x) for x, y in layouts_config.values())
        self.max_turbines = max_turbines
        
        # Create all environments (each wrapped with PerTurbineObservationWrapper)
        self.envs = {}
        for name, (x_pos, y_pos) in layouts_config.items():
            env = WindFarmEnv(x_pos=x_pos, y_pos=y_pos, **base_env_kwargs)
            env = PerTurbineObservationWrapper(env)
            self.envs[name] = env
        
        # Get observation dimensions from a sample env
        sample_env = list(self.envs.values())[0]
        sample_obs, _ = sample_env.reset()
        self.obs_dim_per_turbine = sample_obs.shape[1]
        
        # Store rotor diameter (same for all layouts since same turbine type)
        self.rotor_diameter = sample_env.D
        
        # Initialize with first layout
        self.layout_names = list(layouts_config.keys())
        self.current_layout_name = self.layout_names[0]
        self.current_env = self.envs[self.current_layout_name]
        
        # Define padded observation space
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.max_turbines, self.obs_dim_per_turbine),
            dtype=np.float32
        )
        
        # Define padded action space (1 action per turbine for yaw)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.max_turbines,),
            dtype=np.float32
        )
        
        # Current attention mask (True = padding, should be ignored)
        self._attention_mask = np.ones(self.max_turbines, dtype=bool)
        self._attention_mask[:self.n_turbines] = False
    
    @property
    def n_turbines(self) -> int:
        """Actual number of turbines in current layout (not padded)."""
        return self.current_env.n_turb
    
    @property
    def turbine_positions(self) -> np.ndarray:
        """
        Padded turbine positions (max_turbines, 2).
        Padding positions are zeros.
        """
        actual = np.column_stack([
            self.current_env.x_pos,
            self.current_env.y_pos
        ])
        padded = np.zeros((self.max_turbines, 2), dtype=np.float32)
        padded[:self.n_turbines] = actual
        return padded
    
    @property
    def attention_mask(self) -> np.ndarray:
        """Boolean mask where True = padding (should be ignored by transformer)."""
        return self._attention_mask.copy()
    
    @property
    def mean_wind_direction(self) -> float:
        """Current mean wind direction from the active environment."""
        return self.current_env.wd
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        # Randomly sample a new layout
        self.current_layout_name = self.rng.choice(self.layout_names)
        self.current_env = self.envs[self.current_layout_name]
        
        # Reset the chosen environment
        obs, info = self.current_env.reset(seed=seed, options=options)
        
        # Pad observation
        padded_obs = self._pad_observation(obs)
        
        # Update attention mask
        self._attention_mask = np.ones(self.max_turbines, dtype=bool)
        self._attention_mask[:self.n_turbines] = False
        
        # Add layout info
        info['n_turbines'] = self.n_turbines
        info['layout'] = self.current_layout_name
        info['attention_mask'] = self._attention_mask.copy()
        
        return padded_obs, info
    
    def step(self, action):
        # Only use actions for real turbines (ignore padding actions)
        real_action = action[:self.n_turbines]
        
        obs, reward, terminated, truncated, info = self.current_env.step(real_action)
        
        # Pad observation
        padded_obs = self._pad_observation(obs)
        
        # Add layout info
        info['n_turbines'] = self.n_turbines
        info['attention_mask'] = self._attention_mask.copy()
        
        return padded_obs, reward, terminated, truncated, info
    
    def _pad_observation(self, obs: np.ndarray) -> np.ndarray:
        """Pad observation to (max_turbines, obs_dim)."""
        padded = np.zeros((self.max_turbines, self.obs_dim_per_turbine), dtype=np.float32)
        padded[:obs.shape[0]] = obs
        return padded
    
    def close(self):
        for env in self.envs.values():
            env.close()


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


def get_layout_positions(layout_type: str, wind_turbine) -> Tuple[np.ndarray, np.ndarray]:
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


def make_single_layout_env(args, wind_turbine, seed: int):
    """Create a single-layout environment with per-turbine observations."""
    x_pos, y_pos = get_layout_positions(args.layout_type, wind_turbine)
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
    env = PerTurbineObservationWrapper(env)
    return env


def make_random_layout_env(args, wind_turbine, seed: int):
    """Create a random-layout environment that samples layouts on reset."""
    layout_names = [l.strip() for l in args.random_layouts.split(",")]
    
    # Build layouts config
    layouts_config = {}
    for name in layout_names:
        x_pos, y_pos = get_layout_positions(name, wind_turbine)
        layouts_config[name] = (x_pos, y_pos)
    
    # Base env kwargs (everything except x_pos, y_pos)
    config = make_env_config()
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
    
    env = RandomLayoutEnv(
        layouts_config=layouts_config,
        base_env_kwargs=base_env_kwargs,
        max_turbines=args.max_turbines,
        seed=seed,
    )
    return env


# =============================================================================
# Replay Buffer with Attention Mask Support
# =============================================================================

class TransformerReplayBuffer:
    """Replay buffer that stores attention masks for variable-size farms."""
    
    def __init__(self, capacity: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.buffer = []
        self.position = 0
    
    def add(
        self,
        obs: np.ndarray,           # (max_turbines, obs_dim)
        next_obs: np.ndarray,      # (max_turbines, obs_dim)
        action: np.ndarray,        # (max_turbines, action_dim)
        reward: float,
        done: bool,
        positions: np.ndarray,     # (max_turbines, 2)
        attention_mask: np.ndarray # (max_turbines,) bool, True=padding
    ):
        data = (obs, next_obs, action, reward, done, positions, attention_mask)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.position] = data
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        obs_list, next_obs_list, action_list, positions_list, mask_list = [], [], [], [], []
        rewards, dones = [], []
        
        for obs, next_obs, action, reward, done, positions, mask in batch:
            obs_list.append(obs)
            next_obs_list.append(next_obs)
            action_list.append(action)
            positions_list.append(positions)
            mask_list.append(mask)
            rewards.append(reward)
            dones.append(done)
        
        return {
            "observations": torch.tensor(np.stack(obs_list), device=self.device, dtype=torch.float32),
            "next_observations": torch.tensor(np.stack(next_obs_list), device=self.device, dtype=torch.float32),
            "actions": torch.tensor(np.stack(action_list), device=self.device, dtype=torch.float32),
            "positions": torch.tensor(np.stack(positions_list), device=self.device, dtype=torch.float32),
            "attention_mask": torch.tensor(np.stack(mask_list), device=self.device, dtype=torch.bool),
            "rewards": torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(-1),
            "dones": torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(-1),
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


def get_positions_tensor(positions_np, rotor_d, wind_dir, device, wind_relative: bool = True):
    """Get normalized turbine positions as a tensor."""
    positions_norm = positions_np / rotor_d
    positions_tensor = torch.tensor(positions_norm, dtype=torch.float32, device=device)
    
    if positions_tensor.ndim == 2:
        positions_tensor = positions_tensor.unsqueeze(0)
    
    if wind_relative:
        wind_dir_tensor = torch.tensor([wind_dir], dtype=torch.float32, device=device)
        positions_tensor = transform_to_wind_relative(positions_tensor, wind_dir_tensor)
    
    return positions_tensor


if __name__ == "__main__":
    import stable_baselines3 as sb3
    
    if sb3.__version__ < "2.0":
        raise ValueError("Need stable_baselines3 >= 2.0")
    
    args = tyro.cli(Args)
    
    # Determine training mode
    use_random_layouts = bool(args.random_layouts.strip())
    
    if use_random_layouts:
        layout_names = [l.strip() for l in args.random_layouts.split(",")]
        run_name = f"{args.exp_name}_random_{'_'.join(layout_names)}"
        print(f"Random layout training mode with layouts: {layout_names}")
    else:
        run_name = f"{args.exp_name}_{args.layout_type}"
        print(f"Single layout training mode: {args.layout_type}")
    
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
    
    # Create vectorized environments
    print(f"Creating {args.num_envs} parallel environment(s)...")
    
    def make_env_fn(seed):
        def _init():
            if use_random_layouts:
                return make_random_layout_env(args, wind_turbine, seed)
            else:
                return make_single_layout_env(args, wind_turbine, seed)
        return _init
    
    envs = gym.vector.AsyncVectorEnv(
        [make_env_fn(args.seed + i) for i in range(args.num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    envs = RecordEpisodeVals(envs)
    
    # Get dimensions from a sample env
    sample_env = make_env_fn(args.seed)()
    sample_obs, sample_info = sample_env.reset()
    
    if use_random_layouts:
        n_turbines_max = args.max_turbines
        obs_dim_per_turbine = sample_obs.shape[1]
    else:
        n_turbines_max = sample_obs.shape[0]
        obs_dim_per_turbine = sample_obs.shape[1]
    
    action_dim_per_turbine = 1
    rotor_diameter = sample_env.rotor_diameter if hasattr(sample_env, 'rotor_diameter') else sample_env.D
    sample_env.close()
    
    print(f"Max turbines: {n_turbines_max}, obs_dim={obs_dim_per_turbine}, action_dim={action_dim_per_turbine}")
    
    # Action scaling
    action_high = envs.single_action_space.high[0]
    action_low = envs.single_action_space.low[0]
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
    
    # Entropy tuning (use average turbine count for target entropy)
    if args.autotune:
        target_entropy = -action_dim_per_turbine * n_turbines_max * 0.5  # Rough average
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
    
    # Reset envs and get initial info
    obs, infos = envs.reset(seed=args.seed)
    
    # Deque to track last 1000 step rewards (across all envs)
    step_reward_window = deque(maxlen=1000)
    next_save_step = 0
    next_save_step += args.save_interval


    # Track attention masks per env (for random layouts)
    if use_random_layouts:
        # Get masks from info (vectorized envs return list of infos)
        current_masks = np.stack([info.get('attention_mask', np.zeros(n_turbines_max, dtype=bool)) 
                                   for info in infos.get('_final_info', [{}]*args.num_envs)]) \
                        if '_final_info' in infos else np.zeros((args.num_envs, n_turbines_max), dtype=bool)
        # Actually for AsyncVectorEnv, initial masks come from first reset
        # We'll update them properly below
        current_masks = np.zeros((args.num_envs, n_turbines_max), dtype=bool)
    else:
        # Single layout: no padding needed
        current_masks = np.zeros((args.num_envs, n_turbines_max), dtype=bool)
    
    print(f"\nStarting training for {args.total_timesteps} timesteps with {args.num_envs} env(s)...")
    
    num_updates = args.total_timesteps // args.num_envs
    
    for update in range(num_updates+2):
        global_step += args.num_envs
        
        # Get positions for current layouts
        # For single layout this is constant; for random layouts it changes on reset
        # We use a default wind direction of 270 for position encoding
        positions_np = np.zeros((args.num_envs, n_turbines_max, 2), dtype=np.float32)
        for i in range(args.num_envs):
            # Get positions from underlying env
            if hasattr(envs, 'envs'):
                base_env = envs.envs[i]
                if hasattr(base_env, 'turbine_positions'):
                    pos = base_env.turbine_positions
                else:
                    pos = np.column_stack([base_env.x_pos, base_env.y_pos])
                    # Pad if needed
                    padded = np.zeros((n_turbines_max, 2), dtype=np.float32)
                    padded[:pos.shape[0]] = pos
                    pos = padded
                positions_np[i] = pos / rotor_diameter
        
        # Get action
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(args.num_envs)])
        else:
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
                positions_tensor = torch.tensor(positions_np, dtype=torch.float32, device=device)
                mask_tensor = torch.tensor(current_masks, dtype=torch.bool, device=device) if use_random_layouts else None
                
                # Apply wind-relative transformation
                wind_dir_tensor = torch.full((args.num_envs,), 270.0, dtype=torch.float32, device=device)
                positions_tensor = transform_to_wind_relative(positions_tensor, wind_dir_tensor)
                
                action_tensor, _, _, _ = actor.get_action(obs_tensor, positions_tensor, mask_tensor)
                actions = action_tensor.squeeze(-1).cpu().numpy()
        
        # Step environment
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        
        # Track per-step rewards across envs
        step_reward_window.extend(np.array(rewards).flatten().tolist())


        # Log episode stats
        if "final_info" in infos:
            print(f"global_step={global_step}, episodic_return={np.mean(envs.return_queue):.2f}")
            writer.add_scalar("charts/episodic_return", np.mean(envs.return_queue), global_step)
            writer.add_scalar("charts/episodic_length", np.mean(envs.length_queue), global_step)
            writer.add_scalar("charts/episodic_power", np.mean(envs.mean_power_queue), global_step)
        
        # Handle final observations for truncated episodes
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_obs"][idx]
        
        # Store transitions in replay buffer
        for i in range(args.num_envs):
            done = terminations[i] or truncations[i]
            action_reshaped = actions[i].reshape(-1, action_dim_per_turbine)
            rb.add(
                obs[i], 
                real_next_obs[i], 
                action_reshaped, 
                rewards[i], 
                done, 
                positions_np[i],
                current_masks[i]
            )
            
            # Update mask if episode ended and using random layouts
            if done and use_random_layouts:
                # New mask will come from reset - for now keep current
                # The mask gets updated when we process the next obs
                pass
        
        obs = next_obs
        
        # Training
        if global_step > args.learning_starts and len(rb) >= args.batch_size:
            data = rb.sample(args.batch_size)
            
            # Get mask for this batch
            batch_mask = data["attention_mask"] if use_random_layouts else None
            
            # Compute target Q-values
            with torch.no_grad():
                next_actions, next_log_pi, _, _ = actor.get_action(
                    data["next_observations"], 
                    data["positions"],
                    batch_mask
                )
                qf1_next = qf1_target(data["next_observations"], next_actions, data["positions"], batch_mask)
                qf2_next = qf2_target(data["next_observations"], next_actions, data["positions"], batch_mask)
                min_qf_next = torch.min(qf1_next, qf2_next) - alpha * next_log_pi
                next_q_value = data["rewards"] + (1 - data["dones"]) * args.gamma * min_qf_next
            
            # Update critics
            qf1_value = qf1(data["observations"], data["actions"], data["positions"], batch_mask)
            qf2_value = qf2(data["observations"], data["actions"], data["positions"], batch_mask)
            qf1_loss = F.mse_loss(qf1_value, next_q_value)
            qf2_loss = F.mse_loss(qf2_value, next_q_value)
            qf_loss = qf1_loss + qf2_loss
            
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()
            
            # Update actor (delayed)
            if update % args.policy_frequency == 0:
                for _ in range(args.policy_frequency):
                    actions_pi, log_pi, _, _ = actor.get_action(data["observations"], data["positions"], batch_mask)
                    qf1_pi = qf1(data["observations"], actions_pi, data["positions"], batch_mask)
                    qf2_pi = qf2(data["observations"], actions_pi, data["positions"], batch_mask)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = (alpha * log_pi - min_qf_pi).mean()
                    
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()
                    
                    # Update alpha
                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _, _ = actor.get_action(data["observations"], data["positions"], batch_mask)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()
                        
                        alpha_optimizer.zero_grad()
                        alpha_loss.backward()
                        alpha_optimizer.step()
                        alpha = log_alpha.exp().item()
            
            # Update target networks
            if update % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
            
            # Logging
            if update % 20 == 0:
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                sps = int(global_step / (time.time() - start_time))
                writer.add_scalar("charts/SPS", sps, global_step)
        
                # Mean of last 1000 step rewards (across all envs)
                writer.add_scalar("charts/step_reward_mean_1000",
                                  float(np.mean(step_reward_window)) if len(step_reward_window) > 0 else 0.0,
                                  global_step)
        
        
                print(f"Step {global_step}: SPS={sps}, qf_loss={qf_loss.item():.4f}, actor_loss={actor_loss.item():.4f}, reward_mean_1000={float(np.mean(step_reward_window)) if len(step_reward_window) > 0 else 0.0:.4f}")


        # Save model
        if args.save_model and global_step >= next_save_step:
            save_model(actor, qf1, qf2, global_step, run_name, args.exp_name)
            next_save_step += args.save_interval


    # Final save
    if args.save_model:
        save_model(actor, qf1, qf2, global_step, run_name, args.exp_name)
    
    print("Training finished!")
    envs.close()
    writer.close()