"""
Claude made this
It is a shared module file for Transformer SAC
So we can decide if we want to share various modules between actor and critic
Not yet implemented in main code
"""

# IMPORTS SKIPPED FOR SIMPLICITY
# Assume all necessary imports are here (torch, nn, numpy, gym, etc.)

import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
import gymnasium as gym

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
    track: bool = True  # Enable wandb tracking
    wandb_project_name: str = "transformer_windfarm"
    wandb_entity: Optional[str] = None
    shuffle_turbs: bool = False  # Shuffle turbine order in obs/action

    # === Receptivity Profile Settings ===
    use_pywake_profile: bool = False        # Enable profile encoding
    profile_encoder_hidden: int = 128       # Hidden dim in profile encoder MLP
    rotate_profiles: bool = False           # Rotate profiles to wind-relative frame
    n_profile_directions: int = 360         # Number of directions in profile

    # === Environment Settings ===
    turbtype: str = "DTU10MW"  # Wind turbine type
    TI_type: str = "Random"   # Turbulence intensity sampling
    dt_sim: int = 5           # Simulation timestep (seconds)
    dt_env: int = 10          # Environment timestep (seconds)
    yaw_step: float = 5.0     # Max yaw change per sim step (degrees)
    max_eps: int = 20         # Number of flow passthroughs per episode
    num_envs: int = 1         # Number of parallel environments

    # === Layout Settings ===
    # Comma-separated list of layouts. Single = single-layout, Multiple = multi-layout
    layouts: str = "test_layout"  # e.g., "square_1,square_2,circular_1"
    
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
    
    # === Positional Encoding Settings ===
    # Options: "absolute_mlp", "relative_mlp", "relative_mlp_shared", 
    #          "sinusoidal_2d",
    pos_encoding_type: str = "absolute_mlp"
    # For relative encoding: number of hidden units in the bias MLP
    rel_pos_hidden_dim: int = 64
    # For relative encoding: whether to use separate bias per head
    rel_pos_per_head: bool = True
    
    # === Parameter Sharing Settings ===
    share_pos_encoder: bool = False       # Share positional encoder between actor/critic
    share_profile_encoders: bool = False  # Share receptivity/influence encoders between actor/critic
    share_transformer: bool = False       # Share transformer encoder between actor/critic
    share_input_proj: bool = False        # Share input projection layer between actor/critic
    
    # === SAC Hyperparameters ===
    utd_ratio: float = 1.0           # Update-to-data ratio
    total_timesteps: int = 100_000
    buffer_size: int = int(1e6)
    gamma: float = 0.99           # Discount factor
    tau: float = 0.005            # Target network update rate
    batch_size: int = 256
    learning_starts: int = 5000   # Steps before training starts
    policy_lr: float = 3e-4
    q_lr: float = 3e-4
    policy_frequency: int = 2     # Policy update frequency
    target_network_frequency: int = 1
    alpha: float = 0.2            # Initial entropy coefficient
    autotune: bool = True         # Auto-tune entropy coefficient
    
    # === Gradient Clipping ===
    grad_clip: bool = True
    grad_clip_max_norm: float = 1.0
    
# =============================================================================
# POSITIONAL ENCODING (placeholder classes - actual implementations assumed)
# =============================================================================

VALID_POS_ENCODING_TYPES = [
    "absolute_mlp", "sinusoidal_2d", "polar_mlp",
    "relative_mlp", "relative_mlp_shared", "relative_polar", "relative_polar_shared",
    "alibi", "alibi_directional", "absolute_plus_relative",
]

# Placeholder for actual positional encoding classes
class AbsolutePositionalEncoding(nn.Module):
    def __init__(self, pos_dim: int, embed_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(pos_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
    
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        return self.mlp(positions)

class RelativePositionalBias(nn.Module):
    def __init__(self, num_heads: int, hidden_dim: int, per_head: bool, pos_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.per_head = per_head
        out_dim = num_heads if per_head else 1
        self.mlp = nn.Sequential(
            nn.Linear(pos_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
    
    def forward(self, positions: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Compute pairwise relative positions and return attention bias
        batch, n, _ = positions.shape
        # Simplified - actual implementation would compute pairwise distances
        rel_pos = positions.unsqueeze(2) - positions.unsqueeze(1)  # (batch, n, n, 2)
        bias = self.mlp(rel_pos)  # (batch, n, n, num_heads or 1)
        if not self.per_head:
            bias = bias.expand(-1, -1, -1, self.num_heads)
        return bias.permute(0, 3, 1, 2)  # (batch, num_heads, n, n)

class PyWakeProfileEncoder(nn.Module):
    def __init__(self, n_directions: int, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_directions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
    
    def forward(self, profiles: torch.Tensor) -> torch.Tensor:
        return self.mlp(profiles)


def create_positional_encoding(
    encoding_type: str,
    embed_dim: int,
    pos_embed_dim: int,
    num_heads: int,
    rel_pos_hidden_dim: int = 64,
    rel_pos_per_head: bool = True,
) -> Tuple[Optional[nn.Module], Optional[nn.Module], bool]:
    """
    Factory function to create positional encoding modules.
    
    Returns:
        (pos_encoder, rel_pos_bias, uses_additive_embedding)
    """
    if encoding_type not in VALID_POS_ENCODING_TYPES:
        raise ValueError(f"Unknown pos_encoding_type: {encoding_type}")
    
    if encoding_type == "absolute_mlp":
        pos_encoder = AbsolutePositionalEncoding(pos_dim=2, embed_dim=pos_embed_dim)
        rel_pos_bias = None
        uses_additive_embedding = True
        
    elif encoding_type in ["relative_mlp", "relative_mlp_shared"]:
        pos_encoder = None
        rel_pos_bias = RelativePositionalBias(
            num_heads=num_heads,
            hidden_dim=rel_pos_hidden_dim,
            per_head=(encoding_type == "relative_mlp"),
            pos_dim=2
        )
        uses_additive_embedding = False
        
    elif encoding_type == "absolute_plus_relative":
        pos_encoder = AbsolutePositionalEncoding(pos_dim=2, embed_dim=pos_embed_dim)
        rel_pos_bias = RelativePositionalBias(
            num_heads=num_heads,
            hidden_dim=rel_pos_hidden_dim,
            per_head=rel_pos_per_head,
            pos_dim=2
        )
        uses_additive_embedding = True
    else:
        # Default fallback
        pos_encoder = AbsolutePositionalEncoding(pos_dim=2, embed_dim=pos_embed_dim)
        rel_pos_bias = None
        uses_additive_embedding = True
    
    return pos_encoder, rel_pos_bias, uses_additive_embedding


# =============================================================================
# TRANSFORMER BLOCKS
# =============================================================================

class TransformerEncoderLayer(nn.Module):
    """Standard transformer encoder layer with pre-norm."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_norm = self.norm1(x)
        
        if attn_bias is not None:
            batch_size, num_heads, n, _ = attn_bias.shape
            attn_mask = attn_bias.reshape(batch_size * num_heads, n, n)
        else:
            attn_mask = None
        
        attn_out, attn_weights = self.attn(
            x_norm, x_norm, x_norm,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            average_attn_weights=False
        )
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        
        return x, attn_weights


class TransformerEncoder(nn.Module):
    """Stack of transformer encoder layers."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        all_attn_weights = []
        
        for layer in self.layers:
            x, attn_weights = layer(x, key_padding_mask, attn_bias)
            all_attn_weights.append(attn_weights)
        
        x = self.norm(x)
        
        return x, all_attn_weights


# =============================================================================
# SHARED MODULES CONTAINER
# =============================================================================

@dataclass
class SharedModules:
    """Container for modules that can be shared between actor and critic."""
    pos_encoder: Optional[nn.Module] = None
    rel_pos_bias: Optional[nn.Module] = None
    recep_encoder: Optional[nn.Module] = None
    influence_encoder: Optional[nn.Module] = None
    transformer: Optional[nn.Module] = None
    input_proj: Optional[nn.Module] = None
    uses_additive_embedding: bool = True


def create_shared_modules(
    args: Args,
    embed_dim: int,
    pos_embed_dim: int,
    num_heads: int,
    num_layers: int,
    mlp_ratio: float,
    dropout: float,
) -> SharedModules:
    """
    Create shared modules based on args flags.
    
    Returns a SharedModules container with modules that should be shared.
    Modules that shouldn't be shared are left as None.
    """
    shared = SharedModules()
    
    # Determine if we need additive embedding (needed for input_proj dimension)
    _, _, uses_additive = create_positional_encoding(
        encoding_type=args.pos_encoding_type,
        embed_dim=embed_dim,
        pos_embed_dim=pos_embed_dim,
        num_heads=num_heads,
        rel_pos_hidden_dim=args.rel_pos_hidden_dim,
        rel_pos_per_head=args.rel_pos_per_head,
    )
    shared.uses_additive_embedding = uses_additive
    
    # Create shared positional encoding if requested
    if args.share_pos_encoder:
        pos_encoder, rel_pos_bias, _ = create_positional_encoding(
            encoding_type=args.pos_encoding_type,
            embed_dim=embed_dim,
            pos_embed_dim=pos_embed_dim,
            num_heads=num_heads,
            rel_pos_hidden_dim=args.rel_pos_hidden_dim,
            rel_pos_per_head=args.rel_pos_per_head,
        )
        shared.pos_encoder = pos_encoder
        shared.rel_pos_bias = rel_pos_bias
    
    # Create shared profile encoders if requested
    if args.share_profile_encoders and args.use_pywake_profile:
        shared.recep_encoder = PyWakeProfileEncoder(
            n_directions=args.n_profile_directions,
            embed_dim=embed_dim,
            hidden_dim=args.profile_encoder_hidden,
        )
        shared.influence_encoder = PyWakeProfileEncoder(
            n_directions=args.n_profile_directions,
            embed_dim=embed_dim,
            hidden_dim=args.profile_encoder_hidden,
        )
    
    # Create shared transformer if requested
    if args.share_transformer:
        shared.transformer = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
    
    # Create shared input projection if requested
    if args.share_input_proj:
        if uses_additive:
            shared.input_proj = nn.Linear(embed_dim + pos_embed_dim, embed_dim)
        else:
            shared.input_proj = nn.Linear(embed_dim, embed_dim)
    
    return shared


# =============================================================================
# ACTOR NETWORK
# =============================================================================

LOG_STD_MAX = 2
LOG_STD_MIN = -5


class TransformerActor(nn.Module):
    """
    Transformer-based actor (policy) network for wind farm control.
    
    Supports optional parameter sharing via SharedModules.
    """
    
    def __init__(
        self,
        obs_dim_per_turbine: int,
        action_dim_per_turbine: int = 1,
        embed_dim: int = 128,
        pos_embed_dim: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        action_scale: float = 1.0,
        action_bias: float = 0.0,
        # Positional encoding settings
        pos_encoding_type: str = "absolute_mlp",
        rel_pos_hidden_dim: int = 64,
        rel_pos_per_head: bool = True,
        # Receptivity profile settings
        use_pywake_profile: bool = False,
        profile_encoder_hidden: int = 128,
        n_profile_directions: int = 360,
        # Shared modules (optional)
        shared_modules: Optional[SharedModules] = None,
    ):
        super().__init__()
        
        self.obs_dim_per_turbine = obs_dim_per_turbine
        self.action_dim_per_turbine = action_dim_per_turbine
        self.embed_dim = embed_dim
        self.pos_encoding_type = pos_encoding_type
        self.use_pywake_profile = use_pywake_profile

        # Determine if using shared modules
        use_shared = shared_modules is not None
        
        # === Positional Encoding ===
        if use_shared and shared_modules.pos_encoder is not None:
            # Use shared positional encoder
            self.pos_encoder = shared_modules.pos_encoder
            self.rel_pos_bias = shared_modules.rel_pos_bias
            self.uses_additive_embedding = shared_modules.uses_additive_embedding
        else:
            # Create own positional encoder
            self.pos_encoder, self.rel_pos_bias, self.uses_additive_embedding = \
                create_positional_encoding(
                    encoding_type=pos_encoding_type,
                    embed_dim=embed_dim,
                    pos_embed_dim=pos_embed_dim,
                    num_heads=num_heads,
                    rel_pos_hidden_dim=rel_pos_hidden_dim,
                    rel_pos_per_head=rel_pos_per_head,
                )
        
        # === Profile Encoders ===
        if use_pywake_profile:
            if use_shared and shared_modules.recep_encoder is not None:
                # Use shared profile encoders
                self.recep_encoder = shared_modules.recep_encoder
                self.influence_encoder = shared_modules.influence_encoder
            else:
                # Create own profile encoders
                self.recep_encoder = PyWakeProfileEncoder(
                    n_directions=n_profile_directions,
                    embed_dim=embed_dim,
                    hidden_dim=profile_encoder_hidden,
                )
                self.influence_encoder = PyWakeProfileEncoder(
                    n_directions=n_profile_directions,
                    embed_dim=embed_dim,
                    hidden_dim=profile_encoder_hidden,
                )
        else:
            self.recep_encoder = None
            self.influence_encoder = None

        # === Observation Encoder (never shared - different input for critic) ===
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim_per_turbine, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        
        # === Input Projection ===
        if use_shared and shared_modules.input_proj is not None:
            self.input_proj = shared_modules.input_proj
        else:
            if self.uses_additive_embedding:
                self.input_proj = nn.Linear(embed_dim + pos_embed_dim, embed_dim)
            else:
                self.input_proj = nn.Linear(embed_dim, embed_dim)
        
        # === Transformer ===
        if use_shared and shared_modules.transformer is not None:
            self.transformer = shared_modules.transformer
        else:
            self.transformer = TransformerEncoder(
                embed_dim, num_heads, num_layers, mlp_ratio, dropout
            )
        
        # === Action Heads (never shared - actor-specific) ===
        self.fc_mean = nn.Linear(embed_dim, action_dim_per_turbine)
        self.fc_logstd = nn.Linear(embed_dim, action_dim_per_turbine)
        
        # Action scaling
        self.register_buffer("action_scale", torch.tensor(action_scale, dtype=torch.float32))
        self.register_buffer("action_bias_val", torch.tensor(action_bias, dtype=torch.float32))
    
    def forward(
        self,
        obs: torch.Tensor,
        positions: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        recep_profile: Optional[torch.Tensor] = None,
        influence_profile: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Forward pass returning action distribution parameters."""
        batch_size, n_turbines, _ = obs.shape
        
        # Encode observations
        h = self.obs_encoder(obs)
        
        # Apply positional encoding based on type
        if self.uses_additive_embedding and self.pos_encoder is not None:
            pos_embed = self.pos_encoder(positions)
            h = torch.cat([h, pos_embed], dim=-1)
        
        # Project to embed_dim
        h = self.input_proj(h)
        
        # ADD profile encoding
        if self.use_pywake_profile and recep_profile is not None and influence_profile is not None:
            recep_embed = self.recep_encoder(recep_profile)
            influence_embed = self.influence_encoder(influence_profile)
            h = h + recep_embed + influence_embed

        # Compute relative position bias if using relative encoding
        attn_bias = None
        if self.rel_pos_bias is not None:
            attn_bias = self.rel_pos_bias(positions, key_padding_mask)

        h, attn_weights = self.transformer(h, key_padding_mask, attn_bias)
        
        # Action distribution parameters
        mean = self.fc_mean(h)
        log_std = self.fc_logstd(h)
        
        # Constrain log_std to reasonable range
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        
        return mean, log_std, attn_weights
    
    def get_action(
        self,
        obs: torch.Tensor,
        positions: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        recep_profile: Optional[torch.Tensor] = None,
        influence_profile: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Sample action from policy with log probability."""
        mean, log_std, attn_weights = self.forward(obs, positions, key_padding_mask, 
                                                   recep_profile, influence_profile)
        std = log_std.exp()
        
        normal = torch.distributions.Normal(mean, std)
        if deterministic:
            x_t = mean
        else:
            x_t = normal.rsample()
        
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias_val
        
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        
        if key_padding_mask is not None:
            mask = ~key_padding_mask.unsqueeze(-1)
            log_prob = log_prob * mask.float()
        
        log_prob = log_prob.sum(dim=(-2, -1), keepdim=False).unsqueeze(-1)
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias_val
        
        return action, log_prob, mean_action, attn_weights


# =============================================================================
# CRITIC NETWORK
# =============================================================================

class TransformerCritic(nn.Module):
    """
    Transformer-based critic (Q-function) network.
    
    Supports optional parameter sharing via SharedModules.
    """
    
    def __init__(
        self,
        obs_dim_per_turbine: int,
        action_dim_per_turbine: int = 1,
        embed_dim: int = 128,
        pos_embed_dim: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        # Positional encoding settings
        pos_encoding_type: str = "absolute_mlp",
        rel_pos_hidden_dim: int = 64,
        rel_pos_per_head: bool = True,
        # PyWake profile settings
        use_pywake_profile: bool = False,
        profile_encoder_hidden: int = 128,
        n_profile_directions: int = 360,
        # Shared modules (optional)
        shared_modules: Optional[SharedModules] = None,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.pos_encoding_type = pos_encoding_type
        self.use_pywake_profile = use_pywake_profile
        
        # Determine if using shared modules
        use_shared = shared_modules is not None
        
        # === Positional Encoding ===
        if use_shared and shared_modules.pos_encoder is not None:
            self.pos_encoder = shared_modules.pos_encoder
            self.rel_pos_bias = shared_modules.rel_pos_bias
            self.uses_additive_embedding = shared_modules.uses_additive_embedding
        else:
            self.pos_encoder, self.rel_pos_bias, self.uses_additive_embedding = \
                create_positional_encoding(
                    encoding_type=pos_encoding_type,
                    embed_dim=embed_dim,
                    pos_embed_dim=pos_embed_dim,
                    num_heads=num_heads,
                    rel_pos_hidden_dim=rel_pos_hidden_dim,
                    rel_pos_per_head=rel_pos_per_head,
                )
        
        # === Profile Encoders ===
        if use_pywake_profile:
            if use_shared and shared_modules.recep_encoder is not None:
                self.recep_encoder = shared_modules.recep_encoder
                self.influence_encoder = shared_modules.influence_encoder
            else:
                self.recep_encoder = PyWakeProfileEncoder(
                    n_directions=n_profile_directions,
                    embed_dim=embed_dim,
                    hidden_dim=profile_encoder_hidden,
                )
                self.influence_encoder = PyWakeProfileEncoder(
                    n_directions=n_profile_directions,
                    embed_dim=embed_dim,
                    hidden_dim=profile_encoder_hidden,
                )
        else:
            self.recep_encoder = None
            self.influence_encoder = None

        # === Observation + Action Encoder (never shared - different input) ===
        self.obs_action_encoder = nn.Sequential(
            nn.Linear(obs_dim_per_turbine + action_dim_per_turbine, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        
        # === Input Projection ===
        if use_shared and shared_modules.input_proj is not None:
            self.input_proj = shared_modules.input_proj
        else:
            if self.uses_additive_embedding:
                self.input_proj = nn.Linear(embed_dim + pos_embed_dim, embed_dim)
            else:
                self.input_proj = nn.Linear(embed_dim, embed_dim)
        
        # === Transformer ===
        if use_shared and shared_modules.transformer is not None:
            self.transformer = shared_modules.transformer
        else:
            self.transformer = TransformerEncoder(
                embed_dim, num_heads, num_layers, mlp_ratio, dropout
            )
    
        # === Q-value Head (never shared - critic-specific) ===
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
        recep_profile: Optional[torch.Tensor] = None,
        influence_profile: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute Q-value for observation-action pair."""
        batch_size = obs.shape[0]
        
        # Concatenate obs and action
        x = torch.cat([obs, action], dim=-1)
        
        # Encode
        h = self.obs_action_encoder(x)
        
        # Apply positional encoding
        if self.uses_additive_embedding and self.pos_encoder is not None:
            pos_embed = self.pos_encoder(positions)
            h = torch.cat([h, pos_embed], dim=-1)
        
        # Project to embed_dim
        h = self.input_proj(h)
        
        # ADD profile encoding
        if self.use_pywake_profile and recep_profile is not None and influence_profile is not None:
            recep_embed = self.recep_encoder(recep_profile)
            influence_embed = self.influence_encoder(influence_profile)
            h = h + recep_embed + influence_embed

        # Compute relative position bias
        attn_bias = None
        if self.rel_pos_bias is not None:
            attn_bias = self.rel_pos_bias(positions, key_padding_mask)
        
        # Transformer
        h, _ = self.transformer(h, key_padding_mask, attn_bias)

        # Masked mean pooling over turbines
        if key_padding_mask is not None:
            mask = ~key_padding_mask.unsqueeze(-1)
            h = h * mask.float()
            h_sum = h.sum(dim=1)
            n_real = mask.float().sum(dim=1).clamp(min=1)
            h_pooled = h_sum / n_real
        else:
            h_pooled = h.mean(dim=1)
        
        # Q-value
        q = self.q_head(h_pooled)
        
        return q


# =============================================================================
# HELPER FUNCTION FOR COUNTING PARAMETERS
# =============================================================================

def count_parameters(model: nn.Module, shared_modules: Optional[SharedModules] = None) -> Dict[str, int]:
    """
    Count parameters in a model, distinguishing shared vs unique.
    
    Returns dict with 'total', 'unique', and 'shared' counts.
    """
    total = sum(p.numel() for p in model.parameters())
    
    if shared_modules is None:
        return {'total': total, 'unique': total, 'shared': 0}
    
    # Count shared parameters
    shared_params = set()
    for module in [shared_modules.pos_encoder, shared_modules.rel_pos_bias,
                   shared_modules.recep_encoder, shared_modules.influence_encoder,
                   shared_modules.transformer, shared_modules.input_proj]:
        if module is not None:
            for p in module.parameters():
                shared_params.add(id(p))
    
    shared_count = sum(p.numel() for p in model.parameters() if id(p) in shared_params)
    unique_count = total - shared_count
    
    return {'total': total, 'unique': unique_count, 'shared': shared_count}


def print_sharing_summary(
    actor: TransformerActor,
    qf1: TransformerCritic,
    shared_modules: Optional[SharedModules],
    args: Args
):
    """Print a summary of parameter sharing configuration."""
    print("\n" + "=" * 60)
    print("PARAMETER SHARING SUMMARY")
    print("=" * 60)
    
    # Flags
    print("\nSharing flags:")
    print(f"  share_pos_encoder:      {args.share_pos_encoder}")
    print(f"  share_profile_encoders: {args.share_profile_encoders}")
    print(f"  share_transformer:      {args.share_transformer}")
    print(f"  share_input_proj:       {args.share_input_proj}")
    
    # Count parameters
    actor_counts = count_parameters(actor, shared_modules)
    critic_counts = count_parameters(qf1, shared_modules)
    
    print("\nActor parameters:")
    print(f"  Total:  {actor_counts['total']:,}")
    print(f"  Unique: {actor_counts['unique']:,}")
    print(f"  Shared: {actor_counts['shared']:,}")
    
    print("\nCritic parameters (single Q-network):")
    print(f"  Total:  {critic_counts['total']:,}")
    print(f"  Unique: {critic_counts['unique']:,}")
    print(f"  Shared: {critic_counts['shared']:,}")
    
    # Total unique parameters
    if shared_modules is not None:
        shared_param_count = sum(
            sum(p.numel() for p in m.parameters())
            for m in [shared_modules.pos_encoder, shared_modules.rel_pos_bias,
                     shared_modules.recep_encoder, shared_modules.influence_encoder,
                     shared_modules.transformer, shared_modules.input_proj]
            if m is not None
        )
    else:
        shared_param_count = 0
    
    total_unique = (actor_counts['unique'] + 
                    2 * critic_counts['unique'] +  # qf1 and qf2
                    2 * critic_counts['unique'] +  # target networks
                    shared_param_count)  # shared counted once
    
    total_without_sharing = (actor_counts['total'] + 
                             4 * critic_counts['total'])
    
    print(f"\nTotal trainable parameters: {total_unique:,}")
    print(f"Without sharing would be:   {total_without_sharing:,}")
    if total_without_sharing > 0:
        savings = (1 - total_unique / total_without_sharing) * 100
        print(f"Parameter reduction:        {savings:.1f}%")
    
    print("=" * 60 + "\n")


# =============================================================================
# MAIN TRAINING SCRIPT
# =============================================================================

def main():
    """Main training function."""
    
    # Parse arguments
    args = tyro.cli(Args)
    
    # Parse layouts
    layout_names = [l.strip() for l in args.layouts.split(",")]
    is_multi_layout = len(layout_names) > 1
    
    # Create run name
    run_name = f"{args.exp_name}_{'_'.join(layout_names)}_{args.seed}"
    
    # Add sharing info to run name if any sharing is enabled
    sharing_flags = []
    if args.share_pos_encoder:
        sharing_flags.append("spos")
    if args.share_profile_encoders:
        sharing_flags.append("sprof")
    if args.share_transformer:
        sharing_flags.append("strans")
    if args.share_input_proj:
        sharing_flags.append("sproj")
    if sharing_flags:
        run_name += f"_{'_'.join(sharing_flags)}"
    
    print(f"Run name: {run_name}")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # =========================================================================
    # ENVIRONMENT SETUP (simplified - assumes env setup works)
    # =========================================================================
    
    # Placeholder values - in real code these come from environment
    n_turbines_max = 10
    obs_dim_per_turbine = 64
    action_dim_per_turbine = 1
    rotor_diameter = 178.3
    
    action_high = 30.0
    action_low = -30.0
    action_scale = (action_high - action_low) / 2.0
    action_bias = (action_high + action_low) / 2.0

    # =========================================================================
    # CREATE SHARED MODULES (if any sharing is enabled)
    # =========================================================================
    
    any_sharing = (args.share_pos_encoder or 
                   args.share_profile_encoders or 
                   args.share_transformer or 
                   args.share_input_proj)
    
    if any_sharing:
        print("Creating shared modules...")
        shared_modules = create_shared_modules(
            args=args,
            embed_dim=args.embed_dim,
            pos_embed_dim=args.pos_embed_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            mlp_ratio=args.mlp_ratio,
            dropout=args.dropout,
        )
        # Move shared modules to device
        if shared_modules.pos_encoder is not None:
            shared_modules.pos_encoder = shared_modules.pos_encoder.to(device)
        if shared_modules.rel_pos_bias is not None:
            shared_modules.rel_pos_bias = shared_modules.rel_pos_bias.to(device)
        if shared_modules.recep_encoder is not None:
            shared_modules.recep_encoder = shared_modules.recep_encoder.to(device)
        if shared_modules.influence_encoder is not None:
            shared_modules.influence_encoder = shared_modules.influence_encoder.to(device)
        if shared_modules.transformer is not None:
            shared_modules.transformer = shared_modules.transformer.to(device)
        if shared_modules.input_proj is not None:
            shared_modules.input_proj = shared_modules.input_proj.to(device)
    else:
        shared_modules = None

    # =========================================================================
    # NETWORK SETUP
    # =========================================================================
    
    print("\nCreating networks...")
    print(f"Positional encoding type: {args.pos_encoding_type}")
    
    # Common kwargs for all networks
    common_kwargs = {
        "obs_dim_per_turbine": obs_dim_per_turbine,
        "action_dim_per_turbine": action_dim_per_turbine,
        "embed_dim": args.embed_dim,
        "pos_embed_dim": args.pos_embed_dim,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "mlp_ratio": args.mlp_ratio,
        "dropout": args.dropout,
        "pos_encoding_type": args.pos_encoding_type,
        "rel_pos_hidden_dim": args.rel_pos_hidden_dim,
        "rel_pos_per_head": args.rel_pos_per_head,
        "use_pywake_profile": args.use_pywake_profile,
        "profile_encoder_hidden": args.profile_encoder_hidden,
        "n_profile_directions": args.n_profile_directions,
        "shared_modules": shared_modules,  # Pass shared modules
    }

    # Actor has additional action scaling params
    actor = TransformerActor(
        action_scale=action_scale,
        action_bias=action_bias,
        **common_kwargs,
    ).to(device)

    # Critics - note: qf1 and qf2 share with actor, but NOT with each other's unique params
    # Target networks get their own copies (no sharing with online networks)
    qf1 = TransformerCritic(**common_kwargs).to(device)
    qf2 = TransformerCritic(**common_kwargs).to(device)
    
    # Target networks - create WITHOUT sharing (they maintain separate copies)
    target_kwargs = common_kwargs.copy()
    target_kwargs["shared_modules"] = None  # Targets don't share with online networks
    
    qf1_target = TransformerCritic(**target_kwargs).to(device)
    qf2_target = TransformerCritic(**target_kwargs).to(device)
    
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    
    # Print sharing summary
    print_sharing_summary(actor, qf1, shared_modules, args)
    
    # =========================================================================
    # OPTIMIZER SETUP - IMPORTANT: Handle shared parameters correctly
    # =========================================================================
    
    # Collect all unique parameters for each optimizer
    # Shared parameters should only be in ONE optimizer to avoid double updates
    
    if shared_modules is not None:
        # Get IDs of shared parameters
        shared_param_ids = set()
        shared_params_list = []
        for module in [shared_modules.pos_encoder, shared_modules.rel_pos_bias,
                       shared_modules.recep_encoder, shared_modules.influence_encoder,
                       shared_modules.transformer, shared_modules.input_proj]:
            if module is not None:
                for p in module.parameters():
                    if id(p) not in shared_param_ids:
                        shared_param_ids.add(id(p))
                        shared_params_list.append(p)
        
        # Actor optimizer: actor-unique params + shared params
        actor_unique_params = [p for p in actor.parameters() if id(p) not in shared_param_ids]
        actor_optimizer = optim.Adam(
            actor_unique_params + shared_params_list,
            lr=args.policy_lr
        )
        
        # Critic optimizer: critic-unique params only (shared params handled by actor optimizer)
        critic_unique_params = [p for p in list(qf1.parameters()) + list(qf2.parameters()) 
                                if id(p) not in shared_param_ids]
        q_optimizer = optim.Adam(critic_unique_params, lr=args.q_lr)
        
        print(f"Actor optimizer params: {sum(p.numel() for p in actor_unique_params):,} unique + {sum(p.numel() for p in shared_params_list):,} shared")
        print(f"Critic optimizer params: {sum(p.numel() for p in critic_unique_params):,} unique")
    else:
        # No sharing - standard setup
        actor_optimizer = optim.Adam(actor.parameters(), lr=args.policy_lr)
        q_optimizer = optim.Adam(
            list(qf1.parameters()) + list(qf2.parameters()),
            lr=args.q_lr
        )
    
    # Entropy tuning
    if args.autotune:
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        alpha_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha
        log_alpha = None
        alpha_optimizer = None

    print("\nNetwork and optimizer setup complete!")
    print(f"Device: {device}")
    
    # =========================================================================
    # REST OF TRAINING LOOP (abbreviated)
    # =========================================================================
    
    print("\n[Training loop would continue here...]")
    print("The shared modules are now properly integrated into actor and critics.")


if __name__ == "__main__":
    main()