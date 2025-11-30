"""
Attention Visualization for Transformer Wind Farm Control (v9)

Standalone script to visualize attention patterns from a trained transformer SAC model.

Usage:
    python visualize_attention.py --checkpoint runs/YOUR_RUN/checkpoints/step_100000.pt
    
Features:
    - Load trained model from checkpoint
    - Run policy on environment to extract attention weights
    - Visualize attention as heatmaps (per layer, per head)
    - Overlay attention on farm layout
    - Analyze attention vs. wake physics (upwind/downwind patterns)
    - Create attention animations over an episode

Author: Marcus (DTU Wind Energy)
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass
import math

# =============================================================================
# MODEL COMPONENTS (copied from v9 to be standalone)
# =============================================================================

class PositionalEncoding(nn.Module):
    """Learned positional encoding for turbine (x, y) coordinates."""
    
    def __init__(self, pos_dim: int = 2, embed_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(pos_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
    
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        return self.encoder(positions)


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
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.attn(
            x_norm, x_norm, x_norm,
            key_padding_mask=key_padding_mask,
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
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        all_attn_weights = []
        for layer in self.layers:
            x, attn_weights = layer(x, key_padding_mask)
            all_attn_weights.append(attn_weights)
        x = self.norm(x)
        return x, all_attn_weights


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class TransformerActor(nn.Module):
    """Transformer-based actor (policy) network for wind farm control."""
    
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
        use_farm_token: bool = False,
        action_scale: float = 1.0,
        action_bias: float = 0.0,
    ):
        super().__init__()
        
        self.obs_dim_per_turbine = obs_dim_per_turbine
        self.action_dim_per_turbine = action_dim_per_turbine
        self.embed_dim = embed_dim
        self.use_farm_token = use_farm_token
        
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim_per_turbine, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        
        self.pos_encoder = PositionalEncoding(pos_dim=2, embed_dim=pos_embed_dim)
        self.input_proj = nn.Linear(embed_dim + pos_embed_dim, embed_dim)
        
        if use_farm_token:
            self.farm_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.normal_(self.farm_token, std=0.02)
        
        self.transformer = TransformerEncoder(
            embed_dim, num_heads, num_layers, mlp_ratio, dropout
        )
        
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
        batch_size, n_turbines, _ = obs.shape
        
        h = self.obs_encoder(obs)
        pos_embed = self.pos_encoder(positions)
        h = torch.cat([h, pos_embed], dim=-1)
        h = self.input_proj(h)
        
        if self.use_farm_token:
            farm_tokens = self.farm_token.expand(batch_size, -1, -1)
            h = torch.cat([farm_tokens, h], dim=1)
            if key_padding_mask is not None:
                farm_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=h.device)
                key_padding_mask = torch.cat([farm_mask, key_padding_mask], dim=1)
        
        h, attn_weights = self.transformer(h, key_padding_mask)
        
        if self.use_farm_token:
            h = h[:, 1:, :]
        
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
        
        if key_padding_mask is not None:
            mask = ~key_padding_mask.unsqueeze(-1)
            log_prob = log_prob * mask.float()
        
        log_prob = log_prob.sum(dim=(-2, -1), keepdim=False).unsqueeze(-1)
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        
        return action, log_prob, mean_action, attn_weights


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
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_attention_heatmap(
    attention: np.ndarray,
    n_turbines: int,
    layer_idx: int = -1,
    head_idx: Optional[int] = None,
    ax: plt.Axes = None,
    title: str = None,
    show_values: bool = True,
) -> plt.Axes:
    """
    Plot attention weights as a heatmap.
    
    Args:
        attention: Attention matrix (n_turbines, n_turbines) or (n_heads, n_turb, n_turb)
        n_turbines: Number of turbines
        layer_idx: Layer index (for title)
        head_idx: If provided, shows specific head; if None, averages over heads
        ax: Matplotlib axes
        title: Custom title
        show_values: Whether to annotate cells with values
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Handle different input shapes
    if attention.ndim == 3:  # (n_heads, n_turb, n_turb)
        if head_idx is not None:
            attn_plot = attention[head_idx]
            default_title = f"Layer {layer_idx}, Head {head_idx}"
        else:
            attn_plot = attention.mean(axis=0)
            default_title = f"Layer {layer_idx}, Averaged over Heads"
    else:
        attn_plot = attention
        default_title = f"Attention Matrix"
    
    im = ax.imshow(attn_plot, cmap='Reds', vmin=0, aspect='equal')
    
    # Labels
    ax.set_xticks(range(n_turbines))
    ax.set_yticks(range(n_turbines))
    ax.set_xticklabels([f'T{i}' for i in range(n_turbines)])
    ax.set_yticklabels([f'T{i}' for i in range(n_turbines)])
    ax.set_xlabel('Key (attending FROM)', fontsize=10)
    ax.set_ylabel('Query (attending TO)', fontsize=10)
    
    # Colorbar
    plt.colorbar(im, ax=ax, label='Attention Weight', shrink=0.8)
    
    # Annotate values if small enough
    if show_values and n_turbines <= 8:
        for i in range(n_turbines):
            for j in range(n_turbines):
                val = attn_plot[i, j]
                color = 'white' if val > 0.5 * attn_plot.max() else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                       fontsize=8, color=color)
    
    ax.set_title(title if title else default_title)
    
    return ax


def plot_attention_on_farm(
    attention: np.ndarray,
    positions: np.ndarray,
    rotor_diameter: float,
    wind_direction: float = 270.0,
    ax: plt.Axes = None,
    threshold: float = 0.05,
    title: str = "Attention on Farm Layout",
    figsize: Tuple[int, int] = (10, 8),
    show_wind_arrow: bool = True,
) -> plt.Axes:
    """
    Overlay attention weights on the farm layout.
    
    Arrows show information flow: arrow from j to i means turbine i attends to turbine j.
    
    Args:
        attention: (n_turbines, n_turbines) attention matrix
        positions: (n_turbines, 2) raw positions in meters
        rotor_diameter: Rotor diameter for normalization
        wind_direction: Wind direction in degrees (meteorological convention)
        threshold: Only show attention weights above this threshold
        show_wind_arrow: Whether to show wind direction arrow
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    n_turbines = len(positions)
    pos_norm = positions / rotor_diameter  # Normalize by rotor diameter
    
    # Draw attention arrows first (behind turbines)
    max_attn = attention.max()
    if max_attn > 0:
        for i in range(n_turbines):
            for j in range(n_turbines):
                if i != j and attention[i, j] > threshold:
                    # Arrow from j to i (turbine i attends to turbine j)
                    alpha = min(attention[i, j] / max_attn, 1.0)
                    lw = 1 + 3 * alpha
                    
                    start = pos_norm[j]
                    end = pos_norm[i]
                    
                    # Shorten arrow to not overlap with circles
                    vec = end - start
                    length = np.linalg.norm(vec)
                    if length > 0:
                        vec_norm = vec / length
                        start_adj = start + vec_norm * 0.35
                        end_adj = end - vec_norm * 0.35
                        
                        ax.annotate(
                            '', xy=end_adj, xytext=start_adj,
                            arrowprops=dict(
                                arrowstyle='->,head_width=0.15,head_length=0.1',
                                color='crimson',
                                alpha=alpha * 0.8,
                                lw=lw,
                            ),
                            zorder=5
                        )
    
    # Draw turbines
    for i in range(n_turbines):
        x, y = pos_norm[i]
        circle = Circle((x, y), 0.25, color='steelblue', ec='black', lw=2, zorder=10)
        ax.add_patch(circle)
        ax.text(x, y, str(i), ha='center', va='center',
               fontsize=11, fontweight='bold', color='white', zorder=11)
    
    # Wind direction arrow
    if show_wind_arrow:
        wind_rad = np.radians(270 - wind_direction)  # Convert to math convention
        center = pos_norm.mean(axis=0)
        arrow_length = np.max(np.ptp(pos_norm, axis=0)) * 0.25
        
        # Wind comes FROM this direction
        dx = arrow_length * np.cos(wind_rad)
        dy = arrow_length * np.sin(wind_rad)
        
        # Draw arrow showing wind direction (from upwind to center)
        ax.annotate(
            '', xy=(center[0], center[1]),
            xytext=(center[0] - dx * 2, center[1] - dy * 2),
            arrowprops=dict(arrowstyle='->', color='green', lw=3),
            zorder=3
        )
        ax.text(center[0] - dx * 2.5, center[1] - dy * 2.5 + 0.3,
               f'Wind {wind_direction:.0f}°', ha='center', fontsize=10, color='darkgreen')
    
    # Formatting
    padding = 1.5
    ax.set_xlim(pos_norm[:, 0].min() - padding, pos_norm[:, 0].max() + padding)
    ax.set_ylim(pos_norm[:, 1].min() - padding, pos_norm[:, 1].max() + padding)
    ax.set_aspect('equal')
    ax.set_xlabel('x / D (Rotor Diameters)')
    ax.set_ylabel('y / D (Rotor Diameters)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_all_heads(
    attention_per_head: np.ndarray,
    positions: np.ndarray,
    rotor_diameter: float,
    wind_direction: float = 270.0,
    figsize_per_head: Tuple[int, int] = (5, 4),
    suptitle: str = "Attention by Head",
) -> plt.Figure:
    """
    Plot attention patterns for each attention head in a grid.
    
    Args:
        attention_per_head: (n_heads, n_turbines, n_turbines)
    """
    n_heads = attention_per_head.shape[0]
    n_cols = min(4, n_heads)
    n_rows = (n_heads + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_head[0] * n_cols, figsize_per_head[1] * n_rows)
    )
    axes = np.atleast_2d(axes).flatten()
    
    for h in range(n_heads):
        ax = axes[h]
        plot_attention_on_farm(
            attention_per_head[h],
            positions,
            rotor_diameter,
            wind_direction,
            ax=ax,
            title=f'Head {h}',
            show_wind_arrow=(h == 0)  # Only show wind on first plot
        )
    
    # Hide unused axes
    for h in range(n_heads, len(axes)):
        axes[h].set_visible(False)
    
    plt.suptitle(suptitle, fontsize=14, y=1.02)
    plt.tight_layout()
    
    return fig


def analyze_wake_alignment(
    attention: np.ndarray,
    positions: np.ndarray,
    wind_direction: float,
    rotor_diameter: float,
) -> Dict[str, Any]:
    """
    Analyze whether attention aligns with wake physics.
    
    In wake physics, downwind turbines are affected by upwind turbines.
    So we expect downwind turbines to attend to upwind turbines.
    
    Returns:
        Dictionary with analysis results
    """
    n_turbines = len(positions)
    pos_norm = positions / rotor_diameter
    
    # Convert wind direction to radians (math convention)
    wind_rad = np.radians(270 - wind_direction)
    wind_vec = np.array([np.cos(wind_rad), np.sin(wind_rad)])
    
    upwind_attention = []
    downwind_attention = []
    crosswind_attention = []
    
    for i in range(n_turbines):  # Query turbine (attending)
        for j in range(n_turbines):  # Key turbine (attended to)
            if i == j:
                continue
            
            # Vector from j to i
            vec_ji = pos_norm[i] - pos_norm[j]
            dist = np.linalg.norm(vec_ji)
            if dist < 0.1:
                continue
                
            vec_ji_norm = vec_ji / dist
            
            # Dot product with wind direction
            # Positive = j is upwind of i (i is downwind of j)
            alignment = np.dot(vec_ji_norm, wind_vec)
            
            attn_val = attention[i, j]
            
            if alignment > 0.5:  # j is upwind of i
                upwind_attention.append(attn_val)
            elif alignment < -0.5:  # j is downwind of i
                downwind_attention.append(attn_val)
            else:  # Crosswind
                crosswind_attention.append(attn_val)
    
    results = {
        'upwind_mean': np.mean(upwind_attention) if upwind_attention else 0,
        'downwind_mean': np.mean(downwind_attention) if downwind_attention else 0,
        'crosswind_mean': np.mean(crosswind_attention) if crosswind_attention else 0,
        'n_upwind_pairs': len(upwind_attention),
        'n_downwind_pairs': len(downwind_attention),
        'n_crosswind_pairs': len(crosswind_attention),
        'upwind_values': upwind_attention,
        'downwind_values': downwind_attention,
        'crosswind_values': crosswind_attention,
    }
    
    # Ratio (higher = better alignment with physics)
    if results['downwind_mean'] > 0:
        results['upwind_to_downwind_ratio'] = results['upwind_mean'] / results['downwind_mean']
    else:
        results['upwind_to_downwind_ratio'] = float('inf') if results['upwind_mean'] > 0 else 1.0
    
    return results


def plot_wake_analysis(analysis: Dict[str, Any], ax: plt.Axes = None) -> plt.Axes:
    """Plot wake alignment analysis as a bar chart."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    
    categories = ['Upwind', 'Crosswind', 'Downwind']
    means = [analysis['upwind_mean'], analysis['crosswind_mean'], analysis['downwind_mean']]
    counts = [analysis['n_upwind_pairs'], analysis['n_crosswind_pairs'], analysis['n_downwind_pairs']]
    
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    bars = ax.bar(categories, means, color=colors, edgecolor='black', lw=1.5)
    
    # Add count annotations
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
               f'n={count}', ha='center', va='bottom', fontsize=10)
    
    ax.set_ylabel('Mean Attention Weight')
    ax.set_title('Attention vs. Wake Direction\n(Higher upwind attention = physics-aligned)')
    ax.set_ylim(0, max(means) * 1.3 if max(means) > 0 else 1)
    
    # Add ratio annotation
    ratio = analysis['upwind_to_downwind_ratio']
    if ratio != float('inf'):
        ax.text(0.98, 0.95, f'Upwind/Downwind ratio: {ratio:.2f}',
               transform=ax.transAxes, ha='right', va='top',
               fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    return ax


def create_full_visualization(
    attention_weights: List[torch.Tensor],
    positions: np.ndarray,
    rotor_diameter: float,
    wind_direction: float,
    actions: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create a comprehensive visualization of attention patterns.
    
    Args:
        attention_weights: List of attention tensors from each layer
                          Each: (batch, n_heads, n_tokens, n_tokens)
        positions: (n_turbines, 2) raw positions
        rotor_diameter: For normalization
        wind_direction: Wind direction in degrees
        actions: Optional (n_turbines,) array of actions
        save_path: Optional path to save figure
    """
    # Extract attention from last layer, first batch element
    attn_last = attention_weights[-1][0].cpu().numpy()  # (n_heads, n_turb, n_turb)
    attn_avg = attn_last.mean(axis=0)  # Average over heads
    
    n_heads = attn_last.shape[0]
    n_turbines = attn_last.shape[1]
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Attention on farm layout (large, top-left)
    ax1 = fig.add_subplot(2, 2, 1)
    plot_attention_on_farm(
        attn_avg, positions, rotor_diameter, wind_direction,
        ax=ax1, title='Attention Pattern (Averaged over Heads)'
    )
    
    # 2. Attention heatmap (top-right)
    ax2 = fig.add_subplot(2, 2, 2)
    plot_attention_heatmap(
        attn_avg, n_turbines,
        ax=ax2, title='Attention Matrix (Averaged over Heads)'
    )
    
    # 3. Wake analysis (bottom-left)
    ax3 = fig.add_subplot(2, 2, 3)
    analysis = analyze_wake_alignment(attn_avg, positions, wind_direction, rotor_diameter)
    plot_wake_analysis(analysis, ax=ax3)
    
    # 4. Per-head attention heatmaps (bottom-right)
    ax4 = fig.add_subplot(2, 2, 4)
    # Show attention per head as stacked bars
    head_means = [attn_last[h].mean() for h in range(n_heads)]
    ax4.bar(range(n_heads), head_means, color='steelblue', edgecolor='black')
    ax4.set_xlabel('Attention Head')
    ax4.set_ylabel('Mean Attention Weight')
    ax4.set_title('Average Attention per Head')
    ax4.set_xticks(range(n_heads))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    return fig


# =============================================================================
# MAIN EVALUATION LOOP
# =============================================================================

def load_actor_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    verbose: bool = True,
) -> Tuple[TransformerActor, Dict[str, Any]]:
    """
    Load the actor network from a v9 checkpoint.
    
    Returns:
        (actor, args_dict)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint.get('args', {})
    
    if verbose:
        print(f"Loaded checkpoint from step {checkpoint.get('step', 'unknown')}")
        print(f"  embed_dim: {args.get('embed_dim', 128)}")
        print(f"  num_heads: {args.get('num_heads', 4)}")
        print(f"  num_layers: {args.get('num_layers', 2)}")
    
    # Need to infer obs_dim from saved state dict
    actor_state = checkpoint['actor_state_dict']
    
    # Get obs_dim from first layer of obs_encoder
    obs_encoder_weight = actor_state['obs_encoder.0.weight']
    obs_dim_per_turbine = obs_encoder_weight.shape[1]
    
    if verbose:
        print(f"  obs_dim_per_turbine: {obs_dim_per_turbine}")
    
    # Create actor with same architecture
    actor = TransformerActor(
        obs_dim_per_turbine=obs_dim_per_turbine,
        action_dim_per_turbine=1,
        embed_dim=args.get('embed_dim', 128),
        pos_embed_dim=args.get('pos_embed_dim', 32),
        num_heads=args.get('num_heads', 4),
        num_layers=args.get('num_layers', 2),
        mlp_ratio=args.get('mlp_ratio', 2.0),
        dropout=0.0,  # No dropout at eval
        use_farm_token=args.get('use_farm_token', False),
    ).to(device)
    
    actor.load_state_dict(actor_state)
    actor.eval()
    
    return actor, args


def evaluate_and_visualize(
    checkpoint_path: str,
    device: str = 'cuda',
    n_steps: int = 50,
    save_dir: str = 'attention_plots',
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Main function to evaluate a trained model and visualize attention.
    
    This is a simplified version that creates a mock environment for visualization.
    For full evaluation with the actual WindGym environment, see the extended version.
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    actor, args = load_actor_from_checkpoint(checkpoint_path, device)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Try to create actual environment
    try:
        from WindGym import WindFarmEnv
        from WindGym.wrappers import PerTurbineObservationWrapper
        from WindGym.utils.generate_layouts import generate_square_grid
        from py_wake.examples.data.dtu10mw import DTU10MW
        
        print("Creating WindGym environment...")
        
        # Get turbine
        turbine = DTU10MW()
        
        # Create layout
        layout_type = args.get('layouts', 'test_layout')
        if ',' in layout_type:
            layout_type = layout_type.split(',')[0].strip()
        
        if layout_type == 'test_layout':
            x_pos, y_pos = generate_square_grid(turbine=turbine, nx=2, ny=1, xDist=5, yDist=5)
        elif layout_type == 'square_2x2':
            x_pos, y_pos = generate_square_grid(turbine=turbine, nx=2, ny=2, xDist=5, yDist=5)
        else:
            x_pos, y_pos = generate_square_grid(turbine=turbine, nx=2, ny=1, xDist=5, yDist=5)
        
        # Environment config
        config = {
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
            "power_def": {"Power_reward": "Baseline", "Power_avg": 1, "Power_scaling": 1.0},
            "mes_level": {
                "turb_ws": True, "turb_wd": True, "turb_TI": False, "turb_power": True,
                "farm_ws": False, "farm_wd": False, "farm_TI": False, "farm_power": False,
            },
            "ws_mes": {"ws_current": False, "ws_rolling_mean": True, "ws_history_N": 15,
                      "ws_history_length": 15, "ws_window_length": 1},
            "wd_mes": {"wd_current": False, "wd_rolling_mean": True, "wd_history_N": 15,
                      "wd_history_length": 15, "wd_window_length": 1},
            "yaw_mes": {"yaw_current": False, "yaw_rolling_mean": True, "yaw_history_N": 15,
                       "yaw_history_length": 15, "yaw_window_length": 1},
            "power_mes": {"power_current": False, "power_rolling_mean": True, "power_history_N": 15,
                        "power_history_length": 15, "power_window_length": 1},
        }
        
        # Create environment
        base_env = WindFarmEnv(
            turbine=turbine,
            x_pos=x_pos,
            y_pos=y_pos,
            config=config,
            dt_sim=5,
            dt_env=10,
            n_passthrough=20,
            seed=seed,
        )
        env = PerTurbineObservationWrapper(base_env)
        
        # Get environment properties
        positions = env.turbine_positions
        rotor_diameter = env.rotor_diameter
        n_turbines = env.n_turbines
        
        # Reset
        obs, info = env.reset(seed=seed)
        wind_direction = base_env.wd
        
        print(f"Environment created: {n_turbines} turbines, wind={wind_direction:.1f}°")
        
        # Update action scaling
        action_high = base_env.action_space.high[0]
        action_low = base_env.action_space.low[0]
        actor.action_scale = torch.tensor((action_high - action_low) / 2.0, device=device)
        actor.action_bias = torch.tensor((action_high + action_low) / 2.0, device=device)
        
        # Collect attention over episode
        attention_history = []
        actions_history = []
        
        for step in range(n_steps):
            # Prepare inputs
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            
            # Positions: normalize and transform to wind-relative
            pos_tensor = torch.tensor(positions / rotor_diameter, dtype=torch.float32, device=device).unsqueeze(0)
            wind_tensor = torch.tensor([wind_direction], dtype=torch.float32, device=device)
            pos_transformed = transform_to_wind_relative(pos_tensor, wind_tensor)
            
            # Get action and attention
            with torch.no_grad():
                action, _, mean_action, attn_weights = actor.get_action(
                    obs_tensor, pos_transformed, deterministic=True
                )
            
            # Store attention (last layer, averaged over heads)
            attn_last = attn_weights[-1][0].cpu().numpy()  # (n_heads, n_turb, n_turb)
            attention_history.append(attn_last.mean(axis=0))
            actions_history.append(mean_action[0].cpu().numpy().flatten())
            
            # Step environment
            action_np = action[0].cpu().numpy().flatten()
            obs, reward, terminated, truncated, info = env.step(action_np)
            
            if terminated or truncated:
                break
        
        print(f"Collected {len(attention_history)} steps of attention data")
        
        # Create visualizations
        print("\nGenerating visualizations...")
        
        # 1. Full visualization for final step
        fig = create_full_visualization(
            attn_weights,  # From last step
            positions,
            rotor_diameter,
            wind_direction,
            actions=actions_history[-1],
            save_path=os.path.join(save_dir, 'attention_full.png')
        )
        plt.close(fig)
        
        # 2. Per-head visualization
        fig = plot_all_heads(
            attn_last,  # (n_heads, n_turb, n_turb) from last step
            positions,
            rotor_diameter,
            wind_direction,
            suptitle='Attention by Head (Final Step)'
        )
        plt.savefig(os.path.join(save_dir, 'attention_per_head.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # 3. Attention evolution over episode
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        timesteps = np.linspace(0, len(attention_history)-1, 6, dtype=int)
        
        for idx, t in enumerate(timesteps):
            ax = axes.flatten()[idx]
            plot_attention_on_farm(
                attention_history[t],
                positions,
                rotor_diameter,
                wind_direction,
                ax=ax,
                title=f'Step {t}',
                show_wind_arrow=(idx == 0)
            )
        
        plt.suptitle('Attention Evolution Over Episode', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'attention_evolution.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # 4. Average attention over episode
        avg_attention = np.mean(attention_history, axis=0)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        plot_attention_on_farm(
            avg_attention,
            positions,
            rotor_diameter,
            wind_direction,
            ax=ax1,
            title='Average Attention Over Episode'
        )
        
        analysis = analyze_wake_alignment(avg_attention, positions, wind_direction, rotor_diameter)
        plot_wake_analysis(analysis, ax=ax2)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'attention_average.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"\n✓ Visualizations saved to {save_dir}/")
        print("\nWake Analysis Results:")
        print(f"  Upwind attention (mean): {analysis['upwind_mean']:.4f}")
        print(f"  Downwind attention (mean): {analysis['downwind_mean']:.4f}")
        print(f"  Crosswind attention (mean): {analysis['crosswind_mean']:.4f}")
        print(f"  Upwind/Downwind ratio: {analysis['upwind_to_downwind_ratio']:.2f}")
        
        env.close()
        
        return {
            'attention_history': attention_history,
            'actions_history': actions_history,
            'positions': positions,
            'rotor_diameter': rotor_diameter,
            'wind_direction': wind_direction,
            'wake_analysis': analysis,
        }
        
    except ImportError as e:
        print(f"\nNote: Could not import WindGym ({e})")
        print("Creating synthetic data for demonstration...")
        
        # Create synthetic demo
        n_turbines = 2
        positions = np.array([[0, 0], [5 * 178.3, 0]])  # 2 turbines, 5D apart
        rotor_diameter = 178.3
        wind_direction = 270.0
        
        # Create random attention for demo
        attention = np.random.rand(n_turbines, n_turbines)
        attention = attention / attention.sum(axis=1, keepdims=True)
        
        fig = create_full_visualization(
            [torch.tensor(attention).unsqueeze(0).unsqueeze(0)],
            positions,
            rotor_diameter,
            wind_direction,
            save_path=os.path.join(save_dir, 'attention_demo.png')
        )
        
        print(f"\n✓ Demo visualization saved to {save_dir}/attention_demo.png")
        
        return {'demo': True}


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Visualize attention patterns from trained Transformer SAC model'
    )
    parser.add_argument(
        '--checkpoint', '-c', type=str, required=True,
        help='Path to model checkpoint (e.g., runs/RUN_NAME/checkpoints/step_100000.pt)'
    )
    parser.add_argument(
        '--device', type=str, default='cuda',
        help='Device to use (cuda/cpu)'
    )
    parser.add_argument(
        '--n_steps', type=int, default=50,
        help='Number of steps to run for attention collection'
    )
    parser.add_argument(
        '--save_dir', type=str, default='attention_plots',
        help='Directory to save visualizations'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed'
    )
    
    args = parser.parse_args()
    
    results = evaluate_and_visualize(
        checkpoint_path=args.checkpoint,
        device=args.device,
        n_steps=args.n_steps,
        save_dir=args.save_dir,
        seed=args.seed,
    )
    
    return results


if __name__ == '__main__':
    main()