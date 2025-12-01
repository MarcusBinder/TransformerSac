"""
Attention Evolution During Training

Analyzes how attention patterns evolve across training checkpoints.
Shows when the model learns wake physics and what patterns emerge.

Usage:
    python attention_evolution_training.py --run_dir runs/YOUR_RUN_NAME
    
    # Or specify checkpoints manually:
    python attention_evolution_training.py --checkpoints ckpt1.pt ckpt2.pt ckpt3.pt

Outputs:
    - attention_evolution.png: Attention heatmaps at different training steps
    - wake_alignment_curve.png: Upwind/downwind ratio over training
    - attention_animation.gif: Animated attention evolution (optional)
    - metrics.json: Quantitative metrics over training

Author: Marcus (DTU Wind Energy)
"""

import os
import sys
import argparse
import json
import glob
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import Normalize
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass
import math
from tqdm import tqdm


# =============================================================================
# MODEL COMPONENTS (same as visualize_attention.py)
# =============================================================================

class PositionalEncoding(nn.Module):
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
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 2.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim), nn.Dropout(dropout),
        )
    
    def forward(self, x, key_padding_mask=None):
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.attn(x_norm, x_norm, x_norm, 
                                            key_padding_mask=key_padding_mask,
                                            average_attn_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, attn_weights


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, num_layers: int, mlp_ratio: float = 2.0, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x, key_padding_mask=None):
        all_attn = []
        for layer in self.layers:
            x, attn = layer(x, key_padding_mask)
            all_attn.append(attn)
        return self.norm(x), all_attn


LOG_STD_MAX, LOG_STD_MIN = 2, -5

class TransformerActor(nn.Module):
    def __init__(self, obs_dim_per_turbine: int, action_dim_per_turbine: int = 1,
                 embed_dim: int = 128, pos_embed_dim: int = 32, num_heads: int = 4,
                 num_layers: int = 2, mlp_ratio: float = 2.0, dropout: float = 0.0,
                 use_farm_token: bool = False, action_scale: float = 1.0, action_bias: float = 0.0):
        super().__init__()
        self.obs_dim_per_turbine = obs_dim_per_turbine
        self.embed_dim = embed_dim
        self.use_farm_token = use_farm_token
        
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim_per_turbine, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim))
        self.pos_encoder = PositionalEncoding(pos_dim=2, embed_dim=pos_embed_dim)
        self.input_proj = nn.Linear(embed_dim + pos_embed_dim, embed_dim)
        
        if use_farm_token:
            self.farm_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.normal_(self.farm_token, std=0.02)
        
        self.transformer = TransformerEncoder(embed_dim, num_heads, num_layers, mlp_ratio, dropout)
        self.fc_mean = nn.Linear(embed_dim, action_dim_per_turbine)
        self.fc_logstd = nn.Linear(embed_dim, action_dim_per_turbine)
        self.register_buffer("action_scale", torch.tensor(action_scale))
        self.register_buffer("action_bias", torch.tensor(action_bias))
    
    def forward(self, obs, positions, key_padding_mask=None):
        batch_size = obs.shape[0]
        h = self.obs_encoder(obs)
        pos_embed = self.pos_encoder(positions)
        h = self.input_proj(torch.cat([h, pos_embed], dim=-1))
        
        if self.use_farm_token:
            h = torch.cat([self.farm_token.expand(batch_size, -1, -1), h], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat([
                    torch.zeros(batch_size, 1, dtype=torch.bool, device=h.device),
                    key_padding_mask
                ], dim=1)
        
        h, attn_weights = self.transformer(h, key_padding_mask)
        
        if self.use_farm_token:
            h = h[:, 1:, :]
        
        mean = self.fc_mean(h)
        log_std = torch.tanh(self.fc_logstd(h))
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std, attn_weights
    
    def get_action(self, obs, positions, key_padding_mask=None, deterministic=False):
        mean, log_std, attn_weights = self.forward(obs, positions, key_padding_mask)
        std = log_std.exp()
        
        if deterministic:
            x_t = mean
        else:
            x_t = torch.distributions.Normal(mean, std).rsample()
        
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        
        return action, None, mean_action, attn_weights


def transform_to_wind_relative(positions: torch.Tensor, wind_direction: float) -> torch.Tensor:
    """Transform positions to wind-relative coordinates."""
    theta = (wind_direction - 270.0) * (math.pi / 180.0)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    x, y = positions[..., 0:1], positions[..., 1:2]
    return torch.cat([cos_t * x - sin_t * y, sin_t * x + cos_t * y], dim=-1)


# =============================================================================
# CHECKPOINT UTILITIES
# =============================================================================

def find_checkpoints(run_dir: str) -> List[Tuple[int, str]]:
    """Find all checkpoints in a run directory, sorted by step."""
    checkpoint_dir = os.path.join(run_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        # Try run_dir directly
        checkpoint_dir = run_dir
    
    patterns = ['step_*.pt', 'checkpoint_*.pt', '*.pt']
    checkpoints = []
    
    for pattern in patterns:
        for path in glob.glob(os.path.join(checkpoint_dir, pattern)):
            # Extract step number from filename
            basename = os.path.basename(path)
            try:
                # Try different naming conventions
                if 'step_' in basename:
                    step = int(basename.split('step_')[1].split('.')[0])
                elif 'checkpoint_' in basename:
                    step = int(basename.split('checkpoint_')[1].split('.')[0])
                else:
                    # Try to extract any number
                    import re
                    numbers = re.findall(r'\d+', basename)
                    step = int(numbers[0]) if numbers else 0
                checkpoints.append((step, path))
            except (ValueError, IndexError):
                continue
    
    # Remove duplicates and sort by step
    checkpoints = list(set(checkpoints))
    checkpoints.sort(key=lambda x: x[0])
    
    return checkpoints


def load_actor_from_checkpoint(checkpoint_path: str, device: torch.device) -> Tuple[TransformerActor, Dict]:
    """Load actor from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint.get('args', {})
    actor_state = checkpoint['actor_state_dict']
    
    obs_dim = actor_state['obs_encoder.0.weight'].shape[1]
    
    actor = TransformerActor(
        obs_dim_per_turbine=obs_dim,
        action_dim_per_turbine=1,
        embed_dim=args.get('embed_dim', 128),
        pos_embed_dim=args.get('pos_embed_dim', 32),
        num_heads=args.get('num_heads', 4),
        num_layers=args.get('num_layers', 2),
        mlp_ratio=args.get('mlp_ratio', 2.0),
        use_farm_token=args.get('use_farm_token', False),
    ).to(device)
    
    actor.load_state_dict(actor_state)
    actor.eval()
    
    return actor, args


# =============================================================================
# ATTENTION EXTRACTION
# =============================================================================

def extract_attention_from_checkpoint(
    checkpoint_path: str,
    env,
    device: torch.device,
    n_steps: int = 20,
    n_episodes: int = 3,
) -> Dict[str, Any]:
    """
    Extract attention statistics from a checkpoint.
    
    Runs multiple episodes and averages attention patterns.
    """
    actor, args = load_actor_from_checkpoint(checkpoint_path, device)
    
    # Get env properties
    positions = env.turbine_positions
    rotor_diameter = env.rotor_diameter
    n_turbines = len(positions)
    
    # Update action scaling
    base_env = env.env if hasattr(env, 'env') else env
    action_high = base_env.action_space.high[0]
    action_low = base_env.action_space.low[0]
    actor.action_scale = torch.tensor((action_high - action_low) / 2.0, device=device)
    actor.action_bias = torch.tensor((action_high + action_low) / 2.0, device=device)
    
    all_attention = []
    all_rewards = []
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        wind_direction = getattr(base_env, 'wd', 270.0)
        
        episode_attention = []
        episode_reward = 0
        
        for step in range(n_steps):
            # Prepare inputs
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            pos_norm = positions / rotor_diameter
            pos_t = torch.tensor(pos_norm, dtype=torch.float32, device=device).unsqueeze(0)
            pos_transformed = transform_to_wind_relative(pos_t, wind_direction)
            
            with torch.no_grad():
                action, _, mean_action, attn_weights = actor.get_action(
                    obs_t, pos_transformed, deterministic=True
                )
            
            # Store attention (last layer, averaged over heads)
            attn_last = attn_weights[-1][0].cpu().numpy()  # (n_heads, n_turb, n_turb)
            episode_attention.append(attn_last.mean(axis=0))
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action[0].cpu().numpy().flatten())
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        all_attention.extend(episode_attention)
        all_rewards.append(episode_reward)
    
    # Compute statistics
    attention_array = np.array(all_attention)  # (total_steps, n_turb, n_turb)
    mean_attention = attention_array.mean(axis=0)
    
    # Wake alignment analysis
    wake_analysis = analyze_wake_alignment(mean_attention, positions, wind_direction, rotor_diameter)
    
    return {
        'mean_attention': mean_attention,
        'attention_std': attention_array.std(axis=0),
        'wake_analysis': wake_analysis,
        'mean_reward': np.mean(all_rewards),
        'n_samples': len(all_attention),
    }


def analyze_wake_alignment(attention, positions, wind_direction, rotor_diameter):
    """Analyze attention alignment with wake physics."""
    n_turbines = len(positions)
    pos_norm = positions / rotor_diameter
    
    wind_rad = np.radians(270 - wind_direction)
    wind_vec = np.array([np.cos(wind_rad), np.sin(wind_rad)])
    
    upwind, downwind, crosswind = [], [], []
    
    for i in range(n_turbines):
        for j in range(n_turbines):
            if i == j:
                continue
            vec_ji = pos_norm[i] - pos_norm[j]
            dist = np.linalg.norm(vec_ji)
            if dist < 0.1:
                continue
            alignment = np.dot(vec_ji / dist, wind_vec)
            
            if alignment > 0.5:
                upwind.append(attention[i, j])
            elif alignment < -0.5:
                downwind.append(attention[i, j])
            else:
                crosswind.append(attention[i, j])
    
    upwind_mean = np.mean(upwind) if upwind else 0
    downwind_mean = np.mean(downwind) if downwind else 0
    
    return {
        'upwind_mean': upwind_mean,
        'downwind_mean': downwind_mean,
        'crosswind_mean': np.mean(crosswind) if crosswind else 0,
        'ratio': upwind_mean / downwind_mean if downwind_mean > 0 else float('inf'),
        'n_upwind': len(upwind),
        'n_downwind': len(downwind),
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_attention_evolution_grid(
    results: List[Dict],
    steps: List[int],
    positions: np.ndarray,
    rotor_diameter: float,
    wind_direction: float,
    save_path: str = None,
    max_plots: int = 8,
):
    """Plot attention heatmaps at different training steps."""
    n_checkpoints = len(results)
    
    # Select evenly spaced checkpoints if too many
    if n_checkpoints > max_plots:
        indices = np.linspace(0, n_checkpoints - 1, max_plots, dtype=int)
    else:
        indices = list(range(n_checkpoints))
    
    n_plots = len(indices)
    n_cols = min(4, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.atleast_2d(axes).flatten()
    
    # Find global vmax for consistent colorscale
    vmax = max(results[i]['mean_attention'].max() for i in indices)
    
    for plot_idx, ckpt_idx in enumerate(indices):
        ax = axes[plot_idx]
        attention = results[ckpt_idx]['mean_attention']
        step = steps[ckpt_idx]
        
        im = ax.imshow(attention, cmap='Reds', vmin=0, vmax=vmax)
        ax.set_title(f'Step {step:,}', fontsize=10)
        
        n_turb = attention.shape[0]
        ax.set_xticks(range(n_turb))
        ax.set_yticks(range(n_turb))
        ax.set_xticklabels([f'T{i}' for i in range(n_turb)], fontsize=8)
        ax.set_yticklabels([f'T{i}' for i in range(n_turb)], fontsize=8)
        
        # Add values for small matrices
        if n_turb <= 4:
            for i in range(n_turb):
                for j in range(n_turb):
                    ax.text(j, i, f'{attention[i,j]:.2f}', ha='center', va='center', fontsize=7)
    
    # Hide unused axes
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)
    
    # Add colorbar
    fig.colorbar(im, ax=axes[:n_plots], shrink=0.6, label='Attention Weight')
    
    plt.suptitle('Attention Evolution During Training', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_attention_on_farm_evolution(
    results: List[Dict],
    steps: List[int],
    positions: np.ndarray,
    rotor_diameter: float,
    wind_direction: float,
    save_path: str = None,
    max_plots: int = 6,
):
    """Plot attention overlaid on farm layout at different training steps."""
    n_checkpoints = len(results)
    
    if n_checkpoints > max_plots:
        indices = np.linspace(0, n_checkpoints - 1, max_plots, dtype=int)
    else:
        indices = list(range(n_checkpoints))
    
    n_plots = len(indices)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = np.atleast_2d(axes).flatten()
    
    pos_norm = positions / rotor_diameter
    n_turbines = len(positions)
    
    for plot_idx, ckpt_idx in enumerate(indices):
        ax = axes[plot_idx]
        attention = results[ckpt_idx]['mean_attention']
        step = steps[ckpt_idx]
        
        max_attn = attention.max()
        
        # Draw attention arrows
        if max_attn > 0:
            for i in range(n_turbines):
                for j in range(n_turbines):
                    if i != j and attention[i, j] > 0.05:
                        alpha = min(attention[i, j] / max_attn, 1.0)
                        start, end = pos_norm[j], pos_norm[i]
                        vec = end - start
                        length = np.linalg.norm(vec)
                        if length > 0:
                            vec_n = vec / length
                            ax.annotate('', xy=end - vec_n * 0.3, xytext=start + vec_n * 0.3,
                                       arrowprops=dict(arrowstyle='->', color='crimson',
                                                      alpha=alpha * 0.8, lw=1 + 2 * alpha),
                                       zorder=5)
        
        # Draw turbines
        for i in range(n_turbines):
            circle = Circle(pos_norm[i], 0.25, color='steelblue', ec='black', lw=2, zorder=10)
            ax.add_patch(circle)
            ax.text(pos_norm[i, 0], pos_norm[i, 1], str(i),
                   ha='center', va='center', fontsize=10, fontweight='bold', color='white', zorder=11)
        
        # Wind arrow (only on first plot)
        if plot_idx == 0:
            wind_rad = np.radians(270 - wind_direction)
            center = pos_norm.mean(axis=0)
            arrow_len = np.max(np.ptp(pos_norm, axis=0)) * 0.2
            dx, dy = arrow_len * np.cos(wind_rad), arrow_len * np.sin(wind_rad)
            ax.annotate('', xy=center, xytext=(center[0] - dx * 2, center[1] - dy * 2),
                       arrowprops=dict(arrowstyle='->', color='green', lw=3), zorder=3)
            ax.text(center[0] - dx * 2.5, center[1] - dy * 2.5 + 0.3,
                   f'Wind {wind_direction:.0f}°', ha='center', fontsize=9, color='darkgreen')
        
        ax.set_xlim(pos_norm[:, 0].min() - 1.5, pos_norm[:, 0].max() + 1.5)
        ax.set_ylim(pos_norm[:, 1].min() - 1.5, pos_norm[:, 1].max() + 1.5)
        ax.set_aspect('equal')
        ax.set_title(f'Step {step:,}', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x / D')
        if plot_idx % n_cols == 0:
            ax.set_ylabel('y / D')
    
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Attention Patterns During Training', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_wake_alignment_curve(
    results: List[Dict],
    steps: List[int],
    save_path: str = None,
):
    """Plot wake alignment metrics over training."""
    upwind_means = [r['wake_analysis']['upwind_mean'] for r in results]
    downwind_means = [r['wake_analysis']['downwind_mean'] for r in results]
    crosswind_means = [r['wake_analysis']['crosswind_mean'] for r in results]
    ratios = [r['wake_analysis']['ratio'] for r in results]
    rewards = [r['mean_reward'] for r in results]
    
    # Cap infinite ratios for plotting
    ratios = [min(r, 10) for r in ratios]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Attention by direction
    ax1 = axes[0, 0]
    ax1.plot(steps, upwind_means, 'g-o', label='Upwind', markersize=4)
    ax1.plot(steps, downwind_means, 'r-s', label='Downwind', markersize=4)
    ax1.plot(steps, crosswind_means, 'b-^', label='Crosswind', markersize=4)
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Mean Attention Weight')
    ax1.set_title('Attention by Wind Direction')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log') if steps[-1] > 10000 else None
    
    # Plot 2: Upwind/Downwind ratio
    ax2 = axes[0, 1]
    ax2.plot(steps, ratios, 'purple', marker='o', markersize=4)
    ax2.axhline(y=1.0, color='gray', linestyle='--', label='Equal attention')
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Upwind / Downwind Ratio')
    ax2.set_title('Wake Physics Alignment\n(>1 = model attends more to upwind)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log') if steps[-1] > 10000 else None
    
    # Plot 3: Episode reward
    ax3 = axes[1, 0]
    ax3.plot(steps, rewards, 'orange', marker='o', markersize=4)
    ax3.set_xlabel('Training Step')
    ax3.set_ylabel('Mean Episode Reward')
    ax3.set_title('Performance Over Training')
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log') if steps[-1] > 10000 else None
    
    # Plot 4: Ratio vs Reward correlation
    ax4 = axes[1, 1]
    sc = ax4.scatter(ratios, rewards, c=steps, cmap='viridis', s=50)
    ax4.set_xlabel('Upwind/Downwind Ratio')
    ax4.set_ylabel('Mean Episode Reward')
    ax4.set_title('Physics Alignment vs Performance')
    plt.colorbar(sc, ax=ax4, label='Training Step')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def create_attention_animation(
    results: List[Dict],
    steps: List[int],
    positions: np.ndarray,
    rotor_diameter: float,
    wind_direction: float,
    save_path: str = None,
    fps: int = 2,
):
    """Create animated GIF of attention evolution."""
    pos_norm = positions / rotor_diameter
    n_turbines = len(positions)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    def init():
        ax.clear()
        return []
    
    def update(frame):
        ax.clear()
        
        attention = results[frame]['mean_attention']
        step = steps[frame]
        max_attn = attention.max()
        
        # Attention arrows
        if max_attn > 0:
            for i in range(n_turbines):
                for j in range(n_turbines):
                    if i != j and attention[i, j] > 0.05:
                        alpha = min(attention[i, j] / max_attn, 1.0)
                        start, end = pos_norm[j], pos_norm[i]
                        vec = end - start
                        length = np.linalg.norm(vec)
                        if length > 0:
                            vec_n = vec / length
                            ax.annotate('', xy=end - vec_n * 0.3, xytext=start + vec_n * 0.3,
                                       arrowprops=dict(arrowstyle='->', color='crimson',
                                                      alpha=alpha * 0.8, lw=1 + 2 * alpha))
        
        # Turbines
        for i in range(n_turbines):
            circle = Circle(pos_norm[i], 0.25, color='steelblue', ec='black', lw=2, zorder=10)
            ax.add_patch(circle)
            ax.text(pos_norm[i, 0], pos_norm[i, 1], str(i),
                   ha='center', va='center', fontsize=10, fontweight='bold', color='white', zorder=11)
        
        ax.set_xlim(pos_norm[:, 0].min() - 1.5, pos_norm[:, 0].max() + 1.5)
        ax.set_ylim(pos_norm[:, 1].min() - 1.5, pos_norm[:, 1].max() + 1.5)
        ax.set_aspect('equal')
        ax.set_title(f'Training Step: {step:,}', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x / D')
        ax.set_ylabel('y / D')
        
        return []
    
    anim = FuncAnimation(fig, update, init_func=init, frames=len(results),
                        interval=1000//fps, blit=False)
    
    if save_path:
        anim.save(save_path, writer=PillowWriter(fps=fps))
        print(f"Saved: {save_path}")
    
    plt.close(fig)
    return anim


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Analyze attention evolution during training')
    parser.add_argument('--run_dir', type=str, help='Path to training run directory')
    parser.add_argument('--checkpoints', type=str, nargs='+', help='List of checkpoint paths')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--n_steps', type=int, default=20, help='Steps per episode for evaluation')
    parser.add_argument('--n_episodes', type=int, default=3, help='Episodes per checkpoint')
    parser.add_argument('--save_dir', type=str, default='attention_evolution', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--no_animation', action='store_true', help='Skip animation creation')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Find checkpoints
    if args.checkpoints:
        checkpoint_paths = args.checkpoints
        steps = []
        for path in checkpoint_paths:
            try:
                import re
                numbers = re.findall(r'\d+', os.path.basename(path))
                steps.append(int(numbers[0]) if numbers else 0)
            except:
                steps.append(0)
    elif args.run_dir:
        checkpoints = find_checkpoints(args.run_dir)
        if not checkpoints:
            print(f"No checkpoints found in {args.run_dir}")
            return
        steps, checkpoint_paths = zip(*checkpoints)
        steps, checkpoint_paths = list(steps), list(checkpoint_paths)
    else:
        print("Please provide --run_dir or --checkpoints")
        return
    
    print(f"Found {len(checkpoint_paths)} checkpoints")
    for step, path in zip(steps, checkpoint_paths):
        print(f"  Step {step}: {os.path.basename(path)}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Create environment
    try:
        from WindGym import WindFarmEnv
        from WindGym.wrappers import PerTurbineObservationWrapper
        from WindGym.utils.generate_layouts import generate_square_grid
        from py_wake.examples.data.dtu10mw import DTU10MW
        
        print("\nCreating environment...")
        turbine = DTU10MW()
        
        # Try to infer layout from first checkpoint
        first_ckpt = torch.load(checkpoint_paths[0], map_location='cpu')
        first_args = first_ckpt.get('args', {})
        layout_str = first_args.get('layouts', 'square_2x2')
        if ',' in layout_str:
            layout_str = layout_str.split(',')[0].strip()
        
        print(f"Layout: {layout_str}")
        
        # Generate layout
        if layout_str == 'test_layout':
            x_pos, y_pos = generate_square_grid(turbine=turbine, nx=2, ny=1, xDist=5, yDist=5)
        elif layout_str == 'square_2x2':
            x_pos, y_pos = generate_square_grid(turbine=turbine, nx=2, ny=2, xDist=5, yDist=5)
        elif layout_str == 'square_3x3':
            x_pos, y_pos = generate_square_grid(turbine=turbine, nx=3, ny=3, xDist=5, yDist=5)
        else:
            x_pos, y_pos = generate_square_grid(turbine=turbine, nx=2, ny=2, xDist=5, yDist=5)
        
        config = {
            "yaw_init": "Random", "BaseController": "Local", "ActionMethod": "yaw",
            "Track_power": False, "farm": {"yaw_min": -30, "yaw_max": 30},
            "wind": {"ws_min": 9, "ws_max": 9, "TI_min": 0.05, "TI_max": 0.05, "wd_min": 265, "wd_max": 275},
            "act_pen": {"action_penalty": 0.0, "action_penalty_type": "Change"},
            "power_def": {"Power_reward": "Baseline", "Power_avg": 1, "Power_scaling": 1.0},
            "mes_level": {"turb_ws": True, "turb_wd": True, "turb_TI": False, "turb_power": True,
                         "farm_ws": False, "farm_wd": False, "farm_TI": False, "farm_power": False},
            "ws_mes": {"ws_current": False, "ws_rolling_mean": True, "ws_history_N": 15,
                      "ws_history_length": 15, "ws_window_length": 1},
            "wd_mes": {"wd_current": False, "wd_rolling_mean": True, "wd_history_N": 15,
                      "wd_history_length": 15, "wd_window_length": 1},
            "yaw_mes": {"yaw_current": False, "yaw_rolling_mean": True, "yaw_history_N": 15,
                       "yaw_history_length": 15, "yaw_window_length": 1},
            "power_mes": {"power_current": False, "power_rolling_mean": True, "power_history_N": 15,
                        "power_history_length": 15, "power_window_length": 1},
        }
        
        base_env = WindFarmEnv(
            turbine=turbine, x_pos=x_pos, y_pos=y_pos, config=config,
            dt_sim=5, dt_env=10, n_passthrough=20, seed=args.seed,
        )
        env = PerTurbineObservationWrapper(base_env)
        
        positions = env.turbine_positions
        rotor_diameter = env.rotor_diameter
        wind_direction = base_env.wd
        
        print(f"Environment: {len(positions)} turbines, wind={wind_direction:.1f}°")
        
    except ImportError as e:
        print(f"Could not import WindGym: {e}")
        print("Cannot analyze attention without environment.")
        return
    
    # Extract attention from each checkpoint
    print("\nExtracting attention from checkpoints...")
    results = []
    
    for step, path in tqdm(zip(steps, checkpoint_paths), total=len(steps)):
        try:
            result = extract_attention_from_checkpoint(
                path, env, device, n_steps=args.n_steps, n_episodes=args.n_episodes
            )
            results.append(result)
        except Exception as e:
            print(f"  Error processing {path}: {e}")
            results.append(None)
    
    # Filter out failed checkpoints
    valid_indices = [i for i, r in enumerate(results) if r is not None]
    results = [results[i] for i in valid_indices]
    steps = [steps[i] for i in valid_indices]
    
    if not results:
        print("No valid results to plot")
        return
    
    print(f"\nSuccessfully processed {len(results)} checkpoints")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # 1. Attention heatmap evolution
    plot_attention_evolution_grid(
        results, steps, positions, rotor_diameter, wind_direction,
        save_path=os.path.join(args.save_dir, 'attention_heatmaps.png')
    )
    plt.close()
    
    # 2. Attention on farm layout evolution
    plot_attention_on_farm_evolution(
        results, steps, positions, rotor_diameter, wind_direction,
        save_path=os.path.join(args.save_dir, 'attention_on_farm.png')
    )
    plt.close()
    
    # 3. Wake alignment curves
    plot_wake_alignment_curve(
        results, steps,
        save_path=os.path.join(args.save_dir, 'wake_alignment.png')
    )
    plt.close()
    
    # 4. Animation (optional)
    if not args.no_animation and len(results) >= 3:
        print("Creating animation...")
        create_attention_animation(
            results, steps, positions, rotor_diameter, wind_direction,
            save_path=os.path.join(args.save_dir, 'attention_animation.gif'),
            fps=2
        )
    
    # 5. Save metrics to JSON
    metrics = {
        'steps': steps,
        'upwind_mean': [r['wake_analysis']['upwind_mean'] for r in results],
        'downwind_mean': [r['wake_analysis']['downwind_mean'] for r in results],
        'ratio': [r['wake_analysis']['ratio'] for r in results],
        'mean_reward': [r['mean_reward'] for r in results],
    }
    
    with open(os.path.join(args.save_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    
    print(f"\n✓ All outputs saved to {args.save_dir}/")
    print("\nSummary:")
    print(f"  First checkpoint (step {steps[0]}):")
    print(f"    Upwind/Downwind ratio: {results[0]['wake_analysis']['ratio']:.2f}")
    print(f"    Mean reward: {results[0]['mean_reward']:.2f}")
    print(f"  Last checkpoint (step {steps[-1]}):")
    print(f"    Upwind/Downwind ratio: {results[-1]['wake_analysis']['ratio']:.2f}")
    print(f"    Mean reward: {results[-1]['mean_reward']:.2f}")
    
    env.close()


if __name__ == '__main__':
    main()