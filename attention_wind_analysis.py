"""
Wind Direction Conditioned Attention Analysis

Analyzes how attention patterns change with wind direction.
Designed for models trained with variable wind directions.

Key analyses:
1. Attention patterns binned by wind direction
2. Per-head specialization analysis
3. "Reference turbine" tracking - does it rotate with wind?
4. Upstream/downstream attention conditioned on wind direction

Usage:
    python attention_wind_analysis.py --checkpoint runs/YOUR_RUN/checkpoints/step_100000.pt
    
    # Or analyze multiple checkpoints:
    python attention_wind_analysis.py --run_dir runs/YOUR_RUN

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
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass
import math
from tqdm import tqdm
from collections import defaultdict


# =============================================================================
# MODEL COMPONENTS (same as before)
# =============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, pos_dim: int = 2, embed_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(pos_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim))
    
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
        self.num_heads = num_heads
        
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
    theta = (wind_direction - 270.0) * (math.pi / 180.0)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    x, y = positions[..., 0:1], positions[..., 1:2]
    return torch.cat([cos_t * x - sin_t * y, sin_t * x + cos_t * y], dim=-1)


# =============================================================================
# WIND DIRECTION UTILITIES
# =============================================================================

def get_upwind_downwind_turbines(positions: np.ndarray, wind_direction: float, rotor_diameter: float) -> Dict:
    """
    Classify turbines as upwind or downwind based on wind direction.
    
    Returns dict with:
        - upwind_indices: list of turbine indices that are upwind
        - downwind_indices: list of turbine indices that are downwind
        - wind_axis_positions: position along wind axis for each turbine
    """
    pos_norm = positions / rotor_diameter
    n_turbines = len(positions)
    
    # Wind vector (direction wind is coming FROM)
    wind_rad = np.radians(270 - wind_direction)  # Convert to math convention
    wind_vec = np.array([np.cos(wind_rad), np.sin(wind_rad)])
    
    # Project positions onto wind axis
    # More positive = more upwind
    wind_axis_positions = np.dot(pos_norm, wind_vec)
    
    # Classify based on median (or could use more sophisticated clustering)
    median_pos = np.median(wind_axis_positions)
    
    upwind_indices = np.where(wind_axis_positions >= median_pos)[0].tolist()
    downwind_indices = np.where(wind_axis_positions < median_pos)[0].tolist()
    
    # Find most upwind and most downwind
    most_upwind = int(np.argmax(wind_axis_positions))
    most_downwind = int(np.argmin(wind_axis_positions))
    
    return {
        'upwind_indices': upwind_indices,
        'downwind_indices': downwind_indices,
        'wind_axis_positions': wind_axis_positions,
        'most_upwind': most_upwind,
        'most_downwind': most_downwind,
    }


def get_wind_direction_bin(wind_direction: float, n_bins: int = 4) -> int:
    """Assign wind direction to a bin."""
    # Normalize to [0, 360)
    wd = wind_direction % 360
    bin_size = 360 / n_bins
    return int(wd // bin_size)


def get_bin_label(bin_idx: int, n_bins: int = 4) -> str:
    """Get human-readable label for wind direction bin."""
    bin_size = 360 / n_bins
    start = bin_idx * bin_size
    end = (bin_idx + 1) * bin_size
    
    # Cardinal directions
    directions = {0: 'N', 45: 'NE', 90: 'E', 135: 'SE', 
                  180: 'S', 225: 'SW', 270: 'W', 315: 'NW'}
    
    mid = (start + end) / 2
    closest = min(directions.keys(), key=lambda x: min(abs(x - mid), 360 - abs(x - mid)))
    
    return f"{start:.0f}°-{end:.0f}° ({directions[closest]})"


# =============================================================================
# DATA COLLECTION
# =============================================================================

def collect_attention_data(
    actor: TransformerActor,
    env,
    device: torch.device,
    n_episodes: int = 20,
    n_steps_per_episode: int = 30,
    seed: int = 42,
) -> List[Dict]:
    """
    Collect attention data across multiple episodes with different wind directions.
    
    Returns list of dicts, one per step, containing:
        - wind_direction
        - attention_per_head: (n_heads, n_turb, n_turb)
        - attention_avg: (n_turb, n_turb)
        - action
        - reward
    """
    positions = env.turbine_positions
    rotor_diameter = env.rotor_diameter
    n_turbines = len(positions)
    
    base_env = env.env if hasattr(env, 'env') else env
    
    # Update action scaling
    action_high = base_env.action_space.high[0]
    action_low = base_env.action_space.low[0]
    actor.action_scale = torch.tensor((action_high - action_low) / 2.0, device=device)
    actor.action_bias = torch.tensor((action_high + action_low) / 2.0, device=device)
    
    all_data = []
    
    for ep in tqdm(range(n_episodes), desc="Collecting episodes"):
        obs, info = env.reset(seed=seed + ep)
        wind_direction = getattr(base_env, 'wd', 270.0)
        
        for step in range(n_steps_per_episode):
            # Prepare inputs
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            pos_norm = positions / rotor_diameter
            pos_t = torch.tensor(pos_norm, dtype=torch.float32, device=device).unsqueeze(0)
            pos_transformed = transform_to_wind_relative(pos_t, wind_direction)
            
            with torch.no_grad():
                action, _, mean_action, attn_weights = actor.get_action(
                    obs_t, pos_transformed, deterministic=True
                )
            
            # Extract attention from last layer
            attn_last = attn_weights[-1][0].cpu().numpy()  # (n_heads, n_turb, n_turb)
            
            # Step environment
            action_np = action[0].cpu().numpy().flatten()
            next_obs, reward, terminated, truncated, info = env.step(action_np)
            
            all_data.append({
                'wind_direction': wind_direction,
                'attention_per_head': attn_last,
                'attention_avg': attn_last.mean(axis=0),
                'action': action_np,
                'mean_action': mean_action[0].cpu().numpy().flatten(),
                'reward': reward,
                'episode': ep,
                'step': step,
            })
            
            obs = next_obs
            
            if terminated or truncated:
                break
    
    return all_data


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_by_wind_direction(
    data: List[Dict],
    positions: np.ndarray,
    rotor_diameter: float,
    n_bins: int = 4,
) -> Dict:
    """
    Analyze attention patterns binned by wind direction.
    """
    n_turbines = len(positions)
    
    # Bin the data
    binned_data = defaultdict(list)
    for d in data:
        bin_idx = get_wind_direction_bin(d['wind_direction'], n_bins)
        binned_data[bin_idx].append(d)
    
    results = {}
    
    for bin_idx in range(n_bins):
        if bin_idx not in binned_data or len(binned_data[bin_idx]) == 0:
            continue
        
        bin_data = binned_data[bin_idx]
        
        # Average attention for this bin
        attention_avg = np.mean([d['attention_avg'] for d in bin_data], axis=0)
        attention_per_head = np.mean([d['attention_per_head'] for d in bin_data], axis=0)
        
        # Get representative wind direction for analysis
        mean_wd = np.mean([d['wind_direction'] for d in bin_data])
        
        # Classify upwind/downwind for this wind direction
        ud_info = get_upwind_downwind_turbines(positions, mean_wd, rotor_diameter)
        
        # Compute upwind/downwind attention
        upwind_attn = []
        downwind_attn = []
        
        for i in range(n_turbines):
            for j in range(n_turbines):
                if i == j:
                    continue
                # Is j upwind of i?
                if ud_info['wind_axis_positions'][j] > ud_info['wind_axis_positions'][i]:
                    upwind_attn.append(attention_avg[i, j])
                else:
                    downwind_attn.append(attention_avg[i, j])
        
        upwind_mean = np.mean(upwind_attn) if upwind_attn else 0
        downwind_mean = np.mean(downwind_attn) if downwind_attn else 0
        
        # Find which turbine receives most attention (reference turbine)
        col_sums = attention_avg.sum(axis=0)  # Sum over queries (who attends to this turbine)
        reference_turbine = int(np.argmax(col_sums))
        
        results[bin_idx] = {
            'label': get_bin_label(bin_idx, n_bins),
            'mean_wind_direction': mean_wd,
            'n_samples': len(bin_data),
            'attention_avg': attention_avg,
            'attention_per_head': attention_per_head,
            'upwind_mean': upwind_mean,
            'downwind_mean': downwind_mean,
            'ratio': upwind_mean / downwind_mean if downwind_mean > 0 else float('inf'),
            'reference_turbine': reference_turbine,
            'most_upwind': ud_info['most_upwind'],
            'most_downwind': ud_info['most_downwind'],
            'upwind_indices': ud_info['upwind_indices'],
            'downwind_indices': ud_info['downwind_indices'],
        }
    
    return results


def analyze_per_head(
    data: List[Dict],
    positions: np.ndarray,
    rotor_diameter: float,
) -> Dict:
    """
    Analyze what each attention head specializes in.
    """
    n_turbines = len(positions)
    
    # Get number of heads from first sample
    n_heads = data[0]['attention_per_head'].shape[0]
    
    head_analysis = {}
    
    for head_idx in range(n_heads):
        # Collect attention for this head across all samples
        head_attention = np.mean([d['attention_per_head'][head_idx] for d in data], axis=0)
        
        # Analyze patterns
        # 1. Does this head focus on specific turbines?
        col_sums = head_attention.sum(axis=0)
        focus_turbine = int(np.argmax(col_sums))
        focus_score = col_sums[focus_turbine] / col_sums.sum()
        
        # 2. Does this head do nearest-neighbor attention?
        pos_norm = positions / rotor_diameter
        nn_attention = 0
        far_attention = 0
        n_nn = 0
        n_far = 0
        
        for i in range(n_turbines):
            for j in range(n_turbines):
                if i == j:
                    continue
                dist = np.linalg.norm(pos_norm[i] - pos_norm[j])
                if dist < 6:  # Nearest neighbor (assuming 5D spacing)
                    nn_attention += head_attention[i, j]
                    n_nn += 1
                else:
                    far_attention += head_attention[i, j]
                    n_far += 1
        
        nn_mean = nn_attention / n_nn if n_nn > 0 else 0
        far_mean = far_attention / n_far if n_far > 0 else 0
        
        # 3. Uniformity - is attention spread evenly or concentrated?
        # Entropy-like measure
        attn_flat = head_attention.flatten()
        attn_flat = attn_flat / attn_flat.sum()
        entropy = -np.sum(attn_flat * np.log(attn_flat + 1e-10))
        max_entropy = np.log(len(attn_flat))
        uniformity = entropy / max_entropy
        
        head_analysis[head_idx] = {
            'attention_matrix': head_attention,
            'focus_turbine': focus_turbine,
            'focus_score': focus_score,
            'nearest_neighbor_mean': nn_mean,
            'far_mean': far_mean,
            'nn_preference': nn_mean / far_mean if far_mean > 0 else float('inf'),
            'uniformity': uniformity,  # 0 = concentrated, 1 = uniform
        }
    
    return head_analysis


def analyze_reference_turbine_rotation(
    binned_results: Dict,
    positions: np.ndarray,
    rotor_diameter: float,
) -> Dict:
    """
    Check if the "reference turbine" (most attended to) rotates with wind direction.
    """
    analysis = {
        'reference_matches_downwind': [],
        'reference_matches_upwind': [],
        'details': [],
    }
    
    for bin_idx, result in binned_results.items():
        ref = result['reference_turbine']
        most_downwind = result['most_downwind']
        most_upwind = result['most_upwind']
        
        matches_downwind = ref == most_downwind
        matches_upwind = ref == most_upwind
        
        analysis['reference_matches_downwind'].append(matches_downwind)
        analysis['reference_matches_upwind'].append(matches_upwind)
        analysis['details'].append({
            'bin': result['label'],
            'wind_direction': result['mean_wind_direction'],
            'reference_turbine': ref,
            'most_downwind': most_downwind,
            'most_upwind': most_upwind,
            'matches_downwind': matches_downwind,
            'matches_upwind': matches_upwind,
        })
    
    analysis['pct_matches_downwind'] = np.mean(analysis['reference_matches_downwind']) * 100
    analysis['pct_matches_upwind'] = np.mean(analysis['reference_matches_upwind']) * 100
    
    return analysis


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_attention_by_wind_direction(
    binned_results: Dict,
    positions: np.ndarray,
    rotor_diameter: float,
    save_path: str = None,
):
    """
    Plot attention heatmaps for each wind direction bin.
    """
    n_bins = len(binned_results)
    
    fig, axes = plt.subplots(2, n_bins, figsize=(5 * n_bins, 10))
    
    pos_norm = positions / rotor_diameter
    n_turbines = len(positions)
    
    for idx, (bin_idx, result) in enumerate(sorted(binned_results.items())):
        # Top row: Attention heatmap
        ax1 = axes[0, idx]
        attention = result['attention_avg']
        
        im = ax1.imshow(attention, cmap='Reds', vmin=0)
        ax1.set_title(f"{result['label']}\n(n={result['n_samples']})", fontsize=10)
        ax1.set_xticks(range(n_turbines))
        ax1.set_yticks(range(n_turbines))
        ax1.set_xticklabels([f'T{i}' for i in range(n_turbines)], fontsize=8)
        ax1.set_yticklabels([f'T{i}' for i in range(n_turbines)], fontsize=8)
        
        if n_turbines <= 4:
            for i in range(n_turbines):
                for j in range(n_turbines):
                    ax1.text(j, i, f'{attention[i,j]:.2f}', ha='center', va='center', fontsize=7)
        
        plt.colorbar(im, ax=ax1, shrink=0.7)
        
        # Bottom row: Attention on farm layout
        ax2 = axes[1, idx]
        
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
                            ax2.annotate('', xy=end - vec_n * 0.3, xytext=start + vec_n * 0.3,
                                        arrowprops=dict(arrowstyle='->', color='crimson',
                                                       alpha=alpha * 0.8, lw=1 + 2 * alpha),
                                        zorder=5)
        
        # Draw turbines - color by upwind/downwind
        for i in range(n_turbines):
            if i in result['upwind_indices']:
                color = 'forestgreen'
            else:
                color = 'steelblue'
            
            # Highlight reference turbine
            lw = 4 if i == result['reference_turbine'] else 2
            ec = 'gold' if i == result['reference_turbine'] else 'black'
            
            circle = Circle(pos_norm[i], 0.25, color=color, ec=ec, lw=lw, zorder=10)
            ax2.add_patch(circle)
            ax2.text(pos_norm[i, 0], pos_norm[i, 1], str(i),
                    ha='center', va='center', fontsize=10, fontweight='bold', color='white', zorder=11)
        
        # Wind arrow
        wd = result['mean_wind_direction']
        wind_rad = np.radians(270 - wd)
        center = pos_norm.mean(axis=0)
        arrow_len = 1.0
        dx, dy = arrow_len * np.cos(wind_rad), arrow_len * np.sin(wind_rad)
        ax2.annotate('', xy=center, xytext=(center[0] - dx * 2, center[1] - dy * 2),
                    arrowprops=dict(arrowstyle='->', color='darkgreen', lw=3), zorder=3)
        ax2.text(center[0] - dx * 2.5, center[1] - dy * 2.5,
                f'{wd:.0f}°', ha='center', fontsize=9, color='darkgreen')
        
        ax2.set_xlim(pos_norm[:, 0].min() - 1.5, pos_norm[:, 0].max() + 1.5)
        ax2.set_ylim(pos_norm[:, 1].min() - 1.5, pos_norm[:, 1].max() + 1.5)
        ax2.set_aspect('equal')
        ax2.set_xlabel('x / D')
        if idx == 0:
            ax2.set_ylabel('y / D')
        ax2.grid(True, alpha=0.3)
    
    # Legend
    legend_elements = [
        mpatches.Patch(color='forestgreen', label='Upwind'),
        mpatches.Patch(color='steelblue', label='Downwind'),
        mpatches.Patch(ec='gold', fc='white', lw=3, label='Reference turbine'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=10)
    
    plt.suptitle('Attention Patterns by Wind Direction', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_per_head_analysis(
    head_analysis: Dict,
    positions: np.ndarray,
    rotor_diameter: float,
    save_path: str = None,
):
    """
    Plot attention pattern for each head.
    """
    n_heads = len(head_analysis)
    n_turbines = len(positions)
    pos_norm = positions / rotor_diameter
    
    fig, axes = plt.subplots(2, n_heads, figsize=(4 * n_heads, 8))
    
    for head_idx in range(n_heads):
        result = head_analysis[head_idx]
        attention = result['attention_matrix']
        
        # Top: Heatmap
        ax1 = axes[0, head_idx]
        im = ax1.imshow(attention, cmap='Reds', vmin=0)
        ax1.set_title(f'Head {head_idx}\nFocus: T{result["focus_turbine"]} ({result["focus_score"]:.0%})', fontsize=10)
        ax1.set_xticks(range(n_turbines))
        ax1.set_yticks(range(n_turbines))
        ax1.set_xticklabels([f'T{i}' for i in range(n_turbines)], fontsize=8)
        ax1.set_yticklabels([f'T{i}' for i in range(n_turbines)], fontsize=8)
        plt.colorbar(im, ax=ax1, shrink=0.7)
        
        # Bottom: Farm layout
        ax2 = axes[1, head_idx]
        max_attn = attention.max()
        
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
                            ax2.annotate('', xy=end - vec_n * 0.3, xytext=start + vec_n * 0.3,
                                        arrowprops=dict(arrowstyle='->', color='crimson',
                                                       alpha=alpha * 0.8, lw=1 + 2 * alpha))
        
        for i in range(n_turbines):
            ec = 'gold' if i == result['focus_turbine'] else 'black'
            lw = 4 if i == result['focus_turbine'] else 2
            circle = Circle(pos_norm[i], 0.25, color='steelblue', ec=ec, lw=lw, zorder=10)
            ax2.add_patch(circle)
            ax2.text(pos_norm[i, 0], pos_norm[i, 1], str(i),
                    ha='center', va='center', fontsize=10, fontweight='bold', color='white', zorder=11)
        
        ax2.set_xlim(pos_norm[:, 0].min() - 1.5, pos_norm[:, 0].max() + 1.5)
        ax2.set_ylim(pos_norm[:, 1].min() - 1.5, pos_norm[:, 1].max() + 1.5)
        ax2.set_aspect('equal')
        ax2.set_xlabel('x / D')
        if head_idx == 0:
            ax2.set_ylabel('y / D')
        ax2.grid(True, alpha=0.3)
        
        # Add head characteristics
        chars = f"NN pref: {result['nn_preference']:.1f}x\nUniform: {result['uniformity']:.2f}"
        ax2.text(0.02, 0.98, chars, transform=ax2.transAxes, fontsize=8,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Per-Head Attention Analysis', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_reference_turbine_analysis(
    binned_results: Dict,
    rotation_analysis: Dict,
    save_path: str = None,
):
    """
    Plot analysis of whether reference turbine rotates with wind.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Reference turbine by wind direction
    ax1 = axes[0]
    
    wind_dirs = [d['wind_direction'] for d in rotation_analysis['details']]
    ref_turbines = [d['reference_turbine'] for d in rotation_analysis['details']]
    most_downwind = [d['most_downwind'] for d in rotation_analysis['details']]
    
    x = range(len(wind_dirs))
    width = 0.35
    
    ax1.bar([i - width/2 for i in x], ref_turbines, width, label='Reference Turbine', color='crimson')
    ax1.bar([i + width/2 for i in x], most_downwind, width, label='Most Downwind', color='steelblue')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{wd:.0f}°' for wd in wind_dirs], rotation=45)
    ax1.set_xlabel('Wind Direction')
    ax1.set_ylabel('Turbine Index')
    ax1.set_title('Reference Turbine vs Most Downwind Turbine')
    ax1.legend()
    ax1.set_yticks(range(4))
    
    # Right: Summary statistics
    ax2 = axes[1]
    
    categories = ['Matches\nDownwind', 'Matches\nUpwind', 'Neither']
    pct_downwind = rotation_analysis['pct_matches_downwind']
    pct_upwind = rotation_analysis['pct_matches_upwind']
    pct_neither = 100 - max(pct_downwind, pct_upwind)
    
    values = [pct_downwind, pct_upwind, pct_neither]
    colors = ['steelblue', 'forestgreen', 'gray']
    
    bars = ax2.bar(categories, values, color=colors)
    ax2.set_ylabel('Percentage of Wind Direction Bins')
    ax2.set_title('Does Reference Turbine Rotate with Wind?')
    ax2.set_ylim(0, 100)
    
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.0f}%', ha='center', fontsize=12, fontweight='bold')
    
    # Interpretation
    if pct_downwind > 60:
        interpretation = "✓ Reference turbine tracks downwind position"
    elif pct_upwind > 60:
        interpretation = "✓ Reference turbine tracks upwind position"
    else:
        interpretation = "✗ Reference turbine does NOT rotate with wind"
    
    ax2.text(0.5, 0.02, interpretation, transform=ax2.transAxes, ha='center',
            fontsize=11, style='italic',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_upwind_downwind_by_direction(
    binned_results: Dict,
    save_path: str = None,
):
    """
    Plot upwind vs downwind attention for each wind direction bin.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    bins = sorted(binned_results.keys())
    labels = [binned_results[b]['label'] for b in bins]
    upwind_means = [binned_results[b]['upwind_mean'] for b in bins]
    downwind_means = [binned_results[b]['downwind_mean'] for b in bins]
    ratios = [binned_results[b]['ratio'] for b in bins]
    
    # Cap ratios for plotting
    ratios = [min(r, 5) for r in ratios]
    
    # Left: Stacked bar
    ax1 = axes[0]
    x = range(len(bins))
    width = 0.35
    
    ax1.bar([i - width/2 for i in x], upwind_means, width, label='Upwind Attention', color='forestgreen')
    ax1.bar([i + width/2 for i in x], downwind_means, width, label='Downwind Attention', color='steelblue')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax1.set_xlabel('Wind Direction Bin')
    ax1.set_ylabel('Mean Attention Weight')
    ax1.set_title('Attention to Upwind vs Downwind by Wind Direction')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Right: Ratio
    ax2 = axes[1]
    
    colors = ['forestgreen' if r > 1 else 'steelblue' for r in ratios]
    bars = ax2.bar(x, ratios, color=colors)
    ax2.axhline(y=1.0, color='black', linestyle='--', lw=2, label='Equal attention')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax2.set_xlabel('Wind Direction Bin')
    ax2.set_ylabel('Upwind / Downwind Ratio')
    ax2.set_title('Wake Physics Alignment by Wind Direction\n(>1 = more attention to upwind)')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, ratios):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.2f}', ha='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_summary_dashboard(
    binned_results: Dict,
    head_analysis: Dict,
    rotation_analysis: Dict,
    positions: np.ndarray,
    rotor_diameter: float,
    save_path: str = None,
):
    """
    Create a summary dashboard with key findings.
    """
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    pos_norm = positions / rotor_diameter
    n_turbines = len(positions)
    n_heads = len(head_analysis)
    
    # Top row: Attention by wind direction (4 panels)
    bins = sorted(binned_results.keys())[:4]
    for idx, bin_idx in enumerate(bins):
        ax = fig.add_subplot(gs[0, idx])
        result = binned_results[bin_idx]
        attention = result['attention_avg']
        
        im = ax.imshow(attention, cmap='Reds', vmin=0)
        ax.set_title(f"{result['label'][:10]}\nRef: T{result['reference_turbine']}", fontsize=9)
        ax.set_xticks(range(n_turbines))
        ax.set_yticks(range(n_turbines))
        ax.set_xticklabels([f'{i}' for i in range(n_turbines)], fontsize=7)
        ax.set_yticklabels([f'{i}' for i in range(n_turbines)], fontsize=7)
    
    # Middle left: Per-head summary
    ax_heads = fig.add_subplot(gs[1, :2])
    head_names = [f'Head {i}' for i in range(n_heads)]
    focus_scores = [head_analysis[i]['focus_score'] for i in range(n_heads)]
    focus_turbines = [f"T{head_analysis[i]['focus_turbine']}" for i in range(n_heads)]
    
    bars = ax_heads.bar(head_names, focus_scores, color='steelblue')
    ax_heads.set_ylabel('Focus Score')
    ax_heads.set_title('Per-Head Focus (Higher = More Concentrated)')
    ax_heads.set_ylim(0, 1)
    
    for bar, ft in zip(bars, focus_turbines):
        ax_heads.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     ft, ha='center', fontsize=9)
    
    # Middle right: Upwind/Downwind ratio by wind direction
    ax_ratio = fig.add_subplot(gs[1, 2:])
    ratios = [binned_results[b]['ratio'] for b in bins]
    ratios = [min(r, 5) for r in ratios]
    labels = [binned_results[b]['label'][:8] for b in bins]
    
    colors = ['forestgreen' if r > 1 else 'crimson' for r in ratios]
    ax_ratio.bar(labels, ratios, color=colors)
    ax_ratio.axhline(y=1.0, color='black', linestyle='--', lw=2)
    ax_ratio.set_ylabel('Upwind/Downwind Ratio')
    ax_ratio.set_title('Wake Alignment by Wind Direction')
    ax_ratio.tick_params(axis='x', rotation=45)
    
    # Bottom left: Reference turbine rotation
    ax_rot = fig.add_subplot(gs[2, :2])
    
    wind_dirs = [d['wind_direction'] for d in rotation_analysis['details']]
    ref_turbines = [d['reference_turbine'] for d in rotation_analysis['details']]
    most_downwind = [d['most_downwind'] for d in rotation_analysis['details']]
    
    ax_rot.plot(wind_dirs, ref_turbines, 'o-', label='Reference', color='crimson', markersize=8)
    ax_rot.plot(wind_dirs, most_downwind, 's--', label='Most Downwind', color='steelblue', markersize=8)
    ax_rot.set_xlabel('Wind Direction (°)')
    ax_rot.set_ylabel('Turbine Index')
    ax_rot.set_title('Reference Turbine vs Most Downwind')
    ax_rot.legend()
    ax_rot.set_yticks(range(n_turbines))
    ax_rot.grid(True, alpha=0.3)
    
    # Bottom right: Key findings text
    ax_text = fig.add_subplot(gs[2, 2:])
    ax_text.axis('off')
    
    findings = []
    
    # Check if reference rotates
    if rotation_analysis['pct_matches_downwind'] > 60:
        findings.append("✓ Reference turbine TRACKS most downwind position")
    elif rotation_analysis['pct_matches_upwind'] > 60:
        findings.append("✓ Reference turbine TRACKS most upwind position")
    else:
        findings.append("✗ Reference turbine does NOT rotate with wind")
    
    # Check wake alignment
    avg_ratio = np.mean([binned_results[b]['ratio'] for b in bins])
    if avg_ratio > 1.2:
        findings.append(f"✓ Physics-aligned: avg ratio = {avg_ratio:.2f} (>1)")
    elif avg_ratio < 0.8:
        findings.append(f"✗ Reverse pattern: avg ratio = {avg_ratio:.2f} (<1)")
    else:
        findings.append(f"○ Neutral: avg ratio = {avg_ratio:.2f} (≈1)")
    
    # Head specialization
    focus_scores = [head_analysis[i]['focus_score'] for i in range(n_heads)]
    if max(focus_scores) > 0.5:
        findings.append(f"✓ Clear head specialization (max focus: {max(focus_scores):.0%})")
    else:
        findings.append("○ No clear head specialization")
    
    findings_text = "\n\n".join(findings)
    ax_text.text(0.1, 0.8, "Key Findings:", fontsize=14, fontweight='bold',
                transform=ax_text.transAxes, verticalalignment='top')
    ax_text.text(0.1, 0.65, findings_text, fontsize=11,
                transform=ax_text.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle('Wind Direction Conditioned Attention Analysis', fontsize=16, y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


# =============================================================================
# MAIN
# =============================================================================

def load_actor(checkpoint_path: str, device: torch.device) -> Tuple[TransformerActor, Dict]:
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


def main():
    parser = argparse.ArgumentParser(description='Wind direction conditioned attention analysis')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint')
    parser.add_argument('--run_dir', type=str, help='Path to run directory (uses latest checkpoint)')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--n_episodes', type=int, default=20, help='Episodes to collect')
    parser.add_argument('--n_steps', type=int, default=30, help='Steps per episode')
    parser.add_argument('--n_bins', type=int, default=4, help='Wind direction bins')
    parser.add_argument('--save_dir', type=str, default='attention_wind_analysis', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Find checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    elif args.run_dir:
        ckpts = glob.glob(os.path.join(args.run_dir, 'checkpoints', '*.pt'))
        if not ckpts:
            print(f"No checkpoints in {args.run_dir}")
            return
        checkpoint_path = max(ckpts, key=os.path.getmtime)
    else:
        print("Provide --checkpoint or --run_dir")
        return
    
    print(f"Loading: {checkpoint_path}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load model
    actor, model_args = load_actor(checkpoint_path, device)
    print(f"Loaded model with {actor.num_heads} attention heads")
    
    print("\nModel args:")
    for k, v in model_args.items():
        print(f"  {k}: {v}")
    # Create environment
    try:
        from WindGym import WindFarmEnv
        from WindGym.wrappers import PerTurbineObservationWrapper
        from WindGym.utils.generate_layouts import generate_square_grid
        from py_wake.examples.data.dtu10mw import DTU10MW
        
        turbine = DTU10MW()
        
        layout_str = model_args.get('layouts', 'square_2x2')
        if ',' in layout_str:
            layout_str = layout_str.split(',')[0].strip()
        
        print(f"Layout: {layout_str}")
        # layout_str = "square_2x2"  # Forcing square_2x2 for analysis
        if layout_str == 'test_layout':
            print("Using test layout (2 turbines)")
            x_pos, y_pos = generate_square_grid(turbine=turbine, nx=2, ny=1, xDist=5, yDist=5)
        elif layout_str == 'square_2x2':
            print("Using square 2x2 layout")
            x_pos, y_pos = generate_square_grid(turbine=turbine, nx=2, ny=2, xDist=5, yDist=5)
        else:
            print("Using default square 2x2 layout")
            x_pos, y_pos = generate_square_grid(turbine=turbine, nx=2, ny=2, xDist=5, yDist=5)
        
        # Use wide wind direction range for analysis
        config = {
            "yaw_init": "Random", "BaseController": "Local", "ActionMethod": "yaw",
            "Track_power": False, "farm": {"yaw_min": -30, "yaw_max": 30},
            "wind": {"ws_min": 10, "ws_max": 10, "TI_min": 0.07, "TI_max": 0.07, 
                    "wd_min": 180, "wd_max": 360},  # Full range for analysis
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
        
        print(f"Environment: {len(positions)} turbines")
        
    except ImportError as e:
        print(f"Could not import WindGym: {e}")
        return
    
    # Collect data
    print(f"\nCollecting attention data ({args.n_episodes} episodes, {args.n_steps} steps each)...")
    data = collect_attention_data(actor, env, device, args.n_episodes, args.n_steps, args.seed)
    
    print(f"Collected {len(data)} samples")
    
    # Analyze
    print("\nAnalyzing...")
    
    binned_results = analyze_by_wind_direction(data, positions, rotor_diameter, args.n_bins)
    head_analysis = analyze_per_head(data, positions, rotor_diameter)
    rotation_analysis = analyze_reference_turbine_rotation(binned_results, positions, rotor_diameter)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print(f"\nWind direction bins analyzed: {len(binned_results)}")
    for bin_idx, result in sorted(binned_results.items()):
        print(f"  {result['label']}: ratio={result['ratio']:.2f}, ref=T{result['reference_turbine']}")
    
    print(f"\nReference turbine rotation:")
    print(f"  Matches downwind: {rotation_analysis['pct_matches_downwind']:.0f}%")
    print(f"  Matches upwind: {rotation_analysis['pct_matches_upwind']:.0f}%")
    
    print(f"\nPer-head analysis:")
    for h, result in head_analysis.items():
        print(f"  Head {h}: focus=T{result['focus_turbine']} ({result['focus_score']:.0%}), "
              f"NN pref={result['nn_preference']:.1f}x")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    plot_attention_by_wind_direction(
        binned_results, positions, rotor_diameter,
        save_path=os.path.join(args.save_dir, 'attention_by_wind_direction.png')
    )
    plt.close()
    
    plot_per_head_analysis(
        head_analysis, positions, rotor_diameter,
        save_path=os.path.join(args.save_dir, 'per_head_analysis.png')
    )
    plt.close()
    
    plot_reference_turbine_analysis(
        binned_results, rotation_analysis,
        save_path=os.path.join(args.save_dir, 'reference_turbine_rotation.png')
    )
    plt.close()
    
    plot_upwind_downwind_by_direction(
        binned_results,
        save_path=os.path.join(args.save_dir, 'upwind_downwind_by_direction.png')
    )
    plt.close()
    
    plot_summary_dashboard(
        binned_results, head_analysis, rotation_analysis, positions, rotor_diameter,
        save_path=os.path.join(args.save_dir, 'summary_dashboard.png')
    )
    plt.close()
    
    # Save metrics
    metrics = {
        'n_samples': len(data),
        'n_bins': len(binned_results),
        'binned_results': {
            str(k): {
                'label': v['label'],
                'mean_wind_direction': float(v['mean_wind_direction']),
                'n_samples': v['n_samples'],
                'upwind_mean': float(v['upwind_mean']),
                'downwind_mean': float(v['downwind_mean']),
                'ratio': float(v['ratio']),
                'reference_turbine': int(v['reference_turbine']),
                'most_downwind': int(v['most_downwind']),
            }
            for k, v in binned_results.items()
        },
        'head_analysis': {
            str(k): {
                'focus_turbine': int(v['focus_turbine']),
                'focus_score': float(v['focus_score']),
                'nn_preference': float(v['nn_preference']),
                'uniformity': float(v['uniformity']),
            }
            for k, v in head_analysis.items()
        },
        'rotation_analysis': {
            'pct_matches_downwind': float(rotation_analysis['pct_matches_downwind']),
            'pct_matches_upwind': float(rotation_analysis['pct_matches_upwind']),
        },
    }
    
    with open(os.path.join(args.save_dir, 'analysis_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✓ All outputs saved to {args.save_dir}/")
    
    env.close()


if __name__ == '__main__':
    main()