#!/usr/bin/env python3
"""
Attention Training Analysis

Comprehensive analysis of transformer attention patterns during training.
Tracks how attention and reward evolve as a function of:
1. Training progress (across checkpoints)
2. Wind direction (binned analysis)

Usage:
    # Analyze all checkpoints in a run
    python attention_training_analysis.py --run_dir runs/YOUR_RUN
    
    # Analyze specific checkpoints
    python attention_training_analysis.py --run_dir runs/YOUR_RUN --checkpoints 10000 50000 100000
    
    # Single checkpoint analysis
    python attention_training_analysis.py --checkpoint path/to/checkpoint.pt

Author: Marcus (DTU Wind Energy)
"""

import os
import re
import json
import glob
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
from dataclasses import dataclass, field


# =============================================================================
# MODEL DEFINITION
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
            nn.Linear(hidden_dim, embed_dim), nn.Dropout(dropout))
    
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
            for _ in range(num_layers)])
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
                 use_farm_token: bool = False):
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
        self.register_buffer("action_scale", torch.tensor(1.0))
        self.register_buffer("action_bias", torch.tensor(0.0))
    
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
                    key_padding_mask], dim=1)
        
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
        x_t = mean if deterministic else torch.distributions.Normal(mean, std).rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, None, mean_action, attn_weights


def transform_to_wind_relative(positions: torch.Tensor, wind_direction: float) -> torch.Tensor:
    """Rotate positions so wind appears from 270° (West)."""
    theta = (wind_direction - 270.0) * (math.pi / 180.0)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    x, y = positions[..., 0:1], positions[..., 1:2]
    return torch.cat([cos_t * x - sin_t * y, sin_t * x + cos_t * y], dim=-1)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class CheckpointMetrics:
    """Metrics for a single checkpoint."""
    step: int
    eval_reward_mean: float
    eval_reward_std: float
    attention_uniformity: float
    wake_ratio_mean: float
    focus_score_max: float
    reference_tracks_wind: bool
    binned_results: Dict = field(default_factory=dict)
    head_analysis: Dict = field(default_factory=dict)
    episode_rewards: List[float] = field(default_factory=list)
    episode_wind_dirs: List[float] = field(default_factory=list)


# =============================================================================
# UTILITIES
# =============================================================================

def get_upwind_downwind(positions: np.ndarray, wind_direction: float, rotor_diameter: float) -> Dict:
    """Classify turbines as upwind/downwind based on wind direction."""
    pos_norm = positions / rotor_diameter
    wind_rad = np.radians(270 - wind_direction)
    wind_vec = np.array([np.cos(wind_rad), np.sin(wind_rad)])
    wind_axis = np.dot(pos_norm, wind_vec)
    median = np.median(wind_axis)
    
    return {
        'upwind_indices': np.where(wind_axis >= median)[0].tolist(),
        'downwind_indices': np.where(wind_axis < median)[0].tolist(),
        'wind_axis': wind_axis,
        'most_upwind': int(np.argmax(wind_axis)),
        'most_downwind': int(np.argmin(wind_axis)),
    }


def get_wind_bin(wd: float, n_bins: int = 4) -> int:
    return int((wd % 360) // (360 / n_bins))


def get_bin_label(bin_idx: int, n_bins: int = 4) -> str:
    bin_size = 360 / n_bins
    start, end = bin_idx * bin_size, (bin_idx + 1) * bin_size
    dirs = {0: 'N', 45: 'NE', 90: 'E', 135: 'SE', 180: 'S', 225: 'SW', 270: 'W', 315: 'NW'}
    mid = (start + end) / 2
    closest = min(dirs.keys(), key=lambda x: min(abs(x - mid), 360 - abs(x - mid)))
    return f"{start:.0f}°-{end:.0f}° ({dirs[closest]})"


def extract_step_from_path(path: str) -> int:
    """Extract step number from checkpoint filename."""
    basename = os.path.basename(path)
    match = re.search(r'(\d+)', basename)
    return int(match.group(1)) if match else 0


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_actor(checkpoint_path: str, device: torch.device) -> Tuple[TransformerActor, Dict, int]:
    """Load actor from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = checkpoint.get('args', {})
    actor_state = checkpoint['actor_state_dict']
    
    step = checkpoint.get('global_step', 0)
    if step == 0:
        step = extract_step_from_path(checkpoint_path)
    
    obs_dim = actor_state['obs_encoder.0.weight'].shape[1]
    
    actor = TransformerActor(
        obs_dim_per_turbine=obs_dim,
        embed_dim=args.get('embed_dim', 128),
        pos_embed_dim=args.get('pos_embed_dim', 32),
        num_heads=args.get('num_heads', 4),
        num_layers=args.get('num_layers', 2),
        mlp_ratio=args.get('mlp_ratio', 2.0),
        use_farm_token=args.get('use_farm_token', False),
    ).to(device)
    
    actor.load_state_dict(actor_state)
    actor.eval()
    return actor, args, step


def find_checkpoints(run_dir: str, specific_steps: List[int] = None) -> List[str]:
    """Find checkpoint files in run directory."""
    patterns = [
        os.path.join(run_dir, 'checkpoints', '*.pt'),
        os.path.join(run_dir, '*.pt'),
    ]
    
    all_ckpts = []
    for pattern in patterns:
        all_ckpts.extend(glob.glob(pattern))
    
    # Remove duplicates and sort by step
    ckpt_dict = {}
    for path in all_ckpts:
        step = extract_step_from_path(path)
        if step not in ckpt_dict:
            ckpt_dict[step] = path
    
    if specific_steps:
        # Filter to specific steps (find closest if exact not available)
        result = []
        for target in specific_steps:
            closest = min(ckpt_dict.keys(), key=lambda x: abs(x - target))
            if ckpt_dict[closest] not in result:
                result.append(ckpt_dict[closest])
        return result
    
    return [ckpt_dict[k] for k in sorted(ckpt_dict.keys())]


# =============================================================================
# DATA COLLECTION
# =============================================================================

def collect_episode_data(actor, env, device, n_episodes=20, n_steps=30, seed=42):
    """Collect attention and reward data across episodes."""
    positions = env.turbine_positions
    rotor_diameter = env.rotor_diameter
    base_env = env.env if hasattr(env, 'env') else env
    
    # Set action scaling
    action_high = base_env.action_space.high[0]
    action_low = base_env.action_space.low[0]
    actor.action_scale = torch.tensor((action_high - action_low) / 2.0, device=device)
    actor.action_bias = torch.tensor((action_high + action_low) / 2.0, device=device)
    
    all_data = []
    episode_metrics = {'episode': [], 'wind_direction': [], 'total_reward': []}
    
    for ep in tqdm(range(n_episodes), desc="Episodes", leave=False):
        obs, _ = env.reset(seed=seed + ep)
        wind_dir = getattr(base_env, 'wd', 270.0)
        ep_reward = 0.0
        
        for step in range(n_steps):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            pos_norm = positions / rotor_diameter
            pos_t = torch.tensor(pos_norm, dtype=torch.float32, device=device).unsqueeze(0)
            pos_transformed = transform_to_wind_relative(pos_t, wind_dir)
            
            with torch.no_grad():
                action, _, _, attn_weights = actor.get_action(obs_t, pos_transformed, deterministic=True)
            
            attn_last = attn_weights[-1][0].cpu().numpy()
            action_np = action[0].cpu().numpy().flatten()
            obs, reward, term, trunc, _ = env.step(action_np)
            
            ep_reward += reward
            all_data.append({
                'wind_direction': wind_dir,
                'attention_per_head': attn_last,
                'attention_avg': attn_last.mean(axis=0),
                'reward': reward,
            })
            
            if term or trunc:
                break
        
        episode_metrics['episode'].append(ep)
        episode_metrics['wind_direction'].append(wind_dir)
        episode_metrics['total_reward'].append(ep_reward)
    
    return all_data, episode_metrics


# =============================================================================
# ANALYSIS
# =============================================================================

def analyze_by_wind_direction(data: List[Dict], positions: np.ndarray, 
                               rotor_diameter: float, n_bins: int = 4) -> Dict:
    """Analyze attention patterns binned by wind direction."""
    n_turbines = len(positions)
    binned = defaultdict(list)
    
    for d in data:
        binned[get_wind_bin(d['wind_direction'], n_bins)].append(d)
    
    results = {}
    for bin_idx in range(n_bins):
        if bin_idx not in binned or not binned[bin_idx]:
            continue
        
        bin_data = binned[bin_idx]
        attention_avg = np.mean([d['attention_avg'] for d in bin_data], axis=0)
        mean_wd = np.mean([d['wind_direction'] for d in bin_data])
        mean_reward = np.mean([d['reward'] for d in bin_data])
        
        ud = get_upwind_downwind(positions, mean_wd, rotor_diameter)
        
        # Calculate upwind/downwind attention
        upwind_attn, downwind_attn = [], []
        for i in range(n_turbines):
            for j in range(n_turbines):
                if i == j:
                    continue
                if ud['wind_axis'][j] > ud['wind_axis'][i]:
                    upwind_attn.append(attention_avg[i, j])
                else:
                    downwind_attn.append(attention_avg[i, j])
        
        up_mean = np.mean(upwind_attn) if upwind_attn else 0
        down_mean = np.mean(downwind_attn) if downwind_attn else 0
        
        results[bin_idx] = {
            'label': get_bin_label(bin_idx, n_bins),
            'mean_wd': mean_wd,
            'n_samples': len(bin_data),
            'mean_reward': mean_reward,
            'attention_avg': attention_avg,
            'upwind_mean': up_mean,
            'downwind_mean': down_mean,
            'ratio': up_mean / down_mean if down_mean > 0 else float('inf'),
            'reference_turbine': int(np.argmax(attention_avg.sum(axis=0))),
            'most_upwind': ud['most_upwind'],
            'most_downwind': ud['most_downwind'],
            'upwind_indices': ud['upwind_indices'],
            'downwind_indices': ud['downwind_indices'],
        }
    
    return results


def analyze_per_head(data: List[Dict], positions: np.ndarray, rotor_diameter: float) -> Dict:
    """Analyze per-head attention patterns."""
    n_turbines = len(positions)
    n_heads = data[0]['attention_per_head'].shape[0]
    pos_norm = positions / rotor_diameter
    
    results = {}
    for h in range(n_heads):
        attn = np.mean([d['attention_per_head'][h] for d in data], axis=0)
        
        col_sums = attn.sum(axis=0)
        focus_turb = int(np.argmax(col_sums))
        focus_score = col_sums[focus_turb] / col_sums.sum()
        
        # Nearest neighbor preference
        nn_attn, far_attn, n_nn, n_far = 0, 0, 0, 0
        for i in range(n_turbines):
            for j in range(n_turbines):
                if i == j:
                    continue
                dist = np.linalg.norm(pos_norm[i] - pos_norm[j])
                if dist < 6:
                    nn_attn += attn[i, j]
                    n_nn += 1
                else:
                    far_attn += attn[i, j]
                    n_far += 1
        
        nn_mean = nn_attn / n_nn if n_nn > 0 else 0
        far_mean = far_attn / n_far if n_far > 0 else 0
        
        # Uniformity (entropy-based)
        flat = attn.flatten()
        flat = flat / flat.sum()
        entropy = -np.sum(flat * np.log(flat + 1e-10))
        uniformity = entropy / np.log(len(flat))
        
        results[h] = {
            'attention_matrix': attn,
            'focus_turbine': focus_turb,
            'focus_score': focus_score,
            'nn_preference': nn_mean / far_mean if far_mean > 0 else float('inf'),
            'uniformity': uniformity,
        }
    
    return results


def analyze_checkpoint(actor, env, device, n_episodes=20, n_steps=30, 
                       n_bins=4, seed=42, step=0) -> CheckpointMetrics:
    """Complete analysis for a single checkpoint."""
    positions = env.turbine_positions
    rotor_diameter = env.rotor_diameter
    
    # Collect data
    data, ep_metrics = collect_episode_data(actor, env, device, n_episodes, n_steps, seed)
    
    # Analyze
    binned = analyze_by_wind_direction(data, positions, rotor_diameter, n_bins)
    heads = analyze_per_head(data, positions, rotor_diameter)
    
    # Compute summary metrics
    uniformities = [heads[h]['uniformity'] for h in heads]
    avg_uniformity = np.mean(uniformities)
    
    ratios = [binned[b]['ratio'] for b in binned if binned[b]['ratio'] < float('inf')]
    avg_ratio = np.mean(ratios) if ratios else 1.0
    
    focus_scores = [heads[h]['focus_score'] for h in heads]
    
    # Check if reference tracks wind
    refs = [binned[b]['reference_turbine'] for b in binned]
    downwinds = [binned[b]['most_downwind'] for b in binned]
    upwinds = [binned[b]['most_upwind'] for b in binned]
    
    matches_down = sum(r == d for r, d in zip(refs, downwinds)) / len(refs) if refs else 0
    matches_up = sum(r == u for r, u in zip(refs, upwinds)) / len(refs) if refs else 0
    tracks_wind = matches_down > 0.5 or matches_up > 0.5
    
    return CheckpointMetrics(
        step=step,
        eval_reward_mean=np.mean(ep_metrics['total_reward']),
        eval_reward_std=np.std(ep_metrics['total_reward']),
        attention_uniformity=avg_uniformity,
        wake_ratio_mean=avg_ratio,
        focus_score_max=max(focus_scores),
        reference_tracks_wind=tracks_wind,
        binned_results=binned,
        head_analysis=heads,
        episode_rewards=ep_metrics['total_reward'],
        episode_wind_dirs=ep_metrics['wind_direction'],
    )


# =============================================================================
# VISUALIZATION - SINGLE CHECKPOINT
# =============================================================================

def plot_checkpoint_summary(metrics: CheckpointMetrics, positions: np.ndarray, 
                            rotor_diameter: float, save_path: str = None):
    """Create summary dashboard for a single checkpoint."""
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.3)
    
    pos_norm = positions / rotor_diameter
    n_turbines = len(positions)
    bins = sorted(metrics.binned_results.keys())[:4]
    n_heads = len(metrics.head_analysis)
    
    # Row 0: Episode rewards + by wind direction
    ax0 = fig.add_subplot(gs[0, :2])
    rewards = metrics.episode_rewards
    ax0.plot(range(len(rewards)), rewards, 'o-', color='steelblue', markersize=4, alpha=0.7)
    ax0.axhline(metrics.eval_reward_mean, color='crimson', ls='--', lw=2, 
                label=f'Mean: {metrics.eval_reward_mean:.1f}')
    ax0.fill_between(range(len(rewards)), 
                     metrics.eval_reward_mean - metrics.eval_reward_std,
                     metrics.eval_reward_mean + metrics.eval_reward_std, alpha=0.2, color='crimson')
    ax0.set_xlabel('Episode')
    ax0.set_ylabel('Reward')
    ax0.set_title(f'Evaluation (step {metrics.step:,})')
    ax0.legend()
    ax0.grid(True, alpha=0.3)
    
    ax1 = fig.add_subplot(gs[0, 2:])
    wd_rewards = defaultdict(list)
    for wd, r in zip(metrics.episode_wind_dirs, metrics.episode_rewards):
        wd_rewards[get_wind_bin(wd, 4)].append(r)
    
    labels = [get_bin_label(b, 4)[:8] for b in sorted(wd_rewards.keys())]
    means = [np.mean(wd_rewards[b]) for b in sorted(wd_rewards.keys())]
    stds = [np.std(wd_rewards[b]) for b in sorted(wd_rewards.keys())]
    
    ax1.bar(range(len(labels)), means, yerr=stds, capsize=5, color='forestgreen', alpha=0.7)
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.set_ylabel('Reward')
    ax1.set_title('Reward by Wind Direction')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Row 1: Attention heatmaps by wind direction
    for idx, bin_idx in enumerate(bins):
        ax = fig.add_subplot(gs[1, idx])
        result = metrics.binned_results[bin_idx]
        attn = result['attention_avg']
        
        ax.imshow(attn, cmap='Reds', vmin=0)
        ax.set_title(f"{result['label'][:12]}\nRef: T{result['reference_turbine']}", fontsize=9)
        ax.set_xticks(range(n_turbines))
        ax.set_yticks(range(n_turbines))
        ax.set_xticklabels([f'{i}' for i in range(n_turbines)], fontsize=7)
        ax.set_yticklabels([f'{i}' for i in range(n_turbines)], fontsize=7)
        
        if n_turbines <= 4:
            for i in range(n_turbines):
                for j in range(n_turbines):
                    ax.text(j, i, f'{attn[i,j]:.2f}', ha='center', va='center', fontsize=6)
    
    # Row 2: Per-head focus + wake ratio
    ax_heads = fig.add_subplot(gs[2, :2])
    focus = [metrics.head_analysis[h]['focus_score'] for h in range(n_heads)]
    bars = ax_heads.bar([f'Head {h}' for h in range(n_heads)], focus, color='steelblue')
    ax_heads.axhline(1/n_turbines, color='gray', ls='--', label=f'Uniform ({1/n_turbines:.2f})')
    ax_heads.set_ylabel('Focus Score')
    ax_heads.set_title('Per-Head Focus')
    ax_heads.legend()
    
    for bar, h in zip(bars, range(n_heads)):
        ax_heads.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'T{metrics.head_analysis[h]["focus_turbine"]}', ha='center', fontsize=8)
    
    ax_ratio = fig.add_subplot(gs[2, 2:])
    ratios = [min(metrics.binned_results[b]['ratio'], 3) for b in bins]
    colors = ['forestgreen' if r > 1.1 else ('crimson' if r < 0.9 else 'gray') for r in ratios]
    ax_ratio.bar([metrics.binned_results[b]['label'][:8] for b in bins], ratios, color=colors)
    ax_ratio.axhline(1.0, color='black', ls='--', lw=2)
    ax_ratio.set_ylabel('Upwind/Downwind Ratio')
    ax_ratio.set_title('Wake Alignment (>1 = attends upwind)')
    ax_ratio.tick_params(axis='x', rotation=45)
    
    # Row 3: Reference tracking + findings
    ax_ref = fig.add_subplot(gs[3, :2])
    details = [(metrics.binned_results[b]['mean_wd'], 
                metrics.binned_results[b]['reference_turbine'],
                metrics.binned_results[b]['most_downwind'],
                metrics.binned_results[b]['most_upwind']) for b in bins]
    
    x = range(len(details))
    width = 0.25
    ax_ref.bar([i-width for i in x], [d[1] for d in details], width, label='Reference', color='crimson')
    ax_ref.bar([i for i in x], [d[2] for d in details], width, label='Downwind', color='steelblue')
    ax_ref.bar([i+width for i in x], [d[3] for d in details], width, label='Upwind', color='forestgreen')
    ax_ref.set_xticks(x)
    ax_ref.set_xticklabels([f'{d[0]:.0f}°' for d in details], rotation=45)
    ax_ref.set_ylabel('Turbine Index')
    ax_ref.set_title('Reference vs Upwind/Downwind')
    ax_ref.legend()
    ax_ref.set_yticks(range(n_turbines))
    
    ax_find = fig.add_subplot(gs[3, 2:])
    ax_find.axis('off')
    
    findings = [f"📊 Reward: {metrics.eval_reward_mean:.1f} ± {metrics.eval_reward_std:.1f}"]
    
    if metrics.attention_uniformity > 0.98:
        findings.append(f"❌ Attention uniform ({metrics.attention_uniformity:.3f})")
    elif metrics.attention_uniformity > 0.95:
        findings.append(f"⚠️ Mostly uniform ({metrics.attention_uniformity:.3f})")
    else:
        findings.append(f"✅ Specialized ({metrics.attention_uniformity:.3f})")
    
    if metrics.reference_tracks_wind:
        findings.append("✅ Reference tracks wind")
    else:
        findings.append("⚪ Reference doesn't track")
    
    if metrics.wake_ratio_mean > 1.2:
        findings.append(f"✅ Physics-aligned (ratio={metrics.wake_ratio_mean:.2f})")
    elif metrics.wake_ratio_mean < 0.8:
        findings.append(f"⚪ Reverse (ratio={metrics.wake_ratio_mean:.2f})")
    else:
        findings.append(f"⚪ Neutral (ratio={metrics.wake_ratio_mean:.2f})")
    
    ax_find.text(0.05, 0.95, "Key Findings", fontsize=14, fontweight='bold',
                transform=ax_find.transAxes, va='top')
    ax_find.text(0.05, 0.80, "\n\n".join(findings), fontsize=11,
                transform=ax_find.transAxes, va='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle(f'Attention Analysis - Step {metrics.step:,}', fontsize=16, y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    return fig


# =============================================================================
# VISUALIZATION - TRAINING EVOLUTION
# =============================================================================

def plot_training_evolution(all_metrics: List[CheckpointMetrics], save_path: str = None):
    """Plot how metrics evolve across training checkpoints."""
    steps = [m.step for m in all_metrics]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Reward evolution
    ax = axes[0, 0]
    means = [m.eval_reward_mean for m in all_metrics]
    stds = [m.eval_reward_std for m in all_metrics]
    ax.plot(steps, means, 'o-', color='steelblue', lw=2, markersize=6)
    ax.fill_between(steps, np.array(means) - np.array(stds), 
                    np.array(means) + np.array(stds), alpha=0.3)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Episode Reward')
    ax.set_title('Reward Evolution')
    ax.grid(True, alpha=0.3)
    
    # Attention uniformity
    ax = axes[0, 1]
    uniformities = [m.attention_uniformity for m in all_metrics]
    ax.plot(steps, uniformities, 'o-', color='crimson', lw=2, markersize=6)
    ax.axhline(0.98, color='red', ls=':', alpha=0.5, label='Uniform threshold')
    ax.axhline(0.95, color='orange', ls=':', alpha=0.5, label='Semi-uniform')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Uniformity')
    ax.set_title('Attention Uniformity\n(Lower = More Specialized)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.8, 1.02)
    
    # Wake ratio
    ax = axes[0, 2]
    ratios = [m.wake_ratio_mean for m in all_metrics]
    colors = ['forestgreen' if r > 1.1 else ('crimson' if r < 0.9 else 'gray') for r in ratios]
    ax.scatter(steps, ratios, c=colors, s=80, zorder=5)
    ax.plot(steps, ratios, '-', color='gray', alpha=0.5, zorder=1)
    ax.axhline(1.0, color='black', ls='--', lw=2, alpha=0.7)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Upwind/Downwind Ratio')
    ax.set_title('Wake Physics Alignment\n(>1 = Attends Upwind)')
    ax.grid(True, alpha=0.3)
    
    # Focus score evolution
    ax = axes[1, 0]
    focus = [m.focus_score_max for m in all_metrics]
    ax.plot(steps, focus, 'o-', color='purple', lw=2, markersize=6)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Max Focus Score')
    ax.set_title('Head Specialization')
    ax.grid(True, alpha=0.3)
    
    # Reference tracks wind
    ax = axes[1, 1]
    tracks = [1 if m.reference_tracks_wind else 0 for m in all_metrics]
    ax.plot(steps, tracks, 'o-', color='teal', lw=2, markersize=8)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Tracks Wind (1=Yes)')
    ax.set_title('Reference Turbine Wind Tracking')
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['No', 'Yes'])
    ax.grid(True, alpha=0.3)
    
    # Combined summary
    ax = axes[1, 2]
    ax.axis('off')
    
    # Find best checkpoint
    best_idx = np.argmax([m.eval_reward_mean for m in all_metrics])
    best = all_metrics[best_idx]
    
    summary = [
        f"Training Progress Summary",
        f"─" * 30,
        f"Checkpoints analyzed: {len(all_metrics)}",
        f"Steps: {steps[0]:,} → {steps[-1]:,}",
        f"",
        f"Best checkpoint: step {best.step:,}",
        f"  Reward: {best.eval_reward_mean:.1f} ± {best.eval_reward_std:.1f}",
        f"  Uniformity: {best.attention_uniformity:.3f}",
        f"  Wake ratio: {best.wake_ratio_mean:.2f}",
        f"",
        f"Latest checkpoint: step {all_metrics[-1].step:,}",
        f"  Reward: {all_metrics[-1].eval_reward_mean:.1f}",
        f"  Uniformity: {all_metrics[-1].attention_uniformity:.3f}",
    ]
    
    ax.text(0.1, 0.95, "\n".join(summary), fontsize=10, family='monospace',
           transform=ax.transAxes, va='top',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle('Training Evolution Analysis', fontsize=16, y=0.98)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_attention_evolution_by_wind(all_metrics: List[CheckpointMetrics], 
                                      positions: np.ndarray, rotor_diameter: float,
                                      save_path: str = None):
    """Plot attention heatmaps across training for each wind direction."""
    n_ckpts = len(all_metrics)
    bins = sorted(all_metrics[0].binned_results.keys())[:4]
    n_bins = len(bins)
    
    fig, axes = plt.subplots(n_bins, n_ckpts, figsize=(4*n_ckpts, 4*n_bins))
    
    if n_ckpts == 1:
        axes = axes.reshape(n_bins, 1)
    
    n_turbines = len(positions)
    
    for col, m in enumerate(all_metrics):
        for row, bin_idx in enumerate(bins):
            ax = axes[row, col]
            
            if bin_idx in m.binned_results:
                result = m.binned_results[bin_idx]
                attn = result['attention_avg']
                
                ax.imshow(attn, cmap='Reds', vmin=0, vmax=0.5)
                
                if n_turbines <= 4:
                    for i in range(n_turbines):
                        for j in range(n_turbines):
                            ax.text(j, i, f'{attn[i,j]:.2f}', ha='center', va='center', 
                                   fontsize=7, color='black' if attn[i,j] < 0.3 else 'white')
                
                ax.set_xticks(range(n_turbines))
                ax.set_yticks(range(n_turbines))
                ax.set_xticklabels([f'{i}' for i in range(n_turbines)], fontsize=7)
                ax.set_yticklabels([f'{i}' for i in range(n_turbines)], fontsize=7)
            
            if col == 0:
                label = get_bin_label(bin_idx, 4)[:12]
                ax.set_ylabel(label, fontsize=10)
            
            if row == 0:
                ax.set_title(f'Step {m.step:,}\nR={m.eval_reward_mean:.1f}', fontsize=9)
    
    plt.suptitle('Attention Evolution by Wind Direction', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_wind_direction_evolution(all_metrics: List[CheckpointMetrics], save_path: str = None):
    """Plot how wind-direction-specific metrics evolve."""
    steps = [m.step for m in all_metrics]
    bins = sorted(all_metrics[0].binned_results.keys())[:4]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Wake ratio by wind direction over training
    ax = axes[0, 0]
    for bin_idx in bins:
        ratios = []
        for m in all_metrics:
            if bin_idx in m.binned_results:
                r = m.binned_results[bin_idx]['ratio']
                ratios.append(min(r, 3))
            else:
                ratios.append(np.nan)
        
        label = get_bin_label(bin_idx, 4)[:8]
        ax.plot(steps, ratios, 'o-', label=label, markersize=5)
    
    ax.axhline(1.0, color='black', ls='--', lw=2, alpha=0.5)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Upwind/Downwind Ratio')
    ax.set_title('Wake Ratio by Wind Direction')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Reward by wind direction over training
    ax = axes[0, 1]
    for bin_idx in bins:
        rewards = []
        for m in all_metrics:
            if bin_idx in m.binned_results:
                rewards.append(m.binned_results[bin_idx]['mean_reward'])
            else:
                rewards.append(np.nan)
        
        label = get_bin_label(bin_idx, 4)[:8]
        ax.plot(steps, rewards, 'o-', label=label, markersize=5)
    
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Mean Reward')
    ax.set_title('Reward by Wind Direction')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Reference turbine by wind direction
    ax = axes[1, 0]
    for bin_idx in bins:
        refs = []
        for m in all_metrics:
            if bin_idx in m.binned_results:
                refs.append(m.binned_results[bin_idx]['reference_turbine'])
            else:
                refs.append(np.nan)
        
        label = get_bin_label(bin_idx, 4)[:8]
        ax.plot(steps, refs, 'o-', label=label, markersize=6)
    
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Reference Turbine')
    ax.set_title('Reference Turbine by Wind Direction')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Episode reward distribution at different stages
    ax = axes[1, 1]
    
    indices = [0, len(all_metrics)//2, -1]
    positions_bp = []
    data_bp = []
    labels_bp = []
    
    for i, idx in enumerate(indices):
        m = all_metrics[idx]
        positions_bp.append(i)
        data_bp.append(m.episode_rewards)
        labels_bp.append(f'Step {m.step:,}')
    
    bp = ax.boxplot(data_bp, positions=positions_bp, widths=0.6)
    ax.set_xticks(positions_bp)
    ax.set_xticklabels(labels_bp)
    ax.set_ylabel('Episode Reward')
    ax.set_title('Reward Distribution Over Training')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Wind Direction Metrics Evolution', fontsize=14, y=0.98)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


# =============================================================================
# ENVIRONMENT CREATION
# =============================================================================

def create_environment(model_args: Dict, seed: int = 42):
    """Create WindFarm environment based on model args."""
    from WindGym import WindFarmEnv
    from WindGym.wrappers import PerTurbineObservationWrapper
    from WindGym.utils.generate_layouts import generate_square_grid
    from py_wake.examples.data.dtu10mw import DTU10MW
    
    turbine = DTU10MW()
    layout = model_args.get('layouts', 'square_2x2').split(',')[0].strip()
    
    if layout == 'test_layout':
        x_pos, y_pos = generate_square_grid(turbine=turbine, nx=2, ny=1, xDist=5, yDist=5)
    elif layout == 'square_2x2':
        x_pos, y_pos = generate_square_grid(turbine=turbine, nx=2, ny=2, xDist=5, yDist=5)
    elif layout == 'square_3x3':
        x_pos, y_pos = generate_square_grid(turbine=turbine, nx=3, ny=3, xDist=5, yDist=5)
    else:
        x_pos, y_pos = generate_square_grid(turbine=turbine, nx=2, ny=2, xDist=5, yDist=5)
    
    config = {
        "yaw_init": "Random", "BaseController": "Local", "ActionMethod": "yaw",
        "Track_power": False, "farm": {"yaw_min": -30, "yaw_max": 30},
        "wind": {"ws_min": 9, "ws_max": 9, "TI_min": 0.05, "TI_max": 0.05, 
                "wd_min": 180, "wd_max": 360},
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
    
    base_env = WindFarmEnv(turbine=turbine, x_pos=x_pos, y_pos=y_pos, config=config,
                           dt_sim=5, dt_env=10, n_passthrough=20, seed=seed)
    return PerTurbineObservationWrapper(base_env), layout


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Attention Training Analysis')
    parser.add_argument('--run_dir', type=str, help='Run directory with checkpoints')
    parser.add_argument('--checkpoint', type=str, help='Single checkpoint to analyze')
    parser.add_argument('--checkpoints', type=int, nargs='+', help='Specific checkpoint steps to analyze')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n_episodes', type=int, default=20)
    parser.add_argument('--n_steps', type=int, default=30)
    parser.add_argument('--n_bins', type=int, default=4)
    parser.add_argument('--save_dir', type=str, default='attention_analysis')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_checkpoints', type=int, default=10, help='Max checkpoints for evolution analysis')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Find checkpoints
    if args.checkpoint:
        ckpt_paths = [args.checkpoint]
    elif args.run_dir:
        ckpt_paths = find_checkpoints(args.run_dir, args.checkpoints)
        if not ckpt_paths:
            print(f"No checkpoints found in {args.run_dir}")
            return
        # Limit for evolution analysis
        if len(ckpt_paths) > args.max_checkpoints:
            indices = np.linspace(0, len(ckpt_paths)-1, args.max_checkpoints, dtype=int)
            ckpt_paths = [ckpt_paths[i] for i in indices]
    else:
        print("Provide --run_dir or --checkpoint")
        return
    
    print(f"Analyzing {len(ckpt_paths)} checkpoint(s)")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load first checkpoint to get model args
    _, model_args, _ = load_actor(ckpt_paths[0], device)
    
    # Create environment
    try:
        env, layout = create_environment(model_args, args.seed)
        positions = env.turbine_positions
        rotor_diameter = env.rotor_diameter
        print(f"Layout: {layout}, {len(positions)} turbines")
    except ImportError as e:
        print(f"WindGym error: {e}")
        return
    
    # Analyze each checkpoint
    all_metrics = []
    
    for ckpt_path in tqdm(ckpt_paths, desc="Checkpoints"):
        actor, _, step = load_actor(ckpt_path, device)
        
        print(f"\nStep {step:,}:")
        metrics = analyze_checkpoint(actor, env, device, args.n_episodes, 
                                     args.n_steps, args.n_bins, args.seed, step)
        all_metrics.append(metrics)
        
        print(f"  Reward: {metrics.eval_reward_mean:.1f} ± {metrics.eval_reward_std:.1f}")
        print(f"  Uniformity: {metrics.attention_uniformity:.3f}")
        print(f"  Wake ratio: {metrics.wake_ratio_mean:.2f}")
        
        # Save individual checkpoint summary
        if len(ckpt_paths) <= 5:  # Only for small number of checkpoints
            plot_checkpoint_summary(metrics, positions, rotor_diameter,
                                   save_path=os.path.join(args.save_dir, f'checkpoint_{step}.png'))
            plt.close()
    
    # Generate evolution plots if multiple checkpoints
    if len(all_metrics) > 1:
        print("\nGenerating evolution plots...")
        
        plot_training_evolution(all_metrics, 
                               save_path=os.path.join(args.save_dir, 'training_evolution.png'))
        plt.close()
        
        plot_attention_evolution_by_wind(all_metrics, positions, rotor_diameter,
                                        save_path=os.path.join(args.save_dir, 'attention_evolution.png'))
        plt.close()
        
        plot_wind_direction_evolution(all_metrics,
                                     save_path=os.path.join(args.save_dir, 'wind_direction_evolution.png'))
        plt.close()
    
    # Always generate final checkpoint summary
    plot_checkpoint_summary(all_metrics[-1], positions, rotor_diameter,
                           save_path=os.path.join(args.save_dir, 'final_checkpoint.png'))
    plt.close()
    
    # Save metrics JSON
    metrics_dict = {
        'checkpoints': [{
            'step': m.step,
            'eval_reward_mean': float(m.eval_reward_mean),
            'eval_reward_std': float(m.eval_reward_std),
            'attention_uniformity': float(m.attention_uniformity),
            'wake_ratio_mean': float(m.wake_ratio_mean),
            'focus_score_max': float(m.focus_score_max),
            'reference_tracks_wind': bool(m.reference_tracks_wind),
            'episode_rewards': [float(r) for r in m.episode_rewards],
            'episode_wind_dirs': [float(w) for w in m.episode_wind_dirs],
            'binned_results': {
                str(k): {
                    'label': v['label'],
                    'mean_wd': float(v['mean_wd']),
                    'n_samples': v['n_samples'],
                    'mean_reward': float(v['mean_reward']),
                    'upwind_mean': float(v['upwind_mean']),
                    'downwind_mean': float(v['downwind_mean']),
                    'ratio': float(min(v['ratio'], 100)),
                    'reference_turbine': int(v['reference_turbine']),
                }
                for k, v in m.binned_results.items()
            }
        } for m in all_metrics]
    }
    
    with open(os.path.join(args.save_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    
    print(f"\n✓ All outputs saved to {args.save_dir}/")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if len(all_metrics) > 1:
        print(f"\nTraining progress: {all_metrics[0].step:,} → {all_metrics[-1].step:,}")
        print(f"Reward: {all_metrics[0].eval_reward_mean:.1f} → {all_metrics[-1].eval_reward_mean:.1f}")
        print(f"Uniformity: {all_metrics[0].attention_uniformity:.3f} → {all_metrics[-1].attention_uniformity:.3f}")
        print(f"Wake ratio: {all_metrics[0].wake_ratio_mean:.2f} → {all_metrics[-1].wake_ratio_mean:.2f}")
    
    final = all_metrics[-1]
    print(f"\nFinal checkpoint (step {final.step:,}):")
    
    if final.attention_uniformity > 0.98:
        print("  ❌ Attention is uniform - model not yet specialized")
    elif final.attention_uniformity > 0.95:
        print("  ⚠️ Attention mostly uniform - continue training")
    else:
        print("  ✅ Attention is specialized")
    
    if final.wake_ratio_mean > 1.2:
        print("  ✅ Physics-aligned (attends more to upwind)")
    elif final.wake_ratio_mean < 0.8:
        print("  ⚪ Reverse pattern (attends more to downwind)")
    else:
        print("  ⚪ Neutral attention pattern")
    
    if final.reference_tracks_wind:
        print("  ✅ Reference turbine tracks wind direction")
    else:
        print("  ⚪ Reference turbine doesn't track wind")
    
    env.close()


if __name__ == '__main__':
    main()