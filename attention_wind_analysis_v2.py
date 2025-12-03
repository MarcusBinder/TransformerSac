"""
Wind Direction Conditioned Attention Analysis v2

Analyzes how attention patterns change with wind direction,
with clear convergence indicators based on evaluation performance.

Usage:
    python attention_wind_analysis_v2.py --run_dir runs/YOUR_RUN
    python attention_wind_analysis_v2.py --checkpoint path/to/checkpoint.pt

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
from matplotlib.gridspec import GridSpec
from typing import Optional, List, Dict, Tuple, Any
import math
from tqdm import tqdm
from collections import defaultdict


# =============================================================================
# MODEL COMPONENTS
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
    pos_norm = positions / rotor_diameter
    wind_rad = np.radians(270 - wind_direction)
    wind_vec = np.array([np.cos(wind_rad), np.sin(wind_rad)])
    wind_axis_positions = np.dot(pos_norm, wind_vec)
    median_pos = np.median(wind_axis_positions)
    
    return {
        'upwind_indices': np.where(wind_axis_positions >= median_pos)[0].tolist(),
        'downwind_indices': np.where(wind_axis_positions < median_pos)[0].tolist(),
        'wind_axis_positions': wind_axis_positions,
        'most_upwind': int(np.argmax(wind_axis_positions)),
        'most_downwind': int(np.argmin(wind_axis_positions)),
    }


def get_wind_direction_bin(wind_direction: float, n_bins: int = 4) -> int:
    return int((wind_direction % 360) // (360 / n_bins))


def get_bin_label(bin_idx: int, n_bins: int = 4) -> str:
    bin_size = 360 / n_bins
    start, end = bin_idx * bin_size, (bin_idx + 1) * bin_size
    directions = {0: 'N', 45: 'NE', 90: 'E', 135: 'SE', 180: 'S', 225: 'SW', 270: 'W', 315: 'NW'}
    mid = (start + end) / 2
    closest = min(directions.keys(), key=lambda x: min(abs(x - mid), 360 - abs(x - mid)))
    return f"{start:.0f}°-{end:.0f}° ({directions[closest]})"


# =============================================================================
# DATA COLLECTION
# =============================================================================

def collect_attention_data(actor, env, device, n_episodes=20, n_steps_per_episode=30, seed=42):
    positions = env.turbine_positions
    rotor_diameter = env.rotor_diameter
    base_env = env.env if hasattr(env, 'env') else env
    
    action_high = base_env.action_space.high[0]
    action_low = base_env.action_space.low[0]
    actor.action_scale = torch.tensor((action_high - action_low) / 2.0, device=device)
    actor.action_bias = torch.tensor((action_high + action_low) / 2.0, device=device)
    
    all_data = []
    episode_metrics = {'episode': [], 'wind_direction': [], 'total_reward': []}
    
    for ep in tqdm(range(n_episodes), desc="Collecting episodes"):
        obs, info = env.reset(seed=seed + ep)
        wind_direction = getattr(base_env, 'wd', 270.0)
        episode_reward = 0.0
        
        for step in range(n_steps_per_episode):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            pos_norm = positions / rotor_diameter
            pos_t = torch.tensor(pos_norm, dtype=torch.float32, device=device).unsqueeze(0)
            pos_transformed = transform_to_wind_relative(pos_t, wind_direction)
            
            with torch.no_grad():
                action, _, mean_action, attn_weights = actor.get_action(obs_t, pos_transformed, deterministic=True)
            
            attn_last = attn_weights[-1][0].cpu().numpy()
            action_np = action[0].cpu().numpy().flatten()
            next_obs, reward, terminated, truncated, info = env.step(action_np)
            
            episode_reward += reward
            all_data.append({
                'wind_direction': wind_direction,
                'attention_per_head': attn_last,
                'attention_avg': attn_last.mean(axis=0),
                'reward': reward,
                'episode': ep,
                'step': step,
            })
            
            obs = next_obs
            if terminated or truncated:
                break
        
        episode_metrics['episode'].append(ep)
        episode_metrics['wind_direction'].append(wind_direction)
        episode_metrics['total_reward'].append(episode_reward)
    
    return all_data, episode_metrics


# =============================================================================
# ANALYSIS
# =============================================================================

def analyze_by_wind_direction(data, positions, rotor_diameter, n_bins=4):
    n_turbines = len(positions)
    binned_data = defaultdict(list)
    for d in data:
        binned_data[get_wind_direction_bin(d['wind_direction'], n_bins)].append(d)
    
    results = {}
    for bin_idx in range(n_bins):
        if bin_idx not in binned_data:
            continue
        
        bin_data = binned_data[bin_idx]
        attention_avg = np.mean([d['attention_avg'] for d in bin_data], axis=0)
        mean_wd = np.mean([d['wind_direction'] for d in bin_data])
        mean_reward = np.mean([d['reward'] for d in bin_data])
        ud_info = get_upwind_downwind_turbines(positions, mean_wd, rotor_diameter)
        
        upwind_attn, downwind_attn = [], []
        for i in range(n_turbines):
            for j in range(n_turbines):
                if i == j: continue
                if ud_info['wind_axis_positions'][j] > ud_info['wind_axis_positions'][i]:
                    upwind_attn.append(attention_avg[i, j])
                else:
                    downwind_attn.append(attention_avg[i, j])
        
        upwind_mean = np.mean(upwind_attn) if upwind_attn else 0
        downwind_mean = np.mean(downwind_attn) if downwind_attn else 0
        
        results[bin_idx] = {
            'label': get_bin_label(bin_idx, n_bins),
            'mean_wind_direction': mean_wd,
            'n_samples': len(bin_data),
            'mean_reward': mean_reward,
            'attention_avg': attention_avg,
            'upwind_mean': upwind_mean,
            'downwind_mean': downwind_mean,
            'ratio': upwind_mean / downwind_mean if downwind_mean > 0 else float('inf'),
            'reference_turbine': int(np.argmax(attention_avg.sum(axis=0))),
            'most_upwind': ud_info['most_upwind'],
            'most_downwind': ud_info['most_downwind'],
            'upwind_indices': ud_info['upwind_indices'],
            'downwind_indices': ud_info['downwind_indices'],
        }
    return results


def analyze_per_head(data, positions, rotor_diameter):
    n_turbines = len(positions)
    n_heads = data[0]['attention_per_head'].shape[0]
    head_analysis = {}
    
    for head_idx in range(n_heads):
        head_attention = np.mean([d['attention_per_head'][head_idx] for d in data], axis=0)
        col_sums = head_attention.sum(axis=0)
        focus_turbine = int(np.argmax(col_sums))
        focus_score = col_sums[focus_turbine] / col_sums.sum()
        
        attn_flat = head_attention.flatten()
        attn_flat = attn_flat / attn_flat.sum()
        entropy = -np.sum(attn_flat * np.log(attn_flat + 1e-10))
        uniformity = entropy / np.log(len(attn_flat))
        
        head_analysis[head_idx] = {
            'attention_matrix': head_attention,
            'focus_turbine': focus_turbine,
            'focus_score': focus_score,
            'uniformity': uniformity,
        }
    return head_analysis


def analyze_reference_rotation(binned_results):
    details = []
    for bin_idx, result in binned_results.items():
        details.append({
            'wind_direction': result['mean_wind_direction'],
            'reference_turbine': result['reference_turbine'],
            'most_downwind': result['most_downwind'],
            'most_upwind': result['most_upwind'],
        })
    
    matches_down = [d['reference_turbine'] == d['most_downwind'] for d in details]
    matches_up = [d['reference_turbine'] == d['most_upwind'] for d in details]
    
    return {
        'details': details,
        'pct_matches_downwind': np.mean(matches_down) * 100 if matches_down else 0,
        'pct_matches_upwind': np.mean(matches_up) * 100 if matches_up else 0,
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_summary_dashboard(binned_results, head_analysis, rotation_analysis, episode_metrics, 
                           current_step, positions, rotor_diameter, save_path=None):
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.3)
    
    pos_norm = positions / rotor_diameter
    n_turbines = len(positions)
    n_heads = len(head_analysis)
    bins = sorted(binned_results.keys())
    
    # Row 0: Evaluation rewards
    ax_rewards = fig.add_subplot(gs[0, :2])
    rewards = episode_metrics['total_reward']
    ax_rewards.plot(episode_metrics['episode'], rewards, 'o-', color='steelblue', markersize=4, alpha=0.7)
    ax_rewards.axhline(y=np.mean(rewards), color='crimson', linestyle='--', lw=2, label=f'Mean: {np.mean(rewards):.1f}')
    ax_rewards.fill_between(episode_metrics['episode'], 
                            np.mean(rewards) - np.std(rewards), np.mean(rewards) + np.std(rewards),
                            alpha=0.2, color='crimson')
    ax_rewards.set_xlabel('Episode')
    ax_rewards.set_ylabel('Episode Reward')
    ax_rewards.set_title(f'Evaluation Performance (step {current_step:,})')
    ax_rewards.legend()
    ax_rewards.grid(True, alpha=0.3)
    
    # Reward by wind direction
    ax_wd = fig.add_subplot(gs[0, 2:])
    wd_rewards = defaultdict(list)
    for wd, r in zip(episode_metrics['wind_direction'], episode_metrics['total_reward']):
        wd_rewards[get_wind_direction_bin(wd, 4)].append(r)
    
    labels = [get_bin_label(b, 4)[:8] for b in sorted(wd_rewards.keys())]
    means = [np.mean(wd_rewards[b]) for b in sorted(wd_rewards.keys())]
    stds = [np.std(wd_rewards[b]) for b in sorted(wd_rewards.keys())]
    
    bars = ax_wd.bar(range(len(labels)), means, yerr=stds, capsize=5, color='forestgreen', alpha=0.7)
    ax_wd.set_xticks(range(len(labels)))
    ax_wd.set_xticklabels(labels, rotation=45, ha='right')
    ax_wd.set_ylabel('Episode Reward')
    ax_wd.set_title('Reward by Wind Direction')
    ax_wd.grid(True, alpha=0.3, axis='y')
    
    # Row 1: Attention heatmaps
    for idx, bin_idx in enumerate(bins[:4]):
        ax = fig.add_subplot(gs[1, idx])
        result = binned_results[bin_idx]
        attention = result['attention_avg']
        
        ax.imshow(attention, cmap='Reds', vmin=0)
        ax.set_title(f"{result['label'][:12]}\nRef: T{result['reference_turbine']}", fontsize=9)
        ax.set_xticks(range(n_turbines))
        ax.set_yticks(range(n_turbines))
        ax.set_xticklabels([f'{i}' for i in range(n_turbines)], fontsize=7)
        ax.set_yticklabels([f'{i}' for i in range(n_turbines)], fontsize=7)
        
        if n_turbines <= 4:
            for i in range(n_turbines):
                for j in range(n_turbines):
                    ax.text(j, i, f'{attention[i,j]:.2f}', ha='center', va='center', fontsize=6)
    
    # Row 2: Per-head focus + wake alignment
    ax_heads = fig.add_subplot(gs[2, :2])
    focus_scores = [head_analysis[i]['focus_score'] for i in range(n_heads)]
    uniformities = [head_analysis[i]['uniformity'] for i in range(n_heads)]
    
    bars = ax_heads.bar([f'Head {i}' for i in range(n_heads)], focus_scores, color='steelblue')
    ax_heads.axhline(y=1/n_turbines, color='gray', linestyle='--', label=f'Uniform ({1/n_turbines:.2f})')
    ax_heads.set_ylabel('Focus Score')
    ax_heads.set_title('Per-Head Focus')
    ax_heads.legend()
    
    for bar, h in zip(bars, range(n_heads)):
        ax_heads.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'T{head_analysis[h]["focus_turbine"]}', ha='center', fontsize=8)
    
    ax_ratio = fig.add_subplot(gs[2, 2:])
    ratios = [min(binned_results[b]['ratio'], 3) for b in bins[:4]]
    colors = ['forestgreen' if r > 1.1 else ('crimson' if r < 0.9 else 'gray') for r in ratios]
    ax_ratio.bar([binned_results[b]['label'][:8] for b in bins[:4]], ratios, color=colors)
    ax_ratio.axhline(y=1.0, color='black', linestyle='--', lw=2)
    ax_ratio.set_ylabel('Upwind/Downwind Ratio')
    ax_ratio.set_title('Wake Physics Alignment (>1 = upwind)')
    ax_ratio.tick_params(axis='x', rotation=45)
    
    # Row 3: Reference tracking + findings
    ax_ref = fig.add_subplot(gs[3, :2])
    details = rotation_analysis['details']
    x = range(len(details))
    width = 0.25
    ax_ref.bar([i-width for i in x], [d['reference_turbine'] for d in details], width, label='Reference', color='crimson')
    ax_ref.bar([i for i in x], [d['most_downwind'] for d in details], width, label='Downwind', color='steelblue')
    ax_ref.bar([i+width for i in x], [d['most_upwind'] for d in details], width, label='Upwind', color='forestgreen')
    ax_ref.set_xticks(x)
    ax_ref.set_xticklabels([f'{d["wind_direction"]:.0f}°' for d in details], rotation=45)
    ax_ref.set_ylabel('Turbine Index')
    ax_ref.set_title('Reference vs Upwind/Downwind')
    ax_ref.legend()
    ax_ref.set_yticks(range(n_turbines))
    
    ax_findings = fig.add_subplot(gs[3, 2:])
    ax_findings.axis('off')
    
    avg_uniformity = np.mean(uniformities)
    avg_ratio = np.mean([binned_results[b]['ratio'] for b in bins])
    
    findings = [f"📊 Eval reward: {np.mean(rewards):.1f} ± {np.std(rewards):.1f}"]
    
    if avg_uniformity > 0.98:
        findings.append(f"❌ Attention uniform ({avg_uniformity:.3f}) - needs training")
    elif avg_uniformity > 0.95:
        findings.append(f"⚠️ Attention mostly uniform ({avg_uniformity:.3f})")
    else:
        findings.append(f"✅ Attention specialized ({avg_uniformity:.3f})")
    
    if rotation_analysis['pct_matches_downwind'] > 60:
        findings.append(f"✅ Reference tracks downwind")
    elif rotation_analysis['pct_matches_upwind'] > 60:
        findings.append(f"✅ Reference tracks upwind")
    else:
        findings.append("⚪ Reference doesn't track wind")
    
    if avg_ratio > 1.2:
        findings.append(f"✅ Physics-aligned (ratio={avg_ratio:.2f})")
    elif avg_ratio < 0.8:
        findings.append(f"⚪ Reverse pattern (ratio={avg_ratio:.2f})")
    else:
        findings.append(f"⚪ Neutral (ratio={avg_ratio:.2f})")
    
    ax_findings.text(0.05, 0.95, "Key Findings", fontsize=14, fontweight='bold', transform=ax_findings.transAxes, va='top')
    ax_findings.text(0.05, 0.80, "\n\n".join(findings), fontsize=11, transform=ax_findings.transAxes, va='top',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle(f'Attention Analysis - Step {current_step:,}', fontsize=16, y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    return fig


# =============================================================================
# MAIN
# =============================================================================

def load_actor(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint.get('args', {})
    actor_state = checkpoint['actor_state_dict']
    
    step = checkpoint.get('global_step', 0)
    if step == 0:
        try:
            step = int(''.join(filter(str.isdigit, os.path.basename(checkpoint_path))))
        except:
            step = 0
    
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--run_dir', type=str)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n_episodes', type=int, default=20)
    parser.add_argument('--n_steps', type=int, default=30)
    parser.add_argument('--n_bins', type=int, default=4)
    parser.add_argument('--save_dir', type=str, default='attention_analysis')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    elif args.run_dir:
        ckpts = glob.glob(os.path.join(args.run_dir, 'checkpoints', '*.pt')) or glob.glob(os.path.join(args.run_dir, '*.pt'))
        if not ckpts:
            print("No checkpoints found")
            return
        checkpoint_path = max(ckpts, key=os.path.getmtime)
    else:
        print("Provide --checkpoint or --run_dir")
        return
    
    print(f"Loading: {checkpoint_path}")
    os.makedirs(args.save_dir, exist_ok=True)
    
    actor, model_args, current_step = load_actor(checkpoint_path, device)
    print(f"Step {current_step:,}, {actor.num_heads} heads")
    
    # Create environment
    try:
        from WindGym import WindFarmEnv
        from WindGym.wrappers import PerTurbineObservationWrapper
        from WindGym.utils.generate_layouts import generate_square_grid
        from py_wake.examples.data.dtu10mw import DTU10MW
        
        turbine = DTU10MW()
        layout = model_args.get('layouts', 'square_2x2').split(',')[0].strip()
        
        if layout == 'test_layout':
            x_pos, y_pos = generate_square_grid(turbine=turbine, nx=2, ny=1, xDist=5, yDist=5)
        else:
            x_pos, y_pos = generate_square_grid(turbine=turbine, nx=2, ny=2, xDist=5, yDist=5)
        
        config = {
            "yaw_init": "Random", "BaseController": "Local", "ActionMethod": "yaw",
            "Track_power": False, "farm": {"yaw_min": -30, "yaw_max": 30},
            "wind": {"ws_min": 9, "ws_max": 9, "TI_min": 0.05, "TI_max": 0.05, "wd_min": 180, "wd_max": 360},
            "act_pen": {"action_penalty": 0.0, "action_penalty_type": "Change"},
            "power_def": {"Power_reward": "Baseline", "Power_avg": 1, "Power_scaling": 1.0},
            "mes_level": {"turb_ws": True, "turb_wd": True, "turb_TI": False, "turb_power": True,
                         "farm_ws": False, "farm_wd": False, "farm_TI": False, "farm_power": False},
            "ws_mes": {"ws_current": False, "ws_rolling_mean": True, "ws_history_N": 15, "ws_history_length": 15, "ws_window_length": 1},
            "wd_mes": {"wd_current": False, "wd_rolling_mean": True, "wd_history_N": 15, "wd_history_length": 15, "wd_window_length": 1},
            "yaw_mes": {"yaw_current": False, "yaw_rolling_mean": True, "yaw_history_N": 15, "yaw_history_length": 15, "yaw_window_length": 1},
            "power_mes": {"power_current": False, "power_rolling_mean": True, "power_history_N": 15, "power_history_length": 15, "power_window_length": 1},
        }
        
        base_env = WindFarmEnv(turbine=turbine, x_pos=x_pos, y_pos=y_pos, config=config, dt_sim=5, dt_env=10, n_passthrough=20, seed=args.seed)
        env = PerTurbineObservationWrapper(base_env)
        positions = env.turbine_positions
        rotor_diameter = env.rotor_diameter
        print(f"Layout: {layout}, {len(positions)} turbines")
        
    except ImportError as e:
        print(f"WindGym error: {e}")
        return
    
    # Collect and analyze
    print(f"\nCollecting {args.n_episodes} episodes...")
    data, episode_metrics = collect_attention_data(actor, env, device, args.n_episodes, args.n_steps, args.seed)
    
    print(f"Eval reward: {np.mean(episode_metrics['total_reward']):.2f} ± {np.std(episode_metrics['total_reward']):.2f}")
    
    binned_results = analyze_by_wind_direction(data, positions, rotor_diameter, args.n_bins)
    head_analysis = analyze_per_head(data, positions, rotor_diameter)
    rotation_analysis = analyze_reference_rotation(binned_results)
    
    avg_uniformity = np.mean([head_analysis[h]['uniformity'] for h in head_analysis])
    print(f"Attention uniformity: {avg_uniformity:.3f}", end="")
    print(" ❌" if avg_uniformity > 0.98 else (" ⚠️" if avg_uniformity > 0.95 else " ✅"))
    
    # Plot
    plot_summary_dashboard(binned_results, head_analysis, rotation_analysis, episode_metrics, 
                           current_step, positions, rotor_diameter,
                           save_path=os.path.join(args.save_dir, 'summary_dashboard.png'))
    plt.close()
    
    # Save metrics
    metrics = {
        'checkpoint_step': current_step,
        'eval_reward_mean': float(np.mean(episode_metrics['total_reward'])),
        'eval_reward_std': float(np.std(episode_metrics['total_reward'])),
        'attention_uniformity': float(avg_uniformity),
        'episode_rewards': episode_metrics['total_reward'],
        'episode_wind_directions': episode_metrics['wind_direction'],
    }
    
    with open(os.path.join(args.save_dir, 'analysis_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✓ Saved to {args.save_dir}/")
    env.close()


if __name__ == '__main__':
    main()