"""
Evaluation and Visualization Script for Transformer Wind Farm Control

This script provides:
1. Performance evaluation across different farm layouts
2. Attention pattern visualization overlaid on farm layouts
3. Analysis of attention vs. wake physics (do turbines attend to upwind neighbors?)
4. Comparison plots for zero-shot transfer experiments
5. Checkpointing to resume interrupted evaluations
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import json
import pickle

# Import the transformer model components
from transformer_sac_windfarm_v4 import (
    TransformerActor,
    TransformerCritic,
    make_env,
    get_layout_positions,
    transform_to_wind_relative,
    get_positions_tensor,
    Args,
)

# WindGym imports
from windgym.WindGym import WindFarmEnv
from windgym.WindGym.wrappers import RecordEpisodeVals, PerTurbineObservationWrapper
from windgym.WindGym.utils.generate_layouts import generate_square_grid, generate_cirular_farm


@dataclass
class EvalConfig:
    """Configuration for evaluation."""
    model_path: str
    layouts: List[str]  # Layouts to evaluate on
    n_episodes: int = 10
    seed: int = 42
    device: str = "cuda"
    save_dir: str = "eval_results"
    turbtype: str = "DTU10MW"
    
    # Visualization settings
    plot_attention: bool = True
    plot_attention_episodes: int = 3  # Number of episodes to visualize attention
    attention_timesteps: List[int] = None  # Specific timesteps to visualize (None = evenly spaced)
    
    # Checkpointing
    checkpoint_file: str = None  # Will be set automatically if not provided
    resume: bool = True  # Whether to resume from checkpoint if available


def save_checkpoint(checkpoint_path: str, data: Dict):
    """Save evaluation checkpoint."""
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"  Checkpoint saved to {checkpoint_path}")


def load_checkpoint(checkpoint_path: str) -> Optional[Dict]:
    """Load evaluation checkpoint if it exists."""
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)
        print(f"Loaded checkpoint from {checkpoint_path}")
        return data
    return None


def load_model(
    model_path: str,
    obs_dim_per_turbine: int,
    action_dim_per_turbine: int,
    device: torch.device,
    args: Optional[Args] = None,
) -> Tuple[TransformerActor, TransformerCritic, TransformerCritic]:
    """Load trained transformer model from checkpoint."""
    
    if args is None:
        # Use defaults
        args = Args()
    
    # Determine action scaling (will be overwritten per-env, but need for model init)
    action_scale = 1.0
    action_bias = 0.0
    
    actor = TransformerActor(
        obs_dim_per_turbine=obs_dim_per_turbine,
        action_dim_per_turbine=action_dim_per_turbine,
        embed_dim=args.embed_dim,
        pos_embed_dim=args.pos_embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        mlp_ratio=args.mlp_ratio,
        dropout=0.0,  # No dropout at eval
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
        dropout=0.0,
    ).to(device)
    
    qf2 = TransformerCritic(
        obs_dim_per_turbine=obs_dim_per_turbine,
        action_dim_per_turbine=action_dim_per_turbine,
        embed_dim=args.embed_dim,
        pos_embed_dim=args.pos_embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        mlp_ratio=args.mlp_ratio,
        dropout=0.0,
    ).to(device)
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    actor.load_state_dict(checkpoint["actor"])
    qf1.load_state_dict(checkpoint["qf1"])
    qf2.load_state_dict(checkpoint["qf2"])
    
    actor.eval()
    qf1.eval()
    qf2.eval()
    
    print(f"Loaded model from {model_path}")
    
    return actor, qf1, qf2


def evaluate_on_layout(
    actor: TransformerActor,
    layout: str,
    config: EvalConfig,
    wind_turbine,
    device: torch.device,
    collect_attention: bool = False,
) -> Dict:
    """
    Evaluate the policy on a specific layout.
    
    Returns:
        Dictionary with evaluation metrics and optionally attention data.
    """
    # Create a simple args object for env creation
    env_args = Args()
    env_args.turbtype = config.turbtype
    env_args.layout_type = layout  # Set the layout in args
    
    env = make_env(env_args, wind_turbine, config.seed)
    
    # Update actor action scaling for this env
    action_high = env.action_space.high[0]
    action_low = env.action_space.low[0]
    actor.action_scale = torch.tensor((action_high - action_low) / 2.0, device=device)
    actor.action_bias = torch.tensor((action_high + action_low) / 2.0, device=device)
    
    # Get position info
    turbine_positions_np = env.turbine_positions
    rotor_diameter = env.rotor_diameter
    n_turbines = turbine_positions_np.shape[0]
    
    results = {
        "layout": layout,
        "n_turbines": n_turbines,
        "rotor_diameter": rotor_diameter,
        "episode_returns": [],
        "episode_powers": [],
        "episode_lengths": [],
        "attention_data": [] if collect_attention else None,
    }
    
    for ep in range(config.n_episodes):
        obs, _ = env.reset(seed=config.seed + ep)
        
        # Get wind direction for position transformation
        wind_dir = getattr(env, 'mean_wind_direction', 270.0)
        positions = get_positions_tensor(turbine_positions_np, rotor_diameter, wind_dir, device, wind_relative=True)
        
        episode_return = 0
        episode_power = []
        episode_attention = []
        done = False
        step = 0
        
        while not done:
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action, _, _, attn_weights = actor.get_action(
                    obs_tensor, positions, deterministic=True
                )
                action = action.squeeze(0).cpu().numpy().reshape(-1)
            
            # Collect attention if requested
            if collect_attention and ep < config.plot_attention_episodes:
                # Store attention from last layer, averaged over heads
                # attn_weights[-1] shape: (batch, n_heads, n_turb, n_turb)
                attn_last_layer = attn_weights[-1].squeeze(0).cpu().numpy()  # (n_heads, n_turb, n_turb)
                
                # Handle different possible shapes
                if attn_last_layer.ndim == 3:
                    attn_avg = attn_last_layer.mean(axis=0)  # (n_turb, n_turb)
                elif attn_last_layer.ndim == 2:
                    attn_avg = attn_last_layer  # Already (n_turb, n_turb)
                else:
                    # Unexpected shape, try to reshape
                    print(f"  Warning: Unexpected attention shape {attn_last_layer.shape}, skipping attention collection for this step")
                    attn_avg = None
                
                if attn_avg is not None:
                    # Ensure it's 2D
                    if attn_avg.ndim == 1:
                        # Reshape if flattened (n_turb * n_turb) -> (n_turb, n_turb)
                        side = int(np.sqrt(len(attn_avg)))
                        if side * side == len(attn_avg):
                            attn_avg = attn_avg.reshape(side, side)
                        else:
                            print(f"  Warning: Cannot reshape attention array of length {len(attn_avg)}")
                            attn_avg = None
                    
                    if attn_avg is not None:
                        episode_attention.append({
                            "step": step,
                            "attention": attn_avg.copy(),
                            "positions": turbine_positions_np.copy(),
                            "wind_direction": wind_dir,
                            "obs": obs.copy(),
                            "action": action.copy(),
                        })
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_return += reward
            if hasattr(env, 'current_power'):
                episode_power.append(env.current_power)
            
            step += 1
        
        results["episode_returns"].append(episode_return)
        results["episode_lengths"].append(step)
        if episode_power:
            results["episode_powers"].append(np.mean(episode_power))
        
        if collect_attention and ep < config.plot_attention_episodes:
            results["attention_data"].append(episode_attention)
        
        print(f"    Episode {ep+1}/{config.n_episodes}: return={episode_return:.2f}, length={step}")
    
    # Compute summary statistics
    results["mean_return"] = np.mean(results["episode_returns"])
    results["std_return"] = np.std(results["episode_returns"])
    results["mean_power"] = np.mean(results["episode_powers"]) if results["episode_powers"] else None
    results["mean_length"] = np.mean(results["episode_lengths"])
    
    env.close()
    
    return results


def plot_attention_on_farm(
    attention: np.ndarray,
    positions: np.ndarray,
    wind_direction: float,
    rotor_diameter: float,
    ax: plt.Axes = None,
    title: str = "",
    show_wind_arrow: bool = True,
    threshold: float = 0.1,  # Only show attention weights above this
):
    """
    Visualize attention weights overlaid on farm layout.
    
    Args:
        attention: (n_turbines, n_turbines) attention matrix
        positions: (n_turbines, 2) turbine positions
        wind_direction: Wind direction in degrees
        rotor_diameter: For scaling
        ax: Matplotlib axes
        title: Plot title
        show_wind_arrow: Whether to show wind direction arrow
        threshold: Minimum attention weight to display
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Validate attention shape
    if attention.ndim != 2:
        print(f"Warning: attention has shape {attention.shape}, expected 2D array. Skipping plot.")
        ax.text(0.5, 0.5, f"Invalid attention shape: {attention.shape}", 
                ha='center', va='center', transform=ax.transAxes)
        return ax
    
    n_turbines = len(positions)
    
    if attention.shape[0] != n_turbines or attention.shape[1] != n_turbines:
        print(f"Warning: attention shape {attention.shape} doesn't match {n_turbines} turbines. Skipping plot.")
        ax.text(0.5, 0.5, f"Shape mismatch: {attention.shape} vs {n_turbines} turbines", 
                ha='center', va='center', transform=ax.transAxes)
        return ax
    
    # Normalize positions for plotting
    pos_norm = positions / rotor_diameter
    
    # Plot turbines as circles
    for i in range(n_turbines):
        circle = plt.Circle(
            (pos_norm[i, 0], pos_norm[i, 1]), 
            0.3,  # Radius in rotor diameters
            color='steelblue', 
            ec='black', 
            linewidth=2,
            zorder=10
        )
        ax.add_patch(circle)
        ax.text(
            pos_norm[i, 0], pos_norm[i, 1], 
            str(i), 
            ha='center', va='center', 
            fontsize=10, fontweight='bold', color='white',
            zorder=11
        )
    
    # Plot attention as arrows
    # Color by attention strength
    cmap = plt.cm.Reds
    norm = Normalize(vmin=0, vmax=attention.max())
    
    for i in range(n_turbines):
        for j in range(n_turbines):
            if i != j and attention[i, j] > threshold:
                # Arrow from j to i (j attends to i, or i receives attention from j)
                # Convention: attention[i, j] = how much token i attends to token j
                # So we draw arrow from j -> i (information flows from j to i)
                
                dx = pos_norm[i, 0] - pos_norm[j, 0]
                dy = pos_norm[i, 1] - pos_norm[j, 1]
                
                # Shorten arrow to not overlap with turbine circles
                length = np.sqrt(dx**2 + dy**2)
                if length > 0:
                    # Start and end offsets
                    offset = 0.35 / length
                    start_x = pos_norm[j, 0] + dx * offset
                    start_y = pos_norm[j, 1] + dy * offset
                    end_x = pos_norm[i, 0] - dx * offset
                    end_y = pos_norm[i, 1] - dy * offset
                    
                    arrow = FancyArrowPatch(
                        (start_x, start_y),
                        (end_x, end_y),
                        arrowstyle='->,head_width=0.15,head_length=0.1',
                        color=cmap(norm(attention[i, j])),
                        linewidth=1 + 3 * attention[i, j],  # Thicker = more attention
                        alpha=0.7,
                        zorder=5,
                        connectionstyle="arc3,rad=0.1"  # Slight curve
                    )
                    ax.add_patch(arrow)
    
    # Wind direction arrow
    if show_wind_arrow:
        # Wind comes FROM this direction
        wind_rad = np.radians(wind_direction)
        # Arrow pointing in direction wind is going (opposite of where it comes from)
        wind_dx = -np.sin(wind_rad)  # Met convention
        wind_dy = -np.cos(wind_rad)
        
        # Place arrow in corner
        x_range = pos_norm[:, 0].max() - pos_norm[:, 0].min()
        y_range = pos_norm[:, 1].max() - pos_norm[:, 1].min()
        
        # Handle case where range is 0 (single row/column of turbines)
        x_range = max(x_range, 1.0)
        y_range = max(y_range, 1.0)
        
        arrow_start = (pos_norm[:, 0].min() - 0.15 * x_range, 
                       pos_norm[:, 1].max() + 0.1 * y_range)
        
        ax.annotate(
            '', 
            xy=(arrow_start[0] + wind_dx * 2, arrow_start[1] + wind_dy * 2),
            xytext=arrow_start,
            arrowprops=dict(arrowstyle='->', color='darkgreen', lw=3),
            zorder=15
        )
        ax.text(
            arrow_start[0] + wind_dx, arrow_start[1] + wind_dy + 0.5,
            f'Wind\n{wind_direction:.0f}°',
            ha='center', va='bottom', fontsize=10, color='darkgreen'
        )
    
    # Colorbar for attention
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, label='Attention Weight')
    
    ax.set_xlim(pos_norm[:, 0].min() - 2, pos_norm[:, 0].max() + 2)
    ax.set_ylim(pos_norm[:, 1].min() - 2, pos_norm[:, 1].max() + 2)
    ax.set_aspect('equal')
    ax.set_xlabel('x / D (rotor diameters)')
    ax.set_ylabel('y / D (rotor diameters)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_attention_heatmap(
    attention: np.ndarray,
    ax: plt.Axes = None,
    title: str = "",
):
    """Plot attention as a heatmap matrix."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Validate attention shape
    if attention.ndim != 2:
        print(f"Warning: attention has shape {attention.shape}, expected 2D array. Skipping heatmap.")
        ax.text(0.5, 0.5, f"Invalid attention shape: {attention.shape}", 
                ha='center', va='center', transform=ax.transAxes)
        return ax
    
    n_turbines = attention.shape[0]
    
    sns.heatmap(
        attention,
        ax=ax,
        cmap='Reds',
        vmin=0,
        vmax=attention.max(),
        annot=True if n_turbines <= 10 else False,
        fmt='.2f',
        xticklabels=[f'T{i}' for i in range(n_turbines)],
        yticklabels=[f'T{i}' for i in range(n_turbines)],
        cbar_kws={'label': 'Attention Weight'}
    )
    
    ax.set_xlabel('Key (attends to)')
    ax.set_ylabel('Query (attending)')
    ax.set_title(title)
    
    return ax


def analyze_attention_vs_wake(
    attention_data: List[Dict],
    rotor_diameter: float,
) -> Dict:
    """
    Analyze whether attention correlates with wake physics.
    
    Hypothesis: Turbines should attend more to upwind neighbors.
    
    Returns:
        Dictionary with analysis results.
    """
    upwind_attention = []
    downwind_attention = []
    crosswind_attention = []
    
    for step_data in attention_data:
        attention = step_data["attention"]
        positions = step_data["positions"]
        wind_dir = step_data["wind_direction"]
        
        # Validate attention is 2D
        if attention.ndim != 2:
            print(f"  Warning: Skipping attention data with shape {attention.shape}")
            continue
        
        n_turbines = len(positions)
        
        # Check shape compatibility
        if attention.shape[0] != n_turbines or attention.shape[1] != n_turbines:
            print(f"  Warning: Attention shape {attention.shape} doesn't match {n_turbines} turbines, skipping")
            continue
        
        # Compute wind-relative positions
        wind_rad = np.radians(wind_dir)
        # Unit vector in wind direction (where wind goes TO)
        wind_vec = np.array([-np.sin(wind_rad), -np.cos(wind_rad)])
        
        for i in range(n_turbines):
            for j in range(n_turbines):
                if i == j:
                    continue
                
                # Vector from j to i
                vec_ji = positions[i] - positions[j]
                dist = np.linalg.norm(vec_ji) / rotor_diameter
                
                if dist < 0.1:  # Skip if too close
                    continue
                
                # Project onto wind direction
                # Positive = j is upwind of i
                wind_proj = np.dot(vec_ji, wind_vec)
                
                # Classify relationship
                if wind_proj > 0.5 * dist:  # j is upwind of i
                    upwind_attention.append(attention[i, j])
                elif wind_proj < -0.5 * dist:  # j is downwind of i
                    downwind_attention.append(attention[i, j])
                else:  # Crosswind
                    crosswind_attention.append(attention[i, j])
    
    results = {
        "upwind_attention_mean": np.mean(upwind_attention) if upwind_attention else 0,
        "upwind_attention_std": np.std(upwind_attention) if upwind_attention else 0,
        "downwind_attention_mean": np.mean(downwind_attention) if downwind_attention else 0,
        "downwind_attention_std": np.std(downwind_attention) if downwind_attention else 0,
        "crosswind_attention_mean": np.mean(crosswind_attention) if crosswind_attention else 0,
        "crosswind_attention_std": np.std(crosswind_attention) if crosswind_attention else 0,
        "n_upwind": len(upwind_attention),
        "n_downwind": len(downwind_attention),
        "n_crosswind": len(crosswind_attention),
    }
    
    # Ratio: upwind / downwind attention
    if results["downwind_attention_mean"] > 0:
        results["upwind_downwind_ratio"] = (
            results["upwind_attention_mean"] / results["downwind_attention_mean"]
        )
    else:
        results["upwind_downwind_ratio"] = float('inf')
    
    return results


def plot_performance_comparison(
    all_results: Dict[str, Dict],
    save_path: str = None,
):
    """
    Create bar plot comparing performance across layouts.
    """
    layouts = list(all_results.keys())
    means = [all_results[l]["mean_return"] for l in layouts]
    stds = [all_results[l]["std_return"] for l in layouts]
    n_turbines = [all_results[l]["n_turbines"] for l in layouts]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Absolute returns
    ax1 = axes[0]
    x = np.arange(len(layouts))
    bars = ax1.bar(x, means, yerr=stds, capsize=5, color='steelblue', alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{l}\n({n}T)" for l, n in zip(layouts, n_turbines)], rotation=0)
    ax1.set_ylabel('Episode Return')
    ax1.set_title('Performance Across Farm Layouts')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        ax1.text(
            bar.get_x() + bar.get_width()/2, 
            bar.get_height() + std + 0.02 * max(means),
            f'{mean:.1f}',
            ha='center', va='bottom', fontsize=10
        )
    
    # Plot 2: Normalized by number of turbines (per-turbine performance)
    ax2 = axes[1]
    means_norm = [m / n for m, n in zip(means, n_turbines)]
    stds_norm = [s / n for s, n in zip(stds, n_turbines)]
    bars2 = ax2.bar(x, means_norm, yerr=stds_norm, capsize=5, color='seagreen', alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{l}\n({n}T)" for l, n in zip(layouts, n_turbines)], rotation=0)
    ax2.set_ylabel('Episode Return / Turbine')
    ax2.set_title('Per-Turbine Performance')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved performance comparison to {save_path}")
    
    return fig


def plot_attention_analysis(
    attention_analysis: Dict[str, Dict],
    save_path: str = None,
):
    """
    Plot attention vs wake physics analysis.
    """
    layouts = list(attention_analysis.keys())
    
    # Filter out layouts with no attention data
    layouts = [l for l in layouts if attention_analysis[l].get("n_upwind", 0) > 0 or 
               attention_analysis[l].get("n_downwind", 0) > 0]
    
    if not layouts:
        print("No valid attention data to plot")
        return None
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    x = np.arange(len(layouts))
    width = 0.25
    
    upwind = [attention_analysis[l]["upwind_attention_mean"] for l in layouts]
    downwind = [attention_analysis[l]["downwind_attention_mean"] for l in layouts]
    crosswind = [attention_analysis[l]["crosswind_attention_mean"] for l in layouts]
    
    bars1 = ax.bar(x - width, upwind, width, label='Upwind', color='indianred')
    bars2 = ax.bar(x, downwind, width, label='Downwind', color='steelblue')
    bars3 = ax.bar(x + width, crosswind, width, label='Crosswind', color='seagreen')
    
    ax.set_xticks(x)
    ax.set_xticklabels(layouts)
    ax.set_ylabel('Mean Attention Weight')
    ax.set_title('Attention by Relative Wind Position\n(Higher upwind attention suggests wake-aware learning)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add ratio annotations
    for i, l in enumerate(layouts):
        ratio = attention_analysis[l]["upwind_downwind_ratio"]
        if ratio != float('inf'):
            ax.annotate(
                f'↑/↓={ratio:.2f}',
                xy=(i, max(upwind[i], downwind[i], crosswind[i])),
                xytext=(0, 10),
                textcoords='offset points',
                ha='center',
                fontsize=9,
                color='darkred' if ratio > 1 else 'darkblue'
            )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved attention analysis to {save_path}")
    
    return fig


def create_attention_visualization_grid(
    attention_data: List[Dict],
    rotor_diameter: float,
    layout_name: str,
    timesteps: List[int] = None,
    save_path: str = None,
):
    """
    Create a grid of attention visualizations across timesteps.
    """
    if not attention_data:
        print(f"No attention data for {layout_name}")
        return None
    
    if timesteps is None:
        # Select evenly spaced timesteps
        n_steps = len(attention_data)
        timesteps = [0, n_steps // 4, n_steps // 2, 3 * n_steps // 4, n_steps - 1]
        timesteps = [t for t in timesteps if t < n_steps]
    
    # Filter to valid timesteps
    timesteps = [t for t in timesteps if t < len(attention_data)]
    
    if not timesteps:
        print(f"No valid timesteps for {layout_name}")
        return None
    
    n_plots = len(timesteps)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, t in enumerate(timesteps):
        step_data = attention_data[t]
        plot_attention_on_farm(
            attention=step_data["attention"],
            positions=step_data["positions"],
            wind_direction=step_data["wind_direction"],
            rotor_diameter=rotor_diameter,
            ax=axes[idx],
            title=f'{layout_name} - Step {t}',
            threshold=0.05,
        )
    
    # Hide unused axes
    for idx in range(len(timesteps), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f'Attention Evolution: {layout_name}', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved attention grid to {save_path}")
    
    return fig


def run_full_evaluation(config: EvalConfig):
    """
    Run complete evaluation pipeline with checkpointing.
    """
    # Setup
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(config.save_dir, exist_ok=True)
    
    # Set up checkpoint path
    if config.checkpoint_file is None:
        model_name = os.path.splitext(os.path.basename(config.model_path))[0]
        config.checkpoint_file = os.path.join(config.save_dir, f"checkpoint_{model_name}.pkl")
    
    # Try to load existing checkpoint
    checkpoint_data = None
    if config.resume:
        checkpoint_data = load_checkpoint(config.checkpoint_file)
    
    if checkpoint_data is not None:
        all_results = checkpoint_data.get("all_results", {})
        attention_analysis = checkpoint_data.get("attention_analysis", {})
        completed_layouts = set(checkpoint_data.get("completed_layouts", []))
        print(f"Resuming from checkpoint. Already completed: {completed_layouts}")
    else:
        all_results = {}
        attention_analysis = {}
        completed_layouts = set()
    
    # Wind turbine
    if config.turbtype == "DTU10MW":
        from py_wake.examples.data.dtu10mw import DTU10MW as WT
        rotor_diameter = 178.3  # meters
    elif config.turbtype == "V80":
        from py_wake.examples.data.hornsrev1 import V80 as WT
        rotor_diameter = 80.0
    else:
        raise ValueError(f"Unknown turbtype: {config.turbtype}")
    
    wind_turbine = WT()
    
    # Get dimensions from first layout
    env_args = Args()
    env_args.turbtype = config.turbtype
    env_args.layout_type = config.layouts[0]
    test_env = make_env(env_args, wind_turbine, config.seed)
    sample_obs, _ = test_env.reset()
    obs_dim_per_turbine = sample_obs.shape[1]
    action_dim_per_turbine = 1  # Yaw control
    test_env.close()
    
    print(f"Obs dim per turbine: {obs_dim_per_turbine}")
    print(f"Action dim per turbine: {action_dim_per_turbine}")
    
    # Load model
    actor, qf1, qf2 = load_model(
        config.model_path,
        obs_dim_per_turbine,
        action_dim_per_turbine,
        device,
    )
    
    # Evaluate on each layout
    for layout in config.layouts:
        if layout in completed_layouts:
            print(f"\nSkipping {layout} (already completed)")
            continue
        
        print(f"\nEvaluating on {layout}...")
        
        try:
            results = evaluate_on_layout(
                actor=actor,
                layout=layout,
                config=config,
                wind_turbine=wind_turbine,
                device=device,
                collect_attention=config.plot_attention,
            )
            
            # Get rotor diameter from results
            layout_rotor_d = results.get("rotor_diameter", rotor_diameter)
            
            all_results[layout] = results
            
            print(f"  Mean return: {results['mean_return']:.2f} ± {results['std_return']:.2f}")
            print(f"  Mean length: {results['mean_length']:.1f}")
            
            # Attention analysis
            if config.plot_attention and results["attention_data"]:
                # Flatten attention data from all episodes
                all_attention = []
                for ep_data in results["attention_data"]:
                    all_attention.extend(ep_data)
                
                if all_attention:
                    analysis = analyze_attention_vs_wake(all_attention, layout_rotor_d)
                    attention_analysis[layout] = analysis
                    
                    print(f"  Attention analysis:")
                    print(f"    Upwind mean: {analysis['upwind_attention_mean']:.3f} (n={analysis['n_upwind']})")
                    print(f"    Downwind mean: {analysis['downwind_attention_mean']:.3f} (n={analysis['n_downwind']})")
                    print(f"    Crosswind mean: {analysis['crosswind_attention_mean']:.3f} (n={analysis['n_crosswind']})")
                    if analysis['upwind_downwind_ratio'] != float('inf'):
                        print(f"    Upwind/Downwind ratio: {analysis['upwind_downwind_ratio']:.2f}")
                    
                    # Create attention visualization for first episode
                    if results["attention_data"] and results["attention_data"][0]:
                        create_attention_visualization_grid(
                            attention_data=results["attention_data"][0],
                            rotor_diameter=layout_rotor_d,
                            layout_name=layout,
                            save_path=os.path.join(config.save_dir, f"attention_grid_{layout}.png"),
                        )
                else:
                    print(f"  No valid attention data collected for {layout}")
            
            # Mark as completed and save checkpoint
            completed_layouts.add(layout)
            save_checkpoint(config.checkpoint_file, {
                "all_results": all_results,
                "attention_analysis": attention_analysis,
                "completed_layouts": list(completed_layouts),
            })
            
        except Exception as e:
            print(f"  Error evaluating {layout}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create comparison plots (only if we have results)
    if all_results:
        plot_performance_comparison(
            all_results,
            save_path=os.path.join(config.save_dir, "performance_comparison.png"),
        )
        
        if attention_analysis:
            plot_attention_analysis(
                attention_analysis,
                save_path=os.path.join(config.save_dir, "attention_analysis.png"),
            )
        
        # Save results as JSON (convert numpy types for JSON serialization)
        def convert_to_serializable(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(v) for v in obj]
            return obj
        
        results_summary = {
            layout: {
                "mean_return": res["mean_return"],
                "std_return": res["std_return"],
                "mean_power": res["mean_power"],
                "mean_length": res["mean_length"],
                "n_turbines": res["n_turbines"],
                "episode_returns": res["episode_returns"],
                "episode_lengths": res["episode_lengths"],
                "attention_analysis": attention_analysis.get(layout, {}),
            }
            for layout, res in all_results.items()
        }
        
        results_summary = convert_to_serializable(results_summary)
        
        with open(os.path.join(config.save_dir, "results.json"), "w") as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"\nResults saved to {config.save_dir}/")
    else:
        print("\nNo results to save.")
    
    # Clean up checkpoint after successful completion
    if os.path.exists(config.checkpoint_file) and len(completed_layouts) == len(config.layouts):
        print(f"Evaluation complete. Checkpoint file kept at {config.checkpoint_file}")
    
    return all_results, attention_analysis


def main():
    parser = argparse.ArgumentParser(description="Evaluate Transformer Wind Farm Controller")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--layouts", type=str, default="square_1,square_2,circular_1,circular_2",
                        help="Comma-separated list of layouts to evaluate")
    parser.add_argument("--n_episodes", type=int, default=10, help="Episodes per layout")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--save_dir", type=str, default="eval_results", help="Output directory")
    parser.add_argument("--turbtype", type=str, default="DTU10MW", help="Turbine type")
    parser.add_argument("--no_attention", action="store_true", help="Skip attention visualization")
    parser.add_argument("--no_resume", action="store_true", help="Don't resume from checkpoint")
    
    args = parser.parse_args()
    
    config = EvalConfig(
        model_path=args.model_path,
        layouts=[l.strip() for l in args.layouts.split(",")],
        n_episodes=args.n_episodes,
        seed=args.seed,
        device=args.device,
        save_dir=args.save_dir,
        turbtype=args.turbtype,
        plot_attention=not args.no_attention,
        resume=not args.no_resume,
    )
    
    run_full_evaluation(config)


if __name__ == "__main__":
    main()