"""
Attention Visualization for Transformer Wind Farm Control

This script loads a trained transformer SAC model and creates comprehensive
visualizations of the attention patterns to verify that the model learns
physically meaningful wake interactions.

Usage:
    python visualize_attention.py --model_path runs/experiment_name/actor.pth \
                                   --layout test_layout \
                                   --n_episodes 5 \
                                   --save_dir attention_viz

Author: Marcus (DTU Wind Energy)
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json

# Import the transformer model components
from transformer_sac_windfarm_v9 import (
    TransformerActor,
    TransformerCritic,
    make_env,
    Args,
)

# Import viz utilities
from viz_utils import (
    plot_farm_layout,
    plot_attention_simple,
    plot_attention_by_head,
    create_attention_animation,
)

# WindGym imports
from WindGym import WindFarmEnv
from WindGym.wrappers import RecordEpisodeVals, PerTurbineObservationWrapper
from WindGym.utils.generate_layouts import generate_square_grid, generate_cirular_farm


def load_trained_actor(
    model_path: str,
    obs_dim_per_turbine: int,
    action_dim_per_turbine: int = 1,
    device: torch.device = None,
) -> TransformerActor:
    """
    Load a trained actor from checkpoint.
    
    Args:
        model_path: Path to the actor checkpoint
        obs_dim_per_turbine: Observation dimension per turbine
        action_dim_per_turbine: Action dimension per turbine
        device: Device to load model on
    
    Returns:
        Loaded actor model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create actor with same architecture as training
    # You may need to adjust these hyperparameters to match your trained model
    actor = TransformerActor(
        obs_dim_per_turbine=obs_dim_per_turbine,
        action_dim_per_turbine=action_dim_per_turbine,
        embed_dim=128,
        pos_embed_dim=32,
        num_heads=4,
        num_layers=2,
        mlp_ratio=2.0,
        dropout=0.0,
        use_farm_token=False,
        action_scale=1.0,
        action_bias=0.0,
    ).to(device)
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    actor.load_state_dict(checkpoint)
    actor.eval()
    
    print(f"Loaded actor from {model_path}")
    print(f"  obs_dim_per_turbine: {obs_dim_per_turbine}")
    print(f"  action_dim_per_turbine: {action_dim_per_turbine}")
    print(f"  embed_dim: {actor.embed_dim}")
    print(f"  num_heads: {actor.transformer.num_heads}")
    print(f"  num_layers: {actor.transformer.num_layers}")
    
    return actor


def collect_attention_episode(
    actor: TransformerActor,
    env: gym.Env,
    device: torch.device,
    max_steps: int = 200,
) -> Dict:
    """
    Run one episode and collect attention weights at each timestep.
    
    Args:
        actor: Trained actor model
        env: Wind farm environment
        device: Device for computation
        max_steps: Maximum steps per episode
    
    Returns:
        Dict containing:
            - attention_history: List of attention weights per timestep
            - positions: Turbine positions
            - wind_directions: Wind direction at each timestep
            - actions: Actions taken
            - rewards: Rewards received
            - yaw_angles: Yaw angles at each timestep
    """
    obs, info = env.reset()
    positions = env.turbine_positions
    rotor_diameter = env.rotor_diameter
    
    attention_history = []
    wind_directions = []
    actions_history = []
    rewards_history = []
    yaw_angles_history = []
    
    done = False
    step = 0
    
    while not done and step < max_steps:
        # Prepare inputs
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        
        # Get wind-relative positions
        # Positions are already in meters from the environment
        pos_normalized = positions / rotor_diameter
        
        # Transform to wind-relative coordinates
        wind_dir = env.mean_wind_direction
        wind_rad = np.radians(wind_dir)
        cos_wd = np.cos(wind_rad)
        sin_wd = np.sin(wind_rad)
        rotation_matrix = np.array([
            [cos_wd, sin_wd],
            [-sin_wd, cos_wd]
        ])
        pos_wind_relative = pos_normalized @ rotation_matrix.T
        
        pos_tensor = torch.tensor(
            pos_wind_relative, dtype=torch.float32, device=device
        ).unsqueeze(0)
        
        # Get action and attention weights
        with torch.no_grad():
            action, log_prob, mean_action, attn_weights = actor.get_action(
                obs_tensor, pos_tensor, deterministic=True
            )
        
        # Store attention weights (from all layers)
        # Each element in attn_weights is (batch, n_heads, n_tokens, n_tokens)
        attention_history.append([
            attn.squeeze(0).cpu().numpy() for attn in attn_weights
        ])
        
        wind_directions.append(wind_dir)
        actions_history.append(action.squeeze(0).cpu().numpy())
        yaw_angles_history.append(info.get("yaw angles agent", np.zeros(len(positions))))
        
        # Step environment
        action_flat = action.squeeze().cpu().numpy().flatten()
        obs, reward, terminated, truncated, info = env.step(action_flat)
        done = terminated or truncated
        
        rewards_history.append(reward)
        step += 1
    
    return {
        'attention_history': attention_history,
        'positions': positions,
        'rotor_diameter': rotor_diameter,
        'wind_directions': wind_directions,
        'actions': actions_history,
        'rewards': rewards_history,
        'yaw_angles': yaw_angles_history,
        'n_steps': step,
    }


def visualize_attention_at_timestep(
    attention_weights: List[np.ndarray],
    positions: np.ndarray,
    rotor_diameter: float,
    wind_direction: float,
    timestep: int,
    layer_idx: int = -1,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 10),
):
    """
    Create comprehensive visualization of attention at a single timestep.
    
    Shows:
    - Farm layout with attention arrows for each head
    - Heatmaps for each attention head
    - Average attention across heads
    
    Args:
        attention_weights: List of attention tensors per layer, each (n_heads, n_turb, n_turb)
        positions: (n_turbines, 2) turbine positions
        rotor_diameter: Rotor diameter for normalization
        wind_direction: Wind direction in degrees
        timestep: Current timestep (for title)
        layer_idx: Which layer to visualize (-1 = last)
        save_path: Path to save figure
    """
    attn_layer = attention_weights[layer_idx]  # (n_heads, n_turb, n_turb)
    n_heads = attn_layer.shape[0]
    n_turbines = len(positions)
    
    # Create figure
    n_cols = min(4, n_heads + 1)
    n_rows = (n_heads + 1 + n_cols - 1) // n_cols
    fig = plt.figure(figsize=figsize)
    
    pos_norm = positions / rotor_diameter
    
    # Plot attention for each head
    for h in range(n_heads):
        ax = fig.add_subplot(n_rows, n_cols, h + 1)
        
        attn = attn_layer[h]
        
        # Draw turbines
        for i in range(n_turbines):
            circle = Circle(
                (pos_norm[i, 0], pos_norm[i, 1]), 0.2,
                color='steelblue', ec='black', lw=1.5, zorder=10
            )
            ax.add_patch(circle)
            ax.text(
                pos_norm[i, 0], pos_norm[i, 1], str(i),
                ha='center', va='center', fontsize=8,
                fontweight='bold', color='white', zorder=11
            )
        
        # Draw attention arrows
        max_attn = attn.max()
        threshold = 0.05
        
        for i in range(n_turbines):
            for j in range(n_turbines):
                if i != j and attn[i, j] > threshold:
                    alpha = min(attn[i, j] / max_attn, 1.0)
                    
                    # Arrow from j to i (j provides information to i)
                    start = pos_norm[j]
                    end = pos_norm[i]
                    vec = end - start
                    length = np.linalg.norm(vec)
                    
                    if length > 0:
                        vec_norm = vec / length
                        start_adj = start + vec_norm * 0.25
                        end_adj = end - vec_norm * 0.25
                        
                        ax.annotate(
                            '', xy=end_adj, xytext=start_adj,
                            arrowprops=dict(
                                arrowstyle='->',
                                color='red',
                                alpha=alpha,
                                lw=0.5 + 2*alpha
                            )
                        )
        
        # Wind direction arrow
        center = pos_norm.mean(axis=0)
        wind_rad = np.radians(wind_direction)
        wind_dx = -np.sin(wind_rad) * 1.0
        wind_dy = -np.cos(wind_rad) * 1.0
        ax.annotate(
            '', xy=(center[0] + wind_dx, center[1] + wind_dy),
            xytext=(center[0], center[1]),
            arrowprops=dict(arrowstyle='->', color='green', lw=2)
        )
        
        ax.set_xlim(pos_norm[:, 0].min() - 1.5, pos_norm[:, 0].max() + 1.5)
        ax.set_ylim(pos_norm[:, 1].min() - 1.5, pos_norm[:, 1].max() + 1.5)
        ax.set_aspect('equal')
        ax.set_title(f'Head {h}', fontsize=10)
        ax.set_xlabel('x / D', fontsize=8)
        ax.set_ylabel('y / D', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Average attention across heads
    ax = fig.add_subplot(n_rows, n_cols, n_heads + 1)
    attn_avg = attn_layer.mean(axis=0)
    
    # Draw turbines
    for i in range(n_turbines):
        circle = Circle(
            (pos_norm[i, 0], pos_norm[i, 1]), 0.2,
            color='steelblue', ec='black', lw=1.5, zorder=10
        )
        ax.add_patch(circle)
        ax.text(
            pos_norm[i, 0], pos_norm[i, 1], str(i),
            ha='center', va='center', fontsize=8,
            fontweight='bold', color='white', zorder=11
        )
    
    # Draw attention arrows
    max_attn = attn_avg.max()
    for i in range(n_turbines):
        for j in range(n_turbines):
            if i != j and attn_avg[i, j] > threshold:
                alpha = min(attn_avg[i, j] / max_attn, 1.0)
                start = pos_norm[j]
                end = pos_norm[i]
                vec = end - start
                length = np.linalg.norm(vec)
                
                if length > 0:
                    vec_norm = vec / length
                    start_adj = start + vec_norm * 0.25
                    end_adj = end - vec_norm * 0.25
                    
                    ax.annotate(
                        '', xy=end_adj, xytext=start_adj,
                        arrowprops=dict(
                            arrowstyle='->',
                            color='red',
                            alpha=alpha,
                            lw=0.5 + 2*alpha
                        )
                    )
    
    # Wind direction
    ax.annotate(
        '', xy=(center[0] + wind_dx, center[1] + wind_dy),
        xytext=(center[0], center[1]),
        arrowprops=dict(arrowstyle='->', color='green', lw=2)
    )
    
    ax.set_xlim(pos_norm[:, 0].min() - 1.5, pos_norm[:, 0].max() + 1.5)
    ax.set_ylim(pos_norm[:, 1].min() - 1.5, pos_norm[:, 1].max() + 1.5)
    ax.set_aspect('equal')
    ax.set_title('Average (all heads)', fontsize=10, fontweight='bold')
    ax.set_xlabel('x / D', fontsize=8)
    ax.set_ylabel('y / D', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(
        f'Attention Patterns - Layer {layer_idx} - Step {timestep} - Wind {wind_direction:.0f}°',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved attention visualization to {save_path}")
        plt.close()
    else:
        plt.show()


def visualize_attention_heatmaps(
    attention_weights: List[np.ndarray],
    layer_idx: int = -1,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10),
):
    """
    Create heatmap visualization of attention matrices for all heads.
    
    Args:
        attention_weights: List of attention tensors per layer
        layer_idx: Which layer to visualize (-1 = last)
        save_path: Path to save figure
    """
    attn_layer = attention_weights[layer_idx]  # (n_heads, n_turb, n_turb)
    n_heads = attn_layer.shape[0]
    n_turbines = attn_layer.shape[1]
    
    n_cols = min(4, n_heads + 1)
    n_rows = (n_heads + 1 + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_heads > 1 else [axes]
    
    # Plot each head
    for h in range(n_heads):
        ax = axes[h]
        im = ax.imshow(attn_layer[h], cmap='Reds', vmin=0, vmax=attn_layer[h].max())
        ax.set_title(f'Head {h}')
        ax.set_xlabel('Key (source)')
        ax.set_ylabel('Query (target)')
        ax.set_xticks(range(n_turbines))
        ax.set_yticks(range(n_turbines))
        ax.set_xticklabels([f'T{i}' for i in range(n_turbines)], fontsize=8)
        ax.set_yticklabels([f'T{i}' for i in range(n_turbines)], fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Add values if small enough
        if n_turbines <= 6:
            for i in range(n_turbines):
                for j in range(n_turbines):
                    text = ax.text(
                        j, i, f'{attn_layer[h, i, j]:.2f}',
                        ha='center', va='center', fontsize=7,
                        color='white' if attn_layer[h, i, j] > 0.5 else 'black'
                    )
    
    # Average attention
    ax = axes[n_heads]
    attn_avg = attn_layer.mean(axis=0)
    im = ax.imshow(attn_avg, cmap='Reds', vmin=0, vmax=attn_avg.max())
    ax.set_title('Average (all heads)', fontweight='bold')
    ax.set_xlabel('Key (source)')
    ax.set_ylabel('Query (target)')
    ax.set_xticks(range(n_turbines))
    ax.set_yticks(range(n_turbines))
    ax.set_xticklabels([f'T{i}' for i in range(n_turbines)], fontsize=8)
    ax.set_yticklabels([f'T{i}' for i in range(n_turbines)], fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    if n_turbines <= 6:
        for i in range(n_turbines):
            for j in range(n_turbines):
                text = ax.text(
                    j, i, f'{attn_avg[i, j]:.2f}',
                    ha='center', va='center', fontsize=7,
                    color='white' if attn_avg[i, j] > 0.5 else 'black'
                )
    
    # Hide unused subplots
    for h in range(n_heads + 1, len(axes)):
        axes[h].set_visible(False)
    
    plt.suptitle(f'Attention Heatmaps - Layer {layer_idx}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved heatmap visualization to {save_path}")
        plt.close()
    else:
        plt.show()


def analyze_wake_physics(
    attention_weights: np.ndarray,
    positions: np.ndarray,
    wind_direction: float,
    layer_idx: int = -1,
) -> Dict:
    """
    Analyze whether attention patterns align with wake physics.
    
    In wind farm wake physics:
    - Turbines primarily affect downwind turbines (in wake direction)
    - Attention should be stronger from upwind to downwind turbines
    
    Args:
        attention_weights: List of attention per layer
        positions: (n_turbines, 2) turbine positions
        wind_direction: Wind direction in degrees (meteorological convention)
        layer_idx: Which layer to analyze
    
    Returns:
        Dict with analysis results
    """
    attn = attention_weights[layer_idx].mean(axis=0)  # Average over heads
    n_turbines = len(positions)
    
    # Calculate relative positions in wind direction
    wind_rad = np.radians(wind_direction)
    wind_vec = np.array([-np.sin(wind_rad), -np.cos(wind_rad)])
    
    # For each pair, calculate if j is upwind of i
    upwind_attention = []
    downwind_attention = []
    crosswind_attention = []
    
    for i in range(n_turbines):
        for j in range(n_turbines):
            if i == j:
                continue
            
            # Vector from j to i
            vec_ji = positions[i] - positions[j]
            
            # Project onto wind direction
            projection = np.dot(vec_ji, wind_vec)
            
            # Perpendicular distance (crosswind)
            cross = abs(np.cross(vec_ji, wind_vec))
            
            if projection > 0.5:  # i is downwind of j (j affects i)
                upwind_attention.append(attn[i, j])
            elif projection < -0.5:  # i is upwind of j (j doesn't affect i)
                downwind_attention.append(attn[i, j])
            else:  # Crosswind
                crosswind_attention.append(attn[i, j])
    
    results = {
        'upwind_mean': np.mean(upwind_attention) if upwind_attention else 0,
        'downwind_mean': np.mean(downwind_attention) if downwind_attention else 0,
        'crosswind_mean': np.mean(crosswind_attention) if crosswind_attention else 0,
        'upwind_std': np.std(upwind_attention) if upwind_attention else 0,
        'downwind_std': np.std(downwind_attention) if downwind_attention else 0,
        'crosswind_std': np.std(crosswind_attention) if crosswind_attention else 0,
        'ratio_upwind_downwind': (
            np.mean(upwind_attention) / np.mean(downwind_attention)
            if upwind_attention and downwind_attention and np.mean(downwind_attention) > 0
            else 0
        ),
    }
    
    return results


def plot_wake_analysis(
    episode_data: Dict,
    save_dir: str,
    layer_idx: int = -1,
):
    """
    Create plots analyzing wake physics alignment.
    
    Args:
        episode_data: Data from collect_attention_episode
        save_dir: Directory to save plots
        layer_idx: Which layer to analyze
    """
    attention_history = episode_data['attention_history']
    positions = episode_data['positions']
    wind_directions = episode_data['wind_directions']
    
    # Analyze each timestep
    results_over_time = []
    for t, (attn_weights, wd) in enumerate(zip(attention_history, wind_directions)):
        results = analyze_wake_physics(attn_weights, positions, wd, layer_idx)
        results['timestep'] = t
        results['wind_direction'] = wd
        results_over_time.append(results)
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    timesteps = [r['timestep'] for r in results_over_time]
    upwind = [r['upwind_mean'] for r in results_over_time]
    downwind = [r['downwind_mean'] for r in results_over_time]
    crosswind = [r['crosswind_mean'] for r in results_over_time]
    ratio = [r['ratio_upwind_downwind'] for r in results_over_time]
    
    # Mean attention by direction
    ax = axes[0, 0]
    ax.plot(timesteps, upwind, label='Upwind → Downwind', linewidth=2, color='blue')
    ax.plot(timesteps, downwind, label='Downwind → Upwind', linewidth=2, color='red')
    ax.plot(timesteps, crosswind, label='Crosswind', linewidth=2, color='green')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Mean Attention Weight')
    ax.set_title('Attention by Wake Direction')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Ratio
    ax = axes[0, 1]
    ax.plot(timesteps, ratio, linewidth=2, color='purple')
    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Equal attention')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Upwind / Downwind Ratio')
    ax.set_title('Wake Physics Alignment (Higher = Better)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Distribution comparison
    ax = axes[1, 0]
    all_upwind = []
    all_downwind = []
    all_crosswind = []
    
    for attn_weights, wd in zip(attention_history[::10], wind_directions[::10]):  # Sample every 10 steps
        attn = attn_weights[layer_idx].mean(axis=0)
        n_turbines = len(positions)
        wind_rad = np.radians(wd)
        wind_vec = np.array([-np.sin(wind_rad), -np.cos(wind_rad)])
        
        for i in range(n_turbines):
            for j in range(n_turbines):
                if i == j:
                    continue
                vec_ji = positions[i] - positions[j]
                projection = np.dot(vec_ji, wind_vec)
                
                if projection > 0.5:
                    all_upwind.append(attn[i, j])
                elif projection < -0.5:
                    all_downwind.append(attn[i, j])
                else:
                    all_crosswind.append(attn[i, j])
    
    data = [all_upwind, all_downwind, all_crosswind]
    labels = ['Upwind→Down', 'Downwind→Up', 'Crosswind']
    ax.boxplot(data, labels=labels)
    ax.set_ylabel('Attention Weight')
    ax.set_title('Attention Distribution by Direction')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Summary statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = f"""
    Wake Physics Analysis Summary
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    Mean Attention Weights:
      • Upwind → Downwind:  {np.mean(upwind):.4f} ± {np.std(upwind):.4f}
      • Downwind → Upwind:  {np.mean(downwind):.4f} ± {np.std(downwind):.4f}
      • Crosswind:          {np.mean(crosswind):.4f} ± {np.std(crosswind):.4f}
    
    Ratio (Upwind/Downwind): {np.mean(ratio):.2f}
    
    Interpretation:
      {"✓ Model learns wake physics!" if np.mean(ratio) > 1.2 else "✗ Weak wake physics signal"}
      
      Expected: Upwind attention > Downwind attention
                (upwind turbines affect downwind turbines)
    
    Layer: {layer_idx} (last layer)
    """
    
    ax.text(0.1, 0.5, summary_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle('Wake Physics Alignment Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'wake_physics_analysis.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved wake analysis to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize attention patterns for transformer wind farm control')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained actor checkpoint')
    parser.add_argument('--layout', type=str, default='test_layout',
                        choices=['test_layout', 'train_layout'],
                        help='Layout to evaluate on')
    parser.add_argument('--n_episodes', type=int, default=3,
                        help='Number of episodes to visualize')
    parser.add_argument('--timesteps_per_episode', type=int, default=5,
                        help='Number of timesteps to visualize per episode')
    parser.add_argument('--save_dir', type=str, default='attention_viz',
                        help='Directory to save visualizations')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--layer_idx', type=int, default=-1,
                        help='Which transformer layer to visualize (-1 = last)')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create environment
    print(f"\nCreating environment with layout: {args.layout}")
    env = make_env(layout_name=args.layout, seed=42)
    
    # Get observation dimensions
    obs, _ = env.reset()
    obs_dim_per_turbine = obs.shape[1]  # (n_turbines, obs_dim)
    action_dim_per_turbine = 1
    n_turbines = obs.shape[0]
    
    print(f"  n_turbines: {n_turbines}")
    print(f"  obs_dim_per_turbine: {obs_dim_per_turbine}")
    
    # Load trained actor
    print(f"\nLoading model from: {args.model_path}")
    actor = load_trained_actor(
        args.model_path,
        obs_dim_per_turbine,
        action_dim_per_turbine,
        device
    )
    
    # Run episodes and visualize
    print(f"\nCollecting data from {args.n_episodes} episodes...")
    
    for ep in range(args.n_episodes):
        print(f"\nEpisode {ep + 1}/{args.n_episodes}")
        
        # Collect episode data
        episode_data = collect_attention_episode(actor, env, device, max_steps=200)
        
        print(f"  Completed {episode_data['n_steps']} steps")
        print(f"  Total reward: {sum(episode_data['rewards']):.2f}")
        
        # Create episode directory
        ep_dir = os.path.join(args.save_dir, f'episode_{ep}')
        os.makedirs(ep_dir, exist_ok=True)
        
        # Select timesteps to visualize (evenly spaced)
        n_steps = episode_data['n_steps']
        timestep_indices = np.linspace(0, n_steps - 1, args.timesteps_per_episode, dtype=int)
        
        # Visualize selected timesteps
        for idx in timestep_indices:
            print(f"    Visualizing timestep {idx}...")
            
            attn_weights = episode_data['attention_history'][idx]
            wind_dir = episode_data['wind_directions'][idx]
            
            # Layout + attention arrows
            save_path = os.path.join(ep_dir, f'attention_step_{idx:03d}_layout.png')
            visualize_attention_at_timestep(
                attn_weights,
                episode_data['positions'],
                episode_data['rotor_diameter'],
                wind_dir,
                timestep=idx,
                layer_idx=args.layer_idx,
                save_path=save_path,
            )
            
            # Heatmaps
            save_path = os.path.join(ep_dir, f'attention_step_{idx:03d}_heatmap.png')
            visualize_attention_heatmaps(
                attn_weights,
                layer_idx=args.layer_idx,
                save_path=save_path,
            )
        
        # Wake physics analysis
        print(f"    Analyzing wake physics alignment...")
        plot_wake_analysis(episode_data, ep_dir, layer_idx=args.layer_idx)
        
        # Save episode summary
        summary = {
            'episode': ep,
            'n_steps': episode_data['n_steps'],
            'total_reward': float(sum(episode_data['rewards'])),
            'mean_reward': float(np.mean(episode_data['rewards'])),
            'layout': args.layout,
            'n_turbines': n_turbines,
        }
        
        with open(os.path.join(ep_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
    
    print(f"\nVisualization complete! Results saved to: {args.save_dir}")
    print("\nGenerated visualizations:")
    print("  - attention_step_XXX_layout.png: Farm layout with attention arrows")
    print("  - attention_step_XXX_heatmap.png: Attention matrix heatmaps")
    print("  - wake_physics_analysis.png: Analysis of wake physics alignment")


if __name__ == '__main__':
    main()