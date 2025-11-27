"""
Interactive Visualization Utilities for Transformer Wind Farm Control

Lighter-weight utilities for quick visualization and debugging.
Can be used in Jupyter notebooks or standalone.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import torch
from typing import Optional, List, Tuple
import matplotlib.animation as animation
from IPython.display import HTML


def plot_farm_layout(
    positions: np.ndarray,
    rotor_diameter: float,
    yaw_angles: Optional[np.ndarray] = None,
    wind_direction: float = 270.0,
    ax: plt.Axes = None,
    title: str = "Wind Farm Layout",
    figsize: Tuple[int, int] = (10, 8),
):
    """
    Plot wind farm layout with optional yaw angles.
    
    Args:
        positions: (n_turbines, 2) array of positions
        rotor_diameter: Rotor diameter for normalization
        yaw_angles: Optional (n_turbines,) array of yaw angles in degrees
        wind_direction: Wind direction in degrees (met convention)
        ax: Matplotlib axes
        title: Plot title
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    n_turbines = len(positions)
    pos_norm = positions / rotor_diameter
    
    # Wind direction arrow
    wind_rad = np.radians(wind_direction)
    wind_dx = -np.sin(wind_rad)
    wind_dy = -np.cos(wind_rad)
    
    # Plot turbines
    for i in range(n_turbines):
        x, y = pos_norm[i]
        
        # Turbine circle
        circle = Circle((x, y), 0.25, color='steelblue', ec='black', lw=2, zorder=10)
        ax.add_patch(circle)
        
        # Turbine label
        ax.text(x, y, str(i), ha='center', va='center', 
                fontsize=10, fontweight='bold', color='white', zorder=11)
        
        # Yaw direction indicator
        if yaw_angles is not None:
            yaw_rad = np.radians(yaw_angles[i])
            # Yaw is relative to wind direction
            total_angle = wind_rad + yaw_rad
            dx = -np.sin(total_angle) * 0.5
            dy = -np.cos(total_angle) * 0.5
            ax.arrow(x, y, dx, dy, head_width=0.15, head_length=0.1, 
                    fc='orange', ec='darkorange', lw=1.5, zorder=9)
    
    # Wind arrow in corner
    x_range = pos_norm[:, 0].max() - pos_norm[:, 0].min() + 2
    y_range = pos_norm[:, 1].max() - pos_norm[:, 1].min() + 2
    arrow_x = pos_norm[:, 0].min() - 1
    arrow_y = pos_norm[:, 1].max() + 0.5
    
    ax.annotate('', xy=(arrow_x + wind_dx * 1.5, arrow_y + wind_dy * 1.5),
                xytext=(arrow_x, arrow_y),
                arrowprops=dict(arrowstyle='->', color='green', lw=3))
    ax.text(arrow_x + wind_dx * 0.75, arrow_y + wind_dy * 0.75 + 0.4,
            f'Wind {wind_direction:.0f}°', ha='center', fontsize=10, color='darkgreen')
    
    ax.set_xlim(pos_norm[:, 0].min() - 2, pos_norm[:, 0].max() + 2)
    ax.set_ylim(pos_norm[:, 1].min() - 2, pos_norm[:, 1].max() + 2)
    ax.set_aspect('equal')
    ax.set_xlabel('x / D')
    ax.set_ylabel('y / D')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_attention_simple(
    attention: np.ndarray,
    positions: np.ndarray,
    rotor_diameter: float,
    wind_direction: float = 270.0,
    figsize: Tuple[int, int] = (12, 5),
    threshold: float = 0.05,
):
    """
    Simple side-by-side plot: farm layout + attention heatmap.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    n_turbines = len(positions)
    pos_norm = positions / rotor_diameter
    
    # Left: Farm layout with attention arrows
    ax1 = axes[0]
    
    # Plot turbines
    for i in range(n_turbines):
        circle = Circle((pos_norm[i, 0], pos_norm[i, 1]), 0.25, 
                        color='steelblue', ec='black', lw=2, zorder=10)
        ax1.add_patch(circle)
        ax1.text(pos_norm[i, 0], pos_norm[i, 1], str(i),
                ha='center', va='center', fontsize=10, 
                fontweight='bold', color='white', zorder=11)
    
    # Attention arrows
    max_attn = attention.max()
    for i in range(n_turbines):
        for j in range(n_turbines):
            if i != j and attention[i, j] > threshold:
                alpha = attention[i, j] / max_attn
                
                # Arrow from j to i (info flows from j to i)
                start = pos_norm[j]
                end = pos_norm[i]
                
                # Shorten
                vec = end - start
                length = np.linalg.norm(vec)
                vec_norm = vec / length
                start_adj = start + vec_norm * 0.3
                end_adj = end - vec_norm * 0.3
                
                ax1.annotate('', xy=end_adj, xytext=start_adj,
                            arrowprops=dict(arrowstyle='->', color='red', 
                                          alpha=alpha, lw=1 + 2*alpha))
    
    ax1.set_xlim(pos_norm[:, 0].min() - 1.5, pos_norm[:, 0].max() + 1.5)
    ax1.set_ylim(pos_norm[:, 1].min() - 1.5, pos_norm[:, 1].max() + 1.5)
    ax1.set_aspect('equal')
    ax1.set_xlabel('x / D')
    ax1.set_ylabel('y / D')
    ax1.set_title('Attention on Farm Layout')
    ax1.grid(True, alpha=0.3)
    
    # Right: Heatmap
    ax2 = axes[1]
    im = ax2.imshow(attention, cmap='Reds', vmin=0, vmax=max_attn)
    ax2.set_xticks(range(n_turbines))
    ax2.set_yticks(range(n_turbines))
    ax2.set_xticklabels([f'T{i}' for i in range(n_turbines)])
    ax2.set_yticklabels([f'T{i}' for i in range(n_turbines)])
    ax2.set_xlabel('Key (source)')
    ax2.set_ylabel('Query (target)')
    ax2.set_title('Attention Matrix')
    plt.colorbar(im, ax=ax2, label='Attention Weight')
    
    # Add values to heatmap if small enough
    if n_turbines <= 6:
        for i in range(n_turbines):
            for j in range(n_turbines):
                ax2.text(j, i, f'{attention[i,j]:.2f}', 
                        ha='center', va='center', fontsize=8)
    
    plt.tight_layout()
    return fig


def plot_attention_by_head(
    attention_per_head: np.ndarray,
    positions: np.ndarray,
    rotor_diameter: float,
    figsize_per_head: Tuple[int, int] = (5, 4),
):
    """
    Plot attention patterns for each attention head separately.
    
    Args:
        attention_per_head: (n_heads, n_turbines, n_turbines)
    """
    n_heads = attention_per_head.shape[0]
    n_cols = min(4, n_heads)
    n_rows = (n_heads + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, 
                             figsize=(figsize_per_head[0]*n_cols, figsize_per_head[1]*n_rows))
    axes = np.atleast_2d(axes).flatten()
    
    for h in range(n_heads):
        ax = axes[h]
        im = ax.imshow(attention_per_head[h], cmap='Reds', vmin=0)
        ax.set_title(f'Head {h}')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    for h in range(n_heads, len(axes)):
        axes[h].set_visible(False)
    
    plt.suptitle('Attention by Head', fontsize=12)
    plt.tight_layout()
    return fig


def create_attention_animation(
    attention_history: List[np.ndarray],
    positions: np.ndarray,
    rotor_diameter: float,
    wind_directions: Optional[List[float]] = None,
    interval: int = 200,
    figsize: Tuple[int, int] = (10, 8),
):
    """
    Create an animation of attention patterns over time.
    
    Args:
        attention_history: List of (n_turbines, n_turbines) attention matrices
        positions: (n_turbines, 2) turbine positions
        rotor_diameter: For normalization
        wind_directions: Optional list of wind directions per timestep
        interval: Milliseconds between frames
    
    Returns:
        matplotlib animation object (use HTML(anim.to_jshtml()) in Jupyter)
    """
    n_steps = len(attention_history)
    n_turbines = len(positions)
    pos_norm = positions / rotor_diameter
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Initial setup
    turbine_circles = []
    for i in range(n_turbines):
        circle = Circle((pos_norm[i, 0], pos_norm[i, 1]), 0.25,
                        color='steelblue', ec='black', lw=2, zorder=10)
        ax.add_patch(circle)
        turbine_circles.append(circle)
        ax.text(pos_norm[i, 0], pos_norm[i, 1], str(i),
               ha='center', va='center', fontsize=10,
               fontweight='bold', color='white', zorder=11)
    
    ax.set_xlim(pos_norm[:, 0].min() - 2, pos_norm[:, 0].max() + 2)
    ax.set_ylim(pos_norm[:, 1].min() - 2, pos_norm[:, 1].max() + 2)
    ax.set_aspect('equal')
    ax.set_xlabel('x / D')
    ax.set_ylabel('y / D')
    ax.grid(True, alpha=0.3)
    
    title = ax.set_title('Step 0')
    
    # Store arrow artists
    arrows = []
    
    def update(frame):
        nonlocal arrows
        
        # Remove old arrows
        for arrow in arrows:
            arrow.remove()
        arrows = []
        
        attention = attention_history[frame]
        max_attn = attention.max()
        
        # Draw new arrows
        for i in range(n_turbines):
            for j in range(n_turbines):
                if i != j and attention[i, j] > 0.05:
                    alpha = attention[i, j] / max_attn
                    
                    start = pos_norm[j]
                    end = pos_norm[i]
                    vec = end - start
                    length = np.linalg.norm(vec)
                    vec_norm = vec / length
                    start_adj = start + vec_norm * 0.3
                    end_adj = end - vec_norm * 0.3
                    
                    arrow = FancyArrowPatch(
                        start_adj, end_adj,
                        arrowstyle='->,head_width=0.1,head_length=0.05',
                        color='red',
                        alpha=alpha,
                        lw=1 + 2*alpha,
                        zorder=5
                    )
                    ax.add_patch(arrow)
                    arrows.append(arrow)
        
        title_text = f'Step {frame}'
        if wind_directions is not None:
            title_text += f' | Wind: {wind_directions[frame]:.1f}°'
        title.set_text(title_text)
        
        return arrows + [title]
    
    anim = animation.FuncAnimation(
        fig, update, frames=n_steps, interval=interval, blit=False
    )
    
    return anim


def compare_attention_patterns(
    attention_dict: dict,
    positions_dict: dict,
    rotor_diameters: dict,
    figsize_per: Tuple[int, int] = (5, 4),
):
    """
    Compare attention patterns across different layouts/conditions.
    
    Args:
        attention_dict: {name: attention_matrix}
        positions_dict: {name: positions}
        rotor_diameters: {name: rotor_d}
    """
    names = list(attention_dict.keys())
    n = len(names)
    
    fig, axes = plt.subplots(1, n, figsize=(figsize_per[0]*n, figsize_per[1]))
    if n == 1:
        axes = [axes]
    
    for idx, name in enumerate(names):
        ax = axes[idx]
        attention = attention_dict[name]
        positions = positions_dict[name]
        rotor_d = rotor_diameters[name]
        
        n_turb = len(positions)
        pos_norm = positions / rotor_d
        
        # Plot turbines
        for i in range(n_turb):
            circle = Circle((pos_norm[i, 0], pos_norm[i, 1]), 0.25,
                           color='steelblue', ec='black', lw=2, zorder=10)
            ax.add_patch(circle)
            ax.text(pos_norm[i, 0], pos_norm[i, 1], str(i),
                   ha='center', va='center', fontsize=8,
                   fontweight='bold', color='white', zorder=11)
        
        # Attention arrows
        max_attn = attention.max()
        for i in range(n_turb):
            for j in range(n_turb):
                if i != j and attention[i, j] > 0.05:
                    alpha = attention[i, j] / max_attn
                    start = pos_norm[j]
                    end = pos_norm[i]
                    vec = end - start
                    length = np.linalg.norm(vec)
                    if length > 0:
                        vec_norm = vec / length
                        start_adj = start + vec_norm * 0.3
                        end_adj = end - vec_norm * 0.3
                        ax.annotate('', xy=end_adj, xytext=start_adj,
                                   arrowprops=dict(arrowstyle='->',
                                                  color='red', alpha=alpha,
                                                  lw=1+2*alpha))
        
        ax.set_xlim(pos_norm[:, 0].min() - 1.5, pos_norm[:, 0].max() + 1.5)
        ax.set_ylim(pos_norm[:, 1].min() - 1.5, pos_norm[:, 1].max() + 1.5)
        ax.set_aspect('equal')
        ax.set_title(name)
        ax.set_xlabel('x / D')
        if idx == 0:
            ax.set_ylabel('y / D')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def quick_eval_step(
    actor,
    obs: np.ndarray,
    positions: np.ndarray,
    rotor_diameter: float,
    device: torch.device,
    wind_direction: float = 270.0,
):
    """
    Quick visualization of a single forward pass through the actor.
    
    Useful for debugging and quick inspection.
    """
    actor.eval()
    
    with torch.no_grad():
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        pos_tensor = torch.tensor(positions / rotor_diameter, dtype=torch.float32, device=device).unsqueeze(0)
        
        action, log_prob, mean_action, attn_weights = actor.get_action(
            obs_tensor, pos_tensor, deterministic=True
        )
    
    # Get attention from last layer, average over heads
    attn_last = attn_weights[-1].squeeze(0).cpu().numpy()  # (n_heads, n_turb, n_turb)
    attn_avg = attn_last.mean(axis=0)
    
    # Create visualization
    fig = plt.figure(figsize=(15, 5))
    
    # 1. Farm with attention
    ax1 = fig.add_subplot(131)
    n_turb = len(positions)
    pos_norm = positions / rotor_diameter
    
    for i in range(n_turb):
        circle = Circle((pos_norm[i, 0], pos_norm[i, 1]), 0.25,
                        color='steelblue', ec='black', lw=2, zorder=10)
        ax1.add_patch(circle)
        ax1.text(pos_norm[i, 0], pos_norm[i, 1], str(i),
                ha='center', va='center', fontsize=10,
                fontweight='bold', color='white', zorder=11)
    
    max_attn = attn_avg.max()
    for i in range(n_turb):
        for j in range(n_turb):
            if i != j and attn_avg[i, j] > 0.05:
                alpha = attn_avg[i, j] / max_attn
                start = pos_norm[j]
                end = pos_norm[i]
                vec = end - start
                length = np.linalg.norm(vec)
                vec_norm = vec / length
                ax1.annotate('', xy=end - vec_norm*0.3, xytext=start + vec_norm*0.3,
                            arrowprops=dict(arrowstyle='->', color='red',
                                          alpha=alpha, lw=1+2*alpha))
    
    ax1.set_xlim(pos_norm[:, 0].min() - 1.5, pos_norm[:, 0].max() + 1.5)
    ax1.set_ylim(pos_norm[:, 1].min() - 1.5, pos_norm[:, 1].max() + 1.5)
    ax1.set_aspect('equal')
    ax1.set_title('Attention Pattern')
    ax1.grid(True, alpha=0.3)
    
    # 2. Attention heatmap
    ax2 = fig.add_subplot(132)
    im = ax2.imshow(attn_avg, cmap='Reds')
    ax2.set_xticks(range(n_turb))
    ax2.set_yticks(range(n_turb))
    ax2.set_title('Attention Matrix (avg over heads)')
    plt.colorbar(im, ax=ax2)
    
    # 3. Actions
    ax3 = fig.add_subplot(133)
    actions = mean_action.squeeze().cpu().numpy().flatten()
    ax3.bar(range(n_turb), actions, color='seagreen')
    ax3.set_xticks(range(n_turb))
    ax3.set_xticklabels([f'T{i}' for i in range(n_turb)])
    ax3.set_ylabel('Action (yaw)')
    ax3.set_title('Policy Output')
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    return fig, {
        'action': action.squeeze().cpu().numpy(),
        'attention_avg': attn_avg,
        'attention_per_head': attn_last,
    }


# Example usage in notebook:
"""
# Load model and env
actor, qf1, qf2 = load_model(...)
env = make_env(...)
obs, _ = env.reset()
positions = env.turbine_positions
rotor_d = env.rotor_diameter

# Quick visualization
fig, info = quick_eval_step(actor, obs, positions, rotor_d, device)
plt.show()

# Animation over episode
attention_history = []
obs, _ = env.reset()
for _ in range(100):
    with torch.no_grad():
        obs_t = torch.tensor(obs, device=device).unsqueeze(0)
        pos_t = torch.tensor(positions/rotor_d, device=device).unsqueeze(0)
        action, _, _, attn = actor.get_action(obs_t, pos_t)
        attention_history.append(attn[-1].squeeze().mean(0).cpu().numpy())
    obs, _, done, _, _ = env.step(action.squeeze().cpu().numpy().flatten())
    if done:
        break

anim = create_attention_animation(attention_history, positions, rotor_d)
HTML(anim.to_jshtml())  # In Jupyter
"""