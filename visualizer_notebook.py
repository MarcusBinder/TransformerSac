"""
Attention Visualization Helpers - Notebook Version

Simple functions for interactive attention visualization in Jupyter notebooks.
Can be imported and used with your trained model.

Example usage in notebook:
    
    from attention_viz_notebook import AttentionVisualizer
    
    # Load your model and environment
    viz = AttentionVisualizer(actor, device)
    
    # Get attention for current observation
    result = viz.get_attention(obs, positions, rotor_diameter, wind_direction)
    
    # Plot it
    viz.plot(result)
    
    # Or run an episode and visualize
    viz.run_episode(env, n_steps=50, save_animation=True)
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
from typing import Optional, List, Dict, Tuple, Any
import math


class AttentionVisualizer:
    """Helper class for visualizing attention patterns from a trained transformer actor."""
    
    def __init__(self, actor, device: torch.device = None):
        """
        Args:
            actor: Trained TransformerActor from v9
            device: Torch device (defaults to actor's device)
        """
        self.actor = actor
        self.actor.eval()
        
        if device is None:
            # Try to infer device from actor parameters
            device = next(actor.parameters()).device
        self.device = device
    
    def transform_to_wind_relative(
        self,
        positions: torch.Tensor,
        wind_direction: float
    ) -> torch.Tensor:
        """Transform positions to wind-relative coordinates."""
        angle_offset = wind_direction - 270.0
        theta = torch.tensor(angle_offset * (math.pi / 180.0), device=self.device)
        
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        
        x = positions[..., 0:1]
        y = positions[..., 1:2]
        
        x_rot = cos_theta * x - sin_theta * y
        y_rot = sin_theta * x + cos_theta * y
        
        return torch.cat([x_rot, y_rot], dim=-1)
    
    def get_attention(
        self,
        obs: np.ndarray,
        positions: np.ndarray,
        rotor_diameter: float,
        wind_direction: float,
        deterministic: bool = True,
    ) -> Dict[str, Any]:
        """
        Get attention weights for a single observation.
        
        Args:
            obs: Per-turbine observations, shape (n_turbines, obs_dim)
            positions: Raw turbine positions, shape (n_turbines, 2)
            rotor_diameter: Rotor diameter in meters
            wind_direction: Wind direction in degrees
            deterministic: Whether to use deterministic policy
        
        Returns:
            Dictionary with attention data and actions
        """
        with torch.no_grad():
            # Prepare inputs
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            pos_norm = positions / rotor_diameter
            pos_t = torch.tensor(pos_norm, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            # Transform to wind-relative coordinates
            pos_transformed = self.transform_to_wind_relative(pos_t, wind_direction)
            
            # Get action and attention
            action, log_prob, mean_action, attn_weights = self.actor.get_action(
                obs_t, pos_transformed, deterministic=deterministic
            )
        
        # Extract attention from last layer
        attn_last = attn_weights[-1][0].cpu().numpy()  # (n_heads, n_turb, n_turb)
        
        return {
            'attention_per_head': attn_last,
            'attention_avg': attn_last.mean(axis=0),
            'action': action[0].cpu().numpy().flatten(),
            'mean_action': mean_action[0].cpu().numpy().flatten(),
            'positions': positions,
            'rotor_diameter': rotor_diameter,
            'wind_direction': wind_direction,
            'all_layer_attention': [w[0].cpu().numpy() for w in attn_weights],
        }
    
    def plot(
        self,
        result: Dict[str, Any],
        figsize: Tuple[int, int] = (14, 5),
        threshold: float = 0.05,
    ) -> plt.Figure:
        """
        Plot attention pattern with heatmap and farm overlay.
        
        Args:
            result: Output from get_attention()
            figsize: Figure size
            threshold: Minimum attention to show arrows
        """
        attention = result['attention_avg']
        positions = result['positions']
        rotor_diameter = result['rotor_diameter']
        wind_direction = result['wind_direction']
        actions = result['mean_action']
        
        n_turbines = len(positions)
        pos_norm = positions / rotor_diameter
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # === Plot 1: Farm with attention arrows ===
        ax1 = axes[0]
        max_attn = attention.max()
        
        # Draw attention arrows
        for i in range(n_turbines):
            for j in range(n_turbines):
                if i != j and attention[i, j] > threshold:
                    alpha = min(attention[i, j] / max_attn, 1.0)
                    start = pos_norm[j]
                    end = pos_norm[i]
                    vec = end - start
                    length = np.linalg.norm(vec)
                    if length > 0:
                        vec_n = vec / length
                        ax1.annotate(
                            '', xy=end - vec_n * 0.3, xytext=start + vec_n * 0.3,
                            arrowprops=dict(arrowstyle='->', color='crimson',
                                          alpha=alpha * 0.8, lw=1 + 3 * alpha)
                        )
        
        # Draw turbines
        for i in range(n_turbines):
            circle = Circle(pos_norm[i], 0.25, color='steelblue', ec='black', lw=2, zorder=10)
            ax1.add_patch(circle)
            ax1.text(pos_norm[i, 0], pos_norm[i, 1], str(i),
                    ha='center', va='center', fontsize=10, fontweight='bold', color='white', zorder=11)
        
        # Wind arrow
        wind_rad = np.radians(270 - wind_direction)
        center = pos_norm.mean(axis=0)
        arrow_len = np.max(np.ptp(pos_norm, axis=0)) * 0.25
        dx, dy = arrow_len * np.cos(wind_rad), arrow_len * np.sin(wind_rad)
        ax1.annotate('', xy=center, xytext=(center[0] - dx * 2, center[1] - dy * 2),
                    arrowprops=dict(arrowstyle='->', color='green', lw=3))
        ax1.text(center[0] - dx * 2.5, center[1] - dy * 2.5 + 0.3,
                f'Wind {wind_direction:.0f}°', ha='center', fontsize=9, color='darkgreen')
        
        ax1.set_xlim(pos_norm[:, 0].min() - 1.5, pos_norm[:, 0].max() + 1.5)
        ax1.set_ylim(pos_norm[:, 1].min() - 1.5, pos_norm[:, 1].max() + 1.5)
        ax1.set_aspect('equal')
        ax1.set_xlabel('x / D')
        ax1.set_ylabel('y / D')
        ax1.set_title('Attention Pattern')
        ax1.grid(True, alpha=0.3)
        
        # === Plot 2: Attention heatmap ===
        ax2 = axes[1]
        im = ax2.imshow(attention, cmap='Reds', vmin=0)
        ax2.set_xticks(range(n_turbines))
        ax2.set_yticks(range(n_turbines))
        ax2.set_xticklabels([f'T{i}' for i in range(n_turbines)])
        ax2.set_yticklabels([f'T{i}' for i in range(n_turbines)])
        ax2.set_xlabel('Key (source)')
        ax2.set_ylabel('Query (target)')
        ax2.set_title('Attention Heatmap')
        plt.colorbar(im, ax=ax2, shrink=0.8)
        
        if n_turbines <= 6:
            for i in range(n_turbines):
                for j in range(n_turbines):
                    ax2.text(j, i, f'{attention[i, j]:.2f}', ha='center', va='center', fontsize=8)
        
        # === Plot 3: Actions ===
        ax3 = axes[2]
        bars = ax3.bar(range(n_turbines), actions, color='seagreen', edgecolor='black')
        ax3.set_xticks(range(n_turbines))
        ax3.set_xticklabels([f'T{i}' for i in range(n_turbines)])
        ax3.set_ylabel('Yaw Action')
        ax3.set_title('Policy Output')
        ax3.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax3.set_ylim(-1.1, 1.1)
        
        plt.tight_layout()
        return fig
    
    def plot_per_head(
        self,
        result: Dict[str, Any],
        figsize_per_head: Tuple[int, int] = (4, 4),
    ) -> plt.Figure:
        """Plot attention for each head separately."""
        attn_per_head = result['attention_per_head']
        n_heads = attn_per_head.shape[0]
        n_turbines = attn_per_head.shape[1]
        
        n_cols = min(4, n_heads)
        n_rows = (n_heads + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, 
                                figsize=(figsize_per_head[0] * n_cols, figsize_per_head[1] * n_rows))
        axes = np.atleast_2d(axes).flatten()
        
        for h in range(n_heads):
            ax = axes[h]
            im = ax.imshow(attn_per_head[h], cmap='Reds', vmin=0)
            ax.set_title(f'Head {h}')
            ax.set_xticks(range(n_turbines))
            ax.set_yticks(range(n_turbines))
            plt.colorbar(im, ax=ax, shrink=0.8)
        
        for h in range(n_heads, len(axes)):
            axes[h].set_visible(False)
        
        plt.suptitle('Attention by Head', fontsize=12)
        plt.tight_layout()
        return fig
    
    def analyze_wake_alignment(
        self,
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Analyze whether attention aligns with wake physics."""
        attention = result['attention_avg']
        positions = result['positions']
        rotor_diameter = result['rotor_diameter']
        wind_direction = result['wind_direction']
        
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
        
        return {
            'upwind_mean': np.mean(upwind) if upwind else 0,
            'downwind_mean': np.mean(downwind) if downwind else 0,
            'crosswind_mean': np.mean(crosswind) if crosswind else 0,
            'ratio': (np.mean(upwind) / np.mean(downwind)) if downwind and np.mean(downwind) > 0 else float('inf'),
            'n_upwind': len(upwind),
            'n_downwind': len(downwind),
            'n_crosswind': len(crosswind),
        }
    
    def run_episode(
        self,
        env,
        n_steps: int = 100,
        deterministic: bool = True,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run an episode and collect attention data.
        
        Args:
            env: WindGym environment (wrapped with PerTurbineObservationWrapper)
            n_steps: Maximum steps to run
            deterministic: Use deterministic policy
            verbose: Print progress
        
        Returns:
            Dictionary with attention history and episode data
        """
        obs, info = env.reset()
        
        # Get environment properties
        positions = env.turbine_positions
        rotor_diameter = env.rotor_diameter
        wind_direction = getattr(env.env, 'wd', 270.0)  # Get from base env
        
        attention_history = []
        actions_history = []
        rewards = []
        
        for step in range(n_steps):
            result = self.get_attention(obs, positions, rotor_diameter, wind_direction, deterministic)
            attention_history.append(result['attention_avg'])
            actions_history.append(result['action'])
            
            obs, reward, terminated, truncated, info = env.step(result['action'])
            rewards.append(reward)
            
            if verbose and step % 10 == 0:
                print(f"Step {step}: reward={reward:.4f}")
            
            if terminated or truncated:
                break
        
        if verbose:
            print(f"Episode finished after {len(rewards)} steps, total reward: {sum(rewards):.2f}")
        
        return {
            'attention_history': attention_history,
            'actions_history': actions_history,
            'rewards': rewards,
            'positions': positions,
            'rotor_diameter': rotor_diameter,
            'wind_direction': wind_direction,
        }
    
    def plot_attention_evolution(
        self,
        episode_data: Dict[str, Any],
        n_snapshots: int = 6,
        figsize: Tuple[int, int] = (15, 10),
    ) -> plt.Figure:
        """Plot attention at multiple timesteps during episode."""
        attention_history = episode_data['attention_history']
        positions = episode_data['positions']
        rotor_diameter = episode_data['rotor_diameter']
        wind_direction = episode_data['wind_direction']
        
        n_steps = len(attention_history)
        timesteps = np.linspace(0, n_steps - 1, n_snapshots, dtype=int)
        
        n_cols = min(3, n_snapshots)
        n_rows = (n_snapshots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = np.atleast_2d(axes).flatten()
        
        pos_norm = positions / rotor_diameter
        n_turbines = len(positions)
        
        for idx, t in enumerate(timesteps):
            ax = axes[idx]
            attention = attention_history[t]
            max_attn = attention.max()
            
            # Attention arrows
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
                       ha='center', va='center', fontsize=9, fontweight='bold', color='white', zorder=11)
            
            ax.set_xlim(pos_norm[:, 0].min() - 1.5, pos_norm[:, 0].max() + 1.5)
            ax.set_ylim(pos_norm[:, 1].min() - 1.5, pos_norm[:, 1].max() + 1.5)
            ax.set_aspect('equal')
            ax.set_title(f'Step {t}')
            ax.grid(True, alpha=0.3)
        
        for idx in range(n_snapshots, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Attention Evolution Over Episode', fontsize=14)
        plt.tight_layout()
        return fig


# =============================================================================
# Quick usage functions
# =============================================================================

def quick_attention_plot(
    actor,
    obs: np.ndarray,
    positions: np.ndarray,
    rotor_diameter: float,
    wind_direction: float,
    device: torch.device = None,
) -> plt.Figure:
    """
    One-liner to visualize attention for a single observation.
    
    Example:
        fig = quick_attention_plot(actor, obs, positions, rotor_d, wind_dir)
        plt.show()
    """
    viz = AttentionVisualizer(actor, device)
    result = viz.get_attention(obs, positions, rotor_diameter, wind_direction)
    return viz.plot(result)


def load_actor_from_checkpoint(checkpoint_path: str, device: torch.device = None):
    """
    Load a trained actor from v9 checkpoint.
    
    This function is self-contained with the model definition.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Import model definition (assumes this file or transformer_sac_windfarm_v9 is available)
    # For standalone use, copy the TransformerActor class definition here or import it
    
    try:
        from transformer_sac_windfarm_v9 import TransformerActor
    except ImportError:
        # Use the definition from visualize_attention.py
        from visualize_attention import TransformerActor
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint.get('args', {})
    actor_state = checkpoint['actor_state_dict']
    
    # Infer obs_dim from weights
    obs_dim = actor_state['obs_encoder.0.weight'].shape[1]
    
    actor = TransformerActor(
        obs_dim_per_turbine=obs_dim,
        action_dim_per_turbine=1,
        embed_dim=args.get('embed_dim', 128),
        pos_embed_dim=args.get('pos_embed_dim', 32),
        num_heads=args.get('num_heads', 4),
        num_layers=args.get('num_layers', 2),
        mlp_ratio=args.get('mlp_ratio', 2.0),
        dropout=0.0,
        use_farm_token=args.get('use_farm_token', False),
    ).to(device)
    
    actor.load_state_dict(actor_state)
    actor.eval()
    
    print(f"Loaded actor from step {checkpoint.get('step', 'unknown')}")
    return actor


# =============================================================================
# Example usage
# =============================================================================

EXAMPLE_USAGE = '''
# Example usage in Jupyter notebook:

# 1. Load model
from attention_viz_notebook import AttentionVisualizer, load_actor_from_checkpoint

actor = load_actor_from_checkpoint('runs/YOUR_RUN/checkpoints/step_100000.pt')
device = next(actor.parameters()).device

# 2. Create visualizer
viz = AttentionVisualizer(actor, device)

# 3. With an environment
from WindGym import WindFarmEnv
from WindGym.wrappers import PerTurbineObservationWrapper

env = ...  # Your wrapped environment
obs, info = env.reset()
positions = env.turbine_positions
rotor_d = env.rotor_diameter
wind_dir = env.env.wd

# Get and plot attention
result = viz.get_attention(obs, positions, rotor_d, wind_dir)
fig = viz.plot(result)
plt.show()

# Analyze wake alignment
analysis = viz.analyze_wake_alignment(result)
print(f"Upwind/Downwind ratio: {analysis['ratio']:.2f}")

# Run full episode
episode = viz.run_episode(env, n_steps=100)
fig = viz.plot_attention_evolution(episode)
plt.show()
'''

if __name__ == '__main__':
    print(EXAMPLE_USAGE)