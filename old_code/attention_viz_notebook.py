"""
Interactive Attention Visualization for Jupyter Notebooks

This script provides simple functions to visualize attention patterns
in Jupyter notebooks for quick debugging and exploration.

Example usage:
    from attention_viz_notebook import AttentionVisualizer
    
    # Create visualizer
    viz = AttentionVisualizer(actor, env, device)
    
    # Single step visualization
    viz.visualize_step()
    
    # Episode visualization
    viz.run_episode_with_viz(n_steps=50, plot_every=10)
    
    # Wake physics analysis
    viz.analyze_wake_physics()

Author: Marcus (DTU Wind Energy)
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from typing import Optional, List, Tuple, Dict
from IPython.display import clear_output
import time


class AttentionVisualizer:
    """
    Helper class for visualizing attention patterns in Jupyter notebooks.
    """
    
    def __init__(
        self,
        actor,
        env,
        device: torch.device,
        layer_idx: int = -1,
        head_idx: int = 0,
    ):
        """
        Args:
            actor: Trained TransformerActor model
            env: Wind farm environment (with PerTurbineObservationWrapper)
            device: torch device
            layer_idx: Which transformer layer to visualize (-1 = last)
            head_idx: Which attention head to visualize (or -1 for average)
        """
        self.actor = actor
        self.env = env
        self.device = device
        self.layer_idx = layer_idx
        self.head_idx = head_idx
        
        # Get environment properties
        obs, _ = env.reset()
        self.n_turbines = obs.shape[0]
        self.obs_dim = obs.shape[1]
        self.positions = env.turbine_positions
        self.rotor_diameter = env.rotor_diameter
        
        # History storage
        self.attention_history = []
        self.reward_history = []
        self.action_history = []
        self.wind_dir_history = []
        
        print(f"AttentionVisualizer initialized:")
        print(f"  n_turbines: {self.n_turbines}")
        print(f"  obs_dim_per_turbine: {self.obs_dim}")
        print(f"  rotor_diameter: {self.rotor_diameter}")
    
    def _get_wind_relative_positions(self, wind_dir: float) -> np.ndarray:
        """Transform positions to wind-relative coordinates."""
        pos_normalized = self.positions / self.rotor_diameter
        wind_rad = np.radians(wind_dir)
        cos_wd = np.cos(wind_rad)
        sin_wd = np.sin(wind_rad)
        rotation_matrix = np.array([
            [cos_wd, sin_wd],
            [-sin_wd, cos_wd]
        ])
        return pos_normalized @ rotation_matrix.T
    
    def get_attention_and_action(
        self,
        obs: np.ndarray,
        wind_dir: float,
        deterministic: bool = True,
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Get attention weights and action for a given observation.
        
        Returns:
            attention_weights: List of (n_heads, n_turb, n_turb) per layer
            action: (n_turbines, action_dim) action array
        """
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        pos_wind_rel = self._get_wind_relative_positions(wind_dir)
        pos_tensor = torch.tensor(pos_wind_rel, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            action, _, _, attn_weights = self.actor.get_action(
                obs_tensor, pos_tensor, deterministic=deterministic
            )
        
        # Convert to numpy
        attn_numpy = [attn.squeeze(0).cpu().numpy() for attn in attn_weights]
        action_numpy = action.squeeze(0).cpu().numpy()
        
        return attn_numpy, action_numpy
    
    def plot_attention_single(
        self,
        attention_weights: List[np.ndarray],
        wind_dir: float,
        action: Optional[np.ndarray] = None,
        threshold: float = 0.05,
        figsize: Tuple[int, int] = (16, 5),
    ):
        """
        Plot attention pattern for current state.
        
        Args:
            attention_weights: List of attention per layer
            wind_dir: Wind direction in degrees
            action: Optional actions to display
            threshold: Minimum attention to display
        """
        # Get attention for specified layer
        if self.head_idx == -1:
            # Average over heads
            attn = attention_weights[self.layer_idx].mean(axis=0)
            head_label = "avg"
        else:
            attn = attention_weights[self.layer_idx][self.head_idx]
            head_label = f"head {self.head_idx}"
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        pos_norm = self.positions / self.rotor_diameter
        
        # ===== Plot 1: Farm layout with attention =====
        ax = axes[0]
        
        # Draw turbines
        for i in range(self.n_turbines):
            circle = Circle(
                (pos_norm[i, 0], pos_norm[i, 1]), 0.25,
                color='steelblue', ec='black', lw=2, zorder=10
            )
            ax.add_patch(circle)
            ax.text(
                pos_norm[i, 0], pos_norm[i, 1], str(i),
                ha='center', va='center', fontsize=10,
                fontweight='bold', color='white', zorder=11
            )
        
        # Draw attention arrows
        max_attn = attn.max()
        for i in range(self.n_turbines):
            for j in range(self.n_turbines):
                if i != j and attn[i, j] > threshold:
                    alpha = min(attn[i, j] / max_attn, 1.0)
                    
                    start = pos_norm[j]
                    end = pos_norm[i]
                    vec = end - start
                    length = np.linalg.norm(vec)
                    
                    if length > 0:
                        vec_norm = vec / length
                        start_adj = start + vec_norm * 0.3
                        end_adj = end - vec_norm * 0.3
                        
                        ax.annotate(
                            '', xy=end_adj, xytext=start_adj,
                            arrowprops=dict(
                                arrowstyle='->',
                                color='red',
                                alpha=alpha,
                                lw=0.5 + 2.5*alpha
                            )
                        )
        
        # Wind direction arrow
        center = pos_norm.mean(axis=0)
        wind_rad = np.radians(wind_dir)
        wind_dx = -np.sin(wind_rad) * 1.2
        wind_dy = -np.cos(wind_rad) * 1.2
        ax.annotate(
            '', xy=(center[0] + wind_dx, center[1] + wind_dy),
            xytext=(center[0], center[1]),
            arrowprops=dict(arrowstyle='->', color='green', lw=3)
        )
        ax.text(
            center[0] + wind_dx * 0.5, center[1] + wind_dy * 0.5 + 0.5,
            f'Wind {wind_dir:.0f}°', ha='center', fontsize=10,
            color='darkgreen', fontweight='bold'
        )
        
        ax.set_xlim(pos_norm[:, 0].min() - 2, pos_norm[:, 0].max() + 2)
        ax.set_ylim(pos_norm[:, 1].min() - 2, pos_norm[:, 1].max() + 2)
        ax.set_aspect('equal')
        ax.set_xlabel('x / D', fontsize=11)
        ax.set_ylabel('y / D', fontsize=11)
        ax.set_title(f'Attention Pattern (layer {self.layer_idx}, {head_label})', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # ===== Plot 2: Attention heatmap =====
        ax = axes[1]
        im = ax.imshow(attn, cmap='Reds', vmin=0)
        ax.set_xlabel('Key (source)', fontsize=11)
        ax.set_ylabel('Query (target)', fontsize=11)
        ax.set_title('Attention Matrix', fontsize=12)
        ax.set_xticks(range(self.n_turbines))
        ax.set_yticks(range(self.n_turbines))
        ax.set_xticklabels([f'T{i}' for i in range(self.n_turbines)])
        ax.set_yticklabels([f'T{i}' for i in range(self.n_turbines)])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Add values if small farm
        if self.n_turbines <= 6:
            for i in range(self.n_turbines):
                for j in range(self.n_turbines):
                    color = 'white' if attn[i, j] > attn.max() * 0.5 else 'black'
                    ax.text(
                        j, i, f'{attn[i, j]:.2f}',
                        ha='center', va='center', fontsize=9, color=color
                    )
        
        # ===== Plot 3: Actions =====
        ax = axes[2]
        if action is not None:
            actions_flat = action.flatten()
            colors = ['green' if a > 0 else 'red' for a in actions_flat]
            ax.bar(range(self.n_turbines), actions_flat, color=colors, alpha=0.7, edgecolor='black')
            ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
            ax.set_xlabel('Turbine', fontsize=11)
            ax.set_ylabel('Action (yaw)', fontsize=11)
            ax.set_title('Policy Output', fontsize=12)
            ax.set_xticks(range(self.n_turbines))
            ax.set_xticklabels([f'T{i}' for i in range(self.n_turbines)])
            ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.text(0.5, 0.5, 'No action data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.axis('off')
        
        plt.tight_layout()
        return fig
    
    def visualize_step(
        self,
        obs: Optional[np.ndarray] = None,
        reset: bool = False,
    ):
        """
        Visualize attention for current or provided observation.
        
        Args:
            obs: Observation array (n_turbines, obs_dim). If None, uses current state
            reset: Whether to reset environment first
        """
        if reset or obs is None:
            obs, _ = self.env.reset()
        
        wind_dir = self.env.mean_wind_direction
        attn, action = self.get_attention_and_action(obs, wind_dir)
        
        return self.plot_attention_single(attn, wind_dir, action)
    
    def run_episode_with_viz(
        self,
        n_steps: int = 50,
        plot_every: int = 10,
        save_dir: Optional[str] = None,
    ):
        """
        Run an episode and visualize attention at regular intervals.
        
        Args:
            n_steps: Maximum number of steps
            plot_every: Visualize every N steps
            save_dir: If provided, save figures to this directory
        """
        obs, _ = self.env.reset()
        
        # Clear history
        self.attention_history = []
        self.reward_history = []
        self.action_history = []
        self.wind_dir_history = []
        
        total_reward = 0
        
        for step in range(n_steps):
            wind_dir = self.env.mean_wind_direction
            attn, action = self.get_attention_and_action(obs, wind_dir)
            
            # Store
            self.attention_history.append(attn)
            self.action_history.append(action)
            self.wind_dir_history.append(wind_dir)
            
            # Step
            action_flat = action.flatten()
            obs, reward, terminated, truncated, info = self.env.step(action_flat)
            self.reward_history.append(reward)
            total_reward += reward
            
            # Visualize
            if step % plot_every == 0:
                clear_output(wait=True)
                fig = self.plot_attention_single(attn, wind_dir, action)
                plt.suptitle(f'Step {step} | Reward: {reward:.2f} | Total: {total_reward:.2f}',
                            fontsize=14, fontweight='bold')
                
                if save_dir:
                    import os
                    os.makedirs(save_dir, exist_ok=True)
                    plt.savefig(f'{save_dir}/step_{step:03d}.png', dpi=100, bbox_inches='tight')
                
                plt.show()
                time.sleep(0.1)
            
            if terminated or truncated:
                print(f"Episode ended at step {step}")
                break
        
        print(f"\nEpisode complete!")
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Mean reward: {np.mean(self.reward_history):.2f}")
        print(f"  Steps: {len(self.reward_history)}")
        
        return {
            'attention_history': self.attention_history,
            'reward_history': self.reward_history,
            'action_history': self.action_history,
            'wind_dir_history': self.wind_dir_history,
        }
    
    def analyze_wake_physics(self, verbose: bool = True) -> Dict:
        """
        Analyze whether attention aligns with wake physics.
        
        Requires running an episode first with run_episode_with_viz.
        """
        if not self.attention_history:
            print("No attention history! Run an episode first.")
            return {}
        
        results = []
        
        for attn_weights, wind_dir in zip(self.attention_history, self.wind_dir_history):
            # Average over heads
            if self.head_idx == -1:
                attn = attn_weights[self.layer_idx].mean(axis=0)
            else:
                attn = attn_weights[self.layer_idx][self.head_idx]
            
            # Calculate wind direction vector
            wind_rad = np.radians(wind_dir)
            wind_vec = np.array([-np.sin(wind_rad), -np.cos(wind_rad)])
            
            upwind_attn = []
            downwind_attn = []
            
            for i in range(self.n_turbines):
                for j in range(self.n_turbines):
                    if i == j:
                        continue
                    
                    vec_ji = self.positions[i] - self.positions[j]
                    projection = np.dot(vec_ji, wind_vec)
                    
                    if projection > 0.5:  # i downwind of j
                        upwind_attn.append(attn[i, j])
                    elif projection < -0.5:  # i upwind of j
                        downwind_attn.append(attn[i, j])
            
            if upwind_attn and downwind_attn:
                results.append({
                    'upwind_mean': np.mean(upwind_attn),
                    'downwind_mean': np.mean(downwind_attn),
                    'ratio': np.mean(upwind_attn) / np.mean(downwind_attn) if np.mean(downwind_attn) > 0 else 0,
                })
        
        if not results:
            print("Could not compute wake physics metrics")
            return {}
        
        avg_upwind = np.mean([r['upwind_mean'] for r in results])
        avg_downwind = np.mean([r['downwind_mean'] for r in results])
        avg_ratio = np.mean([r['ratio'] for r in results])
        
        if verbose:
            print("\n" + "="*50)
            print("WAKE PHYSICS ANALYSIS")
            print("="*50)
            print(f"Average attention (upwind → downwind):  {avg_upwind:.4f}")
            print(f"Average attention (downwind → upwind):  {avg_downwind:.4f}")
            print(f"Ratio (upwind/downwind):                {avg_ratio:.2f}")
            print()
            
            if avg_ratio > 1.2:
                print("✓ Model learns wake physics!")
                print("  Attention is stronger from upwind to downwind turbines,")
                print("  which aligns with wake propagation direction.")
            elif avg_ratio > 0.8:
                print("~ Weak wake physics signal")
                print("  Attention pattern is close to uniform.")
            else:
                print("✗ Unexpected attention pattern")
                print("  Model may not be learning wake interactions correctly.")
            
            print("="*50 + "\n")
        
        return {
            'upwind_mean': avg_upwind,
            'downwind_mean': avg_downwind,
            'ratio': avg_ratio,
            'per_step': results,
        }
    
    def plot_attention_evolution(self, sample_every: int = 5, figsize: Tuple[int, int] = (14, 4)):
        """
        Plot how attention evolves over the episode.
        
        Args:
            sample_every: Plot every N steps
            figsize: Figure size
        """
        if not self.attention_history:
            print("No attention history! Run an episode first.")
            return
        
        # Sample timesteps
        indices = list(range(0, len(self.attention_history), sample_every))
        n_samples = len(indices)
        
        if n_samples == 0:
            print("No timesteps to plot")
            return
        
        # Create subplots
        n_cols = min(5, n_samples)
        n_rows = (n_samples + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1] * n_rows))
        if n_samples == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for idx, step in enumerate(indices):
            ax = axes[idx]
            
            # Get attention
            if self.head_idx == -1:
                attn = self.attention_history[step][self.layer_idx].mean(axis=0)
            else:
                attn = self.attention_history[step][self.layer_idx][self.head_idx]
            
            # Plot heatmap
            im = ax.imshow(attn, cmap='Reds', vmin=0)
            ax.set_title(f'Step {step}', fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            
            if idx % n_cols == 0:
                ax.set_ylabel('Query', fontsize=8)
            if idx >= n_samples - n_cols:
                ax.set_xlabel('Key', fontsize=8)
        
        # Hide unused subplots
        for idx in range(n_samples, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Attention Evolution Over Episode', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()


# ==============================================================================
# Example Usage in Jupyter Notebook
# ==============================================================================

"""
# 1. Load model and environment
from transformer_sac_windfarm_v9 import TransformerActor, make_env
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
env = make_env(layout_name='test_layout', seed=42)
actor = load_trained_actor('runs/experiment/actor.pth', obs_dim_per_turbine=..., device=device)

# 2. Create visualizer
from attention_viz_notebook import AttentionVisualizer

viz = AttentionVisualizer(actor, env, device, layer_idx=-1, head_idx=-1)

# 3. Visualize single step
viz.visualize_step(reset=True)

# 4. Run episode with visualization
episode_data = viz.run_episode_with_viz(n_steps=100, plot_every=20)

# 5. Analyze wake physics
wake_analysis = viz.analyze_wake_physics()

# 6. Plot attention evolution
viz.plot_attention_evolution(sample_every=10)
"""