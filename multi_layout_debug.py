"""
Multi-Layout Training Debugging Utilities
=========================================

Comprehensive debugging metrics for transformer-based wind farm control
when training across multiple layouts.

Key debugging categories:
1. Layout-specific performance (per-layout reward/power tracking)
2. Attention physics validation (upwind/downwind attention ratios)
3. Gradient health monitoring
4. Positional encoding effectiveness
5. Buffer composition analysis
6. Q-value statistics
7. Action distribution analysis
8. Wind direction coverage

Usage:
    debug_logger = MultiLayoutDebugLogger(
        layout_names=["2x1", "2x2", "3x1"],
        log_frequency=100,
        attention_log_frequency=500,
    )
    
    # In training loop:
    if debug_logger.should_log_gradients(step):
        debug_logger.log_gradient_norms(actor, qf1, qf2, writer, step)

Author: Claude (debugging assistant for Marcus's transformer wind farm project)
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass, field
import math


@dataclass
class AttentionStats:
    """Statistics about attention patterns for physics validation."""
    upwind_attention_mean: float = 0.0
    downwind_attention_mean: float = 0.0
    self_attention_mean: float = 0.0
    upwind_downwind_ratio: float = 1.0  # > 1.2 indicates good physics learning
    max_attention_turbine: int = -1  # Which turbine receives most attention
    attention_entropy: float = 0.0  # How distributed attention is


@dataclass
class LayoutStats:
    """Per-layout performance tracking."""
    rewards: deque = field(default_factory=lambda: deque(maxlen=1000))
    powers: deque = field(default_factory=lambda: deque(maxlen=1000))
    episode_returns: deque = field(default_factory=lambda: deque(maxlen=100))
    actions: deque = field(default_factory=lambda: deque(maxlen=1000))
    q_values: deque = field(default_factory=lambda: deque(maxlen=1000))
    sample_count: int = 0


class MultiLayoutDebugLogger:
    """
    Comprehensive debugging logger for multi-layout transformer training.
    
    Tracks metrics that help diagnose:
    - Whether model learns all layouts equally
    - Whether attention patterns reflect wake physics
    - Whether training is stable (gradients, Q-values)
    - Whether positional encoding is being utilized
    
    All logging frequencies are configurable and checked via should_log_* methods.
    """
    
    def __init__(
        self,
        layout_names: List[str],
        log_frequency: int = 100,
        attention_log_frequency: int = 500,
        gradient_log_frequency: int = 100,
        q_value_log_frequency: int = 50,
        histogram_frequency: int = 1000,
        diagnostic_print_frequency: int = 2000,
    ):
        """
        Args:
            layout_names: Names of layouts being trained on
            log_frequency: How often to log basic/summary metrics (in steps)
            attention_log_frequency: How often to compute attention stats
            gradient_log_frequency: How often to log gradient stats
            q_value_log_frequency: How often to log Q-value stats
            histogram_frequency: How often to log histograms (expensive)
            diagnostic_print_frequency: How often to print diagnostic summary
        """
        self.layout_names = layout_names
        self.log_frequency = log_frequency
        self.attention_log_frequency = attention_log_frequency
        self.gradient_log_frequency = gradient_log_frequency
        self.q_value_log_frequency = q_value_log_frequency
        self.histogram_frequency = histogram_frequency
        self.diagnostic_print_frequency = diagnostic_print_frequency
        
        # Per-layout tracking
        self.layout_stats: Dict[str, LayoutStats] = {
            name: LayoutStats() for name in layout_names
        }
        
        # Global tracking
        self.wind_direction_history = deque(maxlen=10000)
        self.attention_stats_history = deque(maxlen=1000)
        self.gradient_norm_history = defaultdict(lambda: deque(maxlen=1000))
        
        # Buffer composition tracking
        self.buffer_layout_counts = defaultdict(int)
    
    # =========================================================================
    # FREQUENCY CHECK METHODS - Use these to decide when to log
    # =========================================================================
    
    def should_log(self, step: int) -> bool:
        """Check if basic metrics should be logged at this step."""
        return step > 0 and step % self.log_frequency == 0
    
    def should_log_attention(self, step: int) -> bool:
        """Check if attention analysis should be performed at this step."""
        return step > 0 and step % self.attention_log_frequency == 0
    
    def should_log_gradients(self, step: int) -> bool:
        """Check if gradient norms should be logged at this step."""
        return step > 0 and step % self.gradient_log_frequency == 0
    
    def should_log_q_values(self, step: int) -> bool:
        """Check if Q-value stats should be logged at this step."""
        return step > 0 and step % self.q_value_log_frequency == 0
    
    def should_log_histograms(self, step: int) -> bool:
        """Check if histograms should be logged at this step (expensive)."""
        return step > 0 and step % self.histogram_frequency == 0
    
    def should_print_diagnostics(self, step: int) -> bool:
        """Check if diagnostic summary should be printed at this step."""
        return step > 0 and step % self.diagnostic_print_frequency == 0
        
    # =========================================================================
    # LAYOUT-SPECIFIC METRICS
    # =========================================================================
    
    def log_layout_step(
        self,
        layout_name: str,
        reward: float,
        power: Optional[float] = None,
        actions: Optional[np.ndarray] = None,
        q_value: Optional[float] = None,
    ):
        """Log per-step metrics for a specific layout. Call every step."""
        if layout_name not in self.layout_stats:
            # Handle unknown layouts gracefully
            self.layout_stats[layout_name] = LayoutStats()
            self.layout_names.append(layout_name)
        
        stats = self.layout_stats[layout_name]
        stats.rewards.append(reward)
        stats.sample_count += 1
        
        if power is not None:
            stats.powers.append(power)
        if actions is not None:
            stats.actions.append(actions)
        if q_value is not None:
            stats.q_values.append(q_value)
    
    def log_layout_episode(self, layout_name: str, episode_return: float):
        """Log episode return for a specific layout."""
        if layout_name not in self.layout_stats:
            self.layout_stats[layout_name] = LayoutStats()
            self.layout_names.append(layout_name)
        self.layout_stats[layout_name].episode_returns.append(episode_return)
    
    def get_layout_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get summary metrics per layout."""
        metrics = {}
        total_samples = sum(s.sample_count for s in self.layout_stats.values())
        
        for name, stats in self.layout_stats.items():
            metrics[name] = {
                "mean_reward": float(np.mean(stats.rewards)) if stats.rewards else 0.0,
                "std_reward": float(np.std(stats.rewards)) if len(stats.rewards) > 1 else 0.0,
                "mean_power": float(np.mean(stats.powers)) if stats.powers else 0.0,
                "mean_episode_return": float(np.mean(stats.episode_returns)) if stats.episode_returns else 0.0,
                "sample_count": stats.sample_count,
                "sample_fraction": stats.sample_count / max(1, total_samples),
            }
            
            # Action statistics if available
            if stats.actions:
                all_actions = np.concatenate([np.asarray(a).flatten() for a in stats.actions])
                metrics[name]["action_mean"] = float(np.mean(all_actions))
                metrics[name]["action_std"] = float(np.std(all_actions))
                metrics[name]["action_abs_mean"] = float(np.mean(np.abs(all_actions)))
            
            # Q-value statistics if available
            if stats.q_values:
                metrics[name]["q_mean"] = float(np.mean(stats.q_values))
                metrics[name]["q_std"] = float(np.std(stats.q_values))
        
        return metrics
    
    # =========================================================================
    # ATTENTION PHYSICS VALIDATION
    # =========================================================================
    
    def compute_attention_physics_metrics(
        self,
        attention_weights: List[torch.Tensor],
        positions: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_idx: int = -1,
    ) -> AttentionStats:
        """
        Compute metrics that validate whether attention learns wake physics.
        
        Key insight: In wind farm control, downwind turbines should attend
        more strongly to upwind turbines (which create wakes affecting them).
        
        Args:
            attention_weights: List of attention tensors from transformer
                              Each: (batch, n_heads, n_tokens, n_tokens)
            positions: (batch, n_turbines, 2) wind-relative positions
                      (already transformed so wind comes from negative x)
            attention_mask: (batch, n_turbines) True = padding
            layer_idx: Which layer to analyze
            
        Returns:
            AttentionStats with physics validation metrics
        """
        if not attention_weights:
            return AttentionStats()
        
        # Get attention from specified layer, average over heads
        attn = attention_weights[layer_idx]  # (batch, n_heads, n_tokens, n_tokens)
        attn = attn.mean(dim=1)  # Average over heads: (batch, n_tokens, n_tokens)
        
        # Take first sample for analysis
        attn_sample = attn[0].detach().cpu().numpy()
        pos_sample = positions[0].detach().cpu().numpy()
        
        # Determine number of real turbines
        if attention_mask is not None:
            n_real = int((~attention_mask[0]).sum().item())
        else:
            n_real = pos_sample.shape[0]
        
        attn_sample = attn_sample[:n_real, :n_real]
        pos_sample = pos_sample[:n_real]
        
        # Compute upwind/downwind classification
        # In wind-relative coords (wind from 270°), upwind = positive x
        x_coords = pos_sample[:, 0]
        
        upwind_attn = []
        downwind_attn = []
        self_attn = []
        
        for i in range(n_real):
            for j in range(n_real):
                if i == j:
                    self_attn.append(attn_sample[i, j])
                elif x_coords[j] > x_coords[i]:
                    # j is upwind of i (higher x in wind-relative coords)
                    upwind_attn.append(attn_sample[i, j])
                else:
                    # j is downwind of i
                    downwind_attn.append(attn_sample[i, j])
        
        upwind_mean = float(np.mean(upwind_attn)) if upwind_attn else 0.0
        downwind_mean = float(np.mean(downwind_attn)) if downwind_attn else 0.0
        self_mean = float(np.mean(self_attn)) if self_attn else 0.0
        
        # Compute attention entropy (how distributed is attention?)
        attn_flat = attn_sample.flatten()
        attn_flat = attn_flat[attn_flat > 1e-8]  # Avoid log(0)
        entropy = float(-np.sum(attn_flat * np.log(attn_flat + 1e-8)))
        
        # Find which turbine receives most attention overall
        total_received = attn_sample.sum(axis=0)
        max_attention_turbine = int(np.argmax(total_received))
        
        stats = AttentionStats(
            upwind_attention_mean=upwind_mean,
            downwind_attention_mean=downwind_mean,
            self_attention_mean=self_mean,
            upwind_downwind_ratio=float(upwind_mean / (downwind_mean + 1e-8)),
            max_attention_turbine=max_attention_turbine,
            attention_entropy=entropy,
        )
        
        self.attention_stats_history.append(stats)
        return stats
    
    def log_attention_metrics(
        self,
        attention_weights: List[torch.Tensor],
        positions: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        writer,
        global_step: int,
        layer_idx: int = -1,
        log_image: bool = True,
    ):
        """
        Compute and log attention physics metrics, optionally with visualization.
        
        Call this when should_log_attention(step) returns True.
        
        Args:
            attention_weights: List of attention tensors from transformer
            positions: (batch, n_turbines, 2) wind-relative positions
            attention_mask: (batch, n_turbines) True = padding
            writer: TensorBoard SummaryWriter or wandb
            global_step: Current training step
            layer_idx: Which transformer layer to visualize
            log_image: Whether to log attention visualization image
        """
        stats = self.compute_attention_physics_metrics(
            attention_weights=attention_weights,
            positions=positions,
            attention_mask=attention_mask,
            layer_idx=layer_idx,
        )
        
        writer.add_scalar("debug/attention/upwind_attention", 
                         stats.upwind_attention_mean, global_step)
        writer.add_scalar("debug/attention/downwind_attention", 
                         stats.downwind_attention_mean, global_step)
        writer.add_scalar("debug/attention/upwind_downwind_ratio", 
                         stats.upwind_downwind_ratio, global_step)
        writer.add_scalar("debug/attention/self_attention", 
                         stats.self_attention_mean, global_step)
        writer.add_scalar("debug/attention/entropy", 
                         stats.attention_entropy, global_step)
        
        # Log attention visualization as image
        if log_image:
            try:
                fig = self.create_attention_figure(
                    attention_weights=attention_weights,
                    positions=positions,
                    attention_mask=attention_mask,
                    layer_idx=layer_idx,
                    title=f"Step {global_step} | Ratio: {stats.upwind_downwind_ratio:.2f}"
                )
                if fig is not None:
                    writer.add_figure("debug/attention/visualization", fig, global_step)
                    import matplotlib.pyplot as plt
                    plt.close(fig)
            except Exception as e:
                # Don't crash training if visualization fails
                print(f"Warning: Attention visualization failed: {e}")
        
        # Warning if physics look wrong
        if stats.upwind_downwind_ratio < 1.0 and global_step > 10000:
            print(f"⚠ [Step {global_step}] Attention ratio {stats.upwind_downwind_ratio:.2f} < 1.0 "
                  f"- model may not be learning wake physics correctly")
        
        return stats
    
    def create_attention_figure(
        self,
        attention_weights: List[torch.Tensor],
        positions: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_idx: int = -1,
        head_idx: Optional[int] = None,
        title: str = "",
        sample_idx: int = 0,
    ): #TODO THIS USES THE WRONG X AXIS FOR UPWIND/DOWNWIND: 
        """
        Create a figure showing attention patterns overlaid on farm layout.
        
        Creates a 2-panel figure:
        - Left: Farm layout with attention as arrows (line width = attention strength)
        - Right: Attention heatmap matrix
        
        Args:
            attention_weights: List of attention tensors from transformer
            positions: (batch, n_turbines, 2) wind-relative positions
            attention_mask: (batch, n_turbines) True = padding
            layer_idx: Which layer to visualize (-1 = last)
            head_idx: Which head to visualize (None = average all heads)
            title: Figure title
            sample_idx: Which sample in batch to visualize
            
        Returns:
            matplotlib Figure object
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            from matplotlib.colors import Normalize
            from matplotlib.cm import ScalarMappable
        except ImportError:
            return None
        
        if not attention_weights:
            return None
        
        # Get attention from specified layer
        attn = attention_weights[layer_idx]  # (batch, n_heads, n_tokens, n_tokens)
        
        if head_idx is not None:
            attn = attn[:, head_idx:head_idx+1, :, :]  # Keep dim for consistency
        
        attn = attn.mean(dim=1)  # Average over heads: (batch, n_tokens, n_tokens)
        
        # Get single sample
        attn_matrix = attn[sample_idx].detach().cpu().numpy()
        pos = positions[sample_idx].detach().cpu().numpy()
        
        # Determine number of real turbines
        if attention_mask is not None:
            n_real = int((~attention_mask[sample_idx]).sum().item())
        else:
            n_real = pos.shape[0]
        
        attn_matrix = attn_matrix[:n_real, :n_real]
        pos = pos[:n_real]
        
        # Create figure with 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # =================================================================
        # Left panel: Farm layout with attention arrows
        # =================================================================
        ax1 = axes[0]
        
        # Determine upwind/downwind based on x-coordinate (wind-relative)
        x_coords = pos[:, 0]
        x_mean = x_coords.mean()
        
        # Color turbines by relative position
        colors = ['#2ecc71' if x > x_mean else '#e74c3c' for x in x_coords]  # Green=upwind, Red=downwind
        
        # Plot turbines
        ax1.scatter(pos[:, 0], pos[:, 1], c=colors, s=300, zorder=5, edgecolors='black', linewidths=2)
        
        # Add turbine labels
        for i in range(n_real):
            ax1.annotate(f'T{i}', (pos[i, 0], pos[i, 1]), ha='center', va='center', 
                        fontsize=10, fontweight='bold', color='white', zorder=6)
        
        # Draw attention arrows
        # Normalize attention for line widths
        attn_no_diag = attn_matrix.copy()
        np.fill_diagonal(attn_no_diag, 0)
        max_attn = attn_no_diag.max() if attn_no_diag.max() > 0 else 1.0
        
        # Only draw significant attention (top connections)
        threshold = np.percentile(attn_no_diag[attn_no_diag > 0], 50) if (attn_no_diag > 0).any() else 0
        
        for i in range(n_real):
            for j in range(n_real):
                if i != j and attn_matrix[i, j] > threshold:
                    # Arrow from j to i (j attends to i means info flows j->i)
                    # Actually in attention: row i attends to column j
                    # So turbine i is "looking at" turbine j
                    attn_strength = attn_matrix[i, j] / max_attn
                    
                    # Determine if this is upwind attention (good) or downwind (bad)
                    is_upwind = x_coords[j] > x_coords[i]  # j is upwind of i
                    arrow_color = '#3498db' if is_upwind else '#e67e22'  # Blue=upwind, Orange=downwind
                    
                    # Draw arrow from j to i (info flows from attended to attender)
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    
                    # Shorten arrow to not overlap with turbine markers
                    length = np.sqrt(dx**2 + dy**2)
                    if length > 0:
                        shrink = 0.15  # Shrink from each end
                        dx_shrink = dx * shrink
                        dy_shrink = dy * shrink
                        
                        ax1.annotate(
                            '', 
                            xy=(pos[i, 0] - dx_shrink, pos[i, 1] - dy_shrink),
                            xytext=(pos[j, 0] + dx_shrink, pos[j, 1] + dy_shrink),
                            arrowprops=dict(
                                arrowstyle='->', 
                                color=arrow_color,
                                lw=1 + 3 * attn_strength,
                                alpha=0.4 + 0.5 * attn_strength,
                                connectionstyle='arc3,rad=0.1'
                            ),
                            zorder=3
                        )
        
        # Add wind direction indicator
        ax1.annotate(
            '', xy=(-0.5, 0), xytext=(0.5, 0),
            arrowprops=dict(arrowstyle='->', color='gray', lw=2),
            xycoords='axes fraction'
        )
        ax1.text(0.5, 0.02, 'Wind →', transform=ax1.transAxes, ha='center', 
                fontsize=10, color='gray')
        
        # Legend
        legend_elements = [
            mpatches.Patch(color='#2ecc71', label='Upwind turbine'),
            mpatches.Patch(color='#e74c3c', label='Downwind turbine'),
            mpatches.Patch(color='#3498db', label='Upwind attention (good)'),
            mpatches.Patch(color='#e67e22', label='Downwind attention'),
        ]
        ax1.legend(handles=legend_elements, loc='upper left', fontsize=8)
        
        ax1.set_xlabel('Wind-relative X (rotor diameters)', fontsize=10)
        ax1.set_ylabel('Wind-relative Y (rotor diameters)', fontsize=10)
        ax1.set_title('Attention Flow on Farm Layout', fontsize=12, fontweight='bold')
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        
        # =================================================================
        # Right panel: Attention heatmap
        # =================================================================
        ax2 = axes[1]
        
        # Create heatmap
        im = ax2.imshow(attn_matrix, cmap='Blues', aspect='equal')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        cbar.set_label('Attention Weight', fontsize=10)
        
        # Labels
        ax2.set_xticks(range(n_real))
        ax2.set_yticks(range(n_real))
        ax2.set_xticklabels([f'T{i}' for i in range(n_real)])
        ax2.set_yticklabels([f'T{i}' for i in range(n_real)])
        
        ax2.set_xlabel('Key (attended to)', fontsize=10)
        ax2.set_ylabel('Query (attender)', fontsize=10)
        ax2.set_title('Attention Matrix', fontsize=12, fontweight='bold')
        
        # Add values as text
        for i in range(n_real):
            for j in range(n_real):
                val = attn_matrix[i, j]
                color = 'white' if val > 0.5 * attn_matrix.max() else 'black'
                ax2.text(j, i, f'{val:.2f}', ha='center', va='center', 
                        fontsize=8, color=color)
        
        # Mark upwind/downwind regions
        # Add subtle background coloring based on physics expectation
        for i in range(n_real):
            for j in range(n_real):
                if i != j:
                    is_upwind = x_coords[j] > x_coords[i]
                    if is_upwind:
                        # Should have high attention (good)
                        rect = plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, 
                                            edgecolor='green', linewidth=1, alpha=0.5)
                        ax2.add_patch(rect)
        
        # Overall title
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        return fig
    
    def create_multi_head_attention_figure(
        self,
        attention_weights: List[torch.Tensor],
        positions: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_idx: int = -1,
        sample_idx: int = 0,
        title: str = "",
    ): #TODO THIS USES THE WRONG X AXIS FOR UPWIND/DOWNWIND: 
        """
        Create a figure showing attention patterns for each head separately.
        
        Useful for understanding what different heads specialize in.
        
        Returns:
            matplotlib Figure object
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return None
        
        if not attention_weights:
            return None
        
        # Get attention from specified layer
        attn = attention_weights[layer_idx]  # (batch, n_heads, n_tokens, n_tokens)
        n_heads = attn.shape[1]
        
        # Get positions
        pos = positions[sample_idx].detach().cpu().numpy()
        
        # Determine number of real turbines
        if attention_mask is not None:
            n_real = int((~attention_mask[sample_idx]).sum().item())
        else:
            n_real = pos.shape[0]
        
        pos = pos[:n_real]
        x_coords = pos[:, 0]
        
        # Create subplot grid
        n_cols = min(4, n_heads)
        n_rows = (n_heads + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        if n_heads == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for h in range(n_heads):
            row, col = h // n_cols, h % n_cols
            ax = axes[row, col]
            
            attn_head = attn[sample_idx, h, :n_real, :n_real].detach().cpu().numpy()
            
            # Compute upwind attention ratio for this head
            upwind_attn = []
            downwind_attn = []
            for i in range(n_real):
                for j in range(n_real):
                    if i != j:
                        if x_coords[j] > x_coords[i]:
                            upwind_attn.append(attn_head[i, j])
                        else:
                            downwind_attn.append(attn_head[i, j])
            
            ratio = np.mean(upwind_attn) / (np.mean(downwind_attn) + 1e-8) if downwind_attn else 0
            
            # Plot heatmap
            im = ax.imshow(attn_head, cmap='Blues', aspect='equal')
            ax.set_title(f'Head {h}\nRatio: {ratio:.2f}', fontsize=10)
            ax.set_xticks(range(n_real))
            ax.set_yticks(range(n_real))
            ax.set_xticklabels([f'{i}' for i in range(n_real)], fontsize=8)
            ax.set_yticklabels([f'{i}' for i in range(n_real)], fontsize=8)
        
        # Hide unused subplots
        for h in range(n_heads, n_rows * n_cols):
            row, col = h // n_cols, h % n_cols
            axes[row, col].axis('off')
        
        fig.suptitle(f'{title}\nPer-Head Attention (green border = upwind attention expected)', 
                    fontsize=12, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def get_attention_trend(self, window: int = 100) -> Dict[str, float]:
        """Get trend of attention physics metrics over recent steps."""
        if len(self.attention_stats_history) < 2:
            return {}
        
        recent = list(self.attention_stats_history)[-window:]
        
        return {
            "upwind_downwind_ratio_mean": float(np.mean([s.upwind_downwind_ratio for s in recent])),
            "upwind_downwind_ratio_trend": float(
                np.mean([s.upwind_downwind_ratio for s in recent[-window//2:]]) -
                np.mean([s.upwind_downwind_ratio for s in recent[:window//2]])
            ) if len(recent) >= window else 0.0,
            "attention_entropy_mean": float(np.mean([s.attention_entropy for s in recent])),
            "self_attention_mean": float(np.mean([s.self_attention_mean for s in recent])),
        }
    
    # =========================================================================
    # GRADIENT HEALTH MONITORING
    # =========================================================================
    
    def _compute_grad_norm(self, module: torch.nn.Module) -> float:
        """Helper to compute total gradient norm for a module."""
        total_norm = 0.0
        count = 0
        for p in module.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
                count += 1
        return math.sqrt(total_norm) if count > 0 else 0.0
    
    def log_critic_gradient_norms(
        self,
        qf1: torch.nn.Module,
        qf2: torch.nn.Module,
        writer,
        global_step: int,
    ) -> Dict[str, float]:
        """
        Log gradient norms for critic networks.
        
        IMPORTANT: Call this right after q_optimizer.step() and BEFORE 
        actor_loss.backward(), otherwise you'll get gradients from the 
        actor loss instead of critic loss.
        
        Call this when should_log_gradients(step) returns True.
        """
        grad_norms = {}
        
        for name, qf in [("qf1", qf1), ("qf2", qf2)]:
            if hasattr(qf, 'obs_action_encoder') and qf.obs_action_encoder is not None:
                grad_norms[f"{name}/obs_action_encoder"] = self._compute_grad_norm(qf.obs_action_encoder)
            if hasattr(qf, 'pos_encoder') and qf.pos_encoder is not None:
                grad_norms[f"{name}/pos_encoder"] = self._compute_grad_norm(qf.pos_encoder)
            if hasattr(qf, 'rel_pos_bias') and qf.rel_pos_bias is not None:
                grad_norms[f"{name}/rel_pos_bias"] = self._compute_grad_norm(qf.rel_pos_bias)
            if hasattr(qf, 'transformer') and qf.transformer is not None:
                grad_norms[f"{name}/transformer"] = self._compute_grad_norm(qf.transformer)
            if hasattr(qf, 'q_head') and qf.q_head is not None:
                grad_norms[f"{name}/q_head"] = self._compute_grad_norm(qf.q_head)
            grad_norms[f"{name}/total"] = self._compute_grad_norm(qf)
        
        # Log to writer and history
        for name, norm in grad_norms.items():
            writer.add_scalar(f"debug/gradients/{name}", norm, global_step)
            self.gradient_norm_history[name].append(norm)
        
        return grad_norms
    
    def log_actor_gradient_norms(
        self,
        actor: torch.nn.Module,
        writer,
        global_step: int,
    ) -> Dict[str, float]:
        """
        Log gradient norms for actor network.
        
        Call this right after actor_optimizer.step().
        
        Call this when should_log_gradients(step) returns True.
        """
        grad_norms = {}
        
        # Actor components
        if hasattr(actor, 'obs_encoder') and actor.obs_encoder is not None:
            grad_norms["actor/obs_encoder"] = self._compute_grad_norm(actor.obs_encoder)
        if hasattr(actor, 'pos_encoder') and actor.pos_encoder is not None:
            grad_norms["actor/pos_encoder"] = self._compute_grad_norm(actor.pos_encoder)
        if hasattr(actor, 'rel_pos_bias') and actor.rel_pos_bias is not None:
            grad_norms["actor/rel_pos_bias"] = self._compute_grad_norm(actor.rel_pos_bias)
        if hasattr(actor, 'transformer') and actor.transformer is not None:
            grad_norms["actor/transformer"] = self._compute_grad_norm(actor.transformer)
        
        # Actor action heads
        actor_head_norm = 0.0
        if hasattr(actor, 'fc_mean') and actor.fc_mean.weight.grad is not None:
            actor_head_norm += actor.fc_mean.weight.grad.norm(2).item() ** 2
        if hasattr(actor, 'fc_logstd') and actor.fc_logstd.weight.grad is not None:
            actor_head_norm += actor.fc_logstd.weight.grad.norm(2).item() ** 2
        grad_norms["actor/action_heads"] = math.sqrt(actor_head_norm)
        
        # Total actor norm
        grad_norms["actor/total"] = self._compute_grad_norm(actor)
        
        # Log to writer and history
        for name, norm in grad_norms.items():
            writer.add_scalar(f"debug/gradients/{name}", norm, global_step)
            self.gradient_norm_history[name].append(norm)
        
        return grad_norms
    
    def log_gradient_norms(
        self,
        actor: torch.nn.Module,
        qf1: torch.nn.Module,
        qf2: torch.nn.Module,
        writer,
        global_step: int,
    ) -> Dict[str, float]:
        """
        DEPRECATED: Use log_critic_gradient_norms() and log_actor_gradient_norms() 
        separately for accurate measurements.
        
        This method logs all gradients at once, but the timing makes it tricky:
        - Critic gradients are only valid right after critic backward/step
        - Actor gradients are only valid right after actor backward/step
        
        If you must use this, call it right after actor_optimizer.step() and
        understand that critic gradients will be from actor_loss, not critic_loss.
        """
        grad_norms = {}
        
        # Actor
        actor_norms = self.log_actor_gradient_norms(actor, writer, global_step)
        grad_norms.update(actor_norms)
        
        # Critics (NOTE: these will be from actor_loss if called after actor update!)
        critic_norms = self.log_critic_gradient_norms(qf1, qf2, writer, global_step)
        grad_norms.update(critic_norms)
        
        return grad_norms
    
    def check_gradient_health(self) -> Dict[str, Any]:
        """Check for gradient anomalies."""
        issues = []
        stats = {}
        
        for name, history in self.gradient_norm_history.items():
            if len(history) < 10:
                continue
            
            recent = list(history)[-100:]
            mean_norm = float(np.mean(recent))
            max_norm = float(np.max(recent))
            
            stats[f"{name}_mean"] = mean_norm
            stats[f"{name}_max"] = max_norm
            
            # Check for issues
            if max_norm > 100:
                issues.append(f"Exploding gradients in {name}: max={max_norm:.2f}")
            if mean_norm < 1e-6 and "total" in name:
                issues.append(f"Vanishing gradients in {name}: mean={mean_norm:.2e}")
        
        return {"stats": stats, "issues": issues}
    
    # =========================================================================
    # Q-VALUE ANALYSIS
    # =========================================================================
    
    def log_q_value_stats(
        self,
        qf1_values: torch.Tensor,
        qf2_values: torch.Tensor,
        target_q: torch.Tensor,
        writer,
        global_step: int,
    ) -> Dict[str, float]:
        """
        Log Q-value statistics for stability monitoring.
        
        Call this when should_log_q_values(step) returns True.
        """
        stats = {
            "q1_mean": float(qf1_values.mean().item()),
            "q1_std": float(qf1_values.std().item()),
            "q1_min": float(qf1_values.min().item()),
            "q1_max": float(qf1_values.max().item()),
            "q2_mean": float(qf2_values.mean().item()),
            "q2_std": float(qf2_values.std().item()),
            "q_target_mean": float(target_q.mean().item()),
            "q_target_std": float(target_q.std().item()),
            "q1_q2_diff_mean": float((qf1_values - qf2_values).abs().mean().item()),
            "td_error_magnitude": float((qf1_values - target_q).abs().mean().item()),
        }
        
        for name, value in stats.items():
            writer.add_scalar(f"debug/q_values/{name}", value, global_step)
        
        return stats
    
    # =========================================================================
    # WIND DIRECTION COVERAGE
    # =========================================================================
    
    def log_wind_direction(self, wind_dir: float):
        """Track wind direction for coverage analysis. Call every step."""
        self.wind_direction_history.append(wind_dir)
    
    def get_wind_direction_coverage(self, n_bins: int = 12) -> Dict[str, Any]:
        """
        Analyze wind direction coverage in training.
        
        Returns histogram and uniformity metrics.
        """
        if len(self.wind_direction_history) < 100:
            return {}
        
        wd = np.array(self.wind_direction_history)
        
        # Compute histogram
        bin_edges = np.linspace(0, 360, n_bins + 1)
        counts, _ = np.histogram(wd, bins=bin_edges)
        
        # Normalize to get distribution
        dist = counts / counts.sum()
        
        # Compute uniformity (KL divergence from uniform)
        uniform = np.ones(n_bins) / n_bins
        kl_div = float(np.sum(dist * np.log((dist + 1e-8) / uniform)))
        
        return {
            "wind_dir_mean": float(np.mean(wd)),
            "wind_dir_std": float(np.std(wd)),
            "wind_dir_kl_from_uniform": kl_div,
            "wind_dir_histogram": counts.tolist(),
            "wind_dir_coverage_min_bin": float(dist.min()),
            "wind_dir_coverage_max_bin": float(dist.max()),
        }
    
    # =========================================================================
    # MAIN LOGGING INTERFACE
    # =========================================================================
    
    def log_summary_metrics(
        self,
        writer,
        global_step: int,
        prefix: str = "debug",
    ):
        """
        Log all accumulated summary metrics to wandb/tensorboard.
        
        Call this when should_log(step) returns True.
        """
        
        # Layout-specific metrics
        layout_metrics = self.get_layout_metrics()
        for layout_name, metrics in layout_metrics.items():
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    writer.add_scalar(
                        f"{prefix}/layouts/{layout_name}/{metric_name}",
                        value,
                        global_step
                    )
        
        # Layout balance (are we training equally on all layouts?)
        total_samples = sum(s.sample_count for s in self.layout_stats.values())
        if total_samples > 0:
            for name, stats in self.layout_stats.items():
                writer.add_scalar(
                    f"{prefix}/layout_balance/{name}_fraction",
                    stats.sample_count / total_samples,
                    global_step
                )
        
        # Attention physics trend metrics
        attn_trend = self.get_attention_trend()
        for name, value in attn_trend.items():
            writer.add_scalar(f"{prefix}/attention_trend/{name}", value, global_step)
        
        # Wind direction coverage
        wd_coverage = self.get_wind_direction_coverage()
        for name, value in wd_coverage.items():
            if not isinstance(value, list):
                writer.add_scalar(f"{prefix}/wind_coverage/{name}", value, global_step)
    
    def print_diagnostics(self, global_step: int):
        """
        Print a summary of training diagnostics to console.
        
        Call this when should_print_diagnostics(step) returns True.
        """
        print("\n" + "=" * 60)
        print(f"TRAINING DIAGNOSTICS (Step {global_step})")
        print("=" * 60)
        
        # Layout balance
        print("\nLayout Balance:")
        layout_metrics = self.get_layout_metrics()
        for name, metrics in layout_metrics.items():
            print(f"  {name}: {metrics['sample_fraction']*100:.1f}% of samples, "
                  f"mean_reward={metrics['mean_reward']:.4f}")
        
        # Attention physics
        print("\nAttention Physics:")
        attn_trend = self.get_attention_trend()
        if attn_trend:
            ratio = attn_trend.get('upwind_downwind_ratio_mean', 0)
            status = "✓ Good" if ratio > 1.2 else "⚠ Check" if ratio > 1.0 else "✗ Bad"
            print(f"  Upwind/Downwind ratio: {ratio:.3f} {status}")
            print(f"  (>1.2 indicates model is learning wake physics)")
            trend = attn_trend.get('upwind_downwind_ratio_trend', 0)
            trend_str = "↑" if trend > 0.01 else "↓" if trend < -0.01 else "→"
            print(f"  Trend: {trend_str} ({trend:+.3f})")
        else:
            print("  Not enough data yet")
        
        # Gradient health
        print("\nGradient Health:")
        grad_health = self.check_gradient_health()
        if grad_health["issues"]:
            for issue in grad_health["issues"]:
                print(f"  ⚠ {issue}")
        else:
            print("  ✓ No gradient issues detected")
        
        # Wind direction coverage
        print("\nWind Direction Coverage:")
        wd_coverage = self.get_wind_direction_coverage()
        if wd_coverage:
            print(f"  Mean: {wd_coverage['wind_dir_mean']:.1f}°, "
                  f"Std: {wd_coverage['wind_dir_std']:.1f}°")
            kl = wd_coverage['wind_dir_kl_from_uniform']
            status = "✓ Good" if kl < 0.1 else "⚠ Biased" if kl < 0.3 else "✗ Very biased"
            print(f"  KL from uniform: {kl:.3f} {status}")
        else:
            print("  Not enough data yet")
        
        print("=" * 60 + "\n")


# =============================================================================
# CONVENIENCE FUNCTION FOR SIMPLE INTEGRATION
# =============================================================================

def create_debug_logger(
    layout_names: List[str],
    log_every: int = 100,
) -> MultiLayoutDebugLogger:
    """
    Create a debug logger with sensible defaults.
    
    Args:
        layout_names: List of layout names (e.g., ["2x1", "2x2"])
        log_every: Base logging frequency (other frequencies are multiples)
    
    Returns:
        Configured MultiLayoutDebugLogger
    """
    return MultiLayoutDebugLogger(
        layout_names=layout_names,
        log_frequency=log_every,
        attention_log_frequency=log_every * 5,  # Every 500 steps if log_every=100
        gradient_log_frequency=log_every,
        q_value_log_frequency=log_every // 2,   # Every 50 steps if log_every=100
        histogram_frequency=log_every * 10,
        diagnostic_print_frequency=log_every * 20,
    )


if __name__ == "__main__":
    # Quick test
    logger = create_debug_logger(layout_names=["2x1", "2x2"])
    
    # Simulate some data
    for i in range(1, 2001):
        layout = "2x1" if i % 2 == 0 else "2x2"
        logger.log_layout_step(
            layout_name=layout,
            reward=np.random.randn(),
            power=np.random.uniform(5, 10),
            actions=np.random.randn(4, 1),
        )
        logger.log_wind_direction(np.random.uniform(250, 290))
        
        # Test frequency checks
        if logger.should_print_diagnostics(i):
            logger.print_diagnostics(i)
    
    print("\nFinal diagnostics:")
    logger.print_diagnostics(2000)