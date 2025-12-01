"""
Test Script for Attention Visualization

This script tests the attention visualization functionality
with a dummy/random model to verify everything works.

Run this before using your actual trained model to catch any issues.

Usage:
    python test_attention_viz.py
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

# Try to import the visualization modules
try:
    from attention_viz_notebook import AttentionVisualizer
    print("✓ Successfully imported attention_viz_notebook")
except ImportError as e:
    print(f"✗ Failed to import attention_viz_notebook: {e}")
    exit(1)

try:
    from visualize_attention import (
        visualize_attention_at_timestep,
        visualize_attention_heatmaps,
        analyze_wake_physics,
    )
    print("✓ Successfully imported visualize_attention functions")
except ImportError as e:
    print(f"✗ Failed to import visualize_attention: {e}")
    exit(1)

try:
    from viz_utils import plot_attention_simple
    print("✓ Successfully imported viz_utils")
except ImportError as e:
    print(f"✗ Failed to import viz_utils: {e}")
    exit(1)


class DummyTransformerActor(nn.Module):
    """
    Dummy transformer actor for testing.
    Returns random attention weights.
    """
    
    def __init__(self, obs_dim: int, n_turbines: int, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_turbines = n_turbines
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.embed_dim = 128
        
        # Dummy layers
        self.fc = nn.Linear(obs_dim, 1)
    
    def get_action(
        self,
        obs: torch.Tensor,
        positions: torch.Tensor,
        deterministic: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Generate dummy action and attention weights.
        
        Returns random attention that slightly favors nearby turbines.
        """
        batch_size = obs.shape[0]
        
        # Random action
        action = torch.randn(batch_size, self.n_turbines, 1) * 0.3
        log_prob = torch.zeros(batch_size, 1)
        mean_action = action
        
        # Generate attention weights with some structure
        # (favor nearby turbines)
        attn_weights = []
        
        for layer in range(self.n_layers):
            # (batch, n_heads, n_turb, n_turb)
            attn = torch.zeros(batch_size, self.n_heads, self.n_turbines, self.n_turbines)
            
            for h in range(self.n_heads):
                for b in range(batch_size):
                    # Base: random attention
                    base_attn = torch.rand(self.n_turbines, self.n_turbines)
                    
                    # Add distance bias (closer turbines get more attention)
                    pos_np = positions[b].cpu().numpy()
                    for i in range(self.n_turbines):
                        for j in range(self.n_turbines):
                            if i != j:
                                dist = np.linalg.norm(pos_np[i] - pos_np[j])
                                # Closer = more attention
                                distance_weight = np.exp(-dist / 3.0)
                                base_attn[i, j] *= (1 + distance_weight)
                    
                    # Softmax to make it a proper attention distribution
                    base_attn = torch.softmax(base_attn, dim=1)
                    attn[b, h] = base_attn
            
            attn_weights.append(attn)
        
        return action, log_prob, mean_action, attn_weights


class DummyEnv:
    """
    Dummy environment for testing.
    """
    
    def __init__(self, n_turbines: int = 4, obs_dim: int = 5):
        self.n_turbines = n_turbines
        self.obs_dim = obs_dim
        
        # Create a simple grid layout
        spacing = 5.0
        positions = []
        side = int(np.ceil(np.sqrt(n_turbines)))
        for i in range(n_turbines):
            x = (i % side) * spacing
            y = (i // side) * spacing
            positions.append([x, y])
        
        self.turbine_positions = np.array(positions)
        self.rotor_diameter = 178.0  # DTU 10MW
        self.mean_wind_direction = 270.0
        
        self._step_count = 0
    
    def reset(self):
        self._step_count = 0
        obs = np.random.randn(self.n_turbines, self.obs_dim).astype(np.float32)
        info = {}
        return obs, info
    
    def step(self, action):
        self._step_count += 1
        obs = np.random.randn(self.n_turbines, self.obs_dim).astype(np.float32)
        reward = np.random.rand()
        terminated = self._step_count >= 100
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info


def test_basic_functionality():
    """Test basic visualization functions."""
    print("\n" + "="*70)
    print("TEST 1: Basic Functionality")
    print("="*70)
    
    # Create dummy environment and model
    n_turbines = 4
    obs_dim = 5
    
    env = DummyEnv(n_turbines=n_turbines, obs_dim=obs_dim)
    device = torch.device('cpu')
    
    actor = DummyTransformerActor(
        obs_dim=obs_dim,
        n_turbines=n_turbines,
        n_heads=4,
        n_layers=2
    ).to(device)
    
    print(f"  Created dummy env: {n_turbines} turbines, obs_dim={obs_dim}")
    print(f"  Created dummy actor: {actor.n_heads} heads, {actor.n_layers} layers")
    
    # Test single forward pass
    obs, _ = env.reset()
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    pos_tensor = torch.tensor(
        env.turbine_positions / env.rotor_diameter,
        dtype=torch.float32, device=device
    ).unsqueeze(0)
    
    action, _, _, attn_weights = actor.get_action(obs_tensor, pos_tensor)
    
    print(f"  ✓ Forward pass successful")
    print(f"    Action shape: {action.shape}")
    print(f"    Attention layers: {len(attn_weights)}")
    print(f"    Attention shape: {attn_weights[0].shape}")
    
    return env, actor, device


def test_visualizer_class(env, actor, device):
    """Test the AttentionVisualizer class."""
    print("\n" + "="*70)
    print("TEST 2: AttentionVisualizer Class")
    print("="*70)
    
    try:
        viz = AttentionVisualizer(
            actor=actor,
            env=env,
            device=device,
            layer_idx=-1,
            head_idx=-1,
        )
        print("  ✓ Created AttentionVisualizer")
        
        # Test single step visualization
        print("  Testing single step visualization...")
        fig = viz.visualize_step(reset=True)
        plt.savefig('/tmp/test_single_step.png', dpi=100, bbox_inches='tight')
        plt.close()
        print("  ✓ Single step visualization saved to /tmp/test_single_step.png")
        
        # Test episode visualization (just a few steps)
        print("  Testing episode visualization...")
        episode_data = viz.run_episode_with_viz(n_steps=10, plot_every=5, save_dir='/tmp/test_episode')
        print(f"  ✓ Episode visualization complete ({len(episode_data['attention_history'])} steps)")
        
        # Test wake physics analysis
        print("  Testing wake physics analysis...")
        wake_results = viz.analyze_wake_physics(verbose=False)
        print(f"  ✓ Wake physics analysis complete")
        print(f"    Upwind attention: {wake_results['upwind_mean']:.4f}")
        print(f"    Downwind attention: {wake_results['downwind_mean']:.4f}")
        print(f"    Ratio: {wake_results['ratio']:.2f}")
        
        # Test attention evolution plot
        print("  Testing attention evolution plot...")
        viz.plot_attention_evolution(sample_every=2)
        plt.savefig('/tmp/test_evolution.png', dpi=100, bbox_inches='tight')
        plt.close()
        print("  ✓ Attention evolution plot saved to /tmp/test_evolution.png")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_standalone_functions(env, actor, device):
    """Test standalone visualization functions."""
    print("\n" + "="*70)
    print("TEST 3: Standalone Visualization Functions")
    print("="*70)
    
    try:
        # Get some attention data
        obs, _ = env.reset()
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        pos_tensor = torch.tensor(
            env.turbine_positions / env.rotor_diameter,
            dtype=torch.float32, device=device
        ).unsqueeze(0)
        
        _, _, _, attn_weights = actor.get_action(obs_tensor, pos_tensor)
        
        # Convert to numpy
        attn_numpy = [attn.squeeze(0).cpu().numpy() for attn in attn_weights]
        
        print("  Testing visualize_attention_at_timestep...")
        visualize_attention_at_timestep(
            attention_weights=attn_numpy,
            positions=env.turbine_positions,
            rotor_diameter=env.rotor_diameter,
            wind_direction=270.0,
            timestep=0,
            layer_idx=-1,
            save_path='/tmp/test_timestep.png',
        )
        print("  ✓ Saved to /tmp/test_timestep.png")
        
        print("  Testing visualize_attention_heatmaps...")
        visualize_attention_heatmaps(
            attention_weights=attn_numpy,
            layer_idx=-1,
            save_path='/tmp/test_heatmap.png',
        )
        print("  ✓ Saved to /tmp/test_heatmap.png")
        
        print("  Testing analyze_wake_physics...")
        results = analyze_wake_physics(
            attention_weights=attn_numpy,
            positions=env.turbine_positions,
            wind_direction=270.0,
            layer_idx=-1,
        )
        print(f"  ✓ Wake physics results:")
        print(f"    Upwind mean: {results['upwind_mean']:.4f}")
        print(f"    Downwind mean: {results['downwind_mean']:.4f}")
        print(f"    Ratio: {results['ratio_upwind_downwind']:.2f}")
        
        print("  Testing plot_attention_simple from viz_utils...")
        attn_matrix = attn_numpy[-1].mean(axis=0)  # Last layer, avg over heads
        fig = plot_attention_simple(
            attention=attn_matrix,
            positions=env.turbine_positions,
            rotor_diameter=env.rotor_diameter,
            wind_direction=270.0,
        )
        plt.savefig('/tmp/test_simple.png', dpi=100, bbox_inches='tight')
        plt.close()
        print("  ✓ Saved to /tmp/test_simple.png")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print(" ATTENTION VISUALIZATION TEST SUITE")
    print("="*70)
    print("\nThis script tests the attention visualization functionality")
    print("using dummy models and environments.\n")
    
    # Test 1: Basic functionality
    env, actor, device = test_basic_functionality()
    
    # Test 2: Visualizer class
    success_2 = test_visualizer_class(env, actor, device)
    
    # Test 3: Standalone functions
    success_3 = test_standalone_functions(env, actor, device)
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"  Basic functionality:       ✓ PASS")
    print(f"  AttentionVisualizer class: {'✓ PASS' if success_2 else '✗ FAIL'}")
    print(f"  Standalone functions:      {'✓ PASS' if success_3 else '✗ FAIL'}")
    print("\nGenerated test visualizations in /tmp/:")
    print("  - test_single_step.png")
    print("  - test_episode/")
    print("  - test_evolution.png")
    print("  - test_timestep.png")
    print("  - test_heatmap.png")
    print("  - test_simple.png")
    
    if success_2 and success_3:
        print("\n✓ ALL TESTS PASSED!")
        print("\nYou can now use the visualization tools with your trained model.")
        print("See example_attention_viz.py for usage examples.")
    else:
        print("\n✗ SOME TESTS FAILED")
        print("Please check the error messages above.")
    
    print("="*70 + "\n")


if __name__ == '__main__':
    main()