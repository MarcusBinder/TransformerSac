"""
Simple evaluation script for trained MLP SAC agents (from SAC_windfarm.py).

Usage:
    # Single checkpoint
    python evaluate_mlp.py --checkpoint runs/.../checkpoints/step_X.pt --layout test_layout --episodes 5

    # Entire checkpoints folder (results as function of training steps)
    python evaluate_mlp.py --checkpoint-dir runs/.../checkpoints/ --layout test_layout --episodes 5

    # Parallel evaluation with multiple workers (uses pathos if available)
    python evaluate_mlp.py --checkpoint-dir runs/.../checkpoints/ --layout test_layout --workers 4
"""

import argparse
import re
from functools import partial
from pathlib import Path

import numpy as np
import torch
import xarray as xr

try:
    from pathos.multiprocessing import ProcessingPool as Pool
    HAS_PATHOS = True
except ImportError:
    from multiprocessing import Pool
    HAS_PATHOS = False

from helper_funcs import (
    get_layout_positions,
    make_env_config,
    transform_to_wind_relative_numpy,
    EnhancedPerTurbineWrapper,
    find_checkpoints,
    load_actor_from_checkpoint,
)
from MultiLayoutEnv import MultiLayoutEnv, LayoutConfig
from WindGym import WindFarmEnv
from WindGym.wrappers import PerTurbineObservationWrapper
from SAC_windfarm import MLPActor


# =============================================================================
# ENVIRONMENT CREATION
# =============================================================================

def create_eval_env(layout: str, args: dict, seed: int = 42):
    """Create evaluation environment for a single layout."""

    # Get wind turbine
    if args["turbtype"] == "DTU10MW":
        from py_wake.examples.data.dtu10mw import DTU10MW as WT
    elif args["turbtype"] == "V80":
        from py_wake.examples.data.hornsrev1 import V80 as WT
    else:
        raise ValueError(f"Unknown turbine type: {args['turbtype']}")

    wind_turbine = WT()

    layout_names = [l.strip() for l in layout.split(",")]
    layouts = []
    for name in layout_names:
        x_pos, y_pos = get_layout_positions(name, wind_turbine)
        layouts.append(LayoutConfig(name=name, x_pos=x_pos, y_pos=y_pos))

    # Environment config
    config = make_env_config()

    mes_prefixes = {
        "ws_mes": "ws",
        "wd_mes": "wd",
        "yaw_mes": "yaw",
        "power_mes": "power",
    }
    for mes_type, prefix in mes_prefixes.items():
        config[mes_type][f"{prefix}_history_N"] = args["history_length"]
        config[mes_type][f"{prefix}_history_length"] = args["history_length"]

    base_env_kwargs = {
        "turbine": wind_turbine,
        "n_passthrough": args["max_eps"],
        "TurbBox": "/work/users/manils/rl_timestep/Boxes/V80env/",
        "config": config,
        "turbtype": args["TI_type"],
        "dt_sim": args["dt_sim"],
        "dt_env": args["dt_env"],
        "yaw_step_sim": args["yaw_step"],
    }

    def env_factory(x_pos, y_pos):
        env = WindFarmEnv(x_pos=x_pos, y_pos=y_pos, **base_env_kwargs)
        env.action_space.seed(seed)
        return env

    def combined_wrapper(env):
        env = PerTurbineObservationWrapper(env)
        env = EnhancedPerTurbineWrapper(env, wd_scale_range=args["wd_scale_range"])
        return env

    env = MultiLayoutEnv(
        layouts=layouts,
        env_factory=env_factory,
        per_turbine_wrapper=combined_wrapper,
        seed=seed,
        shuffle=False,  # No shuffling during eval
    )

    return env, wind_turbine


# =============================================================================
# OBSERVATION PREPARATION
# =============================================================================

def prepare_observation_with_positions(
    obs: np.ndarray,
    raw_positions: np.ndarray,
    wind_direction: float,
    rotor_diameter: float,
    include_positions: bool = True,
) -> np.ndarray:
    """
    Prepare observation with optional position features for MLP.
    
    Args:
        obs: (max_turb, obs_dim) observation array
        raw_positions: (max_turb, 2) turbine positions in meters
        wind_direction: Wind direction in degrees
        rotor_diameter: Rotor diameter for normalization
        include_positions: Whether to concatenate positions
    
    Returns:
        Flattened observation array ready for MLP
    """
    if include_positions:
        # Normalize positions
        positions_norm = raw_positions / rotor_diameter
        
        # Apply wind-relative transformation
        # Need to add batch dimension for transform function
        positions_transformed = transform_to_wind_relative_numpy(
            positions_norm[np.newaxis, ...],  # (1, max_turb, 2)
            np.array([wind_direction])  # (1,)
        )[0]  # Remove batch dim -> (max_turb, 2)
        
        # Concatenate positions to observations
        obs_with_pos = np.concatenate([obs, positions_transformed], axis=-1)
        
        # Flatten
        return obs_with_pos.reshape(-1)
    else:
        # Just flatten
        return obs.reshape(-1)


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(
    checkpoint_path: str,
    layout: str,
    num_episodes: int = 5,
    num_steps: int = 200,
    deterministic: bool = False,
    seed: int = 42,
    verbose: bool = False,
):
    """
    Evaluate a trained MLP agent on a specific layout.

    Args:
        checkpoint_path: Path to checkpoint .pt file
        layout: Layout name (e.g., "test_layout", "square_3x3")
        num_episodes: Number of episodes to run
        num_steps: Max steps per episode
        deterministic: Use deterministic actions (mean of policy)
        seed: Random seed
        verbose: Print progress

    Returns:
        dict with evaluation metrics
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    checkpoint, args = load_actor_from_checkpoint(checkpoint_path, device)

    if verbose:
        print(f"Loaded checkpoint from step {checkpoint['step']}")
        print(f"Original training layouts: {args['layouts']}")
        print(f"Evaluating on: {layout}")
        print(f"Include positions: {args.get('include_positions', False)}")

    # Create environment
    env, wind_turbine = create_eval_env(layout, args, seed)

    # Get dimensions
    obs, info = env.reset()
    n_turbines = env.n_turbines
    max_turbines = env.max_turbines
    rotor_diameter = env.rotor_diameter
    obs_dim_per_turbine = obs.shape[1]
    
    # Check if positions are included
    include_positions = args.get("include_positions", False)
    
    # Calculate input dimension for MLP
    if include_positions:
        total_obs_dim = max_turbines * (obs_dim_per_turbine + 2)  # +2 for x,y positions
    else:
        total_obs_dim = max_turbines * obs_dim_per_turbine
    
    action_dim = max_turbines  # One action per turbine

    # Action scaling
    action_high = env.action_space.high[0]
    action_low = env.action_space.low[0]
    action_scale = (action_high - action_low) / 2.0
    action_bias = (action_high + action_low) / 2.0

    if verbose:
        print(f"Layout has {n_turbines} turbines (max: {max_turbines})")
        print(f"Obs dim per turbine: {obs_dim_per_turbine}")
        print(f"Total obs dim for MLP: {total_obs_dim}")

    # Create actor with same architecture
    actor = MLPActor(
        obs_dim=total_obs_dim,
        action_dim=action_dim,
        hidden_dim=args["hidden_dim"],
        num_hidden_layers=args["num_hidden_layers"],
        action_scale=action_scale,
        action_bias=action_bias,
    ).to(device)

    # Load weights
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()

    # Run evaluation
    episode_returns = []
    episode_lengths = []
    episode_powers = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        episode_return = 0.0
        episode_length = 0
        powers = []
        done = False
        
        while not done:
            # Get positions and wind direction
            wind_dir = env.wd
            raw_positions = env.turbine_positions
            
            # Prepare observation (with or without positions)
            obs_flat = prepare_observation_with_positions(
                obs, raw_positions, wind_dir, rotor_diameter, include_positions
            )
            
            # Convert to tensor
            obs_tensor = torch.tensor(obs_flat, dtype=torch.float32, device=device).unsqueeze(0)
            
            # Get attention mask for this layout
            attention_mask = env.attention_mask
            mask_tensor = torch.tensor(attention_mask, dtype=torch.bool, device=device).unsqueeze(0)

            # Get action
            with torch.no_grad():
                action, _, mean_action = actor.get_action(
                    obs_tensor, action_mask=mask_tensor, deterministic=deterministic
                )

            action_np = action.squeeze(0).cpu().numpy()
            
            # MultiLayoutEnv expects padded action (max_turbines) and handles slicing internally

            # Step
            obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated

            episode_return += reward
            episode_length += 1

            if episode_length >= num_steps:
                done = True

            if "Power agent" in info:
                powers.append(info["Power agent"])

        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)
        if powers:
            episode_powers.append(np.mean(powers))

        if verbose:
            print(f"  Episode {ep+1}/{num_episodes}: return={episode_return:.2f}, length={episode_length}")

    env.close()

    results = {
        "mean_return": np.mean(episode_returns),
        "std_return": np.std(episode_returns),
        "mean_length": np.mean(episode_lengths),
        "mean_power": np.mean(episode_powers) if episode_powers else None,
        "episode_returns": episode_returns,
        "episode_lengths": episode_lengths,
        "checkpoint_step": checkpoint["step"],
        "layout": layout,
        "num_episodes": num_episodes,
    }

    if verbose:
        print(f"\nResults:")
        print(f"  Mean return: {results['mean_return']:.2f} +/- {results['std_return']:.2f}")
        print(f"  Mean episode length: {results['mean_length']:.1f}")
        if results['mean_power']:
            print(f"  Mean power: {results['mean_power']:.2f}")

    return results


def _evaluate_single_checkpoint(checkpoint_info, layout, num_episodes, num_steps, deterministic, seed):
    """
    Worker function to evaluate a single checkpoint.
    Designed to be called in parallel by multiprocessing pool.

    Args:
        checkpoint_info: Tuple of (step, checkpoint_path)
        layout: Layout name to evaluate on
        num_episodes: Number of episodes
        num_steps: Max steps per episode
        deterministic: Use deterministic actions
        seed: Base random seed (will be offset by step for reproducibility)

    Returns:
        Tuple of (step, results_dict, error)
    """
    step, checkpoint_path = checkpoint_info

    # Use step-based seed offset for reproducibility across parallel runs
    worker_seed = seed + step % 1000

    try:
        results = evaluate(
            checkpoint_path=checkpoint_path,
            layout=layout,
            num_episodes=num_episodes,
            num_steps=num_steps,
            deterministic=deterministic,
            seed=worker_seed,
            verbose=False,
        )
        return (step, results, None)
    except Exception as e:
        return (step, None, str(e))


def evaluate_checkpoint_dir(
    checkpoint_dir: str,
    layout: str,
    num_episodes: int = 5,
    num_steps: int = 200,
    deterministic: bool = False,
    seed: int = 42,
    output_file: str = None,
    num_workers: int = 1,
    verbose: bool = True,
):
    """
    Evaluate all checkpoints in a directory and track results as a function of training steps.

    Args:
        checkpoint_dir: Path to directory containing checkpoint files
        layout: Layout name to evaluate on
        num_episodes: Number of episodes per checkpoint
        num_steps: Max steps per episode
        deterministic: Use deterministic actions
        seed: Random seed
        output_file: Optional path to save results (NetCDF format)
        num_workers: Number of parallel workers (1 = sequential)
        verbose: Print progress

    Returns:
        dict with results per checkpoint step
    """
    checkpoints = find_checkpoints(checkpoint_dir)

    if not checkpoints:
        raise ValueError(f"No checkpoint files found in {checkpoint_dir}")

    if verbose:
        print(f"Found {len(checkpoints)} checkpoints in {checkpoint_dir}")
        print(f"Steps: {[step for step, _ in checkpoints]}")
        print(f"Evaluating on layout: {layout}")
        if num_workers > 1:
            pool_type = "pathos" if HAS_PATHOS else "multiprocessing"
            print(f"Using {num_workers} parallel workers ({pool_type})")
        print("-" * 60)

    all_results = {
        "checkpoint_dir": checkpoint_dir,
        "layout": layout,
        "num_episodes": num_episodes,
        "num_steps": num_steps,
        "steps": [],
        "mean_returns": [],
        "std_returns": [],
        "mean_lengths": [],
        "mean_powers": [],
        "per_checkpoint": [],
    }

    if num_workers > 1:
        # Parallel evaluation
        worker_fn = partial(
            _evaluate_single_checkpoint,
            layout=layout,
            num_episodes=num_episodes,
            num_steps=num_steps,
            deterministic=deterministic,
            seed=seed,
        )

        with Pool(num_workers) as pool:
            results_list = pool.map(worker_fn, checkpoints)

        # Sort results by step and process
        results_list.sort(key=lambda x: x[0])

        for step, results, error in results_list:
            if error:
                if verbose:
                    print(f"  Step {step}: FAILED - {error}")
                continue

            all_results["steps"].append(step)
            all_results["mean_returns"].append(results["mean_return"])
            all_results["std_returns"].append(results["std_return"])
            all_results["mean_lengths"].append(results["mean_length"])
            all_results["mean_powers"].append(results["mean_power"])
            all_results["per_checkpoint"].append(results)

            if verbose:
                power_str = f", power={results['mean_power']:.2f}" if results['mean_power'] else ""
                print(f"  Step {step}: return={results['mean_return']:.2f} +/- {results['std_return']:.2f}{power_str}")
    else:
        # Sequential evaluation (original behavior)
        for i, (step, checkpoint_path) in enumerate(checkpoints):
            if verbose:
                print(f"\n[{i+1}/{len(checkpoints)}] Evaluating checkpoint at step {step}")

            results = evaluate(
                checkpoint_path=checkpoint_path,
                layout=layout,
                num_episodes=num_episodes,
                num_steps=num_steps,
                deterministic=deterministic,
                seed=seed,
                verbose=False,
            )

            all_results["steps"].append(step)
            all_results["mean_returns"].append(results["mean_return"])
            all_results["std_returns"].append(results["std_return"])
            all_results["mean_lengths"].append(results["mean_length"])
            all_results["mean_powers"].append(results["mean_power"])
            all_results["per_checkpoint"].append(results)

            if verbose:
                power_str = f", power={results['mean_power']:.2f}" if results['mean_power'] else ""
                print(f"  Step {step}: return={results['mean_return']:.2f} +/- {results['std_return']:.2f}{power_str}")

    # Print summary table
    if verbose:
        print("\n" + "=" * 60)
        print("SUMMARY: Results as a function of training steps")
        print("=" * 60)
        print(f"{'Step':>10} | {'Mean Return':>14} | {'Std Return':>12} | {'Mean Power':>12}")
        print("-" * 60)
        for i, step in enumerate(all_results["steps"]):
            power_str = f"{all_results['mean_powers'][i]:.2f}" if all_results['mean_powers'][i] else "N/A"
            print(f"{step:>10} | {all_results['mean_returns'][i]:>14.2f} | {all_results['std_returns'][i]:>12.2f} | {power_str:>12}")
        print("=" * 60)

        # Best checkpoint
        if all_results["mean_returns"]:
            best_idx = np.argmax(all_results["mean_returns"])
            best_step = all_results["steps"][best_idx]
            best_return = all_results["mean_returns"][best_idx]
            print(f"\nBest checkpoint: step {best_step} with mean return {best_return:.2f}")

    # Save results to file if requested
    if output_file and all_results["per_checkpoint"]:
        # Build per-episode returns array (step x episode)
        episode_returns_2d = np.array([
            res["episode_returns"] for res in all_results["per_checkpoint"]
        ])
        episode_lengths_2d = np.array([
            res["episode_lengths"] for res in all_results["per_checkpoint"]
        ])

        # Create xarray Dataset
        ds = xr.Dataset(
            data_vars={
                "mean_return": (["step"], np.array(all_results["mean_returns"])),
                "std_return": (["step"], np.array(all_results["std_returns"])),
                "mean_length": (["step"], np.array(all_results["mean_lengths"])),
                "mean_power": (["step"], np.array([
                    p if p is not None else np.nan for p in all_results["mean_powers"]
                ])),
                "episode_returns": (["step", "episode"], episode_returns_2d),
                "episode_lengths": (["step", "episode"], episode_lengths_2d),
            },
            coords={
                "step": all_results["steps"],
                "episode": np.arange(num_episodes),
            },
            attrs={
                "checkpoint_dir": str(all_results["checkpoint_dir"]),
                "layout": all_results["layout"],
                "num_episodes": all_results["num_episodes"],
                "num_steps": all_results["num_steps"],
                "agent_type": "MLP",
            },
        )

        # Ensure .nc extension
        if not output_file.endswith(".nc"):
            output_file = output_file.rsplit(".", 1)[0] + ".nc" if "." in output_file else output_file + ".nc"

        ds.to_netcdf(output_file)
        if verbose:
            print(f"\nResults saved to {output_file}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained MLP SAC agent (from SAC_windfarm.py)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate single checkpoint
  python evaluate_mlp.py --checkpoint runs/exp1/checkpoints/step_10000.pt --layout test_layout

  # Evaluate all checkpoints in a directory (results vs training steps)
  python evaluate_mlp.py --checkpoint-dir runs/exp1/checkpoints/ --layout test_layout

  # Parallel evaluation with 4 workers (requires pathos or uses multiprocessing)
  python evaluate_mlp.py --checkpoint-dir runs/exp1/checkpoints/ --layout test_layout --workers 4

  # Save results to NetCDF file (xarray Dataset)
  python evaluate_mlp.py --checkpoint-dir runs/exp1/checkpoints/ --layout test_layout --output results.nc
        """,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", type=str, help="Path to single checkpoint file")
    group.add_argument("--checkpoint-dir", type=str, help="Path to checkpoints directory (evaluates all)")
    parser.add_argument("--layout", type=str, required=True, help="Layout to evaluate on")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes")
    parser.add_argument("--steps", type=int, default=200, help="Max steps per episode")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic actions (default: stochastic)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, help="Output file path for results (NetCDF format, only for --checkpoint-dir)")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers for --checkpoint-dir (default: 1)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.checkpoint_dir:
        # Evaluate all checkpoints in directory
        results = evaluate_checkpoint_dir(
            checkpoint_dir=args.checkpoint_dir,
            layout=args.layout,
            num_episodes=args.episodes,
            num_steps=args.steps,
            deterministic=args.deterministic,
            seed=args.seed,
            output_file=args.output,
            num_workers=args.workers,
            verbose=True,
        )
    else:
        # Evaluate single checkpoint
        results = evaluate(
            checkpoint_path=args.checkpoint,
            layout=args.layout,
            num_episodes=args.episodes,
            num_steps=args.steps,
            deterministic=args.deterministic,
            seed=args.seed,
            verbose=True,
        )