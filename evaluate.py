"""
Simple evaluation script for trained transformer SAC agents.

Usage:
    python evaluate.py --checkpoint runs/.../checkpoints/step_X.pt --layout test_layout --episodes 5
"""

import argparse
import numpy as np
import torch
import gymnasium as gym

from transformer_sac_windfarm import TransformerActor
from helper_funcs import (
    get_layout_positions,
    make_env_config,
    transform_to_wind_relative,
    EnhancedPerTurbineWrapper,
)
from MultiLayoutEnv import MultiLayoutEnv, LayoutConfig
from WindGym import WindFarmEnv
from WindGym.wrappers import PerTurbineObservationWrapper


def load_actor_from_checkpoint(checkpoint_path: str, device: torch.device):
    """Load actor network from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint["args"]

    # We need obs_dim_per_turbine - will get from env later
    # For now return checkpoint and args
    return checkpoint, args


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
    print("Creating eval env for layouts:", layout_names)
    layouts = []
    for name in layout_names:
        x_pos, y_pos = get_layout_positions(name, wind_turbine)
        layouts.append(LayoutConfig(name=name, x_pos=x_pos, y_pos=y_pos))
    print("Created layouts:", [l.name for l in layouts])

    # Get layout positions
    # x_pos, y_pos = get_layout_positions(layout, wind_turbine)
    # layouts = [LayoutConfig(name=layout, x_pos=x_pos, y_pos=y_pos)]



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


def evaluate(
    checkpoint_path: str,
    layout: str,
    num_episodes: int = 5,
    num_steps: int = 200,
    deterministic: bool = True,
    seed: int = 42,
    verbose: bool = True,
):
    """
    Evaluate a trained agent on a specific layout.

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

    # Create environment
    env, wind_turbine = create_eval_env(layout, args, seed)

    # Get dimensions
    obs, info = env.reset()
    obs_dim_per_turbine = obs.shape[1]
    n_turbines = env.n_turbines
    rotor_diameter = env.rotor_diameter

    # Action scaling
    action_high = env.action_space.high[0]
    action_low = env.action_space.low[0]
    action_scale = (action_high - action_low) / 2.0
    action_bias = (action_high + action_low) / 2.0

    if verbose:
        print(f"Layout has {n_turbines} turbines, obs_dim={obs_dim_per_turbine}")

    # Create actor with same architecture
    actor = TransformerActor(
        obs_dim_per_turbine=obs_dim_per_turbine,
        action_dim_per_turbine=1,
        embed_dim=args["embed_dim"],
        pos_embed_dim=args["pos_embed_dim"],
        num_heads=args["num_heads"],
        num_layers=args["num_layers"],
        mlp_ratio=args["mlp_ratio"],
        dropout=0.0,  # No dropout during eval
        use_farm_token=args["use_farm_token"],
        action_scale=action_scale,
        action_bias=action_bias,
        pos_encoding_type=args["pos_encoding_type"],
        rel_pos_hidden_dim=args["rel_pos_hidden_dim"],
        rel_pos_per_head=args["rel_pos_per_head"],
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

            # Prepare tensors
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            positions_norm = raw_positions / rotor_diameter
            positions_tensor = torch.tensor(positions_norm, dtype=torch.float32, device=device).unsqueeze(0)
            wind_dir_tensor = torch.tensor([wind_dir], dtype=torch.float32, device=device)
            positions_transformed = transform_to_wind_relative(positions_tensor, wind_dir_tensor)

            # Get action
            with torch.no_grad():
                action, _, mean_action, _ = actor.get_action(
                    obs_tensor, positions_transformed, key_padding_mask=None, deterministic=deterministic
                )

            action_np = action.squeeze(0).squeeze(-1).cpu().numpy()

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained transformer SAC agent")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--layout", type=str, required=True, help="Layout to evaluate on")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes")
    parser.add_argument("--steps", type=int, default=200, help="Max steps per episode")
    parser.add_argument("--deterministic", action="store_true", default=True, help="Use deterministic actions")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic actions")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    deterministic = not args.stochastic

    results = evaluate(
        checkpoint_path=args.checkpoint,
        layout=args.layout,
        num_episodes=args.episodes,
        num_steps=args.steps,
        deterministic=deterministic,
        seed=args.seed,
    )
