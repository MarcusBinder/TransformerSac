"""
Data Collection Script for Wind Farm Pretraining Datasets

Collects per-turbine measurements from WindGym environments and stores them
in HDF5 format for self-supervised pretraining of transformer encoders.

HDF5 Structure:
    layout_<name>.h5
    ├── attrs: layout_name, n_turbines, rotor_diameter, turbine_type
    ├── positions/xy                    # (n_turbines, 2)
    ├── profiles/receptivity            # (n_turbines, n_directions)  [optional]
    ├── profiles/influence              # (n_turbines, n_directions)  [optional]
    └── episodes/
        └── ep_XXXX/
            ├── attrs: n_steps, mean_ws, mean_wd, mean_ti, seed
            ├── ws      # (n_steps, n_turbines)
            ├── wd      # (n_steps, n_turbines)
            ├── yaw     # (n_steps, n_turbines)
            ├── power   # (n_steps, n_turbines)
            └── actions # (n_steps, n_turbines)

Usage:
    python collect_pretrain_data.py \
        --layouts "grid_3x3,grid_5x5,hornsrev" \
        --episodes_per_layout 100 \
        --output_dir ./pretrain_data \
        --n_profile_directions 360 \
        --profile_source geometric

Author: Marcus (DTU Wind Energy)
"""

import os
import argparse
import time
import numpy as np
import h5py
import gymnasium as gym

# WindGym imports
from WindGym import WindFarmEnv

from helper_funcs import (
    get_layout_positions,
    make_env_config,
)

from WindGym.Agents import GreedyAgent, PyWakeAgent

# =============================================================================
# HDF5 WRITER
# =============================================================================

def create_layout_file(
    filepath: str,
    layout_name: str,
    x_pos: np.ndarray,
    y_pos: np.ndarray,
    rotor_diameter: float,
    turbine_type: str = "DTU10MW",
    receptivity: np.ndarray = None,
    influence: np.ndarray = None,
    n_profile_directions: int = 360,
):
    """Create a new HDF5 file for a layout with metadata and positions."""
    with h5py.File(filepath, "w") as f:
        # Layout metadata
        f.attrs["layout_name"] = layout_name
        f.attrs["n_turbines"] = len(x_pos)
        f.attrs["rotor_diameter"] = rotor_diameter
        f.attrs["turbine_type"] = turbine_type

        # Positions
        pos_group = f.create_group("positions")
        pos_group.create_dataset("xy", data=np.stack([x_pos, y_pos], axis=-1).astype(np.float32))

        # Profiles (optional)
        if receptivity is not None and influence is not None:
            prof_group = f.create_group("profiles")
            prof_group.create_dataset("receptivity", data=receptivity.astype(np.float32))
            prof_group.create_dataset("influence", data=influence.astype(np.float32))
            prof_group.attrs["n_directions"] = n_profile_directions

        # Episodes group (populated later)
        f.create_group("episodes")

    print(f"  Created layout file: {filepath}")
    print(f"    n_turbines: {len(x_pos)}, rotor_D: {rotor_diameter:.1f}m")
    if receptivity is not None:
        print(f"    profiles: ({receptivity.shape[0]}, {receptivity.shape[1]})")


def append_episode(
    filepath: str,
    ws: np.ndarray,       # (n_steps, n_turbines)
    wd: np.ndarray,       # (n_steps, n_turbines)
    yaw: np.ndarray,      # (n_steps, n_turbines)
    power: np.ndarray,    # (n_steps, n_turbines)
    actions: np.ndarray,  # (n_steps, n_turbines)
    mean_ws: float,
    mean_wd: float,
    mean_ti: float,
    seed: int = -1,
):
    """Append a completed episode to the layout file."""
    with h5py.File(filepath, "a") as f:
        eps_group = f["episodes"]
        ep_idx = len(eps_group)
        ep = eps_group.create_group(f"ep_{ep_idx:04d}")

        # Episode metadata
        ep.attrs["n_steps"] = ws.shape[0]
        ep.attrs["mean_ws"] = mean_ws
        ep.attrs["mean_wd"] = mean_wd
        ep.attrs["mean_ti"] = mean_ti
        ep.attrs["seed"] = seed

        # Per-step data
        ep.create_dataset("ws", data=ws.astype(np.float32))
        ep.create_dataset("wd", data=wd.astype(np.float32))
        ep.create_dataset("yaw", data=yaw.astype(np.float32))
        ep.create_dataset("power", data=power.astype(np.float32))
        ep.create_dataset("actions", data=actions.astype(np.float32))


# =============================================================================
# ENVIRONMENT FACTORY
# =============================================================================

def make_wind_turbine(turbtype: str):
    """Create wind turbine object."""
    if turbtype == "DTU10MW":
        from py_wake.examples.data.dtu10mw import DTU10MW as WT
    elif turbtype == "V80":
        from py_wake.examples.data.hornsrev1 import V80 as WT
    else:
        raise ValueError(f"Unknown turbine type: {turbtype}")
    return WT()


def make_single_env(
    x_pos: np.ndarray,
    y_pos: np.ndarray,
    wind_turbine,
    config: dict,
    seed: int = 0,
    dt_sim: int = 5,
    dt_env: int = 10,
    yaw_step: float = 5.0,
    max_eps: int = 200,
    TI_type: str = "Random",
) -> gym.Env:
    """Create a single (unwrapped) WindFarmEnv."""
    env = WindFarmEnv(
        x_pos=x_pos,
        y_pos=y_pos,
        turbine=wind_turbine,
        n_passthrough=max_eps,
        TurbBox="/work/users/manils/rl_timestep/Boxes/V80env/",
        config=config,
        turbtype=TI_type,
        dt_sim=dt_sim,
        dt_env=dt_env,
        yaw_step_sim=yaw_step,
        reset_init=False,
    )
    env.action_space.seed(seed)
    return env

# =============================================================================
# AGENTS
# =============================================================================

def greedy_controller(fs, max_yaw_delta):
    """Returns action in [-1, 1] representing desired yaw correction as fraction of max delta."""
    yaw_baseline = fs.windTurbines.yaw
    wind_dir_baseline = fs.get_wind_direction(
        xyz=fs.windTurbines.rotor_positions_xyz, include_wakes=True
    )
    yaw_offset = (fs.wind_direction - wind_dir_baseline) + yaw_baseline

    # Correct toward zero: clip to max step, normalize to [-1, 1]
    action = np.clip(-yaw_offset / max_yaw_delta, -1, 1)
    return action

def pywake_agent(fs, optimal_yaws, max_yaw_delta):
    """Returns action in [-1, 1] representing desired yaw correction as fraction of max delta.

    Inputs:
    - fs: FarmState object from WindGym, providing current turbine states and wind conditions
    - optimal_yaws: array of optimal yaw angles (degrees) computed by PyWake for each turbine
    - max_yaw_delta: maximum yaw change allowed per step (degrees)
    """
    yaw_baseline = fs.windTurbines.yaw
    yaw_offset = optimal_yaws - yaw_baseline
    # Correct toward optimal: clip to max step, normalize to [-1, 1]
    action = np.clip(yaw_offset / max_yaw_delta, -1, 1)
    return action

# =============================================================================
# EPISODE COLLECTION
# =============================================================================

def collect_episode(env, policy="random", max_steps=600) -> dict:
    """
    Run one episode and collect per-turbine measurements.
    
    Args:
        env: WindFarmEnv instance (reset will be called)
        policy: "random" for uniform random actions, "greedy" for zero yaw
        max_steps: safety limit to prevent infinite loops (should not be hit if env is configured properly)
    Returns:
        dict with keys: ws, wd, yaw, power, actions (each (n_steps, n_turb)),
        plus mean_ws, mean_wd, mean_ti scalars
    """
    obs, info = env.reset()
    base_env = env # No base_env here. We just use the env directly since it's not wrapped.
    n_turb = base_env.n_turb

    # Episode-level metadata (constant within episode)
    mean_ws = float(base_env.ws)
    mean_wd = float(base_env.wd)
    mean_ti = float(base_env.ti)

    # Collect step data
    ws_list, wd_list, yaw_list, power_list, action_list = [], [], [], [], []

    done = False
    current_step = 0

    if policy == "pywake":
        py_agent = PyWakeAgent(x_pos=base_env.x_pos, y_pos=base_env.y_pos, 
                       turbine=base_env.turbine, yaw_max=30, yaw_min = -30,
                       look_up=False, env=base_env)
        
        py_agent.update_wind(wind_speed=base_env.ws, wind_direction=base_env.wd, TI=base_env.ti)
        py_agent.optimize()

    while not done:
        # Select action
        if policy == "random":
            action = env.action_space.sample()
        elif policy == "greedy":
            action = greedy_controller(base_env.fs, max_yaw_delta=base_env.yaw_step_env)  # or whatever the env's max delta is
        elif policy == "pywake":
            action = pywake_agent(base_env.fs, py_agent.optimized_yaws, max_yaw_delta=base_env.yaw_step_env)
        else:
            raise ValueError(f"Unknown policy: {policy}")

        # Record pre-step measurements
        measurements = base_env.farm_measurements.get_measurements(scaled=False)
        mes = measurements.reshape(n_turb, 4)  # (n_turb, 4) = [ws, wd, yaw, power]
        ws_list.append(mes[:, 0].copy())
        wd_list.append(mes[:, 1].copy())
        yaw_list.append(mes[:, 2].copy())
        power_list.append(mes[:, 3].copy())
        action_list.append(action.flatten()[:n_turb].copy())

        # Step
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        current_step += 1
        if current_step >= max_steps:
            print(f"Reached max_steps={max_steps}, ending episode.")
            done = True

    return {
        "ws": np.stack(ws_list, axis=0),        # (n_steps, n_turb)
        "wd": np.stack(wd_list, axis=0),
        "yaw": np.stack(yaw_list, axis=0),
        "power": np.stack(power_list, axis=0),
        "actions": np.stack(action_list, axis=0),
        "mean_ws": mean_ws,
        "mean_wd": mean_wd,
        "mean_ti": mean_ti,
    }


# =============================================================================
# PROFILE COMPUTATION
# =============================================================================

def compute_profiles(
    x_pos, y_pos, wind_turbine,
    profile_source: str = "geometric",
    n_directions: int = 360,
):
    """Compute receptivity and influence profiles for a layout."""
    if profile_source.lower() == "geometric":
        from geometric_profiles import compute_layout_profiles_vectorized
        D = wind_turbine.diameter()
        recep, infl = compute_layout_profiles_vectorized(
            x_pos, y_pos,
            rotor_diameter=D,
            k_wake=0.04,
            n_directions=n_directions,
            sigma_smooth=10.0,
            scale_factor=15.0,
        )
    elif profile_source.lower() == "pywake":
        from receptivity_profiles import compute_layout_profiles
        recep, infl = compute_layout_profiles(
            x_pos, y_pos, wind_turbine,
            n_directions=n_directions,
        )
    else:
        raise ValueError(f"Unknown profile_source: {profile_source}")

    return recep, infl


# =============================================================================
# MAIN COLLECTION LOOP
# =============================================================================

def collect_layout_data(
    layout_name: str,
    output_dir: str,
    wind_turbine,
    config: dict,
    episodes_per_layout: int = 100,
    policy: str = "random",
    profile_source: str = "geometric",
    n_profile_directions: int = 360,
    compute_profiles_flag: bool = True,
    base_seed: int = 0,
    max_steps: int = 600,
    # Env kwargs
    dt_sim: int = 5,
    dt_env: int = 10,
    yaw_step: float = 5.0,
    max_eps: int = 60,
    TI_type: str = "Random",
):
    """Collect all episodes for a single layout and save to HDF5."""
    print(f"\n{'='*60}")
    print(f"Collecting data for layout: {layout_name}")
    print(f"{'='*60}")

    # Get positions
    x_pos, y_pos = get_layout_positions(layout_name, wind_turbine)
    rotor_diameter = float(wind_turbine.diameter())
    n_turb = len(x_pos)

    # Compute profiles
    receptivity, influence = None, None
    if compute_profiles_flag:
        print(f"  Computing {profile_source} profiles ({n_profile_directions} dirs)...")
        receptivity, influence = compute_profiles(
            x_pos, y_pos, wind_turbine,
            profile_source=profile_source,
            n_directions=n_profile_directions,
        )
        print(f"  Profiles shape: {receptivity.shape}")

    # Create HDF5 file
    filepath = os.path.join(output_dir, f"layout_{layout_name}.h5")
    create_layout_file(
        filepath=filepath,
        layout_name=layout_name,
        x_pos=x_pos,
        y_pos=y_pos,
        rotor_diameter=rotor_diameter,
        turbine_type=wind_turbine.__class__.__name__,
        receptivity=receptivity,
        influence=influence,
        n_profile_directions=n_profile_directions,
    )

    # Create environment
    env = make_single_env(
        x_pos=x_pos,
        y_pos=y_pos,
        wind_turbine=wind_turbine,
        config=config,
        seed=base_seed,
        dt_sim=dt_sim,
        dt_env=dt_env,
        yaw_step=yaw_step,
        max_eps=max_eps,
        TI_type=TI_type,
    )

    # Collect episodes
    total_steps = 0
    t_start = time.time()

    for ep_idx in range(episodes_per_layout):
        ep_seed = base_seed + ep_idx
        env.action_space.seed(ep_seed)

        episode_data = collect_episode(env, policy=policy, max_steps=max_steps)
        n_steps = episode_data["ws"].shape[0]
        total_steps += n_steps

        append_episode(
            filepath=filepath,
            ws=episode_data["ws"],
            wd=episode_data["wd"],
            yaw=episode_data["yaw"],
            power=episode_data["power"],
            actions=episode_data["actions"],
            mean_ws=episode_data["mean_ws"],
            mean_wd=episode_data["mean_wd"],
            mean_ti=episode_data["mean_ti"],
            seed=ep_seed,
        )

        if (ep_idx + 1) % 10 == 0 or ep_idx == 0:
            elapsed = time.time() - t_start
            eps_per_sec = (ep_idx + 1) / elapsed
            eta = (episodes_per_layout - ep_idx - 1) / eps_per_sec
            print(f"  Episode {ep_idx + 1}/{episodes_per_layout} | "
                  f"steps={n_steps} | ws={episode_data['mean_ws']:.1f} | "
                  f"wd={episode_data['mean_wd']:.0f}° | ti={episode_data['mean_ti']:.3f} | "
                  f"ETA: {eta:.0f}s")

    env.close()

    elapsed = time.time() - t_start
    print(f"\n  Done: {episodes_per_layout} episodes, {total_steps} total steps in {elapsed:.1f}s")
    print(f"  Saved to: {filepath}")
    return filepath


# =============================================================================
# DATASET INSPECTION
# =============================================================================

def inspect_dataset(filepath: str):
    """Print summary of a collected dataset."""
    with h5py.File(filepath, "r") as f:
        print(f"\n{'='*60}")
        print(f"Dataset: {filepath}")
        print(f"{'='*60}")
        print(f"  Layout: {f.attrs['layout_name']}")
        print(f"  Turbines: {f.attrs['n_turbines']}")
        print(f"  Rotor diameter: {f.attrs['rotor_diameter']:.1f}m")
        print(f"  Turbine type: {f.attrs['turbine_type']}")

        n_episodes = len(f["episodes"])
        print(f"  Episodes: {n_episodes}")

        if "profiles" in f:
            print(f"  Profiles: {f['profiles/receptivity'].shape}")

        # Summarize episodes
        total_steps = 0
        ws_vals, wd_vals, ti_vals = [], [], []
        for ep_key in f["episodes"]:
            ep = f[f"episodes/{ep_key}"]
            total_steps += ep.attrs["n_steps"]
            ws_vals.append(ep.attrs["mean_ws"])
            wd_vals.append(ep.attrs["mean_wd"])
            ti_vals.append(ep.attrs["mean_ti"])

        print(f"  Total steps: {total_steps}")
        if ws_vals:
            print(f"  Wind speed:  {np.mean(ws_vals):.1f} ± {np.std(ws_vals):.1f} m/s "
                  f"[{np.min(ws_vals):.1f}, {np.max(ws_vals):.1f}]")
            print(f"  Wind dir:    {np.mean(wd_vals):.0f} ± {np.std(wd_vals):.0f}° "
                  f"[{np.min(wd_vals):.0f}, {np.max(wd_vals):.0f}]")
            print(f"  Turbulence:  {np.mean(ti_vals):.3f} ± {np.std(ti_vals):.3f} "
                  f"[{np.min(ti_vals):.3f}, {np.max(ti_vals):.3f}]")


# =============================================================================
# CLI ENTRYPOINT
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Collect pretraining data for wind farm transformer")

    # Layout and output
    parser.add_argument("--layouts", type=str, required=True,
                        help="Comma-separated layout names (e.g. 'grid_3x3,grid_5x5')")
    parser.add_argument("--output_dir", type=str, default="./pretrain_data",
                        help="Output directory for HDF5 files")
    parser.add_argument("--episodes_per_layout", type=int, default=100,
                        help="Number of episodes to collect per layout")
    parser.add_argument("--policy", type=str, default="random", choices=["random", "greedy", "pywake"],
                        help="Data collection policy")

    # Turbine and environment
    parser.add_argument("--turbtype", type=str, default="DTU10MW")
    parser.add_argument("--TI_type", type=str, default="Random")
    parser.add_argument("--config", type=str, default="basic")
    parser.add_argument("--dt_sim", type=int, default=5)
    parser.add_argument("--dt_env", type=int, default=10)
    parser.add_argument("--yaw_step", type=float, default=5.0)
    parser.add_argument("--max_eps", type=int, default=60)
    parser.add_argument("--max_steps", type=int, default=600)

    # Profiles
    parser.add_argument("--profile_source", type=str, default="geometric",
                        choices=["geometric", "pywake"])
    parser.add_argument("--n_profile_directions", type=int, default=360)
    parser.add_argument("--no_profiles", action="store_true",
                        help="Skip profile computation")

    # Seeds
    parser.add_argument("--seed", type=int, default=0)

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup
    layout_names = [l.strip() for l in args.layouts.split(",")]
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Collecting pretraining data")
    print(f"  Layouts: {layout_names}")
    print(f"  Episodes per layout: {args.episodes_per_layout}")
    print(f"  Policy: {args.policy}")
    print(f"  Output: {args.output_dir}")

    # Create wind turbine
    wind_turbine = make_wind_turbine(args.turbtype)
    config = make_env_config(args.config)

    # Collect data for each layout
    all_files = []
    for li, layout_name in enumerate(layout_names):
        filepath = collect_layout_data(
            layout_name=layout_name,
            output_dir=args.output_dir,
            wind_turbine=wind_turbine,
            config=config,
            episodes_per_layout=args.episodes_per_layout,
            policy=args.policy,
            profile_source=args.profile_source,
            n_profile_directions=args.n_profile_directions,
            compute_profiles_flag=not args.no_profiles,
            base_seed=args.seed + li * 10000,
            dt_sim=args.dt_sim,
            dt_env=args.dt_env,
            yaw_step=args.yaw_step,
            max_eps=args.max_eps,
            TI_type=args.TI_type,
        )
        all_files.append(filepath)

    # Print summaries
    print(f"\n\n{'='*60}")
    print("COLLECTION COMPLETE")
    print(f"{'='*60}")
    for filepath in all_files:
        inspect_dataset(filepath)


if __name__ == "__main__":
    main()