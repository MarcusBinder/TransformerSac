"""
Data Collection Script for Wind Farm Pretraining Datasets

Collects per-turbine measurements from WindGym environments and stores them
in HDF5 format for self-supervised pretraining of transformer encoders.

HDF5 Structure:
    layout_<name>.h5
    ├── attrs: layout_name, n_turbines, rotor_diameter, turbine_type, policy
    ├── positions/xy                    # (n_turbines, 2)
    ├── profiles/receptivity            # (n_turbines, n_directions)  [optional]
    ├── profiles/influence              # (n_turbines, n_directions)  [optional]
    └── episodes/
        └── ep_XXXX/
            ├── attrs: n_steps, mean_ws, mean_wd, mean_ti, seed
            ├── ws           # (n_steps, n_turbines)
            ├── wd           # (n_steps, n_turbines)
            ├── yaw          # (n_steps, n_turbines)
            ├── power        # (n_steps, n_turbines)
            ├── rewards      # (n_steps,)
            ├── actions_wind # (n_steps, n_turbines)  — target setpoint in [-1, 1]
            └── actions_yaw  # (n_steps, n_turbines)  — delta yaw / max_delta, clipped [-1, 1]

Usage:
    python make_dataset.py \\
        --layouts "grid_3x3,grid_5x5,hornsrev" \\
        --episodes_per_layout 100 \\
        --output_dir ./pretrain_data \\
        --n_profile_directions 360 \\
        --profile_source geometric \\
        --n_workers 16

Author: Marcus (DTU Wind Energy)
"""

import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import h5py
import gymnasium as gym
from multiprocessing import Pool, cpu_count

import tyro

# WindGym imports
from WindGym import WindFarmEnv

from helpers.layouts import get_layout_positions
from helpers.env_configs import make_env_config

from WindGym.Agents import GreedyAgent, PyWakeAgent

# TODO - Add power output with NO wakes.

# =============================================================================
# CLI CONFIG
# =============================================================================

@dataclass
class Args:
    """Collect pretraining data for wind farm transformer."""

    # === Layout and Output ===
    layouts: str = "grid_3x3"
    """Comma-separated layout names (e.g. 'grid_3x3,grid_5x5')"""
    output_dir: str = "./pretrain_data"
    """Output directory for HDF5 files"""
    episodes_per_layout: int = 100
    """Number of episodes to collect per layout"""
    policy: str = "random"
    """Data collection policy: 'random', 'greedy', or 'pywake'"""

    # === Turbine and Environment ===
    turbtype: str = "DTU10MW"
    """Wind turbine type"""
    TI_type: str = "Random"
    """Turbulence intensity sampling"""
    config: str = "basic"
    """Environment config preset"""
    dt_sim: int = 5
    """Simulation timestep (seconds)"""
    dt_env: int = 10
    """Environment timestep (seconds)"""
    yaw_step: float = 5.0
    """Max yaw change per sim step (degrees)"""
    max_eps: int = 60
    """Number of flow passthroughs per episode"""
    max_steps: int = 600
    """Safety limit for max steps per episode"""

    # === Profiles ===
    profile_source: str = "geometric"
    """Profile computation source: 'geometric' or 'pywake'"""
    n_profile_directions: int = 360
    """Number of directions in profile"""
    no_profiles: bool = False
    """Skip profile computation"""

    # === Parallelism ===
    n_workers: int = 1
    """Number of parallel workers (0 = all available cores)"""

    # === Seeds ===
    seed: int = 0
    """Base random seed"""


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
    policy: str = "random",
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
        f.attrs["policy"] = policy

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
    ep_idx: int,
    ws: np.ndarray,       # (n_steps, n_turbines)
    wd: np.ndarray,       # (n_steps, n_turbines)
    yaw: np.ndarray,      # (n_steps, n_turbines)
    power: np.ndarray,    # (n_steps, n_turbines)
    actions_wind,   # (n_steps, n_turbines)
    actions_yaw,    # (n_steps, n_turbines)
    rewards,        # (n_steps,)
    mean_ws: float,
    mean_wd: float,
    mean_ti: float,
    seed: int = -1,
):
    """Append a completed episode to the layout file with a specific index."""
    with h5py.File(filepath, "a") as f:
        eps_group = f["episodes"]
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
        ep.create_dataset("rewards",      data=rewards.astype(np.float32))
        ep.create_dataset("actions_wind", data=actions_wind.astype(np.float32))
        ep.create_dataset("actions_yaw",  data=actions_yaw.astype(np.float32))


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

def greedy_controller(fs, yaw_max: float):
    """Wind-based: target yaw that aligns each turbine with its local (wake-inclusive) wind."""
    wind_dir_local = fs.get_wind_direction(xyz=fs.windTurbines.rotor_positions_xyz, include_wakes=True)
    # Target offset = what yaw the turbine needs to be at to face local wind
    target_yaw = wind_dir_local - fs.wind_direction   # negative of the misalignment
    return np.clip(target_yaw / yaw_max, -1.0, 1.0)

def pywake_agent(optimal_yaws, yaw_max):
    """Wind-based: target setpoint as fraction of yaw limit."""
    return np.clip(optimal_yaws / yaw_max, -1.0, 1.0)

# =============================================================================
# EPISODE COLLECTION
# =============================================================================

def collect_episode(env, policy="random", max_steps=600) -> dict:
    """
    Run one episode and collect per-turbine measurements.

    Args:
        env: WindFarmEnv instance (reset will be called)
        policy: "random" for uniform random actions, "greedy" for zero yaw
        max_steps: safety limit to prevent infinite loops
    Returns:
        dict with keys: ws, wd, yaw, power, actions (each (n_steps, n_turb)),
        plus mean_ws, mean_wd, mean_ti scalars
    """
    obs, info = env.reset()
    base_env = env  # No wrapper, use env directly
    n_turb = base_env.n_turb

    # Episode-level metadata (constant within episode)
    mean_ws = float(base_env.ws)
    mean_wd = float(base_env.wd)
    mean_ti = float(base_env.ti)

    # Collect step data
    ws_list, wd_list, yaw_list, power_list = [], [], [], []
    actions_wind_list, actions_yaw_list, reward_list = [], [], []

    done = False
    current_step = 0

    if policy == "pywake":
        py_agent = PyWakeAgent(
            x_pos=base_env.x_pos, y_pos=base_env.y_pos,
            turbine=base_env.turbine, yaw_max=30, yaw_min=-30, # Use 30 because its better for the internal optimizer
            look_up=False, env=base_env,
        )
        py_agent.update_wind(wind_speed=base_env.ws, wind_direction=base_env.wd, TI=base_env.ti)
        py_agent.optimize()

    while not done:
        measurements = base_env.farm_measurements.get_measurements(scaled=False)
        mes = measurements.reshape(n_turb, 4)   # [ws, wd, yaw, power]
        ws_list.append(mes[:, 0].copy())
        wd_list.append(mes[:, 1].copy())
        yaw_list.append(mes[:, 2].copy())
        power_list.append(mes[:, 3].copy())
        yaw_before = mes[:, 2].copy()

        # Select action
        # Wind-based action (target setpoint)
        if policy == "random":
            action = env.action_space.sample()
        elif policy == "greedy":
            action = greedy_controller(base_env.fs, yaw_max=base_env.yaw_max)
        elif policy == "pywake":
            action = pywake_agent(py_agent.optimized_yaws, yaw_max=base_env.yaw_max)
        else:
            raise ValueError(f"Unknown policy: {policy}")

        actions_wind_list.append(action.flatten()[:n_turb].copy())

        # Step
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        reward_list.append(float(reward))

        # Post-step yaw → derive equivalent yaw-based action
        mes_after = base_env.farm_measurements.get_measurements(scaled=False).reshape(n_turb, 4)
        yaw_after = mes_after[:, 2].copy()
        yaw_action = np.clip(
            (yaw_after - yaw_before) / base_env.yaw_step_env, -1.0, 1.0
        )
        actions_yaw_list.append(yaw_action)

        current_step += 1
        if current_step >= max_steps:
            print(f"Reached max_steps={max_steps}, ending episode.")
            done = True

    return {
        "ws":           np.stack(ws_list,           axis=0),
        "wd":           np.stack(wd_list,           axis=0),
        "yaw":          np.stack(yaw_list,          axis=0),
        "power":        np.stack(power_list,        axis=0),
        "rewards":      np.array(reward_list,       dtype=np.float32),
        "actions_wind": np.stack(actions_wind_list, axis=0),
        "actions_yaw":  np.stack(actions_yaw_list,  axis=0),
        "mean_ws": mean_ws,
        "mean_wd": mean_wd,
        "mean_ti": mean_ti,
    }


# =============================================================================
# PARALLEL EPISODE WORKER
# =============================================================================

def _collect_single_episode(args: tuple) -> dict:
    """
    Worker function for parallel episode collection.

    Each worker creates its own environment instance to avoid shared state,
    collects one episode, and returns the data dict (including ep_idx/seed).
    """
    (
        ep_idx, ep_seed, x_pos, y_pos, turbtype, config,
        policy, max_steps, dt_sim, dt_env, yaw_step, max_eps, TI_type,
    ) = args

    # Each worker creates its own env and turbine (no shared state)
    wind_turbine = make_wind_turbine(turbtype)
    env = make_single_env(
        x_pos=x_pos,
        y_pos=y_pos,
        wind_turbine=wind_turbine,
        config=config,
        seed=ep_seed,
        dt_sim=dt_sim,
        dt_env=dt_env,
        yaw_step=yaw_step,
        max_eps=max_eps,
        TI_type=TI_type,
    )

    if env.ActionMethod.lower() == "yaw":
        print("The action method defined in the config was yaw. We change this to wind to ensure the correct action space is used for data collection.")
        env.ActionMethod = "wind"
    episode_data = collect_episode(env, policy=policy, max_steps=max_steps)
    env.close()

    episode_data["ep_idx"] = ep_idx
    episode_data["seed"] = ep_seed
    return episode_data


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
        from helpers.geometric_profiles import compute_layout_profiles_vectorized
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
        from helpers.receptivity_profiles import compute_layout_profiles
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
    n_workers: int = 1,
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
    turbtype = wind_turbine.__class__.__name__

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
    filepath = os.path.join(output_dir, f"layout_{layout_name}_{policy}.h5")
    create_layout_file(
        filepath=filepath,
        layout_name=layout_name,
        x_pos=x_pos,
        y_pos=y_pos,
        rotor_diameter=rotor_diameter,
        turbine_type=turbtype,
        policy=policy,
        receptivity=receptivity,
        influence=influence,
        n_profile_directions=n_profile_directions,
    )

    # Build worker arguments for all episodes
    worker_args = [
        (
            ep_idx, base_seed + ep_idx, x_pos, y_pos, turbtype, config,
            policy, max_steps, dt_sim, dt_env, yaw_step, max_eps, TI_type,
        )
        for ep_idx in range(episodes_per_layout)
    ]

    t_start = time.time()
    total_steps = 0
    effective_workers = min(n_workers, episodes_per_layout)

    if effective_workers <= 1:
        # ── Sequential fallback ──
        print(f"  Running {episodes_per_layout} episodes sequentially...")
        for args in worker_args:
            ep_data = _collect_single_episode(args)
            ep_idx = ep_data["ep_idx"]
            total_steps += ep_data["ws"].shape[0]

            append_episode(
                filepath=filepath,
                ep_idx=ep_idx,
                ws=ep_data["ws"],
                wd=ep_data["wd"],
                yaw=ep_data["yaw"],
                power=ep_data["power"],
                actions_wind=ep_data["actions_wind"],
                actions_yaw=ep_data["actions_yaw"],
                rewards=ep_data["rewards"],
                mean_ws=ep_data["mean_ws"],
                mean_wd=ep_data["mean_wd"],
                mean_ti=ep_data["mean_ti"],
                seed=ep_data["seed"],
            )

            if (ep_idx + 1) % 10 == 0 or ep_idx == 0:
                elapsed = time.time() - t_start
                eps_per_sec = (ep_idx + 1) / elapsed
                eta = (episodes_per_layout - ep_idx - 1) / eps_per_sec
                print(f"  Episode {ep_idx + 1}/{episodes_per_layout} | "
                      f"steps={ep_data['ws'].shape[0]} | ws={ep_data['mean_ws']:.1f} | "
                      f"wd={ep_data['mean_wd']:.0f}\u00b0 | ti={ep_data['mean_ti']:.3f} | "
                      f"ETA: {eta:.0f}s")
    else:
        # ── Parallel collection ──
        print(f"  Running {episodes_per_layout} episodes with {effective_workers} workers...")
        with Pool(processes=effective_workers) as pool:
            results = pool.imap_unordered(_collect_single_episode, worker_args)

            completed = 0
            for ep_data in results:
                ep_idx = ep_data["ep_idx"]
                n_steps = ep_data["ws"].shape[0]
                total_steps += n_steps

                append_episode(
                    filepath=filepath,
                    ep_idx=ep_idx,
                    ws=ep_data["ws"],
                    wd=ep_data["wd"],
                    yaw=ep_data["yaw"],
                    power=ep_data["power"],
                    actions_wind=ep_data["actions_wind"],
                    actions_yaw=ep_data["actions_yaw"],
                    rewards=ep_data["rewards"],
                    mean_ws=ep_data["mean_ws"],
                    mean_wd=ep_data["mean_wd"],
                    mean_ti=ep_data["mean_ti"],
                    seed=ep_data["seed"],
                )

                completed += 1
                if completed % 10 == 0 or completed == 1:
                    elapsed = time.time() - t_start
                    eps_per_sec = completed / elapsed
                    eta = (episodes_per_layout - completed) / eps_per_sec
                    print(f"  Completed {completed}/{episodes_per_layout} | "
                          f"steps={n_steps} | ws={ep_data['mean_ws']:.1f} | "
                          f"wd={ep_data['mean_wd']:.0f}\u00b0 | ti={ep_data['mean_ti']:.3f} | "
                          f"ETA: {eta:.0f}s")

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
        print(f"  Policy: {f.attrs.get('policy', 'N/A')}")

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
            print(f"  Wind speed:  {np.mean(ws_vals):.1f} \u00b1 {np.std(ws_vals):.1f} m/s "
                  f"[{np.min(ws_vals):.1f}, {np.max(ws_vals):.1f}]")
            print(f"  Wind dir:    {np.mean(wd_vals):.0f} \u00b1 {np.std(wd_vals):.0f}\u00b0 "
                  f"[{np.min(wd_vals):.0f}, {np.max(wd_vals):.0f}]")
            print(f"  Turbulence:  {np.mean(ti_vals):.3f} \u00b1 {np.std(ti_vals):.3f} "
                  f"[{np.min(ti_vals):.3f}, {np.max(ti_vals):.3f}]")


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = tyro.cli(Args)

    # Setup
    layout_names = [l.strip() for l in args.layouts.split(",")]
    os.makedirs(args.output_dir, exist_ok=True)

    n_workers = args.n_workers if args.n_workers > 0 else cpu_count()

    print(f"Collecting pretraining data")
    print(f"  Layouts: {layout_names}")
    print(f"  Episodes per layout: {args.episodes_per_layout}")
    print(f"  Policy: {args.policy}")
    print(f"  Workers: {n_workers}")
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
            max_steps=args.max_steps,
            n_workers=n_workers,
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