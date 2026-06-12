"""
Single-cell checkpoint evaluation: ONE checkpoint x ONE eval layout -> ONE NetCDF.

Refactor of eval_ablation.py for SLURM-array parallelism: instead of one
process looping over every run and checkpoint, each invocation evaluates a
single cell so the cluster can fan out one (checkpoint, layout) pair per node
and the campaign is resumable at file granularity (see eval_stage_0.sh).

The episode seeding scheme is copied VERBATIM from eval_ablation.py
(env construction seeds seed+i, batch reset seeds seed + batch*num_envs), so
with the same --num-envs/--seed every checkpoint sees bit-identical inflow
episodes, and results are directly comparable to the old eval numbers.

Usage:
    python eval_checkpoint.py \
        --checkpoint runs/H1_baseline_..._s1/checkpoints/step_350000.pt \
        --eval-layout E4 --out eval_E4_step_350000.nc
"""

import argparse
import os
import re
import sys
import time
import numpy as np
import xarray as xr
import torch

# ---- Imports from your codebase ----
import gymnasium as gym
from config import Args
from networks import TransformerActor, create_profile_encoding
from helpers.agent import WindFarmAgent
from helpers.layouts import get_layout_positions
from helpers.env_configs import make_env_config
from helpers.multi_layout_env import MultiLayoutEnv, LayoutConfig
from helpers.helper_funcs import EnhancedPerTurbineWrapper
from helpers.receptivity_profiles import compute_layout_profiles
from WindGym import WindFarmEnv
from WindGym.wrappers import RecordEpisodeVals, PerTurbineObservationWrapper


def load_actor_from_checkpoint(checkpoint_path: str, device: torch.device):
    """Load actor network from checkpoint (handles old and new formats)."""
    data = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint = data
    args = data["args"]
    # Keep as namespace for attribute access; convert to dict only when needed via vars()
    return checkpoint, args


def create_eval_env(layout: str, args: dict, turbbox_path: str, seed: int = 42, n_envs: int = 1):
    """Create evaluation environment for a single layout."""

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
        layout_to_use = LayoutConfig(name=name, x_pos=x_pos, y_pos=y_pos)

        if args["profile_encoding_type"] is not None:
            if args["profile_source"].lower() == "geometric":
                from helpers.geometric_profiles import compute_layout_profiles_vectorized

                D = wind_turbine.diameter()
                print(f"Computing GEOMETRIC profiles for layout: {name}")
                receptivity_profiles, influence_profiles = compute_layout_profiles_vectorized(
                    x_pos, y_pos,
                    rotor_diameter=D,
                    k_wake=0.04,
                    n_directions=args["n_profile_directions"],
                    sigma_smooth=10.0,
                    scale_factor=15.0,
                )
            elif args["profile_source"].lower() == "pywake":
                print(f"Computing PyWake profiles for layout: {name}")
                receptivity_profiles, influence_profiles = compute_layout_profiles(
                    x_pos, y_pos, wind_turbine,
                    n_directions=args["n_profile_directions"],
                )
            else:
                raise ValueError(
                    f"Unknown profile_source: {args['profile_source']}. "
                    f"Use 'pywake' or 'geometric'."
                )

            layout_to_use.receptivity_profiles = receptivity_profiles
            layout_to_use.influence_profiles = influence_profiles

        layouts.append(layout_to_use)

    config = make_env_config("hard")

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
        "TurbBox": turbbox_path,
        "config": config,
        "turbtype": args["TI_type"],
        "dt_sim": args["dt_sim"],
        "dt_env": args["dt_env"],
        "yaw_step_sim": args["yaw_step"],
    }

    def env_factory(x_pos, y_pos):
        env = WindFarmEnv(x_pos=x_pos,
                          y_pos=y_pos,
                          yaw_init="Zeros",
                          reset_init=False,
                          **base_env_kwargs)
        env.action_space.seed(seed)
        return env

    def combined_wrapper(env):
        env = PerTurbineObservationWrapper(env)
        if args["use_wd_deviation"]:
            env = EnhancedPerTurbineWrapper(env, wd_scale_range=args["wd_scale_range"])
        return env

    def make_env_fn(seed):
        def _init():
            env = MultiLayoutEnv(
                layouts=layouts,
                env_factory=env_factory,
                per_turbine_wrapper=combined_wrapper,
                seed=seed,
                shuffle=False,
                max_episode_steps=9999999,
            )
            return env
        return _init

    env = gym.vector.AsyncVectorEnv(
        [make_env_fn(seed=seed + i) for i in range(n_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    env = RecordEpisodeVals(env)

    return env, wind_turbine


def build_agent(args: dict, env, device: torch.device):
    """Reconstruct the actor + agent from checkpoint args and the eval env
    (extracted from eval_ablation.py's evaluate_run_on_layout)."""
    rotor_diameter = env.env.get_attr('rotor_diameter')[0]
    obs_dim_per_turbine = env.single_observation_space.shape[-1]
    action_high = env.single_action_space.high[0]
    action_low = env.single_action_space.low[0]
    action_scale = (action_high - action_low) / 2.0
    action_bias = (action_high + action_low) / 2.0

    # Profile encoders
    use_profiles = args["profile_encoding_type"] is not None
    if use_profiles and args["share_profile_encoder"]:
        shared_recep_encoder, shared_influence_encoder = create_profile_encoding(
            profile_type=args["profile_encoding_type"],
            embed_dim=args["embed_dim"],
            hidden_channels=args["profile_encoder_hidden"],
        )
        shared_recep_encoder = shared_recep_encoder.to(device)
        shared_influence_encoder = shared_influence_encoder.to(device)
    else:
        shared_recep_encoder = None
        shared_influence_encoder = None

    # Convert to namespace for the constructor
    from argparse import Namespace
    args_ns = Namespace(**args)

    actor = TransformerActor(
        obs_dim_per_turbine=obs_dim_per_turbine,
        action_dim_per_turbine=1,
        embed_dim=args_ns.embed_dim,
        pos_embed_dim=args_ns.pos_embed_dim,
        num_heads=args_ns.num_heads,
        num_layers=args_ns.num_layers,
        mlp_ratio=args_ns.mlp_ratio,
        dropout=0.0,
        action_scale=action_scale,
        action_bias=action_bias,
        pos_encoding_type=args_ns.pos_encoding_type,
        rel_pos_hidden_dim=args_ns.rel_pos_hidden_dim,
        rel_pos_per_head=args_ns.rel_pos_per_head,
        pos_embedding_mode=args_ns.pos_embedding_mode,
        profile_encoding=args_ns.profile_encoding_type,
        profile_encoder_hidden=args_ns.profile_encoder_hidden,
        n_profile_directions=args_ns.n_profile_directions,
        profile_fusion_type=args_ns.profile_fusion_type,
        profile_embed_mode=args_ns.profile_embed_mode,
        shared_recep_encoder=shared_recep_encoder,
        shared_influence_encoder=shared_influence_encoder,
        args=args_ns,
    ).to(device)

    agent = WindFarmAgent(
        actor=actor,
        device=device,
        rotor_diameter=rotor_diameter,
        use_wind_relative=args["use_wind_relative_pos"],
        use_profiles=use_profiles,
        rotate_profiles=args["rotate_profiles"],
    )
    return actor, agent


def evaluate_checkpoint_episodes(env, agent, num_batches, num_envs, num_steps, deterministic, seed=None, progress_prefix=""):
    all_episodes = []

    for batch in range(num_batches):
        batch_start = time.time()
        # Deterministic seeding: each batch gets a unique but reproducible seed
        # so every checkpoint sees the exact same set of episodes.
        if seed is not None:
            batch_seed = seed + batch * num_envs
            obs, infos = env.reset(seed=batch_seed)
        else:
            obs, infos = env.reset()
        wind_speeds = infos["Wind speed Global"]
        wind_directions = infos["Wind direction Global"]

        rewards_per_env = [[0.0] for _ in range(num_envs)]
        powers_per_env = [[] for _ in range(num_envs)]
        baseline_powers_per_env = [[] for _ in range(num_envs)]
        yaws_per_env = [[] for _ in range(num_envs)]

        for i in range(num_envs):
            if "Power agent" in infos:
                powers_per_env[i].append(infos["Power agent"][i])
            if "Power baseline" in infos:
                baseline_powers_per_env[i].append(infos["Power baseline"][i])
            if "yaw angles agent" in infos:
                yaws_per_env[i].append(infos["yaw angles agent"][i])

        for step in range(num_steps):
            actions = agent.act(env, obs, deterministic=deterministic)
            obs, rewards, terminateds, truncateds, infos = env.step(actions)

            for i in range(num_envs):
                rewards_per_env[i].append(rewards[i])
                if "Power agent" in infos:
                    powers_per_env[i].append(infos["Power agent"][i])
                if "Power baseline" in infos:
                    baseline_powers_per_env[i].append(infos["Power baseline"][i])
                if "yaw angles agent" in infos:
                    yaws_per_env[i].append(infos["yaw angles agent"][i])

        for i in range(num_envs):
            episode_data = {
                "wind_speed": wind_speeds[i].item(),
                "wind_direction": wind_directions[i].item(),
                "rewards": rewards_per_env[i],
                "powers": powers_per_env[i],
                "baseline_powers": baseline_powers_per_env[i],
                "yaw_angles": yaws_per_env[i],
                "episode_return": float(sum(rewards_per_env[i])),
                "episode_length": len(rewards_per_env[i]),
                "mean_power": np.mean(powers_per_env[i]) if powers_per_env[i] else None,
                "mean_baseline_power": np.mean(baseline_powers_per_env[i]) if baseline_powers_per_env[i] else None,
            }
            all_episodes.append(episode_data)

        print(f"{progress_prefix}      batch {batch + 1}/{num_batches} "
              f"({time.time() - batch_start:.1f}s)")

    return all_episodes


def parse_run_name(run_name):
    """
    Parse ablation run names like:
        'A1_noSpatial_T1_T2_T3_T4_T5_T6_s1'  -> ('A1_noSpatial', 'T1,T2,T3,T4,T5,T6', 1)
        'A4_smallLayout_T1_T2_T3_s2'           -> ('A4_smallLayout', 'T1,T2,T3', 2)
    Returns (config_name, layout_csv, seed) or (None, None, None).
    """
    match = re.match(r'^(.+?)_((?:[TE]\d+_)*[TE]\d+)_s(\d+)$', run_name)
    if not match:
        return None, None, None
    config_name = match.group(1)
    layout_str = match.group(2).replace("_", ",")  # T1_T2_T3 -> T1,T2,T3
    seed = int(match.group(3))
    return config_name, layout_str, seed


def build_dataset(episodes, num_episodes, num_steps, meta):
    """Pack the episode records into an xarray Dataset with dims
    (episode, time, turbine) -- eval_ablation.py's layout minus the step dim,
    since each output file holds exactly one checkpoint."""
    max_time = num_steps + 1

    sample_yaws = episodes[0]["yaw_angles"]
    n_turbines = len(sample_yaws[0])

    yaw_array = np.full((num_episodes, max_time, n_turbines), np.nan)
    power_array = np.full((num_episodes, max_time), np.nan)
    baseline_power_array = np.full((num_episodes, max_time), np.nan)
    reward_array = np.full((num_episodes, max_time), np.nan)
    episode_return_array = np.full((num_episodes,), np.nan)

    for ei, ep in enumerate(episodes):
        for ti, yaw_snapshot in enumerate(ep["yaw_angles"]):
            yaw_array[ei, ti, :] = yaw_snapshot
        for ti, p in enumerate(ep["powers"]):
            power_array[ei, ti] = p
        for ti, b in enumerate(ep["baseline_powers"]):
            baseline_power_array[ei, ti] = b
        for ti, r in enumerate(ep["rewards"]):
            reward_array[ei, ti] = r
        episode_return_array[ei] = ep["episode_return"]

    # Per-timestep power gain percentage
    with np.errstate(divide="ignore", invalid="ignore"):
        power_gain_pct_ts = (power_array / baseline_power_array - 1.0) * 100.0

    episode_returns = [ep["episode_return"] for ep in episodes]
    episode_powers = [ep["mean_power"] for ep in episodes if ep["mean_power"] is not None]
    baseline_powers = [ep["mean_baseline_power"] for ep in episodes if ep["mean_baseline_power"] is not None]

    mean_power = np.mean(episode_powers) if episode_powers else np.nan
    mean_baseline_power = np.mean(baseline_powers) if baseline_powers else np.nan
    power_gain_pct = (
        (mean_power / mean_baseline_power - 1) * 100
        if episode_powers and baseline_powers else np.nan
    )

    ds = xr.Dataset(
        {
            # Scalar summaries (0-d), same definitions as eval_ablation.py
            "mean_return": ((), np.mean(episode_returns)),
            "std_return": ((), np.std(episode_returns)),
            "mean_power": ((), mean_power),
            "mean_baseline_power": ((), mean_baseline_power),
            "power_gain_pct": ((), power_gain_pct),
            "mean_length": ((), np.mean([ep["episode_length"] for ep in episodes])),
            # Full trajectories
            "yaw_angles": (["episode", "time", "turbine"], yaw_array),
            "power": (["episode", "time"], power_array),
            "baseline_power": (["episode", "time"], baseline_power_array),
            "power_gain_pct_ts": (["episode", "time"], power_gain_pct_ts),
            "reward": (["episode", "time"], reward_array),
            "episode_return": (["episode"], episode_return_array),
            "wind_speed": (["episode"], [ep["wind_speed"] for ep in episodes]),
            "wind_direction": (["episode"], [ep["wind_direction"] for ep in episodes]),
        },
        coords={
            "episode": np.arange(num_episodes),
            "time": np.arange(max_time),
            "turbine": np.arange(n_turbines),
        },
        attrs=meta,
    )
    return ds


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ONE checkpoint on ONE layout, write ONE NetCDF.")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to runs/<run>/checkpoints/step_<N>.pt")
    parser.add_argument("--eval-layout", required=True,
                        help="Eval layout name (e.g. E4), or comma-separated list")
    parser.add_argument("--out", required=True, help="Output NetCDF path")
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--num-steps", type=int, default=250)
    parser.add_argument("--num-envs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--turbbox-path", default="./boxes/")
    parser.add_argument("--stochastic", action="store_true",
                        help="Sample actions instead of deterministic mean")
    cli = parser.parse_args()

    # Line-buffer stdout/stderr so progress shows up promptly in SLURM logs.
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    deterministic = not cli.stochastic
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Identify the cell from the checkpoint path: runs/<run_name>/checkpoints/step_<N>.pt
    ckpt_path = os.path.abspath(cli.checkpoint)
    run_name = os.path.basename(os.path.dirname(os.path.dirname(ckpt_path)))
    step_match = re.search(r"step_(\d+)", os.path.basename(ckpt_path))
    step = int(step_match.group(1)) if step_match else -1
    config_name, train_layout, train_seed = parse_run_name(run_name)

    print(f"[Eval] run={run_name} step={step} layout={cli.eval_layout} "
          f"episodes={cli.num_episodes} steps={cli.num_steps} envs={cli.num_envs} "
          f"seed={cli.seed} deterministic={deterministic} device={device}")

    # ---- Load checkpoint + args (with default backfill for older checkpoints) ----
    checkpoint, args_ns = load_actor_from_checkpoint(ckpt_path, device)
    args = vars(args_ns) if hasattr(args_ns, '__dict__') and not isinstance(args_ns, dict) else args_ns

    # Older checkpoints predate some config flags; backfill from current Args()
    # defaults so attribute/dict access never KeyErrors. Checkpoint values win.
    _defaults = vars(Args())
    _missing = [k for k in _defaults if k not in args]
    args = {**_defaults, **args}
    if _missing:
        print(f"[Eval] Backfilled {len(_missing)} missing args from defaults: {sorted(_missing)}")

    # ---- Env + agent ----
    assert cli.num_episodes % cli.num_envs == 0, (
        f"num_episodes={cli.num_episodes} must be divisible by num_envs={cli.num_envs}"
    )
    num_batches = cli.num_episodes // cli.num_envs

    env, _ = create_eval_env(
        layout=cli.eval_layout, args=args, turbbox_path=cli.turbbox_path,
        seed=cli.seed, n_envs=cli.num_envs,
    )
    actor, agent = build_agent(args, env, device)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()

    # ---- Evaluate ----
    t0 = time.time()
    episodes = evaluate_checkpoint_episodes(
        env, agent, num_batches, cli.num_envs, cli.num_steps, deterministic,
        seed=cli.seed,
    )
    env.close()
    print(f"[Eval] {len(episodes)} episodes in {time.time() - t0:.1f}s")

    # ---- Dataset + atomic save ----
    ds = build_dataset(
        episodes, cli.num_episodes, cli.num_steps,
        meta={
            "run_name": run_name,
            "config_name": config_name or "",
            "train_layout": train_layout or "",
            "train_seed": train_seed if train_seed is not None else -1,
            "step": step,
            "eval_layout": cli.eval_layout,
            "num_episodes": cli.num_episodes,
            "num_steps": cli.num_steps,
            "seed": cli.seed,
            "deterministic": int(deterministic),
            "checkpoint": ckpt_path,
        },
    )

    out_dir = os.path.dirname(os.path.abspath(cli.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    tmp_path = cli.out + ".tmp"
    ds.to_netcdf(tmp_path)
    os.replace(tmp_path, cli.out)

    gain = float(ds["power_gain_pct"])
    print(f"[Done] step {step}: gain={gain:+.2f}%  -> {cli.out}")


if __name__ == "__main__":
    main()
