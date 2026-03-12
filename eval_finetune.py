import os
import re
import sys
import numpy as np
import xarray as xr
import random
import torch

# ---- Imports from your codebase ----
from transformer_sac_windfarm_v26 import *
from agent import WindFarmAgent
from MultiLayoutEnv import MultiLayoutEnv, LayoutConfig


def load_actor_from_checkpoint(checkpoint_path: str, device: torch.device):
    """Load actor network from checkpoint (handles old and new formats)."""
    data = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint = data
    args = data["args"]
    # Keep as namespace for attribute access; convert to dict only when needed via vars()
    return checkpoint, args


def create_eval_env(layout: str, args: dict, seed: int = 42, n_envs: int = 1):
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
                from geometric_profiles import compute_layout_profiles_vectorized

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

    if args["profile_encoding_type"] is not None:
        use_profiles = True
    else:
        use_profiles = False

    if use_profiles:
        profile_registry = [
            (layout.receptivity_profiles, layout.influence_profiles)
            for layout in layouts
        ]
    else:
        profile_registry = None

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
        "TurbBox": TURBBOX_PATH,
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
        [make_env_fn(seed=seed+i) for i in range(n_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    env = RecordEpisodeVals(env)

    return env, wind_turbine


def evaluate_checkpoint_episodes(env, agent, num_batches, num_envs, num_steps, deterministic, seed=None):
    all_episodes = []

    for batch in range(num_batches):
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

    return all_episodes


def parse_run_name(run_name):
    """
    Parse run names like:
        'finetune_E1_ls100_s1'   -> ('finetune_ls100', 'E1', 1)
        'finetune_E3_ls5000_s2'  -> ('finetune_ls5000', 'E3', 2)
    Returns (config_name, layout_str, seed) or (None, None, None).
    """
    match = re.match(r"finetune_(E\d+)_(ls\d+)_s(\d+)", run_name)
    if not match:
        return None, None, None
    layout_str = match.group(1)
    config_name = f"finetune_{match.group(2)}"
    seed = int(match.group(3))
    return config_name, layout_str, seed


def find_safe_output_path(base_path):
    """
    If base_path doesn't exist, return it.
    Otherwise append _try2, _try3, ... before the extension.
    """
    if not os.path.exists(base_path):
        return base_path

    root, ext = os.path.splitext(base_path)
    attempt = 2
    while True:
        candidate = f"{root}_try{attempt}{ext}"
        if not os.path.exists(candidate):
            return candidate
        attempt += 1


def evaluate_run_on_layout(run_name, eval_layout, n_envs):
    checkpoint_dir = os.path.join(BASE_DIR, run_name, "checkpoints")
    output_file = os.path.join(OUTPUT_DIR, f"{run_name}_eval_{eval_layout}_results.nc")

    # Skip if already done
    if os.path.exists(output_file):
        print(f"  [SKIP] Output already exists: {output_file}")
        return

    print(f"Evaluating run '{run_name}' on layout '{eval_layout}' with {n_envs} envs...")

    files = sorted(os.listdir(checkpoint_dir))
    if not files:
        print(f"  [SKIP] No checkpoints found in {checkpoint_dir}")
        return

    # Load args from first checkpoint
    first_path = os.path.join(checkpoint_dir, files[0])

    # Convert args to dict for dict-style access where needed,
    # but keep the original namespace for passing to TransformerActor
    _, args_ns = load_actor_from_checkpoint(first_path, device)
    args = vars(args_ns) if hasattr(args_ns, '__dict__') and not isinstance(args_ns, dict) else args_ns

    # Create env for eval layout
    env, wind_turbine = create_eval_env(layout=eval_layout, args=args, seed=INPUT_SEED, n_envs=n_envs)

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

    # Convert back to namespace for the constructor
    from argparse import Namespace
    args_ns = Namespace(**args) if isinstance(args, dict) else args

    # Build actor
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

    num_envs = env.num_envs
    assert NUM_EPISODES % num_envs == 0, (
        f"NUM_EPISODES={NUM_EPISODES} must be divisible by num_envs={num_envs}"
    )
    num_batches = NUM_EPISODES // num_envs

    # ---- Loop over checkpoints ----
    all_results = {}

    for file in sorted(files):
        step_number = int(file.split("_")[1].split(".")[0])
        file_path = os.path.join(checkpoint_dir, file)

        if VERBOSE:
            print(f"    Checkpoint step {step_number} ({file})")

        checkpoint, _ = load_actor_from_checkpoint(file_path, device)
        actor.load_state_dict(checkpoint["actor_state_dict"])
        actor.eval()

        episodes = evaluate_checkpoint_episodes(
            env, agent, num_batches, num_envs, NUM_STEPS, DETERMINISTIC, seed=INPUT_SEED
        )

        episode_returns = [ep["episode_return"] for ep in episodes]
        episode_powers = [ep["mean_power"] for ep in episodes if ep["mean_power"] is not None]
        baseline_powers = [ep["mean_baseline_power"] for ep in episodes if ep["mean_baseline_power"] is not None]

        all_results[step_number] = {
            "mean_return": np.mean(episode_returns),
            "std_return": np.std(episode_returns),
            "mean_power": np.mean(episode_powers) if episode_powers else np.nan,
            "mean_baseline_power": np.mean(baseline_powers) if baseline_powers else np.nan,
            "power_gain_pct": (
                (np.mean(episode_powers) / np.mean(baseline_powers) - 1) * 100
                if episode_powers and baseline_powers else np.nan
            ),
            "mean_length": np.mean([ep["episode_length"] for ep in episodes]),
            "episodes": episodes,
        }

    env.close()

    # ---- Build xarray Dataset ----
    steps = sorted(all_results.keys())
    n_checkpoints = len(steps)
    max_time = NUM_STEPS + 1

    sample_yaws = all_results[steps[0]]["episodes"][0]["yaw_angles"]
    n_turbines = len(sample_yaws[0])

    yaw_array = np.full((n_checkpoints, NUM_EPISODES, max_time, n_turbines), np.nan)
    power_array = np.full((n_checkpoints, NUM_EPISODES, max_time), np.nan)
    baseline_power_array = np.full((n_checkpoints, NUM_EPISODES, max_time), np.nan)
    reward_array = np.full((n_checkpoints, NUM_EPISODES, max_time), np.nan)
    episode_return_array = np.full((n_checkpoints, NUM_EPISODES), np.nan)

    for ci, s in enumerate(steps):
        for ei, ep in enumerate(all_results[s]["episodes"]):
            yaws = ep["yaw_angles"]
            for ti, yaw_snapshot in enumerate(yaws):
                yaw_array[ci, ei, ti, :] = yaw_snapshot

            powers = ep["powers"]
            for ti, p in enumerate(powers):
                power_array[ci, ei, ti] = p

            bp = ep["baseline_powers"]
            for ti, b in enumerate(bp):
                baseline_power_array[ci, ei, ti] = b

            rews = ep["rewards"]
            for ti, r in enumerate(rews):
                reward_array[ci, ei, ti] = r

            episode_return_array[ci, ei] = ep["episode_return"]

    # Compute per-timestep power gain percentage
    with np.errstate(divide="ignore", invalid="ignore"):
        power_gain_pct_ts = (power_array / baseline_power_array - 1.0) * 100.0

    config_name, layout_str, train_seed = parse_run_name(run_name)

    ds = xr.Dataset(
        {
            "mean_return": ("step", [all_results[s]["mean_return"] for s in steps]),
            "std_return": ("step", [all_results[s]["std_return"] for s in steps]),
            "mean_power": ("step", [all_results[s]["mean_power"] for s in steps]),
            "mean_baseline_power": ("step", [all_results[s]["mean_baseline_power"] for s in steps]),
            "power_gain_pct": ("step", [all_results[s]["power_gain_pct"] for s in steps]),
            "mean_length": ("step", [all_results[s]["mean_length"] for s in steps]),
            "yaw_angles": (["step", "episode", "time", "turbine"], yaw_array),
            "power": (["step", "episode", "time"], power_array),
            "baseline_power": (["step", "episode", "time"], baseline_power_array),
            "power_gain_pct_ts": (["step", "episode", "time"], power_gain_pct_ts),
            "reward": (["step", "episode", "time"], reward_array),
            "episode_return": (["step", "episode"], episode_return_array),
            "wind_speed": (
                ["step", "episode"],
                [[ep["wind_speed"] for ep in all_results[s]["episodes"]] for s in steps],
            ),
            "wind_direction": (
                ["step", "episode"],
                [[ep["wind_direction"] for ep in all_results[s]["episodes"]] for s in steps],
            ),
        },
        coords={
            "step": steps,
            "episode": np.arange(NUM_EPISODES),
            "time": np.arange(max_time),
            "turbine": np.arange(n_turbines),
        },
        attrs={
            "eval_layout": eval_layout,
            "train_layout": layout_str or "",
            "run_name": run_name,
            "num_episodes": NUM_EPISODES,
            "num_steps": NUM_STEPS,
            "seed": INPUT_SEED,
        },
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Safe save: if another job already wrote this file, append _try<N>
    safe_path = find_safe_output_path(output_file)
    if safe_path != output_file:
        print(f"  [WARN] Original output appeared during eval, saving as: {safe_path}")
    ds.to_netcdf(safe_path)
    print(f"  [DONE] Saved {safe_path}")


# ---- Main ----
if __name__ == "__main__":

    # ---- Configuration ----
    BASE_DIR = "/users/nilsenma/runs"
    OUTPUT_DIR = "/users/nilsenma/evals"
    TURBBOX_PATH = "/users/nilsenma/Boxes/V80env/"

    INPUT_SEED = 42
    N_ENVS = 20          # 100 episodes / 20 envs = 5 batches
    NUM_EPISODES = 100
    NUM_STEPS = 250
    DETERMINISTIC = True
    VERBOSE = True

    ALL_EVAL_LAYOUTS = [
        "T1", "T2", "T3", "T4", "T5", "T6",
        "E1", "E2", "E3", "E4", "E5",
    ]

    MULTI_LAYOUT_NAME = "T1-T2-T3-T4-T5-T6"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Evaluate all runs in BASE_DIR
    all_runs = sorted(os.listdir(BASE_DIR))
    all_runs = [r for r in all_runs if r.startswith("finetune")]
    random.shuffle(all_runs)

    for run_name in all_runs:
        config_name, layout_str, seed = parse_run_name(run_name)

        # if layout_str is None:
        #     print(f"[SKIP] Cannot parse run name: {run_name}")
        #     continue
        layout_str="E5"
        print(f"\n{'='*60}")
        print(f"Run: {run_name}  (train_layout={layout_str}, seed={seed})")
        print(f"{'='*60}")

        # Single-layout run: evaluate on its own layout only
        evaluate_run_on_layout(run_name, eval_layout=layout_str, n_envs=N_ENVS)