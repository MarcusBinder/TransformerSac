import os
import re
import numpy as np
import xarray as xr
import random
import torch

# ---- Imports from your codebase ----
from transformer_sac_windfarm_v24 import *
from agent import WindFarmAgent
from MultiLayoutEnv import MultiLayoutEnv, LayoutConfig

# Step 2, get the args from the checkpoint file
def load_actor_from_checkpoint(checkpoint_path: str, device: torch.device):
    """Load actor network from checkpoint (handles old and new formats)."""
    data = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    checkpoint = data
    args = data["args"]
    if hasattr(args, '__dict__'):
        args = vars(args)
    
    return checkpoint, args

# Step 3, create the environment used for evaluation:

def create_eval_env(layout: str, args: dict, seed: int = 42, n_envs: int = 1):
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
    # print("Creating eval env for layouts:", layout_names)
    layouts = []
    for name in layout_names:
        x_pos, y_pos = get_layout_positions(name, wind_turbine)
        layout_to_use = LayoutConfig(name=name, x_pos=x_pos, y_pos=y_pos)
        
        if args["profile_encoding_type"] is not None:
            if args["profile_source"].lower() == "geometric":
                from geometric_profiles import compute_layout_profiles_vectorized
                
                # Get rotor diameter as a float (geometric version doesn't need the full WT object)
                D = wind_turbine.diameter()  # or however DTU10MW exposes this
                
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
            
            layout_to_use.receptivity_profiles = receptivity_profiles  # (n_turbines, n_directions
            layout_to_use.influence_profiles = influence_profiles      # (n_turbines, n_directions
            
        layouts.append(layout_to_use)

    if args["profile_encoding_type"] is not None:
        use_profiles = True
    else:
        use_profiles = False

    # Build profile registry from layouts
    if use_profiles:
        profile_registry = [
            (layout.receptivity_profiles, layout.influence_profiles)
            for layout in layouts
        ]
    else:
        profile_registry = None


    # Environment config
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
        "TurbBox": "/work/users/manils/rl_timestep/Boxes/V80env/",
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
        """Factory function for vectorized environments."""
        def _init():
            env = MultiLayoutEnv(
                layouts=layouts,
                env_factory=env_factory,
                per_turbine_wrapper=combined_wrapper,
                seed=seed,
                shuffle=False,  # No shuffling during eval
                max_episode_steps=9999999, # Set to large during evaluation
            )
            return env
        return _init
        
    env = gym.vector.AsyncVectorEnv(
        [make_env_fn(seed=seed+i) for i in range(n_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    env = RecordEpisodeVals(env)


    return env, wind_turbine


# ---- Evaluation function (per checkpoint) ----
def evaluate_checkpoint_episodes(env, agent, num_batches, num_envs, num_steps, deterministic):
    all_episodes = []

    for batch in range(num_batches):
        obs, infos = env.reset()
        wind_speeds = infos["Wind speed Global"]
        wind_directions = infos["Wind direction Global"]

        rewards_per_env = [[] for _ in range(num_envs)]
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

        # dones = np.zeros(num_envs, dtype=bool)

        for step in range(num_steps):
            actions = agent.act(env, obs, deterministic=deterministic)
            obs, rewards, terminateds, truncateds, infos = env.step(actions)

            for i in range(num_envs):
                # if not dones[i]: # We never run the episode until done. We stop before.
                rewards_per_env[i].append(rewards[i])
                if "Power agent" in infos:
                    powers_per_env[i].append(infos["Power agent"][i])
                if "Power baseline" in infos:
                    baseline_powers_per_env[i].append(infos["Power baseline"][i])
                if "yaw angles agent" in infos:
                    yaws_per_env[i].append(infos["yaw angles agent"][i])

            # dones |= terminateds | truncateds
            # if dones.all():
            #     break

        for i in range(num_envs):
            episode_data = {
                "wind_speed": wind_speeds[i].item(),
                "wind_direction": wind_directions[i].item(),
                "rewards": rewards_per_env[i],
                "powers": powers_per_env[i],
                "baseline_powers": baseline_powers_per_env[i],
                "yaw_angles": yaws_per_env[i],
                "episode_return": sum(rewards_per_env[i]),
                "episode_length": len(rewards_per_env[i]),
                "mean_power": np.mean(powers_per_env[i]) if powers_per_env[i] else None,
                "mean_baseline_power": np.mean(baseline_powers_per_env[i]) if baseline_powers_per_env[i] else None,
            }
            all_episodes.append(episode_data)

    return all_episodes


# ---- Parse run name ----
def parse_run_name(run_name):
    """
    Parse 'multifarm_XXX_separate_sY' -> (layout_str, seed).
    Examples:
        multifarm_E1_separate_s1                 -> ('E1', 1)
        multifarm_T1-T2-T3-T4-T5-T6_separate_s2 -> ('T1-T2-T3-T4-T5-T6', 2)
    """
    match = re.match(r"multifarm_(.+)_separate_s(\d+)", run_name)
    if not match:
        return None, None
    return match.group(1), int(match.group(2))


# ---- Evaluate a single run on a single layout, save to .nc ----
def evaluate_run_on_layout(run_name, eval_layout, n_envs):
    checkpoint_dir = os.path.join(BASE_DIR, run_name, "checkpoints")
    output_file = os.path.join(OUTPUT_DIR, f"{run_name}_eval_{eval_layout}_results.nc")

    if os.path.exists(output_file):
        print(f"  [SKIP] Output already exists: {output_file}")
        return
    print(f"Evaluating run '{run_name}' on layout '{eval_layout}' with {n_envs} envs...")

    files = sorted(os.listdir(checkpoint_dir))
    if not files:
        print(f"  [SKIP] No checkpoints found in {checkpoint_dir}")
        return

    print(f"  Evaluating on layout={eval_layout}, saving to {output_file}")

    # Load args from first checkpoint
    first_path = os.path.join(checkpoint_dir, files[0])
    _, args = load_actor_from_checkpoint(first_path, device)

    # Create env for this eval layout
    env, wind_turbine = create_eval_env(layout=eval_layout, args=args, seed=INPUT_SEED, n_envs=n_envs)
    
    rotor_diameter = env.env.get_attr('rotor_diameter')[0]

    obs_dim_per_turbine = env.single_observation_space.shape[-1]
    action_high = env.single_action_space.high[0]
    action_low = env.single_action_space.low[0]
    action_scale = (action_high - action_low) / 2.0
    action_bias = (action_high + action_low) / 2.0

    # Profile encoders
    if args["profile_encoding_type"] is not None:
        use_profiles = True
        if args["share_profile_encoder"]:
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
    else:
        use_profiles = False
        shared_recep_encoder = None
        shared_influence_encoder = None

    # Build actor
    actor = TransformerActor(
        obs_dim_per_turbine=obs_dim_per_turbine,
        action_dim_per_turbine=1,
        embed_dim=args["embed_dim"],
        pos_embed_dim=args["pos_embed_dim"],
        num_heads=args["num_heads"],
        num_layers=args["num_layers"],
        mlp_ratio=args["mlp_ratio"],
        dropout=0.0,
        action_scale=action_scale,
        action_bias=action_bias,
        pos_encoding_type=args["pos_encoding_type"],
        rel_pos_hidden_dim=args["rel_pos_hidden_dim"],
        rel_pos_per_head=args["rel_pos_per_head"],
        pos_embedding_mode=args["pos_embedding_mode"],
        profile_encoding=args["profile_encoding_type"],
        profile_encoder_hidden=args["profile_encoder_hidden"],
        n_profile_directions=args["n_profile_directions"],
        profile_fusion_type=args["profile_fusion_type"],
        shared_recep_encoder=shared_recep_encoder,
        shared_influence_encoder=shared_influence_encoder,
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
    assert NUM_EPISODES % num_envs == 0
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
            env, agent, num_batches, num_envs, NUM_STEPS, DETERMINISTIC
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
    for ci, s in enumerate(steps):
        for ei, ep in enumerate(all_results[s]["episodes"]):
            yaws = ep["yaw_angles"]
            for ti, yaw_snapshot in enumerate(yaws):
                yaw_array[ci, ei, ti, :] = yaw_snapshot

    ds = xr.Dataset(
        {
            "mean_return": ("step", [all_results[s]["mean_return"] for s in steps]),
            "std_return": ("step", [all_results[s]["std_return"] for s in steps]),
            "mean_power": ("step", [all_results[s]["mean_power"] for s in steps]),
            "mean_baseline_power": ("step", [all_results[s]["mean_baseline_power"] for s in steps]),
            "power_gain_pct": ("step", [all_results[s]["power_gain_pct"] for s in steps]),
            "mean_length": ("step", [all_results[s]["mean_length"] for s in steps]),
            "yaw_angles": (["step", "episode", "time", "turbine"], yaw_array),
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
            "train_layout": parse_run_name(run_name)[0],
            "run_name": run_name,
            "num_episodes": NUM_EPISODES,
            "num_steps": NUM_STEPS,
            # "deterministic": DETERMINISTIC,
            "seed": INPUT_SEED,
        },
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ds.to_netcdf(output_file)
    print(f"  [DONE] Saved {output_file}")


# ---- Main ----
if __name__ == "__main__":

    
    
    # ---- Configuration ----
    BASE_DIR = "/work/users/manils/LargeFarm/runs"           # <-- SET THIS
    OUTPUT_DIR = "/work/users/manils/LargeFarm/evals"  # <-- SET THIS

    INPUT_SEED = 42
    N_ENVS = 30
    NUM_EPISODES = 60
    NUM_STEPS = 250
    DETERMINISTIC = True
    VERBOSE = True

    ALL_EVAL_LAYOUTS = [
        "T1", "T2", "T3", "T4", "T5", "T6",
        "E1", "E2", "E3", "E4", "E5",
    ]

    MULTI_LAYOUT_NAME = "T1-T2-T3-T4-T5-T6"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    all_runs = os.listdir(BASE_DIR)

    # Shuffle to reduce collisions if running multiple instances in parallel
    random.shuffle(all_runs)

    for run_name in all_runs:
        layout_str, seed = parse_run_name(run_name)

        if layout_str is None:
            print(f"[SKIP] Cannot parse run name: {run_name}")
            continue

        print(f"\n{'='*60}")
        print(f"Run: {run_name}  (train_layout={layout_str}, seed={seed})")
        print(f"{'='*60}")


        if layout_str == MULTI_LAYOUT_NAME:
            # Multi-layout run: evaluate on ALL 11 layouts
            eval_layouts_shuffled = ALL_EVAL_LAYOUTS.copy()
            random.shuffle(eval_layouts_shuffled)
            for eval_layout in eval_layouts_shuffled:
                print(f"\nEvaluating on layout: {eval_layout}")
                evaluate_run_on_layout(run_name, eval_layout, n_envs=N_ENVS)

        else:
            # Single-layout run: evaluate on its own layout only
            evaluate_run_on_layout(run_name, eval_layout=layout_str, n_envs=N_ENVS)