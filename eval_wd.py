"""Offline checkpoint eval under a wind-direction SCHEDULE, with optional
corruption or estimator sourcing of the wd fed to the agent.

The WD-estimation ladder's offline workhorse (T0 sensitivity injection, T2
estimator-in-the-loop transfer, parity gates). One invocation = one cell:
one checkpoint x one wd schedule x one ws x one wd source/corruption ->
one NetCDF.

Modeled on eval_checkpoint.py's skeleton but with three deliberate
differences (eval_checkpoint.py is left untouched — it serves the DEL
campaign and hard-codes make_env_config("hard")):

  * The env config comes from the CHECKPOINT's --config preset (e.g.
    hard_2), then make_eval_wind_config pins wd to wd_function(0) and ws to
    --ws, exactly like the trainer's in-training eval specs — so offline
    numbers are comparable to the run's W&B eval/... series.
  * The env gets the named --wd_function (WD_FUNCTIONS registry), so wd can
    ramp mid-episode on either backend (pywake needs windgym >= proj/wdest).
  * The wd fed to the agent's rotation machinery (wind-relative transform +
    profile rotation) is fetched EXPLICITLY each step, optionally corrupted
    (--wd_corrupt, T0) or sourced from the env's sensor-derived estimate
    (--wd_source est, T2), and passed via the existing
    agent.act(..., wind_dirs=) override. This touches ONLY the agent's
    rotation inputs — env physics, observations and yaw re-booking always
    run on the true wd. Both the true and the fed wd are logged per step.

power_ratio accounting mirrors helpers/eval_utils.PolicyEvaluator verbatim
(per-episode step-loop means; batch seeds strided by num_envs) so the
parity gate can compare against the run's W&B final-eval power_ratio.

Usage (T0 cell):
    python eval_wd.py \
        --checkpoint checkpoints_w4/w4_wsnarrow_s1/step_200000.pt \
        --wd_function step_ramp_270_315 --ws 12 \
        --wd_corrupt bias:5 --num-steps 140 --num-envs 20 \
        --out evals/t0/ramp_ws12_bias5_s1.nc
"""

import argparse
import os
import re
import sys
import time

import numpy as np
import torch
import xarray as xr

import gymnasium as gym
from config import Args
from networks import TransformerActor, create_profile_encoding
from helpers.agent import WindFarmAgent
from helpers.env_configs import make_env_config, make_eval_wind_config
from helpers.helper_funcs import EnhancedPerTurbineWrapper
from helpers.layouts import get_layout_positions
from helpers.multi_layout_env import MultiLayoutEnv, LayoutConfig
from helpers.receptivity_profiles import compute_layout_profiles
from helpers.wd_corruption import make_corruptor
from helpers.wd_functions import WD_FUNCTIONS
from WindGym import WindFarmEnv
from WindGym.wrappers import RecordEpisodeVals, PerTurbineObservationWrapper


def load_checkpoint(checkpoint_path: str, device: torch.device):
    data = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = data["args"]
    args = vars(args) if not isinstance(args, dict) else args
    # Older checkpoints predate some config flags; backfill from current Args()
    # defaults so dict access never KeyErrors. Checkpoint values win.
    defaults = vars(Args())
    missing = [k for k in defaults if k not in args]
    args = {**defaults, **args}
    if missing:
        print(f"[eval_wd] Backfilled {len(missing)} missing args from "
              f"defaults: {sorted(missing)}")
    return data, args


def create_eval_env(layout: str, args: dict, cli, wd_fn, seed: int,
                    n_envs: int):
    """One AsyncVectorEnv of MultiLayoutEnv slots pinned to one wd/ws cell."""
    if args["turbtype"] == "DTU10MW":
        from py_wake.examples.data.dtu10mw import DTU10MW as WT
    elif args["turbtype"] == "V80":
        from py_wake.examples.data.hornsrev1 import V80 as WT
    else:
        raise ValueError(f"Unknown turbine type: {args['turbtype']}")
    wind_turbine = WT()

    layouts = []
    for name in [s.strip() for s in layout.split(",") if s.strip()]:
        x_pos, y_pos = get_layout_positions(name, wind_turbine)
        layout_cfg = LayoutConfig(name=name, x_pos=x_pos, y_pos=y_pos)
        if args["profile_encoding_type"] is not None:
            if args["profile_source"].lower() == "geometric":
                from helpers.geometric_profiles import (
                    compute_layout_profiles_vectorized,
                )
                recep, infl = compute_layout_profiles_vectorized(
                    x_pos, y_pos,
                    rotor_diameter=wind_turbine.diameter(),
                    k_wake=0.04,
                    n_directions=args["n_profile_directions"],
                    sigma_smooth=args.get("profile_sigma_smooth", 10.0),
                    scale_factor=15.0,
                    mode=args.get("profile_geom_mode", "wake"),
                )
            elif args["profile_source"].lower() == "pywake":
                recep, infl = compute_layout_profiles(
                    x_pos, y_pos, wind_turbine,
                    n_directions=args["n_profile_directions"],
                )
            else:
                raise ValueError(
                    f"Unknown profile_source: {args['profile_source']}")
            layout_cfg.receptivity_profiles = recep
            layout_cfg.influence_profiles = infl
        layouts.append(layout_cfg)

    # Env config: the checkpoint's preset, mes histories, then the eval wind
    # pin — the same construction the trainer uses for its eval specs.
    config = make_env_config(args["config"])
    for mes_type, prefix in (("ws_mes", "ws"), ("wd_mes", "wd"),
                             ("yaw_mes", "yaw"), ("power_mes", "power"),
                             ("derate_mes", "derate")):
        if mes_type not in config:
            continue
        config[mes_type][f"{prefix}_history_N"] = args["history_length"]
        config[mes_type][f"{prefix}_history_length"] = args["history_length"]
    config = make_eval_wind_config(config, float(wd_fn(0.0)), float(cli.ws))

    backend = cli.backend or args["backend"]
    base_env_kwargs = {
        "turbine": wind_turbine,
        "n_passthrough": args["max_eps"],
        "TurbBox": cli.turbbox_path,
        "config": config,
        "turbtype": cli.TI_type or args["TI_type"],
        "backend": backend,
        "dt_sim": cli.dt_sim if cli.dt_sim is not None else args["dt_sim"],
        "dt_env": cli.dt_env if cli.dt_env is not None else args["dt_env"],
        "yaw_step_sim": args["yaw_step"],
        "wd_function": wd_fn,
    }
    # Obs-affine ws range override (change_wd_4 runs; OBS_SCALING.md). Must
    # match training or the ws obs columns are scaled differently than what
    # the actor saw.
    for k in ("ws_scaling_min", "ws_scaling_max"):
        if args.get(k) is not None:
            base_env_kwargs[k] = float(args[k])
    if cli.wd_source == "est":
        if backend == "pywake":
            raise SystemExit(
                "--wd_source est needs the dynamiks backend: pywake's "
                "adapter hard-codes v=w=0, so a measured local wd does not "
                "exist there.")
        base_env_kwargs["wd_est_tau"] = cli.wd_est_tau
        base_env_kwargs["wd_est_consensus"] = cli.wd_est_consensus

    # No yaw_init override: the trainer's env factories let the config preset
    # decide (hard_2 inherits "Random"), and parity with the in-training eval
    # requires the same seeded initial yaws.
    def env_factory(x_pos, y_pos):
        env = WindFarmEnv(x_pos=x_pos, y_pos=y_pos,
                          reset_init=False, **base_env_kwargs)
        env.action_space.seed(seed)
        return env

    def combined_wrapper(env):
        env = PerTurbineObservationWrapper(env)
        if args["use_wd_deviation"]:
            env = EnhancedPerTurbineWrapper(
                env, wd_scale_range=args["wd_scale_range"])
        if args.get("obs_encoding"):
            import json
            from helpers.obs_encoding import ObsEncodingWrapper
            env = ObsEncodingWrapper(
                env, mode=args["obs_encoding"], turbine=wind_turbine,
                **json.loads(args.get("obs_encoding_kwargs") or "{}"))
        return env

    def make_env_fn(env_seed):
        def _init():
            return MultiLayoutEnv(
                layouts=layouts,
                env_factory=env_factory,
                per_turbine_wrapper=combined_wrapper,
                seed=env_seed,
                shuffle=False,
                max_turbines=args["max_turbines"],
                max_episode_steps=9999999,
            )
        return _init

    env = gym.vector.AsyncVectorEnv(
        [make_env_fn(seed + i) for i in range(n_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    return RecordEpisodeVals(env), wind_turbine


def build_agent(args: dict, env, device: torch.device, checkpoint):
    """Actor + agent from checkpoint args (eval_checkpoint.py's recipe, plus
    the --obs_norm statistics when the run trained with them)."""
    rotor_diameter = env.env.get_attr("rotor_diameter")[0]
    obs_dim_per_turbine = env.single_observation_space.shape[-1]
    action_high = env.single_action_space.high[0]
    action_low = env.single_action_space.low[0]

    use_profiles = args["profile_encoding_type"] is not None
    if use_profiles and args["share_profile_encoder"]:
        shared_recep, shared_infl = create_profile_encoding(
            profile_type=args["profile_encoding_type"],
            embed_dim=args["embed_dim"],
            hidden_channels=args["profile_encoder_hidden"],
        )
        shared_recep = shared_recep.to(device)
        shared_infl = shared_infl.to(device)
    else:
        shared_recep = shared_infl = None

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
        action_scale=(action_high - action_low) / 2.0,
        action_bias=(action_high + action_low) / 2.0,
        pos_encoding_type=args_ns.pos_encoding_type,
        rel_pos_hidden_dim=args_ns.rel_pos_hidden_dim,
        rel_pos_per_head=args_ns.rel_pos_per_head,
        pos_embedding_mode=args_ns.pos_embedding_mode,
        profile_encoding=args_ns.profile_encoding_type,
        profile_encoder_hidden=args_ns.profile_encoder_hidden,
        n_profile_directions=args_ns.n_profile_directions,
        profile_fusion_type=args_ns.profile_fusion_type,
        profile_embed_mode=args_ns.profile_embed_mode,
        shared_recep_encoder=shared_recep,
        shared_influence_encoder=shared_infl,
        args=args_ns,
    ).to(device)

    obs_normalizer = None
    if args.get("obs_norm"):
        from helpers.obs_norm import ObsRunningNorm
        obs_normalizer = ObsRunningNorm(obs_dim_per_turbine, device)
        if "obs_norm_state" in checkpoint and checkpoint["obs_norm_state"]:
            obs_normalizer.load_state_dict(checkpoint["obs_norm_state"])
            print("[eval_wd] Loaded obs_norm statistics from checkpoint")
        else:
            raise SystemExit(
                "Checkpoint trained with --obs_norm but has no "
                "obs_norm_state; refusing to eval with cold statistics.")

    agent = WindFarmAgent(
        actor=actor,
        device=device,
        rotor_diameter=rotor_diameter,
        use_wind_relative=args["use_wind_relative_pos"],
        use_profiles=use_profiles,
        rotate_profiles=args["rotate_profiles"],
        obs_normalizer=obs_normalizer,
    )
    return actor, agent


def run_episodes(env, agent, corruptor, cli, wd_source_attr: str):
    """PolicyEvaluator-style episode batches, with the explicit wd feed.

    Returns a list of per-episode dicts. The fed wd = corrupt(source wd);
    the true env wd is logged alongside regardless of source.
    """
    num_envs, num_steps = cli.num_envs, cli.num_steps
    episodes = []
    for batch in range(cli.num_episodes // num_envs):
        t0 = time.time()
        # Seed stride matches PolicyEvaluator: all-unique seed blocks.
        obs, infos = env.reset(seed=cli.seed + batch * num_envs)

        wd_true0 = np.array(env.env.get_attr("wd"), dtype=np.float64)
        wd_src0 = (wd_true0 if wd_source_attr == "wd" else
                   np.array(env.env.get_attr(wd_source_attr),
                            dtype=np.float64))
        corruptor.reset(wd_src0)

        ws0 = np.asarray(infos["Wind speed Global"], dtype=float)
        wd0 = np.asarray(infos["Wind direction Global"], dtype=float)

        rewards_pe = np.zeros(num_envs)
        powers_pe = [[] for _ in range(num_envs)]
        base_pe = [[] for _ in range(num_envs)]
        yaws_pe = [[] for _ in range(num_envs)]
        wd_true_pe = [[] for _ in range(num_envs)]
        wd_fed_pe = [[] for _ in range(num_envs)]

        for _ in range(num_steps):
            wd_true = np.array(env.env.get_attr("wd"), dtype=np.float64)
            wd_src = (wd_true if wd_source_attr == "wd" else
                      np.array(env.env.get_attr(wd_source_attr),
                               dtype=np.float64))
            wd_fed = corruptor.corrupt(wd_src)

            actions = agent.act(env, obs, deterministic=not cli.stochastic,
                                wind_dirs=wd_fed.astype(np.float32))
            obs, rewards, terms, truncs, infos = env.step(actions)

            rewards_pe += np.asarray(rewards, dtype=float)
            for i in range(num_envs):
                wd_true_pe[i].append(wd_true[i])
                wd_fed_pe[i].append(wd_fed[i])
                if "Power agent" in infos:
                    powers_pe[i].append(float(infos["Power agent"][i]))
                if "Power baseline" in infos:
                    base_pe[i].append(float(infos["Power baseline"][i]))
                if "yaw angles agent" in infos:
                    yaws_pe[i].append(np.asarray(
                        infos["yaw angles agent"][i], dtype=float))

        for i in range(num_envs):
            episodes.append({
                "wind_speed": float(ws0[i]),
                "wind_direction": float(wd0[i]),
                "episode_return": float(rewards_pe[i]),
                "powers": powers_pe[i],
                "baseline_powers": base_pe[i],
                "yaw_angles": yaws_pe[i],
                "wd_true": wd_true_pe[i],
                "wd_fed": wd_fed_pe[i],
            })
        print(f"[eval_wd]   batch {batch + 1}/"
              f"{cli.num_episodes // num_envs} ({time.time() - t0:.1f}s)")
    return episodes


def build_dataset(episodes, cli, meta):
    n_ep, n_t = len(episodes), cli.num_steps
    n_turb = len(episodes[0]["yaw_angles"][0]) if episodes[0]["yaw_angles"] \
        else 0

    def stack(key, fill=np.nan):
        arr = np.full((n_ep, n_t), fill)
        for ei, ep in enumerate(episodes):
            vals = ep[key]
            arr[ei, :len(vals)] = vals
        return arr

    power = stack("powers")
    baseline = stack("baseline_powers")
    yaw = np.full((n_ep, n_t, n_turb), np.nan)
    for ei, ep in enumerate(episodes):
        for ti, snap in enumerate(ep["yaw_angles"]):
            yaw[ei, ti, :] = snap

    # PolicyEvaluator's power_ratio: mean over episodes of per-episode step
    # means, agent / baseline.
    ep_power = np.nanmean(power, axis=1)
    ep_base = np.nanmean(baseline, axis=1)
    mean_power = float(np.mean(ep_power))
    mean_base = float(np.mean(ep_base))
    power_ratio = mean_power / mean_base if mean_base > 0 else np.nan

    ds = xr.Dataset(
        {
            "power_ratio": ((), power_ratio),
            "mean_power": ((), mean_power),
            "mean_baseline_power": ((), mean_base),
            "mean_return": ((), float(np.mean(
                [ep["episode_return"] for ep in episodes]))),
            "power": (["episode", "time"], power),
            "baseline_power": (["episode", "time"], baseline),
            "yaw_angles": (["episode", "time", "turbine"], yaw),
            "wd_true": (["episode", "time"], stack("wd_true")),
            "wd_fed": (["episode", "time"], stack("wd_fed")),
            "episode_power_ratio": (["episode"],
                                    ep_power / np.where(ep_base > 0, ep_base,
                                                        np.nan)),
            "wind_speed": (["episode"],
                           [ep["wind_speed"] for ep in episodes]),
            "wind_direction": (["episode"],
                               [ep["wind_direction"] for ep in episodes]),
        },
        coords={
            "episode": np.arange(n_ep),
            "time": np.arange(n_t),
            "turbine": np.arange(n_turb),
        },
        attrs=meta,
    )
    return ds


def main():
    parser = argparse.ArgumentParser(
        description="Eval ONE checkpoint under ONE wd schedule/source cell, "
                    "write ONE NetCDF.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--wd_function", required=True,
                        choices=sorted(WD_FUNCTIONS))
    parser.add_argument("--ws", type=float, default=12.0)
    parser.add_argument("--wd_source", choices=("true", "est"),
                        default="true",
                        help="What the agent's rotation machinery is fed: "
                             "the env's privileged wd, or the sensor-derived "
                             "estimate (dynamiks only, needs wd_est_tau).")
    parser.add_argument("--wd_corrupt", default="none",
                        help="Corruption spec applied to the source wd "
                             "(see helpers/wd_corruption.py).")
    parser.add_argument("--wd_est_tau", type=float, default=None,
                        help="Estimator EWMA time constant (s); forwarded "
                             "to the env when --wd_source est.")
    parser.add_argument("--wd_est_consensus", default="median",
                        help="Estimator consensus (median/mean/front); "
                             "forwarded when --wd_source est.")
    parser.add_argument("--eval-layout", default="square_5x5")
    parser.add_argument("--backend", default=None,
                        choices=(None, "pywake", "dynamiks"),
                        help="Override the checkpoint's backend.")
    parser.add_argument("--TI_type", default=None,
                        help="Override the checkpoint's TI/turbulence type "
                             "(e.g. MannGenerate).")
    parser.add_argument("--dt_sim", type=int, default=None)
    parser.add_argument("--dt_env", type=int, default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-episodes", type=int, default=None,
                        help="Default: one batch (= --num-envs).")
    parser.add_argument("--num-steps", type=int, default=140)
    parser.add_argument("--num-envs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--turbbox-path", default="./boxes/")
    parser.add_argument("--stochastic", action="store_true")
    cli = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    if cli.num_episodes is None:
        cli.num_episodes = cli.num_envs
    assert cli.num_episodes % cli.num_envs == 0, (
        f"num_episodes={cli.num_episodes} must be divisible by "
        f"num_envs={cli.num_envs}")
    if cli.wd_source == "est" and cli.wd_est_tau is None:
        raise SystemExit("--wd_source est requires --wd_est_tau")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = os.path.abspath(cli.checkpoint)
    run_name = os.path.basename(os.path.dirname(os.path.dirname(ckpt_path)))
    step_match = re.search(r"step_(\d+)", os.path.basename(ckpt_path))
    step = int(step_match.group(1)) if step_match else -1

    checkpoint, args = load_checkpoint(ckpt_path, device)
    wd_fn = WD_FUNCTIONS[cli.wd_function]

    print(f"[eval_wd] run={run_name} step={step} "
          f"schedule={cli.wd_function} ws={cli.ws} source={cli.wd_source} "
          f"corrupt={cli.wd_corrupt} backend={cli.backend or args['backend']} "
          f"episodes={cli.num_episodes} steps={cli.num_steps} "
          f"envs={cli.num_envs} seed={cli.seed} device={device}")

    env, _ = create_eval_env(cli.eval_layout, args, cli, wd_fn,
                             seed=cli.seed, n_envs=cli.num_envs)
    actor, agent = build_agent(args, env, device, checkpoint)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()

    dt_env = cli.dt_env if cli.dt_env is not None else args["dt_env"]
    corruptor = make_corruptor(cli.wd_corrupt, cli.num_envs,
                               dt_env=float(dt_env), seed=cli.seed)
    wd_source_attr = "wd" if cli.wd_source == "true" else "wd_est"

    t0 = time.time()
    episodes = run_episodes(env, agent, corruptor, cli, wd_source_attr)
    env.close()
    print(f"[eval_wd] {len(episodes)} episodes in {time.time() - t0:.1f}s")

    ds = build_dataset(episodes, cli, meta={
        "run_name": run_name,
        "step": step,
        "checkpoint": ckpt_path,
        "wd_function": cli.wd_function,
        "ws": cli.ws,
        "wd_source": cli.wd_source,
        "wd_corrupt": cli.wd_corrupt,
        "wd_est_tau": cli.wd_est_tau if cli.wd_est_tau is not None else -1.0,
        "wd_est_consensus": cli.wd_est_consensus,
        "eval_layout": cli.eval_layout,
        "backend": cli.backend or args["backend"],
        "TI_type": cli.TI_type or args["TI_type"],
        "dt_sim": cli.dt_sim if cli.dt_sim is not None else args["dt_sim"],
        "dt_env": dt_env,
        "num_episodes": cli.num_episodes,
        "num_steps": cli.num_steps,
        "seed": cli.seed,
        "deterministic": int(not cli.stochastic),
    })

    out_dir = os.path.dirname(os.path.abspath(cli.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    tmp = cli.out + ".tmp"
    ds.to_netcdf(tmp)
    os.replace(tmp, cli.out)
    print(f"[Done] power_ratio={float(ds['power_ratio']):.4f} -> {cli.out}")


if __name__ == "__main__":
    main()
