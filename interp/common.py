"""Shared plumbing for the interp pipeline: checkpoint load, delmax env
rebuild, actor rebuild, batch-prep closure.

The env build mirrors transformer_sac_windfarm.py's ENV SETUP block for the
DEL-active path (the same stack every delmax/randlim checkpoint trained on):

    WindFarmEnv(Baseline_comp=True, BaseController="Global",
                turbine=make_derating_turbine(<checkpoint turbtype>))
      -> PerTurbineObservationWrapper [-> EnhancedPerTurbineWrapper]
      -> DELRewardWrapper(...)            # limit seams live here
      -> MultiLayoutEnv(shuffle=False)    # stable turbine order for interp

No TransformReward: we want the raw info labels (del_ratio, del_penalty,
reward_unpenalized, ...) untouched. The actor rebuild follows
eval_tracking_figs.build_agent -- NOT extract_attention's (which hardcodes
action_dim_per_turbine=1 and would silently break yaw+derate checkpoints).
"""

import hashlib
import json
import os
import subprocess
import sys
from argparse import Namespace
from functools import partial

import numpy as np
import torch

_INTERP_DIR = os.path.dirname(os.path.abspath(__file__))
TSAC_DIR = os.path.dirname(_INTERP_DIR)
REPO_ROOT = os.path.dirname(TSAC_DIR)
for _p in (REPO_ROOT, TSAC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import Args
from networks import TransformerActor, create_profile_encoding
from helpers.env_configs import make_env_config
from helpers.layouts import get_layout_positions
from helpers.multi_layout_env import MultiLayoutEnv, LayoutConfig
from helpers.derating_turbine import make_derating_turbine
from helpers.helper_funcs import EnhancedPerTurbineWrapper
from WindGym import WindFarmEnv
from WindGym.wrappers import PerTurbineObservationWrapper
from del_surrogate.reward_wrapper import DELRewardWrapper
from del_surrogate.model import DEFAULT_CHANNEL
# prepare_batch/compute_profiles are the attention-eval rollout skeleton,
# reused as-is (they only touch env attributes MultiLayoutEnv exposes).
from extract_attention import compute_profiles, prepare_batch

# Per-turbine action row convention (per_turbine_wrapper.py): yaw+derate
# checkpoints emit [yaw_i, derate_i] per token, so derate is column 1.
YAW_COL, DERATE_COL = 0, 1

DEL_CHANNEL = DEFAULT_CHANNEL


def load_checkpoint(checkpoint_path: str, device: torch.device):
    """Load the checkpoint dict + its args as a plain dict with any args added
    since the run was launched back-filled from config.Args defaults (copy of
    eval_tracking_figs.load_checkpoint, kept local so interp does not drag in
    that script's WindGym-eval import chain)."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args_obj = checkpoint["args"]
    args = vars(args_obj) if not isinstance(args_obj, dict) else dict(args_obj)

    defaults = vars(Args())
    missing = [k for k in defaults if k not in args]
    args = {**defaults, **args}  # checkpoint values win
    if missing:
        print(f"[interp] Back-filled {len(missing)} missing args from defaults: "
              f"{sorted(missing)}")
    # Backfill guard: config.Args now defaults turbtype to "IEA34", but every
    # checkpoint written before the switch trained on the DTU10MW surrogate.
    # A checkpoint whose args LACK turbtype must rebuild DTU, not the new
    # default. (Checkpoints that saved turbtype keep their saved value.)
    if "turbtype" in missing:
        args["turbtype"] = "DTU10MW"
        print('[interp] Checkpoint predates turbtype arg — pinned to "DTU10MW"')
    return checkpoint, args


def is_randlim(args: dict) -> bool:
    """Whether the checkpoint is goal-conditioned on the DEL limit (extra
    limit-obs column per turbine, U[lo,hi] limit per episode in training)."""
    return bool(args.get("del_limit_random", False))


def _default_turbbox() -> str:
    """The trainers pass ./boxes/ relative to TransformerSac/; fall back to
    windgym's bundled default boxes when that dir does not exist locally
    (TI_type=MannGenerate never reads it, TI_type=Random does)."""
    local = os.path.join(TSAC_DIR, "boxes")
    return local if os.path.isdir(local) else "Default"


def build_delmax_env(
    args: dict,
    *,
    layout_name: str | None = None,
    seed: int = 0,
    limit_mode: str = "train",
    fixed_limit: float | None = None,
    wind_overrides: dict | None = None,
    turbbox_path: str | None = None,
):
    """Rebuild the training env stack for a delmax/randlim checkpoint.

    Args:
        args: back-filled checkpoint args (from load_checkpoint).
        layout_name: layout to build; defaults to the checkpoint's first
            training layout. Must resolve via helpers.layouts.
        seed: seeds the base env's action space + MultiLayoutEnv RNGs.
        limit_mode: "train" reproduces training limit behavior (randlim:
            U[lo,hi] per episode; plain delmax: fixed allowed_increase).
            "fixed" pins every episode to `fixed_limit` (still overridable
            per reset via options={"del_limit": x} and live via
            find_del_wrapper(env).set_limit(x)).
        fixed_limit: required when limit_mode="fixed".
        wind_overrides: optional {"ws":..,"wd":..,"ti":..} pinning
            config["wind"] min=max away from the training inflow (flag the
            capture as OOD in metadata when used). Default: the preset's
            training inflow (ws=10, wd=270, TI=0.06 for these runs).
        turbbox_path: Mann box dir; default TransformerSac/boxes or windgym's
            bundled "Default" when absent.

    Returns:
        (env, config): the MultiLayoutEnv and the resolved env config dict.
    """
    if not str(args["config"]).startswith("power_max_derate"):
        raise ValueError(
            f"Checkpoint config is {args['config']!r}; the interp pipeline "
            "only rebuilds power_max_derate* (delmax/randlim) envs."
        )
    if limit_mode not in ("train", "fixed"):
        raise ValueError(f"limit_mode must be 'train' or 'fixed', got {limit_mode!r}")
    if limit_mode == "fixed" and fixed_limit is None:
        raise ValueError("limit_mode='fixed' requires fixed_limit")

    wt = make_derating_turbine(
        args["turbtype"], iea34_variant=args["iea34_variant"]
    )

    config = make_env_config(args["config"])
    config["ActionMethod"] = args["action_type"]
    if args.get("derate_step_sim") is not None:
        config["derate_step_sim"] = args["derate_step_sim"]
    # Same sensor-history override loop as the trainer (derate_mes only exists
    # in the _dobs preset).
    for mes_type, prefix in {
        "ws_mes": "ws", "wd_mes": "wd", "yaw_mes": "yaw",
        "power_mes": "power", "derate_mes": "derate",
    }.items():
        if mes_type not in config:
            continue
        config[mes_type][f"{prefix}_history_N"] = args["history_length"]
        config[mes_type][f"{prefix}_history_length"] = args["history_length"]
    # DEL-active trainer path: deterministic greedy reference farm.
    config["BaseController"] = "Global"

    if wind_overrides:
        wind_map = {"ws": ("ws_min", "ws_max"), "wd": ("wd_min", "wd_max"),
                    "ti": ("TI_min", "TI_max")}
        for key, val in wind_overrides.items():
            lo_key, hi_key = wind_map[key]
            config["wind"][lo_key] = config["wind"][hi_key] = float(val)

    if layout_name is None:
        layout_name = str(args["layouts"]).split(",")[0].strip()
    x_pos, y_pos = get_layout_positions(layout_name, wt)

    if turbbox_path is None:
        turbbox_path = _default_turbbox()

    base_env_kwargs = dict(
        turbine=wt,
        n_passthrough=args["max_eps"],
        TurbBox=turbbox_path,
        config=config,
        turbtype=args["TI_type"],
        backend=args["backend"],
        dt_sim=args["dt_sim"],
        dt_env=args["dt_env"],
        yaw_step_sim=args["yaw_step"],
        Baseline_comp=True,  # the DEL wrapper's greedy reference farm
    )

    def env_factory(xp, yp):
        env = WindFarmEnv(x_pos=xp, y_pos=yp, reset_init=False, **base_env_kwargs)
        env.action_space.seed(seed)
        return env

    randlim = is_randlim(args)

    def combined_wrapper(env):
        env = PerTurbineObservationWrapper(env)
        if args["use_wd_deviation"]:
            env = EnhancedPerTurbineWrapper(env, wd_scale_range=args["wd_scale_range"])
        # DEL surrogate set follows the checkpoint's turbine (same derivation
        # as the trainers); del_channels drives multi-channel penalties.
        del_turbine = args.get("del_artifact_set") or {
            "IEA34": "iea34", "DTU10MW": "dtu10mw",
        }[args["turbtype"]]
        del_channels = [
            ch.strip() for ch in str(args["del_channels"]).split(",") if ch.strip()
        ]
        del_kwargs = dict(
            turbine=del_turbine,
            channels=del_channels,
            reward_channels=del_channels,
            penalty_scale=args["del_penalty_scale"],
            allowed_increase=args["del_allowed_increase"],
            ti_window=args["del_ti_window"],
            n_r=3,
            n_theta=12,
        )
        if randlim:
            # limit_range switches the limit-obs column ON (obs width +1);
            # obs_ref must stay the training value or the column rescales.
            del_kwargs["limit_range"] = (args["del_limit_lo"], args["del_limit_hi"])
            del_kwargs["limit_obs_ref"] = args["del_limit_obs_ref"]
        if limit_mode == "fixed":
            # fixed_limit pins the episode limit but leaves limit_obs alone,
            # so it is safe for both randlim and plain-delmax envs.
            del_kwargs["fixed_limit"] = float(fixed_limit)
        return DELRewardWrapper(env, **del_kwargs)

    env = MultiLayoutEnv(
        layouts=[LayoutConfig(name=layout_name, x_pos=x_pos, y_pos=y_pos)],
        env_factory=env_factory,
        per_turbine_wrapper=combined_wrapper,
        seed=seed,
        shuffle=False,          # stable turbine order: token i == turbine i
        max_turbines=args["max_turbines"],
        max_episode_steps=None,  # rollout length is the caller's loop, not a truncation
    )
    return env, config


def find_del_wrapper(env) -> DELRewardWrapper:
    """Walk the wrapper chain inside a (single-layout) MultiLayoutEnv to the
    DELRewardWrapper -- the live limit seam (set_limit / fixed_limit). Stable
    across resets because a single-layout MultiLayoutEnv never rebuilds its
    wrapped env."""
    node = env._current_env
    while node is not None:
        if isinstance(node, DELRewardWrapper):
            return node
        node = getattr(node, "env", None)
    raise RuntimeError("No DELRewardWrapper in the env's wrapper chain")


def base_env_of(env):
    """The raw WindFarmEnv under all wrappers (for current_derate, fs, ...)."""
    return env._current_env.unwrapped


def checkpoint_obs_width(checkpoint) -> int:
    return int(checkpoint["actor_state_dict"]["obs_encoder.0.weight"].shape[1])


def build_actor(args: dict, env, checkpoint, device: torch.device) -> TransformerActor:
    """Reconstruct the TransformerActor from saved args + the rebuilt env's
    dims and load the checkpoint weights (modeled on
    eval_tracking_figs.build_agent: vector action scale/bias for yaw+derate)."""
    obs_dim_per_turbine = int(env.observation_space.shape[-1])
    ckpt_width = checkpoint_obs_width(checkpoint)
    if ckpt_width != obs_dim_per_turbine:
        raise RuntimeError(
            f"Obs width mismatch: checkpoint actor expects {ckpt_width}/turbine "
            f"but the rebuilt env produces {obs_dim_per_turbine}/turbine. The "
            "usual cause is the DEL-limit obs column: randlim checkpoints "
            "(del_limit_random=True) carry one extra column per turbine that "
            "plain delmax envs lack (and vice versa). Check that the "
            "checkpoint's del_limit_random/config args survived into the env "
            "build."
        )

    action_dim_per_turbine = int(getattr(env, "action_dim_per_turbine", 1))
    # Scale/bias shape must match the checkpoint's saved buffers: scalars for
    # derate-only, one value per action dim (2,) for yaw+derate.
    if action_dim_per_turbine == 1:
        action_high = float(env.action_space.high.reshape(-1)[0])
        action_low = float(env.action_space.low.reshape(-1)[0])
    else:
        action_high = np.asarray(env.action_space.high[0], dtype=np.float32)
        action_low = np.asarray(env.action_space.low[0], dtype=np.float32)
    action_scale = (action_high - action_low) / 2.0
    action_bias = (action_high + action_low) / 2.0

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

    args_ns = Namespace(**args)
    actor = TransformerActor(
        obs_dim_per_turbine=obs_dim_per_turbine,
        action_dim_per_turbine=action_dim_per_turbine,
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

    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()
    return actor


def make_batch_fn(args: dict, env, device: torch.device):
    """Closure obs -> (obs_t, pos_t, mask_t, rec_t, inf_t) actor-ready tensors
    for the CURRENT env step (reads env.wd live, so wind-relative transforms
    track the episode). Profiles are computed once per layout here."""
    base = base_env_of(env)
    layout = env.current_layout
    rotor_diameter = float(env.rotor_diameter)
    recep_profiles, influ_profiles = compute_profiles(
        args, layout.x_pos, layout.y_pos, rotor_diameter, base.turbine,
    )
    return partial(
        prepare_batch,
        env=env,
        rotor_diameter=rotor_diameter,
        n_turbines=env.n_turbines,
        n_turbines_max=env.max_turbines,
        use_wind_relative=args["use_wind_relative_pos"],
        use_profiles=args["profile_encoding_type"] is not None,
        recep_profiles=recep_profiles,
        influ_profiles=influ_profiles,
        n_profile_directions=args["n_profile_directions"],
        rotate=args["rotate_profiles"],
        device=device,
    )


def git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def args_hash(args: dict) -> str:
    """Stable digest of the checkpoint args (non-serializable values repr'd)."""
    blob = json.dumps({k: repr(v) for k, v in sorted(args.items())})
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def run_name_of(checkpoint_path: str) -> tuple[str, str]:
    """(run_name, step_tag) from runs/<run>/checkpoints/step_<N>.pt."""
    ckpt = os.path.abspath(checkpoint_path)
    run_dir = os.path.dirname(os.path.dirname(ckpt))
    step_tag = os.path.splitext(os.path.basename(ckpt))[0]
    return os.path.basename(run_dir), step_tag


def make_metadata(checkpoint_path: str, checkpoint, args: dict, **extra) -> str:
    """JSON metadata blob stored inside every capture/direction/sweep npz."""
    run_name, step_tag = run_name_of(checkpoint_path)
    meta = dict(
        checkpoint=os.path.abspath(checkpoint_path),
        run_name=run_name,
        step=int(checkpoint.get("step", -1)),
        step_tag=step_tag,
        args_hash=args_hash(args),
        config=args["config"],
        randlim=is_randlim(args),
        obs_width=checkpoint_obs_width(checkpoint),
        del_channel=DEL_CHANNEL,
        action_cols={"yaw": YAW_COL, "derate": DERATE_COL},
        git_rev=git_rev(),
    )
    meta.update(extra)
    return json.dumps(meta)
