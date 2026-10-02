"""Tracking-eval figure generator: ONE tracking checkpoint (derate-only or
yaw+derate) x the track3 training episode -> a stack of flow-field PNGs (and an
optional GIF).

This is the visual counterpart to eval_checkpoint.py (which produces NetCDF for
the power-max yaw trainer). It drives WindGym's figure-generating rollout
``eval_single_fast`` (aka ``AgentEvalFast``) on the SAME fixed track3 episode
used to train the power-tracking Transformer-SAC agent
(``transformer_sac_windfarm_tracking.py``), then renders the derating-oriented
right-column panels (farm power + reference / per-turbine derating / per-turbine
power, plus turbine yaws for yaw+derate agents) that ``agent_eval.py`` draws
when it auto-detects a derate/tracking env. The control mode is read from the
checkpoint's saved args (``config``/``action_type``/``power_schedule``), so
derate-only and yaw+derate checkpoints both rebuild their exact training env.

Two things this script has to bridge:

1. ``eval_single_fast`` was written for flat-obs CleanRL models (``.get_action``)
   or ``.predict`` models -- it has never driven the per-turbine TransformerActor,
   which needs tokens + positions assembled by ``WindFarmAgent``. The tiny
   ``_TransformerEvalModel`` adapter below presents a ``model_type="CleanRL"``
   ``get_action`` face while delegating to ``WindFarmAgent.act`` with precomputed
   (static) wind direction / positions / masks -- which bypasses the async-vector
   ``get_attr`` IPC that ``BatchPreparer.from_envs`` would otherwise do.

2. The trainer inlines its env build (no reusable factory), so we re-derive the
   track3 ``base_env_kwargs`` / greedy / power-reference here, kept faithful to
   ``transformer_sac_windfarm_tracking.py:236-402``. If that trainer config
   drifts, this script must track it.

Usage (from TransformerSac/, via ``pixi run``):
    python eval_tracking_figs.py \
        --checkpoint runs/track3_pywake_100k_s1/checkpoints/step_100050.pt \
        --t-sim 800 \
        --fig-dir runs/track3_pywake_100k_s1/eval_figs/
"""

import argparse
import os
import shutil
import subprocess
import sys
from functools import partial

import numpy as np
import torch

# ---- Imports from the TransformerSac codebase ----
from config import Args
from networks import TransformerActor, create_profile_encoding
from helpers.agent import WindFarmAgent
from helpers.env_configs import make_env_config
from helpers.helper_funcs import EnhancedPerTurbineWrapper
from helpers.derating_turbine import (
    make_derating_turbine,
    make_operating_point_lookup_for,
)
from helpers.power_ref import (
    stepwise_power_ref,
    measure_greedy,
    DEFAULT_SCHEDULE,
    BOOST_SCHEDULE,
)
from WindGym import WindFarmEnv, AgentEvalFast
from WindGym.farm_eval import FarmEval
from WindGym.wrappers import PerTurbineObservationWrapper


# Fixed inflow for the track3 power-tracking episode (matches the
# "power_tracking" env config's ws/TI/wd bounds).
TRACK3_WS = 10.0
TRACK3_TI = 0.06
TRACK3_WD = 270.0


def load_checkpoint(checkpoint_path: str, device: torch.device):
    """Load ``args`` (dict, defaults back-filled) + the full checkpoint dict.

    Mirrors eval_checkpoint.load_actor_from_checkpoint but also back-fills any
    args added since the checkpoint was written, so attribute/dict access never
    KeyErrors on older runs.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args_obj = checkpoint["args"]
    args = vars(args_obj) if not isinstance(args_obj, dict) else dict(args_obj)

    defaults = vars(Args())
    missing = [k for k in defaults if k not in args]
    args = {**defaults, **args}  # checkpoint values win
    if missing:
        print(f"[Eval] Back-filled {len(missing)} missing args from defaults: "
              f"{sorted(missing)}")
    # Backfill guard: config.Args now defaults turbtype to "IEA34", but every
    # checkpoint written before the switch trained on the DTU10MW surrogate —
    # a checkpoint whose args LACK turbtype must rebuild DTU.
    if "turbtype" in missing:
        args["turbtype"] = "DTU10MW"
        print('[Eval] Checkpoint predates turbtype arg — pinned to "DTU10MW"')
    return checkpoint, args


# Per-turbine sensor-history generations of helpers/env_configs._NOW_AND_T2,
# keyed by SAMPLES PER SENSOR. The block was widened from 2 samples over 3
# steps (pywake-era checkpoints, obs 8/turbine) to 3 samples over 10 steps
# (dynamiks-era, obs 11/turbine); the env must be rebuilt with the generation
# a checkpoint was TRAINED on or the actor's obs width won't match. The width
# only pins the sample COUNT, so the sample TIMING is looked up here.
_MES_GENERATIONS = {
    2: dict(current=False, rolling_mean=True, history_N=2, history_length=3,
            window_length=1),
    3: dict(current=False, rolling_mean=True, history_N=3, history_length=10,
            window_length=1),
}


def build_track3_env(args: dict, turbbox_path: str, obs_width: int | None = None,
                     *, baseline_comp: bool = False):
    """Rebuild the exact track3 tracking env used in training (unvectorized).

    Faithful to transformer_sac_windfarm_tracking.py's ENV SETUP block: derate-
    capable DTU10MW surrogate, the checkpoint's saved env config (derate-only
    "power_tracking" or yaw+derate "power_tracking_yaw"), a 3-turbine row at
    x=[0, 6D, 12D], and the saved fraction-of-greedy power reference schedule
    (80/60/70/100% default, or 80/115/70/100% "boost" -- the 115% segment is
    reachable only via yaw wake steering). Greedy is measured once with a
    throwaway probe env at the fixed inflow, with yaw DISABLED so the baseline
    is the true no-steering full-power ceiling, exactly as the trainer does.

    The base env is a ``FarmEval`` (a WindFarmEnv subclass): it adds the
    set_wind_vals/set_yaw_vals surface eval_single_fast calls, and its reset()
    installs a large non-truncating "sandbox" time_max so a fixed-length rollout
    never hits the truncation guard. It is left UNvectorized (eval_single_fast
    asserts no parent_pipes) and NOT wrapped in MultiLayoutEnv (no shuffle), so
    turbine order is stable and matches the positions we hand the actor.

    Returns ``(base_env, wrapped_env, D, greedy)``: ``base_env`` is driven
    directly by eval_single_fast; ``wrapped_env`` is the per-turbine wrapper
    stack on top of the SAME base env, used only to (a) read the actor's
    per-token obs width and (b) transform each flat base observation into
    per-turbine tokens in the eval adapter, exactly as training did; ``greedy``
    is the measured full-power farm baseline (W) the reference schedule scales
    against (interactive_eval.py needs it for its slider range).

    ``obs_width`` is the per-turbine obs width the checkpoint's actor expects
    (its first-layer in_features). When given, the per-turbine sensor blocks
    are rebuilt for the matching _MES_GENERATIONS entry, so checkpoints
    trained before the sensor-history widening still load.

    ``baseline_comp`` builds the FarmEval with a parallel greedy baseline farm
    (Baseline_comp=True, BaseController "Global" so the baseline holds yaw=0 /
    derate=0), which DELRewardWrapper needs to compare DELs against. The greedy
    probe stays single-farm. Default False keeps existing callers unchanged.
    """
    wt = make_derating_turbine(
        args["turbtype"], iea34_variant=args["iea34_variant"]
    )
    D = wt.diameter()
    # Steady-state pitch/RPM table (same table source as the power
    # surrogate) -> eval figures grow blade-pitch and rotor-RPM panels.
    op_lookup = make_operating_point_lookup_for(
        args["turbtype"], iea34_variant=args["iea34_variant"], rotor_diameter=D
    )

    # Env config + action method from the SAVED args (mirrors the trainer).
    # load_checkpoint back-fills defaults, but every track3 checkpoint saved an
    # explicit power_tracking* config; anything else means this checkpoint was
    # not trained by the tracking trainer, so fail loudly.
    if not str(args["config"]).startswith("power_tracking"):
        raise ValueError(
            f"Checkpoint config is {args['config']!r}, expected a "
            "power_tracking* preset -- this script only evals tracking runs."
        )
    config = make_env_config(args["config"])
    config["ActionMethod"] = args["action_type"]

    if obs_width is not None:
        # Per-turbine token = 3 sensors (ws/power/yaw) x samples + broadcast
        # tracking tail (setpoint + error + preview).
        tail = 2 + config["track_def"]["track_obs_preview"]
        samples, rem = divmod(int(obs_width) - tail, 3)
        if rem == 0 and samples in _MES_GENERATIONS:
            block = _MES_GENERATIONS[samples]
            for prefix in ("ws", "power", "yaw"):
                config[f"{prefix}_mes"] = {
                    f"{prefix}_{k}": v for k, v in block.items()
                }
            print(f"[Eval] Checkpoint expects {obs_width} obs/turbine -> "
                  f"{samples}-sample sensor blocks "
                  f"(history_N={block['history_N']}, "
                  f"history_length={block['history_length']}).")
        else:
            print(f"[Eval] WARNING: cannot map checkpoint obs width {obs_width} "
                  f"to a known sensor generation (tail={tail}); keeping the "
                  "current config -- expect a load_state_dict size mismatch "
                  "if the config drifted.")

    # Shared env kwargs. NOTE: no max_time_steps -- FarmEval.reset() sets a large
    # sandbox time_max, and the WindFarmEnv probe only needs a settled reset.
    base_env_kwargs = {
        "turbine": wt,
        "n_passthrough": args["max_eps"],
        "TurbBox": turbbox_path,
        "config": config,
        "turbtype": args["TI_type"],
        "backend": args.get("backend", "pywake"),
        "dt_sim": args["dt_sim"],
        "dt_env": args["dt_env"],
        "yaw_step_sim": args["yaw_step"],
        "op_lookup": op_lookup,
    }

    x_pos = np.arange(3) * 6 * D
    y_pos = np.zeros(3)

    # Greedy (settled waked farm power) from a probe env built WITHOUT the power
    # reference -- exactly the trainer's ordering, so the reference cycle matches.
    # Yaw is disabled on the probe so measure_greedy's all-min action means
    # "full power, no steering" even for yaw+derate configs; otherwise the
    # boost schedule's >100% segments would scale off a steered (wrong) greedy.
    probe_config = {**config, "yaw_action": False}

    def _make_probe_env():
        return WindFarmEnv(
            x_pos=x_pos, y_pos=y_pos, reset_init=False,
            **{**base_env_kwargs, "config": probe_config},
        )

    schedule = (
        BOOST_SCHEDULE if args.get("power_schedule") == "boost" else DEFAULT_SCHEDULE
    )

    print("Measuring greedy farm power (probe env)...")
    greedy = measure_greedy(_make_probe_env)
    print(f"  GREEDY = {greedy / 1e6:.3f} MW; reference cycle (MW): "
          f"{[round(f * greedy / 1e6, 2) for _, f in schedule]}")

    base_env_kwargs["power_ref_function"] = partial(
        stepwise_power_ref, greedy=greedy, schedule=schedule
    )

    if baseline_comp:
        # "Global" holds the baseline farm at yaw=0/derate=0 (true greedy),
        # applied only to the FarmEval build -- the probe above stays single-farm.
        base_env_kwargs["config"] = {**config, "BaseController": "Global"}

    base_env = FarmEval(
        x_pos=x_pos, y_pos=y_pos, reset_init=False, finite_episode=False,
        Baseline_comp=baseline_comp,
        **base_env_kwargs,
    )

    # Same per-turbine wrapper stack the trainer's combined_wrapper builds (minus
    # the reward-scale wrapper, which is irrelevant to figures). Rebuilt from the
    # saved args so obs width matches the checkpoint. We keep the wrapper objects
    # around to transform observations; the ROLLOUT drives base_env directly.
    wrapped = PerTurbineObservationWrapper(base_env)
    if args["use_wd_deviation"]:
        wrapped = EnhancedPerTurbineWrapper(wrapped, wd_scale_range=args["wd_scale_range"])

    return base_env, wrapped, float(D), greedy


def build_agent(args: dict, env, rotor_diameter: float, device: torch.device):
    """Reconstruct the TransformerActor + WindFarmAgent from the saved args and
    the built env's dims (mirrors eval_checkpoint.build_agent, but reads dims off
    the unvectorized env instead of a vector env's get_attr)."""
    obs_dim_per_turbine = env.observation_space.shape[-1]
    # 1 for derate-only, 2 for yaw+derate (the wrapper's per-turbine action
    # width mirrors the base env's act_var).
    action_dim_per_turbine = int(getattr(env.unwrapped, "act_var", 1))
    # Action bounds must reproduce the SHAPE of the checkpoint's saved
    # scale/bias buffers or load_state_dict rejects them: derate-only
    # checkpoints store scalars (shape ()), yaw+derate checkpoints store one
    # value per action dim (shape (2,), from the trainer's
    # ``envs.single_action_space.high[0]`` on the (n_turb, 2) action space).
    if action_dim_per_turbine == 1:
        action_high = float(env.action_space.high.reshape(-1)[0])
        action_low = float(env.action_space.low.reshape(-1)[0])
    else:
        action_high = np.asarray(env.action_space.high[0], dtype=np.float32)
        action_low = np.asarray(env.action_space.low[0], dtype=np.float32)
    action_scale = (action_high - action_low) / 2.0
    action_bias = (action_high + action_low) / 2.0

    print(f"Obs dim per turbine: {obs_dim_per_turbine}")
    print(f"Action dim per turbine: {action_dim_per_turbine}")
    print(f"Action scale/bias: {action_scale} / {action_bias}")

    # Profile encoders (a no-op for the tracking run, which trains with
    # profile_encoding_type=None -> use_profiles=False).
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

    from argparse import Namespace
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

    agent = WindFarmAgent(
        actor=actor,
        device=device,
        rotor_diameter=rotor_diameter,
        use_wind_relative=args["use_wind_relative_pos"],
        use_profiles=use_profiles,
        rotate_profiles=args["rotate_profiles"],
    )
    return actor, agent


def compute_track3_profiles(args: dict, x_pos, y_pos, turbine):
    """Receptivity/influence profiles for the fixed track3 layout, or (None,
    None) when the checkpoint trained without profile encoding.

    Mirrors eval_checkpoint.build_layout_config: the profiles depend only on
    the layout geometry (and profile args), so they are computed once here and
    handed to the eval adapter as precomputed state -- there is no
    MultiLayoutEnv in this unvectorized eval to get_attr them from.
    """
    if args["profile_encoding_type"] is None:
        return None, None

    source = str(args["profile_source"]).lower()
    if source == "geometric":
        from helpers.geometric_profiles import compute_layout_profiles_vectorized

        print("Computing GEOMETRIC profiles for the track3 layout...")
        receptivity, influence = compute_layout_profiles_vectorized(
            x_pos, y_pos,
            rotor_diameter=turbine.diameter(),
            k_wake=0.04,
            n_directions=args["n_profile_directions"],
            sigma_smooth=args.get("profile_sigma_smooth", 10.0),
            scale_factor=15.0,
            mode=args.get("profile_geom_mode", "wake"),
        )
    elif source == "pywake":
        from helpers.receptivity_profiles import compute_layout_profiles

        print("Computing PyWake profiles for the track3 layout...")
        receptivity, influence = compute_layout_profiles(
            x_pos, y_pos, turbine,
            n_directions=args["n_profile_directions"],
        )
    else:
        raise ValueError(
            f"Unknown profile_source: {args['profile_source']}. "
            "Use 'pywake' or 'geometric'."
        )
    return receptivity, influence


def build_obs_transform(base_env, wrapped_env):
    """Return ``flat_obs -> per_turbine_obs`` matching the training wrapper stack.

    Walks the wrapper chain from ``base_env`` up to ``wrapped_env`` and composes
    each wrapper's observation transform, so a flat base observation is turned
    into the exact per-turbine token layout the actor was trained on (including
    the broadcast tracking tail and, if enabled, the WD->deviation transform).
    Reuses the wrappers' own transform code rather than reimplementing it.
    """
    chain = []
    env = wrapped_env
    while env is not base_env:
        chain.append(env)
        env = env.env
    chain.reverse()  # base-most wrapper first

    def transform(flat_obs):
        obs = flat_obs
        for w in chain:
            if isinstance(w, PerTurbineObservationWrapper):
                obs = w._reshape_obs_to_per_turbine(obs)
            elif isinstance(w, EnhancedPerTurbineWrapper):
                obs = w._transform_observation(obs)
            else:
                raise TypeError(f"Unhandled wrapper in obs transform: {type(w)}")
        return obs  # (n_turb, obs_dim_per_token)

    return transform


class _TransformerEvalModel:
    """Adapter presenting a CleanRL ``get_action`` face over ``WindFarmAgent``.

    ``eval_single_fast`` dispatches on ``model_type == "CleanRL"`` and calls
    ``model.get_action(obs_tensor, deterministic=...)`` with a leading batch dim
    (it does ``obs = np.expand_dims(obs, 0)`` first). Because we drive the raw
    base env, ``obs_tensor`` is the FLAT base observation; we transform it into
    per-turbine tokens (``transform``), then forward to ``WindFarmAgent.act``
    with the static wind direction / positions / masks so it never touches the
    (nonexistent) vector-env IPC. The returned action is flattened by the caller
    for ``env.step``.
    """

    model_type = "CleanRL"

    def __init__(self, agent, transform, wind_dirs, raw_positions, masks,
                 receptivity=None, influence=None):
        self.agent = agent
        self._transform = transform
        self._wd = wind_dirs
        self._pos = raw_positions
        self._mask = masks
        # Precomputed (1, n_turb, n_directions) layout profiles for
        # profile-encoding checkpoints; None otherwise.
        self._recep = receptivity
        self._infl = influence

    def get_action(self, obs_tensor, deterministic=False):
        flat = obs_tensor.detach().cpu().numpy()[0]        # (flat_obs_dim,)
        per_turb = self._transform(flat)[None]             # (1, n_turb, obs_dim)
        a = self.agent.act(
            None, per_turb.astype(np.float32), deterministic=deterministic,
            wind_dirs=self._wd, raw_positions=self._pos, masks=self._mask,
            receptivity=self._recep, influence=self._infl,
        )  # (1, n_turb) for act dim 1, (1, n_turb, act_dim) for act dim 2
        a = np.asarray(a)
        if a.ndim == 3:
            # The base env's flat action layout is VARIABLE-major
            # ([yaw_0..yaw_n | derate_0..derate_n], wind_farm_env._adjust_yaws /
            # _apply_derate), but the agent returns turbine-major rows. Transpose
            # before flattening -- same as PerTurbineObservationWrapper.
            # _flatten_action -- so the caller's .flatten() becomes a no-op.
            a = a[0].T.reshape(1, -1)
        return torch.as_tensor(a), None, None  # caller flattens -> (n_turb * act_dim,)


def build_gif(fig_dir: str, name: str, fps: int, frame_stride: int) -> None:
    """Stitch the img_%05d.png frames in ``fig_dir`` into ``<name>.gif`` via
    ffmpeg (two-pass palette for decent quality). No-op with a warning if ffmpeg
    is missing."""
    if shutil.which("ffmpeg") is None:
        print("[GIF] ffmpeg not found on PATH; skipping GIF assembly.")
        return

    out_path = os.path.join(fig_dir, f"{name}.gif")
    frames = os.path.join(fig_dir, "img_%05d.png")
    palette = os.path.join(fig_dir, "_palette.png")

    # The frames are saved with bbox_inches="tight", so their pixel size
    # drifts a few px with tick-label widths — but ffmpeg needs one constant
    # frame size ("Error while filtering: Internal bug" on a mid-stream size
    # change; a scale/pad filter does NOT save the paletteuse pass, since
    # complex filtergraphs cannot reinit on an input size change). Pad the
    # PNGs to the max WxH canvas ON DISK once, then feed uniform frames.
    from PIL import Image

    pngs = sorted(
        os.path.join(fig_dir, f)
        for f in os.listdir(fig_dir)
        if f.startswith("img_") and f.endswith(".png")
    )
    w = h = 0
    for f in pngs:
        with Image.open(f) as im:
            w, h = max(w, im.width), max(h, im.height)
    w += w % 2  # even dims so the same frames also feed mp4/yuv420p encoders
    h += h % 2
    for f in pngs:
        with Image.open(f) as im:
            if im.size == (w, h):
                continue
            canvas = Image.new("RGB", (w, h), "white")
            canvas.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
        canvas.save(f)

    # A [x]-producing video-filter chain that is always non-empty (a bare "[x]"
    # would be an invalid filtergraph). stride>1 decimates every Nth frame and
    # resets timestamps so the GIF stays evenly paced at `fps`.
    if frame_stride > 1:
        vfilter = f"select='not(mod(n\\,{frame_stride}))',setpts=N/({fps}*TB)"
    else:
        vfilter = "null"

    gen = [
        "ffmpeg", "-y", "-framerate", str(fps), "-i", frames,
        "-vf", f"{vfilter},palettegen", "-update", "1", "-frames:v", "1", palette,
    ]
    use = [
        "ffmpeg", "-y", "-framerate", str(fps), "-i", frames, "-i", palette,
        "-lavfi", f"{vfilter}[x];[x][1:v]paletteuse", out_path,
    ]
    print(f"[GIF] Building {out_path} (fps={fps}, stride={frame_stride})...")
    for cmd in (gen, use):
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"[GIF] ffmpeg failed (exit {proc.returncode}):\n"
                  f"{proc.stderr[-2000:]}")
            return
    if os.path.exists(palette):
        os.remove(palette)
    print(f"[GIF] Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate tracking-eval figures for one tracking checkpoint "
                    "(derate-only or yaw+derate; mode read from the checkpoint's "
                    "saved args) on the track3 training episode.")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to runs/<run>/checkpoints/step_<N>.pt")
    parser.add_argument("--t-sim", type=int, default=800,
                        help="Seconds to simulate (<= env horizon; schedule is 800).")
    parser.add_argument("--fig-dir", default=None,
                        help="Directory for the PNG frames + GIF "
                             "(default: <run>/eval_figs/).")
    parser.add_argument("--turbbox-path", default="./boxes/")
    parser.add_argument("--stochastic", action="store_true",
                        help="Sample actions instead of the deterministic mean.")
    parser.add_argument("--no-gif", action="store_true",
                        help="Skip ffmpeg GIF assembly (keep only the PNGs).")
    parser.add_argument("--fps", type=int, default=15, help="GIF frame rate.")
    parser.add_argument("--frame-stride", type=int, default=1,
                        help="Keep every Nth PNG frame in the GIF (all PNGs are "
                             "still written; this only decimates the GIF).")
    parser.add_argument("--turbtype", default=None,
                        choices=["IEA34", "DTU10MW"],
                        help="Override the checkpoint's turbine (default: the "
                             "checkpoint's saved turbtype).")
    parser.add_argument("--iea34-variant", default=None,
                        choices=["annrpm", "minct"],
                        help="Override the checkpoint's IEA34 table variant.")
    cli = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    device = torch.device("cpu")
    deterministic = not cli.stochastic

    ckpt_path = os.path.abspath(cli.checkpoint)
    run_dir = os.path.dirname(os.path.dirname(ckpt_path))  # runs/<run>/
    run_name = os.path.basename(run_dir)
    step = os.path.splitext(os.path.basename(ckpt_path))[0]  # e.g. step_100050
    name = f"{run_name}_{step}"

    fig_dir = cli.fig_dir if cli.fig_dir is not None else os.path.join(run_dir, "eval_figs")
    os.makedirs(fig_dir, exist_ok=True)

    print(f"[Eval] checkpoint={ckpt_path}")
    print(f"[Eval] run={run_name} step={step} t_sim={cli.t_sim} "
          f"deterministic={deterministic} fig_dir={fig_dir}")

    # ---- Load checkpoint, build env + agent ----
    checkpoint, args = load_checkpoint(ckpt_path, device)
    # CLI turbine override (default: the checkpoint's saved args).
    if cli.turbtype is not None:
        print(f"[Eval] turbtype override: {args['turbtype']} -> {cli.turbtype}")
        args["turbtype"] = cli.turbtype
    if cli.iea34_variant is not None:
        args["iea34_variant"] = cli.iea34_variant
    # The actor's first-layer in_features pins the per-turbine sensor
    # generation the checkpoint was trained on (see _MES_GENERATIONS).
    ckpt_obs_width = checkpoint["actor_state_dict"]["obs_encoder.0.weight"].shape[1]
    base_env, wrapped_env, rotor_diameter, _greedy = build_track3_env(
        args, cli.turbbox_path, obs_width=ckpt_obs_width
    )
    # Actor dims come from the WRAPPED (per-turbine) obs/action spaces, matching
    # training; the rollout below drives the base env directly.
    actor, agent = build_agent(args, wrapped_env, rotor_diameter, device)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()

    # ---- Precompute the static per-inference arrays for the adapter ----
    n_turb = base_env.n_turb
    raw_positions = np.stack([base_env.x_pos, base_env.y_pos]).T[None].astype(np.float32)  # (1, n_turb, 2)
    wind_dirs = np.array([TRACK3_WD], dtype=np.float32)
    masks = np.zeros((1, n_turb), dtype=bool)  # 3 turbines, no padding -> all valid
    transform = build_obs_transform(base_env, wrapped_env)
    receptivity, influence = compute_track3_profiles(
        args, base_env.x_pos, base_env.y_pos, base_env.turbine
    )
    if receptivity is not None:
        receptivity = np.asarray(receptivity, dtype=np.float32)[None]  # (1, n_turb, n_dir)
        influence = np.asarray(influence, dtype=np.float32)[None]
    model = _TransformerEvalModel(agent, transform, wind_dirs, raw_positions, masks,
                                  receptivity=receptivity, influence=influence)

    # ---- Run the figure-generating rollout (drives the raw base env) ----
    print(f"[Eval] Running AgentEvalFast for {n_turb} turbines...")
    AgentEvalFast(
        base_env,
        model,
        save_figs=True,
        deterministic=deterministic,
        t_sim=cli.t_sim,
        ws=TRACK3_WS,
        ti=TRACK3_TI,
        wd=TRACK3_WD,
        name=name,
        fig_dir=fig_dir,
    )
    n_png = len([f for f in os.listdir(fig_dir) if f.endswith(".png")])
    print(f"[Eval] Wrote {n_png} PNG frame(s) to {fig_dir}")

    if not cli.no_gif:
        build_gif(fig_dir, name, cli.fps, cli.frame_stride)

    print(f"[Done] {name}")


if __name__ == "__main__":
    main()
