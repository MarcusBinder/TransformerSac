"""Random-agent DEL-penalty GIF on the track3 tracking episode.

Visual sanity check for DELRewardWrapper: drive the track3 yaw+derate tracking
env with RANDOM actions (no checkpoint) and render, per env step, a frame in
the style of the existing power-tracking gifs -- flow field on the left, and on
the right the farm power vs. reference, the penalized vs. unpenalized reward,
the DEL hinge penalty / DEL ratio, and the per-turbine DELs (agent farm vs.
greedy baseline farm). Random yaw+derate actions move the DELs around enough
that the baseline-relative hinge visibly fires.

The env build (greedy probe + power reference + FarmEval) and GIF assembly are
reused from eval_tracking_figs; the rollout + frame rendering live here because
WindGym's eval_single_fast has hardcoded panels (and the windgym submodule is
untouchable). The flow-field panel copies agent_eval.py's idiom.

Usage (from TransformerSac/, via ``pixi run``):
    python random_agent_del_gif.py                       # full 800 s run + GIF
    python random_agent_del_gif.py --t-sim 100 --no-gif  # quick smoke
"""

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from dynamiks.views import XYView
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

from config import Args

# Repo root (parent of TransformerSac/): `del_surrogate` lives there and is
# not installed into the pixi env (same convention as the tracking trainer).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from del_surrogate.model import DEFAULT_CHANNEL
from del_surrogate.reward_wrapper import DELRewardWrapper
from eval_tracking_figs import (
    TRACK3_TI,
    TRACK3_WD,
    TRACK3_WS,
    build_gif,
    build_track3_env,
)


def make_env_args(mode: str) -> dict:
    """Args() defaults + the fixed track3 DEL-eval overrides.

    dt_sim=1 is what the DEL sampler's TI window needs (dynamiks-era configs);
    power_schedule stays "default" (80/60/70/100% of greedy) -- for a random
    agent the reference is only a backdrop, and the boost schedule's 115%
    segment is meaningless without a trained steering policy.
    """
    args = vars(Args())
    args.update(
        config=mode,
        backend="dynamiks",
        TI_type="MannGenerate",
        dt_sim=1,
        dt_env=10,
        power_schedule="default",
    )
    return args


@dataclass
class History:
    """Per-env-step traces for the right-hand panels."""

    t: list = field(default_factory=list)              # s, relative to reset
    farm_power: list = field(default_factory=list)     # W, agent farm
    farm_power_base: list = field(default_factory=list)  # W, baseline farm
    p_ref: list = field(default_factory=list)          # W
    reward: list = field(default_factory=list)
    reward_unpen: list = field(default_factory=list)
    penalty: list = field(default_factory=list)
    ratio: list = field(default_factory=list)
    del_agent: list = field(default_factory=list)      # (n_turb,) per step
    del_base: list = field(default_factory=list)       # (n_turb,) per step
    derates: list = field(default_factory=list)        # (n_turb,) per step


class GrowingLimits:
    """Monotonic-grow y-limits (the pow_max pattern from eval_single_fast):
    axes only ever widen across frames, so the GIF panels never jump."""

    def __init__(self):
        self.lo, self.hi = np.inf, -np.inf

    def update(self, *values):
        for v in values:
            v = np.asarray(v, dtype=float).reshape(-1)
            v = v[np.isfinite(v)]
            if v.size:
                self.lo = min(self.lo, float(v.min()))
                self.hi = max(self.hi, float(v.max()))

    def apply(self, ax, pad=0.05):
        if np.isfinite(self.lo) and np.isfinite(self.hi):
            span = (self.hi - self.lo) or 1.0
            ax.set_ylim(self.lo - pad * span, self.hi + pad * span)


def render_frame(
    frame_idx: int,
    hist: History,
    base_env,
    grid_x,
    grid_y,
    lims: dict,
    t_sim: int,
    ti_window: float,
    allowed_increase: float,
    channel: str,
    fig_dir: str,
):
    """One GIF frame: flow field left, power/reward/penalty/DEL panels right.

    All right panels show the FULL history on a fixed [0, t_sim] x-axis, so
    episodic events (schedule steps, warm-up, rare exceedances) stay in frame
    and the GIF axes never jitter.
    """
    t = np.asarray(hist.t)
    n_turb = base_env.n_turb

    fig = plt.figure(figsize=(15, 7.5))
    grid = (4, 4)
    ax1 = plt.subplot2grid(grid, (0, 0), colspan=2, rowspan=4)

    # ---- Flow field (agent_eval.py idiom; windgym is read-only) ----
    view = XYView(z=70, x=grid_x, y=grid_y, ax=ax1, adaptive=False)
    wt = base_env.fs.windTurbines
    x_turb, y_turb = wt.positions_xyz[:2]
    yaw, tilt = wt.yaw_tilt()

    uvw = base_env.fs.get_windspeed(view, include_wakes=True, xarray=True)
    mesh = ax1.pcolormesh(
        uvw.x.values,
        uvw.y.values,
        uvw[0].T,
        shading="nearest",
        vmin=3,
        vmax=base_env.ws + 2,
    )
    fig.colorbar(mesh, ax=ax1).set_label("Wind speed [m/s]")

    x, y, D = [np.asarray(v) for v in [x_turb, y_turb, wt.diameter()]]
    R = D / 2
    stroke = [path_effects.Stroke(linewidth=2, foreground="black"),
              path_effects.Normal()]
    for ii, (x_, y_, r, yaw_, tilt_) in enumerate(zip(x, y, R, yaw, tilt)):
        for wd_ in np.atleast_1d(base_env.fs.wind_direction):
            ax1.add_artist(Ellipse(
                (x_, y_),
                2 * r * np.sin(np.deg2rad(tilt_)),
                2 * r,
                angle=90 - wd_ + yaw_,
                ec="k", fc="None", lw=2.5,
            ))
            ax1.plot(x_, y_, ".", color="k")
        text = ax1.annotate(ii + 1, (x_ - r, y_ + r), fontsize=10, color="white")
        text.set_path_effects(stroke)
        dtext = ax1.annotate(
            f"{base_env.current_derate[ii]:.2f}",
            (x_ - r, y_ - r), fontsize=10, color="white",
        )
        dtext.set_path_effects(stroke)

    ax1.set_title(f"Flow field at {base_env.fs.time} s")
    ax1.set_aspect("equal")
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")

    # ---- Right column ----
    ax_pow = plt.subplot2grid(grid, (0, 2), colspan=2)
    ax_rew = plt.subplot2grid(grid, (1, 2), colspan=2)
    ax_pen = plt.subplot2grid(grid, (2, 2), colspan=2)
    ax_del = plt.subplot2grid(grid, (3, 2), colspan=2)
    right_axes = [ax_pow, ax_rew, ax_pen, ax_del]

    # Farm power vs. reference (+ thin greedy-baseline farm power).
    p_a = np.asarray(hist.farm_power) / 1e6
    p_b = np.asarray(hist.farm_power_base) / 1e6
    p_r = np.asarray(hist.p_ref) / 1e6
    ax_pow.plot(t, p_a, color="C0", label="agent farm")
    ax_pow.plot(t, p_b, color="gray", lw=1.0, label="greedy baseline")
    ax_pow.plot(t, p_r, "k--", drawstyle="steps-post", label="reference")
    lims["pow"].update(p_a, p_b, p_r)
    lims["pow"].apply(ax_pow)
    ax_pow.set_ylabel("Farm power [MW]")
    ax_pow.legend(loc="upper right", fontsize=7, ncol=3)

    # Penalized vs. unpenalized reward -- the gap IS the DEL penalty.
    rew = np.asarray(hist.reward)
    rew_u = np.asarray(hist.reward_unpen)
    ax_rew.plot(t, rew, color="C0", label="reward (penalized)")
    ax_rew.plot(t, rew_u, color="C2", ls="--", label="unpenalized")
    lims["rew"].update(rew, rew_u)
    lims["rew"].apply(ax_rew)
    ax_rew.set_ylabel("Reward")
    ax_rew.legend(loc="upper right", fontsize=7, ncol=2)

    # DEL hinge penalty (left axis) + DEL ratio vs. threshold (right axis).
    pen = np.asarray(hist.penalty)
    ratio = np.asarray(hist.ratio)
    thr = 1.0 + allowed_increase
    ax_pen.plot(t, pen, color="C3", label="DEL penalty")
    lims["pen"].update(pen, [0.0])
    lims["pen"].apply(ax_pen)
    ax_pen.set_ylabel("DEL penalty", color="C3")
    ax_pen.tick_params(axis="y", labelcolor="C3")
    ax_ratio = ax_pen.twinx()
    ax_ratio.plot(t, ratio, color="C4", ls=":", label="DEL ratio")
    ax_ratio.axhline(thr, color="C4", lw=0.8, alpha=0.7)
    ax_ratio.fill_between(
        t, thr, np.where(np.isfinite(ratio), ratio, thr),
        where=np.isfinite(ratio) & (ratio > thr),
        color="red", alpha=0.25, interpolate=True,
    )
    lims["ratio"].update(ratio, [thr])
    lims["ratio"].apply(ax_ratio)
    ax_ratio.set_ylabel(f"DEL ratio (thr {thr:.2f})", color="C4")
    ax_ratio.tick_params(axis="y", labelcolor="C4")

    # Per-turbine DELs: agent solid vs. baseline dashed, same color per turbine.
    del_a = np.asarray(hist.del_agent)  # (n_frames, n_turb)
    del_b = np.asarray(hist.del_base)
    # Raw surrogate output: the model's docs only promise "training units",
    # so no unit conversion -- the baseline-relative ratio is what matters.
    for i in range(n_turb):
        ax_del.plot(t, del_a[:, i], color=f"C{i}")
        ax_del.plot(t, del_b[:, i], color=f"C{i}", ls="--", alpha=0.6)
    lims["del"].update(del_a, del_b)
    lims["del"].apply(ax_del)
    ax_del.set_ylabel(f"DEL (model units)\n{channel}", fontsize=8)
    handles = [Line2D([], [], color=f"C{i}", label=f"T{i + 1}")
               for i in range(n_turb)]
    handles.append(Line2D([], [], color="k", ls="--", label="baseline"))
    ax_del.legend(handles=handles, loc="upper right", fontsize=7,
                  ncol=n_turb + 1)
    ax_del.set_xlabel("time [s]")

    for ax in right_axes:
        ax.set_xlim(0, t_sim)
        ax.axvspan(0, ti_window, color="0.85", zorder=0)
        ax.grid(alpha=0.4)
        if ax is not ax_del:
            ax.tick_params(labelbottom=False)
    ax_pow.set_title("DEL warm-up band shaded gray", fontsize=8)

    fig.tight_layout()
    # No bbox_inches="tight": constant 1500x750 px frames (even dims), so
    # build_gif's pad pass no-ops and the mp4 encoder gets uniform input.
    fig.savefig(os.path.join(fig_dir, f"img_{frame_idx:05d}.png"), dpi=100)
    plt.close(fig)


def rollout(
    del_env,
    base_env,
    t_sim: int,
    seed: int,
    ti_window: float,
    allowed_increase: float,
    channel: str,
    fig_dir: str,
) -> History:
    """Random-action rollout, one rendered frame per env step."""
    base_env.set_wind_vals(ws=TRACK3_WS, ti=TRACK3_TI, wd=TRACK3_WD)
    _obs, _info = del_env.reset(seed=seed)
    del_env.action_space.seed(seed)
    t0 = float(base_env.fs.time)

    # Flow-field extent, computed once (agent_eval.py idiom).
    D_view = float(np.atleast_1d(base_env.fs.windTurbines.diameter())[0])
    grid_x = np.linspace(-200 + min(base_env.x_pos),
                         300 + max(base_env.x_pos), 200)
    grid_y = np.linspace(min(base_env.y_pos) - 2 * D_view,
                         max(base_env.y_pos) + 2 * D_view, 200)

    hist = History()
    lims = {k: GrowingLimits() for k in ("pow", "rew", "pen", "ratio", "del")}
    n_steps = t_sim // base_env.dt_env
    print(f"[Rollout] {n_steps} random steps (t_sim={t_sim} s, "
          f"dt_env={base_env.dt_env} s, seed={seed})...")

    for step in range(n_steps):
        action = del_env.action_space.sample()
        _obs, reward, terminated, truncated, info = del_env.step(action)
        if terminated or truncated:
            raise RuntimeError(
                f"Env ended at step {step + 1}/{n_steps} (terminated="
                f"{terminated}, truncated={truncated}); the FarmEval sandbox "
                "time_max should make this impossible for t_sim <= 100000 s."
            )
        invariant_err = abs(
            reward - (info["reward_unpenalized"] - info["del_penalty"])
        )
        assert invariant_err < 1e-6, (
            f"reward invariant violated at step {step}: {invariant_err}"
        )

        powers = np.asarray(info["powers"])
        if powers.ndim == 2:  # (sim_samples, n_turb): last sample, farm sum
            farm_p = float(powers[-1].sum())
        else:
            farm_p = float(np.sum(base_env.fs.windTurbines.power()))
        base_powers = np.asarray(info["baseline_powers"])
        farm_p_base = (
            float(base_powers[-1].sum()) if base_powers.ndim == 2
            else float(np.sum(base_env.fs_baseline.windTurbines.power()))
        )

        hist.t.append(float(base_env.fs.time) - t0)
        hist.farm_power.append(farm_p)
        hist.farm_power_base.append(farm_p_base)
        hist.p_ref.append(float(info["Power reference"]))
        hist.reward.append(float(reward))
        hist.reward_unpen.append(float(info["reward_unpenalized"]))
        hist.penalty.append(float(info["del_penalty"]))
        hist.ratio.append(float(info["del_ratio"]))
        hist.del_agent.append(np.asarray(info["loads"][channel], float).copy())
        hist.del_base.append(
            np.asarray(info["loads_baseline"][channel], float).copy()
        )
        hist.derates.append(np.asarray(base_env.current_derate, float).copy())

        render_frame(
            step, hist, base_env, grid_x, grid_y, lims,
            t_sim, ti_window, allowed_increase, channel, fig_dir,
        )
        if (step + 1) % 10 == 0 or step == n_steps - 1:
            print(f"  step {step + 1}/{n_steps}: t={hist.t[-1]:.0f}s "
                  f"P={farm_p / 1e6:.2f}MW ref={hist.p_ref[-1] / 1e6:.2f}MW "
                  f"penalty={hist.penalty[-1]:.4f} ratio={hist.ratio[-1]:.3f}")
    return hist


def build_mp4(fig_dir: str, name: str, fps: int) -> None:
    """img_%05d.png -> <name>.mp4 (frames are already even-dim uniform)."""
    if shutil.which("ffmpeg") is None:
        print("[MP4] ffmpeg not found on PATH; skipping mp4 assembly.")
        return
    out_path = os.path.join(fig_dir, f"{name}.mp4")
    cmd = [
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", os.path.join(fig_dir, "img_%05d.png"),
        "-pix_fmt", "yuv420p", out_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"[MP4] ffmpeg failed (exit {proc.returncode}):\n"
              f"{proc.stderr[-2000:]}")
        return
    print(f"[MP4] Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Random-agent DEL-penalty GIF on the track3 tracking "
                    "episode (no checkpoint; visual check of DELRewardWrapper).")
    parser.add_argument("--mode", default="power_tracking_yaw",
                        choices=["power_tracking", "power_tracking_yaw"],
                        help="Env config preset (yaw+derate default: random yaw "
                             "moves the DELs, so the penalty visibly fires).")
    parser.add_argument("--t-sim", type=int, default=800,
                        help="Seconds to simulate (schedule period is 800).")
    parser.add_argument("--penalty-scale", type=float, default=1.0)
    parser.add_argument("--allowed-increase", type=float, default=0.0,
                        help="Allowed fractional DEL increase over the baseline "
                             "farm max before the hinge fires (0.0 = any "
                             "exceedance is penalized).")
    parser.add_argument("--ti-window", type=float, default=60.0,
                        help="DEL sampler TI window [s]; also the warm-up band.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--no-gif", action="store_true")
    parser.add_argument("--mp4", action="store_true",
                        help="Also encode an mp4 from the frames.")
    parser.add_argument("--fig-dir", default="runs/random_del_gif/eval_figs")
    parser.add_argument("--turbbox-path", default="./boxes/")
    cli = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    fig_dir = os.path.abspath(cli.fig_dir)
    os.makedirs(fig_dir, exist_ok=True)
    # build_gif globs every img_*.png in the dir, so clear stale frames first.
    for f in os.listdir(fig_dir):
        if f.startswith("img_") and f.endswith(".png"):
            os.remove(os.path.join(fig_dir, f))

    name = f"random_del_{cli.mode}_s{cli.seed}"
    print(f"[Setup] mode={cli.mode} t_sim={cli.t_sim} seed={cli.seed} "
          f"penalty_scale={cli.penalty_scale} "
          f"allowed_increase={cli.allowed_increase} fig_dir={fig_dir}")

    env_args = make_env_args(cli.mode)
    base_env, _wrapped, _D, _greedy = build_track3_env(
        env_args, cli.turbbox_path, baseline_comp=True
    )
    del_env = DELRewardWrapper(
        base_env,
        penalty_scale=cli.penalty_scale,
        allowed_increase=cli.allowed_increase,
        ti_window=cli.ti_window,
        n_r=3,
        n_theta=12,
    )
    channel = DEFAULT_CHANNEL

    try:
        hist = rollout(
            del_env, base_env, cli.t_sim, cli.seed,
            cli.ti_window, cli.allowed_increase, channel, fig_dir,
        )

        n_png = len([f for f in os.listdir(fig_dir)
                     if f.startswith("img_") and f.endswith(".png")])
        print(f"[Frames] Wrote {n_png} PNG frame(s) to {fig_dir}")

        if not cli.no_gif:
            build_gif(fig_dir, name, cli.fps, cli.frame_stride)
        if cli.mp4:
            build_mp4(fig_dir, name, cli.fps)

        pen = np.asarray(hist.penalty)
        ratio = np.asarray(hist.ratio)
        finite = np.isfinite(ratio)
        print(f"[Summary] steps={len(pen)} "
              f"penalty_active_steps={int((pen > 0).sum())} "
              f"warmup_nan_steps={int((~finite).sum())} "
              f"max_ratio={np.nanmax(ratio) if finite.any() else float('nan'):.4f} "
              f"total_penalty={pen.sum():.4f} "
              f"mean_reward={np.mean(hist.reward):.4f}")
        print(f"[Done] {name}")
    finally:
        base_env.close()


if __name__ == "__main__":
    main()
