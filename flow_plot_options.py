"""Render 5 candidate flow-field renderings for the track3 derating eval.

The derating-eval GIF's flow panel (``ax1`` in ``WindGym/agent_eval.py``) is
geometrically distorted: the track3 domain is ~2640 m wide (a 3-turbine row at
x=[0, 6D, 12D]) by 400 m tall -- a 6.6:1 aspect -- but ``eval_single_fast`` draws
it in a roughly-square axes box and never calls ``set_aspect``, so the field is
stretched ~6x vertically (rotor disks read as tall lines, wakes as vertical
blobs).

This standalone script reuses the env/agent machinery from
``eval_tracking_figs.py`` to drive the deterministic policy to a chosen env step,
captures ``env.fs`` ONCE, then renders that same flow state 5 different ways so
the user can pick which framing should replace the current panel (phase 2 wires
the winner into ``agent_eval.py``). Only the ax1 framing differs between options;
every option keeps the turbine glyphs + derate labels.

The 5 options:
  1. Equal -- true banner (~6.6:1). set_aspect("equal"), full domain, figure
     sized to the data aspect. Geometrically exact.
  2. Equal -- padded (~2:1). set_aspect("equal") but y padded to +-4D so the row
     sits in a near-2:1 rectangle (more context, less of a razor strip).
  3. Portrait -- rotated 90deg (equal). Transposed so the farm runs vertically
     (wind top->bottom), set_aspect("equal"), tall figure.
  4. Fixed non-equal (~3x vertical). Landscape set_aspect(3): de-distorts vs the
     current ~6x stretch while keeping turbines visibly large.
  5. Equal -- tight + polished (~3.7:1). set_aspect("equal"), y cropped to +-2D,
     axis ticks in meters, grid off, thicker rotor bars -- a cleaner look.

Usage (from TransformerSac/, via ``pixi run``):
    python flow_plot_options.py \
        --checkpoint runs/track3_pywake_100k_s1/checkpoints/step_100050.pt \
        --step 250
"""

import argparse
import os
import sys

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")  # headless: write PNGs, never open a window
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.patches import Ellipse

from dynamiks.views import XYView

# Reuse the exact env/agent builders the GIF generator uses; do NOT reinvent.
from eval_tracking_figs import (
    build_track3_env,
    build_agent,
    build_obs_transform,
    load_checkpoint,
    _TransformerEvalModel,
    TRACK3_WS,
    TRACK3_TI,
    TRACK3_WD,
)

DEFAULT_CHECKPOINT = "runs/track3_pywake_100k_s1/checkpoints/step_100050.pt"


def rollout_to_step(base_env, model, target_step, deterministic=True):
    """Drive the base env to ``target_step`` env-steps and return it settled.

    Mirrors ``eval_single_fast``'s driving loop (set_wind_vals -> reset(seed=0) ->
    CleanRL get_action/step), but skips all the result-array bookkeeping: we only
    need ``env.fs`` in the flow state it reaches at ``target_step``.
    """
    base_env.set_wind_vals(ws=TRACK3_WS, ti=TRACK3_TI, wd=TRACK3_WD)
    obs, _ = base_env.reset(seed=0)

    for _ in range(target_step):
        obs_b = np.expand_dims(obs, 0)
        action, _, _ = model.get_action(
            torch.as_tensor(obs_b, dtype=torch.float32), deterministic=deterministic
        )
        action = action.detach().cpu().numpy().flatten()
        obs, _reward, _term, truncated, _info = base_env.step(action)
        if truncated:
            raise RuntimeError(
                f"Env truncated before reaching --step {target_step}; the flow "
                "sim is cleaned up on truncation. Pick a smaller --step."
            )
    return base_env


def _turbine_glyph_data(env):
    """Pull the turbine positions / orientations / derate values off ``env.fs``.

    Cribbed from ``agent_eval.py:336-357``: the top-view rotor is a thin bar of
    length 2R across the wind (the sin(tilt) term gives its ~zero thickness).
    """
    wt = env.fs.windTurbines
    x_turb, y_turb = wt.positions_xyz[:2]
    yaw, tilt = wt.yaw_tilt()
    x = np.asarray(x_turb)
    y = np.asarray(y_turb)
    # diameter() may return one value per turbine; the row is single-type, so
    # collapse to a scalar radius (matches agent_eval's `D = wt.diameter()` use).
    R = float(np.atleast_1d(wt.diameter())[0]) / 2.0
    derate = getattr(env, "current_derate", None)
    return x, y, R, yaw, tilt, derate


def _label_turbines(ax, xs, ys, R, derate, derate_offset_sign=-1):
    """Draw the white index label (and derate value, if present) per turbine.

    ``derate_offset_sign`` places the derate text below (-1) or, for portrait,
    beside the index -- keeping both readable against the flow colors via a black
    stroke, exactly like the production panel.
    """
    for ii, (xc, yc) in enumerate(zip(xs, ys)):
        idx = ax.annotate(ii + 1, (xc - R, yc + R), fontsize=10, color="white")
        idx.set_path_effects(
            [path_effects.Stroke(linewidth=2, foreground="black"), path_effects.Normal()]
        )
        if derate is not None:
            dt = ax.annotate(
                f"{derate[ii]:.2f}",
                (xc - R, yc + derate_offset_sign * R),
                fontsize=10,
                color="white",
            )
            dt.set_path_effects(
                [path_effects.Stroke(linewidth=2, foreground="black"), path_effects.Normal()]
            )


def render_flow(env, style, out_path):
    """Render ``env.fs``'s current flow state into one PNG per the given style.

    ``style`` is a dict describing this option's framing. All options share the
    same flow-solve (env.fs is fixed); they differ only in view bounds, aspect,
    orientation, and polish. See module docstring for the 5 configured styles.
    """
    x, y, R, yaw, tilt, derate = _turbine_glyph_data(env)
    D = 2.0 * R

    # View bounds. x always spans the row + margins (matches agent_eval's `a`);
    # y half-height is per-style (None -> the default +-200 m from the row).
    x_lo = -200.0 + float(min(env.x_pos))
    x_hi = 300.0 + float(max(env.x_pos))
    y_pad = style.get("y_pad_D")
    if y_pad is None:
        y_lo = -200.0 + float(min(env.y_pos))
        y_hi = 200.0 + float(max(env.y_pos))
    else:
        y_lo = float(min(env.y_pos)) - y_pad * D
        y_hi = float(max(env.y_pos)) + y_pad * D

    a = np.linspace(x_lo, x_hi, 200)
    b = np.linspace(y_lo, y_hi, 200)

    width_m = x_hi - x_lo
    height_m = y_hi - y_lo
    portrait = style.get("portrait", False)

    # Figure sizing: size the canvas to the *displayed* box aspect so equal-aspect
    # options don't get letterboxed into a sliver (renderer.py:240-244 pattern).
    fig_h = style.get("fig_h", 6.0)
    if portrait:
        # Vertical farm: displayed box is height_m(x-data) x width_m(y-data).
        disp_aspect = height_m / width_m  # >1 -> tall
        figsize = (fig_h * disp_aspect + 1.5, fig_h)
    elif style["aspect"] == "equal":
        disp_aspect = width_m / height_m  # >1 -> wide
        figsize = (fig_h * disp_aspect + 1.5, fig_h)
    else:
        # Non-equal: y stretched by `aspect`; displayed box aspect = W/(H*aspect).
        disp_aspect = width_m / (height_m * float(style["aspect"]))
        figsize = (fig_h * disp_aspect + 1.5, fig_h)

    fig, ax = plt.subplots(figsize=figsize, layout="constrained")

    view = XYView(z=70, x=a, y=b, ax=ax, adaptive=False)
    uvw = env.fs.get_windspeed(view, include_wakes=True, xarray=True)  # dims (uvw, x, y)
    u = np.asarray(uvw[0])  # u-component as a plain (x, y) array

    colors = ["k", "gray", "r", "g"] * 5
    if portrait:
        # Swap axes: horizontal = y (cross-stream), vertical = x (downstream).
        mesh = ax.pcolormesh(
            uvw.y.values, uvw.x.values, u,
            shading="nearest", vmin=3, vmax=env.ws + 2,
        )
        for xc, yc, yaw_, tilt_ in zip(x, y, yaw, tilt):
            for wd_ in np.atleast_1d(env.fs.wind_direction):
                ell = Ellipse(
                    (yc, xc),  # (horizontal=y, vertical=x)
                    2 * R,                              # bar length now horizontal
                    2 * R * np.sin(np.deg2rad(tilt_)),  # ~zero thickness
                    angle=yaw_ - wd_,
                    ec="k", fc="None",
                )
                ax.add_artist(ell)
                ax.plot(yc, xc, ".", color="k")
        _label_turbines(ax, y, x, R, derate)
        ax.invert_yaxis()  # wind flows top (upstream) -> bottom (downstream)
        ax.set_xlabel("y [m]")
        ax.set_ylabel("x [m]")
    else:
        mesh = ax.pcolormesh(
            uvw.x.values, uvw.y.values, u.T,
            shading="nearest", vmin=3, vmax=env.ws + 2,
        )
        rotor_lw = style.get("rotor_lw", 1.0)
        for xc, yc, yaw_, tilt_ in zip(x, y, yaw, tilt):
            for wd_ in np.atleast_1d(env.fs.wind_direction):
                ell = Ellipse(
                    (xc, yc),
                    2 * R * np.sin(np.deg2rad(tilt_)),
                    2 * R,
                    angle=90 - wd_ + yaw_,
                    ec="k", fc="None", lw=rotor_lw,
                )
                ax.add_artist(ell)
                ax.plot(xc, yc, ".", color="k")
        _label_turbines(ax, x, y, R, derate)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")

    fig.colorbar(mesh, ax=ax).set_label("Wind speed [m/s]")

    # Aspect. Portrait + the equal options lock 1 y-m == 1 x-m; option 4 stretches.
    if portrait or style["aspect"] == "equal":
        ax.set_aspect("equal")
    else:
        ax.set_aspect(float(style["aspect"]))

    if style.get("hide_ticks", False):
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
    if style.get("grid", False):
        ax.grid(True, alpha=0.3)

    ax.set_title(f"{style['title']}  (t={env.fs.time:.0f} s)")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  wrote {out_path}  [figsize={figsize[0]:.1f}x{figsize[1]:.1f}]")


# The 5 options. Each keeps turbine glyphs + derate labels; only framing differs.
STYLES = [
    {
        "key": "option_1", "title": "1: Equal - true banner (~6.6:1)",
        "aspect": "equal", "y_pad_D": None, "fig_h": 2.4, "hide_ticks": True,
    },
    {
        "key": "option_2", "title": "2: Equal - padded to +-4D (~2:1)",
        "aspect": "equal", "y_pad_D": 4.0, "fig_h": 4.5, "hide_ticks": True,
    },
    {
        "key": "option_3", "title": "3: Portrait - rotated 90deg (equal)",
        "aspect": "equal", "portrait": True, "y_pad_D": None, "fig_h": 6.5,
    },
    {
        "key": "option_4", "title": "4: Fixed non-equal (3x vertical)",
        "aspect": 3.0, "y_pad_D": None, "fig_h": 4.5, "hide_ticks": True,
    },
    {
        "key": "option_5", "title": "5: Equal - tight +-2D + polished (~3.7:1)",
        "aspect": "equal", "y_pad_D": 2.0, "fig_h": 3.2, "rotor_lw": 2.5,
        "grid": False, "hide_ticks": False,
    },
]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help=f"Path to a step_<N>.pt checkpoint (default: {DEFAULT_CHECKPOINT}).")
    parser.add_argument("--step", type=int, default=250,
                        help="Env step to capture the flow state at (default: 250).")
    parser.add_argument("--fig-dir", default=None,
                        help="Output dir for the 5 PNGs (default: <run>/flow_options/).")
    parser.add_argument("--turbbox-path", default="./boxes/")
    parser.add_argument("--stochastic", action="store_true",
                        help="Sample actions instead of the deterministic mean.")
    cli = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    device = torch.device("cpu")
    deterministic = not cli.stochastic

    ckpt_path = os.path.abspath(cli.checkpoint)
    run_dir = os.path.dirname(os.path.dirname(ckpt_path))  # runs/<run>/
    fig_dir = cli.fig_dir or os.path.join(run_dir, "flow_options")
    os.makedirs(fig_dir, exist_ok=True)

    print(f"[Flow] checkpoint={ckpt_path}")
    print(f"[Flow] step={cli.step} deterministic={deterministic} fig_dir={fig_dir}")

    # ---- Build env + agent (identical to the GIF generator) ----
    checkpoint, args = load_checkpoint(ckpt_path, device)
    base_env, wrapped_env, rotor_diameter = build_track3_env(args, cli.turbbox_path)
    actor, agent = build_agent(args, wrapped_env, rotor_diameter, device)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()

    n_turb = base_env.n_turb
    raw_positions = np.stack([base_env.x_pos, base_env.y_pos]).T[None].astype(np.float32)
    wind_dirs = np.array([TRACK3_WD], dtype=np.float32)
    masks = np.zeros((1, n_turb), dtype=bool)
    transform = build_obs_transform(base_env, wrapped_env)
    model = _TransformerEvalModel(agent, transform, wind_dirs, raw_positions, masks)

    # ---- One rollout to the target step; env.fs then holds the shared flow ----
    print(f"[Flow] Rolling out to step {cli.step}...")
    rollout_to_step(base_env, model, cli.step, deterministic=deterministic)
    print(f"[Flow] Captured flow at t={base_env.fs.time:.0f} s; rendering 5 options...")

    # ---- Render the same flow state 5 ways ----
    for style in STYLES:
        render_flow(base_env, style, os.path.join(fig_dir, f"{style['key']}.png"))

    print(f"[Done] 5 options in {fig_dir}")


if __name__ == "__main__":
    main()
