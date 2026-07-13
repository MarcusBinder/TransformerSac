"""Interactive tracking eval: drive the trained track3 agent live.

Same env, same agent, same panels as eval_tracking_figs.py, but the farm
power reference is a matplotlib Slider the user drags while the episode
free-runs -- the agent reacts to the new setpoint within a few steps. The
control mode follows the checkpoint's saved args: derate-only, or yaw+derate
(boost-capable), where a yaw panel and live-rotating rotor ellipses show the
wake steering. The whole point is to *feel* the closed loop: drag the target,
watch the derate labels and the farm-power trace chase the dashed reference.

Two ideas make this work:

1. Live setpoint. ``PowerTrackingManager.reference()`` is the single choke
   point every consumer routes through (``push()`` -> reward deque,
   ``preview()`` -> obs preview, ``_push_tracking`` -> tracking obs), so a
   subclass whose ``reference()`` reads a shared mutable ``SetpointHolder``
   turns the precomputed schedule into a lever. It is installed on the base
   env BEFORE reset, so even reset's warm-up pushes hit the live manager.

2. Persistent artists. eval_tracking_figs' renderer rebuilds the full figure
   and saves a PNG every step -- that, not the physics, was the cost (an env
   step is ms-fast on the steady PyWake backend). Here the figure is built
   once and each frame only ``set_array``s the flow mesh and ``set_data``s
   the time-series lines, which is fast enough to free-run on a GUI timer.

Usage (from TransformerSac/, via ``pixi run``):
    python interactive_eval.py \
        --checkpoint runs/track3_pywake_100k_s1/checkpoints/step_100050.pt

Headless self-test (Agg backend, no window; asserts the slider actually
propagates into the reference, the reward path and the agent's observation):
    python interactive_eval.py --checkpoint ... --smoke 40
"""

import argparse
import sys
from collections import deque

import matplotlib

if __name__ == "__main__":
    # Pin the backend BEFORE the WindGym import below drags in pyplot (the
    # first pyplot import locks the backend). --smoke runs headless on Agg.
    _headless = any(
        a == "--smoke" or a.startswith("--smoke=") for a in sys.argv[1:]
    )
    matplotlib.use("Agg" if _headless else "TkAgg")

import matplotlib.patheffects as path_effects
import numpy as np
import torch
from matplotlib.patches import Ellipse

from dynamiks.views import XYView
from WindGym.core.power_tracking import PowerTrackingManager

# Reuses the checkpoint/env/agent plumbing from the figure generator: the env
# build (incl. the greedy probe), the actor reconstruction, the flat->tokens
# obs transform and the CleanRL-faced eval adapter. Import is clean (that
# module is main-guarded).
from eval_tracking_figs import (
    TRACK3_TI,
    TRACK3_WD,
    TRACK3_WS,
    _TransformerEvalModel,
    build_agent,
    build_obs_transform,
    build_track3_env,
    compute_track3_profiles,
    load_checkpoint,
)

import matplotlib.pyplot as plt  # noqa: E402  (after backend pinning)
from matplotlib.widgets import Button, Slider


class SetpointHolder:
    """Shared mutable setpoint (W). The slider writes it, the env's tracking
    manager reads it every reference() call. Everything runs on the single GUI
    event-loop thread, so no locking is needed."""

    def __init__(self, watts: float):
        self.value = float(watts)


class LiveSetpointManager(PowerTrackingManager):
    """PowerTrackingManager whose reference is the CURRENT holder value.

    reference() is the only method consumers use to read the trajectory
    (push() and preview() both route through it), so overriding it makes the
    reward deque, the setpoint/error observations and info["Power reference"]
    all follow the slider with no env changes.
    """

    def __init__(self, holder: SetpointHolder):
        super().__init__(ref_function=None, preview_steps=0)
        self._holder = holder

    def reset_episode(self, env, time_max: float, delay: float, power_avg: int):
        self._env = env
        self._delay = float(delay)
        # One-entry array satisfies reference()'s trajectory-is-set guard;
        # the override below never actually indexes it.
        self.trajectory = np.array([self._holder.value])
        self.ref_deque = deque(maxlen=power_avg)

    def reference(self, step_idx: int) -> float:
        return float(self._holder.value)


class InteractiveApp:
    """Live view (flow field | farm power+ref / per-turbine derate /
    per-turbine power, plus blade pitch and rotor RPM when the env carries an
    operating-point lookup, plus turbine yaws for a yaw+derate agent) with a
    target-power slider and a Pause/Play button.

    All artists are created once in __init__; tick() only pushes new data into
    them. The free-run loop is a GUI timer (not a while-loop) so slider/button
    events interleave naturally on the Tk event loop.
    """

    def __init__(self, env, model, greedy, holder, obs, *, deterministic,
                 interval_ms, flow_every, window, setpoint_init):
        self.env = env
        self.model = model
        self.greedy = float(greedy)
        self.holder = holder
        self.obs = obs
        self.deterministic = deterministic
        self.interval_ms = int(interval_ms)
        self.flow_every = max(1, int(flow_every))
        self.paused = False
        self.done = False
        self.step_count = 0
        self.last_info = None
        self.timer = None

        n_turb = env.n_turb
        # Steady-state pitch/RPM panels only when the env carries the
        # operating-point lookup (see WindGym.core.OperatingPointLookup).
        self.op_mode = getattr(env, "op_lookup", None) is not None
        # Yaw+derate agents steer while derating; same detection as
        # agent_eval's show_yaw_panel. Derate-only envs keep yaws fixed.
        self.yaw_active = (bool(getattr(env, "yaw_action", True))
                           and bool(getattr(env, "derate_action", False)))
        self.t_deq = deque(maxlen=window)
        self.pfarm_deq = deque(maxlen=window)
        self.ref_deq = deque(maxlen=window)
        self.derate_deq = deque(maxlen=window)   # (n_turb,) per entry
        self.powT_deq = deque(maxlen=window)     # (n_turb,) per entry
        if self.op_mode:
            self.pitch_deq = deque(maxlen=window)  # (n_turb,) per entry
            self.rpm_deq = deque(maxlen=window)    # (n_turb,) per entry
        if self.yaw_active:
            self.yaw_deq = deque(maxlen=window)    # (n_turb,) per entry

        # ---- Figure + axes (created once) ----
        # Wide mode: right-hand block is 3x2 on a (3, 4) grid (pitch/RPM in
        # the extra sub-column, yaws in its bottom cell); otherwise the
        # original (3, 3) single column.
        wide = self.op_mode or self.yaw_active
        self.fig = plt.figure(figsize=(16, 7.5) if wide else (13, 7.5))
        gs = self.fig.add_gridspec(3, 4 if wide else 3)
        self.ax1 = self.fig.add_subplot(gs[:, :2])
        self.ax2 = self.fig.add_subplot(gs[0, 2])
        self.ax3 = self.fig.add_subplot(gs[1, 2])
        self.ax4 = self.fig.add_subplot(gs[2, 2])
        self.ts_axes = [self.ax2, self.ax3, self.ax4]
        self.bottom_axes = [self.ax4]
        if self.op_mode:
            self.ax5 = self.fig.add_subplot(gs[0, 3])
            self.ax6 = self.fig.add_subplot(gs[1, 3])
            self.ts_axes += [self.ax5, self.ax6]
            # Lowest axis of each sub-column carries the time axis labels.
            self.bottom_axes = [self.ax4, self.ax6]
        if self.yaw_active:
            # Yaw panel in the spare (2, 3) cell; it becomes the lowest axis
            # of the right sub-column, so the time label/ticks move to it.
            self.ax7 = self.fig.add_subplot(gs[2, 3])
            self.ts_axes.append(self.ax7)
            self.bottom_axes = [self.ax4, self.ax7]
        self.fig.subplots_adjust(bottom=0.14, hspace=0.45, wspace=0.35)

        # ---- Flow field (ax1): mesh + colorbar + static rotors, once ----
        # Same extent framing as WindGym.agent_eval: x spans the row plus
        # margins, y padded +-2D.
        D_view = float(np.atleast_1d(env.fs.windTurbines.diameter())[0])
        a = np.linspace(-200 + min(env.x_pos), 300 + max(env.x_pos), 200)
        b = np.linspace(min(env.y_pos) - 2 * D_view,
                        max(env.y_pos) + 2 * D_view, 200)
        self.view = XYView(z=70, x=a, y=b, ax=self.ax1, adaptive=False)

        uvw = env.fs.get_windspeed(self.view, include_wakes=True, xarray=True)
        # Fixed color scale at creation (colorbar made once, never rescaled ->
        # no flicker); per frame only the mesh array is swapped.
        self.mesh = self.ax1.pcolormesh(
            uvw.x.values, uvw.y.values, np.asarray(uvw[0].T),
            shading="nearest", vmin=3, vmax=env.ws + 2,
        )
        self.fig.colorbar(self.mesh, ax=self.ax1).set_label("Wind speed [m/s]")

        # Rotor ellipses + index labels are drawn once. For derate-only
        # agents they stay static (fixed wd=270, yaw never moves); for a
        # yaw+derate agent _refresh_plots re-angles each ellipse from the
        # latest yaw sample so the steering is visible in the flow view.
        wt = env.fs.windTurbines
        x_turb, y_turb = wt.positions_xyz[:2]
        yaw, tilt = wt.yaw_tilt()
        x_t, y_t, D_t = [np.asarray(v) for v in [x_turb, y_turb, wt.diameter()]]
        R = D_t / 2
        yaw = np.broadcast_to(np.asarray(yaw), x_t.shape)
        tilt = np.broadcast_to(np.asarray(tilt), x_t.shape)
        wd = float(np.atleast_1d(env.fs.wind_direction)[0])
        self.wd = wd

        self.rotor_ellipses = []
        self.derate_texts = []
        for i, (x_, y_, r, yaw_, tilt_) in enumerate(zip(x_t, y_t, R, yaw, tilt)):
            ellipse = Ellipse(
                (x_, y_), 2 * r * np.sin(np.deg2rad(tilt_)), 2 * r,
                angle=90 - wd + yaw_, ec="k", fc="None", lw=2.5,
            )
            self.ax1.add_artist(ellipse)
            self.rotor_ellipses.append(ellipse)
            self.ax1.plot(x_, y_, ".", color="k")
            label = self.ax1.annotate(i + 1, (x_ - r, y_ + r),
                                      fontsize=10, color="white")
            label.set_path_effects([
                path_effects.Stroke(linewidth=2, foreground="black"),
                path_effects.Normal(),
            ])
            # Live per-turbine derate readout; only set_text() per frame.
            dtext = self.ax1.annotate("0.00", (x_ - r, y_ - r),
                                      fontsize=10, color="white")
            dtext.set_path_effects([
                path_effects.Stroke(linewidth=2, foreground="black"),
                path_effects.Normal(),
            ])
            self.derate_texts.append(dtext)

        self.title = self.ax1.set_title("")
        self.ax1.set_aspect("equal")
        self.ax1.set_xlabel("x [m]")
        self.ax1.set_ylabel("y [m]")

        # ---- Time-series panels: line handles + fixed y-lims, once ----
        (self.line_farm,) = self.ax2.plot([], [], color="orange", label="farm")
        (self.line_ref,) = self.ax2.plot([], [], "k--", label="reference")
        self.ax2.set_ylim(0.0, 1.3 * self.greedy)
        self.ax2.set_title("Farm power [W]")
        self.ax2.legend(loc="upper right", fontsize=8)

        self.derate_lines = [
            self.ax3.plot([], [], label=f"T{i + 1}")[0] for i in range(n_turb)
        ]
        self.ax3.set_ylim(env.derate_min - 0.05, env.derate_max + 0.05)
        self.ax3.set_title("Turbine derating [-]")
        self.ax3.legend(loc="upper right", fontsize=8)

        self.powT_lines = [
            self.ax4.plot([], [], label=f"T{i + 1}")[0] for i in range(n_turb)
        ]
        # Per-turbine power tops out around the freestream (rated-at-ws) power.
        rated = float(np.max(np.atleast_1d(env.rated_power)))
        self.ax4.set_ylim(0.0, 1.2 * rated)
        self.ax4.set_title("Turbine power [W]")
        self.ax4.legend(loc="upper right", fontsize=8)

        if self.op_mode:
            self.pitch_lines = [
                self.ax5.plot([], [], label=f"T{i + 1}")[0] for i in range(n_turb)
            ]
            # Default ranges (same as the eval figures); _refresh_plots grows
            # them whenever the data leaves them — except the pitch max, which
            # is hard-capped at 15 deg (feathered/parked table points ~90 deg
            # would flatten the panel).
            self.pitch_ylim = [0.0, 15.0]
            self.rpm_ylim = [5.0, 8.0]
            self.ax5.set_ylim(*self.pitch_ylim)
            self.ax5.set_title("Blade pitch [deg]")
            self.ax5.legend(loc="upper right", fontsize=8)

            self.rpm_lines = [
                self.ax6.plot([], [], label=f"T{i + 1}")[0] for i in range(n_turb)
            ]
            self.ax6.set_ylim(*self.rpm_ylim)
            self.ax6.set_title("Rotor speed [RPM]")
            self.ax6.legend(loc="upper right", fontsize=8)

        if self.yaw_active:
            self.yaw_lines = [
                self.ax7.plot([], [], label=f"T{i + 1}")[0] for i in range(n_turb)
            ]
            # Grown on the fly like the RPM range (agent_eval uses the same
            # +-5 deg floor and 1.2x running-extreme growth).
            self.yaw_ylim = [-5.0, 5.0]
            self.ax7.set_ylim(*self.yaw_ylim)
            self.ax7.set_title("Turbine yaws [deg]")
            self.ax7.legend(loc="upper right", fontsize=8)

        # Time axis label + tick labels live on the bottom panel of each column.
        for ax in self.bottom_axes:
            ax.set_xlabel("Time [s]")
        for ax in self.ts_axes:
            ax.grid()
            if ax not in self.bottom_axes:
                ax.tick_params(labelbottom=False)

        # ---- Widgets ----
        # Left edge leaves room for the label Slider draws OUTSIDE the axes.
        slider_ax = self.fig.add_axes([0.17, 0.045, 0.41, 0.03])
        self.slider = Slider(slider_ax, "Target [% greedy]", 30.0, 120.0,
                             valinit=100.0 * setpoint_init)
        self.slider.on_changed(self._on_slider)

        btn_ax = self.fig.add_axes([0.66, 0.04, 0.09, 0.05])
        self.button = Button(btn_ax, "Pause")
        self.button.on_clicked(self._on_pause)

        # Seed the panels with the post-reset state so the window never opens
        # empty (reset's warm-up already pushed the live setpoint into obs).
        self._append_sample(
            t=float(env.fs.time),
            powers=np.asarray(env.fs.windTurbines.power(), dtype=float),
            ref=float(env.power_setpoint),
            derate=np.asarray(env.current_derate, dtype=float),
            # reset's warm-up already ran _take_measurements, so these exist.
            pitch=(np.asarray(env.current_pitch, dtype=float)
                   if self.op_mode else None),
            rpm=(np.asarray(env.current_rpm, dtype=float)
                 if self.op_mode else None),
            yaw=(np.asarray(env.fs.windTurbines.yaw, dtype=float)
                 if self.yaw_active else None),
        )
        self._refresh_plots()

    # ---- Widget callbacks ----

    def _on_slider(self, val):
        # Next env.step's _push_tracking propagates this into the reward
        # deque and the agent's tracking observations automatically.
        self.holder.value = (float(val) / 100.0) * self.greedy

    def _on_pause(self, _event):
        # Early-return in tick(); the timer keeps running so the slider and
        # button stay responsive while the sim is frozen.
        self.paused = not self.paused
        self.button.label.set_text("Play" if self.paused else "Pause")
        self.fig.canvas.draw_idle()

    # ---- Per-frame work ----

    def tick(self):
        if self.paused or self.done:
            return

        obs_t = torch.Tensor(np.asarray(self.obs)[None])
        action, _, _ = self.model.get_action(
            obs_t, deterministic=self.deterministic
        )
        action = action.detach().cpu().numpy().flatten()

        self.obs, _reward, _terminated, truncated, info = self.env.step(action)
        self.last_info = info

        if truncated:
            # Truncation triggers the env's resource cleanup (flow sim freed),
            # so the episode cannot continue past it. FarmEval's sandbox
            # time_max is ~100k steps, so this takes ~a day of dragging.
            self.done = True
            if self.timer is not None:
                self.timer.stop()
            self.title.set_text("Episode hit time_max - simulation stopped")
            self.fig.canvas.draw_idle()
            return

        self.step_count += 1
        # info arrays are at sim resolution, (sim_steps_per_env_step, n_turb);
        # take the end-of-step sample.
        self._append_sample(
            t=float(info["time_array"][-1]),
            powers=np.asarray(info["powers"][-1], dtype=float),
            ref=float(info["Power reference"]),
            derate=np.asarray(self.env.current_derate, dtype=float),
            pitch=(np.asarray(info["pitches"][-1], dtype=float)
                   if self.op_mode else None),
            rpm=(np.asarray(info["rpms"][-1], dtype=float)
                 if self.op_mode else None),
            yaw=(np.asarray(info["yaws"][-1], dtype=float)
                 if self.yaw_active else None),
        )

        if self.step_count % self.flow_every == 0:
            uvw = self.env.fs.get_windspeed(
                self.view, include_wakes=True, xarray=True
            )
            self.mesh.set_array(np.asarray(uvw[0].T))

        self._refresh_plots()
        self.fig.canvas.draw_idle()

    def _append_sample(self, *, t, powers, ref, derate, pitch=None, rpm=None,
                       yaw=None):
        self.t_deq.append(t)
        self.pfarm_deq.append(powers.sum())
        self.ref_deq.append(ref)
        self.derate_deq.append(derate.copy())
        self.powT_deq.append(powers.copy())
        if self.op_mode:
            self.pitch_deq.append(pitch.copy())
            self.rpm_deq.append(rpm.copy())
        if self.yaw_active:
            self.yaw_deq.append(yaw.copy())

    def _refresh_plots(self):
        t = np.asarray(self.t_deq)
        self.line_farm.set_data(t, np.asarray(self.pfarm_deq))
        self.line_ref.set_data(t, np.asarray(self.ref_deq))
        derate = np.asarray(self.derate_deq)
        powT = np.asarray(self.powT_deq)
        for i, line in enumerate(self.derate_lines):
            line.set_data(t, derate[:, i])
        for i, line in enumerate(self.powT_lines):
            line.set_data(t, powT[:, i])
        if self.op_mode:
            pitch = np.asarray(self.pitch_deq)
            rpm = np.asarray(self.rpm_deq)
            for i, line in enumerate(self.pitch_lines):
                line.set_data(t, pitch[:, i])
            for i, line in enumerate(self.rpm_lines):
                line.set_data(t, rpm[:, i])
            # Grow the default ranges if the data leaves them (never shrink,
            # so the view stays steady while dragging the slider).
            self.pitch_ylim[0] = min(self.pitch_ylim[0], float(pitch.min()) - 0.5)
            self.rpm_ylim[0] = min(self.rpm_ylim[0], float(rpm.min()) - 0.2)
            self.rpm_ylim[1] = max(self.rpm_ylim[1], float(rpm.max()) + 0.2)
            self.ax5.set_ylim(*self.pitch_ylim)
            self.ax6.set_ylim(*self.rpm_ylim)
        if self.yaw_active:
            yaws = np.asarray(self.yaw_deq)
            for i, line in enumerate(self.yaw_lines):
                line.set_data(t, yaws[:, i])
            # Same growth rule as agent_eval: 1.2x the running data extreme,
            # never shrinking below the +-5 deg floor.
            self.yaw_ylim[0] = min(self.yaw_ylim[0], 1.2 * float(yaws.min()))
            self.yaw_ylim[1] = max(self.yaw_ylim[1], 1.2 * float(yaws.max()))
            self.ax7.set_ylim(*self.yaw_ylim)
            # Re-angle the rotor ellipses from the latest yaw sample so the
            # flow view shows the steering (agent_eval's 90 - wd + yaw).
            for ellipse, yaw_i in zip(self.rotor_ellipses, yaws[-1]):
                ellipse.angle = 90.0 - self.wd + float(yaw_i)

        x1 = t[-1] if len(t) > 1 else t[0] + 1.0
        for ax in self.ts_axes:
            ax.set_xlim(t[0], max(x1, t[0] + 1.0))

        for text, d in zip(self.derate_texts, self.env.current_derate):
            text.set_text(f"{d:.2f}")
        self.title.set_text(
            f"Flow field at {self.env.fs.time:.0f} s - "
            f"target {self.slider.val:.0f}% of greedy "
            f"({self.holder.value / 1e6:.2f} MW)"
        )

    def run(self):
        self.timer = self.fig.canvas.new_timer(interval=self.interval_ms)
        self.timer.add_callback(self.tick)
        self.timer.start()
        plt.show()  # blocks; Tk event loop drives tick() + widgets


def run_smoke(app, holder, greedy, n_steps, out_png):
    """Headless self-test: tick, move the slider programmatically (set_val
    fires on_changed on Agg too), tick again, and assert the new setpoint
    reached (a) info["Power reference"], (b) the reward's reference deque and
    (c) the agent-facing tracking observation. Saves one frame for eyeballing.
    """
    n1 = n_steps // 2
    n2 = n_steps - n1
    for _ in range(n1):
        app.tick()
    obs_before = np.asarray(app.obs).copy()
    setpoint_before = app.env.farm_measurements.get_tracking(scaled=False)[0]

    app.slider.set_val(110.0)
    assert abs(holder.value - 1.10 * greedy) < 1e-6 * greedy, (
        f"slider -> holder failed: {holder.value} vs {1.10 * greedy}"
    )

    for _ in range(n2):
        app.tick()

    ref = float(app.last_info["Power reference"])
    assert abs(ref - 1.10 * greedy) < 1e-6 * greedy, (
        f'info["Power reference"]={ref} did not follow the slider '
        f"(want {1.10 * greedy})"
    )
    reward_ref = float(app.env.power_tracking.ref_deque[-1])
    assert abs(reward_ref - 1.10 * greedy) < 1e-6 * greedy, (
        f"reward ref_deque last={reward_ref} did not follow the slider"
    )
    setpoint_after = app.env.farm_measurements.get_tracking(scaled=False)[0]
    assert abs(setpoint_after - 1.10 * greedy) < 1e-6 * greedy, (
        f"tracking obs setpoint={setpoint_after} did not follow the slider"
    )
    assert not np.array_equal(obs_before, np.asarray(app.obs)), (
        "flat obs identical before/after the setpoint change - "
        "the agent never saw it"
    )
    assert setpoint_before != setpoint_after, (
        "tracking obs setpoint unchanged across the slider move"
    )
    expected_len = min(n_steps + 1, app.t_deq.maxlen)  # +1 = reset seed sample
    assert len(app.t_deq) == expected_len, (
        f"deque grew to {len(app.t_deq)}, expected {expected_len}"
    )

    if app.yaw_active:
        # Yaw plumbing: shapes/finiteness at sim resolution, and actual
        # steering in the 110% boost regime just run (the trained agent ramps
        # a few deg/env-step toward the wake-steering optimum there).
        yaw_shape = (app.env.sim_steps_per_env_step, app.env.n_turb)
        yaws_info = np.asarray(app.last_info["yaws"])
        assert yaws_info.shape == yaw_shape, (
            f'info["yaws"] shape {yaws_info.shape}, want {yaw_shape}'
        )
        assert np.all(np.isfinite(yaws_info)), "non-finite yaw in info"
        final_yaw = np.asarray(app.yaw_deq[-1], dtype=float)
        assert float(np.max(np.abs(final_yaw))) > 1.0, (
            f"no steering at a 110% target: max |yaw| = "
            f"{np.max(np.abs(final_yaw)):.3f} deg"
        )
        assert len(app.yaw_deq) == len(app.t_deq), (
            "yaw deque out of sync with the time deque"
        )
        print(f"[Smoke] Yaw OK: final per-turbine yaws "
              f"{np.round(final_yaw, 2)} deg at a 110% target.")

    if app.op_mode:
        # Operating-point plumbing: shapes/finiteness at sim resolution, and a
        # directional response — dropping the target forces deeper derating,
        # which at ws=10 must pitch the blades up and slow the rotor.
        op_shape = (app.env.sim_steps_per_env_step, app.env.n_turb)
        pitches = np.asarray(app.last_info["pitches"])
        rpms = np.asarray(app.last_info["rpms"])
        assert pitches.shape == op_shape and rpms.shape == op_shape, (
            f"pitches/rpms shapes {pitches.shape}/{rpms.shape}, want {op_shape}"
        )
        assert np.all(np.isfinite(pitches)) and np.all(np.isfinite(rpms)), (
            "non-finite pitch/rpm in info"
        )
        pitch_before = float(np.mean(app.pitch_deq[-1]))
        rpm_before = float(np.mean(app.rpm_deq[-1]))

        app.slider.set_val(40.0)  # deep derate vs the 110% target above
        for _ in range(max(n2, 10)):
            app.tick()

        pitch_after = float(np.mean(app.pitch_deq[-1]))
        rpm_after = float(np.mean(app.rpm_deq[-1]))
        assert pitch_after > pitch_before, (
            f"mean pitch did not rise on a deep derate "
            f"({pitch_before:.2f} -> {pitch_after:.2f} deg)"
        )
        assert rpm_after < rpm_before, (
            f"mean rpm did not drop on a deep derate "
            f"({rpm_before:.2f} -> {rpm_after:.2f} RPM)"
        )
        assert len(app.pitch_deq) == len(app.t_deq) == len(app.rpm_deq), (
            "pitch/rpm deques out of sync with the time deque"
        )
        print(f"[Smoke] Operating point OK: pitch {pitch_before:.2f} -> "
              f"{pitch_after:.2f} deg, rpm {rpm_before:.2f} -> "
              f"{rpm_after:.2f} on a 110% -> 40% target drop.")

    app.fig.savefig(out_png, dpi=100)
    print(f"[Smoke] OK: {n_steps} steps, reference/reward/obs all follow the "
          f"slider. Frame saved to {out_png}")


def main():
    parser = argparse.ArgumentParser(
        description="Interactive tracking eval: live target-power slider over "
                    "the track3 env (derate-only or yaw+derate, from the "
                    "checkpoint args).")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to runs/<run>/checkpoints/step_<N>.pt")
    parser.add_argument("--turbbox-path", default="./boxes/")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stochastic", action="store_true",
                        help="Sample actions instead of the deterministic mean.")
    parser.add_argument("--interval-ms", type=int, default=30,
                        help="GUI timer interval between env steps.")
    parser.add_argument("--flow-every", type=int, default=1,
                        help="Refresh the flow-field mesh every N steps "
                             "(cheap throttle if the grid query bottlenecks).")
    parser.add_argument("--window", type=int, default=120,
                        help="Scrolling time-series window (env steps).")
    parser.add_argument("--setpoint-init", type=float, default=0.8,
                        help="Initial target as a fraction of greedy.")
    parser.add_argument("--smoke", type=int, default=0, metavar="N",
                        help="Headless self-test: run N steps on Agg with a "
                             "programmatic slider change, assert propagation, "
                             "save one frame, exit.")
    parser.add_argument("--smoke-out", default="interactive_smoke.png",
                        help="Frame path for --smoke.")
    cli = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    device = torch.device("cpu")
    deterministic = not cli.stochastic

    # ---- Checkpoint, env, agent (same plumbing as eval_tracking_figs) ----
    checkpoint, args = load_checkpoint(cli.checkpoint, device)
    # The actor's first-layer in_features pins the per-turbine sensor
    # generation the checkpoint was trained on (see _MES_GENERATIONS there).
    ckpt_obs_width = checkpoint["actor_state_dict"]["obs_encoder.0.weight"].shape[1]
    base_env, wrapped_env, rotor_diameter, greedy = build_track3_env(
        args, cli.turbbox_path, obs_width=ckpt_obs_width
    )
    actor, agent = build_agent(args, wrapped_env, rotor_diameter, device)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()

    # ---- Install the live setpoint manager, then reset (order matters:
    # reset's warm-up pushes must already hit the live manager, and
    # set_wind_vals must precede reset -- mirrors agent_eval's ordering) ----
    holder = SetpointHolder(cli.setpoint_init * greedy)
    base_env.power_tracking = LiveSetpointManager(holder)
    base_env.set_wind_vals(ws=TRACK3_WS, ti=TRACK3_TI, wd=TRACK3_WD)
    mode = ("yaw+derate (boost-capable)"
            if (getattr(base_env, "yaw_action", False)
                and getattr(base_env, "derate_action", False))
            else "derate-only")
    print(f"[Interactive] Mode: {mode} (config: {args.get('config')})")
    print("[Interactive] Resetting env (burn-in)...")
    obs, _info = base_env.reset(seed=cli.seed)

    # ---- Static per-inference arrays for the eval adapter ----
    n_turb = base_env.n_turb
    raw_positions = (
        np.stack([base_env.x_pos, base_env.y_pos]).T[None].astype(np.float32)
    )
    wind_dirs = np.array([TRACK3_WD], dtype=np.float32)
    masks = np.zeros((1, n_turb), dtype=bool)
    transform = build_obs_transform(base_env, wrapped_env)
    # Profile-encoded checkpoints (FourierProfileEncoder) need the layout's
    # receptivity/influence profiles at inference; without them the agent's
    # act(None, ...) falls back to env.get_attr and crashes on the first tick.
    receptivity, influence = compute_track3_profiles(
        args, base_env.x_pos, base_env.y_pos, base_env.turbine
    )
    if receptivity is not None:
        receptivity = np.asarray(receptivity, dtype=np.float32)[None]  # (1, n_turb, n_dir)
        influence = np.asarray(influence, dtype=np.float32)[None]
    model = _TransformerEvalModel(agent, transform, wind_dirs, raw_positions,
                                  masks, receptivity=receptivity,
                                  influence=influence)

    app = InteractiveApp(
        base_env, model, greedy, holder, obs,
        deterministic=deterministic,
        interval_ms=cli.interval_ms,
        flow_every=cli.flow_every,
        window=cli.window,
        setpoint_init=cli.setpoint_init,
    )

    if cli.smoke > 0:
        run_smoke(app, holder, greedy, cli.smoke, cli.smoke_out)
        return

    print(f"[Interactive] GREEDY = {greedy / 1e6:.3f} MW; slider 30-120% "
          f"of greedy, start {100 * cli.setpoint_init:.0f}%. Close the window "
          "to exit.")
    app.run()


if __name__ == "__main__":
    main()
