"""Pre-flight for the --obs_agg feature scales (LES-3x3 Stage 4).

Builds ONE les_3x3 dynamiks env (les_recipe, ws 9.5, hold_ramp_270_235_short,
dt_sim 5 / dt_env 10, max_turb_move 12, wd_est on -- the Stage-3 recipe),
steps it with a slew-then-hold yaw policy (random +-25 deg targets every 50
steps, proportional tracking + dither), and for EVERY mode in AGG_MODES (and
L in {15, 30, 60} for mean_std) reduces the same measurement deques exactly
like ObsAggWrapper does. Prints per-mode per-feature p1/p50/p99 in SCALED
units and writes wd_estimation/figs/obs_agg_probe.png. The pass criterion
(plan §Verification 1): every feature has p1..p99 inside (-1, 1) and a spread
p99-p1 > 0.05 -- no saturated and no dead feature. Adjust helpers/obs_agg.py
DISPERSION once if not, then freeze.

Run locally (CPU, ~minutes):
    cd TransformerSac && ../.venv/bin/python obs_agg_probe.py [--steps 200]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from helpers.obs_agg import (  # noqa: E402
    AGG_MODES, QUANTITIES, reduce_quantity, scale_features,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PNG = os.path.join(REPO, "wd_estimation", "figs", "obs_agg_probe.png")

# (label, mode, L) -- the 11 trained arms + raw3 seam mode
PROBE_ARMS = [
    ("raw3", "raw3", 30),
    ("latest", "latest", 30),
    ("mean_std", "mean_std", 30),
    ("mean_std_L15", "mean_std", 15),
    ("mean_std_L60", "mean_std", 60),
    ("ema", "ema", 30),
    ("trend", "trend", 30),
    ("minmax", "minmax", 30),
    ("quantiles", "quantiles", 30),
    ("raw15", "raw15", 30),
    ("spectral", "spectral", 30),
    ("spatial_rel", "spatial_rel", 30),
]
L_MAX = max(L for _, _, L in PROBE_ARMS)


def build_env(ws=9.5, wd_name="hold_ramp_270_235_short", seed=0):
    from py_wake.examples.data.dtu10mw import DTU10MW
    from WindGym import WindFarmEnv
    from helpers.env_configs import make_env_config, make_eval_wind_config
    from helpers.layouts import get_layout_positions
    from helpers.wd_functions import get_wd_function

    turbine = DTU10MW()
    x_pos, y_pos = get_layout_positions("les_3x3", turbine)
    wd_fn = get_wd_function(wd_name)
    config = make_env_config("les_recipe")
    for p in QUANTITIES:
        config[f"{p}_mes"][f"{p}_history_N"] = 3
        config[f"{p}_mes"][f"{p}_history_length"] = L_MAX
    config = make_eval_wind_config(config, float(wd_fn(0.0)), float(ws))
    env = WindFarmEnv(
        x_pos=x_pos, y_pos=y_pos, turbine=turbine, config=config,
        backend="dynamiks", turbtype="MannGenerate", TurbBox="./boxes/",
        dt_sim=5, dt_env=10, yaw_step_sim=5, n_passthrough=5,
        reset_init=False, wd_function=wd_fn, max_turb_move=12.0,
        wd_est_tau=15.0, wd_est_consensus="front",
    )
    env.action_space.seed(seed)
    return env


def snapshot_buffers(env):
    fm = env.farm_measurements
    out = {}
    for q in QUANTITIES:
        cols = [np.asarray(list(getattr(tm, q).measurements), dtype=float)
                for tm in fm.turb_mes]
        m = min(len(c) for c in cols)
        out[q] = np.stack([c[-m:] for c in cols], axis=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--png", default=DEFAULT_PNG)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--npz", default=None, help="dump scaled traces (label/q -> (T,n,K))")
    args = ap.parse_args()

    env = build_env(seed=args.seed)
    tm = env.farm_measurements.turb_mes[0]
    ranges = {"ws": (tm.ws_min, tm.ws_max), "wd": (tm.wd_min, tm.wd_max),
              "yaw": (tm.yaw_min, tm.yaw_max), "power": (0.0, tm.power_max)}
    front = int(np.argmin(env.x_pos))
    dt_env = 10.0
    print(f"env: n_turb={env.n_turb} front={front} ranges={ranges} "
          f"deque L_MAX={L_MAX} steps_on_reset={env.steps_on_reset}")

    rng = np.random.default_rng(args.seed)
    obs, _ = env.reset(seed=args.seed)
    n_turb = env.n_turb
    act = np.zeros(n_turb)
    # scaled feature traces: label -> q -> list of (n_turb, K)
    traces = {lab: {q: [] for q in QUANTITIES} for lab, _, _ in PROBE_ARMS}
    t0 = time.time()
    for step in range(args.steps):
        bufs = snapshot_buffers(env)
        for lab, mode, L in PROBE_ARMS:
            kinds = AGG_MODES[mode].kinds
            for q in QUANTITIES:
                feats = reduce_quantity(mode, q, bufs[q][-L:], L=L,
                                        dt_env=dt_env, front=front)
                lo, hi = ranges[q]
                traces[lab][q].append(scale_features(feats, kinds, q, lo, hi))
        # yaw policy: per-turbine random targets in +-25 deg, re-drawn every
        # ~50 steps, tracked with a proportional slew + small dither. This is
        # what trained policies do (slew, then hold) -- a pure random walk in
        # action units saturates the +-45 band and inflates every dispersion.
        if step % 50 == 0:
            target = rng.uniform(-25.0, 25.0, size=n_turb)
        yaw_now = bufs["yaw"][-1]
        act = np.clip((target - yaw_now) / 10.0 + rng.normal(0.0, 0.1, size=n_turb),
                      -1.0, 1.0)
        obs, r, term, trunc, info = env.step(act.astype(np.float32))
        if trunc or term:
            print(f"episode ended at step {step}; resetting")
            obs, _ = env.reset()
        if step % 25 == 0:
            print(f"step {step}/{args.steps} wd_est={env.wd_est:.1f} "
                  f"({time.time() - t0:.0f}s)")
    env.close()

    if args.npz:
        np.savez_compressed(args.npz, **{f"{lab}/{q}": np.stack(v, axis=0)
                                         for lab in traces for q, v in traces[lab].items()})
        print(f"wrote {args.npz}")

    # ---- report ---------------------------------------------------------
    rows = []   # (label, q, k, kind, p1, p50, p99, ok)
    all_ok = True
    for lab, mode, L in PROBE_ARMS:
        kinds = AGG_MODES[mode].kinds
        print(f"\n== {lab} (mode={mode}, L={L}, K={len(kinds)}: {AGG_MODES[mode].doc})")
        for q in QUANTITIES:
            X = np.stack(traces[lab][q], axis=0)         # (T, n_turb, K)
            for k, kind in enumerate(kinds):
                v = X[:, :, k].ravel()
                p1, p50, p99 = np.percentile(v, [1, 50, 99])
                spread = p99 - p1
                ok = (p1 > -1.0) and (p99 < 1.0) and (spread > 0.05)
                # `t`-type level features of static-ish quantities may
                # legitimately move little; the criterion is still applied
                # (the plan asks for it) and reported per feature.
                all_ok &= bool(ok)
                flag = "" if ok else "   <-- CHECK"
                print(f"  {q:5s}[{k:2d}] {kind:5s} p1={p1:+.3f} p50={p50:+.3f} "
                      f"p99={p99:+.3f} spread={spread:.3f}{flag}")
                rows.append((lab, q, k, kind, p1, p50, p99, ok))
    print("\nALL FEATURES OK" if all_ok else "\nSOME FEATURES FLAGGED (see <-- CHECK)")

    # ---- figure ---------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(PROBE_ARMS)
    fig, axes = plt.subplots(n, 1, figsize=(11, 1.0 + 1.35 * n), sharex=True)
    for ax, (lab, mode, L) in zip(axes, PROBE_ARMS):
        sub = [r for r in rows if r[0] == lab]
        x = np.arange(len(sub))
        for i, (_, q, k, kind, p1, p50, p99, ok) in enumerate(sub):
            col = "#4C72B0" if ok else "#C44E52"
            ax.plot([i, i], [p1, p99], color=col, lw=2.0, solid_capstyle="round")
            ax.plot(i, p50, "o", color=col, ms=4)
        ax.axhline(1, color="0.75", lw=0.8); ax.axhline(-1, color="0.75", lw=0.8)
        ax.axhline(0, color="0.9", lw=0.6)
        ax.set_ylim(-1.15, 1.15)
        ax.set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{q}{k}" for _, q, k, *_ in sub], fontsize=7,
                           rotation=90)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(False)
    axes[0].set_title("obs_agg probe: p1–p50–p99 of every scaled feature "
                      f"({args.steps} steps, les_3x3, ws 9.5, hold_ramp_270_235_short; "
                      "red = saturated or dead)", fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.png), exist_ok=True)
    fig.savefig(args.png, dpi=130)
    print(f"wrote {args.png}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
