"""
Calibrate the frozen empirical ws-CDF for --obs_encoding cdf (change_wd_4).

Rebuilds the change_wd_4 ARM-0 training stack — hard_2 config, grid-DR layout
pool n in [6,16], pywake backend, Random TI, history 3 — runs random actions
for ~5k env steps across ~20 layouts, recovers PHYSICAL per-turbine ws by
inverting the obs affine (the same inversion ObsEncodingWrapper does), pools
over turbines and history slots, and writes the quantile knots to
helpers/ws_cdf_knots.json.

The JSON is COMMITTED to git: fixed seed -> deterministic knots, so every arm,
seed and node warps ws through the IDENTICAL frozen CDF (a per-run calibration
would make the observation function itself run-dependent). Local CPU run is
fine — the change_wd sweeps were all verified locally on the pywake backend.

Usage (repo root):
    uv run python TransformerSac/calibrate_ws_cdf.py
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np

# Script-dir import context, same convention as transformer_sac_windfarm.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from helpers.env_configs import make_env_config
from helpers.layout_gen import generate_layout_pool
from helpers.multi_layout_env import MultiLayoutEnv, LayoutConfig

from WindGym import WindFarmEnv
from WindGym.wrappers import PerTurbineObservationWrapper

SEED = 0
HIST = 3          # arm-0 history_length
N_KNOTS = 33
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "helpers", "ws_cdf_knots.json")


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
        ).decode().strip()
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n_layouts", type=int, default=20)
    parser.add_argument("--episode_steps", type=int, default=250)
    parser.add_argument("--total_steps", type=int, default=5000)
    parser.add_argument("--out", type=str, default=OUT_PATH)
    args = parser.parse_args()

    np.random.seed(SEED)

    # --- Arm-0 env config (mirror transformer_sac_windfarm.py main()) ---
    config = make_env_config("hard_2")
    config["ActionMethod"] = "yaw"
    for mes, prefix in {"ws_mes": "ws", "wd_mes": "wd",
                        "yaw_mes": "yaw", "power_mes": "power"}.items():
        config[mes][f"{prefix}_history_N"] = HIST
        config[mes][f"{prefix}_history_length"] = HIST

    from py_wake.examples.data.dtu10mw import DTU10MW
    wind_turbine = DTU10MW()

    base_env_kwargs = {
        "turbine": wind_turbine,
        "n_passthrough": 20,
        "TurbBox": "./boxes/",
        "config": config,
        "turbtype": "Random",
        "backend": "pywake",
        "dt_sim": 10,
        "dt_env": 10,
        "yaw_step_sim": 5,
    }

    def env_factory(x_pos, y_pos):
        env = WindFarmEnv(x_pos=x_pos, y_pos=y_pos, reset_init=False,
                          **base_env_kwargs)
        env.action_space.seed(SEED)
        return env

    # --- Arm-0 layout distribution: grid-DR pool, counts (nx*ny) in [6,16] ---
    print(f"Generating {args.n_layouts} grid-DR layouts (seed={SEED})...")
    pool = generate_layout_pool(
        pool_size=args.n_layouts,
        n_lo=6, n_hi=16,
        D=wind_turbine.diameter(),
        seed=SEED,
        min_dist_D=3.0,
        screen_headroom=True,
        min_involved_frac=0.5,
        generator="grid",
    )
    layouts = [LayoutConfig(name=name, x_pos=x, y_pos=y) for name, x, y in pool]

    env = MultiLayoutEnv(
        layouts=layouts,
        env_factory=env_factory,
        per_turbine_wrapper=PerTurbineObservationWrapper,
        seed=SEED,
        shuffle=False,
        max_episode_steps=args.episode_steps,
    )

    # --- ws columns + scaling range, read the same way ObsEncodingWrapper does ---
    base = env._get_base_env()
    turb_mes = base.farm_measurements.turb_mes[0]
    offset = turb_mes.n_probes if getattr(turb_mes, "n_probes", 0) else 0
    ws_mes = turb_mes.ws
    n_ws = (1 if ws_mes.current else 0) + (ws_mes.history_N if ws_mes.rolling_mean else 0)
    ws_min, ws_max = float(turb_mes.ws_min), float(turb_mes.ws_max)
    print(f"ws columns {offset}..{offset + n_ws - 1}, scaled from [{ws_min}, {ws_max}]")

    # --- Roll random actions, harvesting physical ws ---
    samples = []
    obs, _ = env.reset(seed=SEED)
    n_episodes = 0
    for step in range(args.total_steps):
        real = ~env.attention_mask  # exclude 0.0-pad rows
        ws_scaled = obs[real, offset:offset + n_ws]
        samples.append((ws_scaled + 1.0) / 2.0 * (ws_max - ws_min) + ws_min)

        obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            obs, _ = env.reset()
            n_episodes += 1
        if (step + 1) % 500 == 0:
            print(f"  step {step + 1}/{args.total_steps} "
                  f"({n_episodes + 1} layouts visited, {sum(s.size for s in samples)} samples)")
    env.close()

    ws = np.concatenate([s.ravel() for s in samples])
    print(f"\nCollected {ws.size} physical ws samples: "
          f"min={ws.min():.2f}, p5={np.percentile(ws, 5):.2f}, "
          f"median={np.median(ws):.2f}, p95={np.percentile(ws, 95):.2f}, "
          f"max={ws.max():.2f} m/s")

    probs = np.linspace(0.0, 1.0, N_KNOTS)
    knots = np.quantile(ws, probs)
    # np.interp needs non-decreasing xp; ties (possible in saturated regions)
    # are collapsed keeping the LAST prob, so the CDF jumps once at the tie.
    knots_u, idx = np.unique(knots, return_index=True)
    # for duplicated knot values keep the highest prob among the ties
    probs_u = np.array([probs[knots == k].max() for k in knots_u])
    if knots_u.size < N_KNOTS:
        print(f"NOTE: {N_KNOTS - knots_u.size} duplicate knots collapsed")

    payload = {
        "probs": probs_u.tolist(),
        "knots": knots_u.tolist(),
        "meta": {
            "script": "TransformerSac/calibrate_ws_cdf.py",
            "git_sha": _git_sha(),
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "seed": SEED,
            "config": "hard_2",
            "backend": "pywake",
            "history_length": HIST,
            "dr": "grid n in [6,16]",
            "n_layouts": args.n_layouts,
            "total_steps": args.total_steps,
            "episode_steps": args.episode_steps,
            "n_samples": int(ws.size),
            "ws_scaling_inverted_from": [ws_min, ws_max],
        },
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {knots_u.size} knots to {args.out}")


if __name__ == "__main__":
    main()
