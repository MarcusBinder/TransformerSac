"""Stage 3: closed-loop steering sweep along a Stage-2 direction.

For each alpha (in SIGMA units -- scaled by the direction artifact's act_std)
the checkpoint's policy is rolled deterministically with a SteeringHook adding
alpha * vector at the direction's site. Episode seeds are PAIRED across
alphas (same turbulence boxes), so curves differ only through the steering.

Outputs (data/interp/<run>/steer/, figs/interp/<run>/):
    steer_{site}_s{seed}.npz            per-step traces + per-alpha aggregates
    steer_alpha_{site}.png              derate / power-ratio / DEL-ratio /
                                        tanh-saturation vs alpha
    steer_plane_{site}.png              power-vs-DEL trade-off plane: steering
                                        curve, and with --limit-sweep (randlim)
                                        the trained limit-knob curve overlaid
                                        -- the headline comparison.

Saturation monitor: fraction of tokens with |tanh^{-1}-side mean| squashed
past 0.99 of the action range, per channel. Alphas whose saturation exceeds
the alpha=0 baseline by >10 pp are flagged (steering there is a clipping
artifact, not a dial) and shaded in the figures.

alpha = 0 is exact: the hook returns None, so the baseline run is the
unsteered policy bit-for-bit (checked against a hookless rollout unless
--no-check-baseline).

Usage (from TransformerSac/):
    ../.venv/bin/python interp/steer_eval.py \
        --checkpoint runs/delmax2x2_B_pen_s1/checkpoints/step_400020.pt \
        --direction data/interp/delmax2x2_B_pen_s1/directions/direction_final_norm.npz
"""

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_INTERP_DIR = os.path.dirname(os.path.abspath(__file__))
_TSAC_DIR = os.path.dirname(_INTERP_DIR)
_REPO_ROOT = os.path.dirname(_TSAC_DIR)
for _p in (_REPO_ROOT, _TSAC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from interp.common import (
    DERATE_COL, YAW_COL, base_env_of, build_actor, build_delmax_env,
    is_randlim, load_checkpoint, make_batch_fn, make_metadata, run_name_of,
)
from interp.hooks import SteeringHook

SAT_THRESH = 0.99      # |squashed mean| beyond this fraction of the range
SAT_EXCESS_PP = 0.10   # flag alphas whose saturation exceeds baseline by this


def parse_args():
    p = argparse.ArgumentParser(description="Stage 3: steering sweep.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--direction", required=True,
                   help="direction_{site}.npz from analyze_directions.py")
    p.add_argument("--alphas", type=str, default="-8,-4,-2,-1,0,1,2,4,8",
                   help="Comma alphas in SIGMA units (act_std multiples).")
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--warmup", type=int, default=20,
                   help="Leading steps excluded from aggregates (recorded).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fixed-limit", type=float, default=None,
                   help="Pin the DEL limit for steering rollouts (randlim "
                        "default: midpoint of the training range, so episodes "
                        "are comparable; ignored for plain delmax).")
    p.add_argument("--limit-sweep", type=str, default=None,
                   help="Comma DEL limits: roll alpha=0 at each to trace the "
                        "NATURAL knob curve (randlim checkpoints only).")
    p.add_argument("--outdir", type=str, default=None)
    p.add_argument("--figdir", type=str, default=None)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--no-check-baseline", action="store_true")
    return p.parse_args()


def rollout(env, actor, batch_fn, *, episodes, steps, seed, options=None,
            collect_actions=False):
    """Paired-seed deterministic rollout -> per-step traces (episodes, steps).
    The caller owns any attached SteeringHook (its alpha is live state)."""
    base = base_env_of(env)
    scale = np.asarray(actor.action_scale.cpu()).reshape(-1)
    bias = np.asarray(actor.action_bias_val.cpu()).reshape(-1)
    tr = {k: [] for k in ("mean_derate", "power_agent", "power_baseline",
                          "del_ratio", "del_margin", "del_penalty",
                          "yaw_activity", "sat_yaw", "sat_derate")}
    actions_all = []
    for ep in range(episodes):
        ep_seed = seed + ep
        torch.manual_seed(ep_seed)
        np.random.seed(ep_seed)
        obs, _ = env.reset(seed=ep_seed, options=options)
        prev_yaw = None
        ep_rows = {k: [] for k in tr}
        ep_actions = []
        for _t in range(steps):
            obs_t, pos_t, mask_t, rec_t, inf_t = batch_fn(obs=obs)
            with torch.no_grad():
                action_t, _, mean_t, _ = actor.get_action(
                    obs_t, pos_t, mask_t, deterministic=True,
                    recep_profile=rec_t, influence_profile=inf_t)
            action = action_t.squeeze(0).cpu().numpy()
            if collect_actions:
                ep_actions.append(action.copy())
            # Saturation: |pre-squash tanh| > SAT_THRESH <=> the squashed mean
            # sits beyond SAT_THRESH of the action range from its bias.
            mean = mean_t[0].cpu().numpy()[: env.n_turbines]
            squash = np.abs((mean - bias) / scale)
            obs, reward, term, trunc, info = env.step(action)
            if term or trunc:
                raise RuntimeError(f"Episode ended early at step {_t}")

            n = env.n_turbines
            # .copy(): np.asarray on an already-float64 source returns a VIEW
            # of windgym's buffer; prev_yaw must not alias live env state.
            yaw = np.asarray(base.fs.windTurbines.yaw_tilt()[0], float)[:n].copy()
            powers = np.asarray(info["powers"], dtype=float)
            if powers.ndim == 2:
                powers = powers[-1]
            bpow = np.asarray(info["baseline_powers"], dtype=float)
            if bpow.ndim == 2:
                bpow = bpow[-1]
            ep_rows["mean_derate"].append(
                float(np.mean(np.asarray(info["derate command"])[:n])))
            ep_rows["power_agent"].append(float(np.nansum(powers[:n])))
            ep_rows["power_baseline"].append(float(np.nansum(bpow[:n])))
            ep_rows["del_ratio"].append(float(info["del_ratio"]))
            ep_rows["del_margin"].append(float(info["del_margin"]))
            ep_rows["del_penalty"].append(float(info["del_penalty"]))
            ep_rows["yaw_activity"].append(
                0.0 if prev_yaw is None else float(np.abs(yaw - prev_yaw).mean()))
            prev_yaw = yaw
            ep_rows["sat_yaw"].append(float((squash[:, YAW_COL] > SAT_THRESH).mean()))
            ep_rows["sat_derate"].append(
                float((squash[:, DERATE_COL] > SAT_THRESH).mean()))
        for k in tr:
            tr[k].append(ep_rows[k])
        if collect_actions:
            actions_all.append(np.stack(ep_actions))
    traces = {k: np.asarray(v, dtype=np.float32) for k, v in tr.items()}
    if collect_actions:
        return traces, np.stack(actions_all)
    return traces


def aggregate(traces, warmup):
    """Post-warmup means per metric (scalar each) + derived power ratio."""
    agg = {}
    for k, v in traces.items():
        agg[k] = float(np.nanmean(v[:, warmup:]))
    agg["power_ratio"] = float(np.nanmean(
        traces["power_agent"][:, warmup:] / traces["power_baseline"][:, warmup:]))
    return agg


def main():
    cli = parse_args()
    device = torch.device(cli.device)
    sys.stdout.reconfigure(line_buffering=True)

    direction = np.load(cli.direction, allow_pickle=False)
    site = str(direction["site"])
    vector = direction["vector"]
    act_std = float(direction["act_std"])
    alphas = [float(a) for a in cli.alphas.split(",")]
    if 0.0 not in alphas:
        alphas.append(0.0)
    alphas = sorted(alphas)

    checkpoint, args = load_checkpoint(cli.checkpoint, device)
    randlim = is_randlim(args)
    limit_sweep = None
    if cli.limit_sweep is not None:
        if not randlim:
            raise SystemExit("--limit-sweep requires a randlim checkpoint "
                             "(plain delmax has no limit input to sweep).")
        limit_sweep = [float(x) for x in cli.limit_sweep.split(",")]
    fixed_limit = cli.fixed_limit
    if fixed_limit is None and randlim:
        fixed_limit = 0.5 * (args["del_limit_lo"] + args["del_limit_hi"])
    steer_options = ({"del_limit": float(fixed_limit)}
                     if (randlim and fixed_limit is not None) else None)

    env, _config = build_delmax_env(args, seed=cli.seed)
    actor = build_actor(args, env, checkpoint, device)
    batch_fn = make_batch_fn(args, env, device)

    run_name, step_tag = run_name_of(cli.checkpoint)
    outdir = cli.outdir or os.path.join(_TSAC_DIR, "data", "interp",
                                        run_name, "steer")
    figdir = cli.figdir or os.path.join(_TSAC_DIR, "figs", "interp", run_name)
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(figdir, exist_ok=True)

    print(f"[steer] run={run_name} site={site} act_std={act_std:.3f} "
          f"alphas(sigma)={alphas}")
    print(f"[steer] episodes={cli.episodes} steps={cli.steps} seed={cli.seed} "
          f"steer_options={steer_options} limit_sweep={limit_sweep}")

    results = {}          # alpha -> traces
    rollout_kwargs = dict(episodes=cli.episodes, steps=cli.steps,
                          seed=cli.seed, options=steer_options)
    try:
        with SteeringHook(actor, site, vector, alpha=0.0) as hook:
            for alpha in alphas:
                hook.set_alpha(alpha * act_std)
                print(f"  alpha={alpha:+g} sigma ({alpha * act_std:+.3f} raw)")
                if alpha == 0.0 and not cli.no_check_baseline:
                    traces, acts_hooked = rollout(
                        env, actor, batch_fn, collect_actions=True,
                        **rollout_kwargs)
                else:
                    traces = rollout(env, actor, batch_fn, **rollout_kwargs)
                results[alpha] = traces

        if not cli.no_check_baseline:
            _, acts_bare = rollout(env, actor, batch_fn, collect_actions=True,
                                   **rollout_kwargs)
            if not np.array_equal(acts_hooked, acts_bare):
                raise RuntimeError(
                    "alpha=0 hooked rollout differs from the hookless rollout "
                    f"(max diff {np.abs(acts_hooked - acts_bare).max():.3e}) "
                    "-- the zero-alpha no-op contract is broken.")
            print("  [check] alpha=0 hooked == hookless (bitwise)")

        knob = {}         # del_limit -> traces (natural-knob curve, alpha=0)
        if limit_sweep:
            with SteeringHook(actor, site, vector, alpha=0.0):
                for limit in limit_sweep:
                    print(f"  limit-knob rollout del_limit={limit:g}")
                    knob[limit] = rollout(
                        env, actor, batch_fn, episodes=cli.episodes,
                        steps=cli.steps, seed=cli.seed,
                        options={"del_limit": float(limit)})
    finally:
        env.close()

    # ---- aggregates + saturation flags ----
    agg = {a: aggregate(t, cli.warmup) for a, t in results.items()}
    base_sat = max(agg[0.0]["sat_yaw"], agg[0.0]["sat_derate"])
    flagged = [a for a in alphas
               if max(agg[a]["sat_yaw"], agg[a]["sat_derate"])
               > base_sat + SAT_EXCESS_PP]
    if flagged:
        print(f"  [saturation] alphas {flagged} exceed baseline tanh-"
              f"saturation by >{SAT_EXCESS_PP:.0%} -- treat as clipping, "
              "not steering.")
    for a in alphas:
        g = agg[a]
        print(f"  alpha={a:+6.2f}: derate={g['mean_derate']:.3f} "
              f"P/P_base={g['power_ratio']:.4f} del_ratio={g['del_ratio']:.3f} "
              f"sat=({g['sat_yaw']:.2f},{g['sat_derate']:.2f})")

    knob_agg = {l: aggregate(t, cli.warmup) for l, t in knob.items()}

    # ---- save ----
    save = {}
    for a, t in results.items():
        for k, v in t.items():
            save[f"steer_a{a:+g}_{k}"] = v
    for l, t in knob.items():
        for k, v in t.items():
            save[f"knob_l{l:g}_{k}"] = v
    save["alphas"] = np.asarray(alphas, dtype=np.float32)
    save["act_std"] = np.float32(act_std)
    save["flagged_alphas"] = np.asarray(flagged, dtype=np.float32)
    save["limit_sweep"] = np.asarray(limit_sweep or [], dtype=np.float32)
    save["agg_json"] = json.dumps(
        {"steer": {str(a): agg[a] for a in alphas},
         "knob": {str(l): knob_agg[l] for l in knob_agg}})
    save["metadata_json"] = make_metadata(
        cli.checkpoint, checkpoint, args, stage="steer", site=site,
        direction=os.path.abspath(cli.direction),
        direction_label=str(direction["label"]),
        alphas=alphas, episodes=cli.episodes, steps=cli.steps,
        warmup=cli.warmup, seed=cli.seed, fixed_limit=fixed_limit,
        limit_sweep=limit_sweep)
    npz_path = os.path.join(outdir, f"steer_{site}_s{cli.seed}.npz")
    np.savez_compressed(npz_path, **save)
    print(f"[steer] wrote {npz_path}")

    # ---- figure 1: metrics vs alpha ----
    metrics = [("mean_derate", "farm-mean derate cmd"),
               ("power_ratio", "power ratio P/P$_{base}$"),
               ("del_ratio", "DEL ratio"),
               ("sat_derate", f"tanh sat. frac (derate ch.)")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for ax, (key, label) in zip(axes.ravel(), metrics):
        vals = [agg[a][key] for a in alphas]
        ax.plot(alphas, vals, "o-", color="C0")
        ax.axhline(agg[0.0][key], color="gray", lw=0.8, ls="--",
                   label=r"$\alpha=0$ baseline")
        for a in flagged:
            ax.axvspan(a - 0.25, a + 0.25, color="red", alpha=0.12)
        ax.set_ylabel(label)
        ax.grid(alpha=0.4)
    for ax in axes[1]:
        ax.set_xlabel(r"steering $\alpha$ [$\sigma$ of projection]")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"{run_name} {step_tag}: steering at {site} "
                 f"(label={str(direction['label'])}; red = saturation-flagged)")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, f"steer_alpha_{site}.png"), dpi=120)
    plt.close(fig)

    # ---- figure 2: power-vs-DEL trade-off plane ----
    fig, ax = plt.subplots(figsize=(7.5, 6))
    xs = [agg[a]["del_ratio"] for a in alphas]
    ys = [agg[a]["power_ratio"] for a in alphas]
    ax.plot(xs, ys, "o-", color="C0", label=f"steering ($\\alpha$ at {site})")
    for a, x, y in zip(alphas, xs, ys):
        ax.annotate(f"{a:+g}", (x, y), fontsize=7,
                    xytext=(4, 4), textcoords="offset points",
                    color="red" if a in flagged else "black")
    ax.plot(xs[alphas.index(0.0)], ys[alphas.index(0.0)], "s", color="C0",
            ms=10, mfc="none", label=r"$\alpha=0$ (trained policy)")
    if knob_agg:
        kx = [knob_agg[l]["del_ratio"] for l in sorted(knob_agg)]
        ky = [knob_agg[l]["power_ratio"] for l in sorted(knob_agg)]
        ax.plot(kx, ky, "^--", color="C3", label="trained limit knob")
        for l in sorted(knob_agg):
            ax.annotate(f"lim {l:g}", (knob_agg[l]["del_ratio"],
                                       knob_agg[l]["power_ratio"]),
                        fontsize=7, color="C3",
                        xytext=(4, -8), textcoords="offset points")
    ax.set_xlabel("DEL ratio (agent max / baseline max)")
    ax.set_ylabel("power ratio P/P$_{base}$")
    ax.grid(alpha=0.4)
    ax.legend(fontsize=8)
    ax.set_title(f"{run_name}: power-vs-DEL plane -- steering vs natural knob")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, f"steer_plane_{site}.png"), dpi=120)
    plt.close(fig)

    print(f"[steer] figures -> {figdir}")


if __name__ == "__main__":
    main()
