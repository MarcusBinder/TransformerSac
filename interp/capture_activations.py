"""Stage 1: capture residual-stream activations during closed-loop rollouts.

Rolls the checkpoint's own policy in its rebuilt training env, buffering every
site's activations plus the exact actor inputs (for offline replay/probing)
and behavior/outcome labels per step. One npz per (del-limit) condition:

    data/interp/<run_name>/capture_lim{L}_s{seed}.npz

Arrays (S = episodes x steps rows, T = max_turbines tokens):
    act_{site}                    (S, T, 128)  float32
    obs, positions, mask          (S, T, obs_w) / (S, T, 2) / (S, T) -- the
                                  tensors AS FED to the actor (wind-relative
                                  positions, rotated profiles)
    recep, influ                  (S, T, n_dirs) when profiles are on
    action, mean_action, log_std  (S, T, A)   A=2 for yaw+derate
    derate_cmd, derate_real, yaw, power_per_turbine, loads, loads_baseline
                                  (S, T)
    power_agent, power_baseline, reward, reward_unpenalized, del_ratio,
    del_margin, del_limit, del_penalty, ws, wd, ti,
    episode_idx, episode_step, is_warmup      (S,)
    metadata_json                 () str

Budgets (Baseline_comp doubles the DWM cost): smoke = 1 ep x 20 steps;
production 2x2 = 8 ep x 300 steps; 3x3 randlim = 7 limits x 2 seeds x 200
steps. Parallelize by launching multiple PROCESSES with disjoint
--del-limits / --seed values -- there is deliberately no multiprocessing here.

After each condition an offline replay verifies determinism: actor.forward on
the saved inputs must reproduce the saved mean_action to <= 1e-6 (disable
with --no-verify).

Usage (from TransformerSac/):
    ../.venv/bin/python interp/capture_activations.py \
        --checkpoint runs/delmax2x2_B_pen_s1/checkpoints/step_400020.pt \
        --episodes 8 --steps 300 --warmup 20
"""

import argparse
import os
import sys

import numpy as np
import torch

_INTERP_DIR = os.path.dirname(os.path.abspath(__file__))
_TSAC_DIR = os.path.dirname(_INTERP_DIR)
_REPO_ROOT = os.path.dirname(_TSAC_DIR)
for _p in (_REPO_ROOT, _TSAC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from interp.common import (
    DEL_CHANNEL, DERATE_COL, base_env_of, build_actor, build_delmax_env,
    is_randlim, load_checkpoint, make_batch_fn, make_metadata, run_name_of,
)
from interp.hooks import ActivationCapture, default_sites


def parse_args():
    p = argparse.ArgumentParser(description="Stage 1: capture activations.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--steps", type=int, default=300,
                   help="Post-warmup steps recorded per episode.")
    p.add_argument("--warmup", type=int, default=20,
                   help="Leading steps per episode flagged is_warmup=True "
                        "(still recorded; covers DEL ti_window + slew-up).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--del-limits", type=str, default=None,
                   help="Comma floats, one capture condition each (randlim "
                        "checkpoints only). Default: training behavior.")
    p.add_argument("--ws", type=float, default=None, help="OOD wind-speed pin.")
    p.add_argument("--wd", type=float, default=None, help="OOD wind-dir pin.")
    p.add_argument("--ti", type=float, default=None, help="OOD TI pin.")
    p.add_argument("--stochastic", action="store_true",
                   help="Sample actions instead of the deterministic mean.")
    p.add_argument("--sites", type=str, default=None,
                   help="Comma site names (default: all).")
    p.add_argument("--layout", type=str, default=None)
    p.add_argument("--outdir", type=str, default=None,
                   help="Default: TransformerSac/data/interp/<run_name>/")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--no-verify", action="store_true")
    return p.parse_args()


def policy_step(actor, obs_t, pos_t, mask_t, rec_t, inf_t, deterministic):
    """One forward + tanh-squash, replicating TransformerActor.get_action but
    returning log_std too (get_action hides it). Single forward pass so the
    ActivationCapture buffer holds exactly this pass."""
    if not actor.use_influence:
        inf_t = None
    with torch.no_grad():
        mean, log_std, _ = actor.forward(obs_t, pos_t, mask_t, rec_t, inf_t)
        x_t = mean if deterministic else torch.distributions.Normal(
            mean, log_std.exp()).rsample()
        action = torch.tanh(x_t) * actor.action_scale + actor.action_bias_val
        mean_action = torch.tanh(mean) * actor.action_scale + actor.action_bias_val
    return action, mean_action, log_std


def rollout_condition(env, actor, batch_fn, sites, *, episodes, steps, warmup,
                      seed, del_limit, deterministic):
    """All episodes of one condition -> dict of stacked arrays."""
    base = base_env_of(env)
    rows = {name: [] for name in (
        "obs", "positions", "mask", "recep", "influ",
        "action", "mean_action", "log_std",
        "derate_cmd", "derate_real", "yaw", "power_per_turbine",
        "loads", "loads_baseline",
        "power_agent", "power_baseline", "reward", "reward_unpenalized",
        "del_ratio", "del_margin", "del_limit", "del_penalty",
        "ws", "wd", "ti", "episode_idx", "episode_step", "is_warmup",
    )}
    acts = {site: [] for site in sites}
    total = warmup + steps

    with ActivationCapture(actor, sites) as cap:
        for ep in range(episodes):
            ep_seed = seed + ep
            torch.manual_seed(ep_seed)
            np.random.seed(ep_seed)
            options = None if del_limit is None else {"del_limit": float(del_limit)}
            obs, info = env.reset(seed=ep_seed, options=options)
            for t in range(total):
                obs_t, pos_t, mask_t, rec_t, inf_t = batch_fn(obs=obs)
                action_t, mean_t, logstd_t = policy_step(
                    actor, obs_t, pos_t, mask_t, rec_t, inf_t, deterministic)
                snap = cap.snapshot()
                for site in sites:
                    acts[site].append(snap[site])

                rows["obs"].append(obs_t[0].cpu().numpy().copy())
                rows["positions"].append(pos_t[0].cpu().numpy().copy())
                rows["mask"].append(mask_t[0].cpu().numpy().copy())
                rows["recep"].append(
                    rec_t[0].cpu().numpy().copy() if rec_t is not None else None)
                rows["influ"].append(
                    inf_t[0].cpu().numpy().copy() if inf_t is not None else None)
                rows["action"].append(action_t[0].cpu().numpy().copy())
                rows["mean_action"].append(mean_t[0].cpu().numpy().copy())
                rows["log_std"].append(logstd_t[0].cpu().numpy().copy())

                # Wind state read PRE-step so it describes the state the
                # captured activations saw (constant per episode for the
                # power_max presets, but keep the alignment honest).
                step_ws, step_wd, step_ti = (
                    float(env.ws), float(env.wd), float(env.ti))
                obs, reward, term, trunc, info = env.step(
                    action_t.squeeze(0).cpu().numpy())
                if term or trunc:
                    raise RuntimeError(
                        f"Episode ended early at step {t} (terminated={term}, "
                        f"truncated={trunc}); max_episode_steps should be None."
                    )

                n_turb = env.n_turbines

                def per_turb(vals):
                    out = np.full(env.max_turbines, np.nan, dtype=np.float32)
                    v = np.asarray(vals, dtype=np.float32).reshape(-1)
                    out[:min(n_turb, v.shape[0])] = v[:n_turb]
                    return out

                powers = np.asarray(info["powers"], dtype=np.float32)
                if powers.ndim == 2:
                    powers = powers[-1]
                base_powers = np.asarray(info["baseline_powers"], dtype=np.float32)
                if base_powers.ndim == 2:
                    base_powers = base_powers[-1]

                rows["derate_cmd"].append(per_turb(info["derate command"]))
                rows["derate_real"].append(per_turb(base.current_derate))
                rows["yaw"].append(per_turb(base.fs.windTurbines.yaw_tilt()[0]))
                rows["power_per_turbine"].append(per_turb(powers))
                rows["loads"].append(per_turb(info["loads"][DEL_CHANNEL]))
                rows["loads_baseline"].append(
                    per_turb(info["loads_baseline"][DEL_CHANNEL]))
                rows["power_agent"].append(float(np.nansum(powers[:n_turb])))
                rows["power_baseline"].append(float(np.nansum(base_powers[:n_turb])))
                rows["reward"].append(float(reward))
                rows["reward_unpenalized"].append(float(info["reward_unpenalized"]))
                rows["del_ratio"].append(float(info["del_ratio"]))
                rows["del_margin"].append(float(info["del_margin"]))
                rows["del_limit"].append(float(info["del_limit"]))
                rows["del_penalty"].append(float(info["del_penalty"]))
                rows["ws"].append(step_ws)
                rows["wd"].append(step_wd)
                rows["ti"].append(step_ti)
                rows["episode_idx"].append(ep)
                rows["episode_step"].append(t)
                rows["is_warmup"].append(t < warmup)

                if (t + 1) % 20 == 0 or t == total - 1:
                    print(f"    ep {ep + 1}/{episodes} step {t + 1}/{total}: "
                          f"mean_derate={np.nanmean(rows['derate_cmd'][-1]):.3f} "
                          f"del_ratio={rows['del_ratio'][-1]:.3f}", flush=True)

    out = {}
    for site in sites:
        out[f"act_{site}"] = np.stack(acts[site]).astype(np.float32)
    for name, vals in rows.items():
        if name in ("recep", "influ"):
            if vals[0] is None:
                continue
            out[name] = np.stack(vals).astype(np.float32)
        elif name == "mask":
            out[name] = np.stack(vals).astype(bool)
        elif name == "is_warmup":
            out[name] = np.asarray(vals, dtype=bool)
        elif name in ("episode_idx", "episode_step"):
            out[name] = np.asarray(vals, dtype=np.int64)
        else:
            out[name] = (np.stack(vals) if isinstance(vals[0], np.ndarray)
                         else np.asarray(vals)).astype(np.float32)
    return out


def verify_replay(data, actor, device, batch_size=256, tol=5e-6):
    """Offline determinism check: actor.forward on the saved inputs must
    reproduce the saved mean_action (proves the npz suffices for probing and
    that capture didn't perturb the policy). tol allows float32 BLAS
    reduction-order drift (replay batches 256 rows, the rollout ran batch 1;
    observed ~1e-6) while still catching real input mismatch (orders of
    magnitude larger)."""
    n = data["obs"].shape[0]
    has_profiles = "recep" in data
    max_diff = 0.0
    for lo in range(0, n, batch_size):
        hi = min(lo + batch_size, n)
        obs_t = torch.as_tensor(data["obs"][lo:hi], device=device)
        pos_t = torch.as_tensor(data["positions"][lo:hi], device=device)
        mask_t = torch.as_tensor(data["mask"][lo:hi], device=device)
        rec_t = (torch.as_tensor(data["recep"][lo:hi], device=device)
                 if has_profiles else None)
        inf_t = (torch.as_tensor(data["influ"][lo:hi], device=device)
                 if has_profiles and "influ" in data and actor.use_influence
                 else None)
        with torch.no_grad():
            mean, _, _ = actor.forward(obs_t, pos_t, mask_t, rec_t, inf_t)
            mean_action = (torch.tanh(mean) * actor.action_scale
                           + actor.action_bias_val).cpu().numpy()
        max_diff = max(max_diff, float(
            np.abs(mean_action - data["mean_action"][lo:hi]).max()))
    if max_diff > tol:
        raise RuntimeError(
            f"Replay determinism FAILED: max |mean_action diff| = {max_diff:.3e} "
            f"> {tol:.0e} -- saved inputs do not reproduce the rollout."
        )
    print(f"  [verify] offline replay OK (max diff {max_diff:.2e})")


def limit_tag(limit) -> str:
    return "none" if limit is None else f"{limit:g}"


def main():
    cli = parse_args()
    device = torch.device(cli.device)
    sys.stdout.reconfigure(line_buffering=True)

    checkpoint, args = load_checkpoint(cli.checkpoint, device)
    randlim = is_randlim(args)

    limits = [None]
    if cli.del_limits is not None:
        if not randlim:
            raise SystemExit(
                "--del-limits requires a randlim (del_limit_random=True) "
                "checkpoint: plain delmax agents never saw a limit input, so "
                "conditioning on one is meaningless."
            )
        limits = [float(x) for x in cli.del_limits.split(",")]

    wind_overrides = {k: v for k, v in
                      (("ws", cli.ws), ("wd", cli.wd), ("ti", cli.ti))
                      if v is not None}

    env, _config = build_delmax_env(
        args, layout_name=cli.layout, seed=cli.seed,
        wind_overrides=wind_overrides or None,
    )
    actor = build_actor(args, env, checkpoint, device)
    batch_fn = make_batch_fn(args, env, device)
    sites = (cli.sites.split(",") if cli.sites is not None
             else default_sites(actor))

    run_name, step_tag = run_name_of(cli.checkpoint)
    outdir = cli.outdir or os.path.join(_TSAC_DIR, "data", "interp", run_name)
    os.makedirs(outdir, exist_ok=True)

    print(f"[capture] run={run_name} {step_tag} randlim={randlim} "
          f"sites={sites}")
    print(f"[capture] conditions={[limit_tag(l) for l in limits]} "
          f"episodes={cli.episodes} steps={cli.warmup}+{cli.steps} "
          f"seed={cli.seed} stochastic={cli.stochastic} "
          f"wind_overrides={wind_overrides or 'training inflow'}")

    try:
        for limit in limits:
            print(f"  condition del_limit={limit_tag(limit)}")
            data = rollout_condition(
                env, actor, batch_fn, sites,
                episodes=cli.episodes, steps=cli.steps, warmup=cli.warmup,
                seed=cli.seed, del_limit=limit,
                deterministic=not cli.stochastic,
            )
            data["metadata_json"] = make_metadata(
                cli.checkpoint, checkpoint, args,
                stage="capture", sites=sites, layout=env.current_layout.name,
                seed=cli.seed, episodes=cli.episodes, steps=cli.steps,
                warmup=cli.warmup, del_limit=limit, stochastic=cli.stochastic,
                wind_overrides=wind_overrides, ood=bool(wind_overrides),
            )
            # Save BEFORE verifying: a tolerance trip must never discard the
            # (expensive) rollout data.
            path = os.path.join(
                outdir, f"capture_lim{limit_tag(limit)}_s{cli.seed}.npz")
            np.savez_compressed(path, **data)
            mb = os.path.getsize(path) / 1e6
            print(f"  wrote {path} ({mb:.1f} MB, "
                  f"S={data['obs'].shape[0]}, T={data['obs'].shape[1]})")
            if not cli.no_verify:
                verify_replay(data, actor, device)
    finally:
        env.close()
    print("[capture] done")


if __name__ == "__main__":
    main()
