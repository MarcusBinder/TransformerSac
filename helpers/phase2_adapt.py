"""Stage 10 phase 2: train the adaptation module phi (RMA / UP-OSI / Lee student).

phi maps each turbine's [obs | commanded-action history] (+ positions,
profiles; a transformer over turbines) to z_hat, the estimate of the actor's
conditioning latent z = mu(e) (raw e when cond_latent_dim == 0).

Protocol (UP-OSI / RMA, DAgger-style):
  1. warm start on the RL replay buffer (true e stored per transition);
  2. ``adapt_rounds`` rounds of ``num_envs x adapt_round_steps`` env steps on
     the TRAINING envs (DR-wrapped, shuffled, SAME_STEP autoreset) with z_hat
     IN THE LOOP (the policy acts on phi's estimate, so phi sees the state
     distribution it will face at deployment), accumulating (phi-input, e)
     pairs in a growing buffer; refit on the aggregate after every round.

Losses: ``latent`` = masked MSE(z_hat, z) over real tokens; ``latent_action``
adds MSE(student mean action | z_hat, teacher mean action | z) 1:1 (Lee et al.
2020 Eq. 1). With ``phase2_train_actor`` the student is a trained copy of the
actor (saved as student_actor_state_dict); otherwise only phi is trained and
the action term only shapes z_hat.

Identifiability diagnostics after each round: per-dim relative RMSE of z_hat
and a linear-probe R^2 of each of the 12 physical parameters from z_hat.
"""
from __future__ import annotations

import copy
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from networks import AdaptationModule, masked_mse


# ----------------------------------------------------------------- targets --
def make_target_z(actor: nn.Module, e: torch.Tensor) -> torch.Tensor:
    """z = mu(e) from the (frozen) actor's parameter encoder; raw e when dz == 0."""
    with torch.no_grad():
        return actor.encode_cond(e).detach()


# ----------------------------------------------------------- batch sources --
class ArrayBatchSource:
    """Uniform batches from a dict of full tensors (tests / held-out probes)."""

    def __init__(self, data: Dict[str, torch.Tensor], batch_size: int, device: torch.device,
                 seed: int = 0, obs_normalizer=None):
        self.data = data
        self.n = int(next(iter(data.values())).shape[0])
        self.batch_size = int(batch_size)
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.obs_normalizer = obs_normalizer

    def sample(self) -> Dict[str, torch.Tensor]:
        idx = torch.as_tensor(self.rng.integers(0, self.n, size=self.batch_size))
        out = {k: v[idx].to(self.device) for k, v in self.data.items()}
        if self.obs_normalizer is not None:
            out["observations"] = self.obs_normalizer.normalize(out["observations"])
        return out


class ReplayBatchSource:
    """Batches from a TransformerReplayBuffer that carries params + action_hist."""

    def __init__(self, rb, batch_size: int, obs_normalizer=None):
        if rb.cond_dim <= 0 or rb.action_hist_dim <= 0:
            raise ValueError("phase 2 needs a replay buffer with params and action_hist columns")
        self.rb = rb
        self.batch_size = int(batch_size)
        self.obs_normalizer = obs_normalizer

    def sample(self) -> Dict[str, torch.Tensor]:
        out = self.rb.sample(self.batch_size)
        if self.obs_normalizer is not None:
            out["observations"] = self.obs_normalizer.normalize(out["observations"])
        return out


# ------------------------------------------------------------------- fitting --
def _mean_action(actor: nn.Module, batch: Dict[str, torch.Tensor], cond: torch.Tensor,
                 cond_is_latent: bool) -> torch.Tensor:
    mean, _, _ = actor(batch["observations"], batch["positions"], batch["attention_mask"],
                       batch.get("receptivity"), batch.get("influence"),
                       cond=cond, cond_is_latent=cond_is_latent)
    return torch.tanh(mean)


def fit_adaptation(adapt: AdaptationModule, actor: nn.Module, source, steps: int,
                   optimizer: torch.optim.Optimizer, loss_kind: str, device: torch.device,
                   student: Optional[nn.Module] = None, log_every: int = 100,
                   writer=None, tag: str = "adapt", step_offset: int = 0) -> List[Dict[str, float]]:
    """Gradient steps on phi (and the student); returns the logged history.

    ``loss_kind``: "latent" | "latent_action". The teacher ``actor`` is used
    under no_grad only (its params may stay requires_grad=False)."""
    if loss_kind not in ("latent", "latent_action"):
        raise ValueError(f"loss_kind must be latent | latent_action, got {loss_kind!r}")
    adapt.train()
    if student is not None:
        student.train()
    hist: List[Dict[str, float]] = []
    acc_mse, acc_act, n_acc = 0.0, 0.0, 0
    for step in range(1, int(steps) + 1):
        batch = source.sample()
        mask = batch["attention_mask"]
        z_hat = adapt(batch["observations"], batch["action_hist"], batch["positions"], mask,
                      batch.get("receptivity"), batch.get("influence"))
        z = make_target_z(actor, batch["params"])
        mse = masked_mse(z_hat.float(), z.float(), mask)
        loss = mse
        act_mse = None
        if loss_kind == "latent_action":
            with torch.no_grad():
                a_teacher = _mean_action(actor, batch, batch["params"], False)
            a_student = _mean_action(student if student is not None else actor, batch, z_hat, True)
            act_mse = masked_mse(a_student.float(), a_teacher.float(), mask)
            loss = loss + act_mse
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        acc_mse += float(mse.detach()); n_acc += 1
        if act_mse is not None:
            acc_act += float(act_mse.detach())
        if step % log_every == 0 or step == steps:
            rec = {"step": step_offset + step, "mse": acc_mse / n_acc}
            if loss_kind == "latent_action":
                rec["action_mse"] = acc_act / n_acc
            if writer is not None:
                writer.add_scalar(f"{tag}/mse", rec["mse"], rec["step"])
                if "action_mse" in rec:
                    writer.add_scalar(f"{tag}/action_mse", rec["action_mse"], rec["step"])
            hist.append(rec)
            acc_mse, acc_act, n_acc = 0.0, 0.0, 0
    return hist


# --------------------------------------------------------------- diagnostics --
def _real_rows(x: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    return x.detach().float().cpu()[~mask.cpu()].numpy()


def per_dim_rel_error(z_hat: torch.Tensor, z: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    """RMSE / std per latent dim over real tokens (0 for a constant dim with zero error)."""
    a, b = _real_rows(z_hat, mask), _real_rows(z, mask)
    rmse = np.sqrt(np.mean((a - b) ** 2, axis=0))
    sd = b.std(axis=0)
    return np.where(sd > 1e-12, rmse / np.maximum(sd, 1e-12), np.where(rmse > 1e-12, np.inf, 0.0))


def linear_probe_r2(z_hat: torch.Tensor, e: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    """Least-squares probe e ~ [z_hat, 1] over real tokens; R^2 per physical
    parameter (NaN for a zero-variance column = an unranged key)."""
    Z, E = _real_rows(z_hat, mask).astype(np.float64), _real_rows(e, mask).astype(np.float64)
    X = np.concatenate([Z, np.ones((Z.shape[0], 1))], axis=1)
    W, *_ = np.linalg.lstsq(X, E, rcond=None)
    res = E - X @ W
    ss_res = (res ** 2).sum(axis=0)
    ss_tot = ((E - E.mean(axis=0)) ** 2).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = np.where(ss_tot > 1e-12, 1.0 - ss_res / ss_tot, np.nan)
    return r2


def identifiability_table(adapt: AdaptationModule, actor: nn.Module, source, n_batches: int,
                          keys) -> Dict[str, Any]:
    """Held-out z MSE, per-dim relative error of z_hat and per-key R^2 of e."""
    adapt.eval()
    zs, zh, es, ms = [], [], [], []
    with torch.no_grad():
        for _ in range(n_batches):
            b = source.sample()
            zh.append(adapt(b["observations"], b["action_hist"], b["positions"], b["attention_mask"],
                            b.get("receptivity"), b.get("influence")).float().cpu())
            zs.append(make_target_z(actor, b["params"]).float().cpu())
            es.append(b["params"].float().cpu())
            ms.append(b["attention_mask"].cpu())
    z_hat, z, e, mask = torch.cat(zh), torch.cat(zs), torch.cat(es), torch.cat(ms)
    r2 = linear_probe_r2(z_hat, e, mask)
    return {
        "mse": float(masked_mse(z_hat, z, mask)),
        "rel_err": per_dim_rel_error(z_hat, z, mask),
        "r2": {k: float(v) for k, v in zip(keys, r2)},
    }


# ----------------------------------------------------------- the DAgger loop --
def _uncompile(module: nn.Module) -> None:
    """Drop a torch.compile'd instance forward (phase 2 calls forward with new
    Python-bool arguments; the uncompiled class method avoids a recapture)."""
    if "forward" in module.__dict__:
        del module.__dict__["forward"]


def run_phase2_guarded(fn, **kwargs):
    """Run phase 2 so that an exception never costs the already-saved RL
    checkpoint: logs the traceback loudly, returns None (the trainer then exits
    0 and the launcher marks the seed done; recovery = ``--adapt_only
    --resume_checkpoint <final ckpt>``)."""
    import traceback
    try:
        return fn(**kwargs)
    except Exception:  # noqa: BLE001 - phase 2 is optional post-processing
        print("\n[adapt] PHASE 2 FAILED (the RL checkpoint is already saved). Traceback:", flush=True)
        traceback.print_exc()
        print("[adapt] recover with: --adapt_only --resume_checkpoint runs/<run>/checkpoints/step_<final>.pt "
              "(same flags, new --exp_name)", flush=True)
        return None


def _adapt_sd(adapt: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in adapt.state_dict().items()}


def run_phase2(args, actor: nn.Module, envs, rb, cond_normalizer, device: torch.device,
               *, trunk_kwargs: dict, rotor_diameter: float, use_profiles: bool,
               obs_normalizer, wd_attr: str, profile_registry, n_turbines_max: int,
               obs_dim: int, action_dim: int, global_step: int, writer=None,
               save_fn=None, stop_fn=None) -> Dict[str, Any]:
    """Train phi after the RL phase; returns adapt / student state dicts + metrics.

    ``save_fn(adapt_state_dict, metrics)`` is called after EVERY round (a
    SIGTERM mid-phase keeps the last fitted phi); ``stop_fn() -> bool`` is
    polled after each round (the trainer's SIGTERM flag) and ends the phase
    early with ``metrics["stopped_early"] = True``. Gate-2 metrics per round:
    ``heldout_*`` = phi evaluated on the round's NEW pairs before it is fitted
    on them (true held-out), ``mse`` / ``rel_err`` / ``r2`` = after the fit on
    the aggregate (held-in).
    """
    from helpers.action_history import N_FEAT_DEFAULT
    from helpers.agent import WindFarmAgent
    from helpers.helper_funcs import (
        get_env_attention_masks, get_env_influence_profiles, get_env_layout_indices,
        get_env_permutations, get_env_raw_positions, get_env_receptivity_profiles,
        get_env_wind_directions,
    )
    from helpers.param_cond import COND_FARM_KEYS, COND_TURB_KEYS, CondFetcher
    from replay_buffer import TransformerReplayBuffer

    t0 = time.time()
    keys = COND_TURB_KEYS + COND_FARM_KEYS
    _uncompile(actor)
    actor.eval()
    for p in actor.parameters():
        p.requires_grad_(False)                       # teacher frozen (mu included)
    student = None
    if args.phase2_train_actor:
        student = copy.deepcopy(actor)
        for p in student.parameters():
            p.requires_grad_(True)
    tk = dict(trunk_kwargs)
    tk["shared_recep_encoder"] = None
    tk["shared_influence_encoder"] = None
    tk.pop("obs_dim_per_turbine", None)
    tk.pop("action_dim_per_turbine", None)
    adapt = AdaptationModule(obs_dim, N_FEAT_DEFAULT, actor.cond_dim, **tk).to(device)
    params = list(adapt.parameters()) + (list(student.parameters()) if student is not None else [])
    opt = torch.optim.Adam(params, lr=args.adapt_lr)
    n_params = sum(p.numel() for p in adapt.parameters())
    print("\n" + "=" * 60)
    print(f"PHASE 2: adaptation module phi ({n_params:,} params, out_dim {actor.cond_dim}, "
          f"loss {args.phase2_loss}{', student actor' if student is not None else ''})")
    print("=" * 60)

    metrics: Dict[str, Any] = {"rounds": []}
    # (a) warm start on the RL buffer
    if rb is not None and len(rb) >= args.batch_size and rb.cond_dim > 0 and rb.action_hist_dim > 0 \
            and args.adapt_warm_steps > 0:
        src = ReplayBatchSource(rb, args.batch_size, obs_normalizer)
        hist = fit_adaptation(adapt, actor, src, args.adapt_warm_steps, opt, args.phase2_loss, device,
                              student=student, log_every=max(1, args.adapt_warm_steps // 10),
                              writer=writer, tag="adapt/warm")
        print(f"[adapt] warm start on {len(rb)} RL transitions: mse {hist[0]['mse']:.4g} -> {hist[-1]['mse']:.4g}")
        metrics["warm"] = hist
    else:
        print("[adapt] no warm start (RL buffer empty / lacks params+action_hist / adapt_warm_steps 0)")

    # (b) DAgger rounds with z_hat in the loop on the TRAINING envs
    cap = max(1, int(args.adapt_rounds) * int(args.num_envs) * int(args.adapt_round_steps))
    round_cap = max(1, int(args.num_envs) * int(args.adapt_round_steps))
    _rb_kw = dict(device=device, rotor_diameter=rotor_diameter, max_turbines=n_turbines_max,
                  obs_dim=obs_dim, action_dim=action_dim, use_wind_relative=args.use_wind_relative_pos,
                  use_profiles=use_profiles, rotate_profiles=args.rotate_profiles,
                  profile_registry=profile_registry, cond_dim=cond_normalizer.cond_dim,
                  action_hist_dim=N_FEAT_DEFAULT)
    adapt_rb = TransformerReplayBuffer(capacity=cap, **_rb_kw)      # DAgger aggregate
    round_rb = TransformerReplayBuffer(capacity=round_cap, **_rb_kw)  # this round only (held-out probe)
    agent_phi = WindFarmAgent(
        actor=student if student is not None else actor, device=device, rotor_diameter=rotor_diameter,
        use_wind_relative=args.use_wind_relative_pos, use_profiles=use_profiles,
        rotate_profiles=args.rotate_profiles, obs_normalizer=obs_normalizer, wd_attr=wd_attr,
        use_action_hist=True, adaptation=adapt,
    )
    fetcher = CondFetcher(envs.env, cond_normalizer)
    obs, _ = envs.reset(seed=int(args.seed) + 7_000_000)
    prev_dones = np.ones(args.num_envs, dtype=bool)
    metrics["stopped_early"] = False
    for r in range(int(args.adapt_rounds)):
        adapt.eval()
        round_rb.position = 0; round_rb.size = 0
        for t in range(int(args.adapt_round_steps)):
            # fetch block (all IPC before the step; every stored field captured BEFORE the step)
            wind_dirs = get_env_wind_directions(envs, attr=wd_attr)
            raw_positions = get_env_raw_positions(envs)
            masks = get_env_attention_masks(envs)
            if use_profiles:
                layout_idx = get_env_layout_indices(envs)
                perms = get_env_permutations(envs)
                recep = get_env_receptivity_profiles(envs)
                infl = get_env_influence_profiles(envs)
            else:
                layout_idx = perms = recep = infl = None
            if prev_dones.any():
                fetcher.refresh()
            e = fetcher.values
            hist = np.array(envs.env.get_attr("action_history"), dtype=np.float32)
            actions = agent_phi.act(envs, obs, deterministic=True, wind_dirs=wind_dirs,
                                    raw_positions=raw_positions, masks=masks, receptivity=recep,
                                    influence=infl, params=e, action_hist=hist)
            next_obs, rewards, term, trunc, infos = envs.step(actions)
            prev_dones = np.logical_or(term, trunc)
            for i in range(args.num_envs):
                _add = dict(
                    obs=obs[i], next_obs=next_obs[i], action=np.asarray(actions[i]).reshape(-1, action_dim),
                    reward=float(rewards[i]), done=bool(prev_dones[i]), raw_positions=raw_positions[i],
                    attention_mask=masks[i], wind_direction=float(wind_dirs[i]),
                    layout_index=(layout_idx[i] if layout_idx is not None else None),
                    permutation=(perms[i] if perms is not None else None),
                    params=e[i], action_hist=hist[i],
                )
                adapt_rb.add(**_add)
                round_rb.add(**_add)
            obs = next_obs
        # held-out probe: phi has not been fitted on this round's pairs yet
        heldout = identifiability_table(adapt, actor, ReplayBatchSource(round_rb, args.batch_size, obs_normalizer),
                                        n_batches=8, keys=keys)
        src = ReplayBatchSource(adapt_rb, args.batch_size, obs_normalizer)
        hist_fit = fit_adaptation(adapt, actor, src, args.adapt_fit_steps, opt, args.phase2_loss, device,
                                  student=student, log_every=max(1, args.adapt_fit_steps // 5),
                                  writer=writer, tag="adapt/fit",
                                  step_offset=r * args.adapt_fit_steps)
        table = identifiability_table(adapt, actor, src, n_batches=8, keys=keys)
        rec = {"round": r + 1, "n_pairs": len(adapt_rb), "fit": hist_fit, **table,
               "heldout_mse": heldout["mse"], "heldout_rel_err": heldout["rel_err"], "heldout_r2": heldout["r2"]}
        metrics["rounds"].append(rec)
        print(f"[adapt] round {r + 1}/{args.adapt_rounds}: {len(adapt_rb)} pairs, "
              f"held-OUT z mse {heldout['mse']:.4g} (pre-fit, new pairs), fit mse {hist_fit[-1]['mse']:.4g}, "
              f"held-in z mse {table['mse']:.4g}, held-out rel err per dim "
              f"{np.array2string(heldout['rel_err'], precision=2)}")
        if writer is not None:
            writer.add_scalar("adapt/mse_round", table["mse"], r + 1)
            writer.add_scalar("adapt/heldout_mse_round", heldout["mse"], r + 1)
            for k, v in table["r2"].items():
                if np.isfinite(v):
                    writer.add_scalar(f"adapt/r2_{k}", v, r + 1)
            for k, v in heldout["r2"].items():
                if np.isfinite(v):
                    writer.add_scalar(f"adapt/heldout_r2_{k}", v, r + 1)
            for j, v in enumerate(table["rel_err"]):
                writer.add_scalar(f"adapt/relerr_dim{j}", float(v), r + 1)
        if save_fn is not None:
            save_fn(_adapt_sd(adapt), metrics)
        if stop_fn is not None and stop_fn():
            print(f"[adapt] stop requested after round {r + 1}: ending phase 2 early (phi saved)")
            metrics["stopped_early"] = True
            break
    if metrics["rounds"]:
        r2 = metrics["rounds"][-1]["heldout_r2"]
        print("[adapt] identifiability (held-out linear probe R^2 of e from z_hat, last round):")
        print("        " + "  ".join(f"{k}={v:.2f}" if np.isfinite(v) else f"{k}=n/a" for k, v in r2.items()))
    print(f"[adapt] phase 2 done in {(time.time() - t0) / 60:.1f} min")
    return {
        "adapt_state_dict": _adapt_sd(adapt),
        "student_actor_state_dict": ({k: v.detach().cpu() for k, v in student.state_dict().items()}
                                     if student is not None else None),
        "metrics": metrics,
    }
