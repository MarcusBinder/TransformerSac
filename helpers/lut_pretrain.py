"""LUT pretraining (lutpre1, plan 2026-10-30): give a plain policy lut3's head start.

``--initial_exploration lut`` replaces the random warm-up with the harness lut3 controller:

1. warm-up (``global_step < learning_starts``): every env acts with the lut3 target plus an
   AR(1) exploration offset; the replay buffer stores the noisy action as the action and the
   CLEAN lut3 action as the row's label (``TransformerReplayBuffer(label_dim=...)``);
2. at ``learning_starts`` (no env steps): ``run_bc`` fits the actor's mean to the labels
   (MSE in the squashed action space) and its pre-tanh std to ``--bc_std``, then the trainer
   saves ``checkpoints/bc.pt`` and rebuilds the actor optimizer at ``--policy_lr``;
3. ``--finetune_freeze_actor_steps``: critics only (the BC actor acts, stochastic);
4. RL fine-tune.

No LUT at run time: the checkpoint is a plain RL row in every harness (no ``action_prior``).

The teacher is the residual arm's prior (helpers/action_prior.py) without the residual: the
robust table (mode 3, sigma 2.3, sign farm) built once per worker, indexed with the prior's OWN
estimator ``env.wd_est_prior`` (tau 60 / front / deadband 0 = the harness lut3 feed), the target
expressed in the agent's yaw frame (``lut_frame``). Labels per ActionMethod:

    wind:  wind_action_from_target(target)                       (the residual arm's inverse)
    yaw:   clip((target - yaw_command_ctrl) / yaw_step_env, -1, 1)

The yaw label integrates against the COMMAND (``env.yaw_command`` in the agent frame), not the
measured yaw: under the turbine-lag DR (tau_yaw / delay_yaw) the measured yaw lags, while the
yaw action moves the command, so the command form reproduces the lut3 row's command trajectory.

Exploration noise is added to the TARGET (deg), AR(1) per turbine with stationary sigma
``--lut_noise_deg`` and correlation time ``--lut_noise_tau_s``, redrawn from its stationary law
at every reset: white noise would average out under the yaw rate limit.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from WindGym.core.yaw_frame import wind_action_from_target

from helpers.action_prior import _arg, lut_frame, prior_point

TEACHER_METHODS = ("yaw", "wind")
NOISE_SEED_SALT = 0x10770   # keeps the noise stream independent of the env's own reset seed


# --------------------------------------------------------------------------- flags
def validate_lut_args(args, config: dict) -> None:
    """Refuse the combinations ``--initial_exploration lut`` cannot run (no-op otherwise)."""
    if _arg(args, "initial_exploration", "random") != "lut":
        return
    if _arg(args, "wd_source") != "est":
        raise ValueError("--initial_exploration lut needs --wd_source est (the teacher's estimator "
                         "sits next to the agent's)")
    if _arg(args, "backend") == "pywake":
        raise ValueError("--initial_exploration lut needs the dynamiks backend (pywake has no "
                         "measured local wd for the estimator)")
    if _arg(args, "action_prior") is not None:
        raise ValueError("--initial_exploration lut and --action_prior are exclusive (the point is a "
                         "plain policy without the LUT at run time)")
    if _arg(args, "resume_checkpoint") is not None:
        raise ValueError("--initial_exploration lut cannot --resume_checkpoint: a resumed run starts past "
                         "the warm-up with an empty (label-free) buffer and would re-run BC on nothing")
    if _arg(args, "load_buffer") is not None:
        raise ValueError("--initial_exploration lut generates its own labelled buffer; --load_buffer "
                         "would skip the warm-up")
    if _arg(args, "action_type") not in TEACHER_METHODS:
        raise ValueError(f"--initial_exploration lut needs --action_type in {TEACHER_METHODS}, "
                         f"got {_arg(args, 'action_type')!r}")
    if int(_arg(args, "learning_starts", 0)) <= 0:
        raise ValueError("--initial_exploration lut needs --learning_starts > 0 (the LUT warm-up)")
    if int(_arg(args, "buffer_size", 0)) < int(_arg(args, "learning_starts", 0)):
        raise ValueError("--buffer_size must hold the whole LUT warm-up (>= --learning_starts)")
    if int(_arg(args, "bc_steps", 0)) < 0:
        raise ValueError("--bc_steps must be >= 0")
    if int(_arg(args, "bc_batch_size", 1)) <= 0:
        raise ValueError("--bc_batch_size must be > 0")
    if not float(_arg(args, "bc_std", 0.0)) > 0:
        raise ValueError("--bc_std must be > 0 (pre-tanh std target)")
    if float(_arg(args, "lut_noise_deg", 0.0)) < 0 or not float(_arg(args, "lut_noise_tau_s", 1.0)) > 0:
        raise ValueError("--lut_noise_deg must be >= 0 and --lut_noise_tau_s > 0")
    if bool((config or {}).get("derate_action", False)):
        raise ValueError("--initial_exploration lut is yaw-only (the preset has derate_action)")


def teacher_point(args, config: dict) -> Tuple[float, float]:
    """(ws, ti) of the teacher table: ``--lut_teacher_ws/--lut_teacher_ti`` when given (lutpre1:
    the harness cell 9.192 / 0.0498 = the table the score's lut3 row uses), else the residual
    prior's point (midpoints of the training bands)."""
    ws, ti = prior_point(args, config)
    ws_o, ti_o = _arg(args, "lut_teacher_ws"), _arg(args, "lut_teacher_ti")
    return (round(float(ws_o), 3) if ws_o is not None else ws,
            round(float(ti_o), 4) if ti_o is not None else ti)


def deg_per_action_unit(method: str, *, yaw_min: float, yaw_max: float, yaw_step_env: float) -> float:
    """Degrees per unit of action: half the yaw band for "wind", the per-env-step budget for "yaw"."""
    if method == "wind":
        return 0.5 * (float(yaw_max) - float(yaw_min))
    if method == "yaw":
        return float(yaw_step_env)
    raise ValueError(f"unknown ActionMethod {method!r}")


# --------------------------------------------------------------------------- label maths
def teacher_action_from_target(target, method: str, *, yaw_min: float, yaw_max: float,
                               yaw_cmd=None, yaw_step_env: Optional[float] = None) -> np.ndarray:
    """The env action that commands the agent-frame yaw ``target`` (per turbine)."""
    t = np.clip(np.asarray(target, dtype=float), yaw_min, yaw_max)
    if method == "wind":
        return wind_action_from_target(t, yaw_min, yaw_max)
    if method == "yaw":
        return np.clip((t - np.asarray(yaw_cmd, dtype=float)) / float(yaw_step_env), -1.0, 1.0)
    raise ValueError(f"the LUT teacher needs ActionMethod in {TEACHER_METHODS}, got {method!r}")


class Ar1Noise:
    """Per-turbine AR(1): x' = phi x + sigma sqrt(1 - phi^2) eps, phi = exp(-dt / tau);
    ``reset`` draws x ~ N(0, sigma^2) (stationary), ``step`` advances one env step."""

    def __init__(self, n: int, sigma: float, tau_s: float, dt_s: float, seed: Optional[int] = None):
        self.n = int(n)
        self.sigma = float(sigma)
        self.phi = math.exp(-float(dt_s) / float(tau_s))
        self.rng = np.random.default_rng(None if seed is None else [int(seed), NOISE_SEED_SALT])
        self.value = np.zeros(self.n)

    def reseed(self, seed: int) -> None:
        self.rng = np.random.default_rng([int(seed), NOISE_SEED_SALT])

    def reset(self) -> np.ndarray:
        self.value = self.sigma * self.rng.standard_normal(self.n)
        return self.value

    def step(self) -> np.ndarray:
        self.value = (self.phi * self.value
                      + self.sigma * math.sqrt(1.0 - self.phi ** 2) * self.rng.standard_normal(self.n))
        return self.value


class LutTeacherWrapper(gym.Wrapper):
    """Innermost wrapper on a WindFarmEnv (the ActionPriorWrapper slot). Passes actions through
    unchanged; ``teacher_actions()`` returns (clean label, exploration action), physical order."""

    lut_teacher = True

    def __init__(self, env: gym.Env, ctrl, noise_deg: float, noise_tau_s: float, seed: Optional[int] = None):
        super().__init__(env)
        base = env.unwrapped
        method = getattr(base, "ActionMethod", None)
        if method not in TEACHER_METHODS:
            raise ValueError(f"the LUT teacher needs ActionMethod in {TEACHER_METHODS}, got {method!r}")
        if getattr(base, "derate_action", False):
            raise ValueError("the LUT teacher is yaw-only (derate_action=True env)")
        if getattr(base, "wd_est_tau", None) is None:
            raise ValueError("the LUT teacher needs the env's wd_est (build it with wd_est_tau: "
                             "--wd_source est)")
        if getattr(base, "prior_wd_est_tau", None) is None:
            raise ValueError("the LUT teacher needs its own estimator (build the env with "
                             "prior_wd_est_tau: action_prior.prior_wd_est_env_kwargs)")
        if float(getattr(ctrl, "deadband_deg", 0.0)) != 0.0:
            raise ValueError("the teacher controller's deadband must be 0 (the estimator holds)")
        if float(noise_deg) < 0 or not float(noise_tau_s) > 0:
            raise ValueError(f"teacher noise needs sigma >= 0 and tau > 0, got {noise_deg}, {noise_tau_s}")
        self.ctrl = ctrl
        self.method = method
        self.yaw_min = float(base.yaw_min)
        self.yaw_max = float(base.yaw_max)
        dt = float(getattr(base, "delay", None) or base.dt_env)    # sim seconds per env step
        self.noise = Ar1Noise(base.n_turb, noise_deg, noise_tau_s, dt, seed=seed)

    def __getattr__(self, name):
        """Forward WindFarmEnv attributes to the outer wrappers (gymnasium 1.x forwards nothing;
        same as ActionPriorWrapper)."""
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.env, name)

    def reset(self, *, seed=None, options=None):
        out = self.env.reset(seed=seed, options=options)
        if seed is not None:
            self.noise.reseed(seed)
        self.noise.reset()
        self.ctrl.reset()
        return out

    def step(self, action):
        out = self.env.step(action)
        self.noise.step()
        return out

    def teacher_target(self) -> np.ndarray:
        """lut3 yaw target (agent frame, clipped to the band) at the current state."""
        base = self.env.unwrapped
        tgt = self.ctrl.target_yaw(float(base.wd_est_prior), wd_frame=lut_frame(base))
        return np.clip(np.asarray(tgt, dtype=float), self.yaw_min, self.yaw_max)

    def teacher_actions(self) -> Tuple[np.ndarray, np.ndarray]:
        base = self.env.unwrapped
        tgt = self.teacher_target()
        noisy = np.clip(tgt + self.noise.value, self.yaw_min, self.yaw_max)
        kw = dict(yaw_min=self.yaw_min, yaw_max=self.yaw_max)
        if self.method == "yaw":
            kw.update(yaw_cmd=np.asarray(base.yaw_command, dtype=float) + float(base.frame_offset),
                      yaw_step_env=float(base.yaw_step_env))
        label = teacher_action_from_target(tgt, self.method, **kw)
        explore = teacher_action_from_target(noisy, self.method, **kw)
        return label.astype(np.float32), explore.astype(np.float32)


# --------------------------------------------------------------------------- BC
def run_bc(actor, rb, *, steps: int, lr: float, batch_size: int, std: float, std_coef: float,
           deg_per_unit: float, device, holdout_frac: float = 0.05, obs_normalizer=None,
           writer=None, log_every: int = 1000, grad_clip: float = 1.0, seed: int = 0) -> Dict[str, float]:
    """Fit ``actor`` to the buffer's labelled rows (in place, own Adam).

    loss = MSE(tanh(mean) * scale + bias, label) + std_coef * (log_std - log std)^2, both masked
    over real turbines. The holdout is the LAST ``holdout_frac`` of the labelled rows in insertion
    order (contiguous in time: neighbouring rows of one episode are near-duplicates, a random split
    would be optimistic). Every actor parameter trains, shared profile encoders included (the
    critics are still untrained at this point)."""
    import torch

    idx = rb.labelled_indices()
    if idx.size == 0:
        raise ValueError("BC needs labelled rows (the LUT warm-up stores them); the buffer has none")
    n_hold = int(round(float(holdout_frac) * idx.size))
    n_hold = min(n_hold, idx.size - 1)
    train_idx, hold_idx = (idx[: idx.size - n_hold], idx[idx.size - n_hold:]) if n_hold > 0 else (idx, idx[:0])
    rng = np.random.default_rng(seed)
    log_std_target = math.log(float(std))
    opt = torch.optim.Adam(actor.parameters(), lr=float(lr))

    def _loss(batch):
        obs = batch["observations"]
        if obs_normalizer is not None:
            obs = obs_normalizer.normalize(obs)
        infl = batch.get("influence") if getattr(actor, "use_influence", True) else None
        mean, log_std, _ = actor(obs, batch["positions"], batch["attention_mask"],
                                 recep_profile=batch.get("receptivity"), influence_profile=infl,
                                 cond=batch.get("params"))
        pred = torch.tanh(mean) * actor.action_scale + actor.action_bias_val
        real = (~batch["attention_mask"]).unsqueeze(-1).expand_as(pred).float()
        n = real.sum().clamp(min=1.0)
        lab = torch.nan_to_num(batch["labels"])
        mse = ((pred - lab).pow(2) * real).sum() / n
        std_err = ((log_std - log_std_target).pow(2) * real).sum() / n
        return mse, std_err

    def _holdout_rms() -> float:
        if hold_idx.size == 0:
            return float("nan")
        sq, cnt = 0.0, 0.0
        with torch.no_grad():
            for k in range(0, hold_idx.size, int(batch_size)):
                b = rb.sample_at(hold_idx[k:k + int(batch_size)])
                mse, _ = _loss(b)
                w = float((~b["attention_mask"]).sum()) * b["labels"].shape[-1]
                sq += float(mse) * w
                cnt += w
        return math.sqrt(sq / max(cnt, 1.0)) * float(deg_per_unit)

    actor.train()
    run_mse = None
    loss_v = float("nan")
    for k in range(int(steps)):
        b = rb.sample_at(rng.choice(train_idx, size=int(batch_size)))
        mse, std_err = _loss(b)
        loss = mse + float(std_coef) * std_err
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=float(grad_clip))
        opt.step()
        m = float(mse.detach())
        run_mse = m if run_mse is None else 0.98 * run_mse + 0.02 * m
        loss_v = float(loss.detach())
        if writer is not None and (k % int(log_every) == 0 or k == int(steps) - 1):
            writer.add_scalar("bc/loss", loss_v, k)
            writer.add_scalar("bc/rms_deg_train", math.sqrt(run_mse) * float(deg_per_unit), k)
            writer.add_scalar("bc/rms_deg_holdout", _holdout_rms(), k)
    stats = {
        "n_labelled": int(idx.size), "n_holdout": int(hold_idx.size), "steps": int(steps),
        "loss": loss_v,
        "rms_deg_train": (math.sqrt(run_mse) * float(deg_per_unit)) if run_mse is not None else float("nan"),
        "rms_deg_holdout": _holdout_rms(),
    }
    return stats
