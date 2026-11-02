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

lutpre2 (2026-11-01) splits the run in two: a COLLECT run (``--buffer_only``) saves the warm-up
buffer at learning_starts BEFORE BC (exactly the teacher rows, every one labelled), with a dataset
fingerprint (``dataset_fingerprint``: every arg that shapes obs, actions, labels or rewards) in its
meta; a TRAIN run ``--load_buffer``s one or more such datasets (appended: pooling), starts its step
counter at the warm-up's last step (``dataset_start_step``), so BC fires at the first iteration and
the freeze window, lr schedules and logged steps keep lutpre1's origin. The envs start fresh (a
dataset cannot restore mid-episode env state). ``check_dataset_meta`` refuses a fingerprint mismatch
except on the poolable axes (``POOLABLE_KEYS``: wd function, wd band, seed).

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

import hashlib
import json
import math
from typing import Dict, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np

from WindGym.core.yaw_frame import wind_action_from_target

from helpers.action_prior import _arg, lut_frame, prior_point

TEACHER_METHODS = ("yaw", "wind")
NOISE_SEED_SALT = 0x10770   # keeps the noise stream independent of the env's own reset seed

# lutpre2: every trainer arg that shapes a dataset row (obs, action, label, reward, done), i.e.
# what two datasets must share to pool and what a train run must share with its dataset. RL-only
# args (lrs, network, buffer, num_envs, logging) are not here: they never touch a stored row.
DATASET_FP_KEYS = (
    # env / physics
    "config", "power_schedule", "backend", "turbtype", "iea34_variant", "TI_type", "precursor_path",
    "dt_sim", "dt_env", "yaw_step", "derate_step_sim", "max_turb_move", "dwm_interpolation",
    "dwm_lateral_cutoff", "veer_min", "veer_max", "tilt", "dr_posterior_path", "dr_keys", "turb_dr",
    "turb_dr_per_turbine", "train_ws_min", "train_ws_max", "ti_min", "ti_max", "train_wd_quant_deg",
    "yaw_min", "yaw_max", "max_episode_steps", "max_eps", "stagger_warmup", "warmup_group_size",
    "warmup_min_episode_steps",
    # layouts / tokens
    "layouts", "max_turbines", "shuffle_turbs", "dr_n_lo", "dr_n_hi", "dr_pool_size", "dr_min_dist_D",
    "dr_screen_headroom", "dr_min_involved_frac", "dr_generator", "dr_grid_spacing_lo",
    "dr_grid_spacing_hi", "dr_grid_exclude_D", "dr_include_layouts", "dr_include_weight",
    "profile_encoding_type", "n_profile_directions",
    # observations
    "history_length", "history_N", "window_length", "use_wd_deviation", "obs_agg", "obs_agg_len",
    "obs_norm", "obs_sensors", "obs_yaw_current", "ws_scaling_min", "ws_scaling_max", "wd_scaling_min",
    "wd_scaling_max", "ti_scaling_min", "ti_scaling_max", "wd_source", "wd_est_tau", "wd_est_consensus",
    "wd_est_deadband", "yaw_frame_source", "cond_source", "cond_norm_json", "cond_action_hist",
    "cond_action_hist_len",
    # actions / reward / done
    "action_type", "power_reward", "power_avg", "power_scaling", "reward_tau", "action_penalty",
    "action_penalty_type", "reversal_penalty", "bootstrap_truncated", "curriculum",
    "del_penalty_scale", "del_log", "load_reward_proxies", "load_penalty",
    # teacher (labels + exploration)
    "lut_teacher_ws", "lut_teacher_ti", "lut_noise_deg", "lut_noise_tau_s",
    "action_prior_wd_est_tau", "action_prior_wd_est_consensus", "action_prior_wd_est_deadband",
)
# the axes datasets may differ on and still pool (Marcus 2026-11-01): wd function, wd band, seed
POOLABLE_KEYS = ("train_wd_function", "wd_min", "wd_max", "seed")


# --------------------------------------------------------------------------- flags
def validate_lut_args(args, config: dict) -> None:
    """Refuse the combinations ``--initial_exploration lut`` cannot run (no-op otherwise)."""
    if _arg(args, "initial_exploration", "random") != "lut":
        if bool(_arg(args, "bc_only", False)):
            raise ValueError("--bc_only needs --initial_exploration lut (BC on a lut dataset)")
        if float(_arg(args, "bc_anchor_coef", 0.0) or 0.0) != 0.0:
            raise ValueError("--bc_anchor_coef needs --initial_exploration lut (the anchor is the bc.pt actor)")
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
    if _arg(args, "load_buffer") is not None and bool(_arg(args, "obs_norm", False)):
        # lutpre2: a dataset carries no normalizer state; BC would normalize with cold stats
        raise ValueError("--load_buffer of a lut dataset cannot run with --obs_norm (the dataset "
                         "carries no normalizer state)")
    if bool(_arg(args, "bc_only", False)) and _arg(args, "load_buffer") is None:
        raise ValueError("--bc_only fits BC on loaded lut datasets: pass --load_buffer")
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
    coef = float(_arg(args, "bc_anchor_coef", 0.0))
    if coef < 0:
        raise ValueError("--bc_anchor_coef must be >= 0")
    if coef > 0 and bool(_arg(args, "compile", False)):
        raise ValueError("--bc_anchor_coef > 0 cannot run with --compile (the anchor deep-copies the actor "
                         "after BC; a compiled forward does not copy)")


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


# --------------------------------------------------------------------------- lutpre2 datasets
def _fp_value(v):
    if isinstance(v, (tuple, list)):
        return [_fp_value(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _fp_value(x) for k, x in v.items()}
    return v


def dataset_fingerprint(args) -> Dict:
    """{key: value} of ``DATASET_FP_KEYS`` (tuples as lists: JSON round-trips equal). The trainer's
    deterministic post-parse coupling (history_N None -> history_length) is resolved here, so the raw
    argv (the LESRL driver) and the trainer's resolved args fingerprint alike; the args the trainer
    rewrites from a checkpoint or a normaliser (pretrain prescan, cond_norm_json) are not, so a dataset
    built that way never matches a raw argv (refused, never silently pooled)."""
    fp = {k: _fp_value(_arg(args, k)) for k in DATASET_FP_KEYS}
    if fp["history_N"] is None:
        fp["history_N"] = fp["history_length"]
    return fp


def fingerprint_hash(fp: Dict) -> str:
    """sha1 (12 hex) of the fingerprint (JSON, sorted keys): the library's directory name."""
    return hashlib.sha1(json.dumps(fp, sort_keys=True, default=str).encode()).hexdigest()[:12]


def dataset_meta(args, label_rows: int) -> Dict:
    """The lutpre2 keys a lut buffer save adds to ``buffer_meta``."""
    fp = dataset_fingerprint(args)
    return {"dataset_fingerprint": fp, "dataset_fp_hash": fingerprint_hash(fp),
            "label_rows": int(label_rows),
            **{k: _fp_value(_arg(args, k)) for k in POOLABLE_KEYS},
            "learning_starts": int(_arg(args, "learning_starts", 0)),
            "num_envs": int(_arg(args, "num_envs", 0))}


def check_dataset_meta(meta: Dict, args, path: str) -> None:
    """Refuse a dataset this run must not train from: not a lut warm-up, no labels, or a
    fingerprint that differs from the run's on any key (the poolable axes are not in it)."""
    if meta.get("warmup_policy") != "lut":
        raise ValueError(f"{path}: warmup_policy {meta.get('warmup_policy')!r} != 'lut' (not a lut3 "
                         "teacher dataset)")
    if "dataset_fingerprint" not in meta:
        raise ValueError(f"{path}: no dataset_fingerprint in the buffer meta (saved before lutpre2: "
                         "re-collect it)")
    if int(meta.get("label_rows", 0)) <= 0:
        raise ValueError(f"{path}: no labelled rows")
    have = json.loads(json.dumps(meta["dataset_fingerprint"], default=str))
    want = json.loads(json.dumps(dataset_fingerprint(args), default=str))
    diff = sorted(k for k in set(have) | set(want) if have.get(k, "<absent>") != want.get(k, "<absent>"))
    if diff:
        raise ValueError(f"{path}: dataset fingerprint differs from this run on "
                         + ", ".join(f"{k} (dataset {have.get(k, '<absent>')!r}, run "
                                     f"{want.get(k, '<absent>')!r})" for k in diff))


def dataset_start_step(learning_starts: int, num_envs: int) -> int:
    """The step counter a dataset-fed run starts at: the warm-up's LAST iteration step (the loop
    adds num_envs first), so the first iteration is the first one >= learning_starts, as in a
    run that collected inline (lutpre1: 49,980 -> BC at 50,010)."""
    n = int(num_envs)
    return -(-int(learning_starts) // n) * n - n


def check_dataset_capacity(rows: Sequence[int], args, start_step: int) -> None:
    """The pooled rows plus the RL steps still to come must fit the buffer (nothing overwritten)."""
    n = int(args.num_envs)
    # the trainer's loop runs num_updates + 2 iterations of num_envs rows each
    rl = 0 if bool(_arg(args, "bc_only", False)) else \
        (max(0, int(args.total_timesteps) - int(start_step)) // n + 2) * n
    need = int(sum(int(r) for r in rows)) + rl
    if need > int(args.buffer_size):
        raise ValueError(f"--buffer_size {int(args.buffer_size)} < {need} (datasets {sum(rows)} rows + "
                         f"{rl} RL steps): raise it to >= {need}")


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

    def teacher_stats(self) -> Dict[str, float]:
        """lutpre2 warm-up logging: rms of the AR(1) offset and mean |lut3 target| (deg)."""
        return {"noise_rms_deg": float(np.sqrt(np.mean(np.square(self.noise.value)))),
                "target_abs_mean_deg": float(np.mean(np.abs(self.teacher_target())))}

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


# --------------------------------------------------------------------------- lutpre3: BC anchor
def anchored_actor_loss(min_qf_pi, log_pi, alpha, a_pi, a_bc, real_mask, *, coef: float, qnorm: bool,
                        deg_per_unit: float):
    """TD3+BC-style actor loss: ``(alpha log_pi - Q).mean() / qnorm + coef * anchor`` with
    ``anchor`` the mean over REAL turbines of ``(a_pi - a_bc)^2`` (squashed action units) and
    ``qnorm`` the detached mean |Q| of the batch (``qnorm=True``; else 1), so beta keeps a fixed
    weight relative to the Q term whatever the critics' scale (TD3+BC's lambda 2.5 = beta 0.4).

    ``a_pi`` (B, T, A) = the actor's squashed MEAN action (differentiable), ``a_bc`` the frozen bc.pt
    actor's (no grad), ``real_mask`` (B, T) True on real turbines. Returns ``(loss, stats)`` with
    stats ``anchor``, ``q_norm`` (floats) and ``anchor_rms_deg`` = sqrt(anchor) * deg_per_unit."""
    import torch

    q = min_qf_pi.float()
    if qnorm:
        q_norm = q.abs().mean().detach().clamp_min(1e-3)
    else:
        q_norm = torch.ones((), dtype=q.dtype, device=q.device)
    real = real_mask.unsqueeze(-1).to(a_pi.dtype)
    n = real.sum().clamp(min=1.0)
    anchor = ((a_pi.float() - a_bc.detach().float()).pow(2) * real).sum() / n
    loss = (alpha * log_pi.float() - q).mean() / q_norm + float(coef) * anchor
    a = anchor.detach()     # stats stay device tensors: the trainer accumulates them without a host sync
    return loss, {"anchor": a, "q_norm": q_norm.detach(), "anchor_rms_deg": a.sqrt() * float(deg_per_unit)}


# --------------------------------------------------------------------------- BC
WD_BANDS = ("wd_ge265", "wd_ramp", "wd_le240")      # the 270 hold / the turn / the 235 end
TURB_ROWS = ("front", "mid", "back")


def holdout_groups(wd, x, real) -> Dict[str, np.ndarray]:
    """BC holdout subsets. Per row (B,): wd band of the stored wd (>= 265 / between / <= 240).
    Per token (B, T): the x-tercile among the row's real tokens (front / mid / back = the 3x3's
    columns for wind from the west half-plane, every lutpre wd; token order is shuffled, so it is
    ranked from the stored raw positions)."""
    wd = np.asarray(wd, dtype=float)
    x = np.asarray(x, dtype=float)
    real = np.asarray(real, dtype=bool)
    out = {"wd_ge265": wd >= 265.0, "wd_le240": wd <= 240.0}
    out["wd_ramp"] = ~(out["wd_ge265"] | out["wd_le240"])
    xs = np.where(real, x, np.inf)
    rank = np.argsort(np.argsort(xs, axis=1, kind="stable"), axis=1)
    n_real = np.maximum(real.sum(axis=1, keepdims=True), 1)
    tercile = np.minimum((3 * rank) // n_real, 2)
    for k, name in enumerate(TURB_ROWS):
        out[name] = real & (tercile == k)
    return out


def run_bc(actor, rb, *, steps: int, lr: float, batch_size: int, std: float, std_coef: float,
           deg_per_unit: float, device, holdout_frac: float = 0.05, obs_normalizer=None,
           writer=None, log_every: int = 250, grad_clip: float = 1.0, seed: int = 0) -> Dict[str, float]:
    """Fit ``actor`` to the buffer's labelled rows (in place, own Adam).

    loss = MSE(tanh(mean) * scale + bias, label) + std_coef * (log_std - log std)^2, both masked
    over real turbines. The holdout is the LAST ``holdout_frac`` of the labelled rows in insertion
    order (contiguous in time: neighbouring rows of one episode are near-duplicates, a random split
    would be optimistic; with pooled datasets: the tail of the last one). Every actor parameter
    trains, shared profile encoders included (the critics are still untrained at this point).

    Logging (every ``log_every`` steps and the last): ``bc/step`` = k first (W&B charts bc/* against
    it), loss, train / holdout rms (deg) and the holdout rms per wd band and turbine row
    (``holdout_groups``)."""
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

    def _forward(batch):
        obs = batch["observations"]
        if obs_normalizer is not None:
            obs = obs_normalizer.normalize(obs)
        infl = batch.get("influence") if getattr(actor, "use_influence", True) else None
        mean, log_std, _ = actor(obs, batch["positions"], batch["attention_mask"],
                                 recep_profile=batch.get("receptivity"), influence_profile=infl,
                                 cond=batch.get("params"))
        pred = torch.tanh(mean) * actor.action_scale + actor.action_bias_val
        return pred, log_std

    def _loss(batch):
        pred, log_std = _forward(batch)
        real = (~batch["attention_mask"]).unsqueeze(-1).expand_as(pred).float()
        n = real.sum().clamp(min=1.0)
        lab = torch.nan_to_num(batch["labels"])
        mse = ((pred - lab).pow(2) * real).sum() / n
        std_err = ((log_std - log_std_target).pow(2) * real).sum() / n
        return mse, std_err

    def _holdout() -> Dict[str, float]:
        """rms (deg) over the holdout, overall and per subset; NaN for an empty subset."""
        keys = ("all",) + WD_BANDS + TURB_ROWS
        sq = dict.fromkeys(keys, 0.0)
        cnt = dict.fromkeys(keys, 0.0)
        if hold_idx.size:
            with torch.no_grad():
                for k in range(0, hold_idx.size, int(batch_size)):
                    rows = hold_idx[k:k + int(batch_size)]
                    b = rb.sample_at(rows)
                    pred, _ = _forward(b)
                    err = (pred - torch.nan_to_num(b["labels"])).pow(2).sum(-1).cpu().numpy()   # (B, T)
                    real = ~b["attention_mask"].cpu().numpy()
                    a_dim = b["labels"].shape[-1]
                    g = holdout_groups(rb._wind_directions[rows], rb._raw_positions[rows, :, 0], real)
                    masks = {"all": real}
                    masks.update({w: real & g[w][:, None] for w in WD_BANDS})
                    masks.update({r: g[r] for r in TURB_ROWS})
                    for key, m in masks.items():
                        sq[key] += float(err[m].sum())
                        cnt[key] += float(m.sum()) * a_dim
        return {key: (math.sqrt(sq[key] / cnt[key]) * float(deg_per_unit) if cnt[key] > 0 else float("nan"))
                for key in keys}

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
            h = _holdout()
            writer.add_scalar("bc/step", k, k)
            writer.add_scalar("bc/loss", loss_v, k)
            writer.add_scalar("bc/rms_deg_train", math.sqrt(run_mse) * float(deg_per_unit), k)
            writer.add_scalar("bc/rms_deg_holdout", h["all"], k)
            for key in WD_BANDS + TURB_ROWS:
                writer.add_scalar(f"bc/rms_deg_holdout_{key}", h[key], k)
    h = _holdout()
    stats = {
        "n_labelled": int(idx.size), "n_holdout": int(hold_idx.size), "steps": int(steps),
        "loss": loss_v,
        "rms_deg_train": (math.sqrt(run_mse) * float(deg_per_unit)) if run_mse is not None else float("nan"),
        "rms_deg_holdout": h["all"],
        **{f"rms_deg_holdout_{key}": h[key] for key in WD_BANDS + TURB_ROWS},
    }
    return stats
