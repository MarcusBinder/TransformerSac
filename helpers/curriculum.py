"""Stage-17 oracle-LUT reward-shaping curriculum (``--curriculum lut``).

Why: the Stage-15/16 Richards arms never learned the negative yaw lobe on the
slow 270 -> 235 turn (yaw sign collapse). Instead of hoping the TD signal finds
it, the TRAINING reward is blended with a similarity term to the PyWake
serial-refine LUT (the open-loop optimum at the env's TRUE wd):

    r = w * r_env + (1 - w) * sim,
    sim = sim_scale * (1 - min(err / sim_norm_deg, 1)),
    err = mean_i |yaw_i - lut_i(wd_true)|   (physical turbine order),

with ``w`` ramped linearly from ``w0`` to ``w1`` over env steps ``start..end``
(``cur_lut_ramp``: 0 -> 1, pure RL afterwards; ``cur_lut_resid``: 0 -> 0.8, a
residual LUT term to the end). ``w == 1`` is bitwise ``r_env``.

Reference frame: the table is built ONCE in the main process (``build_lut``,
sigma 0, yaw_max 45, sign ``farm``) and handed to every env closure as plain
numpy arrays, so py_wake never enters the 30 workers. ``reference_yaw(wd)``
holds the ``ref_hold_center`` row over ``[center - ref_hold_deg, center]`` (a
walk's first 1.5 deg of departure from 270 keeps the 270 setpoint, as the
controllers hold), interpolates per turbine elsewhere and clamps to the grid.

Yaw is booked relative to the TRUE wind frame inside windgym, so ``info["yaw
angles agent"]`` (= ``fs.windTurbines.yaw``) and the LUT share a frame.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import gymnasium as gym
import numpy as np

CURRICULUM_KINDS = ("none", "lut")
LUT_SIGNS = ("farm", "free", "pos", "neg")


def curriculum_weight(step: int, w0: float, w1: float, s0: int, s1: int) -> float:
    """``w(step)``: ``w0`` before ``s0``, ``w1`` from ``s1`` on, linear in between
    (``s0 == s1`` = a step at ``s0``). ``s1 < s0`` raises."""
    s0, s1 = int(s0), int(s1)
    if s1 < s0:
        raise ValueError(f"curriculum_end ({s1}) must be >= curriculum_start ({s0})")
    w0, w1 = float(w0), float(w1)
    if step < s0:
        return w0
    if step >= s1:
        return w1
    return w0 + (w1 - w0) * (step - s0) / (s1 - s0)


def curriculum_push_needed(w_cur: float, w_pushed: Optional[float], w0: float, w1: float,
                           tol: float = 1e-3) -> bool:
    """Should the main loop push ``w_cur`` to the envs? First iteration always; then
    whenever the change exceeds ``tol`` OR ``w_cur`` is an endpoint (``w0`` / ``w1``)
    the envs have not received yet. ``global_step`` advances by ``num_envs`` (30), so
    a plain ``|dw| >= tol`` rule would leave the last push at 0.9994 and the
    ``w == 1`` bitwise-``r_env`` branch would never fire (review finding, 2026-10-21)."""
    if w_pushed is None:
        return True
    if w_cur == w_pushed:
        return False
    if abs(float(w_cur) - float(w_pushed)) >= tol:
        return True
    return float(w_cur) in (float(w0), float(w1))


def validate_curriculum_args(args, layouts: Sequence[Any], eval_layout_names: Sequence[str]) -> None:
    """Reject recipes the single-table reference cannot serve. No-op for ``none``."""
    kind = str(args.curriculum)
    if kind not in CURRICULUM_KINDS:
        raise ValueError(f"--curriculum must be one of {CURRICULUM_KINDS}, got {kind!r}")
    if kind == "none":
        return
    if len(layouts) != 1:
        raise ValueError(f"--curriculum lut builds ONE reference table: exactly one training layout "
                         f"is required (got {[getattr(l, 'name', l) for l in layouts]})")
    name = getattr(layouts[0], "name", layouts[0])
    if list(eval_layout_names) != [name]:
        raise ValueError(f"--curriculum lut: eval layouts {list(eval_layout_names)} must equal the "
                         f"training layout [{name!r}]")
    for key in ("curriculum_w0", "curriculum_w1"):
        v = float(getattr(args, key))
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"--{key} must be in [0, 1], got {v}")
    if int(args.curriculum_start) > int(args.curriculum_end):
        raise ValueError(f"--curriculum_start ({args.curriculum_start}) must be <= "
                         f"--curriculum_end ({args.curriculum_end})")
    if float(args.curriculum_sim_scale) < 0.0:
        raise ValueError(f"--curriculum_sim_scale must be >= 0, got {args.curriculum_sim_scale}")
    if float(args.curriculum_sim_norm_deg) <= 0.0:
        raise ValueError(f"--curriculum_sim_norm_deg must be > 0, got {args.curriculum_sim_norm_deg}")
    if float(args.curriculum_ref_hold_deg) < 0.0:
        raise ValueError(f"--curriculum_ref_hold_deg must be >= 0, got {args.curriculum_ref_hold_deg}")
    if str(args.curriculum_lut_sign) not in LUT_SIGNS:
        raise ValueError(f"--curriculum_lut_sign must be one of {LUT_SIGNS}, got {args.curriculum_lut_sign!r}")
    if float(getattr(args, "reward_scale", 1.0)) != 1.0:
        print(f"[curriculum] note: --reward_scale {args.reward_scale} multiplies the BLENDED reward "
              f"(the wrapper sits inside TransformReward); sim_scale is in pre-scale units")


def build_curriculum_lut(args, wind_turbine, layout, wd_grid: Optional[np.ndarray] = None,
                         verbose: bool = True) -> Dict[str, np.ndarray]:
    """Serial-refine LUT for ``layout`` at the curriculum's (ws, ti): numpy only."""
    from helpers.pywake_lut import DEFAULT_WD_GRID, build_lut
    wd_grid = DEFAULT_WD_GRID.copy() if wd_grid is None else np.asarray(wd_grid, dtype=float)
    x = np.asarray(layout.x_pos, dtype=float)
    y = np.asarray(layout.y_pos, dtype=float)
    grid, lut = build_lut(wind_turbine, x, y, ws=float(args.curriculum_lut_ws),
                          ti=float(args.curriculum_lut_ti), wd_grid=wd_grid, sigma_wd=0.0,
                          yaw_max=45.0, sign=str(args.curriculum_lut_sign), verbose=False)
    lut = np.ascontiguousarray(lut, dtype=float)
    if verbose:
        j = int(np.argmin(np.abs(grid - float(args.curriculum_ref_hold_center))))
        print(f"[pywake_lut] built {len(grid)}-bin table (sigma=0, sign={args.curriculum_lut_sign}, "
              f"ws={float(args.curriculum_lut_ws):g}, ti={float(args.curriculum_lut_ti):g}, yaw_max=45) "
              f"for layout {layout.name} ({len(x)} turbines); row {grid[j]:g}: "
              f"{np.array2string(lut[j], precision=1, suppress_small=True)}", flush=True)
    return {"wd_grid": np.ascontiguousarray(grid, dtype=float), "lut": lut}


class LutCurriculumRewardWrapper(gym.Wrapper):
    """Reward-only wrapper: ``r = w * r_env + (1 - w) * sim`` (see module doc)."""

    curriculum_wrapper = True   # marker for MultiLayoutEnv.set_curriculum_weight

    def __init__(self, env: gym.Env, wd_grid: np.ndarray, lut: np.ndarray, weight: float = 1.0,
                 sim_scale: float = 0.1, sim_norm_deg: float = 45.0, ref_hold_deg: float = 1.5,
                 ref_hold_center: float = 270.0):
        super().__init__(env)
        self._wd_grid = np.asarray(wd_grid, dtype=float)
        self._lut = np.asarray(lut, dtype=float)
        if self._lut.ndim != 2 or self._lut.shape[0] != self._wd_grid.shape[0]:
            raise ValueError(f"lut shape {self._lut.shape} does not match wd_grid {self._wd_grid.shape}")
        if np.any(np.diff(self._wd_grid) <= 0):
            raise ValueError("wd_grid must be strictly increasing")
        self._sim_scale = float(sim_scale)
        self._sim_norm = float(sim_norm_deg)
        self._hold = float(ref_hold_deg)
        self._center = float(ref_hold_center)
        self._weight = 1.0
        self.set_weight(weight)

    # -- weight ---------------------------------------------------------------
    @property
    def weight(self) -> float:
        return self._weight

    def set_weight(self, w: float) -> None:
        self._weight = float(min(max(float(w), 0.0), 1.0))

    # -- reference ------------------------------------------------------------
    def reference_yaw(self, wd: float) -> np.ndarray:
        """LUT setpoints (physical order) at TRUE ``wd``: hold-band -> center row,
        else per-turbine ``np.interp`` with clamping at the grid ends."""
        wd = float(wd)
        if self._hold > 0.0 and (self._center - self._hold) <= wd <= self._center:
            wd = self._center
        wd = float(min(max(wd, self._wd_grid[0]), self._wd_grid[-1]))
        if wd in self._wd_grid:
            return self._lut[int(np.searchsorted(self._wd_grid, wd))].copy()
        return np.array([np.interp(wd, self._wd_grid, self._lut[:, i])
                         for i in range(self._lut.shape[1])], dtype=float)

    # -- gym ------------------------------------------------------------------
    def step(self, action):
        obs, r_env, terminated, truncated, info = self.env.step(action)
        yaw = np.asarray(info["yaw angles agent"], dtype=float).reshape(-1)
        ref = self.reference_yaw(self.env.unwrapped.wd)
        err = float(np.mean(np.abs(yaw - ref)))
        sim = self._sim_scale * (1.0 - min(err / self._sim_norm, 1.0))
        w = self._weight
        r = r_env if w == 1.0 else w * r_env + (1.0 - w) * sim
        info = dict(info)
        info["reward_env"] = r_env
        info["curriculum_sim"] = sim
        info["curriculum_weight"] = w
        info["curriculum_ref_err_deg"] = err
        return obs, r, terminated, truncated, info
