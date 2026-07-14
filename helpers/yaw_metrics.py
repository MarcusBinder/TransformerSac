"""Yaw movement diagnostics shared by training and evaluation.

Position statistics (mean |yaw|, max |yaw|) cannot distinguish a steady
offset from a limit cycle: both produce flat curves. The tracker below
turns consecutive per-step yaw arrays into movement statistics (travel,
reversal rate, duty cycle, slew saturation) so oscillation is visible on
the W&B dashboard. The same class is fed the normalized action stream to
separate policy jitter from actuator-filtered yaw movement.
"""

from typing import Optional

import numpy as np

# Ignore sub-deadband diffs (float noise) for duty cycle / reversals.
YAW_DEADBAND = 0.05


class YawTravelTracker:
    """Accumulates yaw movement stats from consecutive per-step yaw arrays.

    Feed one (num_envs, n_turbines) array per env step via update(); read
    windowed aggregates with compute_and_reset() (scalars, all envs pooled)
    or compute_per_env_and_reset() (one value per env, used by the
    evaluator to attribute episodes to layouts). Counters reset on read;
    diff state carries over so windows stay contiguous.

    A diff is MOVE when |delta| > deadband; sub-deadband jitter counts
    toward travel but not duty cycle or reversals. Rows flagged done are
    NaN-invalidated so a post-autoreset yaw (AutoresetMode.SAME_STEP) is
    never diffed against the pre-reset yaw. A change in array shape
    (layout switch changes n_turbines) drops all diff state and starts
    fresh.
    """

    def __init__(self, deadband: float = YAW_DEADBAND,
                 slew_limit: Optional[float] = None):
        self.deadband = deadband
        self.slew_limit = slew_limit
        self._prev = None        # last yaw array
        self._prev_delta = None  # last per-step diff, NaN where invalid
        self._counters = None

    def _reset_counters(self, num_envs: int):
        self._counters = {
            "sum_abs": np.zeros(num_envs),   # sum |delta| over valid diffs
            "n_delta": np.zeros(num_envs),   # count of valid diffs
            "n_move": np.zeros(num_envs),    # diffs with |delta| > deadband
            "n_sat": np.zeros(num_envs),     # MOVE diffs at >= 0.95 * slew_limit
            "n_pair": np.zeros(num_envs),    # consecutive valid diff pairs
            "n_rev": np.zeros(num_envs),     # pairs with opposing MOVE signs
        }

    def update(self, yaw, done=None):
        """Ingest one step of yaw angles; yaw: (num_envs, n_turbines)."""
        yaw = np.asarray(yaw, dtype=float)
        if yaw.ndim == 1:
            yaw = yaw[None, :]

        if self._prev is None or self._prev.shape != yaw.shape:
            self._prev = yaw.copy()
            self._prev_delta = np.full(yaw.shape, np.nan)
            if self._counters is None or len(self._counters["n_delta"]) != yaw.shape[0]:
                self._reset_counters(yaw.shape[0])
            return

        delta = yaw - self._prev
        if done is not None:
            done = np.asarray(done, dtype=bool).reshape(-1)
            delta[done] = np.nan

        valid = np.isfinite(delta)
        abs_d = np.where(valid, np.abs(delta), 0.0)
        move = valid & (abs_d > self.deadband)

        c = self._counters
        c["sum_abs"] += abs_d.sum(axis=1)
        c["n_delta"] += valid.sum(axis=1)
        c["n_move"] += move.sum(axis=1)
        if self.slew_limit is not None:
            c["n_sat"] += (move & (abs_d >= 0.95 * self.slew_limit)).sum(axis=1)

        prev_valid = np.isfinite(self._prev_delta)
        prev_f = np.where(prev_valid, self._prev_delta, 0.0)
        prev_move = prev_valid & (np.abs(prev_f) > self.deadband)
        pair = valid & prev_valid
        delta_f = np.where(valid, delta, 0.0)
        rev = pair & move & prev_move & (np.sign(delta_f) == -np.sign(prev_f))
        c["n_pair"] += pair.sum(axis=1)
        c["n_rev"] += rev.sum(axis=1)

        self._prev = yaw.copy()
        self._prev_delta = delta

    def compute_and_reset(self) -> dict:
        """Pooled scalar stats over all (env, turbine, step) since last read."""
        c = self._counters
        if c is None or c["n_delta"].sum() == 0:
            return {}
        n_delta = c["n_delta"].sum()
        out = {
            "travel_deg_per_step": float(c["sum_abs"].sum() / n_delta),
            "duty_cycle": float(c["n_move"].sum() / n_delta),
        }
        n_pair = c["n_pair"].sum()
        if n_pair > 0:
            out["reversal_rate"] = float(c["n_rev"].sum() / n_pair)
        if self.slew_limit is not None:
            n_move = c["n_move"].sum()
            out["slew_saturation_frac"] = (
                float(c["n_sat"].sum() / n_move) if n_move > 0 else 0.0
            )
        self._reset_counters(len(c["n_delta"]))
        return out

    def compute_per_env_and_reset(self) -> dict:
        """Per-env stats (arrays of shape (num_envs,)); NaN where undefined."""
        c = self._counters
        if c is None or c["n_delta"].sum() == 0:
            return {}
        with np.errstate(invalid="ignore", divide="ignore"):
            out = {
                "travel_deg_per_step": c["sum_abs"] / c["n_delta"],
                "duty_cycle": c["n_move"] / c["n_delta"],
                "reversal_rate": c["n_rev"] / c["n_pair"],
            }
            if self.slew_limit is not None:
                out["slew_saturation_frac"] = c["n_sat"] / c["n_move"]
        self._reset_counters(len(c["n_delta"]))
        return out
