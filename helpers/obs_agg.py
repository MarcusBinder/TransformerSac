# VENDORED FILE -- do not edit here.
#
# Verbatim copy of helpers/obs_agg.py from the `tracking` branch of
# TransformerSac @ 84d450a (the branch ../paper-derating tracks).
#
# The LES branch never grew this module, but the `change_wd_*.pt` checkpoints
# were trained on `tracking` and carry obs_agg='raw15span' / obs_agg_len=60,
# which RunPretrainedAgentLES_emlhversionTEST.py resolves through AGG_MODES.
# Copied rather than cherry-picked because the two branches diverged at b0a8117
# in opposite directions (LES: HAWC2 + Precursor + six-way obs scaling;
# tracking: wd-schedules, wd-estimation, derating, obs aggregation).
#
# The file is self-contained (dataclasses / typing / gymnasium / numpy only).
# Keep it byte-identical below this header so it can be diffed against
# `tracking` later; port fixes from there rather than patching in place.
"""
Observation-aggregate wrapper for the LES-3x3 Stage-4 sweep (--obs_agg).

WHY. What the agent sees today, per turbine and per quantity (ws / wd / yaw /
power), is the last three raw 10-s samples [t, t-1, t-2] (window_length=1,
history_N=history_length=3): 30 s of memory, no smoothing, no dispersion, no
trend, and wd affine-scaled over 0-360. Stages 2-3 of the LES-3x3 campaign left
the in-ramp window at ~1.0 for every training-wd recipe, so this wrapper swaps
the per-turbine vector for one of ~10 alternative aggregation schemes computed
over a LONGER raw buffer (L env steps, default 30 = 300 s).

CONTRACT.
  * Placement: inside combined_wrapper, after PerTurbineObservationWrapper and
    before TransformReward (the ObsEncodingWrapper slot). It sets
    `_obs_dim_per_turbine` and a 2-D observation_space, which is what
    MultiLayoutEnv._get_obs_dim_from_env reads (helpers/multi_layout_env.py).
    It runs PRE-shuffle/PRE-padding, so cross-turbine features (spatial_rel)
    see the true layout order and never a pad row.
  * Stateless: the trainer / eval_wd enlarge the env's measurement deques to
    max(history_length, obs_agg_len) (history_N stays 3), so the env's reset
    burn-in (steps_on_reset = max(power_avg, max_hist)) fills all L samples
    before the first observation, and every observation() re-reads
    env.unwrapped.farm_measurements.turb_mes[i].{ws,wd,yaw,power}.measurements
    (chronological, physical units, one scalar per entry). FarmMes is rebuilt
    on every reset, so deques are never cached -- only the scale constants
    and x_pos (front turbine = argmin x, same as the estimator's
    consensus=front) are read at __init__.
  * Output: (n_turb, 4K) float32 in [-1, 1], block order [ws | wd | yaw |
    power], each block K features. It REPLACES the base per-turbine vector,
    so it asserts n_farm == 0 (no farm tail / DEL column) and that the inner
    width is exactly 4*history_N.
  * Truncation edge: WindFarmEnv.step frees farm_measurements when its own
    time limit fires (cleanup_on_time_limit) -- the observation of that
    terminal step is then the previously emitted one.

CIRCULAR wd. Every wd reduction works on deviations d = wrap180(wd - c) from
the per-turbine circular mean c of the buffer; level-type outputs are mapped
back as wrap360(c + stat(d)) and env-scaled 0-360 exactly like today
(circ_mean_deg -> float32 -> affine -> clip); dispersion/slope outputs stay in
deviation degrees. EMA on wd runs on unit vectors. Non-wd quantities are
linear.

SCALES (fixed, physical; no --obs_norm by decision). Feature kinds:
  level : env's own (min, max) per quantity -- ws 0/30 (ws_scaling_*), wd
          0/360, yaw the preset band, power 0/Pmax -- the same affine as the
          base obs.
  disp  : std / |F_k| in [0, D_q]           (ddof=0; m == 1 -> 0)
  sdev  : signed deviations / slope*window in [-2 D_q, 2 D_q]
  amp   : spectral |F_k| in [0, 2 D_q] (heavier-tailed than std: a single
          step change of size s in the window gives |F_1| ~ 2s/pi)
  rel   : spatial_rel differences (t - t_front, t - farm mean), fixed
          half-ranges REL_HALF: ws +-6 m/s (7D wake deficits at 9.5 m/s are
          3-4 m/s), wd +-2 D_wd, yaw the level half-range (yaw_max -
          yaw_min)/2 = +-45 deg on les_recipe (front-vs-waked yaw offsets of
          25-40 deg are normal), power +-0.75 Pmax.
  D_q   : ws 2.0 m/s, wd 15 deg, yaw 15 deg, power 0.25*Pmax.
Sanity-checked by TransformerSac/obs_agg_probe.py (p1..p99 of every feature
of every mode inside (-1, 1) with spread > 0.05 on an les_3x3 dynamiks env
under a slew-then-hold yaw policy). Probe-driven adjustments (2026-08-18,
then frozen): D_wd 10 -> 15 (the 35 deg/600 s eval ramp gives std 10 deg over
600 s and slope*L 17.5 deg over 300 s -- both saturated at 10), spectral got
its own 2 D_q range, and rel ws/power got the wider fixed half-ranges above.
Remaining accepted tails: yaw slope*window (a full-rate 5 deg/sim-step slew
exceeds +-30 deg per 60 s by construction) and the p99 of the ws/power
dispersions at waked turbines while an upstream turbine slews.

SHORT BUFFERS (m < L; only if the env ever hands back fewer samples, e.g. the
raw3 seam test with L=3): stats over the available m entries; raw15/raw3 pad
by repeating the oldest sample; slopes need m >= 2 else 0; spectral zero-pads
to L (the FFT length is always L so the periods 300/150/100 s are fixed).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import gymnasium as gym
import numpy as np

QUANTITIES = ("ws", "wd", "yaw", "power")

# Dispersion constants D_q (physical units; power as a fraction of Pmax).
DISPERSION = {"ws": 2.0, "wd": 15.0, "yaw": 15.0, "power": 0.25}
# spatial_rel half-ranges (physical; power as a fraction of Pmax; yaw = None ->
# the level half-range of the env's yaw band).
REL_HALF = {"ws": 6.0, "wd": 2.0 * DISPERSION["wd"], "yaw": None, "power": 0.75}

# EMA time constants [s] and trend short window [env steps]
EMA_TAUS = (30.0, 100.0, 300.0)
TREND_SHORT = 6
N_SPECTRAL = 3
N_RAW = 15


# =============================================================================
# circular helpers
# =============================================================================

def wrap180(x):
    """Wrap angles [deg] into [-180, 180)."""
    return (np.asarray(x, dtype=np.float64) + 180.0) % 360.0 - 180.0


def wrap360(x):
    """Wrap angles [deg] into [0, 360) (mirrors utils.circ_mean_deg's guard)."""
    res = np.asarray(x, dtype=np.float64) % 360.0
    return np.where(res >= 360.0, 0.0, res)


def circmean_deg(x, axis=0):
    """Circular mean along `axis`, in [0, 360)."""
    rad = np.deg2rad(np.asarray(x, dtype=np.float64))
    s = np.mean(np.sin(rad), axis=axis)
    c = np.mean(np.cos(rad), axis=axis)
    return wrap360(np.rad2deg(np.arctan2(s, c)))


# =============================================================================
# mode registry
#   fn(buf, ctx) -> (n_turb, K) PHYSICAL features.
#   buf: (m, n_turb) chronological. For wd it is the DEVIATION buffer d
#        (linear, degrees); ctx["raw"] is the raw wd buffer and ctx["center"]
#        the (n_turb,) circular mean. For linear quantities raw is buf and
#        center is zeros, so `center + stat(buf)` is the level everywhere.
#   kinds: one of level|disp|amp|sdev|rel per feature -> scale table.
# =============================================================================

@dataclass(frozen=True)
class AggSpec:
    fn: Callable[[np.ndarray, dict], np.ndarray]
    kinds: tuple  # per-feature kind, len K
    doc: str

    @property
    def K(self) -> int:
        return len(self.kinds)


def _level(ctx, stat):
    """Map a per-turbine deviation statistic back to an absolute level."""
    lvl = ctx["center"] + stat
    return wrap360(lvl) if ctx["circular"] else lvl


def _ls_slope(buf: np.ndarray) -> np.ndarray:
    """Least-squares slope per env step of buf (m, n) along axis 0; 0 if m<2."""
    m = buf.shape[0]
    if m < 2:
        return np.zeros(buf.shape[1])
    t = np.arange(m, dtype=np.float64) - (m - 1) / 2.0
    return (t[:, None] * (buf - buf.mean(axis=0))).sum(axis=0) / (t * t).sum()


def _pad_oldest(buf: np.ndarray, n: int) -> np.ndarray:
    """Last n rows of buf; if fewer, repeat the oldest row in front."""
    if buf.shape[0] >= n:
        return buf[-n:]
    pad = np.repeat(buf[:1], n - buf.shape[0], axis=0)
    return np.concatenate([pad, buf], axis=0)


def _f_raw3(buf, ctx):
    tail = _pad_oldest(buf, 3)                        # oldest..newest
    return np.stack([_level(ctx, tail[-1 - i]) for i in range(3)], axis=1)


def _f_latest(buf, ctx):
    return _level(ctx, buf[-1])[:, None]


def _f_mean_std(buf, ctx):
    return np.stack([_level(ctx, buf[-1]),
                     _level(ctx, buf.mean(axis=0)),
                     buf.std(axis=0)], axis=1)


def _f_ema(buf, ctx):
    dt = float(ctx["dt_env"])
    outs = [_level(ctx, buf[-1])]
    for tau in EMA_TAUS:
        alpha = 1.0 - np.exp(-dt / tau)
        if ctx["circular"]:
            rad = np.deg2rad(ctx["raw"])
            u = np.stack([np.cos(rad), np.sin(rad)], axis=-1)   # (m, n, 2)
            e = u[0].copy()
            for k in range(1, u.shape[0]):
                e = (1.0 - alpha) * e + alpha * u[k]
            outs.append(wrap360(np.rad2deg(np.arctan2(e[:, 1], e[:, 0]))))
        else:
            e = buf[0].copy()
            for k in range(1, buf.shape[0]):
                e = (1.0 - alpha) * e + alpha * buf[k]
            outs.append(e)
    return np.stack(outs, axis=1)


def _f_trend(buf, ctx):
    short = buf[-TREND_SHORT:]
    return np.stack([_level(ctx, buf[-1]),
                     _level(ctx, short.mean(axis=0)),
                     _ls_slope(short) * TREND_SHORT,
                     _ls_slope(buf) * ctx["L"]], axis=1)


def _f_minmax(buf, ctx):
    return np.stack([_level(ctx, buf[-1]),
                     _level(ctx, buf.min(axis=0)),
                     _level(ctx, buf.max(axis=0)),
                     _level(ctx, buf.mean(axis=0))], axis=1)


def _f_quantiles(buf, ctx):
    q = np.quantile(buf, [0.1, 0.5, 0.9], axis=0)     # (3, n)
    return np.stack([_level(ctx, q[i]) for i in range(3)], axis=1)


def _f_raw15(buf, ctx):
    tail = _pad_oldest(buf, N_RAW)                    # (15, n) chronological
    return _level(ctx, tail).T                        # (n, 15) oldest..newest


def _f_raw15span(buf, ctx):
    """15 evenly-spaced samples SPANNING the whole L-buffer (newest included).

    Stage 6 (IEA22): with --obs_agg_len 60 the raw window covers 600 s at a
    40-s spacing while the obs width stays 15. A NEW registry name -- existing
    raw15 checkpoints (last 15 samples, 150 s) stay bit-reproducible. With
    L == 15 the linspace indices are 0..14, so it is index-identical to raw15
    (asserted in tests/test_obs_agg.py).
    """
    L = int(ctx["L"])
    tail = _pad_oldest(buf, L)                        # (L, n) chronological
    idx = np.round(np.linspace(0, L - 1, N_RAW)).astype(int)
    return _level(ctx, tail[idx]).T                   # (n, 15) oldest..newest


def _f_spectral(buf, ctx):
    L = int(ctx["L"])
    d = buf - buf.mean(axis=0)
    if d.shape[0] < L:
        d = np.concatenate([d, np.zeros((L - d.shape[0], d.shape[1]))], axis=0)
    F = np.fft.rfft(d[-L:], axis=0)                   # (L//2+1, n)
    amp = 2.0 * np.abs(F[1:N_SPECTRAL + 1]) / L       # amplitude of period L/k
    return np.concatenate([_level(ctx, buf[-1])[:, None],
                           _level(ctx, buf.mean(axis=0))[:, None],
                           amp.T], axis=1)


def _f_spatial_rel(buf, ctx):
    raw_t = ctx["raw"][-1]                            # (n,) newest, absolute
    front = raw_t[ctx["front"]]
    if ctx["circular"]:
        d_front = wrap180(raw_t - front)
        d_mean = wrap180(raw_t - circmean_deg(raw_t))
    else:
        d_front = raw_t - front
        d_mean = raw_t - raw_t.mean()
    return np.stack([_level(ctx, buf[-1]), d_front, d_mean], axis=1)


AGG_MODES: Dict[str, AggSpec] = {
    "raw3": AggSpec(_f_raw3, ("level",) * 3,
                    "[t, t-1, t-2] (seam test only; == base obs with L=3)"),
    "latest": AggSpec(_f_latest, ("level",), "[t]"),
    "mean_std": AggSpec(_f_mean_std, ("level", "level", "disp"),
                        "[t, mean_L, std_L]"),
    "ema": AggSpec(_f_ema, ("level",) * 4,
                   "[t, ema_30s, ema_100s, ema_300s]"),
    "trend": AggSpec(_f_trend, ("level", "level", "sdev", "sdev"),
                     "[t, mean_6, slope_6*6, slope_L*L]"),
    "minmax": AggSpec(_f_minmax, ("level",) * 4, "[t, min_L, max_L, mean_L]"),
    "quantiles": AggSpec(_f_quantiles, ("level",) * 3, "[q10_L, q50_L, q90_L]"),
    "raw15": AggSpec(_f_raw15, ("level",) * N_RAW,
                     "last 15 raw samples, chronological"),
    "raw15span": AggSpec(_f_raw15span, ("level",) * N_RAW,
                         "15 samples evenly spanning the L-buffer, "
                         "chronological (== raw15 when L == 15)"),
    "spectral": AggSpec(_f_spectral, ("level", "level") + ("amp",) * N_SPECTRAL,
                        "[t, mean_L, |F_1|, |F_2|, |F_3|] (periods L/1, L/2, L/3)"),
    "spatial_rel": AggSpec(_f_spatial_rel, ("level", "rel", "rel"),
                           "[t, t - t_front, t - mean_j t_j]"),
}


# =============================================================================
# pure reduction (shared by the wrapper and the unit tests)
# =============================================================================

def scale_table(kind: str, q: str, lo: float, hi: float) -> tuple:
    """(lo, hi) affine range for a feature of `kind` on quantity q whose level
    range is [lo, hi] (physical)."""
    if kind == "level":
        return (lo, hi)
    D = DISPERSION[q] * (hi if q == "power" else 1.0)
    if kind == "disp":
        return (0.0, D)
    if kind == "amp":
        return (0.0, 2.0 * D)
    if kind == "sdev":
        return (-2.0 * D, 2.0 * D)
    if kind == "rel":
        half = REL_HALF[q]
        if half is None:
            half = 0.5 * (hi - lo)
        elif q == "power":
            half = half * hi
        return (-half, half)
    raise ValueError(f"unknown feature kind {kind!r}")


def reduce_quantity(mode: str, q: str, buf: np.ndarray, *, L: int,
                    dt_env: float, front: int) -> np.ndarray:
    """PHYSICAL (n_turb, K) features of one quantity from a chronological
    (m, n_turb) buffer. wd is handled circularly (see module doc)."""
    spec = AGG_MODES[mode]
    buf = np.asarray(buf, dtype=np.float64)
    assert buf.ndim == 2 and buf.shape[0] >= 1, buf.shape
    circular = q == "wd"
    if circular:
        center = circmean_deg(buf, axis=0)
        dev = wrap180(buf - center[None, :])
    else:
        center = np.zeros(buf.shape[1])
        dev = buf
    ctx = {"circular": circular, "raw": buf, "center": center, "L": int(L),
           "dt_env": float(dt_env), "front": int(front), "q": q}
    out = np.asarray(spec.fn(dev, ctx), dtype=np.float64)
    assert out.shape == (buf.shape[1], spec.K), (mode, q, out.shape)
    return out


def scale_features(feats: np.ndarray, kinds: Sequence[str], q: str,
                   lo: float, hi: float) -> np.ndarray:
    """Affine-scale PHYSICAL (n, K) features to [-1, 1] and clip -- the env's
    float32 pipeline (float32 cast, then 2*(v-lo)/(hi-lo)-1, then clip)."""
    v = np.asarray(feats, dtype=np.float32)
    los = np.empty(len(kinds), dtype=np.float32)
    his = np.empty(len(kinds), dtype=np.float32)
    for k, kind in enumerate(kinds):
        los[k], his[k] = scale_table(kind, q, lo, hi)
    scaled = 2 * (v - los) / (his - los) - 1
    return np.clip(scaled, -1.0, 1.0).astype(np.float32)


# =============================================================================
# wrapper
# =============================================================================

class ObsAggWrapper(gym.ObservationWrapper):
    """Replace the per-turbine [ws|wd|yaw|power] x H raw-sample vector with an
    aggregate scheme over the env's L-long measurement deques (module doc)."""

    def __init__(self, env: gym.Env, mode: str, obs_agg_len: int, dt_env: float):
        super().__init__(env)
        if mode not in AGG_MODES:
            raise ValueError(f"Unknown obs_agg mode {mode!r}. "
                             f"Valid: {sorted(AGG_MODES)}")
        self.mode = mode
        self.agg_spec = AGG_MODES[mode]
        self.L = int(obs_agg_len)
        self.dt_env = float(dt_env)

        base = env.unwrapped
        fm = base.farm_measurements
        if fm is None:
            raise RuntimeError("ObsAggWrapper needs env.farm_measurements at "
                               "construction (WindFarmEnv builds it in __init__)")
        tm = fm.turb_mes[0]
        n_turb = int(base.n_turb)
        f_in = int(env.observation_space.shape[-1])
        assert len(env.observation_space.shape) == 2 and \
            env.observation_space.shape[0] == n_turb, env.observation_space
        H = int(tm.ws.history_N)
        n_farm = int(base.obs_var) - n_turb * int(base.get_obs_dim_per_turbine())
        if n_farm != 0:
            raise ValueError(f"ObsAggWrapper requires a purely per-turbine obs "
                             f"(n_farm == 0), got n_farm={n_farm}")
        if f_in != 4 * H:
            raise ValueError(
                f"ObsAggWrapper expects the inner per-turbine width to be "
                f"4*history_N = {4 * H} ([ws|wd|yaw|power] raw samples, no "
                f"probes/TI/derate/DEL columns), got {f_in}")
        for q in QUANTITIES:
            mes = getattr(tm, q)
            if mes.measurements.maxlen < self.L:
                raise ValueError(
                    f"ObsAggWrapper: {q} deque holds {mes.measurements.maxlen} "
                    f"< obs_agg_len={self.L}; set config[{q}_mes]"
                    f"['{q}_history_length'] >= obs_agg_len (the trainer / "
                    f"eval_wd do this when --obs_agg is set)")

        # Scale constants (frozen per env; the env's own affine ranges).
        self._ranges = {
            "ws": (float(tm.ws_min), float(tm.ws_max)),
            "wd": (float(tm.wd_min), float(tm.wd_max)),
            "yaw": (float(tm.yaw_min), float(tm.yaw_max)),
            "power": (0.0, float(tm.power_max)),
        }
        self._front = int(np.argmin(np.asarray(base.x_pos, dtype=float)))
        self._n_turb = n_turb
        self.K = self.agg_spec.K
        f_out = 4 * self.K

        # THE integration contract with MultiLayoutEnv._get_obs_dim_from_env.
        self._obs_dim_per_turbine = int(f_out)
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(n_turb, f_out), dtype=np.float32)
        self._last_obs = np.zeros((n_turb, f_out), dtype=np.float32)

        print(f"ObsAggWrapper: mode={mode} K={self.K} ({self.agg_spec.doc}), "
              f"L={self.L} steps x dt_env={self.dt_env:g}s, front turbine "
              f"{self._front}, obs dim per turbine {f_in} -> {f_out}")

    # ------------------------------------------------------------------
    def _buffers(self, fm) -> Optional[Dict[str, np.ndarray]]:
        """Chronological (m, n_turb) physical buffers, one per quantity."""
        out = {}
        for q in QUANTITIES:
            cols = []
            for tm in fm.turb_mes:
                dq = getattr(tm, q).measurements
                cols.append(np.asarray(list(dq), dtype=np.float64).reshape(len(dq)))
            m = min(len(c) for c in cols)
            if m == 0:
                return None
            out[q] = np.stack([c[-m:] for c in cols], axis=1)
        return out

    def observation(self, obs: np.ndarray) -> np.ndarray:
        fm = self.env.unwrapped.farm_measurements
        if fm is None:
            # Env freed its measurements on its own time limit (terminal step).
            return self._last_obs
        bufs = self._buffers(fm)
        if bufs is None:
            return self._last_obs
        blocks: List[np.ndarray] = []
        for q in QUANTITIES:
            feats = reduce_quantity(self.mode, q, bufs[q], L=self.L,
                                    dt_env=self.dt_env, front=self._front)
            lo, hi = self._ranges[q]
            blocks.append(scale_features(feats, self.agg_spec.kinds, q, lo, hi))
        out = np.concatenate(blocks, axis=1).astype(np.float32)
        self._last_obs = out
        return out
