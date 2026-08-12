"""
Observation-encoding wrapper for the change_wd_4 bake-off (see OBS_SCALING.md).

The finding: every ws feature reaching the agent is affine-scaled from a
hardcoded 0-30 m/s while the data lives in ~6-14 m/s, so the signal occupies
~13% of the [-1,1] axis (and the power feature saturates above ~12 m/s). This
wrapper re-encodes the ws columns of the PER-TURBINE observation with more
resolution where the data actually is.

Placement: inside combined_wrapper, AFTER PerTurbineObservationWrapper /
EnhancedPerTurbineWrapper and BEFORE TransformReward. It must be inside
TransformReward because MultiLayoutEnv._get_obs_dim_from_env walks the wrapper
chain outermost-first looking for `_obs_dim_per_turbine`
(helpers/multi_layout_env.py:201-216) — setting that attribute here is the
whole integration contract: networks, replay buffer and evaluators all size
themselves off it. It runs PRE-padding (MultiLayoutEnv pads later), so
cross-turbine features (reldef) never see pad rows.

Layout contract: appended features go at the END of the per-turbine vector and
replace-mode features overwrite the ws columns 1:1, so the base indices
(0..F_in-1, e.g. 0..11 under hard_2 H=3) keep their meaning in every mode.
All emitted features lie in [-1,1] by construction.

Mode registry: each mode is a factory returning (fn, replace) where fn is a
pure numpy map (n_turb, H) PHYSICAL ws [m/s] -> (n_turb, K) features.
  replace=False: obs_out = [obs_base, feats]                    (F_in + K)
  replace=True:  feats[:, :H] overwrite the ws columns in place,
                 feats[:, H:] (if any) are appended at the end   (F_in + K - H)
"""

import json
import os
from typing import Callable, Optional, Tuple

import gymnasium as gym
import numpy as np

# Written by TransformerSac/calibrate_ws_cdf.py (fixed seed, committed to git so
# every arm/node warps through the IDENTICAL frozen CDF).
_CDF_KNOTS_PATH = os.path.join(os.path.dirname(__file__), "ws_cdf_knots.json")


# =============================================================================
# MODE FACTORIES  (pure numpy; ws is PHYSICAL m/s, shape (n_turb, H))
# =============================================================================

def _make_rbf(n_centers: int = 8, c_lo: float = 5.0, c_hi: float = 15.0,
              sigma: float = 1.4) -> Tuple[Callable, bool]:
    """K Gaussian bumps per ws slot — soft one-hot over the live ws band."""
    centers = np.linspace(c_lo, c_hi, n_centers).astype(np.float32)

    def fn(ws):
        # (n, H, 1) - (K,) -> (n, H, K); bumps in (0, 1] subset of [-1, 1]
        z = (ws[:, :, None] - centers) / sigma
        return np.exp(-0.5 * z * z).reshape(ws.shape[0], -1)

    return fn, False


def _make_fourier(bands: tuple = (0.5, 1.0, 2.0, 4.0), lo: float = 4.0,
                  hi: float = 16.0) -> Tuple[Callable, bool]:
    """cos/sin features at multiple frequencies on ws mapped to [0,1] via [lo,hi]."""
    bands_arr = np.asarray(bands, dtype=np.float32)

    def fn(ws):
        t = np.clip((ws - lo) / (hi - lo), 0.0, 1.0)          # (n, H)
        ang = 2.0 * np.pi * t[:, :, None] * bands_arr          # (n, H, B)
        return np.concatenate([np.cos(ang), np.sin(ang)], axis=-1).reshape(ws.shape[0], -1)

    return fn, False


def _make_pyramid(zooms: tuple = ((0.0, 30.0), (6.0, 14.0), (9.0, 12.0))
                  ) -> Tuple[Callable, bool]:
    """Multi-resolution clipped affine zooms. Zoom 0 (the coarse, full-range one)
    overwrites the ws columns in place; the narrow zooms are appended, so base
    indices keep meaning while resolution is added where the data lives."""
    zooms_arr = [(float(a), float(b)) for a, b in zooms]

    def fn(ws):
        outs = [np.clip(2.0 * (ws - a) / (b - a) - 1.0, -1.0, 1.0)
                for a, b in zooms_arr]
        return np.concatenate(outs, axis=-1)                   # (n, H * n_zooms)

    return fn, True


def _make_cdf() -> Tuple[Callable, bool]:
    """Frozen empirical-CDF warp: ws -> 2*CDF(ws)-1, piecewise-linear between
    knots calibrated on the arm-0 training distribution. Equalizes resolution
    per unit of DATA MASS instead of per m/s."""
    if not os.path.exists(_CDF_KNOTS_PATH):
        raise FileNotFoundError(
            f"obs_encoding mode 'cdf' needs {_CDF_KNOTS_PATH}, which is written "
            f"by the calibration script. Run: uv run python "
            f"TransformerSac/calibrate_ws_cdf.py (fixed seed, commit the JSON)."
        )
    with open(_CDF_KNOTS_PATH) as f:
        cal = json.load(f)
    knots = np.asarray(cal["knots"], dtype=np.float64)
    probs = np.asarray(cal["probs"], dtype=np.float64)

    def fn(ws):
        # np.interp clamps outside the knot range -> CDF 0/1 -> feature -1/+1
        return (2.0 * np.interp(ws, knots, probs) - 1.0).astype(np.float32)

    return fn, True


def _make_reldef() -> Tuple[Callable, bool]:
    """Relative deficit: ws_i / max_j(ws_j) per history slot. A direct wake-depth
    readout that is invariant to the freestream level (the max turbine reads 1.0)."""

    def fn(ws):
        farm_max = ws.max(axis=0, keepdims=True)               # (1, H) — pre-padding!
        return ws / np.maximum(farm_max, 1e-6)

    return fn, False


def _make_pcurve(turbine=None, ws_grid_max: float = 30.0,
                 n_grid: int = 301) -> Tuple[Callable, bool]:
    """Physics-informed features: P(ws)/P_rated and Ct(ws) via lookup on the
    turbine's own curves. Places the rated-power knee — the physically decisive
    ws landmark — explicitly into the observation."""
    if turbine is None:
        raise ValueError("obs_encoding mode 'pcurve' requires the turbine object")
    grid = np.linspace(0.0, ws_grid_max, n_grid)
    p_tab = np.asarray(turbine.power(grid), dtype=np.float64)
    ct_tab = np.asarray(turbine.ct(grid), dtype=np.float64)
    p_rated = float(p_tab.max())

    def fn(ws):
        p = np.interp(ws, grid, p_tab) / p_rated
        ct = np.interp(ws, grid, ct_tab)
        return np.concatenate([p, ct], axis=-1).astype(np.float32)  # (n, 2H)

    return fn, False


ENCODING_MODES = {
    "rbf": _make_rbf,
    "pyramid": _make_pyramid,
    "cdf": _make_cdf,
    "fourier": _make_fourier,
    "reldef": _make_reldef,
    "pcurve": _make_pcurve,
}


# =============================================================================
# WRAPPER
# =============================================================================

class ObsEncodingWrapper(gym.ObservationWrapper):
    """Re-encode the ws columns of the per-turbine observation (see module doc)."""

    def __init__(self, env: gym.Env, mode: str, turbine=None, **mode_kwargs):
        super().__init__(env)
        if mode not in ENCODING_MODES:
            raise ValueError(f"Unknown obs_encoding mode '{mode}'. "
                             f"Valid: {sorted(ENCODING_MODES)}")
        self.mode = mode

        # pcurve is the only mode that consumes the turbine object; passing it to
        # the others would blow up their (deliberately strict) signatures.
        factory = ENCODING_MODES[mode]
        if mode == "pcurve":
            mode_kwargs = {"turbine": turbine, **mode_kwargs}
        self._encode, self._replace = factory(**mode_kwargs)

        # Detect ws columns + the env's ACTUAL scaling range (don't hardcode 0/30
        # — stays correct if combined with --ws_scaling_*).
        self._ws_indices, self._ws_min, self._ws_max = self._detect_ws_indices()
        self._n_ws = len(self._ws_indices)
        # ws columns are contiguous ([probes..., ws..., wd..., ...]); the split
        # in observation() relies on that.
        assert self._ws_indices == list(range(self._ws_indices[0],
                                              self._ws_indices[0] + self._n_ws)), \
            f"ws indices not contiguous: {self._ws_indices}"

        f_in = env.observation_space.shape[-1]
        n_turb = env.observation_space.shape[0]
        k = self._encode(np.zeros((1, self._n_ws), dtype=np.float32)).shape[-1]
        if self._replace:
            assert k >= self._n_ws, \
                f"replace-mode encoding must emit >= {self._n_ws} features, got {k}"
            f_out = f_in + k - self._n_ws
        else:
            f_out = f_in + k

        # THE integration contract with MultiLayoutEnv._get_obs_dim_from_env.
        self._obs_dim_per_turbine = int(f_out)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_turb, f_out), dtype=np.float32,
        )

        print(f"ObsEncodingWrapper: mode={mode} ({'replace' if self._replace else 'append'}), "
              f"ws indices {self._ws_indices} scaled from "
              f"[{self._ws_min}, {self._ws_max}], obs dim {f_in} -> {f_out}")

    def _get_base_env(self):
        """Unwrap to the base WindFarmEnv (EnhancedPerTurbineWrapper pattern)."""
        env = self.env
        while hasattr(env, 'env'):
            env = env.env
        return env

    def _detect_ws_indices(self) -> Tuple[list, float, float]:
        """Locate the ws columns by reading the TurbMes configuration directly
        (same approach as EnhancedPerTurbineWrapper._detect_wd_indices)."""
        base_env = self._get_base_env()
        fm = base_env.farm_measurements
        turb_mes = fm.turb_mes[0]  # all turbines share one structure

        if not fm.turb_ws:
            raise ValueError("obs_encoding requires turb_ws measurements, but the "
                             "env config has mes_level.turb_ws=False")

        offset = 0
        if hasattr(turb_mes, 'n_probes') and turb_mes.n_probes > 0:
            offset += turb_mes.n_probes

        ws_mes = turb_mes.ws
        n_ws = (1 if ws_mes.current else 0) + (ws_mes.history_N if ws_mes.rolling_mean else 0)
        if n_ws == 0:
            raise ValueError("obs_encoding: env reports zero ws features")

        return (list(range(offset, offset + n_ws)),
                float(turb_mes.ws_min), float(turb_mes.ws_max))

    def observation(self, obs: np.ndarray) -> np.ndarray:
        i0, i1 = self._ws_indices[0], self._ws_indices[0] + self._n_ws
        # Invert the env's affine scaling back to physical m/s
        ws_phys = (obs[:, i0:i1] + 1.0) / 2.0 * (self._ws_max - self._ws_min) + self._ws_min
        feats = self._encode(ws_phys.astype(np.float32))

        if self._replace:
            parts = [obs[:, :i0], feats[:, :self._n_ws], obs[:, i1:],
                     feats[:, self._n_ws:]]
        else:
            parts = [obs, feats]
        return np.concatenate(parts, axis=-1).astype(np.float32)
