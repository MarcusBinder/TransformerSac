"""Parameter vector e for parameter-conditioned policies (Stage 10: RMA / UP-OSI).

The DR wrappers draw, per episode, six per-turbine turbine-model parameters
(``WindGym.core.turbine_params``: yaw_exp cp_gain ct_gain tau_yaw delay_yaw
tau_power, PHYSICAL units) and seven farm-level DWM / Mann parameters from the
LES-calibrated posterior (k1 k2 d_particle hill_vortex_factor mann_L
mann_GAMMA mann_AE; Stage-10 checkpoints predate hill_vortex_factor and carry
the LEGACY six). This module turns them into one normalised
``(max_turbines, 6 + len(farm_keys))`` float32 vector (13 today, 12 legacy) in
TOKEN order (the actor's per-turbine tokens, i.e. after MultiLayoutEnv's
shuffle and padding):

* turbine columns ``2 (v - lo) / (hi - lo) - 1`` with the TRAINING range from
  ``--turb_dr`` (lo -> -1, hi -> +1); a key without a range is a constant 0
  column (the policy is not told about a parameter it never saw vary);
* farm columns z-scored by the posterior mean / std and clipped to ``+-clip``
  (``source="turbine"`` -> zeros); non-finite inputs (a plain-dynamiks setup
  with ``k1=None``) map to 0; a column whose posterior sd is numerically zero
  (the v2 posterior holds k1 / d_particle fixed, sd ~1e-17) is a constant 0;
* padded rows are all zero (masked everywhere downstream; finite and small).

The normalisation constants are serialised into the checkpoint
(``args.cond_norm_json``) so the LESRL harnesses rebuild the same e from the
nominal turbine, from ``--turb-params`` or from the env's own properties.

NOTE on the nominal farm vector: the LES-calibrated setup (k2 0.0216,
d_particle 0.48, mann_L 29.4, mann_AE 1.0) lies outside the posterior on four
of the six keys (z-scores of +16 / -4 / -6 / +1700 before clipping), so a
``nominal`` / ``expert`` e at the calibrated setup sits at the clip boundary on
those columns. ``--cond_source turbine`` avoids this by construction.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from helpers.dr_turbine import parse_turb_dr

COND_TURB_KEYS: Tuple[str, ...] = (
    "yaw_exp", "cp_gain", "ct_gain", "tau_yaw", "delay_yaw", "tau_power",
)
COND_FARM_KEYS: Tuple[str, ...] = (
    "k1", "k2", "d_particle", "hill_vortex_factor", "mann_L", "mann_GAMMA", "mann_AE",
)
# Stage-10 (rma / uposi) checkpoints serialised this 6-key farm vector.
LEGACY_FARM_KEYS: Tuple[str, ...] = (
    "k1", "k2", "d_particle", "mann_L", "mann_GAMMA", "mann_AE",
)
COND_DIM = len(COND_TURB_KEYS) + len(COND_FARM_KEYS)
COND_SOURCES = ("none", "turbine", "turbine_farm")

# Spec defaults (physical) used when a turbine key is missing / before reset.
# Kept here so the pure-numpy path has no windgym import; checked against
# WindGym.core.turbine_params.TURBINE_PARAM_SPECS in the tests.
TURB_DEFAULTS: Dict[str, float] = {
    "yaw_exp": 0.0, "cp_gain": 1.0, "ct_gain": 1.0,
    "tau_yaw": 0.0, "delay_yaw": 0.0, "tau_power": 0.0,
}

FarmLike = Union[Mapping[str, float], np.ndarray, Sequence[float]]


class ParamNormalizer:
    """Maps physical (turbine, farm) parameters to the conditioning vector e."""

    def __init__(
        self,
        turb_ranges: Mapping[str, Tuple[float, float]],
        farm_mean: Optional[Mapping[str, float]] = None,
        farm_std: Optional[Mapping[str, float]] = None,
        clip: float = 3.0,
        source: str = "turbine_farm",
        farm_keys: Sequence[str] = COND_FARM_KEYS,
    ):
        if source not in COND_SOURCES or source == "none":
            raise ValueError(
                f"cond_source must be one of {COND_SOURCES[1:]} for a ParamNormalizer, got {source!r}"
            )
        unknown = [k for k in turb_ranges if k not in COND_TURB_KEYS]
        if unknown:
            raise ValueError(f"unknown turbine keys in ranges: {unknown}")
        self.turb_ranges: Dict[str, Tuple[float, float]] = {
            k: (float(lo), float(hi)) for k, (lo, hi) in turb_ranges.items()
        }
        self.farm_mean: Dict[str, float] = {k: float(v) for k, v in (farm_mean or {}).items()}
        self.farm_std: Dict[str, float] = {k: float(v) for k, v in (farm_std or {}).items()}
        self.clip = float(clip)
        self.source = source
        self.farm_keys: Tuple[str, ...] = tuple(farm_keys)
        if source == "turbine_farm":
            missing = [k for k in self.farm_keys if k not in self.farm_mean or k not in self.farm_std]
            if missing:
                raise ValueError(f"cond_source turbine_farm needs farm stats for {missing}")
        # Precomputed affine coefficients, turbine: e = a*v + b (a = 0 when unranged)
        a = np.zeros(len(COND_TURB_KEYS), dtype=np.float64)
        b = np.zeros(len(COND_TURB_KEYS), dtype=np.float64)
        for i, k in enumerate(COND_TURB_KEYS):
            if k in self.turb_ranges:
                lo, hi = self.turb_ranges[k]
                if hi > lo:
                    a[i] = 2.0 / (hi - lo)
                    b[i] = -1.0 - a[i] * lo
        self._turb_a, self._turb_b = a, b
        if source == "turbine_farm":
            self._farm_mu = np.array([self.farm_mean[k] for k in self.farm_keys], dtype=np.float64)
            sd = np.array([self.farm_std[k] for k in self.farm_keys], dtype=np.float64)
            # Constant-column rule: a posterior column with no spread (absolute
            # or relative to its mean, e.g. sd 1e-17 on a fixed k1) is inert.
            tol = np.maximum(1e-9, 1e-9 * np.abs(self._farm_mu))
            sd_eff = np.where(sd > tol, sd, 0.0)
            self._farm_inv_sd = np.where(sd_eff > 0, 1.0 / np.where(sd_eff > 0, sd_eff, 1.0), 0.0)
        else:
            self._farm_mu = np.zeros(len(self.farm_keys))
            self._farm_inv_sd = np.zeros(len(self.farm_keys))

    # ------------------------------------------------------------ factories --
    @classmethod
    def from_args(cls, args: Any, posterior: Optional[dict]) -> "ParamNormalizer":
        """Ranges from ``args.turb_dr`` (``name=lo:hi`` items), farm stats from
        the posterior (``{"samples": (N, d), "names": [...]}``)."""
        source = getattr(args, "cond_source", "none")
        ranges = parse_turb_dr(getattr(args, "turb_dr", None) or ())
        mean: Dict[str, float] = {}
        std: Dict[str, float] = {}
        if source == "turbine_farm":
            if posterior is None:
                raise ValueError("cond_source turbine_farm needs --dr_posterior_path (posterior is None)")
            names = list(posterior["names"])
            samples = np.asarray(posterior["samples"], dtype=np.float64)
            missing = [k for k in COND_FARM_KEYS if k not in names]
            if missing:
                raise ValueError(f"posterior lacks farm key(s) {missing} needed by cond_source turbine_farm")
            for k in COND_FARM_KEYS:
                col = samples[:, names.index(k)]
                mean[k] = float(col.mean())
                std[k] = float(col.std())
        return cls(ranges, mean, std, clip=float(getattr(args, "cond_clip", 3.0)), source=source)

    def to_json(self) -> str:
        return json.dumps({
            "turb_keys": list(COND_TURB_KEYS), "farm_keys": list(self.farm_keys),
            "turb_ranges": {k: list(v) for k, v in self.turb_ranges.items()},
            "farm_mean": self.farm_mean, "farm_std": self.farm_std,
            "clip": self.clip, "source": self.source,
        })

    @classmethod
    def from_json(cls, s: str) -> "ParamNormalizer":
        d = json.loads(s)
        farm_keys = tuple(d.get("farm_keys", COND_FARM_KEYS))
        if tuple(d.get("turb_keys", COND_TURB_KEYS)) != COND_TURB_KEYS or \
                farm_keys not in (COND_FARM_KEYS, LEGACY_FARM_KEYS):
            raise ValueError("cond_norm_json key order does not match this code's COND_*_KEYS "
                             "(nor the legacy 6-key farm vector)")
        return cls({k: tuple(v) for k, v in d["turb_ranges"].items()},
                   d.get("farm_mean", {}), d.get("farm_std", {}),
                   clip=d.get("clip", 3.0), source=d.get("source", "turbine_farm"),
                   farm_keys=farm_keys)

    # ------------------------------------------------------------- vectors --
    @property
    def cond_dim(self) -> int:
        return len(COND_TURB_KEYS) + len(self.farm_keys)

    def _farm_array(self, farm: Optional[FarmLike]) -> np.ndarray:
        n_farm = len(self.farm_keys)
        if farm is None:
            return np.full(n_farm, np.nan)
        if isinstance(farm, Mapping):
            # A superset mapping (7 keys into a legacy 6-key normaliser) is fine.
            return np.array([float(farm.get(k, np.nan)) for k in self.farm_keys], dtype=np.float64)
        arr = np.asarray(farm, dtype=np.float64).reshape(-1)
        if arr.shape[0] != n_farm:
            raise ValueError(f"farm params must have {n_farm} entries ({self.farm_keys}), got {arr.shape}")
        return arr

    def vector(self, turb_tok: np.ndarray, farm: Optional[FarmLike], mask: np.ndarray) -> np.ndarray:
        """``turb_tok`` (max_t, 6) physical in token order, ``farm`` len(farm_keys)
        values / dict, ``mask`` (max_t,) True = padding -> e (max_t, cond_dim) float32."""
        turb_tok = np.asarray(turb_tok, dtype=np.float64)
        if turb_tok.ndim != 2 or turb_tok.shape[1] != len(COND_TURB_KEYS):
            raise ValueError(f"turb_tok must be (max_turbines, {len(COND_TURB_KEYS)}), got {turb_tok.shape}")
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        if mask.shape[0] != turb_tok.shape[0]:
            raise ValueError("mask length must equal turb_tok rows")
        max_t = turb_tok.shape[0]
        e = np.zeros((max_t, self.cond_dim), dtype=np.float64)
        e[:, : len(COND_TURB_KEYS)] = turb_tok * self._turb_a + self._turb_b
        if self.source == "turbine_farm":
            z = (self._farm_array(farm) - self._farm_mu) * self._farm_inv_sd
            z = np.clip(np.nan_to_num(z, nan=0.0, posinf=self.clip, neginf=-self.clip), -self.clip, self.clip)
            e[:, len(COND_TURB_KEYS):] = z[None, :]
        e = np.nan_to_num(e, nan=0.0, posinf=0.0, neginf=0.0)
        e[mask] = 0.0
        return e.astype(np.float32)

    def nominal_vector(self, mask: np.ndarray, farm: Optional[FarmLike] = None) -> np.ndarray:
        """e for the nominal turbine (spec defaults) and the given farm values
        (None -> farm columns 0 = posterior mean)."""
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        tok = np.array([[TURB_DEFAULTS[k] for k in COND_TURB_KEYS]] * mask.shape[0], dtype=np.float64)
        if farm is None:
            farm = dict(self.farm_mean) if self.source == "turbine_farm" else None
        return self.vector(tok, farm, mask)

    def vector_from_multilayout(self, ml: Any) -> np.ndarray:
        """e from a MultiLayoutEnv's token-order properties (train / eval / harness)."""
        return self.vector(ml.turbine_params_tok, ml.farm_params, ml.attention_mask)

    def vector_from_base_env(self, base_env: Any, max_turbines: Optional[int] = None,
                             perm: Optional[np.ndarray] = None) -> np.ndarray:
        """e straight from a WindFarmEnv (LESRL harnesses: no MultiLayoutEnv).

        ``base_env.turbine_params`` (None -> defaults) in physical order,
        optionally permuted by ``perm`` (token i = physical perm[i]) and padded
        to ``max_turbines``; farm values from the env's active DWM params over
        its ``dwm_setup`` (see :func:`farm_params_from_base_env`).
        """
        tp = getattr(base_env, "turbine_params", None)
        n = int(getattr(base_env, "n_turbines", 0) or getattr(base_env, "n_turb", 0)
                or (len(next(iter(tp.values()))) if tp else 0))
        if n <= 0:
            raise ValueError("cannot infer the turbine count from base_env")
        tok = turbine_params_matrix(tp, n)
        if perm is not None:
            tok = tok[np.asarray(perm, dtype=np.int64)]
        max_t = int(max_turbines or n)
        out = np.array([[TURB_DEFAULTS[k] for k in COND_TURB_KEYS]] * max_t, dtype=np.float64)
        out[:n] = tok
        mask = np.ones(max_t, dtype=bool)
        mask[:n] = False
        return self.vector(out, farm_params_from_base_env(base_env), mask)


# ---------------------------------------------------------------- helpers --
def turbine_params_matrix(tp: Optional[Mapping[str, Any]], n_turb: int) -> np.ndarray:
    """``{name: (n,) or scalar}`` (WindFarmEnv.turbine_params, None before reset)
    -> (n_turb, 6) physical, spec defaults where a key is missing."""
    out = np.array([[TURB_DEFAULTS[k] for k in COND_TURB_KEYS]] * n_turb, dtype=np.float64)
    if tp:
        for j, k in enumerate(COND_TURB_KEYS):
            if k in tp:
                v = np.asarray(tp[k], dtype=np.float64).reshape(-1)
                out[:, j] = v if v.shape[0] == n_turb else v[0]
    return out


def farm_params_from_base_env(base_env: Any) -> Dict[str, float]:
    """The seven farm keys resolved: this episode's active DWM params
    (``active_dwm_params`` or the private ``_active_dwm_params``) over the
    env's ``dwm_setup`` (k1 / k2 / d_particle / hill_vortex_factor, mann.L /
    gamma / alphaepsilon). Missing or None values -> NaN (the normaliser maps
    NaN to 0). A legacy 6-key normaliser simply ignores the extra key."""
    active = getattr(base_env, "active_dwm_params", None)
    if active is None:
        active = getattr(base_env, "_active_dwm_params", None) or {}
    setup = getattr(base_env, "dwm_setup", None)
    mann = getattr(setup, "mann", None)
    fallback = {
        "k1": getattr(setup, "k1", None), "k2": getattr(setup, "k2", None),
        "d_particle": getattr(setup, "d_particle", None),
        "hill_vortex_factor": getattr(setup, "hill_vortex_factor", None),
        "mann_L": getattr(mann, "L", None), "mann_GAMMA": getattr(mann, "gamma", None),
        "mann_AE": getattr(mann, "alphaepsilon", None),
    }
    out: Dict[str, float] = {}
    for k in COND_FARM_KEYS:
        v = active.get(k, fallback[k]) if isinstance(active, Mapping) else fallback[k]
        try:
            out[k] = float(v) if v is not None else float("nan")
        except (TypeError, ValueError):
            out[k] = float("nan")
    return out


class CondFetcher:
    """Fetches e for every env of a vector env through two ``get_attr`` calls.

    ``envs`` must expose ``get_attr(name) -> list`` (the trainer passes
    ``envs.env``, an AsyncVectorEnv of MultiLayoutEnv). ``values`` is the
    cached ``(num_envs, max_turbines, cond_dim)`` float32 array from the last
    ``refresh()`` (None before the first); callers copy it into the step's
    state so a later refresh cannot alias stored transitions.
    """

    def __init__(self, envs: Any, normalizer: ParamNormalizer):
        self.envs = envs
        self.normalizer = normalizer
        self._values: Optional[np.ndarray] = None

    def refresh(self) -> np.ndarray:
        toks = self.envs.get_attr("turbine_params_tok")
        farms = self.envs.get_attr("farm_params")
        masks = self.envs.get_attr("attention_mask")
        self._values = np.stack([
            self.normalizer.vector(t, f, m) for t, f, m in zip(toks, farms, masks)
        ]).astype(np.float32)
        return self._values.copy()

    @property
    def values(self) -> Optional[np.ndarray]:
        return None if self._values is None else self._values.copy()
