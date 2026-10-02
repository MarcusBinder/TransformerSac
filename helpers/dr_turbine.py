"""Turbine-parameter domain randomization for training (Stage 9, ``--turb_dr``).

Companion to :mod:`helpers.dr_posterior`: that module draws DWM closure / Mann
parameters from the LES-calibrated posterior, this one draws per-turbine
TURBINE-MODEL perturbations (``WindGym.core.turbine_params``: ``yaw_exp``,
``cp_gain``, ``ct_gain``, ``tau_yaw``, ``delay_yaw``, ``tau_power``) from
independent uniform ranges given on the command line as ``name=lo:hi``.

Both samplers feed the same ``DWMRandomizationWrapper`` (one RNG per env,
seeded ``args.seed + i``); :func:`combine_samplers` merges them, posterior
first, so posterior-only runs keep their exact draw stream.

Arrays are in the env's PHYSICAL turbine order. MultiLayoutEnv's shuffle
permutes observations/actions only, and iid per-turbine draws make the
order immaterial; a sampler built for ``max_turbines`` turbines is sliced to
each layout's ``n_turb`` by the env (``normalize_turbine_params``).

Training resets only: the in-training evaluators and every LESRL eval harness
run the nominal turbine unless they pass ``--turb-params`` explicitly.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

Ranges = Dict[str, Tuple[float, float]]
SamplerFn = Callable[[np.random.Generator], dict]


def parse_turb_dr(specs: Optional[Sequence[str]]) -> Ranges:
    """``("yaw_exp=-0.4:0.6", "cp_gain=0.95:1.05")`` -> ``{name: (lo, hi)}``.

    Order is preserved. Bad syntax, non-numeric bounds, ``lo > hi`` and
    duplicate names raise ``ValueError``; an empty/None spec gives ``{}``.
    """
    ranges: Ranges = {}
    for item in specs or ():
        item = str(item).strip()
        if "=" not in item or item.count("=") != 1:
            raise ValueError(f"--turb_dr item {item!r}: expected name=lo:hi")
        name, _, rng = item.partition("=")
        name = name.strip()
        if ":" not in rng or rng.count(":") != 1 or not name:
            raise ValueError(f"--turb_dr item {item!r}: expected name=lo:hi")
        lo_s, _, hi_s = rng.partition(":")
        try:
            lo, hi = float(lo_s), float(hi_s)
        except ValueError as e:
            raise ValueError(f"--turb_dr item {item!r}: non-numeric bound for {name}") from e
        if not (np.isfinite(lo) and np.isfinite(hi)):
            raise ValueError(f"--turb_dr item {item!r}: bounds must be finite")
        if lo > hi:
            raise ValueError(f"--turb_dr item {item!r}: lo > hi")
        if name in ranges:
            raise ValueError(f"--turb_dr: duplicate name {name!r}")
        ranges[name] = (lo, hi)
    return ranges


def validate_turb_dr_names(ranges: Ranges, spec=None) -> None:
    """Check names and ranges against ``WindGym.core.turbine_params.TURBINE_PARAM_SPECS``.

    ``spec`` (name -> object with ``lo``/``hi``) can be injected for tests. A
    windgym without ``turbine_params`` is a version skew: ``RuntimeError`` when
    ranges were requested, silently fine when they were not.
    """
    if not ranges:
        return
    if spec is None:
        try:
            from WindGym.core.turbine_params import TURBINE_PARAM_SPECS as spec
        except ImportError as e:
            raise RuntimeError(
                "--turb_dr requested but this WindGym has no "
                "WindGym.core.turbine_params (needs windgym dev_dynamiks with "
                "turbine-parameter DR)."
            ) from e
    unknown = [k for k in ranges if k not in spec]
    if unknown:
        raise ValueError(
            f"Unknown turbine DR param(s) {unknown}; allowed: {list(spec)}"
        )
    for k, (lo, hi) in ranges.items():
        s = spec[k]
        if lo < s.lo or hi > s.hi:
            raise ValueError(
                f"--turb_dr {k} range {(lo, hi)} outside the WindGym bounds "
                f"[{s.lo}, {s.hi}]"
            )


def make_turbine_sampler(ranges: Ranges, n_turb: int, per_turbine: bool = True) -> SamplerFn:
    """``sampler(rng) -> {name: ndarray (n_turb,)}`` with iid ``U[lo, hi]`` draws.

    ``per_turbine=False`` draws one float per name (farm-wide perturbation).
    """
    ranges = dict(ranges)
    n_turb = int(n_turb)
    if n_turb < 1:
        raise ValueError(f"n_turb must be >= 1, got {n_turb}")

    def _sampler(rng: np.random.Generator) -> dict:
        out = {}
        for k, (lo, hi) in ranges.items():
            if per_turbine:
                out[k] = rng.uniform(lo, hi, size=n_turb).astype(np.float64)
            else:
                out[k] = float(rng.uniform(lo, hi))
        return out

    return _sampler


def combine_samplers(*samplers: Optional[SamplerFn]) -> Optional[SamplerFn]:
    """Merge samplers into one; ``None`` entries are skipped, all-None -> ``None``.

    Samplers are called IN ORDER on the same rng (posterior first keeps a
    posterior-only run's draw stream bit-identical). Overlapping keys raise
    ``KeyError`` at draw time.
    """
    active = [s for s in samplers if s is not None]
    if not active:
        return None
    if len(active) == 1:
        return active[0]

    def _sampler(rng: np.random.Generator) -> dict:
        out: dict = {}
        for s in active:
            d = s(rng)
            dup = set(d) & set(out)
            if dup:
                raise KeyError(f"DR samplers draw overlapping keys: {sorted(dup)}")
            out.update(d)
        return out

    return _sampler


def _fmt(v: float) -> str:
    return f"{v:g}"


def format_turb_dr(ranges: Ranges) -> str:
    """One greppable banner line: ``[TurbDR] yaw_exp~U[-0.4,0.6] cp_gain~U[0.95,1.05]``."""
    if not ranges:
        return "[TurbDR] off"
    return "[TurbDR] " + " ".join(f"{k}~U[{_fmt(lo)},{_fmt(hi)}]" for k, (lo, hi) in ranges.items())
