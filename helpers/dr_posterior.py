"""Self-contained loader for a calibrated DWM-parameter posterior.

The .npz file is produced by the LESRL calibration pipeline and contains:
  - ``samples``     : (N, d) float ndarray of posterior draws
  - ``param_names`` : (d,) array of parameter name strings

This module duplicates the small loader in the LESRL parent's
``calibration/sampling.py`` so that ``transformer_sac_windfarm.py`` runs with
only TransformerSac + windgym on the path. The .npz schema is the contract;
keeping a tiny duplicate is cheaper than coupling to the parent repo.
"""
from __future__ import annotations

import os
from typing import Tuple

import numpy as np


def load_posterior(path: str) -> dict:
    """Load a posterior .npz. Returns dict with keys ``samples`` and ``names``.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    KeyError
        If the file does not have the expected ``samples`` and ``param_names``
        keys.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"posterior file not found: {path}")
    d = np.load(path, allow_pickle=False)
    if "samples" not in d.files or "param_names" not in d.files:
        raise KeyError(
            f"{path} must contain 'samples' (N, d) and 'param_names' (d,) "
            f"arrays; found {sorted(d.files)}"
        )
    return {
        "samples": np.asarray(d["samples"]),
        "names": [str(n) for n in d["param_names"]],
    }


def make_dr_sampler(posterior: dict, keys: Tuple[str, ...]):
    """Build a ``sampler(rng) -> {key: float}`` closure for ``DWMRandomizationWrapper``.

    Empirical bootstrap over the posterior's rows preserves whatever joint
    structure is in the calibration (e.g. k1↔k2 correlation in the LES-fit
    posterior).

    Parameters
    ----------
    posterior : dict
        Output of :func:`load_posterior`.
    keys : tuple[str, ...]
        Subset of ``posterior['names']`` to actually expose to the wrapper.
        Other posterior columns are still drawn jointly with these (so the
        joint structure is preserved across all calibrated dimensions) but
        not returned in the sampler's output dict.
    """
    names = posterior["names"]
    samples = posterior["samples"]
    missing = [k for k in keys if k not in names]
    if missing:
        raise KeyError(
            f"posterior does not contain {missing}; available: {names}"
        )
    name_to_col = {n: i for i, n in enumerate(names)}
    cols = np.array([name_to_col[k] for k in keys], dtype=np.intp)
    n_rows = len(samples)
    keys = tuple(keys)  # freeze order

    def _sampler(rng: np.random.Generator) -> dict:
        idx = int(rng.integers(0, n_rows))
        row = samples[idx, cols]
        return {k: float(v) for k, v in zip(keys, row)}

    return _sampler
