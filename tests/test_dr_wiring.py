"""Stage-7 DR wiring: posterior loader + sampler + trainer-level guards.

The heavy end-to-end (DWMRandomizationWrapper feeding dwm_params into a real
dynamiks reset) lives in windgym's test_dwm_randomization_wrapper.py; here we
gate the TransformerSac-side seams: the .npz contract, key-subset handling,
and the config surface the launcher relies on.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers.dr_posterior import load_posterior, make_dr_sampler  # noqa: E402

POSTERIOR = Path(__file__).resolve().parents[2] / "posterior_les_data_samples.npz"
LES_KEYS = ("k1", "k2", "d_particle", "mann_L", "mann_GAMMA", "mann_AE")


@pytest.fixture(scope="module")
def posterior():
    if not POSTERIOR.exists():
        pytest.skip(f"calibrated posterior not staged at {POSTERIOR}")
    return load_posterior(str(POSTERIOR))


def test_posterior_contract(posterior):
    # The Stage-7 launcher randomizes exactly these six keys.
    assert list(posterior["names"]) == list(LES_KEYS)
    assert posterior["samples"].ndim == 2
    assert posterior["samples"].shape[1] == len(LES_KEYS)
    assert np.all(np.isfinite(posterior["samples"]))


def test_sampler_draws_named_floats(posterior):
    sampler = make_dr_sampler(posterior, keys=LES_KEYS)
    rng = np.random.default_rng(123)
    draw = sampler(rng)
    assert set(draw) == set(LES_KEYS)
    assert all(isinstance(v, float) for v in draw.values())
    # draws are posterior rows: every draw must be an actual sample row
    samples = posterior["samples"]
    row_match = np.all(
        np.isclose(samples, np.array([draw[k] for k in LES_KEYS])[None, :]),
        axis=1,
    )
    assert row_match.any()


def test_sampler_subset_preserves_joint_rows(posterior):
    # A subset sampler must still return values from one joint row.
    sub = ("k1", "k2")
    sampler = make_dr_sampler(posterior, keys=sub)
    rng = np.random.default_rng(7)
    draw = sampler(rng)
    assert set(draw) == set(sub)
    cols = [list(posterior["names"]).index(k) for k in sub]
    row_match = np.all(
        np.isclose(posterior["samples"][:, cols],
                   np.array([draw[k] for k in sub])[None, :]),
        axis=1,
    )
    assert row_match.any()


def test_unknown_key_raises(posterior):
    with pytest.raises(KeyError):
        make_dr_sampler(posterior, keys=("k1", "not_a_param"))


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_posterior("/no/such/posterior.npz")


def test_args_surface():
    # The launcher's flags must exist with DR off by default.
    from config import Args
    a = Args()
    assert a.dr_posterior_path is None
    assert tuple(a.dr_keys) == LES_KEYS
    assert a.veer_min == 0.0 and a.veer_max == 0.0 and a.tilt == 0.0


def test_seeded_sampler_reproducible(posterior):
    sampler = make_dr_sampler(posterior, keys=LES_KEYS)
    d1 = sampler(np.random.default_rng(42))
    d2 = sampler(np.random.default_rng(42))
    assert d1 == d2


# ---- trainer-level guards (validate_dr_setup, module-level in the trainer) --

@pytest.fixture(scope="module")
def trainer():
    # Heavy import (torch + WindGym); shared across the guard tests.
    import transformer_sac_windfarm as tsw
    return tsw


def _args(**over):
    from types import SimpleNamespace
    base = dict(backend="dynamiks", dr_keys=LES_KEYS, TI_type="MannGenerate")
    base.update(over)
    return SimpleNamespace(**base)


def test_guard_noop_without_posterior(trainer):
    # DR off: nothing validated, nothing raised, whatever the other flags say.
    trainer.validate_dr_setup(_args(backend="pywake", TI_type="MannLoad"), None)


def test_guard_pywake_backend_rejected(trainer, posterior):
    with pytest.raises(ValueError, match="pywake"):
        trainer.validate_dr_setup(_args(backend="pywake"), posterior)


def test_guard_mann_keys_require_manngenerate(trainer, posterior):
    with pytest.raises(ValueError, match="MannGenerate"):
        trainer.validate_dr_setup(_args(TI_type="MannLoad"), posterior)


def test_guard_non_mann_subset_allows_other_ti_types(trainer, posterior):
    # Closure-only DR does not constrain the turbulence type.
    trainer.validate_dr_setup(
        _args(dr_keys=("k1", "k2", "d_particle"), TI_type="MannLoad"), posterior
    )


def test_guard_launch_config_passes(trainer, posterior):
    # The exact Stage-7 launch shape must sail through.
    trainer.validate_dr_setup(_args(), posterior)
