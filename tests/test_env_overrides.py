"""helpers/env_overrides.py: --env_overrides (JSON) deep-updates the env config preset.

Lets a run change preset keys the trainer has no flag for (derate_max,
derate_penalty, track_def entries, ...), recorded in the checkpoint args like
every other flag so eval rebuilds the same env.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers.env_overrides import apply_env_overrides  # noqa: E402


def test_empty_is_a_no_op():
    config = {"derate_max": 0.8, "track_def": {"Track_reward": "abs"}}
    apply_env_overrides(config, "{}")
    apply_env_overrides(config, None)
    assert config == {"derate_max": 0.8, "track_def": {"Track_reward": "abs"}}


def test_deep_update_keeps_sibling_keys():
    config = {"derate_max": 0.8, "track_def": {"Track_reward": "abs", "track_obs_preview": 0}}
    apply_env_overrides(config, '{"derate_max": 1.0, "track_def": {"Track_reward": "gaussian"}}')
    assert config["derate_max"] == 1.0
    assert config["track_def"] == {"Track_reward": "gaussian", "track_obs_preview": 0}


@pytest.mark.parametrize("bad", ["[1]", "not json", '"x"'])
def test_bad_json_raises(bad):
    with pytest.raises(ValueError, match="env_overrides"):
        apply_env_overrides({}, bad)
