"""helpers/power_ref.py: build_power_ref picks the tracking reference.

Without a factory it is the fraction-of-greedy step schedule (unchanged
behaviour). With ``power_ref_factory="module:attr"`` the reference comes from
an external, importable callable ``attr(greedy=..., **json kwargs)``, e.g. a
setpoint family of the powertracking repo.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers.power_ref import build_power_ref, greedy_probe_config  # noqa: E402

ENV = SimpleNamespace(delay=1.0)
FACTORY_SRC = '''
def constant(*, greedy, frac, offset=0.0):
    def ref(t, env):
        return frac * greedy + offset
    return ref
'''


def test_default_schedule_is_unchanged():
    ref, desc = build_power_ref(10e6, power_schedule="default")
    assert ref(0.0, ENV) == pytest.approx(8e6)  # first segment 80 %
    assert ref(250.0, ENV) == pytest.approx(6e6)  # second segment 60 %
    assert "default" in desc


def test_boost_schedule():
    ref, _ = build_power_ref(10e6, power_schedule="boost")
    assert ref(250.0, ENV) == pytest.approx(11.5e6)


def test_factory_with_json_kwargs(tmp_path, monkeypatch):
    (tmp_path / "my_refs.py").write_text(FACTORY_SRC)
    monkeypatch.syspath_prepend(str(tmp_path))
    ref, desc = build_power_ref(
        10e6, factory="my_refs:constant", factory_kwargs='{"frac": 0.5, "offset": 1.0}'
    )
    assert ref(0.0, ENV) == pytest.approx(5e6 + 1.0)
    assert "my_refs:constant" in desc


def test_factory_needs_module_and_attribute():
    with pytest.raises(ValueError, match="module:attr"):
        build_power_ref(10e6, factory="no_colon_here")


def test_greedy_probe_config_disables_yaw_and_the_share_inner_loop():
    cfg = {"yaw_action": True, "derate_method": "share", "yaw_deadband": 0.375}
    probe = greedy_probe_config(cfg)
    # under WindGym's inner loop the all-min action means equal deficit shares and
    # the probe would track its default reference instead of running greedy
    assert probe == {"yaw_action": False, "derate_method": "absolute", "yaw_deadband": 0.375}
    assert cfg["derate_method"] == "share"  # the training config is untouched
    for method in ("absolute", "step", None):
        cfg = {"yaw_action": True} if method is None else {"yaw_action": True, "derate_method": method}
        assert greedy_probe_config(cfg) == {**cfg, "yaw_action": False}
