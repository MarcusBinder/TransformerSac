"""The token-order parameter properties live on MultiLayoutEnv only (a bad anchor
once duplicated them onto the LayoutConfig dataclass)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.multi_layout_env import LayoutConfig, MultiLayoutEnv  # noqa: E402


def test_layoutconfig_has_no_env_properties():
    for name in ("turbine_params_tok", "farm_params", "action_history"):
        assert not hasattr(LayoutConfig, name), name
        assert hasattr(MultiLayoutEnv, name), name
    lc = LayoutConfig(name="x", x_pos=np.zeros(2), y_pos=np.zeros(2))
    assert not hasattr(lc, "turbine_params_tok")
