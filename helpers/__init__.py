"""Helper modules for wind farm transformer SAC.

Core utilities (always available):
- helper_funcs: Checkpoint I/O, coordinate transforms, env config
- agent: WindFarmAgent inference wrapper
- eval_utils: PolicyEvaluator for training-time evaluation
- data_loader: PyTorch Dataset for HDF5 pretraining data
- multi_layout_env: Multi-layout environment wrapper
- multi_layout_debug: Debug logging for multi-layout training

Optional (imported lazily due to heavy dependencies):
- geometric_profiles: Geometry-based profile computation (scipy)
- receptivity_profiles: PyWake-based profile computation (py_wake)
"""

from helpers.helper_funcs import (
    save_checkpoint,
    load_checkpoint,
    transform_to_wind_relative,
    EnhancedPerTurbineWrapper,
)
from helpers.layouts import get_layout_positions
from helpers.env_configs import make_env_config
from helpers.agent import WindFarmAgent
from helpers.eval_utils import PolicyEvaluator
from helpers.data_loader import WindFarmPretrainDataset
from helpers.multi_layout_env import MultiLayoutEnv, LayoutConfig
from helpers.multi_layout_debug import MultiLayoutDebugLogger

__all__ = [
    # helper_funcs
    "get_layout_positions",
    "save_checkpoint",
    "load_checkpoint",
    "make_env_config",
    "transform_to_wind_relative",
    "EnhancedPerTurbineWrapper",
    # agent
    "WindFarmAgent",
    # eval_utils
    "PolicyEvaluator",
    # data_loader
    "WindFarmPretrainDataset",
    # multi_layout_env
    "MultiLayoutEnv",
    "LayoutConfig",
    # multi_layout_debug
    "MultiLayoutDebugLogger",
]
