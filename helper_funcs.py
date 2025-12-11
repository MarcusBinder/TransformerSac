from WindGym.utils.generate_layouts import (
    generate_square_grid, 
    generate_cirular_farm, 
    generate_right_triangle_grid,
    generate_line_dots_multiple_thetas
)

import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from typing import Optional, Tuple, List, Dict, Any


def get_layout_positions(layout_type: str, wind_turbine) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get turbine positions for a given layout type.
    
    Args:
        layout_type: Layout identifier string
        wind_turbine: PyWake wind turbine object
    
    Returns:
        x_pos, y_pos: Arrays of turbine positions
    """
    # Import here to avoid circular imports when not using WindGym
    # from WindGym.utils.generate_layouts import generate_square_grid, generate_cirular_farm
    
    layouts = {
        "test_layout": lambda: generate_square_grid(turbine=wind_turbine, nx=2, ny=1, xDist=5, yDist=5),
        "square_2x2": lambda: generate_square_grid(turbine=wind_turbine, nx=2, ny=2, xDist=5, yDist=5),
        "small_triangle": lambda: generate_right_triangle_grid(turbine=wind_turbine, nx=2, ny=3, xDist=5, yDist=5),
        "square_3x3": lambda: generate_square_grid(turbine=wind_turbine, nx=3, ny=3, xDist=5, yDist=5),
        "circular_6": lambda: generate_cirular_farm(n_list=[1, 5], turbine=wind_turbine, r_dist=5),
        "circular_10": lambda: generate_cirular_farm(n_list=[3, 7], turbine=wind_turbine, r_dist=5),
        "tri1": lambda: generate_right_triangle_grid(turbine=wind_turbine, nx=2, ny=2, xDist=5, yDist=5, orientation='lower_left'),
        "tri2": lambda: generate_right_triangle_grid(turbine=wind_turbine, nx=2, ny=2, xDist=5, yDist=5, orientation='lower_right'),
        "tri3": lambda: generate_right_triangle_grid(turbine=wind_turbine, nx=2, ny=2, xDist=5, yDist=5, orientation='upper_left'),
        "tri4": lambda: generate_right_triangle_grid(turbine=wind_turbine, nx=2, ny=2, xDist=5, yDist=5, orientation='upper_right'),
        "5turb1": lambda: generate_line_dots_multiple_thetas(X=3, spacing=5, thetas=[0, 30], turbine=wind_turbine),
        "5turb2": lambda: generate_line_dots_multiple_thetas(X=3, spacing=5, thetas=[0, -30], turbine=wind_turbine),
        "5turb3": lambda: generate_line_dots_multiple_thetas(X=3, spacing=5, thetas=[-30, 30], turbine=wind_turbine),
        "a": lambda: generate_square_grid(turbine=wind_turbine, nx=3, ny=1, xDist=5, yDist=5),
        "b": lambda: generate_square_grid(turbine=wind_turbine, nx=2, ny=2, xDist=5, yDist=5),
        "c": lambda: generate_square_grid(turbine=wind_turbine, nx=4, ny=1, xDist=5, yDist=5),
        "d": lambda: generate_right_triangle_grid(turbine=wind_turbine, nx=2, ny=2, xDist=5, yDist=5, orientation='lower_right'),
        "e": lambda: generate_square_grid(turbine=wind_turbine, nx=3, ny=2, xDist=5, yDist=5),
    }
    
    if layout_type not in layouts:
        raise ValueError(f"Unknown layout type: {layout_type}. Available: {list(layouts.keys())}")
    
    return layouts[layout_type]()


def get_env_wind_directions(envs, num_envs: int) -> np.ndarray:
    """Get current wind direction from each environment."""
    return np.array(envs.env.get_attr('wd'), dtype=np.float32)


def get_env_raw_positions(envs, num_envs: int, n_turbines_max: int) -> np.ndarray:
    """Get raw (unnormalized) turbine positions from each environment."""
    return np.array(envs.env.get_attr('turbine_positions'), dtype=np.float32)


def get_env_attention_masks(envs, num_envs: int, n_turbines_max: int) -> np.ndarray:
    """Get attention masks from each environment."""
    return np.array(envs.env.get_attr('attention_mask'), dtype=bool)

def save_checkpoint(
    actor: nn.Module,
    qf1: nn.Module,
    qf2: nn.Module,
    actor_optimizer: optim.Optimizer,
    q_optimizer: optim.Optimizer,
    step: int,
    run_name: str,
    args: Any,
    log_alpha: Optional[torch.Tensor] = None,
    alpha_optimizer: Optional[optim.Optimizer] = None,
) -> str:
    """
    Save training checkpoint.
    
    Returns:
        Path to saved checkpoint
    """
    checkpoint_dir = f"runs/{run_name}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_path = f"{checkpoint_dir}/step_{step}.pt"
    
    checkpoint = {
        "step": step,
        "actor_state_dict": actor.state_dict(),
        "qf1_state_dict": qf1.state_dict(),
        "qf2_state_dict": qf2.state_dict(),
        "actor_optimizer_state_dict": actor_optimizer.state_dict(),
        "q_optimizer_state_dict": q_optimizer.state_dict(),
        "args": vars(args),
    }
    
    if log_alpha is not None:
        checkpoint["log_alpha"] = log_alpha.detach().cpu()
    if alpha_optimizer is not None:
        checkpoint["alpha_optimizer_state_dict"] = alpha_optimizer.state_dict()
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")
    
    return checkpoint_path

def make_env_config() -> Dict[str, Any]:
    """
    Create base environment configuration.
    
    This configuration is designed for transformer-based control:
    - Per-turbine measurements enabled (ws, wd, yaw)
    - History stacking for temporal context
    - Baseline comparison for reward calculation
    """
    return {
        "yaw_init": "Random",
        "BaseController": "Local",
        "ActionMethod": "yaw",
        "Track_power": False,
        "farm": {"yaw_min": -30, "yaw_max": 30},
        "wind": {
            "ws_min": 10, "ws_max": 10,
            "TI_min": 0.07, "TI_max": 0.07,
            "wd_min": 260, "wd_max": 280,
        },
        "act_pen": {"action_penalty": 0.0, "action_penalty_type": "Change"},
        "power_def": {"Power_reward": "Baseline", "Power_avg": 1, "Power_scaling": 1.0},
        "mes_level": {
            "turb_ws": True,
            "turb_wd": True,  # Will be converted to deviation
            "turb_TI": False,
            "turb_power": True,  # Include power
            "farm_ws": False,
            "farm_wd": False,
            "farm_TI": False,
            "farm_power": False,
        },
        "ws_mes": {
            "ws_current": False,
            "ws_rolling_mean": True,
            "ws_history_N": 15,  # History length
            "ws_history_length": 15,
            "ws_window_length": 1,
        },
        "wd_mes": {
            "wd_current": False,
            "wd_rolling_mean": True,
            "wd_history_N": 15,
            "wd_history_length": 15,
            "wd_window_length": 1,
        },
        "yaw_mes": {
            "yaw_current": False,
            "yaw_rolling_mean": True,
            "yaw_history_N": 15,
            "yaw_history_length": 15,
            "yaw_window_length": 1,
        },
        "power_mes": {
            "power_current": False,
            "power_rolling_mean": True,
            "power_history_N": 15,
            "power_history_length": 15,
            "power_window_length": 1,
        },
    }