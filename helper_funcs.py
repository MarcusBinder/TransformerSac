from WindGym.utils.generate_layouts import (
    generate_square_grid,
    generate_cirular_farm,
    generate_right_triangle_grid,
    generate_line_dots_multiple_thetas,
    generate_diamond_grid,
    generate_staggered_grid,
)

import os
import math
import torch
import gymnasium as gym
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pathlib import Path
import re

from copy import deepcopy
from typing import Optional, Tuple, List, Dict, Any

def soft_update(source, target, tau):
    with torch.no_grad():
        source_params = list(source.parameters())
        target_params = list(target.parameters())
        torch._foreach_lerp_(target_params, source_params, tau)


# =============================================================================
# CHECKPOINT UTILITIES
# =============================================================================


# def load_actor_from_checkpoint(checkpoint_path: str, device: torch.device):
#     """Load actor network from checkpoint."""
#     checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
#     args = checkpoint["args"]

#     # Return checkpoint and args
#     return checkpoint, args



def load_actor_from_checkpoint(checkpoint_path: str, device: torch.device):
    """Load actor network from checkpoint (handles old and new formats)."""
    data = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Check if it's the new dict format or old tuple format
    if isinstance(data, dict) and "actor_state_dict" in data:
        # New format
        checkpoint = data
        args = data["args"]
        if hasattr(args, '__dict__'):
            args = vars(args)
    elif isinstance(data, tuple) and len(data) == 3:
        # Old format: (actor_state_dict, qf1_state_dict, qf2_state_dict)
        checkpoint, args = load_old_sac_checkpoint(checkpoint_path, device)
    else:
        raise ValueError(f"Unknown checkpoint format: {type(data)}")
    
    return checkpoint, args


def find_checkpoints(checkpoint_dir: str) -> list:
    """
    Find all checkpoint files in a directory and return them sorted by step.

    Args:
        checkpoint_dir: Path to checkpoints directory

    Returns:
        List of (step, filepath) tuples sorted by step
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = []

    # Look for .pt files with step numbers in the name
    for filepath in checkpoint_dir.glob("*.pt"):
        # Try to extract step number from filename (e.g., "step_1000.pt", "checkpoint_1000.pt")
        match = re.search(r"(?:step|checkpoint)[_-]?(\d+)", filepath.stem, re.IGNORECASE)
        if match:
            step = int(match.group(1))
            checkpoints.append((step, str(filepath)))
        else:
            # Try to extract just the trailing number from the filename
            match = re.search(r"_(\d+)$", filepath.stem)
            
            
            if match:
                step = match.group(1) or match.group(2)
                step = int(step)
                checkpoints.append((step, str(filepath)))

    # Sort by step number
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints

# =============================================================================
# WIND-RELATIVE COORDINATE TRANSFORMATIONS
# =============================================================================

def transform_to_wind_relative(
    positions: torch.Tensor,
    wind_direction: torch.Tensor
) -> torch.Tensor:
    """
    Transform positions to wind-relative coordinates (PyTorch version).

    Rotates the coordinate system so that wind effectively comes from 270°
    (negative x direction). This makes the learning problem invariant to
    absolute wind direction - the model only needs to learn wake patterns
    in a canonical reference frame.

    Physics intuition:
    - After transformation, "upwind" always means positive x direction
    - Wake effects always propagate in positive x direction
    - Lateral offset is always in y direction

    Args:
        positions: (batch, n_turbines, 2) raw positions in meters
        wind_direction: (batch,) wind direction in degrees (meteorological convention)

    Returns:
        Rotated positions with same shape as input
    """
    # Rotation angle: how much to rotate to align wind with 270°
    angle_offset = wind_direction - 270.0
    theta = angle_offset * (math.pi / 180.0)

    # Handle dimensions for broadcasting
    if theta.dim() == 1:
        theta = theta.unsqueeze(-1).unsqueeze(-1)  # (batch, 1, 1)

    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    x = positions[..., 0:1]
    y = positions[..., 1:2]

    # Standard 2D rotation matrix
    x_rot = cos_theta * x - sin_theta * y
    y_rot = sin_theta * x + cos_theta * y

    return torch.cat([x_rot, y_rot], dim=-1)


def transform_to_wind_relative_numpy(
    positions: np.ndarray,
    wind_direction: np.ndarray
) -> np.ndarray:
    """
    Transform positions to wind-relative coordinates (NumPy version).

    Thin wrapper around the torch version for numpy arrays.

    Args:
        positions: (batch, n_turbines, 2) or (n_turbines, 2) raw positions
        wind_direction: (batch,) or scalar wind direction in degrees

    Returns:
        Rotated positions with same shape as input
    """
    # Handle scalar/1D wind direction
    wind_direction = np.atleast_1d(wind_direction).astype(np.float32)

    # Add batch dim if needed for single sample
    squeeze_batch = positions.ndim == 2
    if squeeze_batch:
        positions = positions[np.newaxis, ...]

    # Convert to torch, transform, convert back
    pos_tensor = torch.from_numpy(positions)
    wd_tensor = torch.from_numpy(wind_direction)

    result = transform_to_wind_relative(pos_tensor, wd_tensor)
    result_np = result.numpy()

    if squeeze_batch:
        result_np = result_np[0]

    return result_np.astype(np.float32)


# =============================================================================
# WIND DIRECTION DEVIATION COMPUTATION
# =============================================================================

def compute_wind_direction_deviation(
    local_wd: np.ndarray,
    mean_wd: float,
    scale_range: float = 90.0
) -> np.ndarray:
    """
    Compute wind direction deviation from mean, scaled to [-1, 1].

    The deviation captures local flow variations due to wakes and turbulence,
    while being invariant to the absolute wind direction (which is encoded
    in the positional encoding via wind-relative coordinates).

    Args:
        local_wd: Per-turbine local wind direction in degrees, shape (n_turbines,)
                  or (n_turbines, history_length) for history
        mean_wd: Farm-level mean wind direction in degrees
        scale_range: Deviation range for scaling. Deviations beyond this are clipped.
                     Default 90° means ±90° maps to [-1, 1].

    Returns:
        Scaled deviation in [-1, 1], same shape as input
    """
    # Compute raw deviation
    deviation = local_wd - mean_wd

    # Wrap to [-180, 180]
    deviation = ((deviation + 180) % 360) - 180

    # Scale to [-1, 1] based on scale_range (±90° -> [-1, 1])
    scaled = np.clip(deviation / scale_range, -1.0, 1.0)

    return scaled.astype(np.float32)


# =============================================================================
# ENHANCED PER-TURBINE WRAPPER
# =============================================================================

class EnhancedPerTurbineWrapper(gym.Wrapper):
    """
    Enhanced wrapper that transforms wind direction observations to DEVIATION from mean.

    This wrapper sits on top of PerTurbineObservationWrapper and transforms the
    absolute wind direction values to deviations from the farm's mean wind direction.

    Why deviation instead of absolute?
    - With wind-relative positional encoding, absolute wind direction is redundant
    - Deviation captures local wake effects and turbulence
    - Makes observations invariant to global wind direction changes

    Observation order per turbine (from TurbMes.get_measurements):
        [probes..., ws..., wd..., yaw..., TI, power...]

    We detect the WD indices by reading the TurbMes configuration directly.
    """

    def __init__(
        self,
        env: gym.Env,
        wd_scale_range: float = 90.0,
    ):
        """
        Args:
            env: Base environment (should be wrapped with PerTurbineObservationWrapper)
            wd_scale_range: Range for wind direction deviation scaling.
                           ±wd_scale_range degrees maps to [-1, 1].
                           Default 90° means ±90° deviation → [-1, 1].
        """
        super().__init__(env)
        self.wd_scale_range = wd_scale_range

        # Get the base WindFarmEnv to access farm_measurements config
        self._base_env = self._get_base_env()

        # Detect wind direction indices from the measurement configuration
        self._wd_indices, self._wd_min, self._wd_max = self._detect_wd_indices()

        if self._wd_indices is not None:
            print(f"EnhancedPerTurbineWrapper: Found WD at indices {self._wd_indices} "
                  f"(scaling from [{self._wd_min}, {self._wd_max}] to deviation)")
        else:
            print("EnhancedPerTurbineWrapper: No wind direction in observations, "
                  "wrapper will pass through unchanged")

    def _get_base_env(self):
        """Unwrap to get the base WindFarmEnv."""
        env = self.env
        while hasattr(env, 'env'):
            env = env.env
        return env

    def _detect_wd_indices(self) -> Tuple[Optional[List[int]], float, float]:
        """
        Detect which observation indices correspond to wind direction.

        Reads directly from TurbMes configuration to determine:
        1. How many probe features come first
        2. How many WS features
        3. Where WD features start and how many there are

        Returns:
            (wd_indices, wd_min, wd_max) or (None, 0, 360) if WD not in obs
        """
        try:
            fm = self._base_env.farm_measurements
            turb_mes = fm.turb_mes[0]  # All turbines have same structure

            # Check if wind direction is enabled
            if not fm.turb_wd:
                return None, 0, 360

            # Count features before wind direction
            offset = 0

            # 1. Probes (if any)
            if hasattr(turb_mes, 'n_probes') and turb_mes.n_probes > 0:
                offset += turb_mes.n_probes

            # 2. Wind speed features
            ws_mes = turb_mes.ws
            n_ws = (1 if ws_mes.current else 0) + (ws_mes.history_N if ws_mes.rolling_mean else 0)
            offset += n_ws

            # 3. Wind direction features start at offset
            wd_mes = turb_mes.wd
            n_wd = (1 if wd_mes.current else 0) + (wd_mes.history_N if wd_mes.rolling_mean else 0)

            if n_wd == 0:
                return None, 0, 360

            wd_indices = list(range(offset, offset + n_wd))
            wd_min = turb_mes.wd_min
            wd_max = turb_mes.wd_max

            return wd_indices, wd_min, wd_max

        except Exception as e:
            print(f"Warning: Could not detect WD indices: {e}")
            return None, 0, 360

    def _transform_observation(self, obs: np.ndarray) -> np.ndarray:
        """
        Transform wind direction values to deviations from mean.

        Args:
            obs: Per-turbine observations, shape (n_turbines, obs_dim)

        Returns:
            Transformed observations with same shape
        """
        if self._wd_indices is None:
            return obs

        # Get mean wind direction from environment
        mean_wd = self._base_env.wd

        # Transform wind direction columns to deviations
        obs_transformed = obs.copy()

        for idx in self._wd_indices:
            if idx < obs.shape[1]:
                # Current values are scaled to [-1, 1] from [wd_min, wd_max]
                # Unscale: val = (scaled + 1) / 2 * (max - min) + min
                wd_scaled = obs[:, idx]
                wd_degrees = (wd_scaled + 1) / 2 * (self._wd_max - self._wd_min) + self._wd_min

                # Compute deviation from mean and rescale
                wd_dev_scaled = compute_wind_direction_deviation(
                    wd_degrees, mean_wd, self.wd_scale_range
                )

                obs_transformed[:, idx] = wd_dev_scaled

        return obs_transformed

    # Pass through all properties from the wrapped env
    @property
    def n_turbines(self) -> int:
        return self.env.n_turbines

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        obs_transformed = self._transform_observation(obs)
        return obs_transformed, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs_transformed = self._transform_observation(obs)
        return obs_transformed, reward, terminated, truncated, info


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
        "3turb": lambda: generate_square_grid(turbine=wind_turbine, nx=3, ny=1, xDist=5, yDist=5),
        "square_2x2": lambda: generate_square_grid(turbine=wind_turbine, nx=2, ny=2, xDist=5, yDist=5),
        "BIG": lambda: generate_square_grid(turbine=wind_turbine, nx=4, ny=4, xDist=6, yDist=6),
        "square_5x5": lambda: generate_square_grid(turbine=wind_turbine, nx=5, ny=5, xDist=6, yDist=6),
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
        "T1": lambda: generate_square_grid(turbine=wind_turbine, nx=3, ny=1, xDist=5, yDist=5),
        "T2": lambda: generate_square_grid(turbine=wind_turbine, nx=2, ny=2, xDist=5, yDist=5),
        "T3": lambda: generate_square_grid(turbine=wind_turbine, nx=4, ny=1, xDist=5, yDist=5),
        "T4": lambda: generate_right_triangle_grid(turbine=wind_turbine, nx=2, ny=2, xDist=5, yDist=5,orientation="lower_right"),
        "T5": lambda: generate_diamond_grid(wind_turbine, n=2, xDist=5, yDist=2.2),
        "T6": lambda: generate_right_triangle_grid(turbine=wind_turbine, nx=3, ny=3, xDist=5, yDist=5,orientation="lower_right"),
        "E1": lambda: generate_square_grid(turbine=wind_turbine, nx=3, ny=2, xDist=5, yDist=5),
        "E2": lambda: generate_square_grid(turbine=wind_turbine, nx=3, ny=3, xDist=5, yDist=5),
        "E3": lambda: generate_square_grid(turbine=wind_turbine, nx=4, ny=3, xDist=5, yDist=5),
        "E4": lambda: (lambda x, y: (y, x))(*generate_staggered_grid(turbine=wind_turbine, nx=2, ny=3, xDist=5, yDist=5, y_stagger_offset=[0, 2.5])),
        "E5": lambda: (np.array([730.0624, 444.7964, 1180.635, 93.5193, 1377.9328, 7.4321]),
                       np.array([1016.8061, 452.8746, 437.7612, 1031.7807, 1061.6575, 2.6086])),
        # --- Grid training layouts ---
        "g1": lambda: generate_square_grid(turbine=wind_turbine, nx=3, ny=2, xDist=5, yDist=5),
        "g2": lambda: generate_square_grid(turbine=wind_turbine, nx=2, ny=3, xDist=5, yDist=5),
        "g3": lambda: (np.array([5.0, 2.5, 7.5, 0.0, 5.0, 10.0]) * wind_turbine.diameter(),
                        np.array([10.0, 5.0, 5.0, 0.0, 0.0, 0.0]) * wind_turbine.diameter()),
        # --- Perturbed training layouts (~1D perturbation from grid) ---
        "p1": lambda: (np.array([0.9, 5.6, 9.2, -0.5, 4.2, 10.8]) * wind_turbine.diameter(),
                        np.array([-0.7, 1.1, -1.0, 5.8, 4.5, 5.5]) * wind_turbine.diameter()),
        "p2": lambda: (np.array([-0.7, 5.8, 1.0, 4.3, -0.4, 5.5]) * wind_turbine.diameter(),
                        np.array([0.9, -0.5, 5.7, 4.6, 10.8, 9.3]) * wind_turbine.diameter()),
        "p3": lambda: (np.array([5.8, 1.7, 8.4, 0.7, 4.3, 10.7]) * wind_turbine.diameter(),
                        np.array([10.7, 5.7, 4.2, 1.0, -0.7, 0.5]) * wind_turbine.diameter()),
        # --- regular training layouts --- found to match the irregular performance
        "r1": lambda: (np.array([0.0, 1248.1, 2496.2, 0.0, 1248.1, 2496.2]), 
                       np.array([0.0, 0.0, 0.0, 891.5, 891.5, 891.5])),
        "r2": lambda: (np.array([0.0, 713.2, 1426.4, 0.0, 713.2, 1426.4]), 
                       np.array([0.0, 0.0, 0.0, 713.2, 713.2, 713.2])),
        "r3": lambda: (np.array([0.0, 713.2, 0.0, 713.2, 0.0, 713.2]), 
                       np.array([0.0, 0.0, 713.2, 713.2, 1426.4, 1426.4])),
        # --- irregular training layouts --- found to match the regular performance
        "ir1": lambda: (np.array([1704.551048268463, 914.3306311063952, 1228.5520434497569, 258.5255410712933, 23.24574142543186, 1781.9532368498412]), 
                        np.array([222.17800042267135, 145.406792381138, 900.5017154692762, 797.6217097748043, 21.26233244323475, 1047.416107140278])),
        "ir2": lambda: (np.array([749.535392785272, 488.31018842263705, 1392.4616322926706, 111.8851795987488, 1747.4330560273631, 1248.5147109705342]), 
                        np.array([990.4950250849097, 64.23999669027903, 576.3010846379485, 885.2083834296628, 1059.4419436654625, 59.40283654123481])),
        "ir3": lambda: (np.array([653.1965502153917, 157.89957952486031, 818.9979507784572, 1518.3458685445662, 1190.252321095436, 40.8895846585575]), 
                        np.array([213.2061968554637, 698.7844673954643, 1056.6153770780397, 895.3812235989902, 441.623693251274, 159.96676745124617])),
        # --- Evaluation layouts ---
        "eval_grid": lambda: (np.array([0, 5, 10, 15, 2.5, 7.5, 12.5, 17.5]) * wind_turbine.diameter(),
                              np.array([0, 0, 0, 0, 5, 5, 5, 5]) * wind_turbine.diameter()),
        "eval_perturb": lambda: (np.array([1.0, 7.5, 4.0, 11.5, 9.0, 0.5, 14.0]) * wind_turbine.diameter(),
                                 np.array([1.5, 0.0, 6.0, 3.5, 8.5, 11.0, 9.5]) * wind_turbine.diameter()),
        # --- Eval layouts but matching the grid performance ---
        "eval_regular": lambda: (np.array([0.0, 891.5, 1783.0, 0.0, 891.5, 1783.0]), np.array([0.0, 0.0, 0.0, 891.5, 891.5, 891.5])),
        "eval_irregular": lambda: (np.array([1235.1614250818666, 614.0772491446635, 32.67493594582077, 590.169988163652, 70.89867195187087, 1603.8511283851528]), 
                                   np.array([872.7611457073143, 47.96788035361005, 951.2489425231132, 972.0429395286828, 366.8012544177941, 95.07779748241096])),
        # --- Larger scale layouts for generalization testing ---
        "20_turb_random_1": lambda: (np.array([1368.8669, 385.5549, 833.9932, 2213.6906, 1469.8883, 881.8675, 2571.9522, 1447.5112, 
                                 1667.5234, 105.8911, 511.6958, 2153.1192, 2055.3987, 2664.1796, 332.8692, 45.0046, 
                                 1804.4568, 2510.6918, 1274.7111, 13.9351]), 
                       np.array([2542.0152, 2537.1629, 1132.1866, 1094.4031, 73.7068, 2108.6526, 1938.4507, 740.5455, 
                                 2077.239, 1413.712, 218.1125, 2577.3377, 566.1241, 650.4798, 1961.9877, 811.7346, 
                                 1525.975, 60.4911, 1710.2677, 6.5215])),
        "20_turb_random_2": lambda: (np.array([229.0687, 2143.0086, 1281.2227, 1964.6266, 1046.3398, 1973.3472, 3.9852, 
                                               798.0741, 1260.5177, 81.1604, 2329.8617, 1502.5601, 571.9873, 2581.5114, 
                                               451.8786, 2572.3619, 1404.8292, 553.8118, 2541.4367, 2647.9218]), 
                                     np.array([633.3497, 1556.9924, 427.2217, 304.0158, 1382.0216, 2557.5368, 2603.5195, 
                                               839.7556, 2068.1294, 1890.7781, 736.4888, 1068.8806, 2296.4378, 1893.9332, 
                                               1370.9419, 2562.3502, 2615.497, 122.4296, 137.4661, 1318.8573])),
        "25_turb_random_1": lambda: (np.array([2271.4054, 146.1116, 2900.1217, 2163.2632, 1938.5667, 2909.3338, 3078.096, 1068.7726, 3556.0506, 481.7541, 3330.7992, 1396.5134, 2806.7926, 1198.5934, 1065.1504, 711.4721, 1782.6285, 52.4427, 1711.6369, 2190.855, 444.1621, 2090.1148, 3365.5959, 3301.0982, 2534.9976]), np.array([962.0594, 58.9375, 3254.8864, 2601.3847, 3334.4683, 9.7655, 1930.8507, 1507.3026, 3497.6588, 2572.8274, 1275.8977, 3174.7183, 853.5914, 535.8966, 2396.3337, 3359.5754, 1516.3653, 3079.7406, 828.6418, 101.1509, 1028.1875, 1975.8867, 452.2298, 2668.2542, 1641.6271])),
        "25_turb_random_2": lambda: (np.array([1825.1559, 514.0732, 1111.991, 2951.5874, 1959.8511, 1175.8233, 477.9927, 725.5214, 2675.8004, 3429.2696, 1930.015, 2223.3645, 141.1882, 682.2611, 3049.7394, 767.468, 2401.558, 2964.3261, 1413.0495, 2331.6862, 3552.2395, 60.0061, 3027.6928, 1887.2085, 70.7286]), np.array([3389.3535, 3382.8839, 1509.5821, 1459.2041, 98.2758, 2811.5368, 1437.5009, 935.4094, 999.9376, 2584.6009, 987.394, 2769.652, 1884.9493, 290.8166, 3071.3369, 2279.8557, 3277.47, 223.6521, 20.7705, 1537.7546, 867.3063, 1082.3128, 2159.8661, 2180.8933, 291.9056])),
    }
    
    if layout_type not in layouts:
        raise ValueError(f"Unknown layout type: {layout_type}. Available: {list(layouts.keys())}")
    
    return layouts[layout_type]()


def get_env_wind_directions(envs) -> np.ndarray:
    """Get current wind direction from each environment."""
    return np.array(envs.env.get_attr('wd'), dtype=np.float32)


def get_env_raw_positions(envs) -> np.ndarray:
    """Get raw (unnormalized) turbine positions from each environment."""
    return np.array(envs.env.get_attr('turbine_positions'), dtype=np.float32)

def get_env_receptivity_profiles(envs) -> np.ndarray:
    """Get receptivity profiles from vectorized environments.
    Returns:
        Shape (num_envs, max_turbines, n_directions)
    """
    if envs.env.get_attr('receptivity_profiles')[0] is None:
        print("The receptivity profiles are None")
        print("I dont know if this will cause issues later, so be careful")
    return np.array(envs.env.get_attr('receptivity_profiles'), dtype=np.float32)

def get_env_layout_indices(envs) -> List[int]:
    """Get current layout index for each env."""
    return list(envs.env.get_attr('current_layout_index'))

def get_env_permutations(envs) -> List[np.ndarray]:
    """Get current turbine permutation for each env."""
    return list(envs.env.get_attr('current_permutation'))

def get_env_influence_profiles(envs) -> np.ndarray:
    """Get influence profiles from vectorized environments.
    Returns:
        Shape (num_envs, max_turbines, n_directions)
    """
    if envs.env.get_attr('influence_profiles')[0] is None:
        print("The influence profiles are None")
        print("I dont know if this will cause issues later, so be careful")
    return np.array(envs.env.get_attr('influence_profiles'), dtype=np.float32)


def get_env_attention_masks(envs) -> np.ndarray:
    """Get attention masks from each environment."""
    return np.array(envs.env.get_attr('attention_mask'), dtype=bool)


def rotate_profiles_tensor(
    profiles: torch.Tensor,
    wind_directions: torch.Tensor,
) -> torch.Tensor:
    """
    Rotate profiles so current wind direction is at index 0.
    
    Args:
        profiles: (batch, max_turb, n_directions)
        wind_directions: (batch,) wind directions in degrees
        n_directions: Number of directions in profile
    
    Returns:
        Rotated profiles with same shape
    """
    n_directions = profiles.shape[2]
    degrees_per_index = 360.0 / n_directions

    shifts = (wind_directions / degrees_per_index).round().long()
    # Build shifted indices: (batch, 1, n_directions)
    base_idx = torch.arange(n_directions, device=profiles.device)
    indices = (base_idx[None, None, :] + shifts[:, None, None]) % n_directions
    return torch.gather(profiles, 2, indices.expand_as(profiles))


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



def load_old_sac_checkpoint(checkpoint_path: str, device: torch.device):
    """Load OLD format SAC checkpoint (tuple of state dicts)."""
    data = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Old format: (actor_state_dict, qf1_state_dict, qf2_state_dict)
    actor_state_dict, qf1_state_dict, qf2_state_dict = data
    
    # Remap old key names to new key names
    # Old: "backbone.X.weight" -> New: "net.X.weight"
    def remap_keys(state_dict, old_prefix="backbone", new_prefix="net"):
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace(old_prefix, new_prefix)
            new_state_dict[new_key] = value
        return new_state_dict
    
    actor_state_dict = remap_keys(actor_state_dict)
    
    # Extract step from filename (e.g., "sac_OLD_longer_e_seed1_1150020.pt" -> 1150020)
    import re
    match = re.search(r'_(\d+)\.pt$', checkpoint_path)
    step = int(match.group(1)) if match else 0
    
    # Create a checkpoint dict in the new format
    default_args = {
        "turbtype": "DTU10MW",
        "history_length": 15,
        "hidden_dim": 256,          # Adjust if different
        "num_hidden_layers": 2,     # Adjust if different
        "include_positions": False,  # Adjust if different
        "dt_sim": 5,
        "dt_env": 10,
        "yaw_step": 5.0,
        "max_eps": 20,
        "TI_type": "Random",
        "wd_scale_range": 90.0,
    }
    
    checkpoint = {
        "step": step,
        "actor_state_dict": actor_state_dict,
        "qf1_state_dict": qf1_state_dict,
        "qf2_state_dict": qf2_state_dict,
        "args": default_args,
    }
    
    return checkpoint, default_args


def _base_config() -> Dict[str, Any]:
    """Base environment configuration for transformer-based control."""
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
            "turb_wd": True,
            "turb_TI": False,
            "turb_power": True,
            "farm_ws": False,
            "farm_wd": False,
            "farm_TI": False,
            "farm_power": False,
        },
        "ws_mes": {
            "ws_current": False,
            "ws_rolling_mean": True,
            "ws_history_N": 15,
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


def _deep_update(base: dict, overrides: dict) -> dict:
    """Recursively update base dict with overrides."""
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


# Registry: name -> overrides from base
ENV_CONFIGS: Dict[str, Dict[str, Any]] = {
    "default": {},
    "big": {
        "BaseController": "PyWake",
        "wind": {
            "ws_min": 9, "ws_max": 9, 
            "wd_min": 225, "wd_max": 315
        },
    },
    # Easy to add more:
    "hard": {
        "wind": {
            "wd_min": 225, "wd_max": 315, 
            "ws_min": 10, "ws_max": 14,
        },  
    },

    "basic": {
        "yaw_init": "Random",
        "BaseController": "Local",
        "ActionMethod": "yaw",
        "Track_power": False,
        "farm": {"yaw_min": -30, "yaw_max": 30},
        "wind": {
            "ws_min": 10, "ws_max": 14,
            "TI_min": 0.07, "TI_max": 0.07,
            "wd_min": 225, "wd_max": 315,
        },
        "act_pen": {"action_penalty": 0.0, "action_penalty_type": "Change"},
        "power_def": {"Power_reward": "Power_avg", "Power_avg": 1, "Power_scaling": 1.0},
        "mes_level": {
            "turb_ws": True,
            "turb_wd": True,
            "turb_TI": False,
            "turb_power": True,
            "farm_ws": False,
            "farm_wd": False,
            "farm_TI": False,
            "farm_power": False,
        },
        "ws_mes": {
            "ws_current": True,
            "ws_rolling_mean": False,
            "ws_history_N": 1,
            "ws_history_length": 1,
            "ws_window_length": 1,
        },
        "wd_mes": {
            "wd_current": True,
            "wd_rolling_mean": False,
            "wd_history_N": 15,
            "wd_history_length": 15,
            "wd_window_length": 1,
        },
        "yaw_mes": {
            "yaw_current": True,
            "yaw_rolling_mean": False,
            "yaw_history_N": 1,
            "yaw_history_length": 1,
            "yaw_window_length": 1,
        },
        "power_mes": {
            "power_current": True,
            "power_rolling_mean": False,
            "power_history_N": 1,
            "power_history_length": 1,
            "power_window_length": 1,
        },
    },

    "wide": {
        "wind": {
            "wd_min": 250, "wd_max": 290, 
            "ws_min": 10, "ws_max": 10,
            "TI_min": 0.07, "TI_max": 0.07,
        },  
    },
    
    "20deg_wd": {
        "wind": {
            "wd_min": 250, "wd_max": 290, 
        },  
    },
}


def make_env_config(name: str = "default") -> Dict[str, Any]:
    """Build an env config by name. Applies overrides on top of the base config."""
    if name not in ENV_CONFIGS:
        available = ", ".join(sorted(ENV_CONFIGS.keys()))
        raise ValueError(f"Unknown env config '{name}'. Available: {available}")
    
    config = deepcopy(_base_config())
    return _deep_update(config, deepcopy(ENV_CONFIGS[name]))


def load_checkpoint(
    checkpoint_path: str,
    actor: nn.Module,
    qf1: nn.Module,
    qf2: nn.Module,
    qf1_target: nn.Module,
    qf2_target: nn.Module,
    actor_optimizer: optim.Optimizer,
    q_optimizer: optim.Optimizer,
    device: torch.device,
    log_alpha: Optional[torch.Tensor] = None,
    alpha_optimizer: Optional[optim.Optimizer] = None,
) -> int:
    """
    Load training checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        actor: Actor network
        qf1: First critic network
        qf2: Second critic network
        qf1_target: First target critic network
        qf2_target: Second target critic network
        actor_optimizer: Actor optimizer
        q_optimizer: Critic optimizer
        device: Torch device
        log_alpha: Optional entropy coefficient (log scale)
        alpha_optimizer: Optional entropy optimizer

    Returns:
        Step number from checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    actor.load_state_dict(checkpoint["actor_state_dict"])
    qf1.load_state_dict(checkpoint["qf1_state_dict"])
    qf2.load_state_dict(checkpoint["qf2_state_dict"])
    qf1_target.load_state_dict(checkpoint["qf1_state_dict"])
    qf2_target.load_state_dict(checkpoint["qf2_state_dict"])
    actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
    q_optimizer.load_state_dict(checkpoint["q_optimizer_state_dict"])

    if log_alpha is not None and "log_alpha" in checkpoint:
        log_alpha.data = checkpoint["log_alpha"].to(device)
    if alpha_optimizer is not None and "alpha_optimizer_state_dict" in checkpoint:
        alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer_state_dict"])

    print(f"Loaded checkpoint from {checkpoint_path} at step {checkpoint['step']}")

    return checkpoint["step"]


def prepare_observation_with_positions(
    obs: np.ndarray,
    raw_positions: np.ndarray,
    wind_directions: np.ndarray,
    rotor_diameter: float,
    include_positions: bool = True,
) -> np.ndarray:
    """
    Prepare observations with position features for MLP-based networks.

    Args:
        obs: (num_envs, max_turb, obs_dim) per-turbine observations
        raw_positions: (num_envs, max_turb, 2) raw positions
        wind_directions: (num_envs,) wind directions
        rotor_diameter: For normalization
        include_positions: Whether to include positions

    Returns:
        Flattened observations: (num_envs, total_dim)
    """
    if include_positions:
        # Normalize and transform positions
        positions_norm = raw_positions / rotor_diameter
        positions_transformed = transform_to_wind_relative_numpy(
            positions_norm, wind_directions
        )

        # Concatenate and flatten
        obs_with_pos = np.concatenate([obs, positions_transformed], axis=-1)
        return obs_with_pos.reshape(obs.shape[0], -1).astype(np.float32)
    else:
        return obs.reshape(obs.shape[0], -1).astype(np.float32)