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

from typing import Optional, Tuple, List, Dict, Any

def soft_update(source, target, tau):
    with torch.no_grad():
        source_params = list(source.parameters())
        target_params = list(target.parameters())
        torch._foreach_lerp_(target_params, source_params, tau)


# =============================================================================
# CHECKPOINT UTILITIES
# =============================================================================


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
                step = match.group(1)
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
    - After transformation, "downwind" always means positive x direction
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
        print("Warning: receptivity profiles are None")
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
        print("Warning: influence profiles are None")
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
    qf1: Optional[nn.Module],
    qf2: Optional[nn.Module],
    actor_optimizer: optim.Optimizer,
    q_optimizer: optim.Optimizer,
    step: int,
    run_name: str,
    args: Any,
    log_alpha: Optional[torch.Tensor] = None,
    alpha_optimizer: Optional[optim.Optimizer] = None,
    tqc_critic: Optional[nn.Module] = None,
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
        "actor_optimizer_state_dict": actor_optimizer.state_dict(),
        "q_optimizer_state_dict": q_optimizer.state_dict(),
        "args": vars(args),
    }

    if tqc_critic is not None:
        checkpoint["tqc_critic_state_dict"] = tqc_critic.state_dict()
    else:
        checkpoint["qf1_state_dict"] = qf1.state_dict()
        checkpoint["qf2_state_dict"] = qf2.state_dict()

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


def load_checkpoint(
    checkpoint_path: str,
    actor: nn.Module,
    qf1: Optional[nn.Module],
    qf2: Optional[nn.Module],
    qf1_target: Optional[nn.Module],
    qf2_target: Optional[nn.Module],
    actor_optimizer: optim.Optimizer,
    q_optimizer: optim.Optimizer,
    device: torch.device,
    log_alpha: Optional[torch.Tensor] = None,
    alpha_optimizer: Optional[optim.Optimizer] = None,
    tqc_critic: Optional[nn.Module] = None,
    tqc_critic_target: Optional[nn.Module] = None,
) -> int:
    """
    Load training checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        actor: Actor network
        qf1: First critic network (None for TQC)
        qf2: Second critic network (None for TQC)
        qf1_target: First target critic network (None for TQC)
        qf2_target: Second target critic network (None for TQC)
        actor_optimizer: Actor optimizer
        q_optimizer: Critic optimizer
        device: Torch device
        log_alpha: Optional entropy coefficient (log scale)
        alpha_optimizer: Optional entropy optimizer
        tqc_critic: TQC critic (None for SAC)
        tqc_critic_target: TQC target critic (None for SAC)

    Returns:
        Step number from checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    actor.load_state_dict(checkpoint["actor_state_dict"])

    if tqc_critic is not None and "tqc_critic_state_dict" in checkpoint:
        tqc_critic.load_state_dict(checkpoint["tqc_critic_state_dict"])
        tqc_critic_target.load_state_dict(checkpoint["tqc_critic_state_dict"])
    else:
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