"""
Wind Farm Agent wrapper for Transformer SAC.

This module provides a clean interface between the actor network and environment
interaction, centralizing all tensor preparation logic in one place.

Usage:
    agent = WindFarmAgent(
        actor=actor,
        device=device,
        rotor_diameter=rotor_diameter,
        use_wind_relative=True,
        use_profiles=True,
        rotate_profiles=False,
    )
    
    # In training/evaluation loop:
    actions = agent.act(envs, obs)

Author: Marcus (DTU Wind Energy)
"""

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from typing import Optional, Tuple, List
from dataclasses import dataclass

from .helper_funcs import transform_to_wind_relative, rotate_profiles_tensor


# =============================================================================
# BATCH PREPARATION
# =============================================================================

@dataclass
class InferenceBatch:
    """Prepared batch ready for actor inference."""
    obs: torch.Tensor                       # (batch, n_turbines, obs_dim)
    positions: torch.Tensor                 # (batch, n_turbines, 2)
    mask: torch.Tensor                      # (batch, n_turbines)
    receptivity: Optional[torch.Tensor] = None   # (batch, n_turbines, n_directions)
    influence: Optional[torch.Tensor] = None     # (batch, n_turbines, n_directions)


class BatchPreparer:
    """
    Prepares batches for actor inference from environment state.
    
    Handles:
    - Querying environment for positions, wind direction, masks, profiles
    - Normalizing positions by rotor diameter
    - Optionally transforming to wind-relative coordinates
    - Optionally rotating profiles to wind-relative frame
    - Converting everything to tensors on the correct device
    """
    
    def __init__(
        self,
        device: torch.device,
        rotor_diameter: float,
        use_wind_relative: bool = True,
        use_profiles: bool = False,
        rotate_profiles: bool = False,
    ):
        """
        Args:
            device: Torch device for tensors
            rotor_diameter: Rotor diameter for position normalization
            use_wind_relative: Whether to transform positions to wind-relative frame
            use_profiles: Whether to include receptivity/influence profiles
            rotate_profiles: Whether to rotate profiles to wind-relative frame
        """
        self.device = device
        self.rotor_diameter = rotor_diameter
        self.use_wind_relative = use_wind_relative
        self.use_profiles = use_profiles
        self.rotate_profiles = rotate_profiles
    
    def from_envs(
        self,
        envs: gym.vector.VectorEnv,
        obs: np.ndarray,
    ) -> InferenceBatch:
        """
        Prepare batch from vectorized environment state.
        
        Args:
            envs: Vectorized environment (AsyncVectorEnv or SyncVectorEnv)
            obs: Current observations, shape (num_envs, n_turbines, obs_dim)
        
        Returns:
            InferenceBatch ready for actor.get_action()
        """
        num_envs = obs.shape[0]
        
        # Query environment state
        wind_dirs = np.array(envs.env.get_attr('wd'), dtype=np.float32)
        raw_positions = np.array(envs.env.get_attr('turbine_positions'), dtype=np.float32)
        masks = np.array(envs.env.get_attr('attention_mask'), dtype=bool)
        
        # Convert observations to tensor
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
        mask_tensor = torch.tensor(masks, dtype=torch.bool, device=self.device)
        
        # Normalize positions by rotor diameter
        positions_norm = raw_positions / self.rotor_diameter
        positions_tensor = torch.tensor(positions_norm, dtype=torch.float32, device=self.device)
        
        # Optionally transform to wind-relative coordinates
        wind_dir_tensor = None
        if self.use_wind_relative:
            wind_dir_tensor = torch.tensor(wind_dirs, dtype=torch.float32, device=self.device)
            positions_tensor = transform_to_wind_relative(positions_tensor, wind_dir_tensor)
        
        # Handle profiles if enabled
        receptivity_tensor = None
        influence_tensor = None
        
        if self.use_profiles:
            # Query profiles from environment
            receptivity = np.array(
                envs.env.get_attr('receptivity_profiles'), dtype=np.float32
            )
            influence = np.array(
                envs.env.get_attr('influence_profiles'), dtype=np.float32
            )
            
            receptivity_tensor = torch.tensor(receptivity, dtype=torch.float32, device=self.device)
            influence_tensor = torch.tensor(influence, dtype=torch.float32, device=self.device)
            
            # Optionally rotate profiles to wind-relative frame
            if self.rotate_profiles:
                if wind_dir_tensor is None:
                    wind_dir_tensor = torch.tensor(wind_dirs, dtype=torch.float32, device=self.device)
                receptivity_tensor = rotate_profiles_tensor(receptivity_tensor, wind_dir_tensor)
                influence_tensor = rotate_profiles_tensor(influence_tensor, wind_dir_tensor)
        
        return InferenceBatch(
            obs=obs_tensor,
            positions=positions_tensor,
            mask=mask_tensor,
            receptivity=receptivity_tensor,
            influence=influence_tensor,
        )


# =============================================================================
# WIND FARM AGENT
# =============================================================================

class WindFarmAgent:
    """
    Wraps actor network with environment interaction logic.
    
    This provides a clean interface for both training and evaluation,
    ensuring consistent tensor preparation across all use cases.
    
    The agent handles:
    - Batch preparation from environment state
    - Action selection (deterministic or stochastic)
    - Train/eval mode switching
    
    Example:
        agent = WindFarmAgent(actor, device, rotor_diameter, ...)
        
        # Training loop
        actions = agent.act(envs, obs)
        
        # Evaluation
        actions = agent.act(envs, obs, deterministic=True)
    """
    
    def __init__(
        self,
        actor: nn.Module,
        device: torch.device,
        rotor_diameter: float,
        use_wind_relative: bool = True,
        use_profiles: bool = False,
        rotate_profiles: bool = False,
    ):
        """
        Args:
            actor: TransformerActor network
            device: Torch device
            rotor_diameter: Rotor diameter for position normalization
            use_wind_relative: Whether to transform positions to wind-relative frame
            use_profiles: Whether to use receptivity/influence profiles
            rotate_profiles: Whether to rotate profiles to wind-relative frame
        """
        self.actor = actor
        self.device = device
        
        self.batch_preparer = BatchPreparer(
            device=device,
            rotor_diameter=rotor_diameter,
            use_wind_relative=use_wind_relative,
            use_profiles=use_profiles,
            rotate_profiles=rotate_profiles,
        )
    
    def act(
        self,
        envs: gym.vector.VectorEnv,
        obs: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        """
        Select actions given current environment state.
        
        Args:
            envs: Vectorized environment
            obs: Current observations, shape (num_envs, n_turbines, obs_dim)
            deterministic: If True, use mean action. If False, sample stochastically.
        
        Returns:
            actions: Action array, shape (num_envs, n_turbines)
        """
        batch = self.batch_preparer.from_envs(envs, obs)
        
        with torch.no_grad():
            action_tensor, _, _, _ = self.actor.get_action(
                batch.obs,
                batch.positions,
                batch.mask,
                deterministic=deterministic,
                recep_profile=batch.receptivity,
                influence_profile=batch.influence,
            )
        
        # Remove action_dim dimension and convert to numpy
        return action_tensor.squeeze(-1).cpu().numpy()
    
    def act_with_log_prob(
        self,
        envs: gym.vector.VectorEnv,
        obs: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Select actions and return additional info (for training diagnostics).
        
        Args:
            envs: Vectorized environment
            obs: Current observations
            deterministic: If True, use mean action
        
        Returns:
            actions: Action array (numpy)
            log_prob: Log probability tensor
            mean_action: Mean action tensor
            attn_weights: List of attention weight tensors
        """
        batch = self.batch_preparer.from_envs(envs, obs)
        
        with torch.no_grad():
            action_tensor, log_prob, mean_action, attn_weights = self.actor.get_action(
                batch.obs,
                batch.positions,
                batch.mask,
                deterministic=deterministic,
                recep_profile=batch.receptivity,
                influence_profile=batch.influence,
            )
        
        return action_tensor.squeeze(-1).cpu().numpy(), log_prob, mean_action, attn_weights
    
    def train(self) -> None:
        """Set actor to training mode."""
        self.actor.train()
    
    def eval(self) -> None:
        """Set actor to evaluation mode."""
        self.actor.eval()
    
    @property
    def parameters(self):
        """Access actor parameters (for optimizer)."""
        return self.actor.parameters()
    
    def state_dict(self):
        """Get actor state dict (for checkpointing)."""
        return self.actor.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load actor state dict."""
        self.actor.load_state_dict(state_dict)