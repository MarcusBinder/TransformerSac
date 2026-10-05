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
    params: Optional[torch.Tensor] = None        # (batch, n_turbines, cond_dim) hidden params e (Stage 10)
    action_hist: Optional[torch.Tensor] = None   # (batch, n_turbines, n_feat) commanded-action history


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
        obs_normalizer=None,
        wd_attr: str = 'wd',
        cond_normalizer=None,
        use_action_hist: bool = False,
    ):
        """
        Args:
            device: Torch device for tensors
            rotor_diameter: Rotor diameter for position normalization
            use_wind_relative: Whether to transform positions to wind-relative frame
            use_profiles: Whether to include receptivity/influence profiles
            rotate_profiles: Whether to rotate profiles to wind-relative frame
            cond_normalizer: Optional helpers.param_cond.ParamNormalizer. When set
                and no `params` are passed to from_envs, e is fetched from the
                GIVEN envs (turbine_params_tok / farm_params / attention_mask
                get_attrs), so the nominal eval envs resolve their own nominal
                e without any DR wrapper (Stage 10).
            use_action_hist: fetch `action_history` from the envs when not
                passed (needs ActionHistoryWrapper in the chain).
            obs_normalizer: Optional ObsRunningNorm (--obs_norm). Applied to the
                obs tensor here so training AND eval act() calls are normalized
                identically (PolicyEvaluator shares this agent).
            wd_attr: env attribute fetched when wind_dirs is not passed in:
                'wd' (privileged true value) or 'wd_est' (sensor-derived
                estimate, --wd_source est). Covers every act() caller that
                lets the agent fetch for itself — notably PolicyEvaluator.
        """
        self.device = device
        self.rotor_diameter = rotor_diameter
        self.use_wind_relative = use_wind_relative
        self.use_profiles = use_profiles
        self.rotate_profiles = rotate_profiles
        self.obs_normalizer = obs_normalizer
        self.wd_attr = wd_attr
        self.cond_normalizer = cond_normalizer
        self.use_action_hist = use_action_hist
    
    def from_envs(
        self,
        envs: gym.vector.VectorEnv,
        obs: np.ndarray,
        wind_dirs: Optional[np.ndarray] = None,
        raw_positions: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
        receptivity: Optional[np.ndarray] = None,
        influence: Optional[np.ndarray] = None,
        params: Optional[np.ndarray] = None,
        action_hist: Optional[np.ndarray] = None,
    ) -> InferenceBatch:
        """
        Prepare batch from vectorized environment state.

        Args:
            envs: Vectorized environment (AsyncVectorEnv or SyncVectorEnv)
            obs: Current observations, shape (num_envs, n_turbines, obs_dim)
            wind_dirs/raw_positions/masks: optional precomputed env state. When
                provided (e.g. the training loop already fetched them), they are
                used directly to avoid duplicate get_attr IPC to the async
                workers. Any left as None is queried here.
            receptivity/influence: optional precomputed layout profiles, shape
                (num_envs, n_turbines, n_directions). Only consulted when
                use_profiles is on; when None they are fetched from the envs
                via get_attr (which needs a MultiLayoutEnv-wrapped vector env,
                so unvectorized eval drivers must pass them in).
            params/action_hist: optional precomputed (num_envs, n_turbines, d)
                hidden-parameter vectors / commanded-action histories (Stage 10).
                When None they are fetched from the envs iff cond_normalizer /
                use_action_hist are set; otherwise left None.

        Returns:
            InferenceBatch ready for actor.get_action()
        """
        num_envs = obs.shape[0]

        # Query environment state (skip the IPC for anything passed in)
        if wind_dirs is None:
            wind_dirs = np.array(envs.env.get_attr(self.wd_attr),
                                 dtype=np.float32)
        if raw_positions is None:
            raw_positions = np.array(envs.env.get_attr('turbine_positions'), dtype=np.float32)
        if masks is None:
            masks = np.array(envs.env.get_attr('attention_mask'), dtype=bool)

        # Convert observations to tensor
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
        mask_tensor = torch.tensor(masks, dtype=torch.bool, device=self.device)
        if self.obs_normalizer is not None:
            obs_tensor = self.obs_normalizer.normalize(obs_tensor)
        
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
            # Query profiles from environment (skip the IPC when precomputed)
            if receptivity is None:
                receptivity = np.array(
                    envs.env.get_attr('receptivity_profiles'), dtype=np.float32
                )
            if influence is None:
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
        
        params_tensor = None
        if params is None and self.cond_normalizer is not None:
            from .param_cond import CondFetcher
            params = CondFetcher(envs.env, self.cond_normalizer).refresh()
        if params is not None:
            params_tensor = torch.as_tensor(np.asarray(params, dtype=np.float32), device=self.device)
        action_hist_tensor = None
        if action_hist is None and self.use_action_hist:
            action_hist = np.array(envs.env.get_attr('action_history'), dtype=np.float32)
        if action_hist is not None:
            action_hist_tensor = torch.as_tensor(np.asarray(action_hist, dtype=np.float32), device=self.device)

        return InferenceBatch(
            obs=obs_tensor,
            positions=positions_tensor,
            mask=mask_tensor,
            receptivity=receptivity_tensor,
            influence=influence_tensor,
            params=params_tensor,
            action_hist=action_hist_tensor,
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
        obs_normalizer=None,
        wd_attr: str = 'wd',
        cond_normalizer=None,
        use_action_hist: bool = False,
        adaptation: Optional[nn.Module] = None,
    ):
        """
        Args:
            actor: TransformerActor network
            device: Torch device
            rotor_diameter: Rotor diameter for position normalization
            use_wind_relative: Whether to transform positions to wind-relative frame
            use_profiles: Whether to use receptivity/influence profiles
            rotate_profiles: Whether to rotate profiles to wind-relative frame
            obs_normalizer: Optional ObsRunningNorm (--obs_norm), see BatchPreparer
            wd_attr: wd source attribute ('wd' or 'wd_est'), see BatchPreparer
            cond_normalizer / use_action_hist: see BatchPreparer (Stage 10)
            adaptation: optional AdaptationModule phi. When set, act() feeds the
                actor z_hat = phi(obs, action_hist, ...) as a latent instead of
                the true e (deployment / phase-2 DAgger rollouts).
        """
        self.actor = actor
        self.device = device
        self.adaptation = adaptation
        # cond contract: the actor's cond_source decides whether cond is a tensor
        # (always) or None (always) -- never mixed (torch.compile single graph).
        self.cond_on = bool(getattr(actor, "cond_on", False))
        if adaptation is not None and not use_action_hist:
            use_action_hist = True

        self.batch_preparer = BatchPreparer(
            device=device,
            rotor_diameter=rotor_diameter,
            use_wind_relative=use_wind_relative,
            use_profiles=use_profiles,
            rotate_profiles=rotate_profiles,
            obs_normalizer=obs_normalizer,
            wd_attr=wd_attr,
            cond_normalizer=cond_normalizer if self.cond_on else None,
            use_action_hist=use_action_hist,
        )

    def _cond_kwargs(self, batch: InferenceBatch) -> dict:
        """cond / cond_is_latent for the actor from a prepared batch."""
        if not self.cond_on:
            return {}
        if self.adaptation is not None:
            if batch.action_hist is None:
                raise ValueError("adaptation module needs action_hist (use_action_hist / ActionHistoryWrapper)")
            z_hat = self.adaptation(batch.obs, batch.action_hist, batch.positions, batch.mask,
                                    batch.receptivity, batch.influence)
            return {"cond": z_hat, "cond_is_latent": True}
        if batch.params is None:
            raise ValueError("cond actor needs params: pass params= or give the agent a cond_normalizer")
        return {"cond": batch.params}
    
    def act(
        self,
        envs: gym.vector.VectorEnv,
        obs: np.ndarray,
        deterministic: bool = False,
        wind_dirs: Optional[np.ndarray] = None,
        raw_positions: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
        receptivity: Optional[np.ndarray] = None,
        influence: Optional[np.ndarray] = None,
        params: Optional[np.ndarray] = None,
        action_hist: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Select actions given current environment state.

        Args:
            envs: Vectorized environment
            obs: Current observations, shape (num_envs, n_turbines, obs_dim)
            deterministic: If True, use mean action. If False, sample stochastically.
            wind_dirs/raw_positions/masks/receptivity/influence: optional
                precomputed env state forwarded to BatchPreparer.from_envs to
                avoid duplicate get_attr IPC (or to supply profiles when there
                is no vector env at all).

        Returns:
            actions: Action array, shape (num_envs, n_turbines)
        """
        batch = self.batch_preparer.from_envs(
            envs, obs, wind_dirs=wind_dirs, raw_positions=raw_positions, masks=masks,
            receptivity=receptivity, influence=influence,
            params=params, action_hist=action_hist,
        )
        
        with torch.no_grad():
            action_tensor, _, _, _ = self.actor.get_action(
                batch.obs,
                batch.positions,
                batch.mask,
                deterministic=deterministic,
                recep_profile=batch.receptivity,
                influence_profile=batch.influence,
                **self._cond_kwargs(batch),
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
                **self._cond_kwargs(batch),
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