"""
Agent-side running observation normalization (--obs_norm, change_wd_4).

Agent-side is the only correct placement in this codebase: a per-env
normalization wrapper would keep 30 INDEPENDENT statistics in the async worker
processes (no way to sync them), and the lazily re-created eval envs would
start cold every cycle. Here one normalizer lives in the main process, updates
from the training stream, and is applied (a) at act() time via BatchPreparer
and (b) on replay batches right after rb.sample — the buffer stores RAW obs,
so one insertion point covers critic, actor and diagnostics, and the
statistics used for a batch are always the freshest ones.

Updates run from step 0 (the random-exploration warmup before learning_starts
already produces on-distribution observations). Pad rows are masked out of the
update — MultiLayoutEnv pads with 0.0, which would otherwise drag every mean
toward zero on small farms.
"""

from typing import Optional

import torch


class ObsRunningNorm:
    """Per-feature running mean/var over the per-turbine observation vector.

    Uses the parallel (Chan et al.) mean/M2 merge, so each masked batch folds
    in with one pass. All state lives in torch tensors on `device`.
    """

    def __init__(self, obs_dim: int, device: torch.device, clip: float = 10.0,
                 eps: float = 1e-8):
        self.obs_dim = obs_dim
        self.clip = clip
        self.eps = eps
        self.mean = torch.zeros(obs_dim, device=device)
        self.var = torch.ones(obs_dim, device=device)
        self.count = torch.zeros((), device=device)

    @torch.no_grad()
    def update(self, obs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> None:
        """Fold a batch of observations into the running statistics.

        Args:
            obs: (..., n_turbines, obs_dim) tensor (any leading batch dims)
            mask: (..., n_turbines) bool, True = PADDING row (attention-mask
                convention). None = all rows real.
        """
        x = obs.reshape(-1, self.obs_dim).float()
        if mask is not None:
            real = ~mask.reshape(-1)
            x = x[real]
        n = x.shape[0]
        if n == 0:
            return

        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)

        delta = batch_mean - self.mean
        tot = self.count + n
        self.mean += delta * (n / tot)
        m2 = self.var * self.count + batch_var * n + delta.pow(2) * (self.count * n / tot)
        self.var = m2 / tot
        self.count = tot

    def normalize(self, obs: torch.Tensor) -> torch.Tensor:
        """(obs - mean) / sqrt(var + eps), clipped. Identity-ish before the
        first update (mean 0, var 1). Broadcasts over leading dims."""
        return torch.clamp((obs - self.mean) / torch.sqrt(self.var + self.eps),
                           -self.clip, self.clip)

    def state_dict(self) -> dict:
        return {
            "mean": self.mean.detach().cpu(),
            "var": self.var.detach().cpu(),
            "count": self.count.detach().cpu(),
        }

    def load_state_dict(self, state: dict) -> None:
        device = self.mean.device
        self.mean = state["mean"].to(device)
        self.var = state["var"].to(device)
        self.count = state["count"].to(device)
