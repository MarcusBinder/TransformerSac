"""Custom multi-head attention for v5 size-generalization experiments.

Drop-in replacement for ``nn.MultiheadAttention`` (single in_proj / out_proj, same
math) that additionally supports, behind flags:

  * log-N logit scaling ("Scalable-Softmax", Nakanishi 2025, arXiv:2501.19399):
    scores *= softplus(s_h) * log(n_real). Restores attention sharpness as the
    number of turbines N grows, countering "attention dilution" (softmax over many
    keys flattens toward 1/N). s_h is a learned per-head scalar.
  * hard LOCAL attention: each turbine attends only to its k nearest neighbours
    ("knn") or to neighbours within a radius in rotor diameters ("radius"). Makes
    the N=25 per-token neighbourhood statistically identical to N=9.
  * entmax15 sparse normalisation (optional; needs the `entmax` package).

With ``logit_scale="none"``, ``local`` disabled and ``softmax_type="softmax"`` it
is numerically identical to ``nn.MultiheadAttention`` (verified by unit test).

Positions are expected in ROTOR-DIAMETER units (the model normalises by D), so the
radius threshold is directly in D.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def neighbour_allow_mask(
    positions: torch.Tensor,                 # (B, N, 2) in rotor-diameter units
    key_padding_mask: Optional[torch.Tensor],  # (B, N) True = padded
    mode: str,                               # "radius" | "knn" | "downwind" | "downwind_knn"
    radius_D: float = 10.0,
    k: int = 5,
    cone_deg: float = 40.0,
) -> torch.Tensor:
    """Return (B, N, N) bool: True where query i is ALLOWED to attend to key j.

    Self-attention (diagonal) is always allowed. Padded keys are never selected by
    knn (their distance is set to +inf); for radius they may pass the threshold but
    are removed later by the key_padding_mask in the attention itself.

    Modes:
      * "radius" / "knn": undirected locality (v5).
      * "downwind" / "downwind_knn": DIRECTED "causal wake graph" (v6). Query i attends
        key j only if j is UPWIND of i (a wake source) AND inside i's upstream cone of
        half-angle ``cone_deg``. Encodes wake causality (wakes propagate downstream only)
        and the upwind/downwind asymmetry. Positions are wind-relative with +x = upwind
        (same convention as positional_encodings._bias), so rel[i,j] = pos[j] - pos[i]
        has dx = x_j - x_i > 0 exactly when j is upwind of i.
        "downwind"     keeps every upwind in-cone source within ``radius_D`` (streamwise cap).
        "downwind_knn" keeps the k nearest upwind in-cone sources -> COUNT-INVARIANT local
        structure (a turbine's neighbourhood at N=25 matches N=9; the size-generalisation
        "d-pattern" fix, Yehudai et al. 2021).
    """
    B, N, _ = positions.shape
    dist = torch.cdist(positions, positions)  # (B, N, N)
    if mode == "radius":
        allow = dist <= radius_D
    elif mode == "knn":
        dd = dist.clone()
        if key_padding_mask is not None:
            dd = dd.masked_fill(key_padding_mask.unsqueeze(1), float("inf"))  # (B,1,N) padded keys
        kk = min(max(k, 1), N)
        idx = dd.topk(kk, dim=-1, largest=False).indices  # (B, N, kk)
        allow = torch.zeros_like(dist, dtype=torch.bool)
        allow.scatter_(-1, idx, True)
    elif mode in ("downwind", "downwind_knn"):
        # rel[i,j] = pos[j] - pos[i] (i = query dim 1, j = key dim 2), matching _bias.py.
        rel = positions.unsqueeze(1) - positions.unsqueeze(2)  # (B, N, N, 2)
        dx, dy = rel[..., 0], rel[..., 1]                      # dx>0: j upwind of i
        tan_half = math.tan(math.radians(cone_deg))
        in_cone = (dx > 0) & (dy.abs() <= tan_half * dx)       # j is an upwind wake source for i
        if mode == "downwind":
            allow = in_cone & (dist <= radius_D)
        else:  # downwind_knn: k nearest in-cone upwind sources
            dd = dist.masked_fill(~in_cone, float("inf"))
            if key_padding_mask is not None:
                dd = dd.masked_fill(key_padding_mask.unsqueeze(1), float("inf"))
            kk = min(max(k, 1), N)
            idx = dd.topk(kk, dim=-1, largest=False).indices   # (B, N, kk)
            allow = torch.zeros_like(dist, dtype=torch.bool)
            allow.scatter_(-1, idx, True)
            allow &= torch.isfinite(dd)                        # drop padded-out topk slots when <k sources
    else:
        raise ValueError(f"unknown local mode {mode!r}")
    eye = torch.eye(N, dtype=torch.bool, device=positions.device).unsqueeze(0)
    return allow | eye


class MaskedScaledAttention(nn.Module):
    """Multi-head self-attention with optional log-N scaling + local masking.

    Mirrors nn.MultiheadAttention(batch_first=True) parameterisation: a single
    ``in_proj`` (3*embed_dim) and ``out_proj``, so weights are interchangeable.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        logit_scale: str = "none",      # "none" | "logn"
        softmax_type: str = "softmax",  # "softmax" | "entmax15"
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.logit_scale = logit_scale
        self.softmax_type = softmax_type

        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        if logit_scale == "logn":
            # per-head scalar s_h, passed through softplus so the multiplier is >0
            self.log_n_scale = nn.Parameter(torch.zeros(num_heads))
        elif logit_scale != "none":
            raise ValueError(f"unknown logit_scale {logit_scale!r}")

    def forward(
        self,
        x: torch.Tensor,                              # (B, N, E)
        key_padding_mask: Optional[torch.Tensor] = None,  # (B, N) True = padded
        attn_bias: Optional[torch.Tensor] = None,     # (B, num_heads, N, N) additive
        local_allow: Optional[torch.Tensor] = None,   # (B, N, N) bool, True = allowed
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, E = x.shape
        qkv = self.in_proj(x).view(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)              # (3, B, h, N, d)
        q, k, v = qkv[0], qkv[1], qkv[2]              # (B, h, N, d)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B,h,N,N)
        if attn_bias is not None:
            scores = scores + attn_bias

        if self.logit_scale == "logn":
            if key_padding_mask is not None:
                n_real = (~key_padding_mask).sum(-1).clamp(min=2).to(scores.dtype)  # (B,)
            else:
                n_real = torch.full((B,), float(N), device=x.device, dtype=scores.dtype).clamp(min=2)
            s = F.softplus(self.log_n_scale).view(1, self.num_heads, 1, 1)
            scores = scores * (s * torch.log(n_real).view(B, 1, 1, 1))

        # Build the disallow mask (B, N, N): padded keys + non-local, but never the
        # diagonal (guarantees >=1 valid key per row -> no all-(-inf) NaN rows; padded
        # query rows are discarded downstream anyway).
        if local_allow is not None or key_padding_mask is not None:
            disallow = torch.zeros(B, N, N, dtype=torch.bool, device=x.device)
            if local_allow is not None:
                disallow |= ~local_allow
            if key_padding_mask is not None:
                disallow |= key_padding_mask.unsqueeze(1)   # (B,1,N) padded keys
            eye = torch.eye(N, dtype=torch.bool, device=x.device).unsqueeze(0)
            disallow &= ~eye
            scores = scores.masked_fill(disallow.unsqueeze(1), torch.finfo(scores.dtype).min)

        if self.softmax_type == "entmax15":
            from entmax import entmax15  # optional dependency
            attn = entmax15(scores, dim=-1)
        else:
            attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)                   # (B, h, N, d)
        out = out.transpose(1, 2).reshape(B, N, E)
        out = self.out_proj(out)
        return out, (attn if need_weights else None)
