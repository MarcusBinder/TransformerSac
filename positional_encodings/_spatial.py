"""Spatial context embeddings from turbine neighborhood structure."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SpatialContextEmbedding(nn.Module):
    """
    Per-turbine embedding based on spatial neighborhood structure.

    For each turbine, computes physics-meaningful summary features
    from the relative positions of all other turbines:
      - Upstream / downstream / lateral neighbor counts at multiple distance thresholds
      - Nearest neighbor distances in each directional sector
      - Angular sector occupancy (turbine count + min distance per sector)
      - Global stats (total neighbors, mean distance)

    These features describe a turbine's "role" in the farm (edge vs interior,
    upstream exposure, downstream influence footprint) without being tied to
    any absolute position. A turbine on the upwind edge of a 5-turbine row
    and the upwind edge of a 20-turbine grid get similar feature vectors.

    Output is added to token embeddings, same as profile encodings.

    Pathway: ADDITIVE (added to token embeddings)
    """
    def __init__(
        self,
        embed_dim: int = 128,
        distance_thresholds: Tuple[float, ...] = (3.0, 5.0, 8.0, 12.0),
        n_angular_sectors: int = 8,
    ):
        """
        Args:
            embed_dim: Output dimension (should match transformer embed_dim)
            distance_thresholds: Distance shells (in rotor diameters) for
                                 counting upstream/downstream/lateral neighbors
            n_angular_sectors: Number of angular bins for sector features
        """
        super().__init__()
        self.distance_thresholds = distance_thresholds
        self.n_angular_sectors = n_angular_sectors

        # Feature count breakdown:
        #   3 * len(thresholds)  — upstream/downstream/lateral counts per threshold
        #   3                    — nearest upstream/downstream/lateral distances
        #   2 * n_sectors        — per-sector count + min distance
        #   2                    — total neighbors, mean distance
        n_threshold_feats = 3 * len(distance_thresholds)
        n_nearest_feats = 3
        n_sector_feats = 2 * n_angular_sectors
        n_global_feats = 2

        self.feature_dim = n_threshold_feats + n_nearest_feats + n_sector_feats + n_global_feats

        self.encoder = nn.Sequential(
            nn.Linear(self.feature_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Sector boundaries (fixed)
        sector_edges = torch.linspace(-math.pi, math.pi, n_angular_sectors + 1)
        self.register_buffer("sector_lo", sector_edges[:-1])
        self.register_buffer("sector_hi", sector_edges[1:])

    def forward(
        self,
        positions: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            positions: (B, N, 2) wind-relative, normalized by rotor diameter
            key_padding_mask: (B, N) True = padding

        Returns:
            embeddings: (B, N, embed_dim)
        """
        B, N, _ = positions.shape
        device = positions.device

        # Pairwise relative positions
        pos_i = positions.unsqueeze(2)  # (B, N, 1, 2)
        pos_j = positions.unsqueeze(1)  # (B, 1, N, 2)
        rel = pos_j - pos_i             # (B, N, N, 2) "from i, where is j?"

        dx = rel[..., 0]  # positive = j is upstream of i (wind-relative)
        dy = rel[..., 1]
        dist = torch.sqrt(dx**2 + dy**2 + 1e-8)  # (B, N, N)
        angle = torch.atan2(dy, dx)                # (B, N, N)

        # Mask: exclude self-pairs and padded turbines
        self_mask = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)  # (1, N, N)
        if key_padding_mask is not None:
            pad_mask = key_padding_mask.unsqueeze(1)  # (B, 1, N) - j is padding
            valid = ~(self_mask | pad_mask)
        else:
            valid = ~self_mask

        # Replace invalid distances with large number
        LARGE = 1e6
        dist_masked = dist.masked_fill(~valid, LARGE)

        # Directional classification (relative to wind)
        is_upstream = dx > torch.abs(dy) * 0.5
        is_downstream = dx < -torch.abs(dy) * 0.5
        is_lateral = ~is_upstream & ~is_downstream

        features = []

        # --- Threshold features: neighbor counts ---
        for thresh in self.distance_thresholds:
            within = (dist_masked < thresh) & valid
            features.append((within & is_upstream).float().sum(dim=-1, keepdim=True))
            features.append((within & is_downstream).float().sum(dim=-1, keepdim=True))
            features.append((within & is_lateral).float().sum(dim=-1, keepdim=True))

        # --- Nearest distances per direction ---
        scale = 10.0
        up_dist = dist_masked.masked_fill(~(is_upstream & valid), LARGE)
        dn_dist = dist_masked.masked_fill(~(is_downstream & valid), LARGE)
        lat_dist = dist_masked.masked_fill(~(is_lateral & valid), LARGE)

        features.append(
            (up_dist.min(dim=-1).values / scale).clamp(max=1.0).unsqueeze(-1)
        )
        features.append(
            (dn_dist.min(dim=-1).values / scale).clamp(max=1.0).unsqueeze(-1)
        )
        features.append(
            (lat_dist.min(dim=-1).values / scale).clamp(max=1.0).unsqueeze(-1)
        )

        # --- Angular sector features ---
        for s in range(self.n_angular_sectors):
            in_sector = (
                (angle >= self.sector_lo[s])
                & (angle < self.sector_hi[s])
                & valid
            )
            features.append(in_sector.float().sum(dim=-1, keepdim=True))
            sector_dist = dist_masked.masked_fill(~in_sector, LARGE)
            features.append(
                (sector_dist.min(dim=-1).values / scale)
                .clamp(max=1.0)
                .unsqueeze(-1)
            )

        # --- Global features ---
        max_thresh = max(self.distance_thresholds)
        n_neighbors = (
            ((dist_masked < max_thresh) & valid).float().sum(dim=-1, keepdim=True)
        )
        valid_count = valid.float().sum(dim=-1, keepdim=True).clamp(min=1)
        mean_dist = (
            dist_masked.masked_fill(~valid, 0.0).sum(dim=-1, keepdim=True)
            / valid_count
            / scale
        )
        features.append(n_neighbors)
        features.append(mean_dist.clamp(max=1.0))

        feat_vec = torch.cat(features, dim=-1)  # (B, N, feature_dim)
        return self.encoder(feat_vec)


class NeighborhoodAggregationEmbedding(nn.Module):
    """
    Learned aggregation of pairwise spatial features into per-token embeddings.

    For each turbine i, attends over all other turbines j using spatial
    features (distance, angle, upstream/downstream) as keys, and aggregates
    into a single context vector.

    This is essentially one round of message passing on the spatial graph
    before the main transformer processes the tokens.

    Pathway: ADDITIVE (added to token embeddings)
    """
    def __init__(
        self,
        embed_dim: int = 128,
        n_heads: int = 4,
        pairwise_dim: int = 6,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        # Pairwise features → key/value
        # Input: (dist_norm, cos_angle, sin_angle, dx_norm, dy_norm, log_dist)
        self.pairwise_dim = pairwise_dim
        self.kv_proj = nn.Linear(pairwise_dim, 2 * embed_dim)

        # Learnable query per head (not per turbine — structure-agnostic)
        self.query = nn.Parameter(torch.randn(1, 1, n_heads, self.head_dim) * 0.02)

        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, positions, key_padding_mask=None):
        """
        positions: (B, N, 2) wind-relative
        Returns: (B, N, embed_dim) per-turbine spatial context
        """
        B, N, _ = positions.shape

        pos_i = positions.unsqueeze(2)  # (B, N, 1, 2)
        pos_j = positions.unsqueeze(1)  # (B, 1, N, 2)
        rel = pos_j - pos_i             # (B, N, N, 2)

        dx, dy = rel[..., 0], rel[..., 1]
        dist = torch.sqrt(dx**2 + dy**2 + 1e-8)
        angle = torch.atan2(dy, dx)

        # Pairwise features
        feats = torch.stack([
            dist,                           #  distance, already normalized by rotor diameter
            torch.cos(angle),                # direction cosine
            torch.sin(angle),                # direction sine
            dx / (dist + 1e-8),              # unit dx
            dy / (dist + 1e-8),              # unit dy
            torch.log1p(dist),               # log distance (soft)
        ], dim=-1)  # (B, N, N, 6)

        # Project to keys and values
        kv = self.kv_proj(feats)  # (B, N, N, 2*embed_dim)
        k, v = kv.chunk(2, dim=-1)

        # Reshape for multi-head attention
        k = k.view(B, N, N, self.n_heads, self.head_dim)  # (B, N, N, H, D)
        v = v.view(B, N, N, self.n_heads, self.head_dim)

        # Query broadcasts: (1, 1, H, D) → (B, N, H, D)
        q = self.query.expand(B, N, -1, -1)

        # Attention: for each turbine i, attend over all j
        # q: (B, N, H, D), k: (B, N, N, H, D)
        attn_logits = torch.einsum('bnhd,bnmhd->bnmh', q, k) / (self.head_dim ** 0.5)

        # Mask self and padding
        self_mask = torch.eye(N, device=positions.device, dtype=torch.bool)
        attn_logits = attn_logits.masked_fill(self_mask.unsqueeze(0).unsqueeze(-1), float('-inf'))

        if key_padding_mask is not None:
            pad_mask = key_padding_mask.unsqueeze(1).unsqueeze(-1)  # (B, 1, N, 1)
            attn_logits = attn_logits.masked_fill(pad_mask, float('-inf'))

        attn_weights = F.softmax(attn_logits, dim=2)  # over j dimension

        # Aggregate: (B, N, N, H, D) weighted sum over N → (B, N, H, D)
        context = torch.einsum('bnmh,bnmhd->bnhd', attn_weights, v)
        context = context.reshape(B, N, self.embed_dim)

        return self.out_proj(context)
