"""Dense Graph Attention Network for spatial positional encoding."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseGATv2Layer(nn.Module):
    """
    GATv2-style attention operating on dense (B, N, N) pairs.

    Implements the "dynamic attention" of Brody et al. (2022):
        e_ij = a^T · LeakyReLU(W_l·h_i + W_r·h_j + W_e·edge_ij)
        a_ij = softmax_j(e_ij)
        h_i' = Σ_j a_ij · W_r·h_j

    All operations are fully batched — no loops over B or N.
    """
    def __init__(self, embed_dim: int, n_heads: int, edge_dim: int, dropout: float = 0.0):
        super().__init__()

        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"

        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.scale = self.head_dim**-0.5  # optional scaling for stability

        # Node projections (left = query-side, right = key/value-side)
        self.W_l = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_r = nn.Linear(embed_dim, embed_dim, bias=False)

        # Edge projection
        self.W_e = nn.Linear(edge_dim, embed_dim, bias=False)

        # Per-head attention vector
        self.attn = nn.Parameter(torch.empty(n_heads, self.head_dim))
        nn.init.xavier_normal_(self.attn.unsqueeze(-1))

        # Output projection (after concatenating heads)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        edge_feats: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            h:          (B, N, D) node embeddings
            edge_feats: (B, N, N, edge_dim) pairwise edge features
            attn_mask:  (B, N, N) bool — True = attend, False = mask out

        Returns:
            out: (B, N, D) updated node embeddings
        """
        B, N, D = h.shape
        H, Dh = self.n_heads, self.head_dim

        l = self.W_l(h)  # (B, N, D)
        r = self.W_r(h)  # (B, N, D)

        # Pairwise interaction: (B, N, 1, D) + (B, 1, N, D) + (B, N, N, D)
        pair = l.unsqueeze(2) + r.unsqueeze(1) + self.W_e(edge_feats)
        pair = F.leaky_relu(pair, negative_slope=0.2)

        # Reshape to multi-head: (B, N, N, H, Dh)
        pair = pair.view(B, N, N, H, Dh)

        # Attention logits: dot with per-head vector → (B, N, N, H)
        logits = (pair * self.attn).sum(dim=-1)

        # Apply mask (padding + self-loops + optional distance cutoff)
        if attn_mask is not None:
            logits = logits.masked_fill(~attn_mask.unsqueeze(-1), float("-inf"))

        weights = F.softmax(logits, dim=2)  # softmax over source dim j
        weights = self.attn_dropout(weights)

        # Values: use W_r·h (standard GATv2) reshaped to (B, N, H, Dh)
        values = r.view(B, N, H, Dh)

        # Weighted aggregation: (B, N_i, N_j, H) × (B, N_j, H, Dh) → (B, N_i, H, Dh)
        out = torch.einsum("bijn,bjnh->binh", weights, values)
        out = out.reshape(B, N, D)

        return self.out_proj(out)

class GATPositionalEncoder(nn.Module):
    """
    Dense Graph Attention Network that produces per-turbine spatial context
    embeddings from positions (and optionally wind conditions).

    Uses a fully vectorized GATv2-style attention mechanism operating on
    dense (B, N, N) pair matrices. No PyG dependency required.

    For wind farms of typical scale (10–100 turbines), dense attention is
    both faster and simpler than sparse message passing, since the graphs
    are fully or near-fully connected anyway.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        edge_dim: int = 8,
        dropout: float = 0.0,
        use_wind_context: bool = False,
        distance_cutoff: Optional[float] = None,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.edge_dim = edge_dim
        self.distance_cutoff = distance_cutoff
        self.use_wind_context = use_wind_context

        # --- Raw edge features (8-dim) → projected edge features ---
        raw_edge_dim = 8
        self.edge_proj = nn.Sequential(
            nn.Linear(raw_edge_dim, edge_dim * 2),
            nn.GELU(),
            nn.Linear(edge_dim * 2, edge_dim),
        )

        # --- Initial node embedding ---
        if use_wind_context:
            self.node_init = nn.Sequential(
                nn.Linear(3, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            )
        else:
            self.node_init_embed = nn.Parameter(
                torch.randn(1, 1, embed_dim) * 0.02
            )

        # --- GAT layers (dense) ---
        self.gat_layers = nn.ModuleList()
        self.norms1 = nn.ModuleList()
        self.norms2 = nn.ModuleList()
        self.ffns = nn.ModuleList()

        for _ in range(n_layers):
            self.gat_layers.append(
                DenseGATv2Layer(embed_dim, n_heads, edge_dim, dropout=dropout)
            )
            self.norms1.append(nn.LayerNorm(embed_dim))
            self.norms2.append(nn.LayerNorm(embed_dim))
            self.ffns.append(
                nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 2),
                    nn.GELU(),
                    nn.Linear(embed_dim * 2, embed_dim),
                )
            )

        self.final_norm = nn.LayerNorm(embed_dim)

    def _compute_dense_edge_feats(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise edge features for all (i, j) pairs in one vectorized pass.

        Args:
            positions: (B, N, 2) wind-relative turbine positions

        Returns:
            edge_feats: (B, N, N, edge_dim) projected edge features
        """
        # Pairwise differences: (B, N, N, 2)
        diff = positions.unsqueeze(2) - positions.unsqueeze(1)
        dx = diff[..., 0]  # (B, N, N)
        dy = diff[..., 1]  # (B, N, N)

        dist = torch.sqrt(dx**2 + dy**2 + 1e-8)
        angle = torch.atan2(dy, dx)
        unit_dx = dx / (dist + 1e-8)
        unit_dy = dy / (dist + 1e-8)

        # Stack raw features: (B, N, N, 8) — same features as original
        raw_feats = torch.stack(
            [
                unit_dx,
                unit_dy,
                dist / 10.0,
                torch.cos(angle),
                torch.sin(angle),
                torch.log1p(dist),
                torch.tanh(unit_dx * 2),
                1.0 - torch.abs(unit_dx),
            ],
            dim=-1,
        )

        # Single batched projection: (B, N, N, 8) → (B, N, N, edge_dim)
        return self.edge_proj(raw_feats)

    def _build_attn_mask(
        self,
        positions: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Build dense attention mask: (B, N, N) where True = attend.

        Handles: no self-loops, padding exclusion, optional distance cutoff.
        """
        B, N, _ = positions.shape
        device = positions.device

        # No self-attention (no self-loops)
        mask = ~torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0)  # (1, N, N)

        # Exclude padding positions from both source and target
        if key_padding_mask is not None:
            valid = ~key_padding_mask  # (B, N) True = real turbine
            # valid_i AND valid_j: (B, N, 1) & (B, 1, N) → (B, N, N)
            mask = mask & valid.unsqueeze(1) & valid.unsqueeze(2)

        # Optional distance cutoff (sparse connectivity for large farms)
        if self.distance_cutoff is not None:
            dist = torch.cdist(positions, positions)  # (B, N, N)
            mask = mask & (dist < self.distance_cutoff)

        return mask

    def forward(
        self,
        positions: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        wind_speed: Optional[torch.Tensor] = None,
        wind_direction: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            positions:        (B, N, 2) wind-relative turbine positions
            key_padding_mask: (B, N) True = padding token
            wind_speed:       (B,) free-stream wind speed [m/s]
            wind_direction:   (B,) wind direction [degrees]

        Returns:
            out: (B, N, embed_dim) spatial context embedding per turbine
        """
        B, N, _ = positions.shape

        # --- Dense edge features (single batched pass) ---
        edge_feats = self._compute_dense_edge_feats(positions)  # (B, N, N, edge_dim)

        # --- Attention mask ---
        attn_mask = self._build_attn_mask(positions, key_padding_mask)  # (B, N, N)

        # --- Initialize node embeddings ---
        if self.use_wind_context and wind_speed is not None:
            wd_rad = wind_direction * (math.pi / 180.0)
            wind_feats = torch.stack(
                [
                    wind_speed / 15.0,
                    torch.cos(wd_rad),
                    torch.sin(wd_rad),
                ],
                dim=-1,
            )  # (B, 3)
            h = self.node_init(wind_feats).unsqueeze(1).expand(-1, N, -1)  # (B, N, D)
        else:
            h = self.node_init_embed.expand(B, N, -1)  # (B, N, D)

        # --- Message passing ---
        for gat, norm1, norm2, ffn in zip(
            self.gat_layers, self.norms1, self.norms2, self.ffns
        ):
            h = h + gat(norm1(h), edge_feats, attn_mask)  # GAT + residual
            h = h + ffn(norm2(h))                          # FFN + residual

        h = self.final_norm(h)

        # --- Zero out padding positions ---
        if key_padding_mask is not None:
            h = h.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        return h
