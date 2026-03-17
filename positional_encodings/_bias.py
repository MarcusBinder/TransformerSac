"""Relative positional biases for wind farm attention mechanisms."""
import math

import torch
import torch.nn as nn
from typing import Optional


class RelativePositionalBias(nn.Module):
    """
    Relative positional bias for attention.

    Computes a learned bias for each pair of positions based on their
    relative displacement. This bias is ADDED to attention logits.

    Physics intuition:
    - rel_pos[i,j] = pos[j] - pos[i] tells us "j is X upwind, Y lateral from i"
    - The learned bias can encode "pay more attention to upwind turbines"
    - Translation invariant: same relative geometry → same bias

    For wind farm control:
    - Positive x in wind-relative coords = upwind
    - The model can learn that upwind turbines are important (wake sources)
    """

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int = 64,
        per_head: bool = True,
        pos_dim: int = 2
    ):
        """
        Args:
            num_heads: Number of attention heads
            hidden_dim: Hidden dimension of bias MLP
            per_head: If True, each head gets its own bias. If False, shared.
            pos_dim: Dimension of position vectors (2 for x, y)
        """
        super().__init__()
        self.num_heads = num_heads
        self.per_head = per_head

        output_dim = num_heads if per_head else 1

        # MLP: relative_position (2D) → bias value(s)
        self.bias_mlp = nn.Sequential(
            nn.Linear(pos_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        positions: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute relative position bias matrix.

        Args:
            positions: (batch, n_tokens, 2) wind-relative positions
            key_padding_mask: (batch, n_tokens) where True = padding

        Returns:
            bias: (batch, num_heads, n_tokens, n_tokens) attention bias
                  Add this to attention logits before softmax.
        """
        batch_size, n_tokens, _ = positions.shape

        # Compute pairwise relative positions
        # pos_i: (batch, n, 1, 2), pos_j: (batch, 1, n, 2)
        pos_i = positions.unsqueeze(2)  # (batch, n, 1, 2)
        pos_j = positions.unsqueeze(1)  # (batch, 1, n, 2)

        # rel_pos[i,j] = pos[j] - pos[i]: "displacement from i to j"
        # If j is upwind of i (positive x), rel_pos has positive x component
        rel_pos = pos_j - pos_i  # (batch, n, n, 2)

        # Reshape for MLP: (batch * n * n, 2)
        rel_pos_flat = rel_pos.reshape(-1, 2)

        # Compute bias values
        bias_flat = self.bias_mlp(rel_pos_flat)  # (batch*n*n, num_heads or 1)

        # Reshape back
        if self.per_head:
            bias = bias_flat.reshape(batch_size, n_tokens, n_tokens, self.num_heads)
            bias = bias.permute(0, 3, 1, 2)  # (batch, num_heads, n, n)
        else:
            bias = bias_flat.reshape(batch_size, n_tokens, n_tokens, 1)
            bias = bias.permute(0, 3, 1, 2)  # (batch, 1, n, n)
            bias = bias.expand(-1, self.num_heads, -1, -1)  # (batch, num_heads, n, n)

        # Apply masking: set bias to large negative for padded positions
        if key_padding_mask is not None:
            # Expand mask: (batch, n) → (batch, 1, 1, n) for keys
            # and (batch, 1, n, 1) for queries
            mask_k = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, n)
            mask_q = key_padding_mask.unsqueeze(1).unsqueeze(3)  # (batch, 1, n, 1)

            # Zero out bias for padded positions (they'll be masked in attention anyway)
            # This prevents any gradient flow through padded positions
            combined_mask = mask_k | mask_q  # (batch, 1, n, n)
            bias = bias.masked_fill(combined_mask, 0.0)

        return bias

class RelativePolarBias(nn.Module):
    """
    Relative positional bias using polar coordinates.

    For each pair (i, j), computes:
    - Δr = distance from i to j
    - θ_ij = angle from i to j relative to wind direction

    The bias MLP learns how attention should depend on:
    - How far apart turbines are
    - Whether j is upwind/downwind/lateral from i
    """

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int = 64,
        per_head: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.per_head = per_head

        output_dim = num_heads if per_head else 1

        # Input: (Δr, θ, sin(θ), cos(θ))
        self.bias_mlp = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        positions: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            positions: (batch, n_tokens, 2) wind-relative positions
            key_padding_mask: (batch, n_tokens) where True = padding

        Returns:
            bias: (batch, num_heads, n_tokens, n_tokens)
        """
        batch_size, n_tokens, _ = positions.shape

        # Compute pairwise relative positions
        pos_i = positions.unsqueeze(2)  # (batch, n, 1, 2)
        pos_j = positions.unsqueeze(1)  # (batch, 1, n, 2)
        rel_pos = pos_j - pos_i  # (batch, n, n, 2)

        # Convert to polar
        dx = rel_pos[..., 0]
        dy = rel_pos[..., 1]

        r = torch.sqrt(dx**2 + dy**2 + 1e-8)
        theta = torch.atan2(dy, dx)

        # Stack polar features
        polar_features = torch.stack([
            r,
            theta,
            torch.sin(theta),
            torch.cos(theta),
        ], dim=-1)  # (batch, n, n, 4)

        # Apply MLP
        polar_flat = polar_features.reshape(-1, 4)
        bias_flat = self.bias_mlp(polar_flat)

        if self.per_head:
            bias = bias_flat.reshape(batch_size, n_tokens, n_tokens, self.num_heads)
            bias = bias.permute(0, 3, 1, 2)
        else:
            bias = bias_flat.reshape(batch_size, n_tokens, n_tokens, 1)
            bias = bias.permute(0, 3, 1, 2)
            bias = bias.expand(-1, self.num_heads, -1, -1)

        if key_padding_mask is not None:
            mask_k = key_padding_mask.unsqueeze(1).unsqueeze(2)
            mask_q = key_padding_mask.unsqueeze(1).unsqueeze(3)
            combined_mask = mask_k | mask_q
            bias = bias.masked_fill(combined_mask, 0.0)

        return bias

class ALiBiPositionalBias(nn.Module):
    """
    Attention with Linear Biases (ALiBi) for 2D spatial positions.

    Simple linear penalty based on distance:
        bias[i,j] = -slope * distance(i, j)

    No learned parameters! Just an inductive bias that nearby turbines
    should attend more to each other.

    Each attention head gets a different slope (geometric sequence),
    allowing different heads to focus on different distance scales.

    Reference: Press et al., "Train Short, Test Long" (2022)
    """

    def __init__(self, num_heads: int, max_distance: float = 20.0):
        """
        Args:
            num_heads: Number of attention heads
            max_distance: Expected maximum distance (in rotor diameters) for slope scaling
        """
        super().__init__()
        self.num_heads = num_heads

        # Geometric sequence of slopes (like original ALiBi)
        # Slopes: 2^(-8/n), 2^(-8*2/n), ..., 2^(-8)
        slopes = torch.tensor([
            2 ** (-8 * (i + 1) / num_heads) for i in range(num_heads)
        ])
        self.register_buffer("slopes", slopes.view(1, num_heads, 1, 1))

    def forward(
        self,
        positions: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            positions: (batch, n_tokens, 2) normalized positions
            key_padding_mask: (batch, n_tokens) where True = padding

        Returns:
            bias: (batch, num_heads, n_tokens, n_tokens)
        """
        batch_size, n_tokens, _ = positions.shape

        # Compute pairwise distances
        pos_i = positions.unsqueeze(2)
        pos_j = positions.unsqueeze(1)
        diff = pos_j - pos_i
        distances = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)  # (batch, n, n)

        # Apply linear penalty with per-head slopes
        distances = distances.unsqueeze(1)  # (batch, 1, n, n)
        bias = -self.slopes * distances  # (batch, num_heads, n, n)

        if key_padding_mask is not None:
            mask_k = key_padding_mask.unsqueeze(1).unsqueeze(2)
            mask_q = key_padding_mask.unsqueeze(1).unsqueeze(3)
            combined_mask = mask_k | mask_q
            bias = bias.masked_fill(combined_mask, 0.0)

        return bias

class DirectionalALiBiPositionalBias(nn.Module):
    """
    Directional ALiBi: Different slopes for upwind vs downwind.

    In wind-relative coordinates (wind from negative x):
    - Upwind (positive x direction): Use upwind_slope
    - Downwind (negative x direction): Use downwind_slope

    This encodes the physical asymmetry: upwind turbines affect
    downwind ones, but not vice versa.

    Learned slopes allow the model to discover the right asymmetry.
    """

    def __init__(self, num_heads: int):
        """
        Args:
            num_heads: Number of attention heads
        """
        super().__init__()
        self.num_heads = num_heads

        # Learnable slopes for upwind and downwind (per head)
        # Initialize with ALiBi-style geometric sequence
        init_slopes = torch.tensor([
            2 ** (-8 * (i + 1) / num_heads) for i in range(num_heads)
        ])

        self.upwind_slopes = nn.Parameter(init_slopes.clone())
        self.downwind_slopes = nn.Parameter(init_slopes.clone() * 0.5)  # Less penalty downwind
        self.lateral_slopes = nn.Parameter(init_slopes.clone() * 0.3)  # Even less for lateral

    def forward(
        self,
        positions: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            positions: (batch, n_tokens, 2) wind-relative positions
            key_padding_mask: (batch, n_tokens) where True = padding

        Returns:
            bias: (batch, num_heads, n_tokens, n_tokens)
        """
        batch_size, n_tokens, _ = positions.shape

        # Compute pairwise relative positions
        pos_i = positions.unsqueeze(2)
        pos_j = positions.unsqueeze(1)
        rel_pos = pos_j - pos_i  # (batch, n, n, 2)

        dx = rel_pos[..., 0]  # Positive = j is upwind of i
        dy = rel_pos[..., 1]

        # Distances
        dist = torch.sqrt(dx**2 + dy**2 + 1e-8)

        # Classify direction: upwind (dx > 0), downwind (dx < 0), lateral (|dy| > |dx|)
        is_upwind = dx > torch.abs(dy)      # Predominantly upwind
        is_downwind = dx < -torch.abs(dy)   # Predominantly downwind
        is_lateral = ~is_upwind & ~is_downwind  # Lateral

        # Apply different slopes based on direction
        slopes_upwind = self.upwind_slopes.view(1, self.num_heads, 1, 1)
        slopes_downwind = self.downwind_slopes.view(1, self.num_heads, 1, 1)
        slopes_lateral = self.lateral_slopes.view(1, self.num_heads, 1, 1)

        dist = dist.unsqueeze(1)  # (batch, 1, n, n)
        is_upwind = is_upwind.unsqueeze(1).float()
        is_downwind = is_downwind.unsqueeze(1).float()
        is_lateral = is_lateral.unsqueeze(1).float()

        # Weighted combination of slopes
        bias = -(
            slopes_upwind * dist * is_upwind +
            slopes_downwind * dist * is_downwind +
            slopes_lateral * dist * is_lateral
        )

        if key_padding_mask is not None:
            mask_k = key_padding_mask.unsqueeze(1).unsqueeze(2)
            mask_q = key_padding_mask.unsqueeze(1).unsqueeze(3)
            combined_mask = mask_k | mask_q
            bias = bias.masked_fill(combined_mask, 0.0)

        return bias

class RelativePositionalBiasAdvanced(nn.Module):
    """
    Improved relative position bias for wind farm attention.

    Key insight: In wind-relative coordinates, the relative position (dx, dy)
    has clear physical meaning:
    - dx > 0: j is downwind of i (j in i's wake)
    - dx < 0: j is upwind of i (i in j's wake)
    - |dy|: lateral separation

    This encoder explicitly decomposes into:
    1. Distance component: How much to attend based on distance
    2. Angular component: How much to attend based on direction
    3. Asymmetry: Upwind/downwind relationships

    Improvements over vanilla RelativePositionalBias:
    - Richer feature representation (not just raw dx, dy)
    - Separate pathways for distance and angle
    - Physics-motivated asymmetry
    - Proper normalization
    """

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int = 64,
        characteristic_distance: float = 5.0,  # In rotor diameters
        use_physics_asymmetry: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.characteristic_distance = characteristic_distance
        self.use_physics_asymmetry = use_physics_asymmetry

        # Feature dimension for MLP input
        # (dx_norm, dy_norm, dist_norm, angle, sin, cos, dist_decay)
        self.feature_dim = 7
        if use_physics_asymmetry:
            # Add asymmetry features: (is_upwind, is_downwind, is_lateral)
            self.feature_dim += 3

        # Main bias MLP (per-head output)
        self.bias_mlp = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_heads),
        )

        # Initialize final layer to small values (start with weak bias)
        nn.init.zeros_(self.bias_mlp[-1].weight)
        nn.init.zeros_(self.bias_mlp[-1].bias)

        # Learnable distance decay slopes (ALiBi-inspired, but learned)
        # Different heads can learn different distance sensitivities
        init_slopes = torch.tensor([
            2 ** (-8 * (i + 1) / num_heads) for i in range(num_heads)
        ])
        self.distance_slopes = nn.Parameter(init_slopes)

    def forward(
        self,
        positions: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            positions: (batch, n_tokens, 2) - wind-relative positions in rotor diameters
            key_padding_mask: (batch, n_tokens) where True = padding

        Returns:
            bias: (batch, num_heads, n_tokens, n_tokens)
        """
        batch_size, n_tokens, _ = positions.shape
        device = positions.device

        # Compute pairwise relative positions: rel_pos[i,j] = pos[j] - pos[i]
        # "From i's perspective, where is j?"
        pos_i = positions.unsqueeze(2)  # (batch, n, 1, 2)
        pos_j = positions.unsqueeze(1)  # (batch, 1, n, 2)
        rel_pos = pos_j - pos_i         # (batch, n, n, 2)

        dx = rel_pos[..., 0]  # Positive = j is downwind of i
        dy = rel_pos[..., 1]  # Lateral displacement

        # Distance (with epsilon for numerical stability)
        dist = torch.sqrt(dx**2 + dy**2 + 1e-8)

        # Normalized features
        dx_norm = dx / (dist + 1e-8)  # Unit direction x
        dy_norm = dy / (dist + 1e-8)  # Unit direction y
        dist_norm = dist / self.characteristic_distance  # Normalized distance

        # Angular features
        angle = torch.atan2(dy, dx)  # Angle from i to j
        sin_angle = torch.sin(angle)
        cos_angle = torch.cos(angle)

        # Soft distance decay (log scale, bounded)
        dist_decay = torch.log1p(dist_norm)  # Smooth, bounded

        # Stack features
        features = [dx_norm, dy_norm, dist_norm, angle, sin_angle, cos_angle, dist_decay]

        if self.use_physics_asymmetry:
            # Soft asymmetry indicators (smooth approximation)
            # is_upwind: j is upwind of i (dx < 0 and mostly aligned with wind)
            # is_downwind: j is downwind of i (dx > 0 and mostly aligned)
            # is_lateral: j is roughly perpendicular to wind

            # Use tanh for smooth transitions
            upwind_score = torch.tanh(-dx_norm * 3)  # High when dx < 0
            downwind_score = torch.tanh(dx_norm * 3)  # High when dx > 0
            lateral_score = 1 - torch.abs(dx_norm)    # High when |dy| >> |dx|

            features.extend([upwind_score, downwind_score, lateral_score])

        # Stack: (batch, n, n, feature_dim)
        features = torch.stack(features, dim=-1)

        # Flatten for MLP: (batch * n * n, feature_dim)
        features_flat = features.reshape(-1, self.feature_dim)

        # MLP: (batch * n * n, num_heads)
        bias_flat = self.bias_mlp(features_flat)

        # Reshape: (batch, n, n, num_heads) -> (batch, num_heads, n, n)
        bias = bias_flat.reshape(batch_size, n_tokens, n_tokens, self.num_heads)
        bias = bias.permute(0, 3, 1, 2)

        # Add learned distance decay (ALiBi-style)
        dist_expanded = dist.unsqueeze(1)  # (batch, 1, n, n)
        distance_bias = -self.distance_slopes.view(1, -1, 1, 1) * dist_expanded
        bias = bias + distance_bias

        # Zero out padded positions
        if key_padding_mask is not None:
            mask_k = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, n)
            mask_q = key_padding_mask.unsqueeze(1).unsqueeze(3)  # (batch, 1, n, 1)
            bias = bias.masked_fill(mask_k | mask_q, 0.0)

        return bias

class RelativePositionalBiasFactorized(nn.Module):
    """
    Alternative: Factorized relative bias with separate distance and angle networks.

    bias(i,j) = distance_bias(dist_ij) * angle_weight(angle_ij)

    This factorization makes it easier to learn interpretable patterns:
    - Distance network: "Closer turbines are more relevant"
    - Angle network: "Upwind turbines are more relevant"
    """

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int = 32,
    ):
        super().__init__()
        self.num_heads = num_heads

        # Distance network: dist -> per-head bias
        self.distance_net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_heads),
        )

        # Angle network: (sin, cos) -> per-head weight
        self.angle_net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_heads),
            nn.Sigmoid(),  # Output in [0, 1] as a multiplicative weight
        )

        # Learnable base decay (per head)
        self.base_decay = nn.Parameter(torch.ones(num_heads) * 0.1)

    def forward(
        self,
        positions: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, n_tokens, _ = positions.shape

        pos_i = positions.unsqueeze(2)
        pos_j = positions.unsqueeze(1)
        rel_pos = pos_j - pos_i

        dx, dy = rel_pos[..., 0], rel_pos[..., 1]
        dist = torch.sqrt(dx**2 + dy**2 + 1e-8)

        # Distance features
        dist_input = dist.unsqueeze(-1)  # (batch, n, n, 1)
        dist_bias = self.distance_net(dist_input.reshape(-1, 1))
        dist_bias = dist_bias.reshape(batch_size, n_tokens, n_tokens, self.num_heads)

        # Angle features
        sin_angle = dy / (dist + 1e-8)
        cos_angle = dx / (dist + 1e-8)
        angle_input = torch.stack([sin_angle, cos_angle], dim=-1)  # (batch, n, n, 2)
        angle_weight = self.angle_net(angle_input.reshape(-1, 2))
        angle_weight = angle_weight.reshape(batch_size, n_tokens, n_tokens, self.num_heads)

        # Combine: base decay + learned adjustment, modulated by angle
        base = -self.base_decay.view(1, 1, 1, -1) * dist.unsqueeze(-1)
        bias = (base + dist_bias) * angle_weight

        # Reshape to (batch, heads, n, n)
        bias = bias.permute(0, 3, 1, 2)

        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2) | key_padding_mask.unsqueeze(1).unsqueeze(3)
            bias = bias.masked_fill(mask, 0.0)

        return bias

class RelativePositionalBiasWithWind(nn.Module):
    """
    Variant that takes wind direction as explicit input.

    Useful if positions are NOT pre-transformed to wind-relative frame.
    The network learns to interpret positions given the current wind direction.
    """

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.num_heads = num_heads

        # Wind direction embedding
        self.wind_embed_dim = 16
        self.wind_embed = nn.Sequential(
            nn.Linear(2, self.wind_embed_dim),  # (sin, cos)
            nn.GELU(),
        )

        # Relative position MLP (takes wind embedding as context)
        # Input: (dx, dy, dist, sin, cos, wind_embed)
        self.bias_mlp = nn.Sequential(
            nn.Linear(5 + self.wind_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_heads),
        )

    def forward(
        self,
        positions: torch.Tensor,          # (batch, n, 2) - global frame
        wind_direction: torch.Tensor,     # (batch,) - degrees
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, n_tokens, _ = positions.shape

        # Wind direction embedding
        wd_rad = wind_direction * math.pi / 180
        wd_features = torch.stack([torch.sin(wd_rad), torch.cos(wd_rad)], dim=-1)
        wd_embed = self.wind_embed(wd_features)  # (batch, wind_embed_dim)

        # Expand wind embedding for all pairs
        wd_embed = wd_embed.unsqueeze(1).unsqueeze(1).expand(-1, n_tokens, n_tokens, -1)

        # Relative positions
        pos_i = positions.unsqueeze(2)
        pos_j = positions.unsqueeze(1)
        rel_pos = pos_j - pos_i

        dx, dy = rel_pos[..., 0], rel_pos[..., 1]
        dist = torch.sqrt(dx**2 + dy**2 + 1e-8)
        sin_angle = dy / (dist + 1e-8)
        cos_angle = dx / (dist + 1e-8)

        # Combine features
        features = torch.stack([dx, dy, dist, sin_angle, cos_angle], dim=-1)
        features = torch.cat([features, wd_embed], dim=-1)

        # MLP
        bias_flat = self.bias_mlp(features.reshape(-1, 5 + self.wind_embed_dim))
        bias = bias_flat.reshape(batch_size, n_tokens, n_tokens, self.num_heads)
        bias = bias.permute(0, 3, 1, 2)

        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2) | key_padding_mask.unsqueeze(1).unsqueeze(3)
            bias = bias.masked_fill(mask, 0.0)

        return bias

class WakeKernelBias(nn.Module):
    """
    Physics-motivated attention bias with very few learnable parameters.

    Models the wake interaction pattern:
      - Distance decay: closer turbines get more attention
      - Upstream bonus: turbines directly upstream matter most
      - Lateral Gaussian falloff: turbines perpendicular to wind matter less

    Only ~3 * num_heads learnable parameters -> essentially impossible to overfit.
    The inductive bias is correct for wake physics; the model just learns the
    right scale/strength per attention head.

    Pathway: ATTENTION BIAS (added to attention logits, like RelativePositionalBias)
    """

    def __init__(self, num_heads: int):
        """
        Args:
            num_heads: Number of attention heads
        """
        super().__init__()
        self.num_heads = num_heads

        # Per-head learnable parameters
        self.dist_decay = nn.Parameter(torch.ones(num_heads) * 0.3)
        self.lateral_width = nn.Parameter(torch.ones(num_heads) * 2.0)
        self.upstream_bonus = nn.Parameter(torch.ones(num_heads) * 1.0)

    def forward(
        self,
        positions: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            positions: (B, N, 2) wind-relative positions (rotor-diameter normalized)
            key_padding_mask: (B, N) True = padding

        Returns:
            bias: (B, num_heads, N, N) — add to attention logits before softmax
        """
        B, N, _ = positions.shape

        pos_i = positions.unsqueeze(2)
        pos_j = positions.unsqueeze(1)
        rel = pos_j - pos_i

        dx = rel[..., 0]  # positive = j downstream of i (wind-relative)
        dy = rel[..., 1]
        dist = torch.sqrt(dx ** 2 + dy ** 2 + 1e-8)

        # Expand for heads: (B, 1, N, N)
        dx = dx.unsqueeze(1)
        dy = dy.unsqueeze(1)
        dist = dist.unsqueeze(1)

        decay = self.dist_decay.abs().view(1, -1, 1, 1)
        width = (self.lateral_width.abs() + 0.1).view(1, -1, 1, 1)
        bonus = self.upstream_bonus.view(1, -1, 1, 1)

        # bias = distance_decay + upstream_bonus + lateral_penalty
        bias = -decay * dist
        bias = bias + bonus * torch.tanh(dx)
        bias = bias - (dy ** 2) / (2 * width ** 2)

        # Mask padded positions
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2) | \
                   key_padding_mask.unsqueeze(1).unsqueeze(3)
            bias = bias.masked_fill(mask, 0.0)

        return bias
