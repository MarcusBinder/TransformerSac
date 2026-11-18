"""Lane descriptor: a per-turbine layout input computed from the wind-frame positions.

An alternative to the layout roses (``profile_encoding_type``), switched on with
``--layout_descriptor``. Each turbine gets the soft occupancy of its neighbourhood in the wind
frame: per side (upstream, downstream) the number of neighbours within a set of streamwise
distances ("within 3, 5, 7, 10, 15 D"), split over lateral bins, counting only the first two
neighbours of each lane. A turbine deep in a long row then looks like the middle of a short
one, and the descriptor does not grow with the farm.

It needs nothing but what the networks already receive: ``positions`` in rotor diameters,
rotated to the wind (+x downstream, ``helpers.helper_funcs.transform_to_wind_relative``) and
the padding mask. No registry, no replay-buffer entry, no rotation.

Origin: ``grid_vector(first=2, cumulative=True)`` of the powertracking layout-descriptor study
(``scripts/analysis/profile_cases.py``, candidates ``lanes2`` and ``lanes2_wide_half``), which
holds the requirements the parameters were chosen against. A test there compares this file
with the study's numpy version.
"""
import json
import math
from typing import Optional, Sequence

import torch
import torch.nn as nn

_FINE = (3.0, 5.0, 7.0, 10.0, 14.0)
_WIDE = (-6.0, -3.0, -1.0, 0.0, 1.0, 3.0, 6.0)

# Parameters of lane_descriptor per preset (all lengths in D).
PRESETS = {
    # first two per lane, within 3..14 D, lateral bins to 2 D
    "lanes2": dict(
        nodes=_FINE, outer=20.0,
        lat_nodes=(-2.0, -1.0, 0.0, 1.0, 2.0), lat_outer=3.0, lat_weights=None,
        first=2, lane=(1.0, 2.0), tau=0.5,
    ),
    # lateral bins to 6 D with the rows beyond 3 D at half weight, bands to 15 D
    "lanes2_wide_half": dict(
        nodes=(3.0, 5.0, 7.0, 10.0, 15.0), outer=22.0,
        lat_nodes=_WIDE, lat_outer=9.0, lat_weights=(0.5, 0.5, 1.0, 1.0, 1.0, 0.5, 0.5),
        first=2, lane=(1.0, 2.0), tau=0.5,
    ),
}


def _bands_within(dist: torch.Tensor, nodes: Sequence[float], outer: float) -> torch.Tensor:
    """(..., band): weight with which a neighbour at streamwise distance ``dist`` counts as
    "within nodes[b]". Linear hats at the nodes (the last one fading to 0 at ``outer``), made
    cumulative, so a neighbour moving away leaves the nearest band first."""
    knots = [*nodes, outer]
    out = []
    for b in range(len(nodes)):
        down = ((knots[b + 1] - dist) / (knots[b + 1] - knots[b])).clamp(0.0, 1.0)
        if b > 0:
            up = ((dist - knots[b - 1]) / (knots[b] - knots[b - 1])).clamp(0.0, 1.0)
            down = torch.minimum(up, down)
        out.append(down)
    return torch.stack(out, dim=-1).cumsum(dim=-1)


def _lateral_hats(lat: torch.Tensor, nodes: Sequence[float], outer: float) -> torch.Tensor:
    """(..., bin): linear hats at ``nodes`` that sum to 1 between the first and the last node
    and fade to 0 at ``-outer`` and ``outer``."""
    k = [-outer, *nodes, outer]
    return torch.stack([
        torch.minimum((lat - k[i]) / (k[i + 1] - k[i]),
                      (k[i + 2] - lat) / (k[i + 2] - k[i + 1])).clamp(0.0, 1.0)
        for i in range(len(nodes))
    ], dim=-1)


def _n_before(dist, lat, present, lane, tau, chunk_elems):
    """(B, i, j): soft number of neighbours k of turbine i that are closer than neighbour j
    on this side and in j's lane; ``present`` (B, i, k) is the weight of k. Summed over k in
    chunks."""
    B, N, K = dist.shape
    out = torch.zeros_like(dist)
    step = max(1, int(chunk_elems) // max(1, B * N * K))
    ks = torch.arange(K, device=dist.device)
    for k0 in range(0, K, step):
        k = ks[k0:k0 + step]
        closer = (0.5 + (dist.unsqueeze(-1) - dist[:, :, None, k]) / (2.0 * tau)).clamp(0.0, 1.0)
        gap = (lat.unsqueeze(-1) - lat[:, :, None, k]).abs()
        same_lane = 0.5 * (1.0 + torch.cos(
            math.pi * ((gap - lane[0]) / (lane[1] - lane[0])).clamp(0.0, 1.0)))
        not_j = (ks[:, None] != k[None, :]).to(dist.dtype)  # (j, k)
        out += (present[:, :, None, k] * closer * same_lane * not_j).sum(dim=-1)
    return out


@torch.no_grad()
def lane_descriptor(
    positions: torch.Tensor,
    key_padding_mask: Optional[torch.Tensor] = None,
    *,
    nodes: Sequence[float],
    outer: float,
    lat_nodes: Sequence[float],
    lat_outer: float,
    lat_weights: Optional[Sequence[float]] = None,
    first: Optional[int] = 2,
    lane: Sequence[float] = (1.0, 2.0),
    tau: float = 0.5,
    chunk_elems: int = 2 ** 24,
) -> torch.Tensor:
    """Lane descriptor of every turbine, (B, N, 2 * len(nodes) * len(lat_nodes)) in float32,
    ordered (side, band, lateral bin) with side 0 = upstream.

    Args:
        positions: (B, N, 2) in rotor diameters, wind frame (+x downstream, +y to the left
            looking downstream).
        key_padding_mask: (B, N), True = padding. Padded turbines are no one's neighbour and
            get a zero descriptor.
        nodes, outer: streamwise distances of the "within x D" bands; the last band fades to
            0 at ``outer``.
        lat_nodes, lat_outer: centres of the lateral bins; the outer bins fade to 0 at
            ``lat_outer``. ``lat_weights`` scales the bins.
        first: count only the first ``first`` neighbours of a lane (None: all). A neighbour
            is weighted by ``clip(first - n_before, 0, 1)``, ``n_before`` being the soft
            number of neighbours on the same side that are closer (ramp of width ``tau``) and
            in the same lane (lateral gap below ``lane[0]``, fading to ``lane[1]``).
        chunk_elems: size limit of the (B, N, N, chunk) tensors of the first-``first`` rule.

    A neighbour fades in between 0.5 and 1.5 D streamwise, so one passing abeam moves from
    the upstream to the downstream side smoothly.
    """
    with torch.autocast(device_type=positions.device.type, enabled=False):
        pos = positions.detach().float()
        B, N, _ = pos.shape
        s = pos[:, None, :, 0] - pos[:, :, None, 0]  # (B, i, j): j downstream of i
        lat = pos[:, None, :, 1] - pos[:, :, None, 1]
        pair = ~torch.eye(N, dtype=torch.bool, device=pos.device)[None]
        if key_padding_mask is not None:
            real = ~key_padding_mask.bool()
            pair = pair & real[:, :, None] & real[:, None, :]
        pair = pair.to(pos.dtype)

        wts = None if lat_weights is None else \
            torch.tensor(lat_weights, dtype=pos.dtype, device=pos.device)
        lat_reach = lat_outer + (lane[1] if first is not None else 0.0)

        sides = []
        for sign in (-1.0, 1.0):
            dist = sign * s
            present = (dist - 0.5).clamp(0.0, 1.0) * pair
            # Only neighbours inside the reach count, or can stand in front of one that does.
            # Keep those (K per turbine: the largest number in the batch), so the cost
            # follows the local density and not the farm size.
            reach = (present > 0) & (dist < outer) & (lat.abs() < lat_reach)
            present = present * reach
            K = max(1, int(reach.sum(dim=-1).max()))
            side_lat = lat
            if K < N:
                idx = reach.to(pos.dtype).topk(K, dim=-1).indices  # (B, i, K)
                dist, side_lat, present = (v.gather(-1, idx) for v in (dist, lat, present))
            if first is not None:
                before = _n_before(dist, side_lat, present, lane, tau, chunk_elems)
                present = present * (first - before).clamp(0.0, 1.0)
            hs = _bands_within(dist, nodes, outer) * present.unsqueeze(-1)  # (B, i, K, band)
            hl = _lateral_hats(side_lat, lat_nodes, lat_outer)  # (B, i, K, bin)
            if wts is not None:
                hl = hl * wts
            sides.append(torch.einsum("nijb,nijc->nibc", hs, hl).reshape(B, N, -1))
        return torch.cat(sides, dim=-1)


class LaneDescriptorEncoder(nn.Module):
    """``lane_descriptor`` of a preset followed by Linear - LayerNorm - GELU - Linear (the
    head of ``FourierProfileEncoder``). ``kwargs`` override the preset's parameters."""

    def __init__(self, embed_dim: int, preset: str = "lanes2_wide_half", **kwargs):
        super().__init__()
        if preset not in PRESETS:
            raise ValueError(f"Unknown layout_descriptor: {preset!r}. Valid: {sorted(PRESETS)}")
        unknown = set(kwargs) - set(PRESETS[preset]) - {"chunk_elems"}
        if unknown:
            raise TypeError(f"Unknown layout_descriptor_kwargs: {sorted(unknown)}")
        self.params = {**PRESETS[preset], **kwargs}
        for key in ("nodes", "lat_nodes", "lane", "lat_weights"):
            if self.params[key] is not None:
                self.params[key] = tuple(float(v) for v in self.params[key])
        if self.params["lat_weights"] is not None and \
                len(self.params["lat_weights"]) != len(self.params["lat_nodes"]):
            raise ValueError("lat_weights needs one weight per lateral bin (lat_nodes)")
        self.feature_dim = 2 * len(self.params["nodes"]) * len(self.params["lat_nodes"])
        self.proj = nn.Sequential(
            nn.Linear(self.feature_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, positions: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """(B, N, 2) positions in D, wind frame -> (B, N, embed_dim)."""
        return self.proj(lane_descriptor(positions, key_padding_mask, **self.params))


def create_layout_descriptor(args, embed_dim: int):
    """``(encoder, proj)`` from ``args.layout_descriptor*``; ``(None, None)`` when the flag
    is off or ``args`` predates it. ``proj`` is the ``Linear(2E, E)`` of the concat mode."""
    name = getattr(args, "layout_descriptor", None) if args is not None else None
    if name is None:
        return None, None
    mode = getattr(args, "layout_descriptor_mode", "concat")
    if mode not in ("add", "concat"):
        raise ValueError(f"Invalid layout_descriptor_mode: {mode}")
    kwargs = json.loads(getattr(args, "layout_descriptor_kwargs", "{}") or "{}")
    encoder = LaneDescriptorEncoder(embed_dim, name, **kwargs)
    proj = nn.Linear(2 * embed_dim, embed_dim) if mode == "concat" else None
    return encoder, proj
