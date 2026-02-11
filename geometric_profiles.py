"""
Geometric approximation of receptivity and influence profiles.

Computes the same (n_turbines, n_directions) profiles as receptivity_profiles.py,
but using only layout geometry — no PyWake simulation needed.

The key insight: a receptivity profile at direction θ is essentially
"how much wake deficit would I receive if the wind came from θ?"
which is dominated by:
  - How many turbines are upstream of me at angle θ
  - How close they are (deficit ∝ 1/x for Gaussian wakes)
  - How aligned they are (Gaussian lateral decay)

Similarly, an influence profile answers "how much can I affect others
if the wind comes from θ?" — same model, but looking downstream.

Drop-in replacement for compute_layout_profiles().
"""

from math import inf
import numpy as np
from scipy.ndimage import gaussian_filter1d
from typing import Optional, Tuple


def _compute_geometric_rose(
    x_pos: np.ndarray,
    y_pos: np.ndarray,
    turbine_index: int,
    n_directions: int = 360,
    rotor_diameter: float = 126.0,
    k_wake: float = 0.04,
    min_distance_D: float = 1.0,
    sigma_smooth: float = 2.0,
    mode: str = "receptivity",
) -> np.ndarray:
    """
    Compute a geometric receptivity or influence rose for one turbine.

    For receptivity (mode='receptivity'):
        For each wind direction θ, find all turbines UPSTREAM of turbine_index.
        Each upstream turbine j contributes:
            contribution_j = (D / x_stream) * exp(-y_lat² / (2*(k*x_stream)²))
        where x_stream is streamwise distance and y_lat is lateral offset.

    For influence (mode='influence'):
        Same, but look DOWNSTREAM instead.

    Args:
        x_pos: Turbine x-coordinates (m)
        y_pos: Turbine y-coordinates (m)
        turbine_index: Index of the turbine to compute the profile for
        n_directions: Number of angular bins (default 360)
        rotor_diameter: Rotor diameter in meters (sets the length scale)
        k_wake: Wake expansion rate (0.04-0.075 typical)
        min_distance_D: Minimum streamwise distance in diameters (avoids singularity)
        sigma_smooth: Gaussian smoothing sigma (in direction bins)
        mode: 'receptivity' (look upstream) or 'influence' (look downstream)

    Returns:
        profile: Shape (n_directions,) — smoothed geometric profile
    """
    D = rotor_diameter
    n_turbines = len(x_pos)

    # Relative positions of all other turbines w.r.t. this one
    dx = x_pos - x_pos[turbine_index]  # (n_turbines,)
    dy = y_pos - y_pos[turbine_index]
    
    # Mask out self
    mask = np.arange(n_turbines) != turbine_index

    directions_deg = np.linspace(0, 360, n_directions, endpoint=False)
    profile = np.zeros(n_directions, dtype=np.float64)

    for i, wd in enumerate(directions_deg):
        # Wind direction convention: wd=0 means wind FROM north (flowing south)
        # Wind vector points in the direction the wind is GOING
        # Using meteorological convention: wd=0 → from N, wd=90 → from E
        wind_rad = np.deg2rad(270 - wd)
        wx = np.cos(wind_rad)
        wy = np.sin(wind_rad)

        # Project relative positions onto streamwise (along wind) and lateral axes
        # Streamwise: positive = downstream of turbine_index
        x_stream = dx * wx + dy * wy  # (n_turbines,)
        # Lateral: perpendicular to wind
        y_lat = -dx * wy + dy * wx    # (n_turbines,)

        if mode == "receptivity":
            # Look UPSTREAM: turbines with negative streamwise distance
            # (they are upstream of us, so their wake hits us)
            upstream = (x_stream < -min_distance_D * D) & mask
            dist = np.abs(x_stream[upstream])
        elif mode == "influence":
            # Look DOWNSTREAM: turbines with positive streamwise distance
            downstream = (x_stream > min_distance_D * D) & mask
            dist = x_stream[downstream]
        else:
            raise ValueError(f"mode must be 'receptivity' or 'influence', got '{mode}'")

        if mode == "receptivity":
            lat = np.abs(y_lat[upstream])
        else:
            lat = np.abs(y_lat[downstream])

        if len(dist) == 0:
            continue

        # Wake-like contribution: axial decay * lateral Gaussian
        # deficit ∝ (D / x) — classic far-wake scaling
        axial_decay = D / dist

        # Gaussian lateral profile: wake width grows as k * x
        wake_sigma = k_wake * dist  # wake half-width at distance x
        lateral_weight = np.exp(-0.5 * (lat / np.maximum(wake_sigma, 1e-6)) ** 2)

        profile[i] = np.sum(axial_decay * lateral_weight)

    # Smooth with circular Gaussian
    if sigma_smooth > 0:
        profile = gaussian_filter1d(profile, sigma=sigma_smooth, mode='wrap')

    return profile.astype(np.float32)


def compute_layout_profiles(
    x_pos: np.ndarray,
    y_pos: np.ndarray,
    rotor_diameter: float = 126.0,
    k_wake: float = 0.04,
    n_directions: int = 360,
    sigma_smooth: float = 2.0,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute geometric receptivity and influence profiles for all turbines.

    Drop-in replacement for receptivity_profiles.compute_layout_profiles().
    Returns the same (n_turbines, n_directions) arrays, but computed from
    pure geometry in milliseconds instead of PyWake simulations.

    Args:
        x_pos: Turbine x-coordinates (m)
        y_pos: Turbine y-coordinates (m)
        rotor_diameter: Rotor diameter in meters
        k_wake: Wake expansion rate (controls lateral spread)
        n_directions: Number of angular bins
        sigma_smooth: Gaussian smoothing sigma
        verbose: Print progress

    Returns:
        (receptivity_profiles, influence_profiles): Each shape (n_turbines, n_directions)
    """
    n_turbines = len(x_pos)
    recep = np.zeros((n_turbines, n_directions), dtype=np.float32)
    influ = np.zeros((n_turbines, n_directions), dtype=np.float32)

    for t in range(n_turbines):
        if verbose:
            print(f"  Geometric profile for turbine {t+1}/{n_turbines}")

        recep[t] = _compute_geometric_rose(
            x_pos, y_pos, t,
            n_directions=n_directions,
            rotor_diameter=rotor_diameter,
            k_wake=k_wake,
            sigma_smooth=sigma_smooth,
            mode="receptivity",
        )
        influ[t] = _compute_geometric_rose(
            x_pos, y_pos, t,
            n_directions=n_directions,
            rotor_diameter=rotor_diameter,
            k_wake=k_wake,
            sigma_smooth=sigma_smooth,
            mode="influence",
        )

    return recep, influ


def compute_layout_profiles_vectorized(
    x_pos: np.ndarray,
    y_pos: np.ndarray,
    rotor_diameter: float = 126.0,
    k_wake: float = 0.04,
    n_directions: int = 360,
    sigma_smooth: float = 10.0,
    scale_factor: float = 15.0,
) -> Tuple[np.ndarray, np.ndarray]:
    print("Calling vectorized geometric profile computation...")
    """
    Fully vectorized version — computes all turbines × all directions at once.

    Much faster for large farms (no Python loops over directions).

    Args / Returns: Same as compute_layout_profiles().
    """
    D = rotor_diameter
    n_turbines = len(x_pos)

    directions_deg = np.linspace(0, 360, n_directions, endpoint=False)
    directions_rad = np.deg2rad(270 - directions_deg)

    # Wind unit vectors: (n_directions,)
    wx = np.cos(directions_rad)
    wy = np.sin(directions_rad)

    # Pairwise relative positions: (n_turbines, n_turbines)
    # dx[i, j] = x_pos[j] - x_pos[i]  (position of j relative to i)
    dx = x_pos[np.newaxis, :] - x_pos[:, np.newaxis]
    dy = y_pos[np.newaxis, :] - y_pos[:, np.newaxis]

    # Project onto streamwise/lateral for all directions at once
    # x_stream[d, i, j] = streamwise distance of turbine j from turbine i for direction d
    # Shape: (n_directions, n_turbines, n_turbines)
    x_stream = dx[np.newaxis, :, :] * wx[:, np.newaxis, np.newaxis] + \
               dy[np.newaxis, :, :] * wy[:, np.newaxis, np.newaxis]
    y_lat = -dx[np.newaxis, :, :] * wy[:, np.newaxis, np.newaxis] + \
             dy[np.newaxis, :, :] * wx[:, np.newaxis, np.newaxis]

    # Self-mask: (n_turbines, n_turbines)
    self_mask = ~np.eye(n_turbines, dtype=bool)

    min_dist = 1.0 * D  # minimum streamwise distance

    # === Receptivity: turbines UPSTREAM (x_stream < 0) ===
    upstream = (x_stream < -min_dist) & self_mask[np.newaxis, :, :]
    dist_up = np.where(upstream, np.abs(x_stream), np.inf)
    lat_up = np.abs(y_lat)

    axial_up = np.where(upstream, D / dist_up, 0.0)
    wake_sigma_up = k_wake * dist_up
    lateral_up = np.where(upstream, np.exp(-0.5 * (lat_up / np.maximum(wake_sigma_up, 1e-6)) ** 2), 0.0)

    # Sum over source turbines (axis=2), result: (n_directions, n_turbines)
    recep_raw = np.sum(axial_up * lateral_up, axis=2).T  # → (n_turbines, n_directions)

    # === Influence: turbines DOWNSTREAM (x_stream > 0) ===
    downstream = (x_stream > min_dist) & self_mask[np.newaxis, :, :]
    dist_down = np.where(downstream, x_stream, np.inf)
    lat_down = np.abs(y_lat)

    axial_down = np.where(downstream, D / dist_down, 0.0)
    wake_sigma_down = k_wake * dist_down
    lateral_down = np.where(downstream, np.exp(-0.5 * (lat_down / np.maximum(wake_sigma_down, 1e-6)) ** 2), 0.0)

    influ_raw = np.sum(axial_down * lateral_down, axis=2).T  # → (n_turbines, n_directions)

    # Smooth each profile
    recep = np.zeros_like(recep_raw)
    influ = np.zeros_like(influ_raw)
    for t in range(n_turbines):
        recep[t] = gaussian_filter1d(recep_raw[t], sigma=sigma_smooth, mode='wrap')
        influ[t] = gaussian_filter1d(influ_raw[t], sigma=sigma_smooth, mode='wrap')
    recep *= scale_factor
    influ *= scale_factor
    return recep.astype(np.float32), influ.astype(np.float32)


# Reuse the same rotate_profiles function (it's pure geometry)
def rotate_profiles(
    profiles: np.ndarray,
    wind_direction: float,
    profile_directions: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Rotate profiles so that current wind direction is at index 0.
    Same as receptivity_profiles.rotate_profiles().
    """
    n_directions = profiles.shape[-1]
    if profile_directions is None:
        degrees_per_index = 360.0 / n_directions
    else:
        degrees_per_index = profile_directions[1] - profile_directions[0]

    shift = int(round(wind_direction / degrees_per_index))
    return np.roll(profiles, -shift, axis=-1)


# =========================================================================
# Convenience: quick test / demo
# =========================================================================
if __name__ == "__main__":
    import time

    # Example: 3x3 grid farm
    spacing = 5 * 126  # 5D spacing
    xs, ys = [], []
    for row in range(3):
        for col in range(3):
            xs.append(col * spacing)
            ys.append(row * spacing)
    x_pos = np.array(xs, dtype=float)
    y_pos = np.array(ys, dtype=float)

    t0 = time.time()
    recep, influ = compute_layout_profiles(x_pos, y_pos, verbose=True)
    t1 = time.time()
    print(f"\nLoop version: {t1-t0:.3f}s")

    t0 = time.time()
    recep_v, influ_v = compute_layout_profiles_vectorized(x_pos, y_pos)
    t1 = time.time()
    print(f"Vectorized version: {t1-t0:.3f}s")

    print(f"\nMax difference (receptivity): {np.max(np.abs(recep - recep_v)):.6f}")
    print(f"Max difference (influence):   {np.max(np.abs(influ - influ_v)):.6f}")
    print(f"\nReceptivity shape: {recep.shape}")
    print(f"Influence shape:   {influ.shape}")