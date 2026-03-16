"""
Receptivity profile computation for wind farm turbines.

Computes the "aerodynamic fingerprint" of each turbine - a 360-degree profile
showing wake-induced losses from each inflow direction.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from typing import Tuple, Optional

# PyWake imports (only needed for computation)
from py_wake.deflection_models.jimenez import JimenezWakeDeflection
from py_wake.literature.gaussian_models import Blondel_Cathelain_2020
from py_wake.turbulence_models import CrespoHernandez
from py_wake.site import UniformSite 


def calc_receptivity_rose(farm, turbine_index: int, metric: str = 'WS_eff', 
                          wd_min: float = 0, wd_max: float = 360, n_wd: int = 36,
                          ws_min: float = 5, ws_max: float = 25, n_ws: int = 20,
                          ti: float = 0.075, sigma_kernel: float = 2.0,
                          rose_directions: int = 360,
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the wake-induced loss for a specific turbine from all directions.
    
    Returns:
        tuple: (smoothed_profile, directions) - profile shape is (rose_directions,)
    """
    wind_directions_pywake = np.linspace(wd_min, wd_max, n_wd, endpoint=True)
    wind_speeds = np.linspace(ws_min, ws_max, n_ws, endpoint=True)
    directions = np.linspace(0, 360, rose_directions, endpoint=False)  # 0-359 degrees

    sim_res = farm(
        x=farm.site.initial_position[:, 0],
        y=farm.site.initial_position[:, 1],
        wd=wind_directions_pywake,
        ws=wind_speeds,
        TI=ti,
        tilt=0,
        yaw=0,
    )

    waked_performance = sim_res[metric].sel(wt=turbine_index)

    if metric == 'WS_eff':
        unwaked_performance = sim_res.WS
    elif metric == 'Power':
        unwaked_performance = farm.windTurbines.power(sim_res.WS)
    else:
        raise ValueError("Metric must be 'WS_eff' or 'Power'")

    wake_loss = unwaked_performance - waked_performance
    receptivity_profile = wake_loss.mean(dim='ws')

    interp = receptivity_profile.interp(wd=directions, method="linear")
    extended = interp.fillna(0)

    smoothed_profile = gaussian_filter1d(
        extended.values,
        sigma=sigma_kernel,
        mode='wrap'
    )

    return smoothed_profile, directions

def calc_influence_rose(farm, turbine_index, scaling_factor=10.0, ti=0.05,
                        ws_min=10, ws_max=20, n_ws=10,
                        wd_min=0, wd_max=360, n_wd=36,
                        yaw_min=-30, yaw_max=30, delta_yaw=5,
                        sigma_kernel=2.0, wake_width_deg=20.0, n_dirs=100,
                        ):
    """
    Calculates a directional influence score (rose) for a turbine based on its 
    influence on all other turbines in the farm.

    Args:
        farm: PyWake wind farm model.
        turbine_index (int): The index of the turbine to assess.
        scaling_factor (float): A factor to scale the resulting influence scores.
        ti (float): Turbulence intensity.
        ws_min (int, float): Minimum wind speed.
        ws_max (int, float): Maximum wind speed.
        n_ws (int): Number of wind speed steps.
        wd_min (int, float): Minimum wind direction.
        wd_max (int, float): Maximum wind direction.
        n_wd (int): Number of wind direction steps for influence calculation.
        yaw_min (int, float): Minimum yaw angle in degrees.
        yaw_max (int, float): Maximum yaw angle in degrees.
        delta_yaw (int, float): Step size for yaw angles.
        sigma_kernel (float): Standard deviation for the Gaussian smoothing kernel.
        wake_width_deg (float): The angular width of the wake cone in degrees.
        n_dirs (int): Number of directions for the output rose.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: Smoothed influence scores and corresponding directions.
    """
    # --- Calculate influence scores ---
    wd = np.linspace(wd_min, wd_max, n_wd, endpoint=False)
    ws = np.linspace(ws_min, ws_max, n_ws, endpoint=True)
    yaw_options = np.arange(yaw_min, yaw_max + delta_yaw, delta_yaw)

    x_coords = farm.site.initial_position[:, 0]
    y_coords = farm.site.initial_position[:, 1]
    n_turbines = len(x_coords)

    turbines_to_check = np.delete(np.arange(n_turbines), turbine_index)
    total_influence = np.zeros(len(turbines_to_check), dtype=float)

    for yaw_test in yaw_options:
        yaws_base = np.zeros(n_turbines)
        yaws_base[turbine_index] = yaw_test

        sim_res_full = farm(x=x_coords, y=y_coords, ws=ws, wd=wd, TI=ti, yaw=yaws_base, tilt=0)

        x_coords_reduced = np.delete(x_coords, turbine_index)
        y_coords_reduced = np.delete(y_coords, turbine_index)
        yaws_reduced = np.delete(yaws_base, turbine_index)
        sim_res_reduced = farm(x=x_coords_reduced, y=y_coords_reduced, ws=ws, wd=wd, TI=ti, yaw=yaws_reduced, tilt=0)

        ws_full = sim_res_full.sel(wt=turbines_to_check).WS_eff
        ws_reduced = sim_res_reduced.WS_eff
        ws_full = ws_full.assign_coords(wt=ws_reduced.wt.values)
        
        diff = np.abs(ws_full - ws_reduced).sum(axis=(1, 2))
        total_influence += diff.values * delta_yaw

    normalization_factor = (yaw_max - yaw_min) * n_ws * n_wd
    influence_score = (total_influence / normalization_factor) * scaling_factor

    # --- Calculate directional rose ---
    directions = np.linspace(0, 360, n_dirs, endpoint=False)
    influence_rose = np.zeros_like(directions, dtype=float)

    x_turb = x_coords[turbine_index]
    y_turb = y_coords[turbine_index]
    
    dx = x_coords - x_turb
    dy = y_coords - y_turb

    for i, wind_dir_deg in enumerate(directions):
        wind_dir_rad = np.deg2rad(270 - wind_dir_deg)
        
        wind_vector_x = np.cos(wind_dir_rad)
        wind_vector_y = np.sin(wind_dir_rad)

        projection = dx * wind_vector_x + dy * wind_vector_y
        is_downstream = projection > 0

        angles_to_turbines = np.arctan2(dy, dx)
        relative_angles = np.abs((angles_to_turbines - wind_dir_rad + np.pi) % (2 * np.pi) - np.pi)
        
        is_in_wake = (relative_angles <= np.deg2rad(wake_width_deg / 2.0)) & is_downstream
        is_in_wake = np.delete(is_in_wake, turbine_index)

        influence_rose[i] = influence_score[is_in_wake].sum()

    smoothed_rose = gaussian_filter1d(influence_rose, sigma=sigma_kernel, mode='wrap')

    return smoothed_rose, directions


def compute_layout_profiles(
    x_pos: np.ndarray,
    y_pos: np.ndarray,
    wind_turbine,
    n_directions: int = 360,
    verbose: bool = True,
) -> np.ndarray:
    """
    Compute receptivity profiles for all turbines in a layout.
    
    Args:
        x_pos: Turbine x positions
        y_pos: Turbine y positions  
        wind_turbine: PyWake wind turbine object
        n_directions: Number of directions in profile (default 360)
        verbose: Print progress
        
    Returns:
        profiles: Array of shape (n_turbines, n_directions)
    """
    n_turbines = len(x_pos)
    
    # Create site and farm model for simulation
    sim_site = UniformSite()
    sim_site.initial_position = np.column_stack([x_pos, y_pos])
    
    sim_farm = Blondel_Cathelain_2020(
        sim_site,
        windTurbines=wind_turbine,
        turbulenceModel=CrespoHernandez(),
        deflectionModel=JimenezWakeDeflection(),
    )
    
    profiles = []
    recep_profiles = []
    influence_profiles = []
    for i in range(n_turbines):
        if verbose:
            print(f'  Computing profile for turbine {i+1}/{n_turbines}')
        
        # Call the receptivity rose calculation
        receptivity, _ = calc_receptivity_rose(
            sim_farm, 
            turbine_index=i,
            rose_directions=n_directions,
        )
        recep_profiles.append(receptivity)
    
        influence, _ = calc_influence_rose(
            sim_farm, 
            turbine_index=i,
            n_dirs=n_directions,
        )
        influence_profiles.append(influence)


    return np.array(recep_profiles, dtype=np.float32), np.array(influence_profiles, dtype=np.float32)


def rotate_profiles(
    profiles: np.ndarray,
    wind_direction: float,
    profile_directions: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Rotate profiles so that current wind direction is at index 0.
    
    Args:
        profiles: Shape (n_turbines, n_directions) or (batch, n_turbines, n_directions)
        wind_direction: Current wind direction in degrees (scalar or batch)
        profile_directions: Direction values (default: 0-359)
        
    Returns:
        Rotated profiles with same shape as input
    """
    n_directions = profiles.shape[-1]
    
    if profile_directions is None:
        # Assume evenly spaced 0 to 360
        degrees_per_index = 360.0 / n_directions
    else:
        degrees_per_index = profile_directions[1] - profile_directions[0]
    
    # Calculate shift amount
    shift = int(round(wind_direction / degrees_per_index))
    
    # np.roll shifts right for positive values, we want wind_dir at index 0
    return np.roll(profiles, -shift, axis=-1)