"""Derating-capable DTU 10MW turbine surrogate.

Plain ``DTU10MW()`` has no ``derate`` input, so WindFarmEnv's
``check_turbine_supports_derating`` rejects it. This module ports
``make_derating_dtu10mw()`` from
``windgym/examples/Example 7 Power tracking RL setup.ipynb`` (cell 38866f2c):
it builds a py_wake ``WindTurbine`` whose ``powerCtFunction`` is a
``PowerCtNDTabular`` over ``["ws", "yaw", "derate"]`` (defaults yaw=0,
derate=0), tabulated from the reduced power+ct surrogate shipped with WindGym.

The surrogate ``.nc`` lives under ``windgym/examples/data/`` in the repo, so we
resolve it relative to this file (repo_root/windgym/...) rather than assuming
the notebook's ``examples/`` working directory.
"""

from pathlib import Path

import xarray as xr

from py_wake.examples.data.dtu10mw._dtu10mw import DTU10MW
from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import DensityScale, PowerCtNDTabular

# TransformerSac/helpers/derating_turbine.py -> parents[2] is the repo root.
SURROGATE_NC = (
    Path(__file__).resolve().parents[2]
    / "windgym/examples/data/dtu10mw_derating_yaw_surrogate.nc"
)

# The un-reduced HAWCStab2 table: same (ws, yaw, derating) grid but with the
# underlying pitch [deg] / tsr / cp variables kept. Used to report the
# steady-state blade pitch and rotor RPM alongside the surrogate power.
FULL_SURROGATE_NC = (
    Path(__file__).resolve().parents[2]
    / "windgym/examples/data/dtu10mw_derating_yaw_surrogate_full.nc"
)


def make_derating_dtu10mw(nc_path: Path = SURROGATE_NC) -> WindTurbine:
    """Build a DTU 10MW WindTurbine with a (ws, yaw, derate) power/ct surrogate.

    Returns a py_wake ``WindTurbine`` (D ~ 178.3 m) whose power curve responds
    to an absolute ``derate`` setpoint in [0, ~0.8], as required by WindFarmEnv's
    derate action path.
    """
    nc_path = Path(nc_path)
    if not nc_path.exists():
        raise FileNotFoundError(
            f"Derating surrogate not found at {nc_path}. Expected the WindGym "
            f"submodule to be checked out at repo_root/windgym."
        )

    ds = xr.load_dataset(nc_path).transpose("ws", "yaw", "derating")
    ref = DTU10MW()

    pctf = PowerCtNDTabular(
        input_keys=["ws", "yaw", "derate"],
        value_lst=[ds.ws.values, ds.yaw.values.astype(float), ds.derating.values],
        power_arr=ds.power.values,
        power_unit="W",
        ct_arr=ds.ct.values,
        default_value_dict={"yaw": 0.0, "derate": 0.0},
        additional_models=[DensityScale(1.225)],
    )
    for gi in pctf.interp:
        gi.bounds = "limit"

    return WindTurbine(
        name="DTU10MW",
        diameter=ref.diameter(),
        hub_height=ref.hub_height(),
        powerCtFunction=pctf,
    )


def make_operating_point_lookup(nc_path: Path = FULL_SURROGATE_NC, rotor_diameter=None):
    """Build the steady-state (ws, yaw, derate) -> (pitch, rpm) lookup.

    Pass the result as WindFarmEnv's / FarmEval's ``op_lookup`` kwarg so eval
    reports blade pitch and rotor RPM per turbine. Defaults to the DTU 10MW
    rotor diameter and the full surrogate table shipped with WindGym.
    """
    from WindGym.core import OperatingPointLookup

    nc_path = Path(nc_path)
    if not nc_path.exists():
        raise FileNotFoundError(
            f"Full derating surrogate not found at {nc_path}. It is the "
            f"un-reduced companion of {SURROGATE_NC.name} (with pitch/tsr "
            f"variables); expected under windgym/examples/data/."
        )
    if rotor_diameter is None:
        rotor_diameter = DTU10MW().diameter()
    return OperatingPointLookup.from_netcdf(nc_path, rotor_diameter=rotor_diameter)
