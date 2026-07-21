"""Derating-capable turbine surrogates (DTU 10MW + IEA 3.4-130).

Plain ``DTU10MW()`` has no ``derate`` input, so WindFarmEnv's
``check_turbine_supports_derating`` rejects it. This module ports
``make_derating_dtu10mw()`` from
``windgym/examples/Example 7 Power tracking RL setup.ipynb`` (cell 38866f2c):
it builds a py_wake ``WindTurbine`` whose ``powerCtFunction`` is a
``PowerCtNDTabular`` over ``["ws", "yaw", "derate"]`` (defaults yaw=0,
derate=0), tabulated from the reduced power+ct surrogate shipped with WindGym.
``make_derating_iea34()`` applies the same recipe to the CCBlade-generated
IEA 3.4-130 tables (``build_iea34_derating_tables.py``), in two variants:

- ``annrpm`` (default): rotor speed pinned to the DLC12 ``wrot_RotSpd`` ANN
  schedule, constant across derating — ct consistent with the HF controller
  the IEA34 load surrogates saw.
- ``minct``: pure min-Ct derating with free rotor speed.

Use the ``make_derating_turbine`` / ``make_operating_point_lookup_for``
dispatchers in trainer/eval code so ``--turbtype`` selects the turbine.

The surrogate ``.nc`` files live under ``windgym/examples/data/`` in the repo,
so we resolve them relative to this file (repo_root/windgym/...) rather than
assuming the notebook's ``examples/`` working directory.
"""

from pathlib import Path

import xarray as xr

from py_wake.examples.data.dtu10mw._dtu10mw import DTU10MW
from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import DensityScale, PowerCtNDTabular

_DATA_DIR = Path(__file__).resolve().parents[2] / "windgym/examples/data"

SURROGATE_NC = _DATA_DIR / "dtu10mw_derating_yaw_surrogate.nc"

# The un-reduced HAWCStab2 table: same (ws, yaw, derating) grid but with the
# underlying pitch [deg] / tsr / cp variables kept. Used to report the
# steady-state blade pitch and rotor RPM alongside the surrogate power.
FULL_SURROGATE_NC = _DATA_DIR / "dtu10mw_derating_yaw_surrogate_full.nc"

IEA34_VARIANTS = ("annrpm", "minct")
IEA34_DEFAULT_VARIANT = "annrpm"
IEA34_DIAMETER = 130.0
IEA34_HUB_HEIGHT = 110.0


def iea34_surrogate_nc(variant: str = IEA34_DEFAULT_VARIANT, full: bool = False) -> Path:
    if variant not in IEA34_VARIANTS:
        raise ValueError(
            f"Unknown IEA34 variant {variant!r}; expected one of {IEA34_VARIANTS}"
        )
    suffix = "_full" if full else ""
    return _DATA_DIR / f"iea34_derating_yaw_surrogate_{variant}{suffix}.nc"


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


def make_derating_iea34(
    variant: str = IEA34_DEFAULT_VARIANT, nc_path: Path | None = None
) -> WindTurbine:
    """Build an IEA 3.4-130 WindTurbine with a (ws, yaw, derate) surrogate.

    Same ``PowerCtNDTabular`` recipe as ``make_derating_dtu10mw``, tabulated
    from the CCBlade-generated table (steady-BEM electrical power, eta=0.936;
    see ``build_iea34_derating_tables.py`` for the variant semantics and the
    accepted 15-20% below-rated offset vs the DLC12 ANN power).
    """
    nc_path = Path(nc_path) if nc_path is not None else iea34_surrogate_nc(variant)
    if not nc_path.exists():
        raise FileNotFoundError(
            f"IEA34 derating surrogate not found at {nc_path}. Generate it "
            "with build_iea34_derating_tables.py (wisdem-env) or pull the "
            "windgym submodule commit that ships it."
        )

    ds = xr.load_dataset(nc_path).transpose("ws", "yaw", "derating")

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
        name=f"IEA34_130_{variant}",
        diameter=IEA34_DIAMETER,
        hub_height=IEA34_HUB_HEIGHT,
        powerCtFunction=pctf,
    )


def make_derating_turbine(
    turbtype: str = "IEA34", iea34_variant: str = IEA34_DEFAULT_VARIANT
) -> WindTurbine:
    """Dispatch on ``--turbtype`` for the derate-action env branch.

    Old checkpoints saved ``turbtype="DTU10MW"`` in their args, so
    checkpoint-driven env rebuilds stay DTU automatically; new runs default
    to IEA34/annrpm via config.Args.
    """
    if turbtype == "IEA34":
        return make_derating_iea34(variant=iea34_variant)
    if turbtype == "DTU10MW":
        return make_derating_dtu10mw()
    raise ValueError(
        f"No derating surrogate for turbtype {turbtype!r} "
        "(expected 'IEA34' or 'DTU10MW')"
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


def make_operating_point_lookup_for(
    turbtype: str = "IEA34",
    iea34_variant: str = IEA34_DEFAULT_VARIANT,
    rotor_diameter=None,
):
    """Turbine-dispatched (ws, yaw, derate) -> (pitch, rpm) lookup.

    The IEA34 tables store tsr on R = 65.0 m (= diameter/2), so the default
    rotor_diameter round-trips the ANN rpm schedule exactly.
    """
    if turbtype == "IEA34":
        from WindGym.core import OperatingPointLookup

        nc_path = iea34_surrogate_nc(iea34_variant, full=True)
        if not nc_path.exists():
            raise FileNotFoundError(
                f"Full IEA34 derating surrogate not found at {nc_path}. "
                "Generate it with build_iea34_derating_tables.py (wisdem-env)."
            )
        if rotor_diameter is None:
            rotor_diameter = IEA34_DIAMETER
        return OperatingPointLookup.from_netcdf(nc_path, rotor_diameter=rotor_diameter)
    if turbtype == "DTU10MW":
        return make_operating_point_lookup(rotor_diameter=rotor_diameter)
    raise ValueError(
        f"No operating-point lookup for turbtype {turbtype!r} "
        "(expected 'IEA34' or 'DTU10MW')"
    )
