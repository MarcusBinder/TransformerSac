"""Plain (non-derating) turbine dispatch by --turbtype name.

One tiny seam so the trainer and every offline eval script build the SAME
turbine object from a checkpoint's ``turbtype`` string. Derate-capable
configs go through helpers/derating_turbine.make_derating_turbine instead;
this helper only serves the plain-turbine else-branch.

Turbines:
  DTU10MW : py_wake's shipped 10 MW reference (D=178.3 m) -- the LES-3x3
            Stage 1-5 campaign turbine.
  V80     : py_wake's Horns Rev 1 Vestas V80 (D=80 m).
  IEA22   : IEA_22MW_H2S (D=284 m, hub 170 m), vendored VERBATIM from the
            LESRL repo (helpers/iea_22_rwt.py + iea_22_rwt.pwr): the turbine
            of the LESRL LES reference case, so the "real"-scale LES-3x3
            arms (Stage 6) are turbine-identical to the LES rows. py_wake
            ships an IEA_22MW_280_RWT module too, but its tabular CSV is
            missing from the installed package, and parity with LESRL's H2S
            power/ct tables (0.95429... electrical factor) is the point.
  IEA22H2 : IEA_22MW_HAWC2Surrogate (Stage 8; same geometry as IEA22). A 2-D
            (ws, yaw) PowerCtNDTabular of HAWC2 aerodynamic rotor power and
            CT, generated from time-domain HAWC2 runs of the exact model the
            LESRL DWM+HAWC2 eval harness uses (helpers/
            iea22_hawc2_ws_yaw_surrogate.nc, vendored VERBATIM from LESRL
            TransformerSac@fa8ed62). NO SimpleYawModel: the table carries the
            yaw loss (~cos^1.65 at 9 m/s vs cos^2.9 in IEA22) and dynamiks
            forwards the ``yaw`` sensor because it is an optional input of
            the tabular. Aero (not electrical) power, so power_max is 23.1 MW
            (IEA22: 22.0 MW) -- obs/reward scaling differs, s8 checkpoints are
            not bitwise-comparable to s7. Train with the LOWERCASE string
            ``iea22h2``: the LESRL harnesses compare ``turbtype.lower()``.

Dispatch is case-insensitive (Stage 7 checkpoints store ``IEA22``, Stage 8
stores ``iea22h2``).
"""
from __future__ import annotations


def make_plain_turbine(turbtype: str):
    """Build the plain py_wake WindTurbine for a --turbtype name (case-insensitive)."""
    key = str(turbtype).upper()
    if key == "DTU10MW":
        from py_wake.examples.data.dtu10mw import DTU10MW
        return DTU10MW()
    if key == "V80":
        from py_wake.examples.data.hornsrev1 import V80
        return V80()
    if key == "IEA22":
        from helpers.iea_22_rwt import IEA_22MW_H2S
        return IEA_22MW_H2S()
    if key == "IEA22H2":
        from helpers.iea_22_rwt import IEA_22MW_HAWC2Surrogate
        return IEA_22MW_HAWC2Surrogate()
    raise ValueError(
        f"Unknown turbine type: {turbtype} "
        "(plain turbines: DTU10MW, V80, IEA22, IEA22H2 [HAWC2 (ws,yaw) surrogate])"
    )
