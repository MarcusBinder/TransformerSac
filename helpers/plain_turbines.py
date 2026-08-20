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
"""
from __future__ import annotations


def make_plain_turbine(turbtype: str):
    """Build the plain py_wake WindTurbine for a --turbtype name."""
    if turbtype == "DTU10MW":
        from py_wake.examples.data.dtu10mw import DTU10MW
        return DTU10MW()
    if turbtype == "V80":
        from py_wake.examples.data.hornsrev1 import V80
        return V80()
    if turbtype == "IEA22":
        from helpers.iea_22_rwt import IEA_22MW_H2S
        return IEA_22MW_H2S()
    raise ValueError(
        f"Unknown turbine type: {turbtype} (plain turbines: DTU10MW, V80, IEA22)"
    )
