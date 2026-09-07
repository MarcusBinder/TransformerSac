# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 12:39:03 2024

@author: mikf
"""

from pathlib import Path

import numpy as np
import pandas as pd
from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import (
    DensityScale,
    PowerCtNDTabular,
    PowerCtTabular,
)

DATA_PATH = Path(__file__).parent

# IEA 22 MW reference turbine
# Based on tabular data found here: https://github.com/IEAWindTask37/IEA-22-280-RWT/blob/main/outputs/01_steady_states/HAWC2/iea-22-280-rwt-steady-states-hawc2.yaml
IEA_22MW_280_RWT_data = pd.read_csv(DATA_PATH / "IEA-22-280-RWT_tabular.csv", sep=';')
IEA_22MW_280_RWT_power_curve = np.array([IEA_22MW_280_RWT_data["Wind [m/s]"], IEA_22MW_280_RWT_data["Power [MW]"]]).T
IEA_22MW_280_RWT_ct_curve = np.array([IEA_22MW_280_RWT_data["Wind [m/s]"], IEA_22MW_280_RWT_data["Thrust Coefficient [-]"]]).T


class IEA_22MW_280_RWT(WindTurbine):
    def __init__(self, method="linear"):
        """
        Parameters
        ----------
        method : {'linear', 'pchip'}
            linear(fast) or pchip(smooth and gradient friendly) interpolation
        """
        WindTurbine.__init__(
            self,
            name="IEA_22MW_280_RWT",
            diameter=284,
            hub_height=170,
            powerCtFunction=PowerCtTabular(
                IEA_22MW_280_RWT_power_curve[:, 0],
                IEA_22MW_280_RWT_power_curve[:, 1],
                "MW",
                IEA_22MW_280_RWT_ct_curve[:, 1],
                method=method,
            ),
        )

class IEA_22MW_H2S(WindTurbine): # This turbine uses data from Riccardo
    def __init__(self, method="linear"):
        """
        Parameters
        ----------
        method : {'linear', 'pchip'}
            linear(fast) or pchip(smooth and gradient friendly) interpolation
        """

        pwr = np.loadtxt(DATA_PATH / "iea_22_rwt.pwr")
        WindTurbine.__init__(
            self,
            name="IEA_22MW_280_RWT",
            diameter=284,
            hub_height=170,
            powerCtFunction=PowerCtTabular(
                ws=pwr[:, 0],
                power=pwr[:, 1] * 0.9542919819763047,
                power_unit="kW",
                ct=pwr[:, 4],
                ws_cutin=3.0,
                ws_cutout=25.0,
                power_idle=0.0,
                ct_idle=0.0,
            ),
        )

def main():
    wt = IEA_22MW_280_RWT()
    print('Diameter', wt.diameter())
    print('Hub height', wt.hub_height())
    ws = np.arange(3, 25)
    import matplotlib.pyplot as plt
    plt.plot(ws, wt.power(ws), '.-', label='power [W]')
    c = plt.plot([], label='ct')[0].get_color()
    plt.legend()
    ax = plt.twinx()
    ax.plot(ws, wt.ct(ws), '.-', color=c)
    plt.show()


if __name__ == '__main__':
    main()


class IEA_22MW_HAWC2Surrogate(WindTurbine):
    """IEA 22 MW turbine as a 2-D (ws, yaw) lookup of HAWC2 aero power and CT.

    Table generated from time-domain HAWC2 runs of the exact model the DWM+HAWC2
    harness uses (LEShawc2files/htc/input_hawc_yaw_actuator_tipcorr.htc, DTU WE
    controller with its hub wind-speed inputs set), uniform laminar inflow,
    yaw stepped through the servo: see scratchpad/iea22_surrogate/ and the
    provenance attrs of the .nc. `power` is AERODYNAMIC rotor power [W] (what
    `RunPretrainedAgentHawc2.py` reports); `ct = 2 T_shaft / (rho A U_inf^2)`,
    the same definition dynamiks' HAWC2WindTurbines.ct() uses.

    Yaw axis is in the dynamiks convention (the `yaw` sensor / `yaw_a`). The
    table carries the yaw loss itself, so PyWake's SimpleYawModel is NOT in the
    additional models (it would swallow `yaw` and double count). dynamiks
    forwards the `yaw` sensor automatically because `yaw` is an optional input
    (PyWakeWindTurbines.get_kwargs). Same pattern as the DTU10MW derating
    surrogate (windgym/Docs/docs/derating.md).
    """

    def __init__(self, nc_path=None):
        import xarray as xr

        nc_path = nc_path or DATA_PATH / "iea22_hawc2_ws_yaw_surrogate.nc"
        ds = xr.load_dataset(nc_path)  # also carries the coarse simulated grid (ws_sim)
        pctf = PowerCtNDTabular(
            input_keys=["ws", "yaw"],
            value_lst=[ds.ws.values.astype(float), ds.yaw.values.astype(float)],
            power_arr=ds.power.transpose("ws", "yaw").values,
            power_unit="W",
            ct_arr=ds.ct.transpose("ws", "yaw").values,
            default_value_dict={"yaw": 0.0},
            additional_models=[DensityScale(1.225)],  # no SimpleYawModel
        )
        for gi in pctf.interp:
            gi.bounds = "limit"  # clamp outside the table instead of raising
        self.surrogate_attrs = dict(ds.attrs)
        WindTurbine.__init__(
            self,
            name="IEA_22MW_280_RWT_HAWC2S",
            diameter=284,
            hub_height=170,
            powerCtFunction=pctf,
        )
