import numpy as np
import argparse
import os
from pathlib import Path

#For the site
from dynamiks.sites import TurbulenceFieldSite
from dynamiks.sites.turbulence_fields import MannTurbulenceField


if __name__ == '__main__':
    #This code generated the turbulence boxes that are used for the seeds.

    parser = argparse.ArgumentParser(description="Generate Mann turbulence boxes for seeding.")
    parser.add_argument("-n", "--num-boxes", type=int, default=10,
                        help="Number of turbulence boxes to generate (default: 10).")
    parser.add_argument("-s", "--seed-start", type=int, default=0,
                        help="Ransom seed starting point (default: 0).")
    parser.add_argument("--nxyz", type=int, nargs=3, default=[2048, 512, 128], metavar=("NX", "NY", "NZ"),
                        help="Grid points per axis (default: 2048 512 128).")
    parser.add_argument("--dxyz", type=float, nargs=3, default=[3.0, 3.0, 3.0], metavar=("DX", "DY", "DZ"),
                        help="Grid spacing per axis in m (default: 3 3 3). The box is periodic, so "
                             "N*d should cover the farm crosswind and the advected episode length; "
                             "coarsen d rather than grow N to keep memory fixed.")
    args = parser.parse_args()

    dx, dy, dz = args.dxyz  # default 3 m; Liew sais to use 1.85
    nx, ny, nz = args.nxyz
    print(f"box: Nxyz=({nx}, {ny}, {nz}) dxyz=({dx}, {dy}, {dz}) -> "
          f"{nx*dx/1e3:.1f} x {ny*dy/1e3:.1f} x {nz*dz/1e3:.2f} km")
    
    # Create the boxes folder if it doesn't exist
    boxes_dir = Path("boxes")
    boxes_dir.mkdir(exist_ok=True)

    print("starting seed: ", args.seed_start)

    for seed in range(args.num_boxes):
        print("Generating turb box number ", seed)

        # Generate the turbulence box, and setup the field
        tf = MannTurbulenceField.generate(alphaepsilon=1.0,
                                        Gamma=3.9,
                                        L=33.6,
                                        Nxyz= (int(nx), int(ny), int(nz)),
                                        dxyz= (dx, dy, dz),
                                        seed=seed+args.seed_start)



        # Name by the ACTUAL Mann seed (not the loop index) so runs with a
        # nonzero --seed-start are honestly named and parallel single-box
        # invocations never collide. Identical names for the default start 0.
        filename = boxes_dir / ("TF_seed_" + str(seed + args.seed_start) + ".nc")
        tf.to_netcdf(filename = str(filename))

    print("Is is done")