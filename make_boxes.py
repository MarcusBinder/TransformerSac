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
    args = parser.parse_args()

    dx = 3.0  #Liew sais to use 1.85
    dy = 3.0
    dz = 3.0
    nx = 2048
    ny = 512
    nz = 128
    
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



        filename = boxes_dir / ("TF_seed_" + str(seed) + ".nc")
        tf.to_netcdf(filename = str(filename))

    print("Is is done")