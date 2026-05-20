import numpy as np
import argparse
import os
from pathlib import Path

#For the site
from dynamiks.sites import TurbulenceFieldSite
from dynamiks.sites.turbulence_fields import MannTurbulenceField


if __name__ == '__main__':
    #This code generated the turbulence boxes that are used for the seeds.
    
    dx = 3.0  #Liew sais to use 1.85
    dy = 3.0
    dz = 3.0
    
    
    nx = 2048
    ny = 512
    nz = 128 
    
    # Create the boxes folder if it doesn't exist
    boxes_dir = Path("boxes")
    boxes_dir.mkdir(exist_ok=True)

    for seed in range(10): # Generate 10 boxes with different seeds
        print("Generating turb box number ", seed)

        # Generate the turbulence box, and setup the field
        tf = MannTurbulenceField.generate(alphaepsilon=1.0, 
                                        Gamma=3.9, 
                                        L=33.6, 
                                        Nxyz= (int(nx), int(ny), int(nz)), 
                                        dxyz= (dx, dy, dz), 
                                        seed=seed)
                                        
                                        
                                        
        filename = boxes_dir / ("TF_seed_" + str(seed))
        tf.to_netcdf(filename = str(filename))

    print("Is is done")