# -*- coding: utf-8 -*-
"""
Created on Tue May  6 09:10:29 2025

@author: leona
"""

## PROBE GENERATION
import os
import math
import numpy as np
from probeinterface import Probe
import matplotlib.pyplot as plt
from probeinterface.plotting import plot_probe
from probeinterface import write_prb, read_prb


'''
############ CHECKED ############


Generate a planar (2D) probe that reflects a MEA device.


1) Generate the grid coordinates for the electrodes;

2) Initialize the Probe object;

3) Set the outer contour of the MEA;

4) Visualize and save.

'''

Save_path = r'C:\Users\leona\Desktop\Prova'
Name_probe = r'CorePlate_6W_38_60.prb'

Number_electrodes =2304
positions = np.zeros((Number_electrodes, 2))



# # CorePlate™ 1W 38/60
# pitch = 60 # um   
# Electrode_size = 21 # um


# # CorePlate™ 1W 27/42L
# pitch = 42 # um   
# Electrode_size = 21 # um


# # CorePlate™ 1W 51/81
# pitch = 81 # um   
# Electrode_size = 21 # um

# CorePlate™ 6W 38/60
pitch = 60 # um   
Electrode_size = 25 # um


# The electrodes are set in square geometry
k =0
n = int(math.sqrt(Number_electrodes))
for i in range(n):
    for j in range(n):
        x = int(i * pitch)
        y = int(j * pitch)
        positions[k] = x, y
        k =k+1



# Generate the probe variable

probe_2d = Probe(ndim=2, si_units='um')
probe_2d.set_contacts(positions=positions, shapes='square', shape_params={'width': Electrode_size})


# Plot the probe
%matplotlib
plot_probe(probe_2d)


# Save
os.chdir(Save_path)
with open(Name_probe, 'w') as f:
    f.write(probe_2d)
