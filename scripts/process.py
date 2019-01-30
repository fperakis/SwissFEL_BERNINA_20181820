#!/usr/bin/env python
"""
Loads a run file (.json) and creates an hdf5 file with 
the average image and the angular average of each shot

Example use:
 ./process.py -r 0000_test01 -s 10

"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import h5py
import time
import sys
import argparse

from escape.parse import swissfel
from jungfrau_utils import apply_gain_pede, apply_geometry

sys.path.insert(0, '../src/')
from data_analysis import *
from integrators import *

# - parsers
parser = argparse.ArgumentParser(description='Analyze a run of p17743 at SwissFEL.')
parser.add_argument('-r', '--run', type=str, required=True, help='run number to process')
parser.add_argument('-s','--shots',type=int, default=0, help='number of shots to process (default: 0 = all)')
parser.add_argument('-t','--threshold',type=int, default=0, help='hit threshold for radial profile (default: 0 = no hits)')
parser.add_argument('-p','--path',type=str,default='/sf/bernina/data/p17743/res/scan_info/',help='path to data')
'/sf/bernina/data/p17743/scratch/hdf5'

args = parser.parse_args()
run  = args.run
path = args.path 
shots = args.shots
threshold = args.threshold

# - process run
process_run(run, path, num_shots=shots, iq_threshold=threshold)
