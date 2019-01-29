from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import h5py
import time
import sys
import argparse

from escape.parse import swissfel
from jungfrau_utils import apply_gain_pede, apply_geometry
#h5py.enable_ipython_completer()

sys.path.insert(0, '../src/')
from data_analysis import *
from integrators import *

# - parsers
parser = argparse.ArgumentParser(description='Analyze a run of xcslr0016. Use MPI!')
parser.add_argument('-r', '--run', type=str, required=True, help='run number to process')
parser.add_argument('-s','--shots',type=int, default=10, help='number of shots to process')
parser.add_argument('-p','--path',type=str,default='/sf/bernina/data/p17743/res/scan_info/',help='path to data')
'/sf/bernina/data/p17743/scratch/hdf5'

args = parser.parse_args()
run  = args.run
path = args.path 
shots = args.shots

# - process run
process_run(run,path,num_shots = shots)
