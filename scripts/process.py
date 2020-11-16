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

import sys
import argparse

sys.path.insert(0, '../src/')
from process import main as process_main

# - parsers
parser = argparse.ArgumentParser(description='Analyze a run of p17743 at SwissFEL.')
parser.add_argument('-r', '--run', type=str, required=True, help='run number to process')
parser.add_argument('-s','--shots',type=int, default=0, help='number of shots to process (default: 0 = all)')
parser.add_argument('-t','--threshold',type=float, default=0, help='hit threshold for radial profile (default: 0 = no hits)')
parser.add_argument('-p','--path',type=str,default='/sf/bernina/data/p17743/res/scan_info/',help='path to data')
parser.add_argument('-l','--taglist',type=str,default='/sf/bernina/data/p17743/res/work/tags/weakice_Taglist_run76_to_79_16_mm.txt',help='text file with shots to save') #default='/sf/bernina/data/p17743/res/work/tags/weakice_Taglist_run165_to_170_75_mm.txt'
#'/das/work/p17/p17743/tags/weakice_Taglist_run165_to_170_75_mm.txt'
#'/sf/bernina/data/p17743/scratch/hdf5'

args = parser.parse_args()

try:
    run = int(args.run)
except ValueError:
    run  = args.run
path = args.path 
shots = args.shots
threshold = args.threshold

# - process run
process_main(run, path=path, num_shots=shots, iq_threshold=threshold, taglist=args.taglist)

