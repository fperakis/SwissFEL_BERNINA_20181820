import numpy as np
import matplotlib.pyplot as plt
import h5py
import time
import sys

sys.path.insert(0, '../src/')
from pedestals import *

def discover_run_h5(run_num,path=None):
    '''
    discovers h5 file of a given run number
    uses: pedestals.discover_files
    '''
    run_file = []
    if path is None:
        run_path = '/sf/bernina/data/p17743/res/work/hdf5/'
    else:
        run_path = path
    h5_files = discover_files(run_path)
    n_file = len(h5_files)

    for h5 in h5_files:
        run_str = '%04d'%run_num
        if run_str in h5:
            run_file = h5
            
    return run_file
