import numpy as np
import matplotlib.pyplot as plt
import h5py
import time
import sys
import os

def get_pedestals(run, data_path = '/sf/bernina/data/p17743/res/scan_info/',
                       pede_path = '/sf/bernina/data/p17743/res/JF_pedestal/'):
    '''
    Script that returns the most recent pedestals filenames for a given run
    '''

    # discover pedestal files
    pede_files = discover_files(pede_path)
    n_files = len(pede_files)

    # separate pedestals from JF03 and JF07
    jf3_files = []
    jf7_files = []
    for i in range(n_files):
        if 'JF03' in pede_files[i]:
            jf3_files = np.append(jf3_files,pede_files[i])
        elif 'JF07' in pede_files[i]:
            jf7_files = np.append(jf7_files,pede_files[i])

    # get dates of jf3
    n_files_jf3 = len(jf3_files)
    jf3_date = np.zeros(n_files_jf3)
    for i in range(n_files_jf3):
        jf3_date[i] = get_date(pede_path + jf3_files[i])

    # get dates of jf7
    n_files_jf7 = len(jf7_files)
    jf7_date = np.zeros(n_files_jf7)
    for i in range(n_files_jf7):
        jf7_date[i] = get_date(pede_path + jf7_files[i])

    # get date of .json file
    file_path = '%srun%s.json'%(data_path,run)
    data_date = get_date(file_path)

    # find recent jf3 file
    diff = np.abs(jf3_date-data_date)
    index_min = np.where(diff == np.min(diff))[0][0]
    recent_jf3 = jf3_files[index_min]
    recent_jf3_path = pede_path + recent_jf3

    # find recent jf7 file
    diff = np.abs(jf7_date-data_date)
    index_min = np.where(diff == np.min(diff))[0][0]
    recent_jf7 = jf7_files[index_min]
    recent_jf7_path = pede_path + recent_jf7

    return recent_jf3_path, recent_jf7_path 


def get_date(filepath):
    '''
    Returns the when file was last modified
    '''
    stat = os.stat(filepath)
    return stat.st_mtime


def discover_files(path):
    '''
    Looks in the given directory and returns the filenames
    '''
    for (dirpath, dirnames, filenames) in os.walk(path):
        break
    return filenames


