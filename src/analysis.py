import numpy as np
import h5py
import time
import sys

sys.path.insert(0, '../src/')
from data_analysis import *
from integrators import *

def load_processed_data(run):
    '''
    loads data from processed h5
    '''

    h5path = '/sf/bernina/data/p17743/res/work/hdf5/run%s.h5'%run
    h5file = h5py.File(h5path,'r')
    img = h5file['JF7/2D_sum'][:]
    Iq = h5file['JF7/I_Q'][:]
    r = h5file['JF7/Q_bins'][:]
    
    try:
        i0 = h5file['JF3/i0'].value
    except KeyError:
        i0 = h5file['JF7/i0'].value
    
    nshots = h5file['JF7/num_shots'].value
        
    laser_i0 = h5file['SARES20/i0'].value
    try:
        laser_on = h5file['SARES20/laser_on'].value
    except KeyError:
        laser_on = h5file['BERNINA/laser_on'].value
    event_ID = h5file['pulse_id'].value
          
    print('run%s: %d shots' % (run, h5file['JF7/num_shots'].value))
    h5file.close()
        
    return img,Iq,r,int(nshots),i0,laser_i0,laser_on,event_ID #maybe use dictionary here


def do_histogram(a,bi,bf,db):
    '''
    finally a good histogram function
    '''
    bins = np.arange(bi-db/2.,bf+db/2.+db,db)
    y,x_tmp = np.histogram(a,bins=bins)
    x = np.array([(bins[j]+bins[j+1])/2. for j in range(len(bins)-1)])
    return x,y


def find_hits(Iq, threshold=0.015, r_min=30, r_max=80):
    '''
    finds the shots that hit water droplets based on
    a simple threshold on the average over a q-range
    note: give Iq in photon/pix units/i0
    '''
    metric = np.average(Iq[:,r_min:r_max],axis=1)
    hits = metric>threshold
    return metric,hits

def find_ice(Iq, q, threshold=0.1, filter_length=5, q_min=1.0, q_max=4.5):
    '''
    finds the shots that hit ice droplets
    based on maximum gradient of median filtered intensities
    note: give Iq in photon/pix units/i0
    '''
    median_filtered_Iq = median_filter(Iq, filter_length=filter_length)
    ice_range_indices = np.where((q > q_min) & (q < q_max))
    metric = np.array([np.max(np.abs(i[ice_range_indices]-i[(ice_range_indices[0]+1)])) for i in median_filtered_Iq])
    hits = metric>threshold
    return metric,hits

def median_filter(arr, filter_length=5):
    '''
    median filter of 1D or 2D ndarray along fast-scan
    '''
    if arr.ndim == 1:
        median_filtered_arr = np.zeros((arr.shape[0]-filter_length,))
        for i in range(median_filtered_arr.shape[0]):
            median_filtered_arr[i] = np.median(arr[i:i+filter_length])
    else:
        median_filtered_arr = np.zeros((arr.shape[0], arr.shape[1]-filter_length))
        for i in range(median_filtered_arr.shape[0]):
            for j in range(median_filtered_arr.shape[1]):
                median_filtered_arr[i][j] = np.median(arr[i,j:j+filter_length])
    return median_filtered_arr

def normalize(array, low, high, subtract=False):
    if subtract:
        n = np.max(array[low:high])
        m = array.min()
        norm_array = (array-m) / (n-m) # normalized between 0 and 1
    else:
        n = np.sum(array[low:high])
        norm_array = array / n # normalized to area
    return norm_array

def pump_probe_signal(Iq,hits,laser_on,misses=None,r_min=20,r_max=30):
    '''
    calculate the pump probe signal
    '''
    # averages
    hit_avg = np.average(Iq[hits,:], axis=0) 
    if misses is None:
        misses = np.logical_not(hits)
    miss_avg = np.average(Iq[misses,:], axis=0)
    
    # misses (laser on/off)
    off_misses = np.average(Iq[misses * np.logical_not(laser_on),:], axis=0)
    on_misses  = np.average(Iq[misses * laser_on,:], axis=0)

    # hits (laser on/off)
    off_hits = np.average(Iq[hits * np.logical_not(laser_on),:], axis=0)
    on_hits  = np.average(Iq[hits * laser_on,:], axis=0)

    # signal
    #diff_signal = on_hits - off_hits
    diff_signal = normalize(on_hits, r_min, r_max) - normalize(off_hits, r_min, r_max) # / normalize(off_hit, l, h)
    return hit_avg,miss_avg,diff_signal

