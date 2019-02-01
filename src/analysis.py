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
        i0 = h5file['JF7/i0'].value
    except KeyError:
        i0 = h5file['JF3/i0'].value
    
    nshots = h5file['JF7/num_shots'].value
        
    laser_i0 = h5file['SARES20/i0'].value
    laser_on = h5file['BERNINA/laser_on'].value
    event_ID = h5file['BERNINA/event_ID'].value
          
    print('run%s: %d shots' % (run, h5file['JF7/num_shots'].value))
    h5file.close()
        
    return img,Iq,r,nshots,i0,laser_i0,laser_on,event_ID #maybe use dictionary here


def do_histogram(a,bi,bf,db):
    '''
    finally a good histogram function
    '''
    bins = np.arange(bi-db,bf+db,db)
    y,x_tmp = np.histogram(a,bins=bins)
    x = np.array([(bins[j]+bins[j+1])/2. for j in range(len(bins)-1)])
    return x,y


def find_hits(Iq,threshold = 0.5, q_min = 30, q_max = 80):
    '''
    finds the shots that hit water droplets based on
    a simple threshold on the average over a q-range
    note: give Iq in photon/pix units
    '''
    metric = np.average(Iq[:,q_min:q_max],axis=1)
    hits = metric>threshold
    return metric,hits

def normalize(array, low, high):
    n = np.sum(array[low:high])
    norm_array = array / n
    return norm_array

def pump_probe_signal(Iq,hits,laser_on,q_min=20,q_max=30):
    '''
    calculate the pump probe signal
    '''
    # averages
    hit_avg = np.average(Iq[hits,:], axis=0) 
    miss_avg = np.average(Iq[np.logical_not(hits),:], axis=0) 

    # misses (laser on/off)
    off_misses = np.average(Iq[np.logical_not(hits) * np.logical_not(laser_on),:], axis=0)
    on_misses  = np.average(Iq[np.logical_not(hits) * laser_on,:], axis=0)

    # hits (laser on/off)
    off_hits = np.average(Iq[hits * np.logical_not(laser_on),:], axis=0)
    on_hits  = np.average(Iq[hits * laser_on,:], axis=0)

    # signal
    #diff_signal = on_hits - off_hits
    diff_signal = normalize(on_hits, q_min, q_max) - normalize(off_hits, q_min, q_max) # / normalize(off_hit, l, h)
    return hit_avg,miss_avg,diff_signal

