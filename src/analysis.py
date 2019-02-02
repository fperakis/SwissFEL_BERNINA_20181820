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

def AFF(q, atom, choice='AFF'):
    '''
    calculate the AFFs using five Gaussians fitted by
    Waasmeier et al., Acta Crystallographica Section A (1995)
    atom must be 'O' for oxygen or 'H' for hydrogen
    q specifies the range of q-values for which the AFF
    should be calculated in inverse Angstrom [A-1].
    if choice is 'AFF'  = independent atomic form factor
                 'MAFF' = modified atomic form factor
    '''
    s = q/4/np.pi # crystallographer definition of q
    
    if atom == 'O':
        a = [2.960427, 2.508818, 0.637853, 0.722838, 1.142756]
        b = [14.182259, 5.936858, 0.112726, 34.958481, 0.390240]
        c = 0.027014
        f0 = a[0]*np.exp(-b[0]*s*s) + a[1]*np.exp(-b[1]*s*s) +
             a[2]*np.exp(-b[2]*s*s) + a[3]*np.exp(-b[3]*s*s) +
             a[4]*np.exp(-b[4]*s*s) + c
        if choice == 'MAFF':
            # choose modified atomic form factor (MAFF)
            alpha_O = 0.1075
            alpha_H = -4*alpha_O
            delta = 2.01 # A-1
            f0 = f0*(1 + alpha_O*np.exp(-((4*np.pi*s)**2)/2/(delta*delta)))
    elif atom == 'H':
        a = [0.413048, 0.294953, 0.187491, 0.080701, 0.023736]
        b = [15.569946, 32.398468, 5.711404, 61.889874, 1.334118]
        c = 0.000049
        f0 = a[0]*np.exp(-b[0]*s*s) + a[1]*np.exp(-b[1]*s*s) +
             a[2]*np.exp(-b[2]*s*s) + a[3]*np.exp(-b[3]*s*s) +
             a[4]*np.exp(-b[4]*s*s) + c
        if choice == 'MAFF':
            # choose modified atomic form factor (MAFF)
            alpha_O = 0.1075
            alpha_H = -4*alpha_O
            delta = 2.01 # A-1
            f0 = f0*(1 + alpha_H*np.exp(-((4*np.pi*s)**2)/2/(delta*delta)))
    else:
        print('unknown atom: %s' % atom)
        f0 = np.zeros_like(q)
    return f0

def Iq_normalization(q, Iq, nominator, rho=0.1, q_min, q_max, denominator=None, choice='la'):
    '''
    normalize I(q) to S(q) using the Warren normalization (large-angle method = la)
    or Krogh-Moe normalization (integral method = int), set by the choice argument
    nominator is the molecular form factor squared <F^2> (in electron units)
    denominator is the spherical part of the molecular form factor squared <F>^2 (in electron units)
    denominator is usually approximated to be equal to nominator, both can be in atom or molecular basis
    rho is the atomic density (in atoms/A^3)
    q_min and q_max (in A-1) sets the q-limits for the method
    q and Iq are ndarrays with the same shape of momentum transfer (in A-1) and radial intensity, respectively
    '''
    if denominator is None:
        denominator = nominator
    q_indices = (q > q_min) & (q < q_max)
    if choice == 'la':
        norm = np.average(nominator[q_indices]/Iq[q_indices])
    else:
        int_nom = np.trapz(q*q*nominator/denominator, x=q)
        int_denom = np.trapz(q*q*Iq/denominator, x=q)
        norm = (int_nom-2*np.pi*np.pi*rho)/int_denom
    Sq = (norm*Iq-nominator)/denominator
    return norm, Sq
