import numpy as np
import h5py
import time
import sys

sys.path.insert(0, '../src/')
from scipy.signal import medfilt
from data_analysis import *
from integrators import *

def load_processed_data(run,path=None):
    '''
    loads data from processed h5
    '''
    
    if path is None:
        h5path = '/sf/bernina/data/p17743/res/work/hdf5/run%s.h5'%run
    else:
        h5path = path+'run%s.h5'%run
    h5file = h5py.File(h5path,'r')
    try:
        sum = h5file['JF7/2D_sum'][:]
    except KeyError:
        sum = np.zeros(( 4432, 4215 )) # bad solution: fix this in file instead
    Iq = h5file['JF7/I_Q'][:]
    r = h5file['JF7/Q_bins'][:]
    
    try:
        i0 = h5file['JF3/i0'].value
    except KeyError:
        i0 = h5file['JF7/i0'].value
    
    nshots = h5file['JF7/num_shots'].value

    try:
        nhits = np.int(h5file['JF7/num_hits'].value)
        sum_hits = h5file['JF7/2D_sum_hits'].value
        Iq_thr = h5file['JF7/I_threshold'].value
    except KeyError:
        nhits = 0
        sum_hits = np.zeros_like(sum)
        Iq_thr = 0
    
    laser_i0 = h5file['SARES20/i0'].value
    try:
        laser_on = h5file['SARES20/laser_on'].value
    except KeyError:
        laser_on = h5file['BERNINA/laser_on'].value
    event_ID = h5file['pulse_id'].value
          
    print('run%s: %d shots' % (run, h5file['JF7/num_shots'].value))
    h5file.close()
        
    return sum,Iq,r,int(nshots),sum_hits,Iq_thr,nhits,i0,laser_i0,laser_on,event_ID #maybe use dictionary here


def do_histogram(a,bi,bf,db):
    '''
    finally a good histogram function
    '''
    bins = np.arange(bi-db/2.,bf+db/2.+db,db)
    y,x_tmp = np.histogram(a,bins=bins)
    x = np.array([(bins[j]+bins[j+1])/2. for j in range(len(bins)-1)])
    return x,y


def find_hits(Iq, threshold=0.015, r_min=200, r_max=400):
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
    median_filtered_Iq = medfilt(Iq, (1, filter_length))

    ice_range_idx = (q > q_min) * (q < q_max)
    diff = np.abs(median_filtered_Iq[:,ice_range_idx] - median_filtered_Iq[:,np.roll(ice_range_idx, 1)])
    metric = np.max(diff, axis=1)
    hits = ( metric > threshold )

    #omedian_filtered_Iq = median_filter(Iq, filter_length=filter_length)
    #ice_range_indices = np.where((q > q_min) & (q < q_max))
    #ometric = np.array([np.max(np.abs(i[ice_range_indices]-i[(ice_range_indices[0]+1)])) for i in omedian_filtered_Iq])
    #oldhits = metric>threshold

    #print(np.sum(np.abs(ometric - metric)))
    #print(np.sum(hits - oldhits))

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

def pump_probe_signal(Iq,hits,laser_on,misses=None,r_min=200,r_max=400):
    '''
    calculate the pump probe signal
    '''
    # cast laser_on to boolean (otherwise it messes up the code -ask TJ)
    laser_on = laser_on.astype(bool)

    # averages
    if misses is None:
        misses = np.logical_not(hits)
    Iq_miss_avg = np.average(Iq[misses,:], axis=0)
    # hit sanity check
    if hits.sum() > 0:
        Iq_hit_avg = np.average(Iq[hits,:], axis=0)
    else:
        return np.zeros_like(Iq_miss_avg),Iq_miss_avg,np.zeros_like(Iq_miss_avg)
    
    # misses (laser on/off)
    Iq_off_misses = np.average(Iq[misses * np.logical_not(laser_on),:], axis=0)
    Iq_on_misses  = np.average(Iq[misses * laser_on,:], axis=0)

    # hits (laser on/off)
    hits_off = hits * np.logical_not(laser_on)
    hits_on = hits * laser_on
    if (hits_on.sum() > 0) and (hits_off.sum() > 0):
        Iq_off_hits = np.average(Iq[hits_off,:], axis=0)
        Iq_on_hits  = np.average(Iq[hits_on,:], axis=0)
    else:
        return Iq_hit_avg,Iq_miss_avg,np.zeros_like(Iq_miss_avg)
    
    # signal
    #diff_signal = on_hits - off_hits
    diff_signal = normalize(Iq_on_hits, r_min, r_max) - normalize(Iq_off_hits, r_min, r_max) # / normalize(Iq_off_hits, l, h)
    return Iq_hit_avg,Iq_miss_avg,diff_signal

def subtract_background(Iq,hits,i0,nshots,misses=None):
    '''
    Calculates the average of missed shots and subtracts it as a background
    '''

    # in case there are no hits 
    if hits.sum() == 0:
        return None
    
    # calculate background based on normalised misses
    if misses is None:
        miss = np.logical_not(hits)
    else:
        miss = misses
    Iq_background = np.average(Iq[miss],axis=0,weights=i0[miss])
    
    # subtract background
    Iq_corr = np.zeros_like(Iq[hits])
    
    for i in range(np.sum(hits)):
        norm = i0[hits][i]/np.average(i0[hits])
        Iq_corr[i] = Iq[hits][i]/norm - Iq_background/norm
    
    return Iq_corr


def pump_probe_signal_2(Iq,hits,laser_on,r_min=200,r_max=300):
    '''
    calculate the pump probe signal
    '''
    # in case there are no hits return en empty array
    if hits.sum() > 0:
        # important: cast laser_on to boolean (otherwise it messes up the code -ask TJ)
        laser_on_hits = laser_on[hits].astype(bool)   
        laser_off_hits = np.logical_not(laser_on_hits)
 
        # laser on and off shots
        if (laser_on_hits.sum() > 0) and (laser_off_hits.sum() > 0):
            Iq_on_avg = np.average(Iq[laser_on_hits],axis=0)
            Iq_off_avg = np.average(Iq[laser_off_hits],axis=0)
        else:
            return np.zeros_like(np.average(Iq,axis=0))
        
        # pump-probe difference signal
        diff_signal = normalize(Iq_on_avg, r_min, r_max) - normalize(Iq_off_avg, r_min, r_max)
        return diff_signal
    else:
        return np.zeros_like(np.average(Iq,axis=0))
    

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
        f0 = a[0]*np.exp(-b[0]*s*s) + a[1]*np.exp(-b[1]*s*s) + a[2]*np.exp(-b[2]*s*s) + a[3]*np.exp(-b[3]*s*s) + a[4]*np.exp(-b[4]*s*s) + c
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
        f0 = a[0]*np.exp(-b[0]*s*s) + a[1]*np.exp(-b[1]*s*s) + a[2]*np.exp(-b[2]*s*s) + a[3]*np.exp(-b[3]*s*s) + a[4]*np.exp(-b[4]*s*s) + c
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

def Iq_normalization(q, Iq, nominator, q_min, q_max, rho=0.1, denominator=None, choice='la'):
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
    q_indices = np.where((q >= q_min) & (q <= q_max))
    if choice == 'la':
        norm = np.average(nominator[q_indices]/Iq[q_indices])
        print('Large-angle normalization: %.2f' % norm)
    else:
        int_nom = np.trapz(q[q_indices]*q[q_indices]*nominator[q_indices]/denominator[q_indices], x=q[q_indices])
        int_denom = np.trapz(q[q_indices]*q[q_indices]*Iq[q_indices]/denominator[q_indices], x=q[q_indices])
        norm = (int_nom-2*np.pi*np.pi*rho)/int_denom
        print('Integral normalization: %.2f' % norm)
    Sq = (norm*Iq-nominator)/denominator
    return norm, Sq
