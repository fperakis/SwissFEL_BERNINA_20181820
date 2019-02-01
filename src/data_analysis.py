from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from escape.parse import swissfel
import h5py
from jungfrau_utils import apply_gain_pede, apply_geometry
import time,sys

sys.path.insert(0, '../src/')
from integrators import *
from pedestals import*
from masking import *

def process_run(run,path,num_shots=0,iq_threshold=0,photon_energy=9500):
    '''
    Script that processes a given run by doing the following:
    * loads files from raw data (.json)
    * applies corrections (gain, pedestals, geometry)
    * calculates average 2D image
    * angular integrator
    '''

    # load data
    file_path = '%srun%s.json'%(path,run)
    print('-- Loading data:%s'%file_path)
    data = swissfel.parseScanEco_v01(file_path,createEscArrays=True,memlimit_mD_MB=50)
    jf7 = data['JF07T32V01'] # large JungFrau data
    jf3 = data['JF03T01V01'] # i0 monitor data 
    total_shots = jf7.data.shape[jf7.eventDim]
    if (num_shots>total_shots) or (num_shots==0):
        num_shots = total_shots
    i0 = np.zeros(total_shots)
    
    # get event IDs at 100Hz and match with JF pulse IDs at 25 Hz
    jf_pulse_id = jf7.eventIds[:num_shots] # event ID at 25 Hz
    evcodes = data['SAR-CVME-TIFALL5:EvtSet'] # trigger codes in 256 channels at 100 Hz
    laser_on_100Hz = evcodes.data[:,20].compute() # laser trigger at 100 Hz
    pulse_id = evcodes.eventIds # event ID at 100 Hz
    matched_id = np.isin(pulse_id, jf_pulse_id) # matched IDs at 25 Hz in 100 Hz arrays
    assert (np.sum(matched_id) ==  len(jf_pulse_id))
    print('-- Asserting that %d matched IDs sum up to %d JF7 event IDs' % (np.sum(matched_id), len(jf_pulse_id)))
    laser_on = laser_on_100Hz[matched_id].astype(np.bool) # laser trigger at 25 Hz
    
    # get laser i0
    laser_i0_100Hz = data['SARES20-LSCP9-FNS:CH1:VAL_GET'].data.compute()
    laser_i0 = laser_i0_100Hz[matched_id]
    
    # load corrections
    gains,pede,noise,mask = load_corrections(run)
    gains_i0,pede_i0,noise_i0,mask_i0 = load_corrections_i0(run)
    mask_double_pixels(mask)

    # apply corrections and geometry
    t0 = time.time()
    icorr = apply_gain_pede(jf7.data[0].compute(),G=gains, P=pede, pixel_mask=mask)
    icorr_sum = apply_geometry(icorr,'JF07T32V01')
    hcorr_sum = np.zeros_like(icorr_sum)
    mask_geom = ~apply_geometry(~(mask>0),'JF07T32V01')
    mask_inv = np.logical_not(mask_geom) #inverted: 0 masked, 1 not masked

    # initialise for angular integration
    rad_dist = radial_distances(icorr_sum,center=(2117,2222))
    ra = RadialAverager(rad_dist, mask_inv)
    r  = ra.bin_centers
    roi_min = 5
    roi_max = 80
    iq = ra(icorr_sum)
    iqs = np.zeros((num_shots, iq.shape[0]))
    iqs[0] = iq
    if iq_threshold > 0:
        num_hits = 0
        if iq[roi_min:roi_max].mean() > iq_threshold:
            hcorr_sum += icorr_sum
            num_hits += 1
            print('run%s - s.%i - %.1f Hz - %.2f photon/pix - HIT'%(run,0,1.0/(time.time() - t0),np.mean(icorr_sum[mask_inv])*1000/photon_energy))
        else:
            print('run%s - s.%i - %.1f Hz - %.2f photon/pix'%(run,0,1.0/(time.time() - t0),np.mean(icorr_sum[mask_inv])*1000/photon_energy))
    else:
        print('run%s - s.%i - %.1f Hz - %.2f photon/pix'%(run,0,1.0/(time.time() - t0),np.mean(icorr_sum[mask_inv])*1000/photon_energy))

    
    # loop over all shots
    for i_shot in range(1,num_shots):
        t1 = time.time()
        i0[i_shot] = get_i0(i_shot,jf3,gains_i0,pede_i0,mask_i0)
        icorr = apply_gain_pede(jf7.data[i_shot].compute(),G=gains, P=pede, pixel_mask=mask)
        icorr_geom = apply_geometry(icorr,'JF07T32V01')
        #r, iq = angular_average(icorr_geom, rad=rad_dist,mask=mask_inv)
        iq = ra(icorr_geom)
        iqs[i_shot] = iq
        icorr_sum += icorr_geom
        if (iq_threshold > 0) and (iq[roi_min:roi_max].mean() > iq_threshold):
            hcorr_sum += icorr_sum
            num_hits += 1
            print('run%s - s.%i - %.1f Hz - %.2f photon/pix - HIT'%(run,i_shot,1.0/(time.time() - t1),np.mean(icorr_geom[mask_inv])*1000/photon_energy))
        else:
            print('run%s - s.%i - %.1f Hz - %.2f photon/pix'%(run,i_shot,1.0/(time.time() - t1),np.mean(icorr_geom[mask_inv])*1000/photon_energy))
    if iq_threshold > 0:
        print('-- Processed %d shots with %d hits: %.01f%%'%(num_shots, num_hits, 100*num_hits/num_shots))
        print('-- Analyzed data in: %d min, %d s'%((time.time()-t0)/60, (time.time()-t0)%60))
    else:
        print('-- Processed %d shots in %d min, %d s'%(num_shots, (time.time()-t0)/60, (time.time()-t0)%60))
    
    save_data = {"JF7":{"2D_sum":icorr_sum, "num_shots":num_shots, "Q_bins":r, "I_Q":iqs}, "JF3":{"i0":i0}, "SARES20":{"i0":laser_i0}, "BERNINA":{"event_ID":jf_pulse_id, "laser_on":laser_on}}
    if iq_threshold > 0:
        save_data["JF7"]["2D_sum_hits"] = hcorr_sum
        save_data["JF7"]["num_hits"] = num_hits
        save_data["JF7"]["I_threshold"] = iq_threshold
    save_path = '/sf/bernina/data/p17743/res/work/hdf5/run%s.h5'%run
    #save_path = './run%s.h5'%run
    print('-- Saving data: %s'%save_path)
    save_h5(save_path,save_data)
    return


def load_corrections(run):
    '''
    Loads the corrections for the jungfrau07 detector (16Mpix)
    '''
    jf3_pede_file, jf7_pede_file = get_pedestals(run)
 
    gain_file = '/sf/bernina/config/jungfrau/gainMaps/JF07T32V01/gains.h5'
    pede_file = jf7_pede_file#'/sf/bernina/data/p17743/res/waterJet_tests/JFpedestal/pedestal_20190125_1507.JF07T32V01.res.h5'
    mask_file = '/sf/bernina/data/p17743/res/JF_pedestals/pedestal_20190115_1551.JF07T32V01.res.h5'
    #try:
    #    if (np.int(run.split('_')[0]) > 26):
    #        pede_file = '/sf/bernina/data/p17743/res/JF_pedestal/pedestal_20190130_1925.JF07T32V01.res.h5'
    #except ValueError:
    #    pass
    with h5py.File(gain_file,'r') as f:
        gains = f['gains'].value
    with h5py.File(pede_file,'r') as f:
        pede = f['gains'].value
    with h5py.File(mask_file,'r') as f:
        noise = f['gainsRMS'].value
        mask = f['pixel_mask'].value
    print('using pedestals from: %s', pede_file)
    return gains,pede,noise,mask


def save_h5(save_path,save_dict):
    '''
    Saves the processed data in h5.
    Takes as input a nested dictionary with {"detector_name":{"dataname": dataset}}
    '''

    h5f = h5py.File(save_path, 'w')

    for group in save_dict:
        h5g = h5f.create_group(group)
        for dataset in save_dict[group]:
            if (dataset.find("num") >= 0) or (dataset.find("event") >= 0):
                h5g.create_dataset(dataset, data=save_dict[group][dataset], dtype='u8') # unsigned 64-bit integer
            elif (dataset.find("laser_on") >= 0):
                h5g.create_dataset(dataset, data=save_dict[group][dataset], dtype='?') # boolean
            else:
                h5g.create_dataset(dataset, data=save_dict[group][dataset], dtype='f') # floating point (32-bit?)
    
    h5f.close()
    return


def load_corrections_i0(run):
    '''
    Loads the corrections for jungfrau03 detector (small one - i0 monitor)
    '''

    jf3_pede_file, jf7_pede_file = get_pedestals(run)
    with h5py.File('/sf/bernina/config/jungfrau/gainMaps/JF03T01V01/gains.h5','r') as f:
        gains = f['gains'].value
    with h5py.File(jf3_pede_file,'r') as f:
        pede = f['gains'].value
    with h5py.File('/sf/bernina/data/p17743/res/JF_pedestal/pedestal_20190115_1551.JF03T01V01.res.h5','r') as f:
        noise = f['gainsRMS'].value
        mask = f['pixel_mask'].value

    return gains,pede,noise,mask


def get_i0(jf3_image,gains,pede,mask):
    '''
    calculates the i0 from an ROI of the  small jungfrau detector (JF3)
    '''
    # parameters
    X1,X2 = 10,240 #260,500
    Y1,Y2 = 260,500

    icorr = apply_gain_pede(jf3_image,G=gains, P=pede, pixel_mask=mask)
    icorr_geom = apply_geometry(icorr,'JF03T01V01')
    i0 = np.average(icorr_geom[X1:X2,Y1:Y2])

    return i0

