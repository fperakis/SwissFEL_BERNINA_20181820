from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from escape.parse import swissfel
import h5py
from jungfrau_utils import apply_gain_pede, apply_geometry
import time
import sys
from data_analysis import load_corrections, save_h5, load_corrections_i0, get_i0
from smalldata import *

sys.path.insert(0, '../src/')
from integrators import *

# global
iq_threshold=0
photon_energy=9500
path = '/sf/bernina/data/p17743/res/scan_info/'

def yield_shots(run, num_shots=0):
    '''
    Script that processes a given run by doing the following:
    * loads files from raw data (.json)
    * applies corrections (gain, pedestals, geometry)
    * calculates average 2D image
    * angular integrator
    '''

    # load data
    file_path = '%srun%s.json' % (path,run)
    print('-- Loading data:%s' % file_path)

    data = swissfel.parseScanEco_v01(file_path,createEscArrays=True,memlimit_mD_MB=50)
    jf7 = data['JF07T32V01'] # large JungFrau data
    jf3 = data['JF03T01V01'] # i0 monitor data 
    total_shots = jf7.data.shape[jf7.eventDim]

    if (num_shots > total_shots) or (num_shots==0):
        num_shots = total_shots
    
    # get event IDs at 100Hz and match with JF pulse IDs at 25 Hz
    jf_pulse_id = jf7.eventIds[:num_shots] # event ID at 25 Hz
    evcodes = data['SAR-CVME-TIFALL5:EvtSet'] # trigger codes in 256 channels at 100 Hz
    laser_on_100Hz = evcodes.data[:,20] # laser trigger at 100 Hz

    pulse_id = evcodes.eventIds # event ID at 100 Hz
    matched_id = np.isin(pulse_id, jf_pulse_id) # matched IDs at 25 Hz in 100 Hz arrays
    assert (np.sum(matched_id) ==  len(jf_pulse_id))
    print('-- Asserting that %d matched IDs sum up to %d JF7 event IDs'
          '' % (np.sum(matched_id), len(jf_pulse_id)))
    
    # get laser i0
    laser_i0_100Hz = data['SARES20-LSCP9-FNS:CH1:VAL_GET'].data

    # SHOT LOOP HERE !!! 
    for i in range(num_shots):
        event = {'pulse_id' : pulse_id[i],
                 'jf7' : jf7.data[i].compute(),
                 'jf3' : jf3.data[i].compute(),
                 'laser_on' : laser_on_100Hz[matched_id][i].compute(),
                 'laser_i0' : laser_i0_100Hz[matched_id][i].compute() 
                }
        yield event
    
    return


def main(run, num_shots=0):

    t0 = time.time()

    save_path = '/sf/bernina/data/p17743/res/work/hdf5/TEST-run%s.h5' % run
    shot_gen = yield_shots(run, num_shots=num_shots)

    smd = SmallData(save_path, 'pulse_id')
    ds = MPIDataSource(smd, shot_gen, global_gather_interval=3) 

    # load corrections
    gains,pede,noise,mask = load_corrections(run)
    gains_i0,pede_i0,noise_i0,mask_i0 = load_corrections_i0()

    mask_geom = ~apply_geometry(~(mask>0),'JF07T32V01')
    mask_inv = np.logical_not(mask_geom) #inverted: 0 masked, 1 not masked


    # initialize accumulator arrays
    icorr_sum = np.zeros((4432, 4215))
    hcorr_sum = np.zeros((4432, 4215))


    # initialise for angular integration
    rad_dist = radial_distances(icorr_sum, center=(2117,2222))
    ra = RadialAverager(rad_dist, mask_inv)
    r  = ra.bin_centers

    roi_min = 5
    roi_max = 80

    # event loop
    for i_shot, event in enumerate(ds.events()):
        t1 = time.time()

        i0 = get_i0(event['jf3'], gains_i0, pede_i0, mask_i0)

        icorr      = apply_gain_pede(event['jf7'],
                                G=gains, P=pede, pixel_mask=mask)
        icorr_geom = apply_geometry(icorr,'JF07T32V01')
        icorr_sum += icorr_geom

        iq         = ra(icorr_geom)

        print(event['laser_i0'])
        smd.event({
                   'JF7' : 
                      {'I_Q': iq},
                     "JF3": 
                       {"i0": i0}, 
                     "SARES20":
                       {"i0": event['laser_i0']}, 
                     "BERNINA":
                       {"event_ID": event['pulse_id'], 
                        "laser_on": event['laser_on']}
                     },
                    pulse_id = event['pulse_id'])

        if (iq_threshold > 0) and (iq[roi_min:roi_max].mean() > iq_threshold):
            hcorr_sum += icorr_sum
            num_hits += 1
            is_hit = 1
        else:
            is_hit = 0
        print('run%s - s.%i - %.1f Hz - %.2f photon/pix - HIT = %d'
              '' % (run,
                    i_shot,
                    1.0/(time.time() - t1),
                    np.mean(icorr_geom[mask_inv])*1000/photon_energy,
                    is_hit)
             )

    # RUN SUMMARY PRINT
    if iq_threshold > 0:
        print('-- Processed %d shots with %d hits: %.01f%%'
              ''%(num_shots, num_hits, 100*num_hits/num_shots))
        print('-- Analyzed data in: %d min, %d s'
              ''%((time.time()-t0)/60, (time.time()-t0)%60))
    else:
        print('-- Processed %d shots in %d min, %d s'
              '' % (num_shots, (time.time()-t0)/60, (time.time()-t0)%60))

#    save_data = {"JF7":
#                  {"2D_sum":    icorr_sum, 
#                   "num_shots": num_shots, 
#                   "Q_bins":    r}, 
#                 }
#    if iq_threshold > 0:
#        save_data["JF7"]["2D_sum_hits"] = hcorr_sum
#        save_data["JF7"]["num_hits"]    = num_hits
#        save_data["JF7"]["I_threshold"] = iq_threshold

    # save to small data file
#    smd.save(**save_data)
    smd.save()
    return
    

if __name__ == '__main__':
    main('0019_droplets_10um_2mm', num_shots=10)

