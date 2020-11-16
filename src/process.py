#from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from escape.parse import swissfel
import h5py
import csv
from jungfrau_utils import apply_gain_pede, apply_geometry
import time
import sys

sys.path.insert(0, '../src/')
from smalldata import *
from integrators import *
from masking import *
from h5_handling import *
from data_analysis import load_corrections, save_h5, load_corrections_i0, get_i0

class ShotYielder:

    def __init__(self, run, path, num_shots=0):
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

        print('found %d total shots' % total_shots)
        print('will process:', num_shots)
        
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
        
        # get spectra
        spectrum = data['SARFE10-PSSS059:FPICTURE.spectrum']
        
        self.jf_pulse_id = jf_pulse_id
        self.jf7         = jf7
        self.jf3         = jf3
        self.spectrum    = spectrum
        self.laser_on_100Hz = laser_on_100Hz
        self.laser_i0_100Hz = laser_i0_100Hz
        self.matched_id = matched_id

        return 

    
    def __call__(self, i):
        event = {'pulse_id' : self.jf_pulse_id[i],
                 'jf7' :      self.jf7.data[i].compute(),
                 'jf3' :      self.jf3.data[i].compute(),
                 'spectrum' : self.spectrum.data[i].compute(),
                 'laser_on' : self.laser_on_100Hz[self.matched_id][i].compute(),
                 'laser_i0' : self.laser_i0_100Hz[self.matched_id][i].compute() 
                }
        return event


def main(run, photon_energy=9500, iq_threshold=0, num_shots=0, 
         path = '/sf/bernina/data/p17743/res/scan_info/', taglist=None):

    t0 = time.time()

    if type(run) is int:
        run_file = discover_run_h5(run, path=path)
        if run_file.startswith('run'):
            run = run_file[3:].split('.')[0]
    elif type(run) is not str:
        print('invalid format on run:', type(run))
    save_path = '/sf/bernina/data/p17743/res/work/hdf5/run%s.h5' % run
    shot_gen = ShotYielder(run, path, num_shots=num_shots)
    tags = []
    if taglist is not None:
        try:
            with open(taglist) as csvfile:
                tagreader = csv.reader(csvfile, delimiter=' ', quotechar='#') 
                for row in tagreader: 
                    tags.append(np.int64(row))
            tags = np.array(tags).flatten()
            print('found %d tags to save from: %s' % (tags.shape[0], taglist))
        except Exception as e:
            print('could not read file: %s' % taglist)
            print(e)
    
    smd = SmallData(save_path, 'pulse_id')

    if num_shots == 0:
        ds = MPIDataSource(smd, shot_gen, global_gather_interval=100)
    else:
        ds = MPIDataSource(smd, shot_gen, global_gather_interval=100, break_after=num_shots)

    # load corrections
    gains,pede,noise,mask = load_corrections(run)
    gains_i0,pede_i0,noise_i0,mask_i0 = load_corrections_i0(run)
    mask_double_pixels(mask)
    mask_geom = ~apply_geometry(~(mask>0),'JF07T32V01')
    mask_inv = np.logical_not(mask_geom) #inverted: 0 masked, 1 not masked


    # initialize accumulator arrays
    icorr_sum = np.zeros((4432, 4215))
    hcorr_sum = np.zeros((4432, 4215))
    num_hits = 0
    if len(tags) > 0:
        tcorr = np.zeros((tags.shape[0], 4432, 4215))
        num_tags = 0


    # initialise for angular integration
    #rad_dist = radial_distances(icorr_sum, center=(2117,2222))
    rad_dist = radial_distances(icorr_sum, center=(2223,2118))
    ra = RadialAverager(rad_dist, mask_inv, n_bins=1000)
    r  = ra.bin_centers
    
    roi_min = 200
    roi_max = 400
    photon_threshold = 6.0

    # event loop
    for i_shot, event in enumerate(ds.events()):
        t1 = time.time()

        i0 = get_i0(event['jf3'], gains_i0, pede_i0, mask_i0)

        icorr = apply_gain_pede(event['jf7'],
                                     G=gains, P=pede, pixel_mask=mask)
        icorr[ icorr < photon_threshold ] = 0.0 # remove zero photon noise
        icorr_geom = apply_geometry(icorr,'JF07T32V01')
        icorr_sum += icorr_geom

        iq         = ra(icorr_geom)

        smd.event({
                   'JF7': 
                       {'I_Q': iq},
                    "JF3": 
                       {"i0": float(i0)},
                    "SARFE10": 
                       {'spectrum': event['spectrum']},
                    "SARES20":
                       {"i0": float(event['laser_i0']),
                        "laser_on": int(event['laser_on'])}, 
                     },
                    pulse_id = int(event['pulse_id']))
        
        if (iq_threshold > 0) and (iq[roi_min:roi_max].mean()/i0 > iq_threshold):
            hcorr_sum += icorr_geom
            num_hits += 1
            is_hit = 1
        else:
            is_hit = 0
        if int(event['pulse_id']) in tags:
            tcorr[np.where(int(event['pulse_id']) == tags)[0]] = icorr_geom
            num_tags += 1
        print('run%s - pid.%d - s.%i - %.1f Hz - %.2f photon/pix - HIT = %d - TAG = %d'
              '' % (run,
                    int(event['pulse_id']),
                    i_shot,
                    1.0/(time.time() - t1),
                    np.mean(icorr_geom[mask_inv])*1000/photon_energy,
                    is_hit,
                    int(event['pulse_id']) in tags)
             )

    # RUN SUMMARY PRINT
    if iq_threshold > 0:
        if len(tags) > 0:
            print('-- Processed %d shots with %d hits and %d tags: %.03f%%'
                  ''%(num_shots, num_hits, num_tags, 100*num_hits/num_shots))
        else:
            print('-- Processed %d shots with %d hits: %.03f%%'
                  ''%(num_shots, num_hits, 100*num_hits/num_shots))
        print('-- Analyzed data in: %d min, %d s'
              ''%((time.time()-t0)/60, (time.time()-t0)%60))
    else:
        print('-- Processed %d shots in %d min, %d s'
              '' % (num_shots, (time.time()-t0)/60, (time.time()-t0)%60))



    # SAVE AGGREGATE / ACCUMULATOR DATA
    smd.sum(icorr_sum)
    smd.sum(hcorr_sum)
    smd.sum(num_hits)
    if len(tags) > 0:
        smd.sum(tcorr)
        smd.sum(num_tags)
        smd.sum(tags)
    
    save_data = {"JF7":
                  {"2D_sum":    icorr_sum, 
                   "num_shots": num_shots, 
                   "Q_bins":    r}, 
                 }

    if iq_threshold > 0:
        save_data["JF7"]["2D_sum_hits"] = hcorr_sum
        save_data["JF7"]["num_hits"]    = num_hits
        save_data["JF7"]["I_threshold"] = iq_threshold

    if len(tags) > 0:
        save_data["JF7"]["2D_tags"] = tcorr
        save_data["JF7"]["taglist"] = tags
        save_data["JF7"]["num_tags"] = num_tags
    
    # save to small data file
    smd.save(save_data)

    return
    

if __name__ == '__main__':
    main('0019_droplets_10um_2mm', num_shots=10)

