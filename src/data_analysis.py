from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from escape.parse import swissfel
import h5py
from jungfrau_utils import apply_gain_pede, apply_geometry
h5py.enable_ipython_completer()
import time

sys.path.insert(0, '../src/')
from integrators import *

def process_run(run):
    '''
    Script that processes a given run by doing the following:
    * loads files from raw data (.json)
    * applies corrections (gain, pedestals, geometry)
    * calculates average 2D image
    * angular integrator
    '''

    # load data
    path = '/sf/bernina/data/p17743/res/scan_info/run%s.json'%run
    data = swissfel.parseScanEco_v01(path,createEscArrays=True,memlimit_mD_MB=50)
    jf7 = data['JF07T32V01'] # JungFrau data
    num_shots = jf7.data.shape[jf7.eventDim]
 
    # load corrections
    gains,pede,noise,mask = load_corrections()

    # apply corrections and geometry
    icorr = apply_gain_pede(jf7.data[0].compute(),G=gains, P=pede, pixel_mask=mask)
    icorr_geom = apply_geometry(icorr,'JF07T32V01')

    # initialise for angular integration
    rad_dist = radial_distances(icorr_sum)
    r, iq = angular_average(icorr_sum, rad=rad_dist) # memory error with mask, why?
    iqs = np.zeros((num_shots, iq.shape[0]))
    iqs[0] = iq

    # loop over all shots
    for i_shot in range(1,10):#num_shots):
        t1 = time.time()
        icorr = apply_gain_pede(jf7.data[i_shot].compute(),G=gains, P=pede, pixel_mask=mask)
        icorr_geom += apply_geometry(icorr,'JF07T32V01')
        r, iq = angular_average(icorr_geom, rad=rad_dist) # memory error with mask, why?
        iqs[i_shot] = iq
        icorr_sum += icorr_geom

        print('%.1f Hz'%(1.0/(time.time() - t1)))

    save_data = np.array([icorr_geom, num_shots,r,iqs])
    save_path = '/sf/bernina/data/p17743/scratch/hdf5/run%s.h5'%run
    print(save_path)
    save_h5(save_path,save_data)

    return



def load_corrections():
    '''
    Loads the corrections from a predefined path
    '''

    with h5py.File('/sf/bernina/config/jungfrau/gainMaps/JF07T32V01/gains.h5','r') as f:
        gains = f['gains'].value
    with h5py.File('/sf/bernina/data/p17743/res/JF_pedestals/pedestal_20190115_1551.JF07T32V01.res.h5','r') as f:
        pede = f['gains'].value
        noise = f['gainsRMS'].value
        mask = f['pixel_mask'].value
    return gains,pede,noise,mask


def save_h5(save_path,save_data):
    '''
    Saves the processed data in h5.
    '''

    avg_img_2d,num_shots,r,iqs = save_data

    h5f = h5py.File(save_path, 'w')

    IMG_2D = h5f.create_group("JF7")
    IMG_2D.create_dataset("2D_img", data = avg_img_2d, dtype = 'f')
    IMG_2D.create_dataset("num_shots", data = num_shots, dtype = 'i')
    IMG_2D.create_dataset("Q_bins", data = r, dtype = 'f')
    IMG_2D.create_dataset("I_Q", data = iqs, dtype = 'f')

    h5f.close()
    return
