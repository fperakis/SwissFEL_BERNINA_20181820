from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from escape.parse import swissfel
import h5py
from jungfrau_utils import apply_gain_pede, apply_geometry
h5py.enable_ipython_completer()
import time

def process_run(run):
    '''
    Script that processes a given run by doing the following:
    * loads files from raw data (.json)
    * applies corrections (gain, pedestals, geometry)
    * calculates average 2D image
    * angular integrator (not yet implemented)
    '''

    # load data
    path = '/sf/bernina/data/p17743/res/scan_info/run%s.json'%run
    data = swissfel.parseScanEco_v01(path,createEscArrays=True,memlimit_mD_MB=50)
    jf7 = data['JF07T32V01'] # JungFrau data
    num_shots = jf7.data.shape[jf7.eventDim]
 
    # load corrections
    pede,noise,mask = load_corrections()

    # apply corrections and geometry
    icorr = apply_gain_pede(jf7.data[0].compute(),G=gains, P=pede, pixel_mask=mask)
    icorr_geom = apply_geometry(icorr,'JF07T32V01')
    for i_shot in range(num_shots):
        t1 = time.time()
        icorr = apply_gain_pede(jf7.data[i_shot].compute(),G=gains, P=pede, pixel_mask=mask)
        icorr_geom += apply_geometry(icorr,'JF07T32V01')
        print('%.1f Hz'%(1.0/(time.time() - t1)))

    save_data = np.array([icorr_geom, num_shots])
    save_path = '/sf/bernina/data/p17743/res/scratch/hdf5/run%.h5'
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
    return pede,noise,mask


def save_h5(save_path,save_data):
    '''
    Saves the processed data in h5.
    '''

    avg_img_2d,num_shots = save_data

    h5f = h5py.File(save_path, 'w')

    IMG_2D = h5f.create_group("Average_2D_Image")
    IMG_2D.create_dataset("2D_img", data = avg_img_2d, dtype = i)
    IMG_2D.create_dataset("num_shots", data = num_shots, dtype = i)

    h5f.close()

