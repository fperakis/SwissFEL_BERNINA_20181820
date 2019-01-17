import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import h5py
from os import walk

def discover_files(path):
    '''
    Looks in the given directory and returns the filenames,
    '''
    for (dirpath, dirnames, filenames) in walk(path):
        break
    return dirnames,filenames

def get_XRD_pattern(h5file,thr = 0):
    '''
    Load the XAS data, applies a threshold (thr) and returns the integrated intensity
    '''
    XRD_image = h5file['/Laser/BaslerImage2'].value
    XRD_image[XRD_image<thr]=0
    XRD_sum = np.sum(XRD_image,axis=2)
    return XRD_sum

def get_FEL_Spectrum(h5file):
    '''
    Loads and returns the FEL spectrum and calibrated energy
    '''
    # constants
    h   = 4.135667662*10**(-18) #ev s
    c   = 299792458             #m/s

    # load stuff
    Intensity      = h5file['photon_diagnostics/Spectrometer/hor_spectrum']
    WavelenghtSpan = h5file['photon_diagnostics/Spectrometer/WavelengthSpan'].value
    Pixel2micron   = h5file['photon_diagnostics/Spectrometer/Pixel2micron'].value
    Wavelenght     = h5file['/photon_diagnostics/Spectrometer/Wavelength'].value

    # get energy axis (in eV)
    ind    = np.arange(0,1000)-500 # index array
    x      = Wavelenght + ind*Pixel2micron*WavelenghtSpan/1000
    Energy = h*c/x*1e12

    return Intensity,Energy


def get_i0(h5file, offset_range = [240,290]):
    '''
    Loads and corrects the FEL spectrum.
    Correction includes the subtraction of a backrgound estimated over the 
    given "offset_range".
    Returns the i0 of each shot and the FEL spectrum averaged over shots
    '''
    x1,x2 = offset_range
    FEL_intensity,Energy = get_FEL_Spectrum(h5file)

    # subtract background in FEL spectrum (offset)
    FEL_intensity = np.array(FEL_intensity,dtype=float) # change type to float
    n_shots = FEL_intensity.shape[0]
    for j in range(int(n_shots)):
        offset = np.average(FEL_intensity[j,x1:x2])
        FEL_intensity[j,:] -= float(offset)
    FEL_intensity[FEL_intensity<0]=0

    # average FEL spectrum (over all shots)
    Spectrum = np.average(FEL_intensity,axis=0)

    # calculate i0 (integrate FEL spectrum of each shot)
    i0 = np.sum(FEL_intensity,axis=1)

    return i0,Energy, Spectrum


def save_data_h5(filename,data):
    '''
    Saves the processed data in h5.
    '''

    XAS_int,thr,XES_spectrum,i0,Energy,Spectrum,laser_int,dt = data

    h5f = h5py.File(filename, 'w')

    n_shots = len(XAS_int)
    XAS= h5f.create_group("XAS")
    XAS.create_dataset("intensity",  data = XAS_int,    dtype='i')
    XAS.create_dataset("threshold",  data = thr,        dtype='i')

    FEL= h5f.create_group("FEL")
    FEL.create_dataset("i0",         data= i0,          dtype='f')
    FEL.create_dataset("energy",     data=Energy,       dtype='f')
    FEL.create_dataset("spectrum",   data = Spectrum,   dtype='f')

    LASER = h5f.create_group("LASER")
    LASER.create_dataset("intensity", data = laser_int, dtype='f')
    LASER.create_dataset("delay",     data = dt,        dtype='f')

    h5f.close()
    return


