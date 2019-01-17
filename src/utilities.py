import numpy as np
import scipy.fftpack

# analysis tools from scikit-beam (https://github.com/scikit-beam/scikit-beam/tree/master/skbeam/core)
import skbeam.core.roi as roi
import skbeam.core.utils as utils

def ring_mask(data,center,inner_radius=0,width=1,spacing=0,num_rings=10):
    '''
    Creates a ring mask with cocentric rings of given radius, width and spacing 
    center = (cx, cy)
    '''
    edges = roi.ring_edges(inner_radius, width, spacing, num_rings)
    ring_mask = roi.rings(edges, center, data.shape)

    return ring_mask

def find_local_minima(x,y):
    miny = np.r_[True, y[1:] < y[:-1]] & np.r_[y[:-1] < y[1:],True]
    min_pos = x[miny==True]
    return min_pos

def find_local_maxima(x,y):
    miny = np.r_[True, y[1:] > y[:-1]] & np.r_[y[:-1] > y[1:],True]
    max_pos = x[miny==True]
    return max_pos

def FT_low_pass_filter(x,y, cutoff):
    '''
    FT low pass filter
    '''
    N = len(x)
    w = scipy.fftpack.rfft(y)
    f = scipy.fftpack.rfftfreq(N, x[1]-x[0])
    spectrum = w**2

    cutoff_value = spectrum[1:].max()/cutoff
    cutoff_idx = spectrum < cutoff_value #(spectrum.max()/cutoff)
    w2 = w.copy()
    w2[cutoff_idx] = 0
    spectrum2 = w2**2
    y2 = scipy.fftpack.irfft(w2)

    return y2


def FT_high_pass_filter(x,y, cutoff):
    '''
    FT high pass filter
    '''
    N = len(x)
    w = scipy.fftpack.rfft(y)
    f = scipy.fftpack.rfftfreq(N, x[1]-x[0])
    spectrum = w**2

    cutoff_value = spectrum[1:].max()/cutoff
    cutoff_idx = spectrum > cutoff_value #(spectrum.max()/cutoff)
    w2 = w.copy()
    w2[cutoff_idx] = 0
    spectrum2 = w2**2
    y2 = scipy.fftpack.irfft(w2)

    return y2

