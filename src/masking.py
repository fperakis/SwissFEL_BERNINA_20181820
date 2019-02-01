import numpy as np
import matplotlib.pyplot as plt
import h5py
import time
import sys
import os

def mask_double_pixels(mask):
    '''
    script that masks the double pixels in the raw (un-reconstructed) mask
    '''
    nx = 256
    ny = 512
    on = 10 #from dmitri
    n_modules = 32

    # mask borders
    for i in range(4):
        mask[:,i] = on
        for j in range(4):
            mask[:,-(i+j)] = on

    # vertical stripes
    for i in range(4):
        mask[:,i*nx] = on
        for j in range(1,4):
            mask[:,(nx-j)+i*nx] = on

    # horizontal stripes
    for i in range(n_modules):
        mask[ny*i, :] = on
        mask[ny*i + nx, :] = on
        for j in range(4):
            mask[ny*i + (nx-j), :] = on
            mask[ny*i + (ny-(j+1)), :] = on

    return mask
