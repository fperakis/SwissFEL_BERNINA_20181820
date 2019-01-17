import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

def line(x,a,b):
    return a*x+b

def gaussian(x,a,b,c,d):
    # in this form the c is the FWHM
    return np.abs(a)*np.exp(-4*np.log(2)*(x-b)**2./(c**2))+d

def exponential(x,a,b,c):
    return np.abs(a)*np.exp(-x/(np.abs(b)))+c

def fit(function,x,y,p0=None,sigma=None,bounds=None):
    '''
    fits a function and return the fit resulting parameters and curve
    '''
    popt,pcov = curve_fit(function,x,y,p0=p0,sigma=sigma)
    #x = np.arange(0,1e4)
    curve = function(x,*popt)
    perr = np.sqrt(np.diag(pcov))
    return popt,x,curve,perr

