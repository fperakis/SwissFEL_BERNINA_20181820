import numpy as np

def iceIc_peaks():
    '''
    returns the ice Ic (cubic) peak positions at T = 88 K (in Angst-1) 
    taken from Nature 1960, 188, 1144)
    http://www.nature.com/nature/journal/v188/n4757/abs/1881144a0.html
    '''
    ice_2theta =np.array([24.26,40.11,47.43])
    xrd_energy = 8.05 #keV at Cu K-alpha (using a copper anode)
    h = 4.135667516*1e-18 #kev*sec
    c = 3*1e8 #m/s
    lambda_ = h*c/xrd_energy*1e10 #1.5498 Angst # wavelength of the X-rays

    
    # -- convert degrees to theta
    ice_q = 4.*np.pi*np.sin(ice_2theta/2.*np.pi/180.)/lambda_

    return ice_q

def iceIh_peaks():
    '''
    returns the ice Ih (hexagonal) peak positions at T = 88 K (in Angst-1) 
    taken from Nature 1960, 188, 1144)
    http://www.nature.com/nature/journal/v188/n4757/abs/1881144a0.html
    '''
    ice_2theta =np.array([22.82,24.26,25.89,33.55,40.09,43.70,46.62,47.41,48.34,53.24])
    xrd_energy = 8.05 #keV at Cu K-alpha (using a copper anode)
    h = 4.135667516*1e-18 #kev*sec
    c = 3*1e8 #m/s
    lambda_ = h*c/xrd_energy*1e10 #1.5498 Angst # wavelength of the X-rays

    
    # -- convert degrees to theta
    ice_q = 4.*np.pi*np.sin(ice_2theta/2.*np.pi/180.)/lambda_

    return ice_q


