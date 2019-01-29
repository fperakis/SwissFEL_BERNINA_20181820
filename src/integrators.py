import numpy as np

ELEMENTARY_CHARGE = 1.602176634E-19 # C
SPEED_OF_LIGHT = 299792458 # m/s
PLANCK_CONSTANT = 6.62607015E-34 # Js


class RadialAverager(object):

    def __init__(self, q_values, mask, n_bins=101):
        """
        Parameters
        ----------
        q_values : np.ndarray (float)
            For each pixel, this is the momentum transfer value of that pixel
        mask : np.ndarray (int)
            A boolean (int) saying if each pixel is masked or not
        n_bins : int
            The number of bins to employ. If `None` guesses a good value.
        """

        self.q_values = q_values
        self.mask = mask
        self.n_bins = n_bins

        # figure out the number of bins to use
        if n_bins != None:
            self.n_bins = n_bins
            self._bin_factor = float(self.n_bins-1) / self.q_values.max()
        else:
            self._bin_factor = 25.0
            self.n_bins = (self.q_values.max() * self._bin_factor) + 1

        self._bin_assignments = np.floor( q_values * self._bin_factor ).astype(np.int32)
        self._normalization_array = (np.bincount( self._bin_assignments.flatten(), weights=self.mask.flatten() ) \
                                    + 1e-100).astype(np.float)

        assert self.n_bins == self._bin_assignments.max() + 1
        self._normalization_array = self._normalization_array[:self.n_bins]

        return

    def __call__(self, image):
        """
        Bin pixel intensities by their momentum transfer.
        
        Parameters
        ----------            
        image : np.ndarray
            The intensity at each pixel, same shape as pixel_pos


        Returns
        -------
        bin_centers : ndarray, float
            The q center of each bin.

        bin_values : ndarray, int
            The average intensity in the bin.
        """

        if not (image.shape == self.q_values.shape):
            raise ValueError('`image` and `q_values` must have the same shape')
        if not (image.shape == self.mask.shape):
            raise ValueError('`image` and `mask` must have the same shape')

        weights = image.flatten() * self.mask.flatten()
        bin_values = np.bincount(self._bin_assignments.flatten(), weights=weights)
        bin_values /= self._normalization_array

        assert bin_values.shape[0] == self.n_bins

        return bin_values

    @property
    def bin_centers(self):
        return np.arange(self.n_bins) / self._bin_factor


def angular_average(image, mask=None, rad=None, center=None, threshold=None, nx=None,
                     pixel_size=None, photon_energy=None, detector_distance=None, min_x=None, max_x=None):
    """Azimuthal average of the image data
    The azimuthal average is also known as the radial profile

    Parameters
    ----------
    image : array
        Image to compute the average as a function of radius
    center : tuple
        The beam position in the image in pixel units
        argument order should be (row, col)
        defaults to image center
    mask  : arrayint, optional
        Boolean array with 1s (and 0s) to be used (or not) in the average
    threshold : int, optional
        Ignore counts above `threshold` [not implemented]
        default is off
    nx : int, optional
        number of bins in x
        defaults to 1 bin/pixel
    pixel_size : tuple, optional
        The size of a pixel (in SI units).
        argument order should be np.float  [(pixel_height, pixel_width) not implemented]
        default is 1
    min_x : float, optional number of pixels
        Left edge of first bin defaults to minimum value of x
    max_x : float, optional number of pixels
        Right edge of last bin defaults to maximum value of x
    
    Returns
    -------
    bin_centers : np.ndarray
        The center of each bin in R. shape is (nx, )
    rad_profile : np.ndarray
        Radial profile of the image. shape is (nx, ).
    """
    
    if rad is None:
        # compute the radii for image[y, x]
        x = np.arange(image.shape[1])
        y = np.arange(image.shape[0])
        xx, yy = np.meshgrid(x, y)
        if center is None:
            xx = xx.astype(np.float64) - (image.shape[1]-1)/2.
            yy = yy.astype(np.float64) - (image.shape[0]-1)/2.
        else:
            xx -= center[1]
            yy -= center[0]
        rad = np.sqrt(xx*xx + yy*yy)
        del x, y
        del xx, yy
    assert rad.shape == image.shape
    
    if (photon_energy is not None) and (detector_distance is not None) and (pixel_size is not None):
        # convert to momentum transfer in A-1
        rad = q_scale(rad, photon_energy=photon_energy, detector_distance=detector_distance, pixel_size=pixel_size)
        if nx is None:
            nx = 1000 # set scale so not 1 bin/A-1 is used
    
    # histogram the intensities and normalize by number of pixels in each bin to obtain average intensity
    if nx is None:
        nBins = np.arange(np.floor(rad.min()), np.ceil(rad.max())+1)
    else:
        nBins = np.int(nx)
    if mask is None:
        bin_values, bin_edges = np.histogram(rad, weights=image, bins=nBins) 
        bin_normalizations = np.histogram(rad, bins=bin_edges)
    else:
        bin_values, bin_edges = np.histogram(rad[mask], weights=image[mask], bins=nBins)
        bin_normalizations = np.histogram(rad[mask], bins=bin_edges)
    rad_profile = bin_values[np.where(bin_normalizations[0] > 0)]/bin_normalizations[0][np.where(bin_normalizations[0] > 0)]
    bin_centers = np.array([(bin_edges[i] + bin_edges[i+1])/2 for i in range(len(bin_values))])
    
    # run into 'High memory usage error', try to delete
    del rad
    del bin_values, bin_edges, bin_normalizations
    return bin_centers, rad_profile

def radial_distances(image, center=None):
    """ Calculates radial distances of the pixels in the image, can be used as input
    to angular_average to speed up algorithm.

    Parameters
    ----------
    image : array
        Image to compute the average as a function of radius
    center : tuple
        The beam position in the image in pixel units
        argument order should be (row, col)
        defaults to image center
    
    Returns
    -------
    rad : np.ndarray
        Radial distances of the pixels in image. shape is same as image (ny, nx).
    """
    
    # compute the radii for image[y, x]
    x = np.arange(image.shape[1])
    y = np.arange(image.shape[0])
    xx, yy = np.meshgrid(x, y)
    if center is None:
        xx = xx.astype(np.float64) - (image.shape[1]-1)/2.
        yy = yy.astype(np.float64) - (image.shape[0]-1)/2.
    else:
        xx -= center[1]
        yy -= center[0]
    rad = np.sqrt(xx*xx + yy*yy)
    return rad

def q_scale(pixel_distances, photon_energy=9500, detector_distance=0.140, pixel_size=75E-6):
    """
    Compute the momentum transfer (A-1) for a radial profile in pixels
    
    Parameters
    ----------
    pixelDistances : np.ndarray
        The pixel distances (center of each bin) from the beam center, in pixel units
    
    Returns
    -------
    q : np.ndarray
        The momentum transfer (A-1)
    """
    wavelength = PLANCK_CONSTANT*SPEED_OF_LIGHT/(photon_energy*ELEMENTARY_CHARGE)
    # convert to momentum transfer in A-1
    q = 2.*np.pi*2.*np.sin(0.5*np.arctan2(pixel_distances*pixel_size, detector_distance))/(wavelength*1E10)
    return q


