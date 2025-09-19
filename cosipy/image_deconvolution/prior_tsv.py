import healpy as hp
import numpy as np

from .prior_base import PriorBase

from .allskyimage import AllSkyImageModel

class PriorTSV(PriorBase):
    """
    Total Squared Variation (TSV) prior for all-sky image models.

    This prior implements a smoothness constraint by penalizing differences between neighboring pixels.

    Parameters
    ----------
    coefficient : float
        Scaling coefficient for the TSV prior.
    model : AllSkyImageModel
        All-sky image model to which the prior will be applied.

    Attributes
    ----------
    usable_model_classes : list
        List containing AllSkyImageModel as the only compatible model class.
    neighbour_pixel_index : numpy.ndarray
        Array of shape (8, npix) containing indices of neighboring pixels.
        Some pixels have only 7 neighboring pixels. In this case, healpy returns -1
        as the index of a neighboring pixel, but it can cause calculation errors in
        this code. So, such a pixel index is replaced with its own pixel index.
    num_neighbour_pixels : numpy.ndarray
        Array of shape (npix,) containing the number of valid neighbors for each pixel.
    """

    usable_model_classes = [AllSkyImageModel]

    def __init__(self, coefficient, model):

        super().__init__(coefficient, model)

        if self.model_class == AllSkyImageModel:

            nside = model.axes['lb'].nside
            npix  = model.axes['lb'].npix
            nest = False if model.axes['lb'].scheme == 'RING' else True

            theta, phi = hp.pix2ang(nside = nside, ipix = np.arange(npix), nest = nest)

            self.neighbour_pixel_index = hp.get_all_neighbours(nside = nside, theta = theta, phi = phi, nest = nest) # Its shape is (8, num. of pixels)

            self.num_neighbour_pixels = np.sum(self.neighbour_pixel_index >= 0, axis = 0) # Its shape is (num. of pixels)
            
            # replace -1 with its pixel index
            for idx, ipixel in np.argwhere(self.neighbour_pixel_index == -1):
                self.neighbour_pixel_index[idx, ipixel] = ipixel

    def log_prior(self, model):
        """
        Calculate the logarithm of the TSV prior probability.

        Parameters
        ----------
        model : AllSkyImageModel
            Model for which to calculate the log prior.

        Returns
        -------
        float
            The logarithm of the TSV prior probability.
        """
        if self.model_class == AllSkyImageModel:

            diff = (model[:] - model[self.neighbour_pixel_index]).value
            # Its shape is (8, num. of pixels, num. of energies)

            return -1.0 * self.coefficient * np.sum(diff**2)
    
    def grad_log_prior(self, model): 
        """
        Calculate the gradient of the log TSV prior.

        Parameters
        ----------
        model : AllSkyImageModel
            Model for which to calculate the gradient.

        Returns
        -------
        numpy.ndarray
            Gradient of the log prior, with the same units as the model.
        """
        if self.model_class == AllSkyImageModel:

            diff = (model[:] - model[self.neighbour_pixel_index]).value

            return -1.0 * self.coefficient * 4 * np.sum(diff, axis = 0) / model.unit
