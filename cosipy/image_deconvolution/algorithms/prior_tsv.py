import healpy as hp
import numpy as np

from .prior_base import PriorBase

from ..models.allskyimage import AllSkyImageModel

class PriorTSV(PriorBase):
    """
    Total Squared Variation (TSV) prior for all-sky image models.

    This prior implements a smoothness constraint by penalizing squared
    differences between neighboring pixels.

    Parameters
    ----------
    parameter : dict
        Parameters for the TSV prior.
    model : AllSkyImageModel
        All-sky image model to which the prior will be applied.

    Attributes
    ----------
    usable_model_classes : list
        List containing AllSkyImageModel as the only compatible model class.
    neighbour_pixel_index : numpy.ndarray
        Array of shape (-1, npix) containing indices of neighboring pixels.
        For the case of HealPix, some pixels have only 7 neighboring pixels. 
        In this case, healpy returns -1 as the index of a neighboring pixel, 
        but it can cause calculation errors in this code. So, such a pixel 
        index is replaced with its own pixel index.
    num_neighbour_pixels : numpy.ndarray
        Array of shape (npix,) containing the number of valid neighbors for each pixel.

    Notes
    -----
    **Mathematical definition**

    The log TSV prior is defined as:

    .. math::

        \\log P_{\\mathrm{TSV}}(\\lambda) =
        -c_{\\mathrm{TSV}} \\sum_{i} \\sum_{j \\in \\sigma(i)} (\\lambda_i - \\lambda_j)^2

    where :math:`\\lambda_i` is the flux in pixel :math:`i`,
    :math:`\\sigma(i)` is the set of neighboring pixels of pixel :math:`i`,
    and :math:`c_{\mathrm{TSV}}` is the regularization coefficient (``coefficient`` in YAML).

    The gradient with respect to :math:`\\lambda_i` is:

    .. math::

        \\frac{\\partial \\log P_{\\mathrm{TSV}}}{\\partial \\lambda_i} =
        -4c_{\\mathrm{TSV}} \\sum_{j \\in \\sigma(i)} (\\lambda_i - \\lambda_j)

    **YAML parameter block**

    .. code-block:: yaml

        prior:
            TSV:
                coefficient: 1.0e+6   # regularization strength lambda (dimensionless)
    """

    usable_model_classes = [AllSkyImageModel]

    def __init__(self, parameter, model):

        super().__init__(parameter, model)

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
            Gradient of the log prior, in units inverse to the model.
        """

        if self.model_class == AllSkyImageModel:

            diff = (model[:] - model[self.neighbour_pixel_index]).value

            return -1.0 * self.coefficient * 4 * np.sum(diff, axis = 0) / model.unit
