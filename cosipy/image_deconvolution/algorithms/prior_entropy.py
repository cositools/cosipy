import astropy.units as u
import healpy as hp
import numpy as np

from .prior_base import PriorBase

from ..models.allskyimage import AllSkyImageModel

class PriorEntropy(PriorBase):
    """
    Maximum Entropy prior for all-sky image models.

    This prior encodes a preference for solutions that are as smooth and
    featureless as possible relative to a reference map, penalizing
    deviation of the reconstructed image from the reference.

    Parameters
    ----------
    parameter : dict
        Parameters for the entropy prior.
    model : AllSkyImageModel
        All-sky image model to which the prior will be applied.

    Attributes
    ----------
    reference_map : astropy.units.Quantity
        The reference (default) image :math:`m` against which entropy is measured.
        Typically set to a flat map representing the prior expectation of the image.

    Notes
    -----
    **Mathematical definition**

    The log maximum entropy prior is defined as:

    .. math::

        \\log P_{\\mathrm{ME}}(\\lambda) =
        c_{\\mathrm{ME}} \\sum_{i} \\lambda_i \\left(1 - \\log\\frac{\\lambda_i}{m_i}\\right)

    where :math:`\lambda_i` is the flux in pixel :math:`i`,
    :math:`m_i` is the reference map value in pixel :math:`i`,
    and :math:`c_{\mathrm{ME}}` is the regularization coefficient.

    The gradient with respect to :math:`\lambda_i` is:

    .. math::

        \\frac{\\partial \\log P_{\\mathrm{ME}}}{\\partial \\lambda_i} =
        -c_{\\mathrm{ME}} \\log\\frac{\\lambda_i}{m_i}

    **YAML parameter block**

    .. code-block:: yaml

        prior:
            entropy:
                coefficient: 1.0          # regularization strength lambda (dimensionless)
                reference_map:
                    value: 1.0e-4         # reference map value (uniform map)
                    unit: "cm-2 s-1 sr-1"
    """

    usable_model_classes = [AllSkyImageModel]

    def __init__(self, parameter, model):

        super().__init__(parameter, model)

        self.reference_map = self.parameter['reference_map']['value'] * u.Unit(self.parameter['reference_map']['unit'])

    def log_prior(self, model):
        """
        Calculate the logarithm of the entropy prior probability.

        Parameters
        ----------
        model : AllSkyImageModel
            Model for which to calculate the log prior.

        Returns
        -------
        float
            The logarithm of the entropy prior probability.
        """

        if self.model_class == AllSkyImageModel:

            image_ratio = (model/self.reference_map).contents.to('').value
            
            return self.coefficient * np.sum(model.contents.value * (1 - np.log(image_ratio)))
    
    def grad_log_prior(self, model): 
        """
        Calculate the gradient of the log entropy prior.

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

            image_ratio = (model/self.reference_map).contents.to('').value

            return -1 * self.coefficient * np.log(image_ratio) / model.unit
