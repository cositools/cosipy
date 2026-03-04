import astropy.units as u
import healpy as hp
import numpy as np

from .prior_base import PriorBase

from .allskyimage import AllSkyImageModel

class PriorEntropy(PriorBase):
    """
    Entropy prior for all-sky image models.

    Parameters
    ----------
    parameter : dict
        Parameters for the entropy prior.
    model : AllSkyImageModel
        All-sky image model to which the prior will be applied.
    """

    usable_model_classes = [AllSkyImageModel]

    def __init__(self, parameter, model):

        super().__init__(parameter, model)

        self.prior_map = self.parameter['prior_map']['value'] * u.Unit(self.parameter['prior_map']['unit'])

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

            image_ratio = (model/self.prior_map).contents.to('').value
            
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

            image_ratio = (model/self.prior_map).contents.to('').value

            return -1 * self.coefficient * np.log(image_ratio) / model.unit
