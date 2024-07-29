from abc import ABC, abstractmethod
import astropy.units as u
import numpy as np
import copy
from histpy import Histogram

class ModelBase(Histogram, ABC):
    """
    A base class of the model, i.e., a gamma-ray flux sky, and a gamma-ray distributions in a 3D space.

    Subclasses must override these methods. The `ImageDeconvolution` class will use them in the initialization process.

    Methods:
    - instantiate_from_parameters(cls, parameter)
    - set_values_from_parameters(self, parameter)
    """

    def __init__(self, edges, contents = None, sumw2 = None,
                 labels=None, axis_scale = None, sparse = None, unit = None):

        super().__init__(edges, contents = contents, sumw2 = sumw2, 
                         labels = labels, axis_scale = axis_scale, sparse = sparse, unit = unit)

    @classmethod
    @abstractmethod
    def instantiate_from_parameters(cls, parameter):
        """
        Return an instantiate of the class using given parameters.

        Parameters
        ----------
        parameter : py:class:`yayc.Configurator`
            Parameters for the specified algorithm.

        Returns
        -------
        py:class:`ModelBase`
        """

        raise NotImplementedError

    @abstractmethod
    def set_values_from_parameters(self, parameter):
        """
        Set values accordinng to the give parameters. 

        Parameters
        ----------
        parameter : py:class:`yayc.Configurator`
            Parameters for the specified algorithm.
        """

        raise NotImplementedError

    def mask_pixels(self, mask, fill_value = 0):
        """
        Mask pixels

        Parameters
        ----------
        mask: :py:class:`histpy.histogram.Histogram`
        fill_value: float or :py:class:`astropy.units.quantity.Quantity`
        """

        if not isinstance(fill_value, u.quantity.Quantity) and self.unit is not None:
            fill_value *= self.contents.unit

        model_new = copy.deepcopy(self)
        model_new[:] = np.where(mask.contents, model_new.contents, fill_value)

        return model_new
