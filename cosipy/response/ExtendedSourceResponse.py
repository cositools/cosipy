import numpy as np

import astropy.units as u

from astromodels import Function3D

from histpy import Histogram

from .functions import get_integrated_extended_model
from .functions_3d import get_integrated_extended_model_3d

class ExtendedSourceResponse(Histogram):
    """
    A class to represent and manipulate extended source response data.

    This class provides methods to load data from HDF5 files, access contents,
    units, and axes information, and calculate expectations based on sky models.

    Methods
    -------
    get_expectation(allsky_image_model)
        Calculate expectation based on an all-sky image model.
    get_expectation_from_astromodel(source)
        Calculate expectation from an astronomical model source.

    Notes
    -----
    Currently, the axes of the response must be ['NuLambda', 'Ei', 'Em', 'Phi', 'PsiChi'].
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize an ExtendedSourceResponse object.
        """

        kwargs['track_overflow'] = False
        kwargs['sparse'] = False

        super().__init__(*args, **kwargs)

        self.post_init()

    def post_init(self):
        """
        Do init operations specific to our subclass
        """
        if not tuple(self.axes.labels) == ('NuLambda', 'Ei', 'Em', 'Phi', 'PsiChi'):
            # 'NuLambda' should be 'lb' if it is in the gal. coordinates?
            raise ValueError(f"The input axes {self.axes.labels} is not supported by ExtendedSourceResponse class.")

        # unit required for get_expectation() input
        self._exp_unit = (u.s * u.cm**2 * u.sr)**(-1)

    @classmethod
    def _open(cls, name='hist'):
        """
        Load response from an HDF5 group.

        Parameters
        ----------
        name : str, optional
            The name of the histogram group (default is 'hist').

        Returns
        -------
        ExtendedSourceResponse
            A new instance of ExtendedSourceResponse with loaded data.

        Raises
        ------
        ValueError
            If the shape of the contents does not match the axes.
        """

        resp = super()._open(name)

        resp.track_overflow(False)

        if resp.is_sparse:
            resp = resp.to_dense()

        resp.post_init()

        return resp

    def get_expectation(self, allsky_image_model):
        """
        Calculate expectation based on an all-sky image model.

        Parameters
        ----------
        allsky_image_model : Histogram
            The all-sky image model to use for calculation.

        Returns
        -------
        Histogram
            A histogram representing the calculated expectation.
        """

        if allsky_image_model.axes[:2] != self.axes[:2] or \
           allsky_image_model.unit != self._exp_unit:
            raise ValueError(f"The input allskymodel mismatches with the extended source response.")

        contents = np.tensordot(allsky_image_model.contents, self.contents, axes=((0,1), (0,1)))
        contents *= self.axes[0].pixarea()

        return Histogram(edges=self.axes[2:], contents=contents, copy_contents=False)

    def get_expectation_from_astromodel(self, source):
        """
        Calculate expectation from an astromodels extended source model.

        This method creates an AllSkyImageModel based on the current axes configuration,
        sets its values from the provided astromodels extended source model, and then
        calculates the expectation using the get_expectation method.

        Parameters
        ----------
        source : astromodels.ExtendedSource
            An astromodels extended source model object. This model represents
            the spatial and spectral distribution of an extended astronomical source.

        Returns
        -------
        Histogram
            A histogram representing the calculated expectation based on the
            provided extended source model.
        """

        if isinstance(source.spatial_shape, Function3D):
            allsky_image_model = get_integrated_extended_model_3d(source,
                                                                  image_axis = self.axes[0],
                                                                  energy_axis = self.axes[1])
        else:
            allsky_image_model = get_integrated_extended_model(source,
                                                               image_axis = self.axes[0],
                                                               energy_axis = self.axes[1])

        return self.get_expectation(allsky_image_model)
