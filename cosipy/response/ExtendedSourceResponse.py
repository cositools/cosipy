import numpy as np

import astropy.units as u

from astromodels import Function3D

from histpy import Histogram

from .functions import (
    get_integrated_extended_model,
    get_integrated_spectral_model,
)
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
    get_spatial_response_from_astromodel(source)
        Convolve a 2D source morphology with the response while retaining Ei.
    get_expectation_from_spatial_response(source, spatial_response)
        Fold a spectrum through a precomputed spatial response.
    get_galprop_unit_expectation(source)
        Calculate the detector expectation for a GalpropHealpixModel with K=1.

    Notes
    -----
    Currently, the axes of the response must be
    ['NuLambda', 'Ei', 'Em', 'Phi', 'PsiChi'].
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
            raise ValueError(
                f"The input axes {self.axes.labels} is not supported by "
                "ExtendedSourceResponse class."
            )

        # unit required for get_expectation() input
        self._exp_unit = (u.s * u.cm**2 * u.sr)**(-1)

    @classmethod
    def _open(cls, name='hist', **kwargs):
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

        resp = super()._open(name, **kwargs)

        resp.track_overflow(False)

        resp = resp.to_dense(copy=False)

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
            raise ValueError(
                "The input allskymodel mismatches with the extended source response."
            )

        contents = np.tensordot(
            allsky_image_model.contents,
            self.contents,
            axes=((0, 1), (0, 1)),
        )
        contents *= self.axes[0].pixarea()

        return Histogram(
            edges=self.axes[2:],
            contents=contents,
            copy_contents=False,
        )

    def get_spatial_response_from_astromodel(self, source):
        """
        Convolve a 2D spatial morphology with the extended-source response.

        The expensive sky contraction is performed while the true-energy axis
        is retained. For a fixed morphology this result can be cached and
        reused while spectral parameters change.

        Parameters
        ----------
        source : astromodels.ExtendedSource
            Extended source with an energy-independent (2D) spatial model.

        Returns
        -------
        Histogram
            Spatially convolved response with axes
            ``(Ei, Em, Phi, PsiChi)``.
        """

        if source.spatial_shape.n_dim != 2:
            raise ValueError(
                "Spatial-response caching is only supported for "
                "energy-independent (2D) spatial models."
            )

        image_axis = self.axes[0]

        l, b = image_axis.pix2ang(
            np.arange(image_axis.npix),
            lonlat=True,
        )

        # Keep exactly the same spatial normalization and unit behavior used by
        # get_integrated_extended_model().
        normalized_map = source.spatial_shape(l, b) / u.sr

        # Contract only the sky dimension:
        #
        #   (NuLambda) x
        #   (NuLambda, Ei, Em, Phi, PsiChi)
        #
        # -> (Ei, Em, Phi, PsiChi)
        contents = np.tensordot(
            normalized_map,
            self.contents,
            axes=(0, 0),
        )

        contents *= image_axis.pixarea()

        return Histogram(
            edges=self.axes[1:],
            contents=contents,
            copy_contents=False,
        )

    def get_expectation_from_spatial_response(self, source, spatial_response):
        """
        Fold a spectrum through an already spatially-convolved response.

        Parameters
        ----------
        source : astromodels.ExtendedSource
            Source whose spectral model should be folded through the response.
        spatial_response : Histogram
            Cached histogram produced by
            :meth:`get_spatial_response_from_astromodel`, with axes
            ``(Ei, Em, Phi, PsiChi)``.

        Returns
        -------
        Histogram
            Detector expectation with axes ``(Em, Phi, PsiChi)``.
        """

        if spatial_response.axes != self.axes[1:]:
            raise ValueError(
                "The spatial response axes do not match the extended-source "
                "response Ei/detector axes."
            )

        integrated_flux = get_integrated_spectral_model(
            spectrum=source.spectrum.main.shape,
            energy_axis=self.axes[1],
        )

        # Contract only Ei:
        #
        #   (Ei) x (Ei, Em, Phi, PsiChi)
        #
        # -> (Em, Phi, PsiChi)
        contents = np.tensordot(
            integrated_flux.contents,
            spatial_response.contents,
            axes=(0, 0),
        )

        return Histogram(
            edges=self.axes[2:],
            contents=contents,
            copy_contents=False,
        )

    def get_galprop_unit_expectation(self, source):
        """
        Calculate the detector expectation for a GalpropHealpixModel with K=1.

        ``GalpropHealpixModel`` is an energy-dependent spatial model and cannot
        use the 2D spatial-response factorization above.  Its only fit parameter,
        however, is an overall normalization K.  The full sky+energy response
        contraction can therefore be performed once at K=1 and rescaled during
        subsequent likelihood evaluations.

        Parameters
        ----------
        source : astromodels.ExtendedSource
            Extended source using ``GalpropHealpixModel``.

        Returns
        -------
        Histogram
            Detector expectation for unit GALPROP normalization.
        """

        from cosipy.threeml.custom_functions import GalpropHealpixModel

        if not isinstance(source.spatial_shape, GalpropHealpixModel):
            raise TypeError(
                "get_galprop_unit_expectation requires a GalpropHealpixModel."
            )

        galprop = source.spatial_shape
        original_norm = float(galprop.K.value)

        # functions_3d.py caches the integrated GALPROP cube on the model
        # instance.  Force one calculation for this response Ei axis so a stale
        # cube from a different response/slice cannot be reused inadvertently.
        galprop._result = None

        if original_norm == 0.0:
            galprop.K.value = 1.0

        try:
            get_integrated_extended_model_3d(
                source,
                image_axis=self.axes[0],
                energy_axis=self.axes[1],
            )

            unit_flux = np.array(
                galprop.intg_flux,
                dtype=float,
                copy=True,
            )
        finally:
            galprop.K.value = original_norm

        unit_flux_map = Histogram(
            (self.axes[0], self.axes[1]),
            contents=unit_flux,
            unit=self._exp_unit,
            copy_contents=False,
        )

        return self.get_expectation(unit_flux_map)

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
            allsky_image_model = get_integrated_extended_model_3d(
                source,
                image_axis=self.axes[0],
                energy_axis=self.axes[1],
            )
        else:
            allsky_image_model = get_integrated_extended_model(
                source,
                image_axis=self.axes[0],
                energy_axis=self.axes[1],
            )

        return self.get_expectation(allsky_image_model)
