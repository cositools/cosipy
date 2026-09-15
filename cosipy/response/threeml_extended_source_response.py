from copy import copy as shallow_copy, deepcopy

from astromodels.sources import Source, ExtendedSource

from histpy import Axes, Histogram

from cosipy.data_io import EmCDSBinnedData

from cosipy.polarization.polarization_axis import PolarizationAxis
from cosipy.threeml.util import to_linear_polarization

from cosipy.response import ExtendedSourceResponse

from cosipy.interfaces import BinnedThreeMLSourceResponseInterface

import logging
logger = logging.getLogger(__name__)


__all__ = ["BinnedThreeMLExtendedSourceResponse"]


class BinnedThreeMLExtendedSourceResponse(BinnedThreeMLSourceResponseInterface):
    """
    COSI 3ML plugin.

    Parameters
    ----------
    dr:
        Extended source response handle, or the file path
    """

    def __init__(self,
                 data: EmCDSBinnedData,
                 precomputed_psr: ExtendedSourceResponse,
                 polarization_axis: PolarizationAxis = None,
                 ):
        """
        Parameters
        ----------
        precomputed_psr:
            Precomputed point source response for all pixel, a.k.a
            ExtendedSourceResponse.
        polarization_axis:
            The desired effective binning of the photon polarization
            angle (aka Pol).  This also defined the polarization
            coordinate system and convention.

        """

        # TODO: FullDetectorResponse -> BinnedInstrumentResponseInterface

        # Interface inputs
        self._source = None

        # Other implementation inputs
        self._data = data

        self._response = precomputed_psr
        self._polarization_axis = polarization_axis

        # Cache
        # Prevent unnecessary calculations and new memory allocations

        # See this issue for the caveats of comparing models
        # https://github.com/threeML/threeML/issues/645
        self._last_convolved_source_dict = None

        self._expectation = None

        # The ESR changes for each morphology, but for an energy-independent
        # (2D) morphology its expensive sky convolution is independent of all
        # spectral parameters.  Cache that intermediate response separately.
        self._last_spatial_model_dict = None
        self._spatial_response = None

        # GalpropHealpixModel is a fixed sky+energy template with an overall
        # normalization K.  Cache its complete detector expectation at K=1.
        self._last_galprop_template_key = None
        self._galprop_unit_expectation = None

        # The PSR change for each direction, but it's the same for all
        # spectrum parameters

        # Source location cached separately since changing the response
        # for a given direction is expensive
        #self._last_convolved_source_skycoord = None
        self._esr = None

    @property
    def axes(self) -> Axes:
        return self._data.axes

    def clear_cache(self):

        self._last_convolved_source_dict = None
        self._expectation = None

        self._last_spatial_model_dict = None
        self._spatial_response = None

        self._last_galprop_template_key = None
        self._galprop_unit_expectation = None

        #self._last_convolved_source_skycoord = None
        self._esr = None

    def copy(self) -> "BinnedThreeMLExtendedSourceResponse":
        """
        Safe copy to use for multiple sources
        Returns
        -------
        A copy than can be used safely to convolve another source
        """
        new = shallow_copy(self)
        new.clear_cache()
        return new

    def set_source(self, source: Source):
        """
        The source is passed as a reference and its parameters can
        change. Remember to check if it changed since the last time
        the user called expectation.

        """
        if not isinstance(source, ExtendedSource):
            raise TypeError("I only know how to handle extended sources!")

        polarization = to_linear_polarization(source.spectrum.main.polarization)

        if (polarization.degree.value != 0 and
                self._polarization_axis is None):
            raise RuntimeError("This response can't handle a polarized source.")

        self._source = source

    def expectation(self, copy = True) -> Histogram:
        # TODO: check coordsys from axis
        # TODO: Earth occ always true in this case

        if self._data is None or self._source is None:
            raise RuntimeError("Call set_source() first.")

        # Import locally to avoid adding a package-level dependency/circular
        # import during response module initialization.
        from cosipy.threeml.custom_functions import GalpropHealpixModel

        is_galprop = isinstance(self._source.spatial_shape, GalpropHealpixModel)

        # See this issue for the caveats of comparing models
        # https://github.com/threeML/threeML/issues/645
        source_dict = self._source.to_dict()

        # GalpropHealpixModel.to_dict() currently stores the file and frame but
        # not the GALPROP file-version selector. Include it in the local cache
        # key so set_version() invalidates the expectation cache correctly.
        if is_galprop:
            source_dict = deepcopy(source_dict)
            source_dict["_cosipy_galprop_version"] = self._source.spatial_shape._gal_version

        #coord = self._source.position.sky_coord

        # Use cached expectation if nothing has changed
        if self._expectation is not None and self._last_convolved_source_dict == source_dict:
            if copy:
                return self._expectation.copy()
            else:
                return self._expectation

        # Expectation calculation
        # For ExtendedSource response, the psr has been already
        # computed for each position in the sky so we just need to
        # compute the expectation.  Check if the source position
        # change, since these operations are expensive
        if self._esr is None:
            logger.info("... Reading Extended source response ...")
            self._esr = self._response
            logger.info(f"--> done (source name : {self._source.name})")

        # --------------------------------------------------------------
        # Compute source expectation
        # --------------------------------------------------------------

        # CASE 1: GALPROP is a fixed sky+energy template whose only fit
        # parameter is the overall normalization K.  Cache the complete
        # detector expectation for K=1.
        if is_galprop:

            galprop = self._source.spatial_shape

            energy_edges = self._esr.axes[1].edges
            energy_edge_values = getattr(energy_edges, "value", energy_edges)

            template_key = (
                str(galprop._fitsfile),
                galprop._frame,
                galprop._gal_version,
                tuple(energy_edge_values),
            )

            if (
                self._galprop_unit_expectation is None
                or self._last_galprop_template_key != template_key
            ):

                logger.info(
                    "... Convolving GALPROP template with extended-source response ..."
                )

                self._galprop_unit_expectation = (
                    self._esr.get_galprop_unit_expectation(self._source)
                )
                self._last_galprop_template_key = template_key

                logger.info("--> GALPROP response convolution done")

            norm = float(galprop.K.value)

            # Preserve axes and units exactly while rescaling the cached
            # unit-normalization detector expectation.
            self._expectation = self._galprop_unit_expectation.copy()
            self._expectation *= norm

        # CASE 2: an ordinary energy-independent spatial morphology can be
        # separated into morphology(sky) x spectrum(E).  Cache only the sky
        # convolution and refold the changing spectrum over Ei.
        elif self._source.spatial_shape.n_dim == 2:

            spatial_model_dict = self._source.spatial_shape.to_dict()

            if (
                self._spatial_response is None
                or self._last_spatial_model_dict != spatial_model_dict
            ):

                logger.info(
                    "... Convolving spatial morphology with "
                    "extended-source response ..."
                )

                self._spatial_response = (
                    self._esr.get_spatial_response_from_astromodel(self._source)
                )
                self._last_spatial_model_dict = deepcopy(spatial_model_dict)

                logger.info("--> spatial response convolution done")

            self._expectation = (
                self._esr.get_expectation_from_spatial_response(
                    self._source,
                    self._spatial_response,
                )
            )

        # CASE 3: generic energy-dependent spatial models cannot in general
        # be factorized.  Preserve the existing COSIPy behavior.
        else:

            self._expectation = self._esr.get_expectation_from_astromodel(
                self._source
            )

        # Check if axes match
        if self._data.axes != self._expectation.axes:
            raise ValueError(
                "Currently, the expectation axes must exactly match the detector "
                "response measurement axes"
            )

        # Cache. Use dict and copy since the internal variables can change
        # See this issue for the caveats of comparing models
        # https://github.com/threeML/threeML/issues/645
        self._last_convolved_source_dict = source_dict
        #self._last_convolved_source_skycoord = coord.copy()

        # Copy to prevent user to modify our cache
        if copy:
            return self._expectation.copy()
        else:
            return self._expectation
