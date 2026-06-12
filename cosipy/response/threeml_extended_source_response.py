import copy

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
                 data:EmCDSBinnedData,
                 precomputed_psr: ExtendedSourceResponse,
                 polarization_axis:PolarizationAxis = None,
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
        #self._last_convolved_source_skycoord = None
        self._esr = None

    def copy(self) -> "BinnedThreeMlPointSourceResponse":
        """
        Safe copy to use for multiple sources
        Returns
        -------
        A copy than can be used safely to convolve another source
        """
        new = copy.copy(self)
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

    def expectation(self, copy = True)-> Histogram:
        # TODO: check coordsys from axis
        # TODO: Earth occ always true in this case

        if self._data is None or  self._source is None:
            raise RuntimeError("Call set_source() first.")

        # See this issue for the caveats of comparing models
        # https://github.com/threeML/threeML/issues/645
        source_dict = self._source.to_dict()

        #coord = self._source.position.sky_coord

        # Use cached expectation if nothing has changed
        if self._expectation is not None and self._last_convolved_source_dict == source_dict :
            if copy:
                return self._expectation.copy()
            else:
                return self._expectation

        # Expectation calculation
        # For ExtendedSource response, the psr has been already
        # computed for each position in the sky so we just need to
        # compute the expectation.  Check if the source position
        # change, since these operations are expensive
        if self._esr is None :
            logger.info("... Reading Extended source response ...")
            self._esr = self._response
            logger.info(f"--> done (source name : {self._source.name})")

        # Convolve with spectrum
        self._expectation = self._esr.get_expectation_from_astromodel(self._source)

        # Check if axes match
        if self._data.axes != self._expectation.axes:
            raise ValueError("Currently, the expectation axes must exactly match the detector response measurement axes")

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
