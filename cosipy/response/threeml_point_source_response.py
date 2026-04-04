import logging
from pathlib import Path
from typing import Union

import histpy
from mhealpy import HealpixBase

from cosipy.data_io import EmCDSBinnedData
from cosipy.interfaces.instrument_response_interface import BinnedInstrumentResponseInterface
from cosipy.polarization.polarization_axis import PolarizationAxis
from cosipy.threeml.util import to_linear_polarization

logger = logging.getLogger(__name__)

import copy

from astromodels.sources import Source, PointSource
from scoords import SpacecraftFrame
from histpy import Axes, Histogram, Axis, HealpixAxis
from cosipy.interfaces import BinnedThreeMLSourceResponseInterface, BinnedDataInterface, DataInterface

from cosipy.response import FullDetectorResponse, PointSourceResponse
from cosipy.spacecraftfile import SpacecraftHistory, SpacecraftAttitudeMap

from mhealpy import HealpixMap

__all__ = ["BinnedThreeMLPointSourceResponse"]

class BinnedThreeMLPointSourceResponse(BinnedThreeMLSourceResponseInterface):
    """
    COSI 3ML plugin.

    Parameters
    ----------
    dr:
        Full detector response handle, or the file path
    sc_history:
        Contains the information of the orientation: timestamps (astropy.Time) and attitudes (scoord.Attitude) that describe
        the spacecraft for the duration of the data included in the analysis
    """

    def __init__(self,
                 data:EmCDSBinnedData,
                 instrument_response: BinnedInstrumentResponseInterface,
                 sc_history: SpacecraftHistory,
                 energy_axis:Axis,
                 polarization_axis:PolarizationAxis = None,
                 nside = None
                 ):
        """

        Parameters
        ----------
        instrument_response:
            A BinnedInstrumentResponseInterface capable of providing the differential
            effective area in local coordinates as a function of direction, energy and
            polarization.
        sc_history:
            The SpacecraftHistory describing the SC orbit and attitude vs time.
        energy_axis:
            The desired effective binning of the photon energy (aka Ei)
        polarization_axis:
            The desired effective binning of the photon polarization angle (aka Pol).
            This also defined the polarization coordinate system and convention.
        nside:
            - If transformation from local to an inertial system is needed, the spacecraft
            attitude will be first discretized based on this nside.
            - If local, this is the nside of the dwell time map
        """

        # TODO: FullDetectorResponse -> BinnedInstrumentResponseInterface


        # Interface inputs
        self._source = None

        # Other implementation inputs
        self._data = data

        self._sc_ori = sc_history
        self._response = instrument_response
        self._energy_axis = energy_axis
        self._polarization_axis = polarization_axis
        self._nside = nside

        # Cache
        # Prevent unnecessary calculations and new memory allocations

        # See this issue for the caveats of comparing models
        # https://github.com/threeML/threeML/issues/645
        self._last_convolved_source_dict = None

        self._expectation = None

        # The PSR change for each direction, but it's the same for all spectrum parameters

        # Source location cached separately since changing the response
        # for a given direction is expensive
        self._last_convolved_source_skycoord = None

        self._psr = None

    @property
    def axes(self) -> histpy.Axes:
        return self._data.axes

    def clear_cache(self):

        self._last_convolved_source_dict = None
        self._expectation = None
        self._last_convolved_source_skycoord = None
        self._psr = None

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
        The source is passed as a reference and it's parameters
        can change. Remember to check if it changed since the
        last time the user called expectation.
        """
        if not isinstance(source, PointSource):
            raise TypeError("I only know how to handle point sources!")

        if not hasattr(source.spectrum, 'main'):

            for item in source.spectrum.to_dict():

                polarization = to_linear_polarization(source.components[item].polarization)

                if polarization.degree.value != 0 and self._polarization_axis is None:
                    raise RuntimeError("This response can't handle a polarized source.")

        self._source = source

    def expectation(self, copy = True)-> Histogram:
        # TODO: check coordsys from axis
        # TODO: Earth occ always true in this case

        if self._data is None:
            raise RuntimeError("Call set_source() first.")

        if self._source is None:
            raise RuntimeError("Call set_source() first.")

        # See this issue for the caveats of comparing models
        # https://github.com/threeML/threeML/issues/645
        source_dict = self._source.to_dict()

        coord = self._source.position.sky_coord

        # Use cached expectation if nothing has changed
        if self._expectation is not None and self._last_convolved_source_dict == source_dict:
            if copy:
                return self._expectation.copy()
            else:
                return self._expectation

        # Expectation calculation

        # Check if the source position change, since these operations
        # are expensive
        if self._psr is None or coord != self._last_convolved_source_skycoord:

            coordsys = self._data.axes["PsiChi"].coordsys

            logger.info("... Calculating point source response ...")

            if isinstance(coordsys, SpacecraftFrame):
                # Local coordinates

                dwell_time_map = self._sc_ori.get_dwell_map(coord, nside = self._nside)

                self._psr = PointSourceResponse.from_dwell_time_map(self._data,
                                                                    self._response,
                                                                    dwell_time_map,
                                                                    self._energy_axis,
                                                                    self._polarization_axis)

            else:
                # Inertial e..g. galactic

                scatt_map = self._sc_ori.get_scatt_map(nside=self._nside,
                                                       target_coord=coord,
                                                       earth_occ=True)

                self._psr = PointSourceResponse.from_scatt_map(coord,
                                                               self._data,
                                                               self._response,
                                                               scatt_map,
                                                               self._energy_axis,
                                                               self._polarization_axis)

            logger.info(f"--> done (source name : {self._source.name})")

        # Convolve with spectrum

        if hasattr(self._source.spectrum, 'main'):

            self._expectation = self._psr.get_expectation(self._source.spectrum.main.shape)

        else:

            component_counter = 0

            for item in self._source.spectrum.to_dict():

                if component_counter == 0:
                    self._expectation = self._psr.get_expectation(getattr(self._source.spectrum, item).shape,
                                                                  self._source.components[item].polarization)
                else:
                    self._expectation += self._psr.get_expectation(getattr(self._source.spectrum, item).shape,
                                                                   self._source.components[item].polarization)
                    
                component_counter += 1

        # Check if axes match
        if self._data.axes != self._expectation.axes:
            raise ValueError(
                "Currently, the expectation axes must exactly match the detector response measurement axes")

        # Cache. Use dict and copy since the internal variables can change
        # See this issue for the caveats of comparing models
        # https://github.com/threeML/threeML/issues/645
        self._last_convolved_source_dict = source_dict
        self._last_convolved_source_skycoord = coord.copy()

        # Copy to prevent user to modify our cache
        if copy:
            return self._expectation.copy()
        else:
            return self._expectation
