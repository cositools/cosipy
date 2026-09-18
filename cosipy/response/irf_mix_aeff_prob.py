from typing import Iterable

from cosipy.interfaces.data_interface import EventDataInterface
from cosipy.interfaces.instrument_response_interface import FarFieldSpectralInstrumentResponseFunctionInterface
from cosipy.interfaces.photon_parameters import PhotonListInterface


class IRFMixAeffProbUnpolarized(FarFieldSpectralInstrumentResponseFunctionInterface):
    """
    Testing/validation helper that mixes two independent
    :class:`~cosipy.interfaces.instrument_response_interface.FarFieldSpectralInstrumentResponseFunctionInterface`
    implementations:

    - ``aeff_source`` provides the total effective area
      (``effective_area_cm2``).
    - ``diff_aeff_source`` provides the differential effective area
      (``differential_effective_area_cm2``).

    This is not meant as a physically self-consistent response -- the two
    sources are evaluated independently and are not guaranteed to agree on
    the total effective area they each imply -- but it is useful to
    validate one component of a response against another independently
    (e.g. comparing a histogram-based differential shape against a
    neural-network response's total effective area, or vice versa)
    without having to build a combined response.

    Parameters
    ----------
    aeff_source : FarFieldSpectralInstrumentResponseFunctionInterface
        Source of the total effective area (``effective_area_cm2``).
    diff_aeff_source : FarFieldSpectralInstrumentResponseFunctionInterface
        Source of the differential effective area
        (``differential_effective_area_cm2``).

    Raises
    ------
    ValueError
        If ``aeff_source`` and ``diff_aeff_source`` do not declare the
        same ``photon_list_type``/``event_data_type``.
    """

    def __init__(self,
                 aeff_source: FarFieldSpectralInstrumentResponseFunctionInterface,
                 diff_aeff_source: FarFieldSpectralInstrumentResponseFunctionInterface):

        if (aeff_source.photon_list_type is not diff_aeff_source.photon_list_type
                or aeff_source.event_data_type is not diff_aeff_source.event_data_type):
            raise ValueError("aeff_source and diff_aeff_source must handle the same "
                              "photon_list_type and event_data_type.")

        self.photon_list_type = aeff_source.photon_list_type
        self.event_data_type = aeff_source.event_data_type

        self._aeff_source = aeff_source
        self._diff_aeff_source = diff_aeff_source

    def _effective_area_cm2(self, photons: PhotonListInterface) -> Iterable[float]:
        return self._aeff_source._effective_area_cm2(photons)

    def _differential_effective_area_cm2(self, photons: PhotonListInterface, events: EventDataInterface) -> Iterable[float]:
        return self._diff_aeff_source._differential_effective_area_cm2(photons, events)

    def _random_events(self, photons: PhotonListInterface) -> EventDataInterface:
        """
        Not implemented: neither source is guaranteed to sample events
        consistent with this class's mixed effective area, so no attempt is
        made to pick one.
        """
        raise NotImplementedError("random_events not implemented for IRFMixAeffProbUnpolarized.")
