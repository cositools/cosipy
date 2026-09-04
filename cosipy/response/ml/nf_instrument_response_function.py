from typing import Iterable, Optional, List, Union

import numpy as np

import torch

from cosipy.interfaces.data_interface import EmCDSEventDataInSCFrameInterface
from cosipy.interfaces.instrument_response_interface import FarFieldSpectralInstrumentResponseFunctionInterface
from cosipy.interfaces.photon_parameters import PhotonListWithDirectionAndEnergyInSCFrameInterface
from cosipy.data_io.EmCDSUnbinnedData import EmCDSEventDataInSCFrameFromArrays
from cosipy.response.ml.NFResponse import NFResponse
from cosipy.response.relative_irf_hist import IRFRelativeHistUnpolarized
from cosipy.util.iterables import asarray


class UnpolarizedNFFarFieldInstrumentResponseFunction(FarFieldSpectralInstrumentResponseFunctionInterface):

    event_data_type = EmCDSEventDataInSCFrameInterface
    photon_list_type = PhotonListWithDirectionAndEnergyInSCFrameInterface

    def __init__(self, response: NFResponse,):
        if response.is_polarized:
            raise ValueError("The provided NNResponse is polarized, but UnpolarizedNNFarFieldInstrumentResponseFunction only supports unpolarized responses.")
        self._response = response

    def init_compute_pool(self, devices: Optional[List[Union[str, int, torch.device]]]=None):
        self._response.init_compute_pool(devices)

    def shutdown_compute_pool(self):
        self._response.shutdown_compute_pool()

    @property
    def active_pool(self) -> bool: return self._response.active_pool

    @staticmethod
    def _get_context(photons: PhotonListWithDirectionAndEnergyInSCFrameInterface):
        lon = torch.as_tensor(asarray(photons.direction_lon_rad_sc, dtype=np.float32))
        lat = torch.as_tensor(asarray(photons.direction_lat_rad_sc, dtype=np.float32))
        en  = torch.as_tensor(asarray(photons.energy_keV, dtype=np.float32))

        lat = -lat + (np.pi / 2)
        return torch.stack([lon, lat, en], dim=1)

    @staticmethod
    def _get_source(events: EmCDSEventDataInSCFrameInterface):
        lon = torch.as_tensor(asarray(events.scattered_lon_rad_sc, dtype=np.float32))
        lat = torch.as_tensor(asarray(events.scattered_lat_rad_sc, dtype=np.float32))
        phi = torch.as_tensor(asarray(events.scattering_angle_rad, dtype=np.float32))
        en  = torch.as_tensor(asarray(events.energy_keV, dtype=np.float32))

        lat = -lat + (np.pi / 2)
        return torch.stack([en, phi, lon, lat], dim=1)

    def _effective_area_cm2(self, photons: PhotonListWithDirectionAndEnergyInSCFrameInterface) -> Iterable[float]:
        context = self._get_context(photons)

        return np.asarray(self._response.evaluate_effective_area(context))

    def _event_probability(self, photons: PhotonListWithDirectionAndEnergyInSCFrameInterface, events: EmCDSEventDataInSCFrameInterface) -> Iterable[float]:
        source = self._get_source(events)
        context = self._get_context(photons)

        return np.asarray(self._response.evaluate_density(context, source))

    def _random_events(self, photons: PhotonListWithDirectionAndEnergyInSCFrameInterface) -> EmCDSEventDataInSCFrameInterface:
        context = self._get_context(photons)
        samples = self._response.sample_density(context)
        samples[:, 3].mul_(-1).add_(np.pi/2)
        samples = np.asarray(samples)

        return EmCDSEventDataInSCFrameFromArrays(
            samples[:, 0], # Energy
            samples[:, 2], # Lon
            samples[:, 3], # Lat
            samples[:, 1]  # Phi
        )


class IRFRelativeHistWithNFAeffUnpolarized(FarFieldSpectralInstrumentResponseFunctionInterface):
    """
    Testing/validation helper that mixes two unpolarized far-field response
    implementations:

    - The total effective area comes from an
      :class:`UnpolarizedNFFarFieldInstrumentResponseFunction` (the
      neural-network response).
    - The differential effective area comes from an
      :class:`IRFRelativeHistUnpolarized` (the histogram-based response).

    This is not meant as a physically self-consistent response -- the two
    sources are evaluated independently and are not guaranteed to agree on
    the total effective area they each imply -- but it is useful to compare
    the histogram's differential shape against the NN response's total
    effective area (or vice versa) without having to build a combined
    ``aeff`` histogram (see :class:`IRFRelativeHistUnpolarized`'s ``aeff``
    constructor argument for that alternative).

    Parameters
    ----------
    aeff_irf : UnpolarizedNFFarFieldInstrumentResponseFunction
        Source of the total effective area (``effective_area_cm2``).
    diff_aeff_irf : IRFRelativeHistUnpolarized
        Source of the differential effective area
        (``differential_effective_area_cm2``).
    """

    event_data_type = EmCDSEventDataInSCFrameInterface
    photon_list_type = PhotonListWithDirectionAndEnergyInSCFrameInterface

    def __init__(self,
                 aeff_irf: UnpolarizedNFFarFieldInstrumentResponseFunction,
                 diff_aeff_irf: IRFRelativeHistUnpolarized):

        if aeff_irf.photon_list_type is not self.photon_list_type or aeff_irf.event_data_type is not self.event_data_type:
            raise ValueError("aeff_irf is expected to handle the same photon/event types as "
                              "IRFRelativeHistWithNFAeffUnpolarized.")

        if diff_aeff_irf.photon_list_type is not self.photon_list_type or diff_aeff_irf.event_data_type is not self.event_data_type:
            raise ValueError("diff_aeff_irf is expected to handle the same photon/event types as "
                              "IRFRelativeHistWithNFAeffUnpolarized.")

        self._aeff_irf = aeff_irf
        self._diff_aeff_irf = diff_aeff_irf

    def init_compute_pool(self, devices: Optional[List[Union[str, int, torch.device]]]=None):
        self._aeff_irf.init_compute_pool(devices)

    def shutdown_compute_pool(self):
        self._aeff_irf.shutdown_compute_pool()

    @property
    def active_pool(self) -> bool: return self._aeff_irf.active_pool

    def _effective_area_cm2(self, photons: PhotonListWithDirectionAndEnergyInSCFrameInterface) -> Iterable[float]:
        return self._aeff_irf._effective_area_cm2(photons)

    def _differential_effective_area_cm2(self, photons: PhotonListWithDirectionAndEnergyInSCFrameInterface, events: EmCDSEventDataInSCFrameInterface) -> Iterable[float]:
        return self._diff_aeff_irf._differential_effective_area_cm2(photons, events)

    def _random_events(self, photons: PhotonListWithDirectionAndEnergyInSCFrameInterface) -> EmCDSEventDataInSCFrameInterface:
        """
        Not implemented: neither source is guaranteed to sample events
        consistent with this class's mixed effective area, so no attempt is
        made to pick one.
        """
        raise NotImplementedError("random_events not implemented for IRFRelativeHistWithNFAeffUnpolarized.")
