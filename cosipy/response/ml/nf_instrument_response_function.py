from typing import Iterable, Optional, List, Union

import numpy as np

from cosipy.interfaces.data_interface import EmCDSEventDataInSCFrameInterface
from cosipy.interfaces.instrument_response_interface import FarFieldSpectralInstrumentResponseFunctionInterface
from cosipy.interfaces.photon_parameters import PhotonListWithDirectionAndEnergyInSCFrameInterface
from cosipy.data_io.EmCDSUnbinnedData import EmCDSEventDataInSCFrameFromArrays
from cosipy.response.ml.NFResponse import NFResponse
from cosipy.util.iterables import asarray

import torch


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
