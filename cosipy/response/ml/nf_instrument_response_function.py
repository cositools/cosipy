from typing import Iterable, Optional, List, Union

import numpy as np

from cosipy.interfaces.data_interface import EmCDSEventDataInSCFrameInterface
from cosipy.interfaces.instrument_response_interface import FarFieldSpectralInstrumentResponseFunctionInterface
from cosipy.interfaces.photon_parameters import PhotonListWithDirectionAndEnergyInSCFrameInterface
from cosipy.data_io.EmCDSUnbinnedData import EmCDSEventDataInSCFrameFromArrays
from cosipy.response.ml.NFResponse import NFResponse
from cosipy.util.iterables import asarray
from cosipy.event_selection import EnergySelector
from cosipy.interfaces.event_selection import EventSelectorInterface
from cosipy.response.ml.NFNormalizationMap import NFNormalizationMap, IEnergyList
from cosipy.response.ml import NFNormalizationDensity

import torch


class UnpolarizedNFFarFieldInstrumentResponseFunction(FarFieldSpectralInstrumentResponseFunctionInterface):
    
    event_data_type = EmCDSEventDataInSCFrameInterface
    photon_list_type = PhotonListWithDirectionAndEnergyInSCFrameInterface
    
    def __init__(self, 
                 response: NFResponse,
                 selector: Optional[EventSelectorInterface] = None):
        if response.is_polarized:
            raise ValueError("The provided NNResponse is polarized, but UnpolarizedNNFarFieldInstrumentResponseFunction only supports unpolarized responses.")
        self._response = response
        
        self._load_selector(selector)
    
    def _load_selector(self, selector: Optional[EventSelectorInterface] = None):
        self._norm_map = None
        self._selector = None
        if selector is not None:
            if not isinstance(selector, EnergySelector):
                raise ValueError("This response implementation only supports EnergySelector.")
            else:
                self._selector = selector
                self._menergy_min_list = getattr(selector, 'menergy_keV_min', None)
                self._menergy_max_list = getattr(selector, 'menergy_keV_max', None)
                if self._menergy_min_list is None and self._menergy_max_list is None:
                    # TODO: log warning that no map needs to be initialized
                    self._selector = None
                else:
                    if self._menergy_min_list is None:
                        self._menergy_min_list = [np.nan for _ in self._menergy_max_list]
                    if self._menergy_max_list is None:
                        self._menergy_max_list = [np.nan for _ in self._menergy_min_list]
                        
                    self._menergy_intervals = list(zip(self._menergy_min_list, self._menergy_max_list))
    
    @property
    def norm_map(self) -> Optional[NFNormalizationMap]:
        return self._norm_map
    
    # TODO: Init map when already one set? How to changer parameters afterwards? Different types of norm maps...
    # TODO: Check if map matches selector type and selector intervals?

    @norm_map.setter
    def norm_map(self, custom_map: NFNormalizationMap):
        """Allows users to inject a pre-configured normalization map directly."""
        if not isinstance(custom_map, NFNormalizationMap):
            raise TypeError("Provided map must be an instance of NFNormalizationMap.")
        self._norm_map = custom_map
    
    def init_normalization_map(self, 
                                nfdensity: NFNormalizationDensity, 
                                ienergy_keV: IEnergyList, 
                                **kwargs):
        
        if self._selector is None:
            raise ValueError("Cannot initialize normalization map: No selector was provided to the response.")
        else:
            self._norm_map = NFNormalizationMap(
                nfdensity=nfdensity,
                ienergy_keV=ienergy_keV,
                menergy_keV=self._menergy_intervals,
                **kwargs
            )
    
    def _valid_events(self, events: EmCDSEventDataInSCFrameInterface) -> torch.Tensor:
        if self._selector is None:
            return torch.ones(events.nevents, dtype=torch.bool)
        else:
            return torch.tensor(self._selector._select(events), dtype=torch.bool)
    
    def _get_norm_factor(self, context: torch.Tensor) -> Union[np.ndarray, float]:
        if self._selector is None:
            factor = 1.0
        else:
            if self._norm_map is not None:
                factor = self._norm_map.query_normalization(pol_rad=context[:, 1], az_rad=context[:, 0], ienergy_keV=context[:, 2])
            else:
                raise ValueError("A selector was provided to the response, but no normalization map was initialized.")
        
        return factor
    
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
    
    # TODO: Implement diff instead of area and pdf separately + same in the folding class so not to query renorm
    
    def _differential_effective_area_cm2(self, photons: PhotonListWithDirectionAndEnergyInSCFrameInterface, events: EmCDSEventDataInSCFrameInterface) -> Iterable[float]:
        context = self._get_context(photons)
        source = self._get_source(events)
        selection = self._valid_events(events)
        if torch.all(selection):
            diff_area = self._response.evaluate_effective_area(context) * self._response.evaluate_density(context, source)
        else:
            diff_area = torch.zeros(len(selection), dtype=torch.float32)
            diff_area[selection] = self._response.evaluate_effective_area(context[selection]) * self._response.evaluate_density(context[selection], source[selection])
        
        return np.asarray(diff_area)
    
    def _effective_area_cm2(self, photons: PhotonListWithDirectionAndEnergyInSCFrameInterface) -> Iterable[float]: 
        context = self._get_context(photons)
        factor = self._get_norm_factor(context)
        
        return np.asarray(self._response.evaluate_effective_area(context)) * factor
    
    def _event_probability(self, photons: PhotonListWithDirectionAndEnergyInSCFrameInterface, events: EmCDSEventDataInSCFrameInterface) -> Iterable[float]:
        source = self._get_source(events) # TODO: add factor
        context = self._get_context(photons)
        factor = self._get_norm_factor(context)
        selection = self._valid_events(events)
        
        if torch.all(selection):
            densities = self._response.evaluate_density(context, source)
        else:
            densities = torch.zeros(len(selection), dtype=torch.float32)
            densities[selection] = self._response.evaluate_density(context[selection], source[selection])
        
        return np.asarray(densities) * 1/factor
    
    def _random_events(self, photons: PhotonListWithDirectionAndEnergyInSCFrameInterface) -> EmCDSEventDataInSCFrameInterface:
        context = self._get_context(photons) # TODO: add filter
        samples = self._response.sample_density(context)
        samples[:, 3].mul_(-1).add_(np.pi/2)
        samples = np.asarray(samples)
        
        return EmCDSEventDataInSCFrameFromArrays(
            samples[:, 0], # Energy
            samples[:, 2], # Lon
            samples[:, 3], # Lat
            samples[:, 1]  # Phi
        )
