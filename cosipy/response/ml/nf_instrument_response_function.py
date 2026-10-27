from typing import Iterable, Optional, List, Union

import numpy as np

from cosipy.interfaces.data_interface import EmCDSEventDataInSCFrameInterface
from cosipy.interfaces.instrument_response_interface import FarFieldSpectralInstrumentResponseFunctionInterface
from cosipy.interfaces.photon_parameters import PhotonListWithDirectionAndEnergyInSCFrameInterface
from cosipy.data_io.EmCDSUnbinnedData import EmCDSEventDataInSCFrameFromArrays
from cosipy.response.ml.NFResponse import NFResponse
from cosipy.util.iterables import asarray
from cosipy.interfaces.event_selection import EventSelectorInterface
from cosipy.response.ml.NFNormalizationMap import NFNormalizationMap
from .NFSelectorNormalizationMixin import EnergySelectorNormalizationMixin

import torch


class UnpolarizedNFFarFieldInstrumentResponseFunction(EnergySelectorNormalizationMixin, FarFieldSpectralInstrumentResponseFunctionInterface):
    
    event_data_type = EmCDSEventDataInSCFrameInterface
    photon_list_type = PhotonListWithDirectionAndEnergyInSCFrameInterface
    
    def __init__(self, 
                 response: NFResponse,
                 selection: Optional[EventSelectorInterface] = None):
        if response.is_polarized:
            raise ValueError("The provided NNResponse is polarized, but UnpolarizedNNFarFieldInstrumentResponseFunction only supports unpolarized responses.")
        self._response = response
        
        self._type_map = NFNormalizationMap
        self._load_selector(selection)
        
        self._oversample_factor = 5
    
    def init_compute_pool(self, devices: Optional[List[Union[str, int, torch.device]]]=None):
        self._response.init_compute_pool(devices)
    
    def shutdown_compute_pool(self):
        self._response.shutdown_compute_pool()
    
    @property
    def oversample_factor(self) -> int:
        return self._oversample_factor
    
    @oversample_factor.setter
    def oversample_factor(self, value: int):
        if value <= 0:
            raise ValueError("Oversample factor must be a positive integer.")
        self._oversample_factor = value
    
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
        factor = self._get_norm_factor(
            pol_rad=context[:, 1], 
            az_rad=context[:, 0], 
            ienergy_keV=context[:, 2]
        )
        
        return np.asarray(self._response.evaluate_effective_area(context)) * factor
    
    def _event_probability(self, photons: PhotonListWithDirectionAndEnergyInSCFrameInterface, events: EmCDSEventDataInSCFrameInterface) -> Iterable[float]:
        source = self._get_source(events)
        context = self._get_context(photons)
        factor = self._get_norm_factor(
            pol_rad=context[:, 1], 
            az_rad=context[:, 0], 
            ienergy_keV=context[:, 2]
        )
        selection = self._valid_events(events)
        
        if torch.all(selection):
            densities = self._response.evaluate_density(context, source)
        else:
            densities = torch.zeros(len(selection), dtype=torch.float32)
            densities[selection] = self._response.evaluate_density(context[selection], source[selection])
        
        return np.asarray(densities) * 1/factor
    
    def _random_events(self, photons: PhotonListWithDirectionAndEnergyInSCFrameInterface) -> EmCDSEventDataInSCFrameInterface:
        context = self._get_context(photons)
        
        if self._selector is None:
            samples = self._response.sample_density(context)
            samples[:, 3].mul_(-1).add_(np.pi/2)
            samples = np.asarray(samples)
                    
            return EmCDSEventDataInSCFrameFromArrays(
                samples[:, 0], # Energy
                samples[:, 2], # Lon
                samples[:, 3], # Lat
                samples[:, 1]  # Phi
            )
        else:
            n_photons = context.shape[0]
            
            final_energy = torch.empty(n_photons, dtype=torch.float32)
            final_lon = torch.empty(n_photons, dtype=torch.float32)
            final_lat = torch.empty(n_photons, dtype=torch.float32)
            final_phi = torch.empty(n_photons, dtype=torch.float32)

            filled = torch.zeros(n_photons, dtype=torch.bool)
            active_indices = torch.arange(n_photons, dtype=torch.long)
            
            
            while len(active_indices) > 0:
                sub_context = context[active_indices] # Shape: (M, 3)
                M = len(sub_context)
                
                # Kontext K-mal duplizieren, um parallel K Samples pro aktivem Photon zu erzeugen
                sub_context_expanded = sub_context.repeat_interleave(self._oversample_factor, dim=0) # Shape: (M * K, 3)
                sub_samples = self._response.sample_density(sub_context_expanded) # Shape: (M * K, 4)

                sub_en = sub_samples[:, 0]
                sub_phi = sub_samples[:, 1]
                sub_lon = sub_samples[:, 2]
                sub_lat = -sub_samples[:, 3] + np.pi/2

                sub_events = EmCDSEventDataInSCFrameFromArrays(
                    np.asarray(sub_en),
                    np.asarray(sub_lon),
                    np.asarray(sub_lat),
                    np.asarray(sub_phi)
                )

                # Prüfen, welche der M * K Samples gültig sind
                selection = self._valid_events(sub_events) # Shape: (M * K,) bool

                # Reshape zu (M, K) -> Jedes der M Photonen hat K Versuche
                selection_matrix = selection.view(M, self._oversample_factor)
                
                # Welche Photonen haben mindestens einen gültigen Versuch?
                has_valid = selection_matrix.any(dim=1) # (M,) bool
                
                if has_valid.any():
                    valid_photon_local_indices = torch.where(has_valid)[0]
                    orig_indices = active_indices[valid_photon_local_indices]

                    # Finde den Index des *ersten* gültigen Versuchs (0 bis K-1) für jedes dieser Photonen
                    first_valid_sample_idx = selection_matrix[valid_photon_local_indices].int().argmax(dim=1)
                    
                    # Flache Indizes im expandierten (M * K) Tensor berechnen: photon_idx * K + sample_idx
                    flat_indices = valid_photon_local_indices * self._oversample_factor + first_valid_sample_idx

                    final_energy[orig_indices] = sub_en[flat_indices]
                    final_lon[orig_indices] = sub_lon[flat_indices]
                    final_lat[orig_indices] = sub_lat[flat_indices]
                    final_phi[orig_indices] = sub_phi[flat_indices]

                    filled[orig_indices] = True

                # Restliche, noch nicht gefüllte Photonen für den nächsten Loop-Durchlauf ermitteln
                active_indices = torch.where(~filled)[0]

            return EmCDSEventDataInSCFrameFromArrays(
                np.asarray(final_energy),
                np.asarray(final_lon),
                np.asarray(final_lat),
                np.asarray(final_phi)
            )