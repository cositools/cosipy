from typing import Dict, Iterable, Type, Optional

from astropy import units as u
import numpy as np

from cosipy import SpacecraftHistory
from cosipy.interfaces.event import EventInterface
from cosipy.interfaces.data_interface import TimeTagEmCDSEventDataInSCFrameInterface
from cosipy.data_io.EmCDSUnbinnedData import TimeTagEmCDSEventInSCFrameInterface
from cosipy.interfaces.background_interface import BackgroundDensityInterface
from cosipy.util.iterables import asarray

from cosipy.background_estimation.ml.NFBackground import NFBackground
import torch

class FreeNormNFUnbinnedBackground(BackgroundDensityInterface):
    
    def __init__(self,
                 model: NFBackground,
                 data: TimeTagEmCDSEventDataInSCFrameInterface,
                 sc_history: SpacecraftHistory,
                 label: str = "bkg_norm"):
        
        self._expected_counts = None
        self._expectation_density = None
        self._model = model
        self._data = data
        self._sc_history = sc_history
        
        self._accum_livetime = self._sc_history.cumulative_livetime().to_value(u.s)
        
        self._norm = 1
        self._label = label
        self._offset: Optional[float] = 1e-12
    
    @property
    def event_type(self) -> Type[EventInterface]:
        return TimeTagEmCDSEventInSCFrameInterface
    
    @property
    def offset(self) -> Optional[float]:
        return self._offset
    
    @offset.setter
    def offset(self, offset: Optional[float]):
        if (offset is not None) and (offset < 0):
            raise ValueError("The offset cannot be negative.")
        self._offset = offset
    
    @property
    def norm(self) -> u.Quantity:
        self._update_cache(counts_only=True)
        return u.Quantity(self._norm * self._expected_counts/self._accum_livetime, u.Hz)
    
    @norm.setter
    def norm(self, norm: u.Quantity):
        self._update_cache(counts_only=True)
        self._norm = norm.to_value(u.Hz) * self._accum_livetime/self._expected_counts
    
    def set_parameters(self, **parameters: u.Quantity) -> None:
        self.norm = parameters[self._label]
    
    @property
    def parameters(self) -> Dict[str, u.Quantity]:
        return {self._label: self.norm}
    
    def _integrate_rate(self) -> float:
        mid_times = torch.as_tensor((self._sc_history.obstime[:-1] + (self._sc_history.obstime[1:] - self._sc_history.obstime[:-1]) / 2).utc.unix).view(-1, 1)
        rate = self._model.evaluate_rate(mid_times)
        return torch.sum(rate * torch.as_tensor(self._sc_history.livetime)).item()
    
    def _compute_density(self):
        self._energy_m_keV = torch.as_tensor(asarray(self._data.energy_keV, dtype=np.float32))
        self._phi_rad = torch.as_tensor(asarray(self._data.scattering_angle_rad, dtype=np.float32))
        self._lon_scatt = torch.as_tensor(asarray(self._data.scattered_lon_rad_sc, dtype=np.float32))
        self._lat_scatt = torch.as_tensor(asarray(self._data.scattered_lat_rad_sc, dtype=np.float32))
        source = torch.stack((self._energy_m_keV, self._phi_rad, self._lon_scatt, np.pi/2 - self._lat_scatt), dim=1)
        
        time = torch.as_tensor(self._data.time.utc.unix).view(-1, 1)
        if torch.any((time < self._sc_history.tstart.utc.unix) | (time > self._sc_history.tstop.utc.unix)):
            raise ValueError("Input times are outside the spacecraft history range")
        interval_ratios = torch.as_tensor(self._sc_history.livetime.to_value(u.s) / self._sc_history.intervals_duration.to_value(u.s))
        factor = torch.searchsorted(torch.as_tensor(self._sc_history.obstime.utc.unix), time.view(-1), right=True) - 1

        return np.asarray(self._model.evaluate_density(time, source) * self._model.evaluate_rate(time) * interval_ratios[factor], dtype=np.float64)
    
    def _update_cache(self, counts_only=False):
        if self._expected_counts is None:
            self._expected_counts = self._integrate_rate()
            
        if (self._expectation_density is None) and (not counts_only):
            active_pool = self._model.active_pool
            if not active_pool:
                self._model.init_compute_pool()
            self._expectation_density = self._compute_density()
            if not active_pool:
                self._model.shutdown_compute_pool()
    
    def expected_counts(self) -> float:
        self._update_cache()
        
        return self._expected_counts * self._norm
    
    def expectation_density(self) -> Iterable[float]:
        self._update_cache()
        
        result = self._expectation_density * self._norm
        
        if self._offset is not None:
            return result + self._offset
        else:
            return result
    