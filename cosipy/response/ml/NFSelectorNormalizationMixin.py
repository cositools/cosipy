from typing import Optional, Union, List
import numpy as np
import torch
from astropy.units import Quantity
from astropy.time import Time

from cosipy.event_selection import EnergySelector
from cosipy.interfaces.event_selection import EventSelectorInterface
from .NFNormalizationBase import NFNormalizationMapBase

class EnergySelectorNormalizationMixin:
    
    _type_map: NFNormalizationMapBase
    
    def _load_selector(self, selector: Optional[EventSelectorInterface] = None):
        self._norm_map = None
        self._selector = None
        if selector is not None:
            if not isinstance(selector, EnergySelector):
                raise ValueError("This implementation only supports EnergySelector.")
            else:
                self._selector = selector
                menergy_list = selector.energy_ranges_keV
                if len(menergy_list) == 0:
                    raise ValueError("The provided EnergySelector has no energy ranges defined.")
                else:
                    self._energy_ranges = selector.energy_ranges

    @property
    def norm_map(self) -> Optional[NFNormalizationMapBase]:
        return self._norm_map

    @norm_map.setter
    def norm_map(self, custom_map: NFNormalizationMapBase):
        if not isinstance(custom_map, self._type_map):
            raise TypeError("Provided map must be an instance of {}".format(self._type_map.__name__))
        if self._norm_map is not None:
            raise ValueError("Normalization map has already been set. To change it, please reset the selector first.")
        compatibility_check = custom_map.check_compatibility(menergy=self._energy_ranges)
        if not compatibility_check:
            raise ValueError("Provided normalization map is not compatible with the current energy ranges.")
        self._norm_map = custom_map
    
    def _valid_events(self, events) -> torch.Tensor:
        if self._selector is None:
            return torch.ones(events.nevents, dtype=torch.bool)
        else:
            return torch.tensor(self._selector._select(events), dtype=torch.bool)
        
    def init_normalization_map(self, 
                               nfdensity: Optional[NFNormalizationMapBase] = None,
                               intvar: Optional[Union[Quantity, Time]] = None,
                               **kwargs
                               ):
        if self._selector is None:
            raise ValueError(f"Cannot initialize normalization map: No selector was provided to the {self.__class__.__name__}.")
        if self._norm_map is None:
            self._norm_map = self._type_map(
                nfdensity, intvar,
                menergy=self._energy_ranges,
                **kwargs)   
    
    def _get_norm_factor(self, *args, **kwargs) -> Union[np.ndarray, float]:
        if self._selector is None:
            factor = 1.0
        else:
            if self._norm_map is None:
                raise ValueError(
                    f"A selector was provided to the {self.__class__.__name__}, "
                    "but no normalization map was initialized."
                )
            else:
                factor = self._norm_map.query_normalization(*args, **kwargs)
            
        return factor