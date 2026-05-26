from typing import Optional, Union, List
import numpy as np
import torch

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
    def norm_map(self) -> Optional[NFNormalizationMapBase]:
        return self._norm_map
    # TODO: Init map when already one set? How to changer parameters afterwards? Different types of norm maps...
    # TODO: Check if map matches selector type and selector intervals?
    # TODO: Changing map has influence on expectation chache (e.g. background counts)
    @norm_map.setter
    def norm_map(self, custom_map: NFNormalizationMapBase):
        if not isinstance(custom_map, self._type_map):
            raise TypeError("Provided map must be an instance of {}".format(self._type_map.__name__))
        self._norm_map = custom_map
    
    def _valid_events(self, events) -> torch.Tensor:
        if self._selector is None:
            return torch.ones(events.nevents, dtype=torch.bool)
        else:
            return torch.tensor(self._selector._select(events), dtype=torch.bool)
        
    def init_normalization_map(self, 
                               nfdensity: Optional[NFNormalizationMapBase] = None,
                               intvar: Optional[List] = None,
                               **kwargs
                               ):
        if self._selector is None:
            raise ValueError(f"Cannot initialize normalization map: No selector was provided to the {self.__class__.__name__}.")
        else:
            self._norm_map = self._type_map(
                nfdensity, intvar,
                menergy_keV=self._menergy_intervals,
                **kwargs
            )
    
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