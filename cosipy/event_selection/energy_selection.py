import numpy as np

from cosipy.interfaces.event_selection import EventSelectorInterface
from cosipy.interfaces import EventDataInterface
from cosipy.util.iterables import itertools_batched, asarray

from typing import Union, List, Optional, Iterable

Numeric = Union[float, int, np.number]

class EnergySelector(EventSelectorInterface):
    
    def __init__(self, 
                 menergy_keV_min: Optional[Union[Numeric, List[Numeric]]] = None, 
                 menergy_keV_max: Optional[Union[Numeric, List[Numeric]]] = None, 
                 batch_size: Optional[int] = None):
        """
        Selects events that fall within ANY of the measured energy intervals defined by
        corresponding pairs of (menergy_keV_min, menergy_keV_max).

        Valid combinations:
        - (None, None): No energy constraints
        - (Scalar, None): Single lower bound only
        - (None, Scalar): Single upper bound only  
        - (Scalar, Scalar): Single energy interval
        - (List, List): Multiple energy intervals (same length required)

        Parameters
        ----------
        menergy_keV_min : numeric, list of numeric, or None
            Minimum measured energy(ies) in keV [inclusive]. If list, menergy_keV_max 
            must also be a list of the same length.
        menergy_keV_max : numeric, list of numeric, or None
            Maximum measured energy(ies) in keV [exclusive]. If list, menergy_keV_min 
            must also be a list of the same length.
        batch_size : int, default None
            Number of events to process at once.
            If None, all values are processed in a single batch.
            This parameter only affects iteration when the expectation density
            is provided as an iterator that is not a numpy array. If it is already
            an array, batching is not applied.
        """
        
        min_is_list = isinstance(menergy_keV_min, list)
        max_is_list = isinstance(menergy_keV_max, list)

        if menergy_keV_min is not None and menergy_keV_max is not None:
            if min_is_list != max_is_list:
                raise ValueError("menergy_keV_min and menergy_keV_max must both be scalar or both be list.")
                
        elif menergy_keV_min is None and menergy_keV_max is not None:
            if max_is_list:
                raise ValueError("When menergy_keV_min is None, menergy_keV_max must not be a list.")
                
        elif menergy_keV_min is not None and menergy_keV_max is None:
            if min_is_list:
                raise ValueError("When menergy_keV_max is None, menergy_keV_min must not be a list.")
                
        else:
            pass

        if menergy_keV_min is not None:
            if not min_is_list:
                menergy_keV_min = [menergy_keV_min]

        if menergy_keV_max is not None:
            if not max_is_list:
                menergy_keV_max = [menergy_keV_max]
        
        if menergy_keV_min is not None and menergy_keV_max is not None:
            if len(menergy_keV_min) != len(menergy_keV_max):
                raise ValueError("menergy_keV_min and menergy_keV_max must have same length.")
            else:
                for mn, mx in zip(menergy_keV_min, menergy_keV_max):
                    if mn >= mx:
                        raise ValueError("menergy_keV_min must be strictly less than menergy_keV_max.")

        self._menergy_min_list = menergy_keV_min
        self._menergy_max_list = menergy_keV_max
        self._batch_size = batch_size
    
    @property
    def menergy_keV_min(self) -> Optional[List[Numeric]]: return self._menergy_min_list

    @property
    def menergy_keV_max(self) -> Optional[List[Numeric]]: return self._menergy_max_list
        
    def _select(self, events:EventDataInterface, early_stop:bool = True) -> Iterable[bool]:
        
        def process_chunk(energy: np.ndarray):
            
            if self._menergy_min_list is None and self._menergy_max_list is None:
                return np.ones_like(energy, dtype=bool)

            if self._menergy_min_list is None:
                result = energy < self._menergy_max_list[0]

            elif self._menergy_max_list is None:
                result = energy >= self._menergy_min_list[0]

            else:
                result = np.zeros(len(energy), dtype=bool)
                
                for e_min, e_max in zip(self._menergy_min_list, self._menergy_max_list):
                    result |= (energy >= e_min) & (energy < e_max)

            return result

        def process_in_chunks(events):

            for chunk in itertools_batched(events, self._batch_size):

                energies = []

                for event in chunk:
                    energies.append(event.energy_keV)

                energies = asarray(energies, dtype=np.float64)

                result = process_chunk(energies)

                yield from result
        
        if getattr(self, '_batch_size', None) is None or isinstance(getattr(events, 'energy_keV', None), np.ndarray):
            return process_chunk(asarray(events.energy_keV, dtype=np.float64))
        else:
            return process_in_chunks(events)
