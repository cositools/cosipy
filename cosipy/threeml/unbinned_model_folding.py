import itertools
from typing import Optional, Iterable, Union

import numpy as np
from astromodels import Model, PointSource, ExtendedSource
from pathlib import Path

from cosipy.interfaces import UnbinnedThreeMLModelFoldingInterface, UnbinnedThreeMLSourceResponseInterface
from cosipy.interfaces.source_response_interface import CachedUnbinnedThreeMLSourceResponseInterface
from cosipy.response.threeml_response import ThreeMLModelFoldingCacheSourceResponsesMixin
from cosipy.util.iterables import asarray
from cosipy.util.iterables import itertools_batched

class UnbinnedThreeMLModelFolding(UnbinnedThreeMLModelFoldingInterface, ThreeMLModelFoldingCacheSourceResponsesMixin):

    def __init__(self,
                 point_source_response = UnbinnedThreeMLSourceResponseInterface,
                 extended_source_response: UnbinnedThreeMLSourceResponseInterface = None,
                 batch_size: Optional[int] = None):

        # Interface inputs
        self._model = None

        # Implementation inputs
        self._psr = point_source_response
        self._esr = extended_source_response
        self._batch_size = batch_size

        if (self._psr is not None) and (self._esr is not None) and self._psr.event_type != self._esr.event_type:
            raise RuntimeError("Point and Extended Source Response must handle the same event type")

        self._event_type = self._psr.event_type

        # Cache
        # Each source has its own cache.
        # We could cache the sum of all sources, but I thought
        # it was not worth it for the typical use case. Usually
        # at least one source changes in between call
        self._cached_model_dict = None
        self._source_responses = {}

    @property
    def event_type(self):
        return self._event_type

    def set_model(self, model: Model):
        """
        The model is passed as a reference and it's parameters
        can change. Remember to check if it changed since the
        last time the user called expectation.
        """
        self._model = model

    def expected_counts(self) -> float:
        """
        Total expected counts
        """

        self._cache_source_responses()

        return sum(s.expected_counts() for s in self._source_responses.values())

    def _expectation_density_batched_gen(self, sources: list) -> Iterable[float]:
        batched_sources = [itertools_batched(s, self._batch_size) for s in sources]
        
        for chunks in zip(*batched_sources):
            densities = [asarray(c, dtype=np.float64) for c in chunks]
            
            yield from np.add.reduce(densities)
    
    def expectation_density(self) -> Iterable[float]:
        self._cache_source_responses()
        
        if not self._source_responses:
            return np.array([], dtype=np.float64)
        
        sources = [ex.expectation_density() for ex in self._source_responses.values()]
        
        if (self._batch_size is None) or all(hasattr(s, "__len__") for s in sources):
            densities = [asarray(s, dtype=np.float64) for s in sources]
            return np.add.reduce(densities)
        else:
            return self._expectation_density_batched_gen(sources)


class CachedUnbinnedThreeMLModelFolding(UnbinnedThreeMLModelFolding):
    def __init__(self,
                 point_source_response: Optional[UnbinnedThreeMLSourceResponseInterface] = None,
                 extended_source_response: Optional[UnbinnedThreeMLSourceResponseInterface] = None, 
                 batch_size: Optional[int] = None):
        
        super().__init__(point_source_response=point_source_response, 
                         extended_source_response=extended_source_response,
                         batch_size=batch_size)
        
        self._base_filename = "_source_response_cache.h5"
        
    def init_cache(self):
        """
        Forces the creation of response objects for each source in the model.
        """
        self._cache_source_responses()
        
        for response in self._source_responses.values():
            if isinstance(response, CachedUnbinnedThreeMLSourceResponseInterface):
                response.init_cache()
    
    def save_caches(self, directory: Union[str, Path], cache_only: Optional[Iterable[str]] = None):
        """Saves only the responses that implement the cache interface."""
        self.init_cache()
            
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        
        for name, response in self._source_responses.items():
            if (cache_only is not None) and (name not in set(cache_only)):
                continue
            if isinstance(response, CachedUnbinnedThreeMLSourceResponseInterface):
                filepath = dir_path / f"{name}{self._base_filename}"
                response.cache_to_file(filepath)
    
    def load_caches(self, directory: Union[str, Path], load_only: Optional[Iterable[str]] = None):
        """Loads available cache files into compatible response objects."""
        self._cache_source_responses()

        dir_path = Path(directory)
        for name, response in self._source_responses.items():
            if (load_only is not None) and (name not in set(load_only)):                
                continue
            if isinstance(response, CachedUnbinnedThreeMLSourceResponseInterface):
                filepath = dir_path / f"{name}{self._base_filename}"
                if filepath.exists():
                    response.cache_from_file(filepath)
