import cosipy
import pytest

from unittest.mock import MagicMock, patch

from cosipy.interfaces import UnbinnedThreeMLSourceResponseInterface
from typing import Iterable, Type
from astromodels.sources import Source
from cosipy.interfaces.event import TimeTagEmCDSEventInSCFrameInterface, EmCDSEventInSCFrameInterface
from cosipy.interfaces import EventInterface
from astromodels import PointSource, ExtendedSource
from cosipy import test_data
import shutil
import numpy as np

from cosipy.threeml.unbinned_model_folding import (
    UnbinnedThreeMLModelFolding, 
    CachedUnbinnedThreeMLModelFolding
)

data_path = test_data.path

class MockResponse(UnbinnedThreeMLSourceResponseInterface):
    """Simulates a source response."""
    def __init__(self, counts=10.0, density=None, event_type=TimeTagEmCDSEventInSCFrameInterface):
        self._counts = counts
        self._density = density if density is not None else [1.0, 1.0, 1.0]
        self._event_type = event_type
        self.source_set = None

    def set_source(self, source):
        self.source_set = source

    def copy(self):
        return MockResponse(self._counts, self._density, self._event_type)

    def expected_counts(self) -> float:
        return self._counts

    def expectation_density(self) -> Iterable[float]:
        return self._density
    
    @property
    def event_type(self) -> Type[EventInterface]:
        return self._event_type

class MockCachedResponse(MockResponse):
    """Simulates a response that supports caching to disk."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.init_called = False
        self.saved_path = None
        self.loaded_path = None

    def init_cache(self):
        self.init_called = True

    def cache_to_file(self, path):
        self.saved_path = path

    def cache_from_file(self, path):
        self.loaded_path = path

def test_folding_init_event_type_mismatch():
    """Verify that inconsistent event types raise a RuntimeError."""
    psr = MockResponse(counts = 5.0, density = [1.0, 2.0], event_type=TimeTagEmCDSEventInSCFrameInterface)
    esr = MockResponse(counts = 5.0, density = [1.0, 2.0], event_type=EmCDSEventInSCFrameInterface)
    
    with pytest.raises(RuntimeError):
        UnbinnedThreeMLModelFolding(point_source_response=psr, extended_source_response=esr)

def test_cache_source_responses_no_model():
    """Ensure RuntimeError if expected_counts is called before set_model."""
    folding = UnbinnedThreeMLModelFolding(point_source_response=MockResponse())
    with pytest.raises(RuntimeError):
        folding.expected_counts()
        
def test_cache_source_responses_logic():
    """Test the full lifecycle of the Mixin: mapping sources to responses."""
    mock_model = MagicMock()
    mock_source = MagicMock(spec=PointSource)
    mock_model.sources = {"src1": mock_source}
    mock_model.to_dict.return_value = {"src1": "params_v1"}

    psr = MockResponse(counts=10.0)
    folding = UnbinnedThreeMLModelFolding(point_source_response=psr)
    folding.set_model(mock_model)

    assert folding.expected_counts() == 10.0
    assert "src1" in folding._source_responses
    assert folding._source_responses["src1"].source_set == mock_source

    assert folding._cache_source_responses() is False 

    mock_model.to_dict.return_value = {"src1": "params_v2"}
    assert folding._cache_source_responses() is True

def test_mixin_missing_response_errors():
    """Verify errors when model has a source type but the folding lacks the response."""
    mock_model = MagicMock()
    mock_model.sources = {"ext": MagicMock(spec=ExtendedSource)}
    mock_model.to_dict.return_value = {"ext": "data"}

    folding = UnbinnedThreeMLModelFolding(point_source_response=MockResponse())
    folding.set_model(mock_model)

    with pytest.raises(RuntimeError):
        folding.expected_counts()

def test_expectation_density_with_batching():
    """Test the batching generator path in UnbinnedThreeMLModelFolding."""
    def gen_density():
        yield from [1.0, 2.0, 3.0, 4.0]

    mock_model = MagicMock()
    mock_model.sources = {"s1": MagicMock(spec=PointSource)}
    mock_model.to_dict.return_value = {"s1": "v1"}
    
    psr = MockResponse(density=gen_density())
    folding = UnbinnedThreeMLModelFolding(point_source_response=psr, batch_size=2)
    folding.set_model(mock_model)

    result = list(folding.expectation_density())
    assert result == [1.0, 2.0, 3.0, 4.0]
    assert folding.event_type == TimeTagEmCDSEventInSCFrameInterface

def test_expectation_density_empty_model():
    """Verify that a model with no sources returns an empty iterable."""
    folding = UnbinnedThreeMLModelFolding(point_source_response=MockResponse())
    
    mock_model = MagicMock()
    mock_model.sources = {}
    mock_model.to_dict.return_value = {}
    folding.set_model(mock_model)
    
    result = folding.expectation_density()
    
    assert list(result) == []

def test_expectation_density_fast_track_multi_source():
    """Test the 'fast path' where we sum multiple sources that have __len__."""
    s1_dens = np.array([1.0, 2.0, 3.0])
    s2_dens = np.array([0.5, 0.5, 0.5])
    
    mock_model = MagicMock()
    mock_model.sources = {
        "src1": MagicMock(spec=PointSource),
        "src2": MagicMock(spec=PointSource)
    }
    mock_model.to_dict.return_value = {"src1": 1, "src2": 2}

    psr = MockResponse(density=s1_dens) 
    folding = UnbinnedThreeMLModelFolding(point_source_response=psr)
    folding.set_model(mock_model)

    folding._cache_source_responses()
    folding._source_responses["src1"]._density = s1_dens
    folding._source_responses["src2"]._density = s2_dens

    result = folding.expectation_density()
    
    expected = np.array([1.5, 2.5, 3.5])
    np.testing.assert_allclose(result, expected)

def test_cached_folding_init_cache():
    """Verify init_cache propagates to underlying responses."""
    res_a = MockCachedResponse()
    folding = CachedUnbinnedThreeMLModelFolding(point_source_response=res_a)
    
    folding._source_responses = {"src_a": res_a}
    
    with patch.object(folding, '_cache_source_responses'):
        folding.init_cache()
        assert res_a.init_called is True

def test_cached_folding_save_and_load_with_cleanup():
    """
    Verify saving/loading logic using the library's test_data path.
    Ensures files are created, verified, and strictly cleaned up.
    """
    output_dir = data_path / "temp_cache_test"
    
    res_a = MockCachedResponse()
    res_b = MockCachedResponse()
    
    folding = CachedUnbinnedThreeMLModelFolding(point_source_response=res_a)
    folding._source_responses = {"src_a": res_a, "src_b": res_b}
    
    try:
        with patch.object(folding, '_cache_source_responses'):
            folding.save_caches(output_dir, cache_only=["src_a"])
            
            expected_file_a = output_dir / "src_a_source_response_cache.h5"
            assert res_a.saved_path == expected_file_a
            assert res_b.saved_path is None
            
            expected_file_a.touch()
            
            folding.load_caches(output_dir, load_only=["src_a"])
            assert res_a.loaded_path == expected_file_a
            
            folding.load_caches(output_dir, load_only=["src_b"])
            assert res_b.loaded_path is None
            
    finally:
        if output_dir.exists():
            shutil.rmtree(output_dir)

def test_cached_folding_isinstance_branches():
    """
    Targets the 'False' branch of isinstance(...) checks in 
    init_cache, save_caches, and load_caches.
    """
    output_dir = data_path / "branch_coverage_temp"
    
    res_std = MockResponse() 
    
    folding = CachedUnbinnedThreeMLModelFolding(point_source_response=res_std)
    folding._source_responses = {"src_std": res_std}
    
    try:
        with patch.object(folding, '_cache_source_responses'):
            folding.init_cache()
            
            folding.save_caches(output_dir)
            dummy_file = output_dir / "src_std_source_response_cache.h5"
            assert not (dummy_file).exists()
            
            output_dir.mkdir(parents=True, exist_ok=True)
            dummy_file.touch()
            
            folding.load_caches(output_dir)
            
            assert True 
    finally:
        if output_dir.exists():
            shutil.rmtree(output_dir)
