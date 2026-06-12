import cosipy
import pytest

if not cosipy.with_ml:
    pytest.skip(reason="Optional [ml] dependencies not installed", allow_module_level=True) 

from cosipy.response.ml import NFWorkerState

def test_nf_worker_state_initialization():
    """
    Sanity check to ensure NFWorkerState is importable and 
    variables are initialized to None as expected for coverage.
    """
    assert NFWorkerState.worker_device is None
    assert NFWorkerState.density_module is None
    assert NFWorkerState.area_module is None
    assert NFWorkerState.progress_queue is None

def test_nf_worker_state_settable():
    """
    Verify that the variables can be updated.
    """
    NFWorkerState.worker_device = "cpu"
    assert NFWorkerState.worker_device == "cpu"
    
    NFWorkerState.worker_device = None