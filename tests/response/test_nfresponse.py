import pytest

import cosipy
if not cosipy.with_ml:
    pytest.skip(reason="Optional [ml] dependencies not installed", allow_module_level=True)

import torch
import torch.multiprocessing as mp
from unittest.mock import MagicMock, patch
import queue

from cosipy.response.ml.NFResponse import (
    ResponseDensityApproximation,
    AreaApproximation,
    update_response_worker_settings,
    init_response_worker,
    evaluate_area_task,
    NFResponse
)
import cosipy.response.ml.NFWorkerState as NFWorkerState


@patch("cosipy.response.ml.NFResponse.UnpolarizedDensityCMLPDGaussianCARQSFlow")
def test_response_density_approximation(mock_density_model):
    """Test mapping of major versions to correct Density Models."""
    mock_instance = MagicMock()
    mock_instance.context_dim = 2
    mock_instance.source_dim = 3
    mock_density_model.return_value = mock_instance

    approx = ResponseDensityApproximation(1, {}, 'cpu', 100, None)
    assert approx._expected_context_dim == 2
    assert approx._expected_source_dim == 3
    mock_density_model.assert_called_once()

    with pytest.raises(ValueError, match="Unsupported major version 99"):
        ResponseDensityApproximation(99, {}, 'cpu', 100, None)

@patch("cosipy.response.ml.NFResponse.UnpolarizedAreaSphericalHarmonicsExpansion")
def test_area_approximation(mock_area_model):
    """Test mapping and evaluation logic of AreaApproximation."""
    mock_instance = MagicMock()
    mock_instance.context_dim = 2
    mock_area_model.return_value = mock_instance

    approx = AreaApproximation(1, {}, 'cpu', 100, None)
    assert approx._expected_context_dim == 2
    mock_area_model.assert_called_once()

    with pytest.raises(ValueError, match="Unsupported major version 99"):
        AreaApproximation(99, {}, 'cpu', 100, None)

    context = torch.randn(5, 2)
    mock_instance.evaluate_effective_area.return_value = torch.ones(5)
    
    cb = MagicMock()
    result = approx.evaluate_effective_area(context, progress_callback=cb)
    
    assert torch.equal(result, torch.ones(5))
    args, kwargs = mock_instance.evaluate_effective_area.call_args
    assert len(args) == 2 
    assert kwargs['progress_callback'] == cb

    bad_context = torch.randn(5, 3)
    with pytest.raises(ValueError, match="Feature mismatch"):
        approx.evaluate_effective_area(bad_context)

@patch("cosipy.response.ml.NFResponse.update_density_worker_settings")
def test_update_response_worker_settings(mock_update_density):
    """Test worker settings updates map correctly to the internal models."""
    NFWorkerState.area_module = MagicMock()
    
    update_response_worker_settings(('area_batch_size', 500_000))
    assert NFWorkerState.area_module._model.batch_size == 500_000
    mock_update_density.assert_not_called()
    
    update_response_worker_settings(('area_compile_mode', 'default'))
    assert NFWorkerState.area_module._model.compile_mode == 'default'
    mock_update_density.assert_not_called()
    
    update_response_worker_settings(('density_batch_size', 100))
    mock_update_density.assert_called_with(('density_batch_size', 100))

    mock_update_density.side_effect = ValueError("Unknown attribute: unknown")
    with pytest.raises(ValueError, match="Unknown attribute"):
        update_response_worker_settings(('unknown', 'value'))
    
    NFWorkerState.area_module = None

@patch("cosipy.response.ml.NFResponse.init_density_worker")
@patch("cosipy.response.ml.NFResponse.AreaApproximation")
def test_init_response_worker(mock_area_approx, mock_init_density):
    """Test that the response worker initializes both density and area approximations."""
    device_queue = MagicMock()
    progress_queue = MagicMock()
    NFWorkerState.worker_device = 'cpu'
    
    init_response_worker(
        device_queue, progress_queue, 1, 
        {'area': 'in'}, {'density': 'in'}, 
        300_000, 100_000, 'default', 'max-autotune'
    )
    
    mock_init_density.assert_called_once()
    mock_area_approx.assert_called_once_with(1, {'area': 'in'}, 'cpu', 300_000, 'default')
    assert NFWorkerState.area_module is not None
    
    NFWorkerState.area_module = None
    NFWorkerState.worker_device = None

def test_evaluate_area_task():
    """Test multiprocessing task unpacking and queue callbacks for area evaluation."""
    NFWorkerState.area_module = MagicMock()
    NFWorkerState.progress_queue = MagicMock()
    
    context = torch.randn(10, 2)
    indices = torch.tensor([1, 4, 7])
    
    evaluate_area_task((context, indices))
    
    args, kwargs = NFWorkerState.area_module.evaluate_effective_area.call_args
    assert torch.equal(args[0], context[indices])
    
    cb = kwargs['progress_callback']
    cb(3)
    NFWorkerState.progress_queue.put.assert_called_with(3)
    
    NFWorkerState.area_module = None
    NFWorkerState.progress_queue = None

@patch("torch.load")
@patch("torch.multiprocessing.get_context")
def test_nfresponse_properties_and_initialization(mock_get_context, mock_load):
    """Verify NFResponse properties, setters, and checkpoint validation."""
    mock_load.return_value = {
        'version': '1.0.0',
        'density_input': {},
        'is_polarized': True,
        'area_input': {}
    }
    
    mock_ctx = MagicMock()
    mock_get_context.return_value = mock_ctx
    mock_pool = MagicMock()
    mock_ctx.Pool.return_value = mock_pool
    
    resp = NFResponse("fake_path.ckpt", area_batch_size=200_000, area_compile_mode="max-autotune-no-cudagraphs")
    assert resp.is_polarized is True
    assert resp.area_batch_size == 200_000
    assert resp.area_compile_mode == "max-autotune-no-cudagraphs"
    
    resp._pool = mock_pool
    resp._num_workers = 2
    
    resp.area_batch_size = 400_000
    assert resp.area_batch_size == 400_000
    mock_pool.map.assert_called_with(resp._update_worker, [('area_batch_size', 400_000)] * 2)
    
    resp.area_compile_mode = "default"
    assert resp.area_compile_mode == "default"
    mock_pool.map.assert_called_with(resp._update_worker, [('area_compile_mode', 'default')] * 2)
    
    with pytest.raises(ValueError, match="positive integer"):
        resp.area_batch_size = -50
    with pytest.raises(ValueError, match="positive integer"):
        resp.area_batch_size = "100"

@patch("torch.load")
@patch("torch.multiprocessing.get_context")
def test_nfresponse_evaluate_effective_area_orchestration(mock_get_context, mock_load):
    """Test evaluate_effective_area map_async calls, memory sharing, and progress bars."""
    mock_load.return_value = {
        'version': '1.0.0', 'density_input': {},
        'is_polarized': False, 'area_input': {}
    }
    
    mock_ctx = MagicMock()
    mock_get_context.return_value = mock_ctx
    mock_pool = MagicMock()
    mock_queue = MagicMock()
    
    resp = NFResponse("fake.ckpt", show_progress=True)
    resp._pool = mock_pool
    resp._num_workers = 2
    resp._progress_queue = mock_queue
    
    mock_async = MagicMock()
    mock_async.ready.side_effect = [False, True]
    mock_async.get.return_value = [torch.ones(5), torch.ones(5)]
    mock_pool.map_async.return_value = mock_async
    
    mock_queue.get_nowait.side_effect = [5, queue.Empty, 5, queue.Empty]
    mock_queue.empty.side_effect = [False, False, True]
    
    context = torch.randn(10, 2)
    
    with patch.object(context, 'is_shared', return_value=False):
        with patch.object(context, 'share_memory_') as mock_share:
            
            result = resp.evaluate_effective_area(context)
            
            assert result.shape == (10,)
            assert torch.all(result == 1.0)
            mock_share.assert_called_once()
            mock_pool.map_async.assert_called_once()
            assert resp._pool is not None 

@patch("torch.load")
@patch("torch.multiprocessing.get_context")
def test_nfresponse_evaluate_effective_area_temp_pool(mock_get_context, mock_load):
    """Ensure evaluate_effective_area correctly opens and closes a temp pool when needed."""
    mock_load.return_value = {
        'version': '1.0.0', 'density_input': {},
        'is_polarized': False, 'area_input': {}
    }
    mock_ctx = MagicMock()
    mock_get_context.return_value = mock_ctx
    
    resp = NFResponse("fake.ckpt", show_progress=False)
    resp._devices = [] 
    
    context = torch.randn(10, 2)

    with pytest.raises(RuntimeError, match="No compute pool initialized"):
        resp.evaluate_effective_area(context, devices=None)

    mock_pool = MagicMock()
    mock_queue = MagicMock()
    
    def fake_init(devices=None):
        resp._pool = mock_pool
        resp._num_workers = len(devices) if devices else 1
        resp._progress_queue = mock_queue

    def fake_shutdown():
        resp._pool = None
        resp._num_workers = 0

    with patch.object(resp, 'init_compute_pool', side_effect=fake_init) as mock_init, \
         patch.object(resp, 'shutdown_compute_pool', side_effect=fake_shutdown) as mock_shutdown:
        
        mock_async = MagicMock()
        mock_async.ready.return_value = True
        mock_async.get.return_value = [torch.randn(10)]
        mock_pool.map_async.return_value = mock_async
        mock_queue.empty.return_value = True

        resp.evaluate_effective_area(context, devices=['cpu'])
        
        mock_init.assert_called_with(['cpu'])
        mock_shutdown.assert_called_once()
        assert resp._pool is None 
        
@patch("torch.load")
@patch("torch.multiprocessing.get_context")
def test_nfresponse_evaluate_area_already_shared(mock_get_context, mock_load):
    """
    Test that evaluate_effective_area skips calling share_memory_() 
     if the context tensor is already in shared memory.
    """
    mock_load.return_value = {
        'version': '1.0.0', 'density_input': {},
        'is_polarized': False, 'area_input': {}
    }
    
    mock_ctx = MagicMock()
    mock_get_context.return_value = mock_ctx
    mock_pool = MagicMock()
    
    resp = NFResponse("fake.ckpt", show_progress=False)
    resp._pool = mock_pool
    resp._num_workers = 1
    resp._progress_queue = MagicMock()
    
    mock_async = MagicMock()
    mock_async.ready.return_value = True
    mock_async.get.return_value = [torch.ones(5)]
    mock_pool.map_async.return_value = mock_async
    
    context = torch.randn(5, 2)
    
    with patch.object(context, 'is_shared', return_value=True):
        with patch.object(context, 'share_memory_') as mock_share:
            
            resp.evaluate_effective_area(context)
            
            mock_share.assert_not_called()        

        