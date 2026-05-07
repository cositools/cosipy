import pytest

import cosipy
if not cosipy.with_ml:
    pytest.skip(reason="Optional [ml] dependencies not installed", allow_module_level=True) 

from unittest.mock import MagicMock, patch
from typing import Dict

from cosipy.response.ml.NFBase import (
    BaseMLP, ConditionalDiagGaussian, NNDensityInferenceWrapper,
    build_cmlp_diaggaussian_base, build_c_arqs_flow, NFBase,
    cuda_cleanup_task, update_density_worker_settings,
    init_density_worker, evaluate_density_task,
    sample_density_task, BaseModel, RateModel,
    DensityModel, DensityApproximation
)
import cosipy.response.ml.NFWorkerState as NFWorkerState

import normflows as nf
import queue
import torch
import torch.nn as nn

def test_nn_components():
    """
    Check if basic NN component output shapes are as expected.
    Ensure the InferenceWrapper returns samples and valid probabilities.
    """
    mlp = BaseMLP(2, 4, 10, 1)
    assert mlp(torch.randn(5, 2)).shape == (5, 4)

    encoder = lambda x: torch.zeros(x.shape[0], 4) 
    dist = ConditionalDiagGaussian(shape=2, context_encoder=encoder)
    ctx = torch.randn(5, 2)
    z, log_p = dist(num_samples=5, context=ctx)
    assert z.shape == (5, 2)
    
    dist2 = ConditionalDiagGaussian(shape=[2,], context_encoder=encoder)
    assert dist2.shape == (2,)
    
    lp = dist.log_prob(z, context=ctx)
    assert lp.shape == (5,)

    mock_inner = MagicMock()
    mock_inner.log_prob.return_value = torch.tensor([0.0])
    mock_inner.sample.return_value = (torch.tensor([[1.0]]), None)
    wrapper = NNDensityInferenceWrapper(mock_inner)
    
    assert wrapper(source=torch.tensor([[0.5]]), mode="inference") == 1.0
    assert wrapper(source=torch.tensor([[0.5]]), context=torch.tensor([[1.5]]), mode="inference") == 1.0
    assert wrapper(n_samples=1, mode="sampling") == 1.0
    assert wrapper(context=torch.tensor([[1.5]]), n_samples=1, mode="sampling") == 1.0
    with pytest.raises(ValueError, match="Unknown mode: invalid_mode"):
        wrapper(mode="invalid_mode")
    
def test_flow_builders():
    """
    Test the helper functions that construct the normflows objects.
    """
    base = build_cmlp_diaggaussian_base(4, 6, 10, 1)
    
    assert isinstance(base, ConditionalDiagGaussian)
    assert base.shape == (3,)
    assert isinstance(base.context_encoder, BaseMLP)
    
    num_layers = 2
    latent_dim = 3
    context_dim = 4
    num_bins = 8
    hidden_units = 16
    res_blocks = 2
    
    flow = build_c_arqs_flow(
        base, num_layers, latent_dim, context_dim, 
        num_bins, hidden_units, res_blocks
    )
    
    assert isinstance(flow, nf.ConditionalNormalizingFlow)
    assert len(flow.flows) == num_layers * 2
    assert isinstance(flow.flows[0], nf.flows.AutoregressiveRationalQuadraticSpline)
    assert isinstance(flow.flows[1], nf.flows.LULinearPermute)

@patch("torch.load")
@patch("torch.multiprocessing.get_context")
def test_nfbase_lifecycle(mock_get_context, mock_load):
    """Test full lifecycle of NFBase class."""
    mock_load.return_value = {
        'version': '1.0.0',
        'density_input': {'feature_dim': 10},
        'extra_key': 'present'
    }
    
    update_fn = MagicMock()
    pool_fn = MagicMock()
    
    mock_ctx = MagicMock()
    mock_get_context.return_value = mock_ctx
    mock_pool = MagicMock()
    mock_ctx.Pool.return_value = mock_pool
    mock_queue = MagicMock()
    mock_ctx.Queue.return_value = mock_queue

    with pytest.raises(KeyError, match="'missing_key'"):
        NFBase("path.ckpt", update_fn, pool_fn, additional_required_keys=['missing_key'])
    
    base = NFBase("path.ckpt", update_fn, pool_fn, additional_required_keys=['extra_key'],
                  show_progress=False, density_batch_size=100_000)
    assert base._major_version == 1
    
    assert base.density_batch_size == 100_000
    assert base.show_progress == False
    
    assert base.density_compile_mode == "default"
    assert base.active_pool is False
    base.clean_compute_pool()    
    base.shutdown_compute_pool()
    
    with pytest.raises(ValueError, match="boolean"):
        base.show_progress = "yes"
    
    with pytest.raises(ValueError, match="positive integer"):
        base.density_batch_size = 0
        
    with pytest.raises(ValueError, match="positive integer"):
        base.density_batch_size = "0"
    
    base2 = NFBase("path.ckpt", update_fn, pool_fn, devices=["cpu", "cuda:1"])
    assert base2.devices == ["cpu", "cuda:1"]
    
    with pytest.raises(ValueError, match="list"):
        base2.devices = "cpu"
    
    base._pool = mock_pool
    base._num_workers = 2
    base.density_batch_size = 500
    mock_pool.map.assert_called_with(update_fn, [('density_batch_size', 500)] * 2)
    
    base._devices = []
    with pytest.raises(RuntimeError, match="no devices provided"):
        base.init_compute_pool()
    
    base._pool = None
    base.init_compute_pool(devices=['cpu', 'cpu'])
    assert base._num_workers == 2
    mock_ctx.Pool.assert_called_once()
    assert base._pool is not None
    assert base._progress_queue is not None
    
    base.clean_compute_pool()
    mock_pool.map.assert_called_with(cuda_cleanup_task, range(2))
    
    base.shutdown_compute_pool()
    mock_pool.close.assert_called_once()
    mock_queue.close.assert_called_once()
    assert base._pool is None

@patch("torch.load")
@patch("torch.multiprocessing.get_context")
def test_nfbase_sampling_orchestration(mock_get_context, mock_load):
    """Tests NFBase.sample_density and its interaction with the worker pool."""
    mock_load.return_value = {'version': '1.0.0', 'density_input': {}}
    
    mock_ctx = MagicMock()
    mock_get_context.return_value = mock_ctx
    mock_pool = MagicMock()
    mock_ctx.Pool.return_value = mock_pool
    mock_queue = MagicMock()
    mock_ctx.Queue.return_value = mock_queue
    
    base = NFBase("fake.ckpt", MagicMock(), MagicMock(), show_progress=True)
    base._pool = mock_pool
    base._num_workers = 2
    base._progress_queue = mock_queue
    
    mock_async = MagicMock()
    mock_async.ready.side_effect = [False, True]
    mock_async.get.return_value = [torch.randn(5, 1), torch.randn(5, 1)]
    mock_pool.map_async.return_value = mock_async
    
    mock_queue.get_nowait.side_effect = [5, 5, queue.Empty, 5, queue.Empty]
    mock_queue.empty.side_effect = [False, False, True]
    
    context = torch.randn(10, 2)
    with patch.object(context, 'is_shared', return_value=False):
        with patch.object(context, 'share_memory_') as mock_share:
            result = base.sample_density(context)
            mock_share.assert_called_once()
            
    assert result.shape == (10, 1)
    mock_pool.map_async.assert_called_once()
    
    mock_pool.map_async.reset_mock()
    mock_async.ready.side_effect = [False, True]
    mock_queue.get_nowait.side_effect = [5, 5, queue.Empty, 5, queue.Empty]
    mock_queue.empty.side_effect = [False, False, True]
    
    with patch.object(context, 'is_shared', return_value=True):
        with patch.object(context, 'share_memory_') as mock_share:
            result2 = base.sample_density(context)
            mock_share.assert_not_called()
    
@patch("torch.load")
@patch("torch.multiprocessing.get_context")
def test_sample_density_temp_pool_logic(mock_get_context, mock_load):
    """Tests sample_density temp pool creation, shutdown, and error handling."""
    mock_load.return_value = {'version': '1.0.0', 'density_input': {}}
    mock_ctx = MagicMock()
    mock_get_context.return_value = mock_ctx
    
    base = NFBase("fake.ckpt", MagicMock(), MagicMock(), show_progress=False)
    base._devices = []
    
    context = torch.randn(10, 2)

    mock_pool = MagicMock()
    mock_queue = MagicMock()
    
    def fake_init(devices=None):
        base._pool = mock_pool
        base._num_workers = len(devices) if devices else 1
        base._progress_queue = mock_queue

    def fake_shutdown():
        base._pool = None
        base._num_workers = 0

    with pytest.raises(RuntimeError, match="No compute pool initialized"):
        base.sample_density(context, devices=None)

    with patch.object(base, 'init_compute_pool', side_effect=fake_init) as mock_init:
        with patch.object(base, 'shutdown_compute_pool', side_effect=fake_shutdown) as mock_shutdown:
            
            mock_async = MagicMock()
            mock_async.ready.return_value = True
            mock_async.get.return_value = [torch.randn(10, 1)]
            mock_pool.map_async.return_value = mock_async
            mock_queue.empty.return_value = True

            base.sample_density(context, devices=['cpu', 'cpu'])
            
            mock_init.assert_called_with(['cpu', 'cpu'])
            mock_shutdown.assert_called_once()
            assert base._pool is None

@patch("torch.load")
@patch("torch.multiprocessing.get_context")
def test_evaluate_density_temp_pool_logic(mock_get_context, mock_load):
    """Tests evaluate_density temp pool creation, shutdown, and error handling."""
    mock_load.return_value = {'version': '1.0.0', 'density_input': {}}
    mock_ctx = MagicMock()
    mock_get_context.return_value = mock_ctx
    
    base = NFBase("fake.ckpt", MagicMock(), MagicMock(), show_progress=False)
    base._devices = []
    
    context = torch.randn(10, 2)
    source = torch.randn(10, 3)

    mock_pool = MagicMock()
    mock_queue = MagicMock()
    
    def fake_init(devices=None):
        base._pool = mock_pool
        base._num_workers = len(devices) if devices else 1
        base._progress_queue = mock_queue

    def fake_shutdown():
        base._pool = None
        base._num_workers = 0

    with pytest.raises(RuntimeError, match="No compute pool initialized"):
        base.evaluate_density(context, source, devices=None)

    with patch.object(base, 'init_compute_pool', side_effect=fake_init) as mock_init:
        with patch.object(base, 'shutdown_compute_pool', side_effect=fake_shutdown) as mock_shutdown:
            
            mock_async = MagicMock()
            mock_async.ready.return_value = True
            mock_async.get.return_value = [torch.randn(10)]
            mock_pool.map_async.return_value = mock_async
            mock_queue.empty.return_value = True

            base.evaluate_density(context, source, devices=['cpu'])
            
            mock_init.assert_called_with(['cpu'])
            mock_shutdown.assert_called_once()
            assert base._pool is None

@patch("torch.load")
@patch("torch.multiprocessing.get_context")
def test_nfbase_evaluate_density_orchestration(mock_get_context, mock_load):
    """Tests evaluate_density with persistent pool, progress bars, and memory sharing."""
    mock_load.return_value = {'version': '1.0.0', 'density_input': {}}
    
    mock_ctx = MagicMock()
    mock_get_context.return_value = mock_ctx
    mock_pool = MagicMock()
    mock_queue = MagicMock()
    
    base = NFBase("fake.ckpt", MagicMock(), MagicMock(), show_progress=True)
    
    base._pool = mock_pool
    base._num_workers = 2
    base._progress_queue = mock_queue
    
    mock_async = MagicMock()

    mock_async.ready.side_effect = [False, True]
    mock_async.get.return_value = [torch.randn(5), torch.randn(5)]
    mock_pool.map_async.return_value = mock_async
    
    mock_queue.get_nowait.side_effect = [5, queue.Empty, 5, queue.Empty]
    mock_queue.empty.side_effect = [False, False, True]
    
    context = torch.randn(10, 2)
    source = torch.randn(10, 3)
    
    with patch.object(context, 'is_shared', return_value=False), \
         patch.object(source, 'is_shared', return_value=False):
        
        with patch.object(context, 'share_memory_') as mock_share_ctx, \
             patch.object(source, 'share_memory_') as mock_share_src:
            
            result = base.evaluate_density(context, source)
            
            assert result.shape == (10,)
            mock_share_ctx.assert_called_once()
            mock_share_src.assert_called_once()
            mock_pool.map_async.assert_called_once()
            
            assert base._pool is not None
    
@patch("torch.load")
@patch("torch.multiprocessing.get_context")
def test_init_compute_pool_reinitialization(mock_get_context, mock_load, capsys):
    """
    Tests the branch where init_compute_pool is called while a pool is already active.
    Verifies:
    1. The warning message is printed.
    2. shutdown_compute_pool is called.
    3. A new pool is successfully initialized.
    """
    mock_load.return_value = {'version': '1.0.0', 'density_input': {}}
    
    mock_ctx = MagicMock()
    mock_get_context.return_value = mock_ctx
    mock_pool = MagicMock()
    mock_ctx.Pool.return_value = mock_pool
    
    base = NFBase("fake.ckpt", MagicMock(), MagicMock())
    
    existing_pool = MagicMock()
    base._pool = existing_pool
    
    def mock_shutdown_side_effect():
        base._pool = None

    with patch.object(base, 'shutdown_compute_pool', side_effect=mock_shutdown_side_effect) as mock_shutdown:
        base.init_compute_pool(devices=['cpu'])
        
        mock_shutdown.assert_called_once()
        
        captured = capsys.readouterr()
        assert "Warning: Pool already initialized" in captured.out
        
        assert base._pool == mock_pool
        mock_ctx.Pool.assert_called_once()

def test_cuda_cleanup_task():
    """Test CUDA cache clearing logic for both available and unavailable CUDA."""
    
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.empty_cache") as mock_empty:
        assert cuda_cleanup_task(None) is True
        mock_empty.assert_called_once()
        
    with patch("torch.cuda.is_available", return_value=False), \
         patch("torch.cuda.empty_cache") as mock_empty:
        assert cuda_cleanup_task(None) is True
        mock_empty.assert_not_called()

@patch("cosipy.response.ml.NFWorkerState.density_module")
def test_update_density_worker_settings(mock_density_module):
    """Test updating settings on the worker's internal model."""
    
    mock_model = MagicMock()
    mock_density_module._model = mock_model
    
    update_density_worker_settings(('density_batch_size', 500))
    assert mock_model.batch_size == 500
    
    update_density_worker_settings(('density_compile_mode', 'max-autotune'))
    assert mock_model.compile_mode == 'max-autotune'
    
    with pytest.raises(ValueError, match="Unknown attribute: unknown"):
        update_density_worker_settings(('unknown', 123))

@patch("torch.cuda.set_device")
def test_init_density_worker(mock_set_device):
    """Test worker initialization for both CPU and CUDA devices."""
    
    mock_queue = MagicMock()
    mock_progress = MagicMock()
    mock_approx_class = MagicMock()
    
    mock_queue.get.return_value = "cuda:0"
    init_density_worker(
        mock_queue, mock_progress, 1, {}, 100, "default", mock_approx_class
    )
    
    assert NFWorkerState.worker_device.type == 'cuda'
    assert NFWorkerState.worker_device.index == 0
    assert NFWorkerState.progress_queue == mock_progress
    mock_set_device.assert_called_with(NFWorkerState.worker_device)
    mock_approx_class.assert_called_once()
    
    NFWorkerState.worker_device = None
    NFWorkerState.progress_queue = None
    NFWorkerState.density_module = None
    NFWorkerState.area_module = None
    
    mock_set_device.reset_mock()
    mock_queue.get.return_value = "cpu"
    
    init_density_worker(
        mock_queue, mock_progress, 1, {}, 100, "default", mock_approx_class
    )
    
    assert NFWorkerState.worker_device.type == 'cpu'
    mock_set_device.assert_not_called()
    
    NFWorkerState.worker_device = None
    NFWorkerState.progress_queue = None
    NFWorkerState.density_module = None
    NFWorkerState.area_module = None

def test_evaluate_density_task():
    """Test evaluate_density task slicing and callback logic."""
    
    NFWorkerState.density_module = MagicMock()
    NFWorkerState.progress_queue = MagicMock()
    
    context = torch.randn(10, 2)
    source = torch.randn(10, 3)
    indices = torch.tensor([1, 3, 5])
    
    evaluate_density_task((context, source, indices))
    
    args, kwargs = NFWorkerState.density_module.evaluate_density.call_args
    assert torch.equal(args[0], context[indices])
    assert torch.equal(args[1], source[indices])
    
    callback = kwargs['progress_callback']
    callback(10)
    NFWorkerState.progress_queue.put.assert_called_with(10)
    
    NFWorkerState.worker_device = None
    NFWorkerState.progress_queue = None
    NFWorkerState.density_module = None
    NFWorkerState.area_module = None

def test_sample_density_task():
    """Test sample_density task slicing and callback logic."""
    
    NFWorkerState.density_module = MagicMock()
    NFWorkerState.progress_queue = MagicMock()
    
    context = torch.randn(10, 2)
    indices = torch.tensor([0, 2, 4])
    
    sample_density_task((context, indices))
    
    args, kwargs = NFWorkerState.density_module.sample_density.call_args
    assert torch.equal(args[0], context[indices])
    
    callback = kwargs['progress_callback']
    callback(5)
    NFWorkerState.progress_queue.put.assert_called_with(5)
    
    NFWorkerState.worker_device = None
    NFWorkerState.progress_queue = None
    NFWorkerState.density_module = None
    NFWorkerState.area_module = None

class DummyBaseModel(BaseModel):
    def _init_model(self, input: Dict):
        return nn.Linear(2, 2)
    
    @property
    def context_dim(self) -> int:
        return 2

def test_basemodel_batch_size():
    """Test batch_size property and setter."""
    model = DummyBaseModel(compile_mode=None, batch_size=10, worker_device='cpu', input={})
    
    assert model.batch_size == 10
    
    model.batch_size = 20
    assert model.batch_size == 20
    
    assert model.context_dim == 2
    
    with pytest.raises(ValueError, match="Batch size must be a positive integer"):
        model.batch_size = -1
        
    with pytest.raises(ValueError, match="Batch size must be a positive integer"):
        model.batch_size = "10"

def test_basemodel_compile_mode_caching():
    """Test compile_mode property and setter."""
    model = DummyBaseModel(compile_mode=None, batch_size=10, worker_device='cpu', input={})
    
    assert model.compile_mode is None
    assert model._model_op == model._base_model
    
    with patch("torch.compile") as mock_compile:
        mock_compile.return_value = "compiled_mock_model"
        
        model.compile_mode = "max-autotune"
        assert model.compile_mode == "max-autotune"
        assert model._model_op == "compiled_mock_model"
        assert "max-autotune" in model._compiled_cache
        mock_compile.assert_called_once_with(model._base_model, mode="max-autotune")
        
        mock_compile.reset_mock()
        model.compile_mode = "max-autotune"
        mock_compile.assert_not_called()
        
        mock_compile.return_value = "compiled_mock_model_2"
        model.compile_mode = "default"
        assert model._model_op == "compiled_mock_model_2"
        assert "default" in model._compiled_cache
        
        mock_compile.reset_mock()
        model.compile_mode = "max-autotune"
        mock_compile.assert_not_called()
        assert model._model_op == "compiled_mock_model"

class DummyRateModel(RateModel):
    def _unpack_rate_input(self, rate_input: Dict):
        self.unpacked = True
        
    @property
    def context_dim(self) -> int:
        return 1
        
    def evaluate_rate(self, *args):
        return torch.tensor([1.0])

def test_rate_model_initialization():
    """Test RateModel initialization."""
    model = DummyRateModel(rate_input={"test": 1})
    assert model.unpacked is True
    assert model.context_dim == 1
    assert model.evaluate_rate() == torch.tensor([1.0])
    
class DummyDensityModel(DensityModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._model_op = MagicMock()

    def _init_model(self, input: Dict):
        return None

    @property
    def context_dim(self) -> int: return 2

    @property
    def source_dim(self) -> int: return 3

    def _transform_context(self, *args):
        return args[0]

    def _inverse_transform_coordinates(self, *args):
        batch_len = args[-1].shape[0]
        return torch.ones((batch_len, self.source_dim))

    def _valid_samples(self, *args):
        batch_len = args[-1].shape[0]
        mask = torch.ones(batch_len, dtype=torch.bool)
        if batch_len == self.batch_size:
            mask[0] = False
        return mask

    def _transform_coordinates(self, *args):
        ctx = args[0]
        src = torch.ones(ctx.shape[0], self.source_dim)
        jac = torch.full((ctx.shape[0],), 2.0)
        return ctx, src, jac

def test_density_model_sample_density():
    """Test sample_density method."""
    model = DummyDensityModel(compile_mode=None, batch_size=4, worker_device='cpu', input={})
    
    model._model_op.return_value = torch.ones(4, model.source_dim)
    
    context = torch.randn(10, 2)
    progress_mock = MagicMock()
    
    result = model.sample_density(context, progress_callback=progress_mock)
    
    assert result.shape == (10, 3)
    assert progress_mock.call_count == 3+1
    
    result_no_cb = model.sample_density(context)
    assert result_no_cb.shape == (10, 3)

def test_density_model_evaluate_density():
    """Test evaluate_density method."""
    model = DummyDensityModel(compile_mode=None, batch_size=5, worker_device='cpu', input={})
    
    model._model_op.return_value = torch.full((5,), 5.0) 
    
    context = torch.randn(10, 2)
    source = torch.randn(10, 3)
    progress_mock = MagicMock()
    
    result = model.evaluate_density(context, source, progress_callback=progress_mock)
    
    assert result.shape == (10,)
    assert torch.all(result == 10.0)
    
    assert progress_mock.call_count == 2
    progress_mock.assert_any_call(5)
    
    result_no_cb = model.evaluate_density(context, source) 
    assert result_no_cb.shape == (10,)

class DummyDensityApproximation(DensityApproximation):
    def _setup_model(self):
        self._model = MagicMock()
        self._expected_context_dim = 2
        self._expected_source_dim = 3

def test_density_approximation_evaluate_density():
    """Test evaluate_density method."""
    approx = DummyDensityApproximation(1, {}, 'cpu', 10, None)
    
    context = torch.randn(10, 2)
    source = torch.randn(10, 3)
    
    approx.evaluate_density(context, source)
    
    args, _ = approx._model.evaluate_density.call_args
    assert len(args) == 5
    
    bad_context = torch.randn(10, 5)
    with pytest.raises(ValueError, match="context has 5"):
        approx.evaluate_density(bad_context, source)
        
    bad_source = torch.randn(10, 1)
    with pytest.raises(ValueError, match="source has 1"):
        approx.evaluate_density(context, bad_source)

def test_density_approximation_sample_density():
    """Test sample_density method."""
    approx = DummyDensityApproximation(1, {}, 'cpu', 10, None)
    
    context = torch.randn(10, 2)
    
    approx.sample_density(context)
    args, _ = approx._model.sample_density.call_args
    assert len(args) == 2
    
    bad_context = torch.randn(10, 1)
    with pytest.raises(ValueError, match="context has 1"):
        approx.sample_density(bad_context)
