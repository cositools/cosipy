import pytest

import cosipy
if not cosipy.with_ml:
    pytest.skip(reason="Optional [ml] dependencies not installed", allow_module_level=True)

import torch
import torch.multiprocessing as mp
from unittest.mock import MagicMock, patch

from cosipy.background_estimation.ml.NFBackground import (
    BackgroundDensityApproximation,
    BackgroundRateApproximation,
    init_background_worker,
    NFBackground
)

class TestBackgroundDensityApproximation:

    @patch('cosipy.background_estimation.ml.NFBackground.DensityApproximation.__init__', return_value=None)
    @patch('cosipy.background_estimation.ml.NFBackground.TotalBackgroundDensityCMLPDGaussianCARQSFlow')
    def test_setup_model_valid_version(self, mock_flow_cls, mock_base_init):
        """Test that major_version=1 correctly initializes the flow model and sets dimensions."""
        
        mock_flow_instance = MagicMock()
        mock_flow_instance.context_dim = 4
        mock_flow_instance.source_dim = 3
        mock_flow_cls.return_value = mock_flow_instance

        approx = BackgroundDensityApproximation()
        
        approx._major_version = 1
        approx._density_input = {"fake": "input"}
        approx._worker_device = "cpu"
        approx._batch_size = 128
        approx._compile_mode = "default"

        approx._setup_model()

        mock_flow_cls.assert_called_once_with({"fake": "input"}, "cpu", 128, "default")
        
        assert approx._expected_context_dim == 4
        assert approx._expected_source_dim == 3

    @patch('cosipy.background_estimation.ml.NFBackground.DensityApproximation.__init__', return_value=None)
    def test_setup_model_invalid_version(self, mock_base_init):
        """Ensure unsupported versions raise a ValueError."""
        approx = BackgroundDensityApproximation()
        approx._major_version = 999 
        
        with pytest.raises(ValueError, match="Unsupported major version 999 for Density Approximation"):
            approx._setup_model()

class TestBackgroundRateApproximation:

    @patch('cosipy.background_estimation.ml.NFBackground.TotalDC4BackgroundRate')
    def test_init_and_setup_valid_version(self, mock_rate_cls):
        """Test that major_version=1 correctly initializes the rate model."""
        mock_rate_instance = MagicMock()
        mock_rate_instance.context_dim = 2
        mock_rate_cls.return_value = mock_rate_instance

        approx = BackgroundRateApproximation(1, {"fake": "rate_input"})

        mock_rate_cls.assert_called_once_with({"fake": "rate_input"})
        assert approx._expected_context_dim == 2

    def test_setup_model_invalid_version(self):
        """Ensure unsupported versions raise a ValueError."""
        with pytest.raises(ValueError, match="Unsupported major version 2 for Rate Approximation"):
            BackgroundRateApproximation(2, {})

    @patch('cosipy.background_estimation.ml.NFBackground.TotalDC4BackgroundRate')
    def test_evaluate_rate_valid_dimensions(self, mock_rate_cls):
        """Test list unpacking logic for contexts with matching dimensions."""
        mock_rate_instance = MagicMock()
        mock_rate_instance.context_dim = 2
        mock_rate_instance.evaluate_rate.return_value = torch.tensor([5.0, 6.0])
        mock_rate_cls.return_value = mock_rate_instance

        approx = BackgroundRateApproximation(1, {})
        
        context = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = approx.evaluate_rate(context)

        args = mock_rate_instance.evaluate_rate.call_args[0]
        assert len(args) == 2
        assert torch.allclose(args[0], torch.tensor([1.0, 3.0]))
        assert torch.allclose(args[1], torch.tensor([2.0, 4.0]))
        
        assert torch.allclose(result, torch.tensor([5.0, 6.0]))

    @patch('cosipy.background_estimation.ml.NFBackground.TotalDC4BackgroundRate')
    def test_evaluate_rate_invalid_dimensions(self, mock_rate_cls):
        """Ensure dimension mismatches raise a clear ValueError."""
        mock_rate_instance = MagicMock()
        mock_rate_instance.context_dim = 2
        mock_rate_cls.return_value = mock_rate_instance

        approx = BackgroundRateApproximation(1, {})
        
        context = torch.tensor([[1.0, 2.0, 3.0]]) 
        
        with pytest.raises(ValueError, match="Feature mismatch: MagicMock expects 2 features, but context has 3."):
            approx.evaluate_rate(context)

@patch('cosipy.background_estimation.ml.NFBackground.init_density_worker')
def test_init_background_worker(mock_init_density):
    """Test the wrapper to ensure it injects the BackgroundDensityApproximation class."""
    dq = mp.Queue()
    pq = mp.Queue()
    
    init_background_worker(
        device_queue=dq, progress_queue=pq, major_version=1,
        density_input={"test": "data"}, density_batch_size=50,
        density_compile_mode="default"
    )
    
    mock_init_density.assert_called_once_with(
        dq, pq, 1, {"test": "data"}, 50, "default", BackgroundDensityApproximation
    )

def mock_nfbase_init_side_effect(self, *args, **kwargs):
    self._major_version = 1
    self._ckpt = {'rate_input': {'dummy': 'rate_data'}}
    self._density_input = {'dummy': 'density_data'}
    self._density_batch_size = 100_000
    self._density_compile_mode = 'default'

class TestNFBackground:

    @patch('cosipy.background_estimation.ml.NFBackground.NFBase.__init__', autospec=True, side_effect=mock_nfbase_init_side_effect)
    @patch('cosipy.background_estimation.ml.NFBackground.BackgroundRateApproximation')
    def test_init_and_pool_arguments(self, mock_rate_approx, mock_base_init):
        """Test initialization, component loading, and dynamic retrieval of pool arguments."""
        model = NFBackground("fake_path.h5", density_batch_size=100_000, density_compile_mode="default")
        
        mock_base_init.assert_called_once()
        
        mock_rate_approx.assert_called_once_with(1, {'dummy': 'rate_data'})
        
        expected_pool_args = [
            1,                            
            {'dummy': 'density_data'},    
            100_000,                    
            'default'                     
        ]
        assert model._pool_arguments == expected_pool_args

    @patch('cosipy.background_estimation.ml.NFBackground.NFBase.__init__', autospec=True, side_effect=mock_nfbase_init_side_effect)
    @patch('cosipy.background_estimation.ml.NFBackground.BackgroundRateApproximation')
    def test_evaluate_rate(self, mock_rate_approx_cls, mock_base_init):
        """Ensure evaluate_rate seamlessly passes tensors to the internal rate approximation."""
        mock_rate_instance = MagicMock()
        mock_rate_instance.evaluate_rate.return_value = torch.tensor([10.0])
        mock_rate_approx_cls.return_value = mock_rate_instance
        
        model = NFBackground("fake_path.h5")
        
        context = torch.tensor([[1.0, 2.0]])
        result = model.evaluate_rate(context)
        
        assert torch.allclose(result, torch.tensor([10.0]))
        mock_rate_instance.evaluate_rate.assert_called_once_with(context)