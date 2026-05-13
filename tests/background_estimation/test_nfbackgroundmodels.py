import pytest

import cosipy
if not cosipy.with_ml:
    pytest.skip(reason="Optional [ml] dependencies not installed", allow_module_level=True)

import numpy as np
import torch
from unittest.mock import MagicMock, patch

from cosipy.background_estimation.ml.NFBackgroundModels import (
    TotalDC4BackgroundRate,
    TotalBackgroundDensityCMLPDGaussianCARQSFlow
)

@pytest.fixture
def dummy_rate_input():
    return {
        "slew_duration": 10.0,
        "obs_duration": 50.0,
        "start_time": 1000.0,
        "offset": 5.0,
        "slope": 0.1,
        "buildup": ((1.0, 2.0), (10.0, 20.0)),
        "scale": 0.5,
        "cutoff": (90.0, (1.0, 1.0, 1.0), (0.1, 0.1, 0.1), (0.0, 30.0, 60.0)),
        "outlocs": torch.tensor([500.0, 900.0, 1500.0]),
        "saa_decay": ((2.0, 3.0), (15.0, 30.0))
    }

@pytest.fixture
def dummy_density_input():
    return {
        "model_state_dict": {},
        "bins": 8,
        "hidden_units": 64,
        "residual_blocks": 2,
        "total_layers": 3,
        "context_size": 5,
        "mlp_hidden_units": 32,
        "mlp_hidden_layers": 2,
        "menergy_cuts": (100.0, 10000.0),
        "phi_cuts": (0.0, np.pi),
        "start_time": 1000.0,
        "total_time": 10000.0,
        "period": 5400.0, 
        "slew_duration": 600.0,
        "obs_duration": 3000.0,
        "outlocs": torch.tensor([500.0, 900.0, 1500.0])
    }

class TestTotalDC4BackgroundRate:
    
    def test_context_dim_property(self, dummy_rate_input):
        model = TotalDC4BackgroundRate(dummy_rate_input)
        assert model.context_dim == 1

    def test_unpack_rate_input(self, dummy_rate_input):
        """Ensure all elements from the dictionary are mapped to the correct instance variables."""
        model = TotalDC4BackgroundRate(dummy_rate_input)
        
        assert model._slew_duration == 10.0
        assert model._offset == 5.0
        assert model._buildup_A == (1.0, 2.0)
        assert model._cutoff_mu == (0.0, 30.0, 60.0)
        assert torch.allclose(model._outlocs, torch.tensor([500.0, 900.0, 1500.0]))

    def test_static_math_methods(self):
        """Test the pure mathematical equations for buildup and decay."""
        t = torch.tensor([10.0])
        
        buildup_res = TotalDC4BackgroundRate._buildup(t, A=4.0, T=10.0)
        assert torch.allclose(buildup_res, torch.tensor([2.0]))
        
        decay_res = TotalDC4BackgroundRate._decay(t, A=4.0, T=10.0)
        assert torch.allclose(decay_res, torch.tensor([2.0]))
        
        vm_res = TotalDC4BackgroundRate._von_mises(torch.tensor([0.0]), T=10.0, A=2.0, kappa=1.0, mu=0.0)
        assert torch.allclose(vm_res, torch.tensor([2.0 * np.exp(1.0)], dtype=torch.float32))

    def test_pointing_scale(self, dummy_rate_input):
        """Test the sigmoid boundary logic inside pointing scale."""
        model = TotalDC4BackgroundRate(dummy_rate_input)
        
        res = model._pointing_scale(torch.tensor([0.0, 60.0, 120.0]), scale=0.5, k0=10.0)
        assert res.shape == (3,)
        assert np.isclose(res[0], res[2])
        
    def test_saa_decay(self, dummy_rate_input):
        """Test SAA decay correctly identifies the proper last exit time using searchsorted."""
        model = TotalDC4BackgroundRate(dummy_rate_input)
        
        time_mins = torch.tensor([0.0, 10.0]) 
        
        decay = model._saa_decay(time_mins, A=(2.0, 3.0), T=(15.0, 30.0))
        assert decay.shape == (2,)
        assert np.all(np.isclose(decay,
                                 model._decay(time_mins - torch.tensor([-100/60, 500/60]), 2.0, 15.0) +
                                 model._decay(time_mins - torch.tensor([-100/60, 500/60]), 3.0, 30.0)))
        assert torch.all(decay > 0) 

    def test_evaluate_rate(self, dummy_rate_input):
        """Test the full aggregation method."""
        model = TotalDC4BackgroundRate(dummy_rate_input)
        
        abs_times = torch.tensor([1000.0, 1060.0, 1120.0])
        rates = model.evaluate_rate(abs_times)
        
        assert rates.shape == (3,)
        assert rates.dtype == torch.float32

class TestTotalBackgroundDensity:

    @patch('cosipy.background_estimation.ml.NFBackgroundModels.NNDensityInferenceWrapper')
    @patch('cosipy.background_estimation.ml.NFBackgroundModels.build_c_arqs_flow')
    @patch('cosipy.background_estimation.ml.NFBackgroundModels.build_cmlp_diaggaussian_base')
    def test_init_and_properties(self, mock_base_builder, mock_flow_builder, mock_wrapper, dummy_density_input):
        """Test that the flow builds correctly from dict parameters and properties read out correctly."""
        model = TotalBackgroundDensityCMLPDGaussianCARQSFlow(
            density_input=dummy_density_input,
            worker_device="cpu",
            batch_size=128,
            compile_mode=None
        )
        
        assert model.context_dim == 1
        assert model.source_dim == 4
        
        assert model._menergy_cuts == (100.0, 10000.0)
        assert model._total_time == 10000.0
        
        mock_base_builder.assert_called_once()
        mock_flow_builder.assert_called_once()
        mock_wrapper.assert_called_once()

    @patch('cosipy.background_estimation.ml.NFBackgroundModels.TotalBackgroundDensityCMLPDGaussianCARQSFlow._load_model', return_value=None)
    def test_inverse_transform_coordinates(self, mock_load, dummy_density_input):
        """Test the physics to normalized-coordinate inverse mappings."""
        model = TotalBackgroundDensityCMLPDGaussianCARQSFlow(dummy_density_input, "cpu", 128)
        
        nem = torch.tensor([0.0])  
        nphi = torch.tensor([0.5]) 
        npsi = torch.tensor([0.25])
        nchi = torch.tensor([0.5]) 
        dummy = torch.tensor([0.0])
        
        res = model._inverse_transform_coordinates(nem, nphi, npsi, nchi, dummy)
        
        assert res.shape == (1, 4)
        np.testing.assert_allclose(res[0, 0].item(), 100.0)
        np.testing.assert_allclose(res[0, 1].item(), np.pi / 2)
        np.testing.assert_allclose(res[0, 2].item(), np.pi / 2)
        np.testing.assert_allclose(res[0, 3].item(), np.pi / 2)

    @patch('cosipy.background_estimation.ml.NFBackgroundModels.TotalBackgroundDensityCMLPDGaussianCARQSFlow._load_model', return_value=None)
    def test_transform_coordinates(self, mock_load, dummy_density_input):
        """Test calculation of transformed context, source, and jacobian."""
        model = TotalBackgroundDensityCMLPDGaussianCARQSFlow(dummy_density_input, "cpu", 128)
        
        time = torch.tensor([1000.0])
        em = torch.tensor([1000.0])
        phi = torch.tensor([np.pi])
        scatt_az = torch.tensor([np.pi])
        scatt_pol = torch.tensor([np.pi/2])
        
        ctx, src, jac = model._transform_coordinates(time, em, phi, scatt_az, scatt_pol)
        
        assert ctx.shape == (1, 5)
        assert src.shape == (1, 4) 
        assert jac.shape == (1,)
        
        np.testing.assert_allclose(src[0, 0].item(), 0.5)

    @patch('cosipy.background_estimation.ml.NFBackgroundModels.TotalBackgroundDensityCMLPDGaussianCARQSFlow._load_model', return_value=None)
    def test_valid_samples(self, mock_load, dummy_density_input):
        """Test the logical masking bounds for validation checks."""
        model = TotalBackgroundDensityCMLPDGaussianCARQSFlow(dummy_density_input, "cpu", 128)
        
        nem = torch.tensor([0.0, -1.0, 0.5])   
        nphi = torch.tensor([0.5, 1.5, 0.5])   
        npsi = torch.tensor([0.5, 0.5, -0.1])  
        nchi = torch.tensor([0.5, 0.5, 0.5])
        dummy = torch.tensor([0.0, 0.0, 0.0])
        
        mask = model._valid_samples(nem, nphi, npsi, nchi, dummy)
        
        assert mask.tolist() == [True, False, False]