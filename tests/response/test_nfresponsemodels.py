import pytest

import cosipy
if not cosipy.with_ml:
    pytest.skip(reason="Optional [ml] dependencies not installed", allow_module_level=True)

import torch
import numpy as np
from unittest.mock import MagicMock, patch

# Assuming the import paths align with your library structure
from cosipy.response.ml.NFResponseModels import (
    UnpolarizedAreaSphericalHarmonicsExpansion,
    UnpolarizedDensityCMLPDGaussianCARQSFlow
)

@pytest.fixture
def dummy_area_input():
    """Provides a minimal valid dictionary for initializing the Area model."""
    lmax = 1
    poly_degree = 2
    poly_coeffs = torch.ones((2, 3, 4), dtype=torch.float64)
    
    return {
        'lmax': lmax,
        'poly_degree': poly_degree,
        'poly_coeffs': poly_coeffs
    }

@pytest.fixture
def dummy_density_input():
    """Provides a minimal valid dictionary for initializing the Density model."""
    return {
        "model_state_dict": {},
        "bins": 2,
        "hidden_units": 4,
        "residual_blocks": 1,
        "total_layers": 1,
        "context_size": 4,
        "mlp_hidden_units": 4,
        "mlp_hidden_layers": 1,
        "menergy_cuts": [10.0, 10000.0],
        "phi_cuts": [0.0, np.pi]
    }

class TestUnpolarizedAreaSphericalHarmonicsExpansion:

    @patch("sphericart.torch.SphericalHarmonics")
    def test_init_and_context_dim(self, mock_sh, dummy_area_input):
        """Test correct extraction of inputs and context dimensions."""
        model = UnpolarizedAreaSphericalHarmonicsExpansion(
            dummy_area_input, 'cpu', batch_size=10, compile_mode=None
        )
        assert model.context_dim == 3
        assert model._lmax == 1
        assert model._poly_degree == 2
        mock_sh.assert_called_once_with(1)

    @patch("sphericart.torch.SphericalHarmonics")
    def test_convert_coefficients(self, mock_sh, dummy_area_input):
        """Verify the m=0, m>0, and m<0 branches in ALM coefficient conversion."""
        dummy_area_input['poly_coeffs'][0, :, :] = 2.0 
        dummy_area_input['poly_coeffs'][1, :, :] = 3.0 
        
        model = UnpolarizedAreaSphericalHarmonicsExpansion(dummy_area_input, 'cpu', 10, None)
        
        assert model._conv_coeffs.shape == (3, 4) 
        
        assert torch.all(model._conv_coeffs[:, 0] == 2.0)
        assert torch.allclose(model._conv_coeffs[:, 1], torch.tensor(3 * np.sqrt(2), dtype=torch.float64))
        assert torch.allclose(model._conv_coeffs[:, 3], torch.tensor(-2 * np.sqrt(2), dtype=torch.float64))

    @patch("sphericart.torch.SphericalHarmonics")
    def test_horner_eval(self, mock_sh, dummy_area_input):
        """Test Horner's method for polynomial evaluation."""
        model = UnpolarizedAreaSphericalHarmonicsExpansion(dummy_area_input, 'cpu', 10, None)
        
        model._conv_coeffs = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64)
        
        x = torch.tensor([2.0], dtype=torch.float32)
        result = model._horner_eval(x)
        
        assert torch.allclose(result, 1*x**2 + 2*x + 3)
        assert result.dtype == torch.float32

    @patch("sphericart.torch.SphericalHarmonics")
    def test_compute_spherical_harmonics(self, mock_sh, dummy_area_input):
        """Test the Cartesian mapping before passing to sphericart."""
        model = UnpolarizedAreaSphericalHarmonicsExpansion(dummy_area_input, 'cpu', 10, None)
        
        dir_az = torch.tensor([0.0, np.pi/2])
        dir_polar = torch.tensor([np.pi/2, np.pi/2])
        
        model._compute_spherical_harmonics(dir_az, dir_polar)
        
        args, _ = model._sh_calculator.call_args
        xyz = args[0]
        
        assert torch.allclose(xyz[0], torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32), atol=1e-6)
        assert torch.allclose(xyz[1], torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32), atol=1e-6)

    @patch("sphericart.torch.SphericalHarmonics")
    def test_evaluate_effective_area(self, mock_sh, dummy_area_input):
        """Test batching, progress callbacks, and tensor clamping."""
        model = UnpolarizedAreaSphericalHarmonicsExpansion(dummy_area_input, 'cpu', batch_size=2, compile_mode=None)
        
        model._sh_calculator = MagicMock(return_value=torch.ones((2, 4)))
        
        model._model_op = MagicMock(return_value=torch.tensor([[-5.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]))
        
        dir_az = torch.tensor([0.0, 0.1, 0.2, 0.4])
        dir_polar = torch.tensor([0.0, 0.1, 0.2, 0.15])
        energy = torch.tensor([100.0, 100.0, 100.0, 120.0])
        
        cb = MagicMock()
        result = model.evaluate_effective_area(dir_az, dir_polar, energy, progress_callback=cb)
        
        assert cb.call_count == 2
        cb.assert_any_call(2)
        
        assert result.shape == (4,)
        assert result[0] == 0.0
        assert result[1] == 4.0
        assert torch.all(result >= 0)
        
        result = model.evaluate_effective_area(dir_az, dir_polar, energy)
        
        assert result.shape == (4,)
        assert result[0] == 0.0
        assert result[1] == 4.0
        assert torch.all(result >= 0)

class TestUnpolarizedDensityCMLPDGaussianCARQSFlow:

    @patch("cosipy.response.ml.NFResponseModels.build_cmlp_diaggaussian_base")
    @patch("cosipy.response.ml.NFResponseModels.build_c_arqs_flow")
    @patch("cosipy.response.ml.NFResponseModels.NNDensityInferenceWrapper")
    def test_init_and_build(self, mock_wrapper, mock_flow, mock_base, dummy_density_input):
        """Test instantiation, dimensions, and model building."""
        mock_model = MagicMock()
        mock_flow.return_value = mock_model
        
        model = UnpolarizedDensityCMLPDGaussianCARQSFlow(
            dummy_density_input, 'cpu', batch_size=10, compile_mode=None
        )
        
        assert model.context_dim == 3
        assert model.source_dim == 4
        assert model._context_size == dummy_density_input["context_size"]
        assert model._mlp_hidden_units == dummy_density_input["mlp_hidden_units"]
        mock_model.load_state_dict.assert_called_once()
        mock_wrapper.assert_called_once_with(mock_model)

    @patch.object(UnpolarizedDensityCMLPDGaussianCARQSFlow, "_load_model", return_value=MagicMock())
    def test_get_vector(self, mock_load, dummy_density_input):
        """Test spherical to cartesian logic."""
        model = UnpolarizedDensityCMLPDGaussianCARQSFlow(dummy_density_input, 'cpu', 10, None)
        
        phi_sc = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        theta_sc = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        
        vec = model._get_vector(phi_sc, theta_sc)
        
        expected = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        torch.testing.assert_close(vec, expected)

    @patch.object(UnpolarizedDensityCMLPDGaussianCARQSFlow, "_load_model", return_value=MagicMock())
    def test_convert_conventions(self, mock_load, dummy_density_input):
        """Test the kinematics conversion and ensure zeta wrapping logic is covered."""
        model = UnpolarizedDensityCMLPDGaussianCARQSFlow(dummy_density_input, 'cpu', 10, None)
        
        dir_az_sc = torch.tensor([[1., 0.], [0., 1.]])
        dir_polar_sc = torch.tensor([[1., 0.], [1., 0.]])
        scatt_az_sc = torch.tensor([[0., 1.], [1., 0.]])
        scatt_polar_sc = torch.tensor([[1., 0.], [0., 1.]])
        
        ei = torch.tensor([500.0, 500.0])
        em = torch.tensor([250.0, 250.0])
        phi = torch.tensor([0.5, 0.5])
        
        with patch.object(model, '_get_vector') as mock_get_vec:
            mock_get_vec.side_effect = [
                torch.tensor([[0., 0., 1.], [0., 0., 1.]]),
                torch.tensor([[1., -1., 0.], [1., 1., 0.]])
            ]
            
            eps, theta, zeta = model._convert_conventions(
                dir_az_sc, dir_polar_sc, ei, em, phi, scatt_az_sc, scatt_polar_sc
            )
            
            assert torch.allclose(eps, em/ei - 1)
            assert eps.shape == (2,)
            assert theta.shape == (2,)
            assert zeta.shape == (2,)
            assert torch.all(zeta >= 0)

    @patch.object(UnpolarizedDensityCMLPDGaussianCARQSFlow, "_load_model", return_value=MagicMock())
    def test_inverse_transform_coordinates(self, mock_load, dummy_density_input):
        """Test inverse coordinate transform and psi_cds wrapping branch."""
        model = UnpolarizedDensityCMLPDGaussianCARQSFlow(dummy_density_input, 'cpu', 10, None)
        
        neps, nphi, ntheta, nzeta = [torch.tensor([-0.5, 0.5]) for _ in range(4)]
        dir_az = torch.tensor([0.0, np.pi/2])
        dir_pol = torch.tensor([np.pi/4, np.pi/4])
        ei = torch.tensor([500.0, 1000.0])
        
        with patch('torch.bmm') as mock_bmm:
            mock_bmm.return_value = torch.tensor([[1.0, -1.0, 0.0], [1.0, 1.0, 0.0]]).unsqueeze(-1)
            
            res = model._inverse_transform_coordinates(neps, nphi, ntheta, nzeta, dir_az, dir_pol, ei)
            
            assert res.shape == (2, 4)
            assert res[0, 0] == 500.0 * (0.5 + 1) 
            assert torch.all(res[:, 2] >= 0)      

    @patch.object(UnpolarizedDensityCMLPDGaussianCARQSFlow, "_load_model", return_value=MagicMock())
    def test_transform_coordinates_jacobian_clamping(self, mock_load, dummy_density_input):
        """Test Jacobian evaluation, specifically clamping of Inf/Negative values."""
        model = UnpolarizedDensityCMLPDGaussianCARQSFlow(dummy_density_input, 'cpu', 10, None)
        
        dir_az = dir_pol = scatt_az = scatt_pol = torch.tensor([0.0, 0.0])
        ei = torch.tensor([500.0, 0.0])
        em = torch.tensor([250.0, 250.0])
        phi = torch.tensor([0.5, 0.5])
        
        with patch.object(model, '_convert_conventions', return_value=(torch.ones(2), torch.ones(2), torch.ones(2))), \
             patch.object(model, '_transform_context', return_value=torch.ones((2, 3))):
             
             ctx, src, jac = model._transform_coordinates(dir_az, dir_pol, ei, em, phi, scatt_az, scatt_pol)
             
             assert ctx.shape == (2, 3)
             assert src.shape == (2, 4)
             assert jac.shape == (2,)
             assert jac[1] == 0.0 

    @patch.object(UnpolarizedDensityCMLPDGaussianCARQSFlow, "_load_model", return_value=MagicMock())
    def test_transform_context(self, mock_load, dummy_density_input):
        """Test physical variable scaling into NN context variables."""
        model = UnpolarizedDensityCMLPDGaussianCARQSFlow(dummy_density_input, 'cpu', 10, None)
        
        dir_az = torch.tensor([0.0, np.pi/2])
        dir_pol = torch.tensor([0.0, np.pi])
        ei = torch.tensor([100.0, 10000.0])
        
        ctx = model._transform_context(dir_az, dir_pol, ei)
        assert ctx.shape == (2, model._context_size)

    @patch.object(UnpolarizedDensityCMLPDGaussianCARQSFlow, "_load_model", return_value=MagicMock())
    def test_valid_samples(self, mock_load, dummy_density_input):
        """Test the boolean masking rules."""
        model = UnpolarizedDensityCMLPDGaussianCARQSFlow(dummy_density_input, 'cpu', 10, None)
        
        def run_mask(neps, nphi, ntheta, nzeta, ei):
            tensors = [torch.tensor([val]) for val in [neps, nphi, ntheta, nzeta, 0.0, 0.0, ei]]
            return model._valid_samples(*tensors)[0].item()

        assert run_mask(neps=0.5, nphi=0.5, ntheta=0.3, nzeta=0.5, ei=1000.0) is True
        
        assert run_mask(neps=1.1, nphi=0.5, ntheta=0.3, nzeta=0.5, ei=1000.0) is False   
        assert run_mask(neps=-10.0, nphi=0.5, ntheta=0.3, nzeta=0.5, ei=1000.0) is False 
        assert run_mask(neps=0.5, nphi=1.5, ntheta=0.3, nzeta=0.5, ei=1000.0) is False   
        assert run_mask(neps=0.5, nphi=0.5, ntheta=1.5, nzeta=0.5, ei=1000.0) is False   
        assert run_mask(neps=0.5, nphi=0.1, ntheta=0.1, nzeta=0.5, ei=1000.0) is False   