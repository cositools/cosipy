import pytest

import cosipy
if not cosipy.with_ml:
    pytest.skip(reason="Optional [ml] dependencies not installed", allow_module_level=True)

import numpy as np
import torch
from unittest.mock import MagicMock

from astropy import units as u
from astropy.time import Time

from cosipy.data_io.EmCDSUnbinnedData import TimeTagEmCDSEventInSCFrameInterface
from cosipy.background_estimation.ml.NFBackground import NFBackground
from cosipy.interfaces.data_interface import TimeTagEmCDSEventDataInSCFrameInterface
from cosipy.interfaces.event import EventInterface

from cosipy.background_estimation.ml.nf_unbinned_background import FreeNormNFUnbinnedBackground


@pytest.fixture
def mock_sc_history():
    """Provides a realistic SpacecraftHistory mock using real Astropy units/times."""
    sc_history = MagicMock()
    
    obstime = Time([1000.0, 1010.0, 1020.0], format='unix')
    sc_history.obstime = obstime
    sc_history.tstart = obstime[0]
    sc_history.tstop = obstime[-1]
    
    sc_history.livetime = [9.0, 8.5] * u.s
    sc_history.intervals_duration = [10.0, 10.0] * u.s

    sc_history.cumulative_livetime.return_value = 17.5 * u.s
    
    return sc_history

@pytest.fixture
def mock_data():
    """Provides a realistic Data mock using real Astropy times."""
    data = MagicMock(spec=TimeTagEmCDSEventDataInSCFrameInterface)
    
    data.energy_keV = [500.0, 1000.0]
    data.scattering_angle_rad = [0.5, 1.0]
    data.scattered_lon_rad_sc = [0.1, 0.2]
    data.scattered_lat_rad_sc = [0.3, 0.4]
    
    data.time = Time([1005.0, 1015.0], format='unix')
    
    return data

@pytest.fixture
def mock_model():
    """Provides a mocked NFBackground model that returns predictable tensors."""
    model = MagicMock(spec=NFBackground)
    
    model.evaluate_rate.return_value = torch.tensor([2.0, 3.0])
    model.evaluate_density.return_value = torch.tensor([0.5, 0.6])
    
    model.active_pool = True
    return model


@pytest.fixture
def background_instance(mock_model, mock_data, mock_sc_history):
    """Instantiates the background class with all dependencies."""
    return FreeNormNFUnbinnedBackground(
        model=mock_model,
        data=mock_data,
        sc_history=mock_sc_history,
        label="test_bkg_norm"
    )


class TestFreeNormNFUnbinnedBackground:

    def test_init_and_properties(self, background_instance):
        """Test initial state, event type, and parameters property."""
        assert background_instance.event_type == TimeTagEmCDSEventInSCFrameInterface
        assert background_instance.offset == 1e-12
        assert background_instance._label == "test_bkg_norm"
        assert background_instance._accum_livetime == 17.5 
        
        params = background_instance.parameters
        assert "test_bkg_norm" in params
        assert isinstance(params["test_bkg_norm"], u.Quantity)

    def test_offset_validation(self, background_instance):
        """Test the offset setter logic, including negative value guardrails."""
        background_instance.offset = 0.0
        assert background_instance.offset == 0.0
        
        background_instance.offset = None
        assert background_instance.offset is None
        
        with pytest.raises(ValueError, match="The offset cannot be negative."):
            background_instance.offset = -1.0

    def test_integrate_rate(self, background_instance, mock_model):
        """Test that the integration correctly multiplies rates by livetime."""
        expected_counts = background_instance._integrate_rate()
        
        assert expected_counts == 43.5
        mock_model.evaluate_rate.assert_called_once()
        
        passed_times = mock_model.evaluate_rate.call_args[0][0]
        assert torch.allclose(passed_times, torch.tensor([[1005.0], [1015.0]], dtype=torch.float64))

    def test_compute_density(self, background_instance, mock_model):
        """Test the complex density computation and bin mapping logic."""
        densities = background_instance._compute_density()
        
        assert isinstance(densities, np.ndarray)
        assert densities.dtype == np.float64
        np.testing.assert_allclose(densities, [0.9, 1.53])
        
        passed_source = mock_model.evaluate_density.call_args[0][1]
        assert passed_source.shape == (2, 4)
        assert torch.allclose(passed_source[0, 3], torch.tensor(np.pi/2 - 0.3, dtype=torch.float32))

    def test_compute_density_time_bounds_error(self, background_instance, mock_data):
        """Ensure evaluating events outside the spacecraft history raises an error."""
        mock_data.time = Time([900.0, 1050.0], format='unix')
        
        with pytest.raises(ValueError, match="Input times are outside the spacecraft history range"):
            background_instance._compute_density()

    def test_norm_setter_and_getter(self, background_instance):
        """Test scaling logic when norm is manipulated."""
        
        initial_norm_qty = background_instance.norm
        assert initial_norm_qty.unit == u.Hz
        np.testing.assert_allclose(initial_norm_qty.value, 43.5 / 17.5)
        
        background_instance.norm = 5.0 * u.Hz
        
        np.testing.assert_allclose(background_instance._norm, 5.0 * (17.5 / 43.5))
        
        background_instance.set_parameters(test_bkg_norm=10.0 * u.Hz)
        assert background_instance.norm.value == 10.0

    def test_expected_counts_method(self, background_instance):
        """Test that expected counts scale by the internal norm."""
        assert background_instance.expected_counts() == 43.5
        
        background_instance._norm = 2.0
        assert background_instance.expected_counts() == 87.0

    def test_expectation_density_method(self, background_instance):
        """Test final density calculations, including norm scaling and offset addition."""
        
        res1 = background_instance.expectation_density()
        np.testing.assert_allclose(res1, [0.9 + 1e-12, 1.53 + 1e-12])
        
        background_instance._norm = 2.0
        background_instance.offset = None
        res2 = background_instance.expectation_density()
        np.testing.assert_allclose(res2, [1.8, 3.06])

    def test_compute_pool_management(self, background_instance, mock_model):
        """Ensure lazy evaluation accurately initializes and shutdowns compute pools."""
        mock_model.active_pool = False
        
        background_instance.expectation_density()
        
        mock_model.init_compute_pool.assert_called_once()
        mock_model.shutdown_compute_pool.assert_called_once()