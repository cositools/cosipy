import pytest

import cosipy
if not cosipy.with_ml:
    pytest.skip(reason="Optional [ml] dependencies not installed", allow_module_level=True)

import numpy as np
import torch
from unittest.mock import MagicMock, patch

from cosipy.response.ml.nf_instrument_response_function import UnpolarizedNFFarFieldInstrumentResponseFunction
from cosipy.response.ml.NFResponse import NFResponse
from cosipy.interfaces.photon_parameters import PhotonListWithDirectionAndEnergyInSCFrameInterface
from cosipy.data_io.EmCDSUnbinnedData import EmCDSEventDataInSCFrameFromArrays


@pytest.fixture
def mock_unpolarized_response():
    """Mock an NFResponse that is explicitly unpolarized."""
    mock_resp = MagicMock(spec=NFResponse)
    mock_resp.is_polarized = False
    return mock_resp

@pytest.fixture
def mock_polarized_response():
    """Mock an NFResponse that is polarized."""
    mock_resp = MagicMock(spec=NFResponse)
    mock_resp.is_polarized = True
    return mock_resp

@pytest.fixture
def mock_photons():
    """Mock a PhotonListWithDirectionAndEnergyInSCFrameInterface with predictable values."""
    photons = MagicMock(spec=PhotonListWithDirectionAndEnergyInSCFrameInterface)
    photons.direction_lon_rad_sc = np.array([0.0, np.pi/2, np.pi])
    photons.direction_lat_rad_sc = np.array([0.0, np.pi/4, -np.pi/2])
    photons.energy_keV = np.array([100.0, 500.0, 1000.0])
    return photons

@pytest.fixture
def mock_events():
    """Mock an EmCDSEventDataInSCFrameInterface with predictable values."""
    events = MagicMock(spec=EmCDSEventDataInSCFrameFromArrays)
    events.scattered_lon_rad_sc = np.array([0.1, 0.2, 0.3])
    events.scattered_lat_rad_sc = np.array([0.0, np.pi/4, -np.pi/4])
    events.scattering_angle_rad = np.array([0.5, 1.0, 1.5])
    events.energy_keV = np.array([100.0, 500.0, 1000.0])
    return events


class TestUnpolarizedNFFarFieldInstrumentResponseFunction:

    def test_initialization_unpolarized(self, mock_unpolarized_response):
        """Ensure initialization succeeds when the response is unpolarized."""
        irf = UnpolarizedNFFarFieldInstrumentResponseFunction(mock_unpolarized_response)
        assert irf._response == mock_unpolarized_response

    def test_initialization_polarized_raises_error(self, mock_polarized_response):
        """Ensure initialization raises a ValueError if the response is polarized."""
        with pytest.raises(ValueError, match="only supports unpolarized responses"):
            UnpolarizedNFFarFieldInstrumentResponseFunction(mock_polarized_response)

    def test_pool_delegation(self, mock_unpolarized_response):
        """Test that pool management methods cleanly delegate to the underlying NFResponse."""
        irf = UnpolarizedNFFarFieldInstrumentResponseFunction(mock_unpolarized_response)

        irf.init_compute_pool(['cpu', 'cuda:0'])
        mock_unpolarized_response.init_compute_pool.assert_called_once_with(['cpu', 'cuda:0'])

        irf.shutdown_compute_pool()
        mock_unpolarized_response.shutdown_compute_pool.assert_called_once()

        mock_unpolarized_response.active_pool = True
        assert irf.active_pool is True

    def test_get_context_lat_colat_conversion(self, mock_photons):
        """
        Test that _get_context correctly extracts arrays, constructs the tensor,
        and converts latitude to colatitude (colat = -lat + pi/2).
        """
        expected_lon = [0.0, np.pi/2, np.pi]
        expected_colat = [np.pi/2, np.pi/4, np.pi]
        expected_en = [100.0, 500.0, 1000.0]
        
        expected_tensor = torch.tensor(
            list(zip(expected_lon, expected_colat, expected_en)), dtype=torch.float32
        )

        context_tensor = UnpolarizedNFFarFieldInstrumentResponseFunction._get_context(mock_photons)

        assert context_tensor.shape == (3, 3)
        torch.testing.assert_close(context_tensor, expected_tensor)

    def test_get_source_lat_colat_conversion(self, mock_events):
        """
        Test that _get_source correctly extracts arrays, constructs the tensor,
        and converts latitude to colatitude (colat = -lat + pi/2).
        """

        expected_en = [100.0, 500.0, 1000.0]
        expected_phi = [0.5, 1.0, 1.5]
        expected_lon = [0.1, 0.2, 0.3]
        expected_colat = [np.pi/2, np.pi/4, 3*np.pi/4]

        expected_tensor = torch.tensor(
            list(zip(expected_en, expected_phi, expected_lon, expected_colat)), dtype=torch.float32
        )

        source_tensor = UnpolarizedNFFarFieldInstrumentResponseFunction._get_source(mock_events)

        assert source_tensor.shape == (3, 4)
        torch.testing.assert_close(source_tensor, expected_tensor)

    def test_effective_area_cm2(self, mock_unpolarized_response, mock_photons):
        """Test effective area evaluation and output formatting."""
        irf = UnpolarizedNFFarFieldInstrumentResponseFunction(mock_unpolarized_response)
        
        mock_output = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float32)
        mock_unpolarized_response.evaluate_effective_area.return_value = mock_output
        
        eff_area = irf._effective_area_cm2(mock_photons)
        
        args, _ = mock_unpolarized_response.evaluate_effective_area.call_args
        assert args[0].shape == (3, 3)
        
        assert isinstance(eff_area, np.ndarray)
        np.testing.assert_array_equal(eff_area, np.array([10.0, 20.0, 30.0]))

    def test_event_probability(self, mock_unpolarized_response, mock_photons, mock_events):
        """Test density evaluation orchestration with context and source tensors."""
        irf = UnpolarizedNFFarFieldInstrumentResponseFunction(mock_unpolarized_response)
        
        mock_output = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float32)
        mock_unpolarized_response.evaluate_density.return_value = mock_output
        
        probs = irf._event_probability(mock_photons, mock_events)
        
        args, _ = mock_unpolarized_response.evaluate_density.call_args
        assert args[0].shape == (3, 3)
        assert args[1].shape == (3, 4)
        
        assert isinstance(probs, np.ndarray)
        np.testing.assert_allclose(probs, np.array([0.1, 0.5, 0.9]), atol=1e-7)

    def test_random_events_colat_to_lat_recovery(self, mock_unpolarized_response, mock_photons):
        """
        Test that sampling returns EmCDSEventDataInSCFrameFromArrays
        and correctly converts colatitude back to latitude.
        """
        irf = UnpolarizedNFFarFieldInstrumentResponseFunction(mock_unpolarized_response)
        
        mock_samples = torch.tensor([
            [100.0, 0.5, 0.1, np.pi/2],
            [500.0, 1.0, 0.2, np.pi/4],
            [1000.0, 1.5, 0.3, np.pi]
        ], dtype=torch.float32)
        
        mock_unpolarized_response.sample_density.return_value = mock_samples

        events = irf._random_events(mock_photons)

        mock_unpolarized_response.sample_density.assert_called_once()
        
        assert isinstance(events, EmCDSEventDataInSCFrameFromArrays)
        
        np.testing.assert_allclose(events.energy_keV, [100.0, 500.0, 1000.0], atol=1e-7)
        np.testing.assert_allclose(events.scattering_angle_rad, [0.5, 1.0, 1.5], atol=1e-7)
        np.testing.assert_allclose(events.scattered_lon_rad_sc, [0.1, 0.2, 0.3], atol=1e-7)
        np.testing.assert_allclose(events.scattered_lat_rad_sc, [0.0, np.pi/4, -np.pi/2], atol=1e-7)