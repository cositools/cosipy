import pytest
import cosipy

if not cosipy.with_ml:
    pytest.skip(reason="Optional [ml] dependencies not installed", allow_module_level=True) 

from unittest.mock import MagicMock, patch
import numpy as np
import torch
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u
from astromodels import PointSource
from cosipy import test_data
from pathlib import Path

from cosipy.threeml.ml.optimized_unbinned_folding import UnbinnedThreeMLPointSourceResponseIRFAdaptive
from cosipy.spacecraftfile import SpacecraftHistory
from cosipy.interfaces.instrument_response_interface import FarFieldSpectralInstrumentResponseFunctionInterface
from cosipy.interfaces.data_interface import TimeTagEmCDSEventDataInSCFrameInterface
from cosipy.interfaces.event import TimeTagEmCDSEventInSCFrameInterface
from cosipy.response.ml.nf_instrument_response_function import UnpolarizedNFFarFieldInstrumentResponseFunction

@pytest.fixture
def mock_data():
    """Mocks TimeTagEmCDSEventDataInSCFrameInterface"""
    data = MagicMock(spec=TimeTagEmCDSEventDataInSCFrameInterface)
    data.time.utc.unix = np.array([1000.0, 1001.0, 1002.0])
    data.nevents = 3
    data.energy_keV = np.array([500.0, 1000.0, 2000.0])
    data.scattering_angle_rad = np.array([0.1, 0.2, 0.3])
    data.scattered_lon_rad_sc = np.array([0.5, 0.6, 0.7])
    data.scattered_lat_rad_sc = np.array([0.1, 0.2, 0.3])
    return data

@pytest.fixture
def mock_sc_history():
    """Mocks SpacecraftHistory"""
    sc = MagicMock(spec=SpacecraftHistory)
    sc.obstime = Time([1000.0, 1001.0, 1002.0], format='unix')
    sc.livetime = [1.0, 1.0, 1.0] * u.s
    sc.intervals_duration = [1.0, 1.0, 1.0] * u.s
    
    mock_interp = MagicMock()
    mock_interp.location.spherical.distance.km = 7000.0 
    mock_interp.earth_zenith = SkyCoord(0, 0, unit='deg')
    sc.interp.return_value = mock_interp
    return sc

@pytest.fixture
def mock_irf():
    """Mocks FarFieldSpectralInstrumentResponseFunctionInterface"""
    irf = MagicMock(spec=FarFieldSpectralInstrumentResponseFunctionInterface)
    irf.effective_area_cm2.return_value = np.ones(3)
    irf.event_probability.return_value = np.ones(3)
    irf.active_pool = True 
    return irf

@pytest.fixture
def mock_source():
    """Mocks astromodels PointSource"""
    source = MagicMock(spec=PointSource)
    
    mock_position = MagicMock()
    mock_position.sky_coord = SkyCoord(10, 10, unit='deg')
    
    source.position = mock_position
    
    source.to_dict.return_value = {"mock": "source_dict"}
    source.side_effect = lambda energies: np.ones_like(energies)
    return source

@pytest.fixture
def irf_adaptive(mock_data, mock_irf, mock_sc_history):
    return UnbinnedThreeMLPointSourceResponseIRFAdaptive(
        data=mock_data,
        irf=mock_irf,
        sc_history=mock_sc_history,
        show_progress=False,
        force_energy_node_caching=False,
        reduce_memory=True
    )


class TestUnbinnedThreeMLPointSourceResponseIRFAdaptive:

    def test_initialization_and_properties(self, irf_adaptive):
        """Test getters and setters, and ensure valid assignments don't raise errors."""
        irf_adaptive.force_energy_node_caching = True
        assert irf_adaptive.force_energy_node_caching is True

        irf_adaptive.show_progress = True
        assert irf_adaptive.show_progress is True

        irf_adaptive.reduce_memory = False
        assert irf_adaptive.reduce_memory is False

        irf_adaptive.set_integration_parameters(
            total_energy_nodes=(100, 500),
            peak_nodes=(20, 15),
            peak_widths=(0.05, 0.15),
            energy_range=(200., 8000.),
            cache_batch_size=2000000,
            integration_batch_size=2000001,
            offset=1e-11
        )
        assert irf_adaptive.total_energy_nodes == (100, 500)
        assert irf_adaptive.energy_range == (200., 8000.)
        assert irf_adaptive.event_type == TimeTagEmCDSEventInSCFrameInterface
        assert irf_adaptive.peak_nodes == (20, 15)
        irf_adaptive.peak_widths = (0.15, 0.15)
        assert irf_adaptive.peak_widths == (0.15, 0.15)
        assert irf_adaptive.cache_batch_size == 2000000
        assert irf_adaptive.integration_batch_size == 2000001
        assert irf_adaptive.offset == 1e-11

    def test_property_type_errors(self, irf_adaptive):
        """Test that invalid types trigger ValueError on setters."""
        with pytest.raises(ValueError, match="must be a boolean"):
            irf_adaptive.force_energy_node_caching = "True"
            
        with pytest.raises(ValueError, match="must be a boolean"):
            irf_adaptive.show_progress = "Yes"
            
        with pytest.raises(ValueError, match="must be a boolean"):
            irf_adaptive.reduce_memory = 1

    def test_integration_parameter_value_errors(self, irf_adaptive):
        """Test the validation logic in set_integration_parameters."""
        
        with pytest.raises(ValueError, match="Too many nodes per peak"):
            irf_adaptive.total_energy_nodes = (10, 500) 

        with pytest.raises(ValueError, match="must be at least 1"):
            irf_adaptive.peak_nodes = (0, 10)

        with pytest.raises(ValueError, match="needs to be increasing"):
            irf_adaptive.energy_range = (10000., 100.)

        with pytest.raises(ValueError, match="cannot be smaller than"):
            irf_adaptive.cache_batch_size = 1

        with pytest.raises(ValueError, match="cannot be smaller than"):
            irf_adaptive.integration_batch_size = 1

        with pytest.raises(ValueError, match="cannot be negative"):
            irf_adaptive.offset = -2.0

    def test_memory_savings_warning(self, irf_adaptive, caplog):
        """Test _check_memory_savings triggers a warning if inefficient."""
        irf_adaptive._n_events = 10 
        irf_adaptive._total_energy_nodes = (60, 500)

        irf_adaptive._integration_batch_size = 400 
        irf_adaptive.reduce_memory = True
        
        assert "reduce_memory will increase the memory usage" in caplog.text
        
    def test_memory_reduction(self, irf_adaptive):
        """Test reduce_memory functionality."""
        irf_adaptive.reduce_memory = True
        irf_adaptive._irf_cache = torch.ones((5, 5), dtype=torch.float32)
        irf_adaptive._irf_energy_node_cache = np.ones((5, 5), dtype=np.float32)
        
        irf_adaptive.reduce_memory = False
        assert irf_adaptive._irf_cache.dtype == torch.float64
        assert irf_adaptive._irf_energy_node_cache.dtype == np.float64
        
        irf_adaptive.reduce_memory = True
        assert irf_adaptive._irf_cache.dtype == torch.float32
        assert irf_adaptive._irf_energy_node_cache.dtype == np.float32
        
        irf_adaptive._irf_cache = None
        
        irf_adaptive.reduce_memory = False
        assert irf_adaptive._irf_energy_node_cache.dtype == np.float64
        
        irf_adaptive.reduce_memory = True
        assert irf_adaptive._irf_energy_node_cache.dtype == np.float32
        
        irf_adaptive._irf_cache = torch.ones((5, 5), dtype=torch.float32)
        irf_adaptive._irf_energy_node_cache = None
        
        irf_adaptive.reduce_memory = False
        assert irf_adaptive._irf_cache.dtype == torch.float64
        
        irf_adaptive.reduce_memory = True
        assert irf_adaptive._irf_cache.dtype == torch.float32

    def test_node_building_math(self, irf_adaptive):
        """Test the static methods for building and scaling nodes."""
        n, w = irf_adaptive._build_nodes(5)
        assert n.shape == (1, 5)
        assert w.shape == (1, 5)

        split = irf_adaptive._build_split_nodes(10, 3)
        assert len(split) == 3
        assert split[0][0].shape[1] == 4
        assert split[1][0].shape[1] == 3
        
        E1 = torch.tensor([100.0])
        E2 = torch.tensor([200.0])
        EC = torch.tensor([150.0])
        nodes_u = torch.tensor([[-1.0, 0.0, 1.0]])
        weights_u = torch.tensor([[1.0, 1.0, 1.0]])


        out_n, out_w = irf_adaptive._scale_nodes_exp(E1, E2, nodes_u, weights_u)
        assert out_n.shape == (1, 3)
        assert torch.all(out_n <= E2)
        assert torch.all(out_n >= E1)


        out_n, out_w = irf_adaptive._scale_nodes_center(E1, E2, EC, nodes_u, weights_u)
        assert out_n.shape == (1, 3)
        assert np.isclose(out_n[0, 1], EC)
        assert torch.all(out_n <= E2)
        assert torch.all(out_n >= E1)

    def test_init_node_pool(self, irf_adaptive):
        """
        Test that node pools are initialized with correct dimensions and 
        background nodes are calculated correctly based on peak node allocations.
        """

        irf_adaptive._peak_widths = [0.1, 0.2]
        irf_adaptive._peak_nodes = [10, 5]
        irf_adaptive._total_energy_nodes = (50, 0) 
        
        with patch.object(irf_adaptive, '_build_nodes', side_effect=lambda n: (torch.zeros((1, n)), torch.ones((1, n)))) as mock_build, \
             patch.object(irf_adaptive, '_build_split_nodes', side_effect=lambda n, s: [(torch.zeros((1, n//s)), torch.ones((1, n//s)))]*s) as mock_split:
            
            irf_adaptive._init_node_pool()
            
            expected_widths = torch.tensor([0.1, 0.1, 0.2, 0.2], dtype=torch.float32)
            torch.testing.assert_close(irf_adaptive._width_tensor, expected_widths)
            
            assert mock_build.call_args_list[0][0][0] == 10 

            assert mock_build.call_args_list[1][0][0] == 5

            assert mock_build.call_args_list[2][0][0] == 40
            
            mock_split.assert_any_call(35, 2)
            
            mock_split.assert_any_call(30, 3)

    def test_peak_calculations(self, irf_adaptive):
        """Test kinematic peak calculations."""
        energy_m_keV = torch.tensor([2000.0,])
        phi_rad = torch.tensor([0.5])
        phi_geo_rad = np.pi - phi_rad
        
        esc_peak = irf_adaptive._get_escape_peak(energy_m_keV, phi_rad)
        assert esc_peak.shape == (1,)
        
        miss_peak = irf_adaptive._get_missing_energy_peak(phi_geo_rad, energy_m_keV, phi_rad)
        assert miss_peak.shape == (1,)
        miss_peak = irf_adaptive._get_missing_energy_peak(phi_geo_rad, energy_m_keV, phi_rad, inverse=True)
        assert miss_peak.shape == (1,)
        
        energy_m_keV = torch.tensor([50.0, 15000.0])
        
        miss_peak = irf_adaptive._get_missing_energy_peak(phi_geo_rad, energy_m_keV, phi_rad)
        assert torch.all(torch.isnan(miss_peak))
        esc_peak = irf_adaptive._get_escape_peak(energy_m_keV, phi_rad)
        assert torch.all(torch.isnan(esc_peak))

    def test_cache_init_clearing_and_copy(self, irf_adaptive, mock_source):
        """Test clear_cache and copy functionality."""
        irf_adaptive.set_source(mock_source)
        irf_adaptive._irf_cache = torch.ones((10, 10))
        irf_adaptive._area_cache = np.ones(10)
        
        irf_adaptive.clear_cache()
        assert irf_adaptive._irf_cache is None
        assert irf_adaptive._area_cache is None

        irf_adaptive.set_source(mock_source)
        new_instance = irf_adaptive.copy()
        assert new_instance._source is None
        assert new_instance._irf_cache is None
        
        with patch.object(irf_adaptive, '_update_cache') as mock_update:
            irf_adaptive.init_cache()
            mock_update.assert_called_once()

    def test_set_source_type_error(self, irf_adaptive):
        """Ensures set_source rejects non-PointSource objects."""
        with pytest.raises(TypeError, match="Please provide a PointSource"):
            irf_adaptive.set_source("Not A Source")
            
    def test_cache_from_file_not_found(self, irf_adaptive):
        """Test loading a non-existent file raises the correct error."""
        with pytest.raises(FileNotFoundError):
            irf_adaptive.cache_from_file("non_existent_file_12345.h5")
    
    def test_file_io_cache_full_state(self, irf_adaptive):
        """Test cache_to_file and cache_from_file with all attributes fully populated."""
        data_path = test_data.path
        test_dir = data_path / "unbinned_cache_full_test"
        test_dir.mkdir(parents=True, exist_ok=True)
        cache_file = Path(str(test_dir / "full_cache.h5"))
        
        irf_adaptive._total_energy_nodes = (12, 12)
        irf_adaptive._peak_nodes = (2, 2)
        irf_adaptive._peak_widths = (0.5, 0.5)
        irf_adaptive._energy_range = (10.0, 1000.0)
        irf_adaptive._cache_batch_size = 500
        irf_adaptive._integration_batch_size = 1000
        irf_adaptive._show_progress = True
        irf_adaptive._force_energy_node_caching = False
        irf_adaptive._reduce_memory = True
        irf_adaptive._offset = 1.23

        irf_adaptive._irf_cache = torch.ones((5, 5), dtype=torch.float32)
        irf_adaptive._irf_energy_node_cache = np.ones((5, 5), dtype=np.float64) * 2.0
        irf_adaptive._area_cache = np.ones(5, dtype=np.float64) * 3.0
        irf_adaptive._area_energy_node_cache = np.ones(5, dtype=np.float64) * 4.0
        irf_adaptive._exp_events = 42.5
        irf_adaptive._exp_density = torch.ones(5, dtype=torch.float32) * 5.0

        irf_adaptive._last_convolved_source_dict_number = {"flux": 1.0, "index": -2.0}
        irf_adaptive._last_convolved_source_dict_density = {"flux": 2.0, "cutoff": 500}

        irf_adaptive._last_convolved_source_skycoord = SkyCoord(15.0, -20.0, unit='deg', frame='galactic')

        try:
            irf_adaptive.cache_to_file(cache_file)
            assert cache_file.exists()

            new_instance = UnbinnedThreeMLPointSourceResponseIRFAdaptive(
                data=irf_adaptive._data,
                irf=irf_adaptive._irf,
                sc_history=irf_adaptive._sc_ori
            )
            
            with patch.object(new_instance, '_init_node_pool') as mock_init_pool:
                new_instance.cache_from_file(cache_file)
                mock_init_pool.assert_called_once()

            assert new_instance._total_energy_nodes == (12, 12)
            assert new_instance._peak_nodes == (2, 2)
            assert new_instance._peak_widths == (0.5, 0.5)
            assert new_instance._energy_range == (10.0, 1000.0)
            assert new_instance._cache_batch_size == 500
            assert new_instance._integration_batch_size == 1000
            assert new_instance._show_progress is True
            assert new_instance._force_energy_node_caching is False
            assert new_instance._reduce_memory is True
            assert new_instance._offset == 1.23

            assert torch.equal(new_instance._irf_cache, irf_adaptive._irf_cache)
            assert np.array_equal(new_instance._irf_energy_node_cache, irf_adaptive._irf_energy_node_cache)
            assert np.array_equal(new_instance._area_cache, irf_adaptive._area_cache)
            assert np.array_equal(new_instance._area_energy_node_cache, irf_adaptive._area_energy_node_cache)
            assert new_instance._exp_events == 42.5
            assert torch.equal(new_instance._exp_density, irf_adaptive._exp_density)

            assert new_instance._last_convolved_source_dict_number == {"flux": 1.0, "index": -2.0}
            assert new_instance._last_convolved_source_dict_density == {"flux": 2.0, "cutoff": 500}

            sc = new_instance._last_convolved_source_skycoord
            assert sc.frame.name == 'galactic'
            assert np.isclose(sc.spherical.lon.deg, 15.0)
            assert np.isclose(sc.spherical.lat.deg, -20.0)
            
        finally:
            if cache_file.exists(): cache_file.unlink()
            if test_dir.exists(): test_dir.rmdir()
    
    def test_file_io_cache_none_values(self, irf_adaptive):
        """Test cache_to_file and cache_from_file when optional attributes are None."""
        data_path = test_data.path
        test_dir = data_path / "unbinned_cache_none_test"
        test_dir.mkdir(parents=True, exist_ok=True)
        cache_file = Path(str(test_dir / "none_cache.h5"))

        irf_adaptive._offset = None
        irf_adaptive._irf_cache = None
        irf_adaptive._irf_energy_node_cache = None
        irf_adaptive._area_cache = None
        irf_adaptive._area_energy_node_cache = None
        irf_adaptive._exp_events = None
        irf_adaptive._exp_density = None
        irf_adaptive._last_convolved_source_dict_number = None
        irf_adaptive._last_convolved_source_dict_density = None
        irf_adaptive._last_convolved_source_skycoord = None

        try:
            irf_adaptive.cache_to_file(cache_file)

            new_instance = UnbinnedThreeMLPointSourceResponseIRFAdaptive(
                data=irf_adaptive._data, irf=irf_adaptive._irf, sc_history=irf_adaptive._sc_ori
            )
            
            with patch.object(new_instance, '_init_node_pool') as mock_init_pool:
                new_instance.cache_from_file(cache_file)
                mock_init_pool.assert_not_called()

            assert new_instance._offset is None
            assert new_instance._irf_cache is None
            assert new_instance._last_convolved_source_skycoord is None
            
        finally:
            if cache_file.exists(): cache_file.unlink()
            if test_dir.exists(): test_dir.rmdir()
    
    def test_file_io_cache_skycoord_equinox(self, irf_adaptive):
        """Test saving and loading a SkyCoord that specifically requires an equinox value."""
        data_path = test_data.path
        test_dir = data_path / "unbinned_cache_equinox_test"
        test_dir.mkdir(parents=True, exist_ok=True)
        cache_file = Path(str(test_dir / "equinox_cache.h5"))
        
        irf_adaptive._last_convolved_source_skycoord = SkyCoord(
            10.0, 20.0, unit='deg', frame='fk5', equinox='J2000.0'
        )
        
        try:
            irf_adaptive.cache_to_file(cache_file)

            new_instance = UnbinnedThreeMLPointSourceResponseIRFAdaptive(
                data=irf_adaptive._data, irf=irf_adaptive._irf, sc_history=irf_adaptive._sc_ori
            )
            new_instance.cache_from_file(cache_file)

            sc = new_instance._last_convolved_source_skycoord
            assert sc.frame.name == 'fk5'
            assert sc.equinox.value == 'J2000.000'
            
        finally:
            if cache_file.exists(): cache_file.unlink()
            if test_dir.exists(): test_dir.rmdir()
            
    @patch('cosipy.threeml.ml.optimized_unbinned_folding.UnbinnedThreeMLPointSourceResponseIRFAdaptive._update_cache')
    def test_expected_counts(self, mock_update, irf_adaptive, mock_source):
        """Test expected_counts basic caching logic."""
        irf_adaptive.set_source(mock_source)
        irf_adaptive._area_cache = np.array([1.0, 2.0])
        irf_adaptive._area_energy_node_cache = np.array([100.0, 200.0])
        irf_adaptive._exp_events = None
        
        temp_count = mock_source.call_count
        counts = irf_adaptive.expected_counts()
        assert isinstance(counts, float)
        assert np.isclose(counts, 3.0)
        mock_update.assert_called()
        assert mock_source.call_count == temp_count + 1
        
        temp_count = mock_source.call_count
        counts = irf_adaptive.expected_counts()
        assert np.isclose(counts, 3.0)
        assert mock_source.call_count == temp_count
        
        temp_count = mock_source.call_count
        mock_source.to_dict.return_value = {"mock": "source_dict_2"}
        counts = irf_adaptive.expected_counts()
        assert np.isclose(counts, 3.0)
        assert mock_source.call_count == temp_count + 1

    @patch('cosipy.threeml.ml.optimized_unbinned_folding.UnbinnedThreeMLPointSourceResponseIRFAdaptive._update_cache')
    def test_expectation_density_all_branches(self, mock_update, irf_adaptive, mock_source):
        """Test expectation_density across all caching and batching branches."""
        irf_adaptive.set_source(mock_source)
        
        n_events = 3
        n_nodes = 6
        irf_adaptive._n_events = n_events
        irf_adaptive._total_energy_nodes = [n_nodes, n_nodes]
        irf_adaptive._irf_cache = torch.ones((n_events, n_nodes))
        
        irf_adaptive._integration_batch_size = n_nodes * n_events
        irf_adaptive._irf_energy_node_cache = np.ones((n_events, n_nodes))
        irf_adaptive._exp_density = None
        mock_source.to_dict.return_value = {"mock": "state1"}
        
        density = irf_adaptive.expectation_density()
        assert density.shape == (n_events,)
        assert np.allclose(density, [6, 6, 6])
        
        irf_adaptive._integration_batch_size = n_nodes * 2 
        irf_adaptive._exp_density = None
        mock_source.to_dict.return_value = {"mock": "state2"}
        
        density = irf_adaptive.expectation_density()
        assert density.shape == (n_events,)
        assert np.allclose(density, [6, 6, 6])
        
        irf_adaptive._irf_energy_node_cache = None
        irf_adaptive._exp_density = None
        mock_source.to_dict.return_value = {"mock": "state3"}
        
        irf_adaptive._sc_coord_sph_cache = SkyCoord(
            l=np.zeros(n_events)*u.deg, 
            b=np.zeros(n_events)*u.deg, 
            frame="galactic"
        ).spherical
        
        irf_adaptive._energy_m_keV = np.zeros(n_events)
        irf_adaptive._phi_rad = np.zeros(n_events)
        
        with patch.object(irf_adaptive, '_get_CDS_coordinates') as mock_cds, \
             patch.object(irf_adaptive, '_get_nodes') as mock_get_nodes:
            
            mock_cds.return_value = (np.zeros(n_events), np.zeros(n_events))
            
            mock_get_nodes.side_effect = lambda e, p, pg, pig: (
                np.ones((len(e), n_nodes)), 
                np.ones((len(e), n_nodes))
            )
            
            density = irf_adaptive.expectation_density()
            
            assert density.shape == (n_events,)
            assert np.allclose(density, [6, 6, 6])
            assert mock_get_nodes.call_count == 2 
            
        irf_adaptive._offset = 1.0
        irf_adaptive._exp_density = None
        mock_source.to_dict.return_value = {"mock": "state4"}
        irf_adaptive._irf_energy_node_cache = np.ones((n_events, n_nodes))
        irf_adaptive._integration_batch_size = n_nodes * n_events
        
        density_with_offset = irf_adaptive.expectation_density()
        assert np.allclose(density_with_offset, [7.0, 7.0, 7.0])
        
        irf_adaptive._offset = None
        density_without_offset = irf_adaptive.expectation_density()
        assert np.allclose(density_without_offset, [6.0, 6.0, 6.0])

    def test_earth_occ(self, irf_adaptive, mock_sc_history):
        """Test Earth occultation static logic based on distance and zenith."""
        
        visible_src = SkyCoord(0, 0, unit='deg')
        occ_src = SkyCoord(115, 0, unit='deg')  
        
        mock_ori = mock_sc_history.interp(None)
        
        occ_index_visible = irf_adaptive._earth_occ(visible_src, mock_ori)
        occ_index_occ = irf_adaptive._earth_occ(occ_src, mock_ori)
        
        assert occ_index_visible == 1.0
        assert occ_index_occ == 0.0

    def test_get_cds_coordinates(self, irf_adaptive):
        """Test the transformation from Sky coordinates to Compton Data Space (CDS)."""
        lon_src = torch.tensor([0.0, np.pi/2, 0.7])
        lat_src = torch.tensor([0.1, 0.0, 0.3])
        
        phi_geo, phi_igeo = irf_adaptive._get_CDS_coordinates(lon_src, lat_src)
        
        assert phi_geo.shape == lon_src.shape
        assert phi_igeo.shape == lon_src.shape
        assert np.isclose(phi_geo[2], 0.0)
        assert torch.allclose(phi_geo + phi_igeo, torch.tensor(np.pi))
        assert not torch.any(torch.isnan(phi_geo))
        
    def test_get_target_in_sc_frame_logic(self, irf_adaptive):
        """Test the math/logic of sky-to-spacecraft coordinate transformation."""
        source_coord = SkyCoord([0,], [0,], unit='deg', frame='icrs')
        
        mock_ori = MagicMock()
        mock_ori.attitude.frame = 'icrs'
        mock_ori.attitude.rot.inv().as_matrix.return_value = np.eye(3)
        
        result = irf_adaptive._get_target_in_sc_frame(source_coord, mock_ori)
        
        assert isinstance(result, SkyCoord)
        assert result.representation_type.name == 'spherical'
        
        assert np.isclose(result.lon.deg, 0.0)
        assert np.isclose(result.lat.deg, 0.0)
        
    def test_update_cache_runtime_error(self, irf_adaptive):
        """Ensure error is raised if source is not set."""
        irf_adaptive._source = None
        with pytest.raises(RuntimeError, match="Call set_source"):
            irf_adaptive._update_cache()   
        
    def test_update_cache_recalculation_logic(self, irf_adaptive, mock_source):
        """Verify the branching logic for no/area/pdf recalculation."""
        irf_adaptive.set_source(mock_source)
        
        irf_adaptive._sc_coord_sph_cache = None
        irf_adaptive._area_cache = None
        irf_adaptive._irf_cache = None
        irf_adaptive._last_convolved_source_skycoord = None
        
        with patch.object(irf_adaptive, '_compute_area') as mock_area, \
             patch.object(irf_adaptive, '_compute_density') as mock_dens, \
             patch.object(irf_adaptive, '_init_node_pool') as mock_init, \
             patch.object(irf_adaptive, '_get_target_in_sc_frame') as mock_trans:
            
            mock_trans.return_value = SkyCoord(np.zeros(10), np.zeros(10), unit='deg').spherical
            irf_adaptive._inv_idx = slice(None) 
            
            irf_adaptive._update_cache()
            
            mock_area.assert_called_once()
            mock_dens.assert_called_once()
            mock_init.assert_called_once()
            assert irf_adaptive._last_convolved_source_skycoord == mock_source.position.sky_coord
            
            mock_area.reset_mock()
            mock_dens.reset_mock()
            mock_init.reset_mock()

            irf_adaptive._area_cache = np.ones(1)
            irf_adaptive._irf_cache = torch.ones(1)

            irf_adaptive._update_cache()
            mock_area.assert_not_called()
            mock_dens.assert_not_called()
            
            irf_adaptive._area_cache = None
            irf_adaptive._update_cache()
            mock_area.assert_called_once()
            mock_dens.assert_not_called() 
            
            mock_area.reset_mock()
            mock_dens.reset_mock()
            mock_init.reset_mock()
            irf_adaptive._area_cache = np.ones(1)
            irf_adaptive._irf_cache = torch.ones(1)
            
            irf_adaptive._irf_cache = None
            irf_adaptive._update_cache()
            mock_dens.assert_called_once()
            mock_area.assert_not_called()
            
            mock_area.reset_mock()
            mock_dens.reset_mock()
            mock_init.reset_mock()
            irf_adaptive._area_cache = np.ones(1)
            irf_adaptive._irf_cache = torch.ones(1)

            new_coord = SkyCoord(11, 10, unit='deg', frame='icrs')
            mock_source.position.sky_coord = new_coord
            irf_adaptive._irf_energy_node_cache = np.ones(1)
            
            irf_adaptive._update_cache()
            
            mock_area.assert_called_once()
            mock_dens.assert_called_once()
            assert irf_adaptive._irf_energy_node_cache is None
            assert irf_adaptive._last_convolved_source_skycoord == new_coord
            
            mock_irf = MagicMock(spec=UnpolarizedNFFarFieldInstrumentResponseFunction)
            mock_irf.active_pool = False 
            
            old_irf = irf_adaptive._irf
            irf_adaptive._irf = mock_irf
            
            final_coord = SkyCoord(20, 20, unit='deg', frame='icrs')
            mock_source.position.sky_coord = final_coord
            
            irf_adaptive._update_cache()
            
            mock_irf.init_compute_pool.assert_called_once()
            mock_irf.shutdown_compute_pool.assert_called_once()
            
            mock_irf = MagicMock(spec=UnpolarizedNFFarFieldInstrumentResponseFunction)
            mock_irf.active_pool = True
            
            old_irf = irf_adaptive._irf
            irf_adaptive._irf = mock_irf
            
            final_coord = SkyCoord(20, 21, unit='deg', frame='icrs')
            mock_source.position.sky_coord = final_coord
            
            irf_adaptive._update_cache()
            
            mock_irf.init_compute_pool.assert_not_called()
            mock_irf.shutdown_compute_pool.assert_not_called()
            
            irf_adaptive._irf = old_irf
            
            irf_adaptive.force_energy_node_caching = True
            with patch.object(irf_adaptive, '_compute_nodes') as mock_nodes:
                irf_adaptive._irf_energy_node_cache = np.array([1.0])
                irf_adaptive._update_cache()
                mock_nodes.assert_not_called()
                irf_adaptive._irf_energy_node_cache = None
                irf_adaptive._update_cache()
                mock_nodes.assert_called_once()
        
    def test_compute_area_progress_bar_toggle(self, irf_adaptive, mock_source):
        """Ensure tqdm respects the show_progress attribute."""
        irf_adaptive.set_source(mock_source)
        irf_adaptive._sc_ori = MagicMock()
        irf_adaptive._sc_ori.livetime.to_value.return_value = np.ones(1)
        irf_adaptive._get_target_in_sc_frame = MagicMock(return_value=SkyCoord(l=[0,], b=[0,], unit='deg', frame='galactic').spherical)
        irf_adaptive._earth_occ = MagicMock(return_value=np.ones(1))
        irf_adaptive._irf.effective_area_cm2 = MagicMock(return_value=np.ones(10))

        with patch("cosipy.threeml.ml.optimized_unbinned_folding.tqdm") as mock_tqdm:

            irf_adaptive.show_progress = False
            irf_adaptive._compute_area()

            args, kwargs = mock_tqdm.call_args
            assert kwargs['disable'] is True
            

            irf_adaptive.show_progress = True
            irf_adaptive._compute_area()
            args, kwargs = mock_tqdm.call_args
            assert kwargs['disable'] is False
        
    def test_compute_area_earth_occultation(self, irf_adaptive, mock_source):
        """Verify that Earth occultation (0 weights) correctly zeroes out area contributions."""
        irf_adaptive.set_source(mock_source)
        irf_adaptive._total_energy_nodes = (5, 2)
        irf_adaptive._energy_range = (10, 100)
        irf_adaptive._cache_batch_size = 100
        
        n_time = 5
        irf_adaptive._sc_ori = MagicMock()
        irf_adaptive._sc_ori.livetime.to_value.return_value = np.ones(n_time)
        
        mock_sc_coords = SkyCoord(np.zeros(n_time), np.zeros(n_time), unit='deg', frame='galactic').spherical
        irf_adaptive._get_target_in_sc_frame = MagicMock(return_value=mock_sc_coords)
        
        irf_adaptive._earth_occ = MagicMock(return_value=np.zeros(n_time))
        
        irf_adaptive._irf.effective_area_cm2 = MagicMock(return_value=np.ones(10) * 100.0)

        irf_adaptive._compute_area()

        assert np.all(np.isclose(irf_adaptive._area_cache, 0.0))
    
    def test_compute_area_logic_and_batching(self, irf_adaptive, mock_source):
        """Test the quadrature setup, batching buffers, and integration in _compute_area."""
        
        n_energy = 4 
        n_time = 10   
        cache_batch_size = 8 
        
        irf_adaptive._total_energy_nodes = (10, n_energy)
        irf_adaptive._energy_range = (10.0, 100.0)
        irf_adaptive._cache_batch_size = cache_batch_size
        irf_adaptive.set_source(mock_source)
        
        irf_adaptive._sc_ori = MagicMock()
        irf_adaptive._sc_ori.livetime.to_value.return_value = np.ones(n_time)

        mock_sc_coords = SkyCoord(np.linspace(0, 10, n_time), np.zeros(n_time), unit='deg', frame='galactic').spherical 
        irf_adaptive._get_target_in_sc_frame = MagicMock(return_value=mock_sc_coords)
        
        irf_adaptive._earth_occ = MagicMock(return_value=np.ones(n_time))

        def side_effect(photons):
            return np.ones(len(photons.energy)) * 50.0
        
        irf_adaptive._irf.effective_area_cm2 = MagicMock(side_effect=side_effect)

        irf_adaptive._compute_area()

        assert len(irf_adaptive._area_energy_node_cache) == n_energy
        assert np.all(irf_adaptive._area_energy_node_cache >= 10.0)
        assert np.all(irf_adaptive._area_energy_node_cache <= 100.0)

        assert irf_adaptive._irf.effective_area_cm2.call_count == 5
        
        assert isinstance(irf_adaptive._area_cache, np.ndarray)
        assert irf_adaptive._area_cache.shape == (n_energy,)

        assert np.all(irf_adaptive._area_cache > 0)

    def test_compute_density_logic_and_batching(self, irf_adaptive, mock_source):
        """Test the event batching, buffer management, and IRF integration in _compute_density."""
        
        n_energy = 2   
        n_events = 6   
        batch_size = 4 
        
        irf_adaptive._total_energy_nodes = (n_energy, 5)
        irf_adaptive._n_events = n_events
        irf_adaptive._cache_batch_size = batch_size
        irf_adaptive._reduce_memory = True
        irf_adaptive._force_energy_node_caching = True
        irf_adaptive.set_source(mock_source)
        
        irf_adaptive._energy_m_keV = np.array([100, 200, 300, 400, 500, 600])
        irf_adaptive._phi_rad = np.zeros(n_events)
        irf_adaptive._lon_scatt = np.zeros(n_events)
        irf_adaptive._lat_scatt = np.zeros(n_events)
        irf_adaptive._livetime_ratio = np.ones(n_events)
        irf_adaptive._inv_idx = np.arange(n_events)
        
        irf_adaptive._sc_coord_sph_cache = SkyCoord(np.zeros(n_events), np.zeros(n_events), unit='deg', frame="galactic").spherical
        irf_adaptive._get_CDS_coordinates = MagicMock(return_value=(np.zeros(n_events), np.zeros(n_events)))
        irf_adaptive._earth_occ = MagicMock(return_value=np.ones(n_events))

        def mock_get_nodes(e, p, pg, pig):
            curr_n = len(e)
            return np.ones((curr_n, n_energy)) * 50.0, torch.ones((curr_n, n_energy))
        
        irf_adaptive._get_nodes = MagicMock(side_effect=mock_get_nodes)
        
        irf_adaptive._irf.effective_area_cm2 = MagicMock(return_value=np.ones(batch_size))
        irf_adaptive._irf.event_probability = MagicMock(return_value=np.ones(batch_size))

        irf_adaptive._compute_density()

        assert irf_adaptive._irf_cache.shape == (n_events, n_energy)
        assert irf_adaptive._irf_cache.dtype == torch.float32
        
        assert irf_adaptive._irf_energy_node_cache.shape == (n_events, n_energy)
        assert np.all(irf_adaptive._irf_energy_node_cache == 50.0)

        assert irf_adaptive._irf.effective_area_cm2.call_count == 3
        assert irf_adaptive._irf.event_probability.call_count == 3

    def test_compute_density_memory_modes(self, irf_adaptive, mock_source):
        """Verify that reduce_memory correctly toggles between float32 and float64."""
        irf_adaptive.set_source(mock_source)
        irf_adaptive._n_events = 2
        irf_adaptive._total_energy_nodes = (2, 2)
        irf_adaptive._cache_batch_size = 100
        irf_adaptive._energy_m_keV = np.array([100, 200])
        irf_adaptive._phi_rad = np.zeros(2)
        irf_adaptive._livetime_ratio = np.ones(2)
        irf_adaptive._inv_idx = np.arange(2)
        irf_adaptive._sc_coord_sph_cache = SkyCoord([0, 0], [0, 0], unit='deg', frame="galactic").spherical
        
        irf_adaptive._get_CDS_coordinates = MagicMock(return_value=(np.zeros(2), np.zeros(2)))
        irf_adaptive._get_nodes = MagicMock(return_value=(np.ones((2,2)), torch.ones((2,2))))
        irf_adaptive._earth_occ = MagicMock(return_value=np.ones(2))
        irf_adaptive._irf.effective_area_cm2 = MagicMock(return_value=np.ones(4))
        irf_adaptive._irf.event_probability = MagicMock(return_value=np.ones(4))

        irf_adaptive._reduce_memory = True
        irf_adaptive._compute_density()
        assert irf_adaptive._irf_cache.dtype == torch.float32

        irf_adaptive._reduce_memory = False
        irf_adaptive._compute_density()
        assert irf_adaptive._irf_cache.dtype == torch.float64

    def test_compute_density_occultation_masking(self, irf_adaptive, mock_source):
        """Ensure that the Earth occultation index correctly zero-shields the density cache."""
        irf_adaptive.set_source(mock_source)
        irf_adaptive._n_events = 1
        irf_adaptive._total_energy_nodes = (1, 1)
        irf_adaptive._cache_batch_size = 10
        irf_adaptive._energy_m_keV = np.array([100])
        irf_adaptive._phi_rad = np.zeros(1)
        irf_adaptive._livetime_ratio = np.ones(1)
        irf_adaptive._inv_idx = np.array([0])
        irf_adaptive._sc_coord_sph_cache = SkyCoord([0,], [0,], unit='deg', frame="galactic").spherical
        
        irf_adaptive._get_CDS_coordinates = MagicMock(return_value=(np.zeros(1), np.zeros(1)))
        irf_adaptive._get_nodes = MagicMock(return_value=(np.ones((1,1)), torch.ones((1,1))))
        irf_adaptive._irf.effective_area_cm2 = MagicMock(return_value=np.ones(1))
        irf_adaptive._irf.event_probability = MagicMock(return_value=np.ones(1))
        
        irf_adaptive._earth_occ = MagicMock(return_value=np.array([0.0]))

        irf_adaptive._compute_density()
        
        assert torch.all(irf_adaptive._irf_cache == 0.0)

    def test_compute_nodes_logic(self, irf_adaptive):
        """Test that _compute_nodes correctly populates the energy node cache."""
        
        n_events = 5
        irf_adaptive._n_events = n_events
        irf_adaptive._energy_m_keV = np.linspace(100, 500, n_events)
        irf_adaptive._phi_rad = np.zeros(n_events)
        irf_adaptive._reduce_memory = True
        
        irf_adaptive._sc_coord_sph_cache = SkyCoord(
            np.zeros(n_events), np.zeros(n_events), unit='deg', frame="galactic"
        ).spherical

        mock_cds = (torch.zeros(n_events), torch.zeros(n_events))

        n_energy_nodes = 3
        mock_nodes = np.ones((n_events, n_energy_nodes)) * 42.0
        mock_weights = np.ones((n_events, n_energy_nodes))
        
        with patch.object(irf_adaptive, '_get_CDS_coordinates', return_value=mock_cds) as mock_get_cds, \
             patch.object(irf_adaptive, '_get_nodes', return_value=(mock_nodes, mock_weights)) as mock_get_nodes:
            
            irf_adaptive._compute_nodes()
            
            mock_get_cds.assert_called_once()
            mock_get_nodes.assert_called_once()
            
            assert irf_adaptive._irf_energy_node_cache is not None
            assert irf_adaptive._irf_energy_node_cache.shape == (n_events, n_energy_nodes)
            np.testing.assert_array_equal(irf_adaptive._irf_energy_node_cache, 42.0)
            
            assert irf_adaptive._irf_energy_node_cache.dtype == np.float32

    def test_compute_nodes_precision_modes(self, irf_adaptive):
        """Verify that the node cache respects the reduce_memory setting."""
        n_events = 2
        irf_adaptive._n_events = n_events
        irf_adaptive._energy_m_keV = np.array([100, 200])
        irf_adaptive._phi_rad = np.zeros(2)
        irf_adaptive._sc_coord_sph_cache = SkyCoord([0, 0], [0, 0], unit='deg', frame="galactic").spherical
        
        mock_nodes = np.array([[1.0], [2.0]])
        
        with patch.object(irf_adaptive, '_get_CDS_coordinates', return_value=(torch.zeros(2), torch.zeros(2))), \
             patch.object(irf_adaptive, '_get_nodes', return_value=(mock_nodes, None)):
                 
            irf_adaptive._reduce_memory = True
            irf_adaptive._compute_nodes()
            assert irf_adaptive._irf_energy_node_cache.dtype == np.float32
            
            irf_adaptive._reduce_memory = False
            irf_adaptive._compute_nodes()
            assert irf_adaptive._irf_energy_node_cache.dtype == np.float64

    def test_compute_nodes_empty_cache_error(self, irf_adaptive):
        """Test that the method fails if sc_coord_sph_cache is None."""
        irf_adaptive._sc_coord_sph_cache = None
        
        with pytest.raises(AttributeError):
            irf_adaptive._compute_nodes()

    def test_get_nodes_peak_branching(self, irf_adaptive):
        """Verify that _get_nodes correctly groups events by peak count (1, 2, or 3)."""
        
        n_events = 3
        n_nodes = 10
        irf_adaptive._total_energy_nodes = (n_nodes, 5)
        irf_adaptive._width_tensor = torch.tensor([1.0, 1.0, 1.0, 1.0])
        
        e = torch.tensor([100.0, 120.0, 140.0])
        p = torch.tensor([0.1, 0.1, 0.1])
        pg = torch.tensor([0.2, 0.2, 0.2])
        pig = torch.tensor([0.3, 0.3, 0.3])
        
        with patch.object(irf_adaptive, '_get_escape_peak') as mock_escape, \
             patch.object(irf_adaptive, '_get_missing_energy_peak') as mock_missing, \
             patch.object(irf_adaptive, '_fill_nodes') as mock_fill:
            
            mock_escape.return_value = torch.tensor([np.nan, 150.0, 250.0])
            
            mock_missing.side_effect = [
                torch.tensor([np.nan, np.nan, 200.0]), 
                torch.tensor([np.nan, np.nan, np.nan])  
            ]

            nodes, weights = irf_adaptive._get_nodes(e, p, pg, pig)

            assert mock_fill.call_count == 3
            
            assert nodes.shape == (n_events, n_nodes)
            assert weights.shape == (n_events, n_nodes)
            
            last_call_args = mock_fill.call_args_list[2] 
            passed_peaks = last_call_args[0][4] 
            
            expected_sorted_peaks = torch.tensor([[140.0, 200.0, 250.0]])
            torch.testing.assert_close(passed_peaks, expected_sorted_peaks)

    def test_get_nodes_single_event_consistency(self, irf_adaptive):
        """Ensure the view/squeeze logic handles a single event without crashing."""
        irf_adaptive._total_energy_nodes = (10, 5)
        irf_adaptive._width_tensor = torch.zeros(4)
        
        e = torch.tensor([511.0])
        p = torch.tensor([0.0])
        pg = torch.tensor([0.0])
        pig = torch.tensor([0.0])
        
        with patch.object(irf_adaptive, '_get_escape_peak', return_value=torch.tensor([np.nan])), \
             patch.object(irf_adaptive, '_get_missing_energy_peak', return_value=torch.tensor([np.nan])), \
             patch.object(irf_adaptive, '_fill_nodes'):
            
            nodes, weights = irf_adaptive._get_nodes(e, p, pg, pig)
            assert nodes.shape == (1, 10)
        
        with patch.object(irf_adaptive, '_get_escape_peak', return_value=torch.tensor([515.0])), \
             patch.object(irf_adaptive, '_get_missing_energy_peak', return_value=torch.tensor([np.nan])), \
             patch.object(irf_adaptive, '_fill_nodes'):
            
            nodes, weights = irf_adaptive._get_nodes(e, p, pg, pig)
            assert nodes.shape == (1, 10)
            
    def test_fill_nodes_index_assignment(self, irf_adaptive):
        """Ensure that the data is written to the correct slices of the output tensors."""
        p_nodes = (torch.zeros((1, 10)), torch.ones((1, 10))) 
        b_nodes = (torch.zeros((1, 5)), torch.ones((1, 5)))  
        
        irf_adaptive._nodes_primary = p_nodes
        irf_adaptive._nodes_bkg_1 = b_nodes
        irf_adaptive._energy_range = (100.0, 1000.0)
        
        nodes_out = torch.zeros((1, 15))
        weights_out = torch.zeros((1, 15))
        
        with patch.object(irf_adaptive, '_scale_nodes_center', return_value=(torch.ones((1, 10)), torch.ones((1, 10)))), \
             patch.object(irf_adaptive, '_scale_nodes_exp', return_value=(torch.full((1, 5), 2.0), torch.full((1, 5), 2.0))):
             
             irf_adaptive._fill_nodes(nodes_out, weights_out, torch.tensor([0]), 1, 
                                      torch.tensor([[500.0]]), torch.tensor([[50.0]]))
             
             assert torch.all(nodes_out[0, 0:10] == 1.0)
             assert torch.all(nodes_out[0, 10:15] == 2.0)
    
    def test_fill_nodes_logic_and_clamping(self, irf_adaptive):
        """
        Verify that _fill_nodes correctly calculates energy boundaries (clamping)
        and calls the scaling methods with the appropriate node pools for each mode.
        """
        p_nodes = (torch.zeros((1, 5)), torch.ones((1, 5))) 
        s_nodes = (torch.zeros((1, 4)), torch.ones((1, 4))) 
        b1_nodes = (torch.zeros((1, 15)), torch.ones((1, 15))) 
        b2_nodes = [(torch.zeros((1, 6)), torch.ones((1, 6))), 
                    (torch.zeros((1, 5)), torch.ones((1, 5)))] 
        b3_nodes = [(torch.zeros((1, 3)), torch.ones((1, 3))),
                    (torch.zeros((1, 2)), torch.ones((1, 2))),
                    (torch.zeros((1, 2)), torch.ones((1, 2)))] 

        irf_adaptive._nodes_primary = p_nodes
        irf_adaptive._nodes_secondary = s_nodes
        irf_adaptive._nodes_bkg_1 = b1_nodes
        irf_adaptive._nodes_bkg_2 = b2_nodes
        irf_adaptive._nodes_bkg_3 = b3_nodes
        
        irf_adaptive._energy_range = (100.0, 1000.0) 

        n_events = 3
        n_nodes_total = 20
        nodes_out = torch.zeros((n_events, n_nodes_total))
        weights_out = torch.zeros((n_events, n_nodes_total))

        with patch.object(irf_adaptive, '_scale_nodes_center') as mock_center, \
             patch.object(irf_adaptive, '_scale_nodes_exp') as mock_exp:
            
            mock_center.side_effect = lambda e1, e2, ec, n, w: (torch.ones_like(n), torch.ones_like(w))
            mock_exp.side_effect = lambda e1, e2, n, w: (torch.ones_like(n), torch.ones_like(w))

            idx1 = torch.tensor([0])
            peaks1 = torch.tensor([[110.0]])
            delta1 = torch.tensor([[20.0]])
            
            irf_adaptive._fill_nodes(nodes_out, weights_out, idx1, 1, peaks1, delta1)
            
            args_center = mock_center.call_args_list[0][0]
            assert float(args_center[1]) == 130.0 
            assert float(args_center[2]) == 110.0 
            
            args_exp = mock_exp.call_args_list[0][0]
            assert float(args_exp[0]) == 130.0 
            assert float(args_exp[1]) == 1000.0 

            mock_center.reset_mock()
            mock_exp.reset_mock()
            idx2 = torch.tensor([1])
            peaks2 = torch.tensor([[300.0, 400.0]])
            delta2 = torch.tensor([[60.0, 60.0]])
            
            irf_adaptive._fill_nodes(nodes_out, weights_out, idx2, 2, peaks2, delta2)
            
            assert mock_center.call_count == 2
            assert mock_exp.call_count == 2
            
            args_prim = mock_center.call_args_list[0][0]
            args_sec = mock_center.call_args_list[1][0]
            assert float(args_prim[1]) == 350.0 
            assert float(args_sec[0]) == 350.0 

            mock_center.reset_mock()
            mock_exp.reset_mock()
            idx3 = torch.tensor([2])
            peaks3 = torch.tensor([[200.0, 500.0, 800.0]])
            delta3 = torch.tensor([[10.0, 10.0, 10.0]])
            
            irf_adaptive._fill_nodes(nodes_out, weights_out, idx3, 3, peaks3, delta3)
            
            assert mock_center.call_count == 3
            assert mock_exp.call_count == 3
            
            args_last_exp = mock_exp.call_args_list[-1][0]
            assert float(args_last_exp[1]) == 1000.0
            
            with pytest.raises(ValueError, match="Unknown folding mode"):
                irf_adaptive._fill_nodes(nodes_out, weights_out, idx3, 4, peaks3, delta3)
            
        