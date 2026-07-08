from cosipy.threeml.custom_functions import PointSource
import numpy as np
import pytest
from unittest.mock import MagicMock, patch


# ==========================================
# FIXTURES & STUBS FOR EXTERNAL DEPENDENCIES
# ==========================================
@pytest.fixture(autouse=True)
def mock_external_dependencies():
    """Patches all global framework dependencies required by PointSource."""
    with patch('cosipy.threeml.custom_functions.log') as mock_log, \
         patch('cosipy.threeml.custom_functions.SkyDirection') as mock_sky_dir, \
         patch('cosipy.threeml.custom_functions.SpectralComponent') as mock_spectral_comp, \
         patch('cosipy.threeml.custom_functions.SourceType') as mock_src_type, \
         patch('cosipy.threeml.custom_functions.get_units') as mock_get_units, \
         patch('cosipy.threeml.custom_functions.use_astromodels_memoization') as mock_memo, \
         patch('cosipy.threeml.custom_functions.dict_to_list') as mock_dict_to_list, \
         patch('cosipy.threeml.custom_functions.sp_int') as mock_sp_int, \
         patch('cosipy.threeml.custom_functions.Source.__init__') as mock_source_init, \
         patch('cosipy.threeml.custom_functions.Node.__init__') as mock_node_init, \
         patch('cosipy.threeml.custom_functions.Node._add_child') as mock_add_child:
        
        # Configure default behavior for unit settings
        mock_units = MagicMock()
        mock_units.energy = "keV"
        mock_units.area = "cm2"
        mock_units.time = "s"
        mock_get_units.return_value = mock_units
        
        yield {
            "log": mock_log,
            "SkyDirection": mock_sky_dir,
            "SpectralComponent": mock_spectral_comp,
            "sp_int": mock_sp_int,
        }


@pytest.fixture
def dummy_shape():
    """Returns a mock 1D spectral shape function."""
    shape = MagicMock()
    shape.parameters = {}
    return shape


# ==========================================
# INITIALIZATION & VALIDATION TESTS
# ==========================================

def test_init_with_valid_ra_dec():
    """Verifies successful initialization using Right Ascension and Declination."""
    source = PointSource(
        source_name="test_source", 
        ra=125.6, 
        dec=-75.3, 
        spectral_shape=dummy_shape
    )
    
    # Check that SkyDirection was instantiated correctly with floats
    mock_external_dependencies["SkyDirection"].assert_called_once_with(ra=125.6, dec=-75.3)
    # Check that a default "main" spectral component was wrapped
    mock_external_dependencies["SpectralComponent"].assert_called_once_with("main", dummy_shape, None)


def test_init_with_valid_galactic(dummy_shape, mock_external_dependencies):
    """Verifies successful initialization using Galactic coordinates."""
    source = PointSource(
        source_name="test_source", 
        l=15.67, 
        b=80.75, 
        spectral_shape=dummy_shape
    )
    mock_external_dependencies["SkyDirection"].assert_called_once_with(l=15.67, b=80.75)


def test_init_position_xor_failure(dummy_shape):
    """Ensures initialization fails if multiple or no position specs are provided."""
    # Scenario 1: Both Equatorial and Galactic given
    with pytest.raises(AssertionError):
        PointSource("src", ra=10, dec=20, l=30, b=40, spectral_shape=dummy_shape)
        
    # Scenario 2: No position elements given
    with pytest.raises(AssertionError):
        PointSource("src", spectral_shape=dummy_shape)


def test_init_invalid_ra_dec_types(dummy_shape):
    """Ensures initialization catches values that cannot be cast to floats."""
    with pytest.raises(AssertionError):
        PointSource("src", ra="not_a_number", dec=-75.3, spectral_shape=dummy_shape)


def test_init_spectrum_xor_failure():
    """Ensures initialization fails if both spectral shape and components list are passed (or neither)."""
    # Scenario 1: Both given
    with pytest.raises(AssertionError):
        PointSource("src", ra=10, dec=20, spectral_shape=MagicMock(), components=[MagicMock()])
        
    # Scenario 2: Neither given
    with pytest.raises(AssertionError):
        PointSource("src", ra=10, dec=20)


# ==========================================
# EVALUATION & CALLABLE MECHANICS TESTS
# ==========================================

def test_call_without_tag_array(dummy_shape):
    """Validates simple spectrum summation over an array input without code integration."""
    source = PointSource("src", ra=10, dec=20, spectral_shape=dummy_shape)
    
    # Mock two spectral components inside the source
    comp1 = MagicMock(return_value=np.array([10.0, 20.0]))
    comp2 = MagicMock(return_value=np.array([5.0, 5.0]))
    source.components = {"c1": comp1, "c2": comp2}
    
    x_input = np.array([1.0, 2.0])
    results = source(x_input)
    
    # Assert components were called and outputs summed
    comp1.assert_called_once_with(x_input, None)
    comp2.assert_called_once_with(x_input, None)
    assert np.allclose(results, [15.0, 25.0])


def test_call_without_tag_scalar(dummy_shape):
    """Ensures scalar input arguments yield native scalar outputs."""
    source = PointSource("src", ra=10, dec=20, spectral_shape=dummy_shape)
    comp = MagicMock(return_value=np.array([42.0]))
    source.components = {"main": comp}
    
    results = source(5.0)
    assert isinstance(results, float)
    assert results == 42.0


def test_call_with_tag_no_integration(dummy_shape):
    """Tests evaluating a time/energy-varying setup at a static point (b is None)."""
    source = PointSource("src", ra=10, dec=20, spectral_shape=dummy_shape)
    
    comp = MagicMock(return_value=np.array([100.0]))
    source.components = {"main": comp}
    
    mock_var = MagicMock()
    tag_tuple = (mock_var, 5.0, None)  # Variable, static target value, No integration upper boundary
    
    results = source(1.0, tag=tag_tuple)
    
    # Confirm the integration variable was correctly updated before evaluation
    assert mock_var.value == 5.0
    assert results == 100.0


def test_call_with_tag_integration(dummy_shape, mock_external_dependencies):
    """Validates vector-based integration routing when tracking custom bounds."""
    source = PointSource("src", ra=10, dec=20, spectral_shape=dummy_shape)
    
    # Mock scipy quad_vec output format: (integral_array, error_array)
    mock_external_dependencies["sp_int"].quad_vec.return_value = (np.array([200.0]), None)
    
    mock_var = MagicMock()
    tag_tuple = (mock_var, 1.0, 5.0)  # Integrate variable from 1.0 to 5.0
    
    results = source(1.0, tag=tag_tuple)
    
    # Formula: integrals / (b - a) -> 200.0 / (5.0 - 1.0) = 50.0
    assert results == 50.0
    mock_external_dependencies["sp_int"].quad_vec.assert_called_once()


# ==========================================
# PARAMETER REFLECTION PROPERTIES TESTS
# ==========================================

def test_parameter_properties(dummy_shape):
    """Verifies logic behind parameter state parsing (free vs fixed)."""
    source = PointSource("src", ra=10, dec=20, spectral_shape=dummy_shape)
    
    # Set up mock parameters
    par_free = MagicMock(free=True, path="spec.main.raw.par1")
    par_fixed = MagicMock(free=False, path="spec.main.raw.par2")
    par_pos = MagicMock(free=False, path="pos.ra")
    
    # Assign parameters to internal component dependencies
    mock_comp = MagicMock()
    mock_comp.shape.parameters = {"p1": par_free, "p2": par_fixed}
    source._components = {"main": mock_comp}
    
    mock_pos = MagicMock()
    mock_pos.parameters = {"ra": par_pos}
    source.position = mock_pos
    
    # Assert assertions match state conditions
    assert source.has_free_parameters is True
    assert "spec.main.raw.par1" in source.free_parameters
    assert "spec.main.raw.par2" not in source.free_parameters
    assert len(source.parameters) == 3
