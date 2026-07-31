from cosipy.threeml.custom_functions import CosipyPointSource
import numpy as np
import pytest
from astromodels import Powerlaw
from astromodels.core.sky_direction import SkyDirection
from astromodels.core.spectral_component import SpectralComponent

def test_construct_with_equatorial_position():
        src = PointSource("test_source", ra=125.6, dec=-75.3, spectral_shape=Powerlaw())
        assert src.name == "test_source"
        assert pytest.approx(src.position.ra.value, rel=1e-6) == 125.6
        assert pytest.approx(src.position.dec.value, rel=1e-6) == -75.3
 
def test_construct_with_galactic_position():
        src = PointSource("test_source", l=15.67, b=80.75, spectral_shape=Powerlaw())
        assert pytest.approx(src.position.l.value, rel=1e-6) == 15.67
        assert pytest.approx(src.position.b.value, rel=1e-6) == 80.75
 
def test_construct_with_sky_position_object():
        sky_pos = SkyDirection(ra=10.0, dec=20.0)
        src = PointSource("test_source", sky_position=sky_pos, spectral_shape=Powerlaw())
        assert pytest.approx(src.position.ra.value, rel=1e-6) == 10.0
        assert pytest.approx(src.position.dec.value, rel=1e-6) == 20.0
 
def test_ra_dec_coerced_to_float():
        src = PointSource("test_source", ra="125.6", dec="-75.3", spectral_shape=Powerlaw())
        assert pytest.approx(src.position.ra.value, rel=1e-6) == 125.6
 
def test_no_position_raises():
        with pytest.raises(AssertionError):
            PointSource("test_source", spectral_shape=Powerlaw())
 
def test_partial_equatorial_position_raises():
        # ra given without dec -> falls through to the galactic branch,
        # which will also be incomplete -> should raise
        with pytest.raises(AssertionError):
            PointSource("test_source", ra=125.6, spectral_shape=Powerlaw())
 
def test_both_equatorial_and_galactic_raises():
        with pytest.raises(AssertionError):
            PointSource(
                "test_source", ra=125.6, dec=-75.3, l=15.67, b=80.75,
                spectral_shape=Powerlaw(),
            )
 
def test_equatorial_and_sky_position_raises():
        sky_pos = SkyDirection(ra=10.0, dec=20.0)
        with pytest.raises(AssertionError):
            PointSource(
                "test_source", ra=125.6, dec=-75.3, sky_position=sky_pos,
                spectral_shape=Powerlaw(),
            )
 
def test_invalid_ra_dec_type_raises():
        with pytest.raises(AssertionError):
            PointSource("test_source", ra="not_a_number", dec=-75.3,
                         spectral_shape=Powerlaw())
 
 
# ---------------------------------------------------------------------------
# Construction: spectral shape vs. components
# ---------------------------------------------------------------------------
 
 
def test_single_spectral_shape_creates_main_component():
        src = PointSource("test_source", ra=125.6, dec=-75.3, spectral_shape=Powerlaw())
        assert "main" in src.components
        assert len(src.components) == 1
 
def test_multiple_components():
        c1 = SpectralComponent("component1", Powerlaw())
        c2 = SpectralComponent("component2", Powerlaw())
        src = PointSource("test_source", ra=125.6, dec=-75.3, components=[c1, c2])
        assert set(src.components.keys()) == {"component1", "component2"}
        assert len(src.components) == 2
 
def test_no_spectrum_raises():
        with pytest.raises(AssertionError):
            PointSource("test_source", ra=125.6, dec=-75.3)
 
def test_both_shape_and_components_raises():
        c1 = SpectralComponent("component1", Powerlaw())
        with pytest.raises(AssertionError):
            PointSource(
                "test_source", ra=125.6, dec=-75.3,
                spectral_shape=Powerlaw(), components=[c1],
            )
 
def test_components_are_children_of_spectrum_node():
        src = PointSource("test_source", ra=125.6, dec=-75.3, spectral_shape=Powerlaw())
        assert "spectrum" in [child.name for child in src._get_children()]
 
 
# ---------------------------------------------------------------------------
# __call__
# ---------------------------------------------------------------------------
 
 
def test_call_scalar_input_returns_scalar():
        src = PointSource("test_source", ra=125.6, dec=-75.3, spectral_shape=Powerlaw())
        result = src(1.0)
        assert np.isscalar(result) or np.ndim(result) == 0
 
def test_call_array_input_returns_array():
        src = PointSource("test_source", ra=125.6, dec=-75.3, spectral_shape=Powerlaw())
        x = np.array([1.0, 2.0, 3.0])
        result = src(x)
        assert np.ndim(result) == 1
        assert len(result) == 3
 
def test_call_sums_multiple_components():
        c1 = SpectralComponent("component1", Powerlaw())
        c2 = SpectralComponent("component2", Powerlaw())
        src = PointSource("test_source", ra=125.6, dec=-75.3, components=[c1, c2])
 
        x = np.array([1.0, 2.0, 3.0])
        combined = src(x)
 
        expected = c1.shape(x) + c2.shape(x)
        np.testing.assert_allclose(combined, expected)
 
def test_call_matches_direct_component_evaluation():
        pl = Powerlaw()
        src = PointSource("test_source", ra=125.6, dec=-75.3, spectral_shape=pl)
        x = np.array([1.0, 10.0, 100.0])
        np.testing.assert_allclose(src(x), pl(x))
 
 
# ---------------------------------------------------------------------------
# has_free_parameters / free_parameters / parameters
# ---------------------------------------------------------------------------
 
 
def _make_source():
        return PointSource("test_source", ra=125.6, dec=-75.3, spectral_shape=Powerlaw())
 
def test_has_free_parameters_true_by_default():
        # Powerlaw's K and index are free by default; position is fixed
        src = _make_source()
        assert src.has_free_parameters is True
 
def test_has_free_parameters_false_when_all_fixed():
        src = _make_source()
        for par in src.parameters.values():
            par.free = False
        assert src.has_free_parameters is False
 
def test_position_fixed_by_default():
        src = _make_source()
        for par in src.position.parameters.values():
            assert par.free is False
 
def test_free_parameters_subset_of_parameters():
        src = _make_source()
        assert set(src.free_parameters.keys()) <= set(src.parameters.keys())
 
def test_free_parameters_updates_when_freezing():
        src = _make_source()
        n_free_before = len(src.free_parameters)
        assert n_free_before > 0
 
        # Freeze every parameter
        for par in src.parameters.values():
            par.free = False
        assert len(src.free_parameters) == 0
 
def test_freeing_position_adds_to_free_parameters():
        src = _make_source()
        src.position.ra.free = True
        assert src.position.ra.path in src.free_parameters
 
def test_parameters_keys_are_paths():
        src = _make_source()
        for path, par in src.parameters.items():
            assert path == par.path
 
def test_parameters_includes_position_and_spectrum():
        src = _make_source()
        keys = src.parameters.keys()
        assert any("position" in k for k in keys)
        assert any("main" in k for k in keys)
 
 
# ---------------------------------------------------------------------------
# Representation
# ---------------------------------------------------------------------------
 
 
def test_repr_base_text_output():
        src = PointSource("test_source", ra=125.6, dec=-75.3, spectral_shape=Powerlaw())
        text_repr = src._repr__base(rich_output=False)
        assert "test_source" in str(text_repr)
 
def test_str_does_not_raise():
        src = PointSource("test_source", ra=125.6, dec=-75.3, spectral_shape=Powerlaw())
        # Should not raise regardless of underlying repr mechanism
        str(src)
