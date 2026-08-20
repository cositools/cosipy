"""
Unit tests for CosipyExtendedSource.
"""

import collections

import numpy as np
import pytest
from astropy.io import fits

from astromodels import (
    SpectralComponent,
    Powerlaw,
    Constant,
    Gaussian_on_sphere,
    SpatialTemplate_2D
)

from cosipy.threeml.custom_functions import CosipyExtendedSource 

# ---------------------------------------------------------------------------
# Fake FITS cube generation for the 3D (energy-dependent template) branch
# ---------------------------------------------------------------------------
 
NE, NL, NB = 3, 4, 4
 
 
def _make_fits_file(path: str, ne: int = NE, nl: int = NL, nb: int = NB) -> str:
    """Write a minimal 3-D FITS cube and return its path."""
    data = np.random.rand(ne, nl, nb).astype(np.float64)
    hdu = fits.PrimaryHDU(data)
    # Mandatory WCS keys for a 3D cube
    hdu.header["CDELT1"] = 0.5
    hdu.header["CDELT2"] = 0.5
    hdu.header["CDELT3"] = 0.2
    hdu.header["CRVAL1"] = 0.0
    hdu.header["CRVAL2"] = 0.0
    hdu.header["CRVAL3"] = 5.0
    hdu.header["CRPIX1"] = nl // 2
    hdu.header["CRPIX2"] = nb // 2
    hdu.header["CRPIX3"] = 1
    hdu.header["NAXIS1"] = nl
    hdu.header["NAXIS2"] = nb
    hdu.header["NAXIS3"] = ne
    hdu.writeto(path, overwrite=True)
    return path
 
 
@pytest.fixture
def fits_cube_path(tmp_path):
    return _make_fits_file(str(tmp_path / "test_cube.fits"))
 
 
def _load_3d_template(fits_path):
    spatial = SpatialTemplate_2D()
    spatial.load_file(fits_path)
    assert spatial.n_dim == 3
    return spatial
 
 
def _make_2d_source(spectral_shape=None, components=None):
    spatial = Gaussian_on_sphere()
    if spectral_shape is None and components is None:
        spectral_shape = Powerlaw()
    return CosipyExtendedSource(
        "test_ext_source", spatial, spectral_shape=spectral_shape, components=components
    )
 
 
def _make_3d_template_source(fits_path):
    spatial = _load_3d_template(fits_path)
    return CosipyExtendedSource("test_ext_source", spatial)
 
 
def _make_3d_with_spectrum_source(fits_path):
    spatial = _load_3d_template(fits_path)
    return CosipyExtendedSource("test_ext_source", spatial, spectral_shape=Powerlaw())
 
 
# ---------------------------------------------------------------------------
# Construction: 2D spatial shape
# ---------------------------------------------------------------------------
 
def test_2d_single_spectral_shape_creates_main_component():
    src = _make_2d_source(spectral_shape=Powerlaw())
    assert "main" in src.components
    assert len(src.components) == 1
 
 
def test_2d_components_list():
    c1 = SpectralComponent("component1", Powerlaw())
    c2 = SpectralComponent("component2", Powerlaw())
    src = _make_2d_source(components=[c1, c2])
    assert set(src.components.keys()) == {"component1", "component2"}
 
 
def test_2d_both_shape_and_components_raises():
    c1 = SpectralComponent("component1", Powerlaw())
    with pytest.raises(AssertionError):
        _make_2d_source(spectral_shape=Powerlaw(), components=[c1])
 
 
def test_2d_neither_shape_nor_components_raises():
    with pytest.raises(AssertionError):
        CosipyExtendedSource("test_ext_source", Gaussian_on_sphere())
 
 
def test_2d_spatial_shape_stored_and_added_as_child():
    src = _make_2d_source()
    assert src.spatial_shape is not None
    assert "spectrum" in [child.name for child in src._get_children()]
 
 
# ---------------------------------------------------------------------------
# Construction: 3D spatial shape
# ---------------------------------------------------------------------------
 
def test_3d_template_defaults_to_constant_main_component(fits_cube_path):
    src = _make_3d_template_source(fits_cube_path)
    assert "main" in src.components
    assert isinstance(src.components["main"].shape, Constant)
 
 
def test_3d_with_explicit_spectral_shape(fits_cube_path):
    src = _make_3d_with_spectrum_source(fits_cube_path)
    assert "main" in src.components
    assert isinstance(src.components["main"].shape, Powerlaw)
 
 
def test_3d_both_shape_and_components_raises(fits_cube_path):
    spatial = _load_3d_template(fits_cube_path)
    c1 = SpectralComponent("component1", Powerlaw())
    with pytest.raises(AssertionError):
        CosipyExtendedSource(
            "test_ext_source",
            spatial,
            spectral_shape=Powerlaw(),
            components=[c1],
        )
 
 
def test_3d_components_list(fits_cube_path):
    spatial = _load_3d_template(fits_cube_path)
    c1 = SpectralComponent("component1", Powerlaw())
    c2 = SpectralComponent("component2", Powerlaw())
    src = CosipyExtendedSource("test_ext_source", spatial, components=[c1, c2])
    assert set(src.components.keys()) == {"component1", "component2"}
 
 
def test_invalid_ndim_raises_runtime_error(fits_cube_path):
    bad_shape = _load_3d_template(fits_cube_path)
    bad_shape.n_dim = 1  # force an invalid value to hit the else/RuntimeError branch
    with pytest.raises(RuntimeError):
        CosipyExtendedSource("test_ext_source", bad_shape, spectral_shape=Powerlaw())
 
 
# ---------------------------------------------------------------------------
# __call__
# ---------------------------------------------------------------------------
 
def test_call_2d_type_mismatch_raises_assertion():
    src = _make_2d_source()
    with pytest.raises(AssertionError):
        src(1.0, 1.0, np.array([1.0]))
 
 
def test_call_2d_returns_array_like_result():
    src = _make_2d_source()
    lon = np.array([10.0, 20.0])
    lat = np.array([5.0, 5.0])
    energies = np.array([1.0, 10.0, 100.0])
    result = src(lon, lat, energies)
    assert result is not None
    assert not np.any(np.isnan(np.asarray(result)))
 
 
def test_call_3d_delegates_to_spatial_shape(fits_cube_path):
    src = _make_3d_with_spectrum_source(fits_cube_path)
    lon = np.array([10.0])
    lat = np.array([5.0])
    energies = np.array([1.0, 10.0])
    result = src(lon, lat, energies)
    assert result is not None
    assert not np.any(np.isnan(np.asarray(result)))
 
 
# ---------------------------------------------------------------------------
# get_spatially_integrated_flux
# ---------------------------------------------------------------------------
 
def test_get_spatially_integrated_flux_returns_expected_shape():
    src = _make_2d_source()
    energies = np.array([1.0, 10.0, 100.0])
    flux = src.get_spatially_integrated_flux(energies)
    assert len(np.atleast_1d(flux)) == len(energies)
 
 
def test_get_spatially_integrated_flux_accepts_scalar_or_list():
    src = _make_2d_source()
    flux = src.get_spatially_integrated_flux(10.0)
    assert flux is not None
 
 
# ---------------------------------------------------------------------------
# has_free_parameters / free_parameters / parameters
# ---------------------------------------------------------------------------
 
def test_has_free_parameters_true_by_default():
    src = _make_2d_source()
    assert src.has_free_parameters is True
 
 
def test_has_free_parameters_false_when_all_fixed():
    src = _make_2d_source()
    for par in src.parameters.values():
        par.free = False
    assert src.has_free_parameters is False
 
 
def test_free_parameters_subset_of_parameters():
    src = _make_2d_source()
    assert set(src.free_parameters.keys()) <= set(src.parameters.keys())
 
 
def test_free_parameters_empty_when_all_fixed():
    src = _make_2d_source()
    for par in src.parameters.values():
        par.free = False
    assert len(src.free_parameters) == 0
 
 
def test_parameters_keys_are_paths():
    src = _make_2d_source()
    for path, par in src.parameters.items():
        assert path == par.path
 
 
def test_parameters_includes_spatial_and_spectral_params():
    src = _make_2d_source()
    keys = src.parameters.keys()
    assert any("main" in k for k in keys)
    # spatial shape params should also be present (e.g. lon0/lat0/sigma)
    spatial_param_names = set(src.spatial_shape.parameters.keys())
    assert any(any(name in k for name in spatial_param_names) for k in keys)
 
 
def test_3d_template_parameters_included(fits_cube_path):
    src = _make_3d_with_spectrum_source(fits_cube_path)
    spatial_param_names = set(src.spatial_shape.parameters.keys())
    keys = src.parameters.keys()
    assert any(any(name in k for name in spatial_param_names) for k in keys)
 
 
# ---------------------------------------------------------------------------
# Representation
# ---------------------------------------------------------------------------
 
def test_repr_base_text_output_contains_name():
    src = _make_2d_source()
    text_repr = src._repr__base(rich_output=False)
    assert "test_ext_source" in str(text_repr)
 
 
def test_str_does_not_raise():
    src = _make_2d_source()
    str(src)
 
 
# ---------------------------------------------------------------------------
# get_boundaries
# ---------------------------------------------------------------------------
 
def test_get_boundaries_delegates_to_spatial_shape(fits_cube_path):
    src = _make_3d_with_spectrum_source(fits_cube_path)
    boundaries = src.get_boundaries()
    assert boundaries == src.spatial_shape.get_boundaries()
 
 
def test_get_boundaries_returns_two_tuples_for_2d_source():
    src = _make_2d_source()
    boundaries = src.get_boundaries()
    assert len(boundaries) == 2
    assert len(boundaries[0]) == 2
    assert len(boundaries[1]) == 2
 
 
if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
 
