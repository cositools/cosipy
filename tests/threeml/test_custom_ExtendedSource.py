"""
Unit tests for CosipyExtendedSource.
"""

import collections

import numpy as np
import pytest

from astromodels import (
    SpectralComponent,
    Powerlaw,
    Constant,
    Gaussian_on_sphere,
)

from cosipy.threeml.custom_functions import CosipyExtendedSource 

# ---------------------------------------------------------------------------
# Fakes for the 3D (energy-dependent template) branch
# ---------------------------------------------------------------------------

class _FakeParameter:
    def __init__(self, path, free=True):
        self.path = path
        self.free = free


class _FakeSpatialShape3D:
    """Minimal stand-in for a 3D (energy-dependent) spatial template."""

    def __init__(self, name="fake_template", free_params=True):
        self.n_dim = 3
        self.name = name
        self._set_units_calls = []
        self._parameters = collections.OrderedDict(
            {
                f"{name}.par1": _FakeParameter(f"{name}.par1", free=free_params),
            }
        )

    def set_units(self, *args):
        self._set_units_calls.append(args)

    @property
    def parameters(self):
        return self._parameters

    def __call__(self, lon, lat, energies):
        lon = np.atleast_1d(lon)
        energies = np.atleast_1d(energies)
        return np.ones((len(lon), len(energies)))

    def to_dict(self, minimal=False):
        return {"template": self.name}

    def get_boundaries(self):
        return ((0.0, 360.0), (-90.0, 90.0))


def _make_2d_source(spectral_shape=None, components=None):
    spatial = Gaussian_on_sphere()
    if spectral_shape is None and components is None:
        spectral_shape = Powerlaw()
    return CosipyExtendedSource(
        "test_ext_source", spatial, spectral_shape=spectral_shape, components=components
    )


def _make_3d_template_source():
    spatial = _FakeSpatialShape3D()
    return CosipyExtendedSource("test_ext_source", spatial)


def _make_3d_with_spectrum_source():
    spatial = _FakeSpatialShape3D()
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

def test_3d_template_defaults_to_constant_main_component():
    src = _make_3d_template_source()
    assert "main" in src.components
    assert isinstance(src.components["main"].shape, Constant)


def test_3d_with_explicit_spectral_shape():
    src = _make_3d_with_spectrum_source()
    assert "main" in src.components
    assert isinstance(src.components["main"].shape, Powerlaw)


def test_3d_both_shape_and_components_raises():
    c1 = SpectralComponent("component1", Powerlaw())
    with pytest.raises(AssertionError):
        CosipyExtendedSource(
            "test_ext_source",
            _FakeSpatialShape3D(),
            spectral_shape=Powerlaw(),
            components=[c1],
        )


def test_3d_components_list():
    c1 = SpectralComponent("component1", Powerlaw())
    c2 = SpectralComponent("component2", Powerlaw())
    src = CosipyExtendedSource("test_ext_source", _FakeSpatialShape3D(), components=[c1, c2])
    assert set(src.components.keys()) == {"component1", "component2"}


def test_invalid_ndim_raises_runtime_error():
    bad_shape = _FakeSpatialShape3D()
    bad_shape.n_dim = 1
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


def test_call_3d_delegates_to_spatial_shape():
    src = _make_3d_with_spectrum_source()
    lon = np.array([10.0])
    lat = np.array([5.0])
    energies = np.array([1.0, 10.0])
    result = src(lon, lat, energies)
    assert result is not None


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


def test_3d_fake_shape_parameters_included():
    src = _make_3d_with_spectrum_source()
    keys = src.parameters.keys()
    assert any("par1" in k for k in keys)


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

def test_get_boundaries_delegates_to_spatial_shape():
    src = _make_3d_with_spectrum_source()
    boundaries = src.get_boundaries()
    assert boundaries == ((0.0, 360.0), (-90.0, 90.0))


def test_get_boundaries_returns_two_tuples_for_2d_source():
    src = _make_2d_source()
    boundaries = src.get_boundaries()
    assert len(boundaries) == 2
    assert len(boundaries[0]) == 2
    assert len(boundaries[1]) == 2


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
