from types import SimpleNamespace

import astropy.units as u
import numpy as np
import pytest
from astromodels import (
    Constant,
    Cutoff_powerlaw,
    Model,
    PointSource,
    Powerlaw,
    clone_model,
)

from cosipy.response.integrals import get_integral_values
from cosipy.threeml import BinnedSED, freeze_binned_sed_bins


def _response(nbins=8):
    edges = np.geomspace(100.0, 10000.0, nbins + 1) * u.keV
    return SimpleNamespace(
        axes={"Ei": SimpleNamespace(edges=edges, nbins=nbins)}
    )


def _powerlaw(index=-1.7):
    shape = Powerlaw()
    shape.K.value = 3.0e-5
    shape.piv.value = 300.0
    shape.index.value = index
    return shape


def test_binned_sed_defaults_to_powerlaw_index_minus_two():
    spectrum = BinnedSED.from_response(_response())

    assert isinstance(spectrum.spectral_shape, Powerlaw)
    assert spectrum.spectral_shape.index.value == -2.0
    assert "index" not in spectrum.parameters


def test_binned_sed_snapshots_and_freezes_spectral_shape():
    input_shape = _powerlaw()
    initial_fluxes = np.array([1.0e-6, 2.0e-6, 3.0e-6, 4.0e-6])
    spectrum = BinnedSED.from_response(
        _response(),
        input_shape,
        ei_bin_indices=range(2, 6),
        initial_fluxes=initial_fluxes,
    )

    assert isinstance(spectrum, BinnedSED)
    assert spectrum.n_bins == 4
    assert spectrum._cosipy_ei_bin_indices == (2, 3, 4, 5)
    assert all(
        not parameter.free
        for parameter in spectrum.spectral_shape.parameters.values()
    )
    assert all(parameter.free for parameter in spectrum.normalizations)
    assert np.allclose(spectrum(spectrum.pivots), initial_fluxes)

    frozen_index = spectrum.spectral_shape.index.value
    input_shape.index.value = -3.0
    assert spectrum.spectral_shape.index.value == frozen_index


def test_binned_sed_uses_all_response_bins_by_default():
    response = _response(nbins=7)
    spectrum = BinnedSED.from_response(response, _powerlaw())

    assert spectrum.n_bins == 7
    assert spectrum._cosipy_ei_bin_indices == tuple(range(7))
    assert len(spectrum.normalizations) == 7


def test_binned_sed_zero_outside_range():
    spectrum = BinnedSED.from_response(_response(nbins=5), _powerlaw())
    edges = spectrum.bin_edges

    flux = spectrum(np.array([0.5 * edges[0], 2.0 * edges[-1]]))

    assert np.array_equal(flux, [0.0, 0.0])


def test_powerlaw_shape_has_expected_within_bin_values_and_integrals():
    response = _response(nbins=4)
    initial_fluxes = np.array([1e-6, 2e-6, 3e-6, 4e-6])
    spectrum = BinnedSED.from_response(
        response,
        _powerlaw(index=-1.0),
        initial_fluxes=initial_fluxes,
    )
    edges = spectrum.bin_edges
    expected_integral = sum(
        k
        * np.sqrt(edges[i] * edges[i + 1])
        * np.log(edges[i + 1] / edges[i])
        for i, k in enumerate(initial_fluxes)
    )

    assert np.allclose(spectrum(spectrum.pivots), initial_fluxes)
    assert np.isclose(
        spectrum.integral(edges[0], edges[-1]),
        expected_integral,
    )


def test_cutoff_shape_retains_curvature_inside_each_bin():
    response = _response(nbins=4)
    shape = Cutoff_powerlaw()
    shape.K.value = 2.0e-5
    shape.piv.value = 200.0
    shape.index.value = -2.0
    shape.xc.value = 450.0
    initial_fluxes = np.array([1.0e-6, 2.0e-6, 3.0e-6, 4.0e-6])
    spectrum = BinnedSED.from_response(
        response,
        shape,
        initial_fluxes=initial_fluxes,
    )

    energy = 0.75 * spectrum.bin_edges[2] + 0.25 * spectrum.bin_edges[3]
    expected = initial_fluxes[2] * shape(energy) / shape(spectrum.pivots[2])
    assert np.isclose(spectrum(energy), expected)


def test_binned_sed_supports_units():
    spectrum = BinnedSED.from_response(_response(nbins=3), _powerlaw())
    flux_unit = 1.0 / (u.keV * u.cm**2 * u.s)
    spectrum.set_units(u.keV, flux_unit)

    values = spectrum(spectrum.pivots * u.keV)
    assert values.unit == flux_unit
    assert np.allclose(
        values.value,
        [parameter.value for parameter in spectrum.normalizations],
    )


def test_binned_sed_rejects_zero_pivot_shape():
    shape = Constant()
    shape.k.value = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        BinnedSED.from_response(_response(), shape)


def test_existing_bin_freezing_utility_accepts_shape_driven_sed():
    spectrum = BinnedSED.from_response(_response(), _powerlaw())

    frozen = freeze_binned_sed_bins(spectrum, [1, 3])

    assert frozen == (1, 3)
    assert spectrum.K1.value == 0.0
    assert spectrum.K1.free is False
    assert spectrum.K3.value == 0.0
    assert spectrum.K3.free is False


def test_astromodels_model_exposes_only_bin_normalizations_as_free():
    spectrum = BinnedSED.from_response(_response(), _powerlaw())
    model = Model(
        PointSource("test_source", l=0.0, b=0.0, spectral_shape=spectrum)
    )

    free_paths = tuple(model.free_parameters)
    assert len(free_paths) == spectrum.n_bins
    assert all(path.rsplit(".", 1)[-1].startswith("K") for path in free_paths)
    assert "index" not in spectrum.parameters


def test_astromodels_model_clone_restores_frozen_spectral_shape():
    shape = _powerlaw(index=-1.35)
    spectrum = BinnedSED.from_response(_response(), shape)
    model = Model(
        PointSource("test_source", l=0.0, b=0.0, spectral_shape=spectrum)
    )

    cloned_model = clone_model(model)
    cloned_spectrum = cloned_model.test_source.spectrum.main.shape

    assert cloned_spectrum.spectral_shape.index.value == -1.35
    assert all(
        not parameter.free
        for parameter in cloned_spectrum.spectral_shape.parameters.values()
    )
    assert np.allclose(
        cloned_spectrum(cloned_spectrum.pivots),
        [parameter.value for parameter in cloned_spectrum.normalizations],
    )


def test_response_integrals_match_direct_binned_sed_integrals():
    response = _response(nbins=5)
    spectrum = BinnedSED.from_response(
        response,
        _powerlaw(index=-1.35),
        initial_fluxes=np.geomspace(1.0e-7, 5.0e-6, 5),
    )
    values = get_integral_values(spectrum, response.axes["Ei"].edges.value)
    expected = np.asarray(
        [
            spectrum.integral(low, high)
            for low, high in zip(spectrum.bin_edges[:-1], spectrum.bin_edges[1:])
        ]
    )

    assert np.allclose(values, expected)
