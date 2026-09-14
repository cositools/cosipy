from types import SimpleNamespace

import numpy as np
import pytest
import astropy.units as u

from cosipy.threeml import (
    BinnedSED,
    check_binned_sed_response,
    find_unconstrained_sed_bins,
    freeze_binned_sed_bins,
    profile_likelihood_upper_limit,
)


class _DummyEiAxis:
    def __init__(self, nbins=9):
        self.edges = np.geomspace(100.0, 10000.0, nbins + 1) * u.keV
        self.nbins = nbins


class _DummyResponse:
    def __init__(self, nbins=9):
        self.axes = {"Ei": _DummyEiAxis(nbins)}


class _DummyExpectation:
    def __init__(self, contents):
        self.contents = np.asarray(contents, dtype=float)


class _DummySourceResponse:
    def __init__(self, spectrum, zero_bin=2):
        self._source = SimpleNamespace(
            spectrum=SimpleNamespace(
                main=SimpleNamespace(shape=spectrum)
            )
        )
        self.zero_bin = zero_bin

    def expectation(self, copy=False):
        spectrum = self._source.spectrum.main.shape
        values = np.array([par.value for par in spectrum.normalizations])
        active = int(np.argmax(values))

        if values[active] == 0.0 or active == self.zero_bin:
            return _DummyExpectation(np.zeros((2, 2)))

        return _DummyExpectation(np.full((2, 2), values[active]))


def test_response_configuration_sets_dynamic_parameters():
    response = _DummyResponse(nbins=9)
    initial_fluxes = np.geomspace(1e-8, 1e-6, 5)

    spectrum = BinnedSED.from_response(
        response,
        ei_bin_indices=range(2, 7),
        initial_fluxes=initial_fluxes,
        index=-2.0,
    )

    expected_edges = response.axes["Ei"].edges[2:8].to_value(u.keV)
    actual_fluxes = np.array([par.value for par in spectrum.normalizations])

    assert spectrum.n_bins == 5
    assert np.allclose(spectrum.bin_edges, expected_edges)
    assert np.allclose(actual_fluxes, initial_fluxes)
    assert spectrum.index.value == -2.0
    assert spectrum._cosipy_ei_bin_indices == tuple(range(2, 7))
    assert all(par.free for par in spectrum.normalizations)


def test_find_and_freeze_unconstrained_sed_bins():
    templates = np.ones((4, 3, 2))
    templates[1] = 0.0
    templates[3] = 0.0

    bins = find_unconstrained_sed_bins(
        templates,
        bin_indices=[5, 6, 7, 8],
    )

    assert np.array_equal(bins, [6, 8])

    spectrum = BinnedSED.from_response(_DummyResponse(nbins=6))
    frozen = freeze_binned_sed_bins(spectrum, [2, 5])

    assert frozen == (2, 5)
    assert spectrum.K2.value == 0.0
    assert spectrum.K5.value == 0.0
    assert spectrum.K2.free is False
    assert spectrum.K5.free is False


def test_check_binned_sed_response():
    spectrum = BinnedSED.from_response(
        _DummyResponse(nbins=8),
        ei_bin_indices=range(3, 8),
    )
    source_response = _DummySourceResponse(spectrum, zero_bin=2)

    original_values = np.array([par.value for par in spectrum.normalizations])

    with pytest.warns(UserWarning):
        result = check_binned_sed_response(
            spectrum,
            source_response,
            freeze=True,
            verbose=False,
        )

    assert np.array_equal(result["local_indices"], [2])
    assert np.array_equal(result["response_indices"], [5])
    assert result["template_totals"][2] == 0.0

    for i, par in enumerate(spectrum.normalizations):
        if i == 2:
            assert par.value == 0.0
            assert par.free is False
        else:
            assert par.value == original_values[i]


def test_profile_likelihood_upper_limit():
    best = 2.0e-6
    sigma = 0.4e-6
    nll_best = 12.3

    def profile_nll(value):
        return nll_best + 0.5 * ((value - best) / sigma) ** 2

    ul = profile_likelihood_upper_limit(
        profile_nll,
        nll_best=nll_best,
        best_value=best,
        sigma=sigma,
    )

    expected = best + sigma * np.sqrt(2.705543)

    assert np.isclose(ul, expected, rtol=1e-5)
