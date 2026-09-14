from types import SimpleNamespace

import numpy as np
import astropy.units as u

from cosipy.threeml import BinnedSED


def _response(nbins=8):
    edges = np.geomspace(100.0, 10000.0, nbins + 1) * u.keV
    return SimpleNamespace(
        axes={"Ei": SimpleNamespace(edges=edges, nbins=nbins)}
    )


def test_binned_sed_arbitrary_number_of_response_bins():
    response = _response(nbins=8)
    initial_fluxes = np.array([1e-6, 2e-6, 3e-6, 4e-6])

    spectrum = BinnedSED.from_response(
        response,
        ei_bin_indices=range(2, 6),
        initial_fluxes=initial_fluxes,
    )

    assert isinstance(spectrum, BinnedSED)
    assert spectrum.n_bins == 4
    assert spectrum._cosipy_ei_bin_indices == (2, 3, 4, 5)

    expected_edges = response.axes["Ei"].edges[2:7].to_value(u.keV)
    assert np.allclose(spectrum.bin_edges, expected_edges)

    flux = spectrum(spectrum.pivots)
    assert np.allclose(flux, initial_fluxes)


def test_binned_sed_uses_all_response_bins_by_default():
    response = _response(nbins=7)
    spectrum = BinnedSED.from_response(response)

    assert spectrum.n_bins == 7
    assert spectrum._cosipy_ei_bin_indices == tuple(range(7))
    assert len(spectrum.normalizations) == 7


def test_binned_sed_zero_outside_range():
    response = _response(nbins=5)
    spectrum = BinnedSED.from_response(response)
    edges = spectrum.bin_edges

    flux = spectrum(np.array([0.5 * edges[0], 2.0 * edges[-1]]))

    assert np.array_equal(flux, [0.0, 0.0])


def test_binned_sed_integral_index_minus_one():
    response = _response(nbins=4)
    initial_fluxes = np.array([1e-6, 2e-6, 3e-6, 4e-6])
    spectrum = BinnedSED.from_response(
        response,
        initial_fluxes=initial_fluxes,
        index=-1.0,
    )

    edges = spectrum.bin_edges
    expected = 0.0

    for i, k in enumerate(initial_fluxes):
        epiv = np.sqrt(edges[i] * edges[i + 1])
        expected += k * epiv * np.log(edges[i + 1] / edges[i])

    assert np.isclose(
        spectrum.integral(edges[0], edges[-1]),
        expected,
    )
