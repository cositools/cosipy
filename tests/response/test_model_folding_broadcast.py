import numpy as np
import astropy.units as u
import pytest

from histpy import Histogram, Axis, Axes, HealpixAxis

from threeML import Powerlaw, PointSource, Model

from cosipy.response import BinnedThreeMLModelFolding
from cosipy.response.threeml_response import _add_broadcast


def _axes(extra_label=None, extra_nbins=4):

    em_axis = Axis([100, 200, 300] * u.keV, label='Em')
    phi_axis = Axis([0, 90, 180] * u.deg, label='Phi')
    psichi_axis = HealpixAxis(nside=1, scheme='ring', coordsys='galactic', label='PsiChi')

    axes = []

    if extra_label is not None:
        axes.append(Axis(np.arange(extra_nbins + 1), label=extra_label))

    axes += [em_axis, phi_axis, psichi_axis]

    return Axes(axes, copy_axes=False)


class _StubData:
    """Minimal stand-in for a BinnedDataInterface, just exposing .axes."""

    def __init__(self, axes):
        self._axes = axes

    @property
    def axes(self):
        return self._axes


class _StubSourceResponse:
    """
    Minimal BinnedThreeMLSourceResponseInterface implementation that
    always returns a fixed-shape expectation (scaled by the source's
    spectral normalization K), regardless of what source is set.
    """

    def __init__(self, axes, value=1.0):
        self._axes = axes
        self._value = value
        self._source = None

    @property
    def axes(self):
        return self._axes

    def copy(self):
        return _StubSourceResponse(self._axes, self._value)

    def set_source(self, source):
        self._source = source

    def expectation(self, copy=True):
        hist = Histogram(self._axes)
        hist[:] = self._value * self._source.spectrum.main.shape.K.value
        return hist


def _make_model_and_response(value=3.0):

    spectrum = Powerlaw()
    spectrum.K.value = 2.0
    spectrum.K.unit = 1 / u.cm / u.cm / u.s / u.keV
    spectrum.piv.value = 1000
    spectrum.piv.unit = u.keV
    spectrum.index.value = -2

    source = PointSource("src", l=0, b=0, spectral_shape=spectrum)

    stub_psr = _StubSourceResponse(_axes(), value=value)

    return source, stub_psr


def test_add_broadcast_exact_match():

    axes = _axes()
    accumulator = Histogram(axes)
    accumulator[:] = 1.0

    contribution = Histogram(axes)
    contribution[:] = 2.0

    _add_broadcast(accumulator, contribution)

    assert np.all(accumulator.contents == 3.0)


def test_add_broadcast_extra_leading_axis():

    full_axes = _axes(extra_label='Depth', extra_nbins=4)
    plain_axes = _axes()

    accumulator = Histogram(full_axes)
    accumulator[:] = 1.0

    contribution = Histogram(plain_axes)
    contribution[:] = 2.0

    _add_broadcast(accumulator, contribution)

    assert accumulator.axes.labels.tolist() == full_axes.labels.tolist()
    assert np.all(accumulator.contents == 3.0)


def test_add_broadcast_rejects_contribution_with_extra_axis():

    plain_axes = _axes()
    full_axes = _axes(extra_label='Depth', extra_nbins=4)

    accumulator = Histogram(plain_axes)
    contribution = Histogram(full_axes)

    with pytest.raises(ValueError):
        _add_broadcast(accumulator, contribution)


def test_model_folding_broadcasts_depth_independent_response():
    """
    A point-source response that doesn't know about the extra 'Depth'
    axis (because the response is assumed not to vary with depth, only
    the background does) must still be usable in a Depth-segmented fit:
    BinnedThreeMLModelFolding should broadcast its expectation across
    all Depth bins.
    """

    n_depth = 4

    source, stub_psr = _make_model_and_response(value=3.0)

    data = _StubData(_axes(extra_label='Depth', extra_nbins=n_depth))

    folding = BinnedThreeMLModelFolding(data=data, point_source_response=stub_psr)
    folding.set_model(Model(source))

    expectation = folding.expectation()

    assert set(expectation.axes.labels) == {'Depth', 'Em', 'Phi', 'PsiChi'}
    assert expectation.axes['Depth'].nbins == n_depth

    # K=2.0, stub value=3.0 -> each (depth-independent) bin should be 6.0
    assert np.allclose(expectation.contents, 6.0)

    # Every depth slice must be identical, since the response doesn't
    # depend on it.
    dense = expectation.to_dense(copy=False).contents
    for i in range(1, n_depth):
        assert np.allclose(dense[0], dense[i])
