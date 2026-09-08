import numpy as np

import astropy.units as u
from histpy import Axis, Axes, HealpixAxis, Histogram
from scoords import SpacecraftFrame

from cosipy.polarization import PolarizationAxis, StereographicConvention
from cosipy.response.relative_irf_hist import IRFRelativeHistUnpolarized


def _make_irf_hist(nside=2, seed=0):
    rng = np.random.default_rng(seed)

    axes = Axes([
        HealpixAxis(nside=nside, scheme='ring', coordsys=SpacecraftFrame(), label='NuLambda'),
        Axis(np.geomspace(100, 1000, 4) * u.keV, label='Ei', scale='log'),
        Axis(np.linspace(-0.5, 0.5, 5), label='Epsilon'),
        Axis(np.linspace(0, 180, 7) * u.deg, label='Phi'),
        Axis(np.linspace(-90, 90, 5) * u.deg, label='Theta'),
        PolarizationAxis(np.linspace(0, 360, 9) * u.deg, convention=StereographicConvention(), label='Zeta'),
    ])

    contents = rng.random(axes.nbins)

    return Histogram(axes, contents=contents, unit=u.cm * u.cm)


def _make_aeff_hist(nside=4, seed=1):
    rng = np.random.default_rng(seed)

    axes = Axes([
        HealpixAxis(nside=nside, scheme='ring', coordsys=SpacecraftFrame(), label='NuLambda'),
        Axis(np.geomspace(50, 2000, 6) * u.keV, label='Ei', scale='log'),
    ])

    contents = rng.random(axes.nbins) * 100

    return Histogram(axes, contents=contents, unit=u.cm * u.cm)


class TestFromH5Aeff:

    def test_auto_detects_aeff_group(self, tmp_path):
        """If the file has an "AEFF" group, from_h5() should use it as the
        total effective area, even though its NuLambda/Ei grid differs
        from the irf's own."""

        path = tmp_path / "irf_with_aeff.h5"
        irf_hist = _make_irf_hist()
        aeff_hist = _make_aeff_hist()

        irf_hist.write(str(path), name="IRF", overwrite=True)
        aeff_hist.write(str(path), name="AEFF", overwrite=True)

        model = IRFRelativeHistUnpolarized.from_h5(path)

        assert model._tot_aeff.axes.labels.tolist() == ['NuLambda', 'Ei']
        assert model._tot_aeff.contents.shape == aeff_hist.contents.shape

    def test_falls_back_to_projection_without_aeff_group(self, tmp_path):
        """Without an "AEFF" group, from_h5() should behave as before:
        the total effective area is the irf's own NuLambda/Ei projection."""

        path = tmp_path / "irf_only.h5"
        irf_hist = _make_irf_hist()
        irf_hist.write(str(path), name="IRF", overwrite=True)

        model = IRFRelativeHistUnpolarized.from_h5(path)

        assert model._tot_aeff.contents.shape == (irf_hist.axes['NuLambda'].nbins, irf_hist.axes['Ei'].nbins)

    def test_explicit_aeff_kwarg_takes_precedence(self, tmp_path):
        """An aeff passed explicitly to from_h5() should win over any
        "AEFF" group present in the file."""

        path = tmp_path / "irf_with_aeff.h5"
        irf_hist = _make_irf_hist()
        file_aeff_hist = _make_aeff_hist(nside=4)
        irf_hist.write(str(path), name="IRF", overwrite=True)
        file_aeff_hist.write(str(path), name="AEFF", overwrite=True)

        override_aeff_hist = _make_aeff_hist(nside=8, seed=2)

        model = IRFRelativeHistUnpolarized.from_h5(path, aeff=override_aeff_hist)

        assert model._tot_aeff.contents.shape == override_aeff_hist.contents.shape
        assert model._tot_aeff.contents.shape != file_aeff_hist.contents.shape

    def test_explicit_positional_aeff_takes_precedence(self, tmp_path):
        """Same as above, but with aeff passed positionally."""

        path = tmp_path / "irf_with_aeff.h5"
        irf_hist = _make_irf_hist()
        file_aeff_hist = _make_aeff_hist(nside=4)
        irf_hist.write(str(path), name="IRF", overwrite=True)
        file_aeff_hist.write(str(path), name="AEFF", overwrite=True)

        override_aeff_hist = _make_aeff_hist(nside=8, seed=3)

        model = IRFRelativeHistUnpolarized.from_h5(path, override_aeff_hist)

        assert model._tot_aeff.contents.shape == override_aeff_hist.contents.shape
