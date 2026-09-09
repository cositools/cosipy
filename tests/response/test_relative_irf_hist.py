from concurrent.futures import ThreadPoolExecutor

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


class _FakePhotonList:
    def __init__(self, lon_rad, lat_rad, energy_keV):
        self.direction_lon_rad_sc = lon_rad
        self.direction_lat_rad_sc = lat_rad
        self.energy_keV = energy_keV


class _FakeEventData:
    def __init__(self, lon_rad, lat_rad, phi_rad, energy_keV):
        self.scattered_lon_rad_sc = lon_rad
        self.scattered_lat_rad_sc = lat_rad
        self.scattering_angle_rad = phi_rad
        self.energy_keV = energy_keV


def _make_photons_and_events(n, seed=42):
    """Random, but physically-plausible-enough photon/event arrays: energies
    within the irf's Ei axis range, directions on the sphere, scattering
    angles in [0, pi]. Values don't need to be physically self-consistent
    (e.g. the event doesn't need to be a real Compton-scattered version of
    the photon) -- these tests only check that parallelizing the interp()
    call doesn't change its result, not that the result is physically
    correct (that's covered elsewhere by the class's other tests/usage)."""

    rng = np.random.default_rng(seed)

    photon_lon = rng.uniform(-np.pi, np.pi, n)
    photon_lat = rng.uniform(-np.pi / 2, np.pi / 2, n)
    photon_energy = rng.uniform(100, 1000, n)

    psichi_lon = rng.uniform(-np.pi, np.pi, n)
    psichi_lat = rng.uniform(-np.pi / 2, np.pi / 2, n)
    phi_kin = rng.uniform(0, np.pi, n)
    measured_energy = rng.uniform(100, 1000, n)

    photons = _FakePhotonList(photon_lon, photon_lat, photon_energy)
    events = _FakeEventData(psichi_lon, psichi_lat, phi_kin, measured_energy)

    return photons, events


class TestParallelInterp:
    """nthreads > 1 (with a small npoints_parallel_thresh so the parallel
    path actually triggers at small, fast-to-test N) must give the same
    results as
    nthreads=1 -- chunking the input and concatenating the per-chunk
    results should be numerically transparent, since each point's
    interpolation is independent of every other point's."""

    def test_executor_only_created_when_nthreads_greater_than_one(self):
        serial_model = IRFRelativeHistUnpolarized(_make_irf_hist(), nthreads=1)
        assert serial_model._executor is None

        parallel_model = IRFRelativeHistUnpolarized(_make_irf_hist(), nthreads=4)
        assert isinstance(parallel_model._executor, ThreadPoolExecutor)

    def test_effective_area_cm2_matches_serial(self):
        # Separate Histogram instances per model rather than sharing one
        # via .copy(): Histogram.copy() doesn't deep-copy its individual
        # Axis objects, and IRFRelativeHistUnpolarized.__init__ mutates
        # them in place (unit standardization) even when copy=True, so two
        # models built from copies of the *same* underlying histogram
        # would corrupt each other's axes.
        serial_model = IRFRelativeHistUnpolarized(_make_irf_hist(), nthreads=1)
        # npoints_parallel_thresh=1 -> threshold is nthreads*thresh=4
        # points, so the parallel path triggers well below the N used here.
        parallel_model = IRFRelativeHistUnpolarized(_make_irf_hist(), nthreads=4, npoints_parallel_thresh=1)

        # N=37: deliberately not a multiple of nthreads, to exercise
        # np.array_split's uneven-chunk behavior.
        photons, _ = _make_photons_and_events(37)

        serial_result = serial_model._effective_area_cm2(photons)
        parallel_result = parallel_model._effective_area_cm2(photons)

        np.testing.assert_allclose(parallel_result, serial_result)

    def test_differential_effective_area_cm2_matches_serial(self):
        serial_model = IRFRelativeHistUnpolarized(_make_irf_hist(), nthreads=1)
        parallel_model = IRFRelativeHistUnpolarized(_make_irf_hist(), nthreads=4, npoints_parallel_thresh=1)

        photons, events = _make_photons_and_events(37)

        serial_result = serial_model._differential_effective_area_cm2(photons, events)
        parallel_result = parallel_model._differential_effective_area_cm2(photons, events)

        np.testing.assert_allclose(parallel_result, serial_result)

    def test_matches_serial_below_npoints_parallel_thresh(self):
        """Same as above, but with N below nthreads*npoints_parallel_thresh,
        so the parallel model's _parallel_interp() takes its direct,
        single-threaded branch -- should trivially still match, and
        confirms that branch is exercised too."""

        serial_model = IRFRelativeHistUnpolarized(_make_irf_hist(), nthreads=1)
        parallel_model = IRFRelativeHistUnpolarized(_make_irf_hist(), nthreads=4, npoints_parallel_thresh=1000)

        photons, events = _make_photons_and_events(5)

        np.testing.assert_allclose(
            parallel_model._effective_area_cm2(photons),
            serial_model._effective_area_cm2(photons))
        np.testing.assert_allclose(
            parallel_model._differential_effective_area_cm2(photons, events),
            serial_model._differential_effective_area_cm2(photons, events))
