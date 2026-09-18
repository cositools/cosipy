from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import astropy.units as u
from histpy import Axis, Axes, HealpixAxis, Histogram
from scoords import SpacecraftFrame

from cosipy.event_selection.distance_selection import DistanceSelector
from cosipy.event_selection.energy_selection import EnergySelector
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


def _make_uniform_irf_hist(nside=2, n_eps=8, eps_range=(-0.8, 0.8)):
    """A single-Ei-bin irf histogram with all contents equal to 1 cm^2
    and a uniform Epsilon axis, so the selection fraction for any cut
    has an exact, hand-computable closed form (constant density ->
    the piecewise-linear integration model is exact, not just an
    approximation)."""

    axes = Axes([
        HealpixAxis(nside=nside, scheme='ring', coordsys=SpacecraftFrame(), label='NuLambda'),
        Axis([100., 1000.] * u.keV, label='Ei'),
        Axis(np.linspace(*eps_range, n_eps + 1), label='Epsilon'),
        Axis(np.linspace(0, 180, 7) * u.deg, label='Phi'),
        Axis(np.linspace(-90, 90, 5) * u.deg, label='Theta'),
        PolarizationAxis(np.linspace(0, 360, 9) * u.deg, convention=StereographicConvention(), label='Zeta'),
    ])

    return Histogram(axes, contents=np.ones(axes.nbins), unit=u.cm * u.cm)


class TestEnergySelections:
    """selections=... on IRFRelativeHistUnpolarized should scale
    _tot_aeff, per (NuLambda, Ei), by the fraction of effective area
    whose measured energy falls inside the selection -- and never
    touch _diff_aeff."""

    def test_non_energy_selector_raises_type_error(self):
        with pytest.raises(TypeError):
            IRFRelativeHistUnpolarized(_make_irf_hist(), selections=DistanceSelector())

    def test_selections_none_matches_omitting_the_parameter(self):
        baseline = IRFRelativeHistUnpolarized(_make_irf_hist(seed=1))
        explicit_none = IRFRelativeHistUnpolarized(_make_irf_hist(seed=1), selections=None)

        np.testing.assert_array_equal(explicit_none._tot_aeff.contents, baseline._tot_aeff.contents)

    def test_full_range_cut_leaves_tot_aeff_unchanged(self):
        """A cut wide enough to enclose every Ei bin's full measured-energy
        range should hit the exact fast path -- _tot_aeff byte-for-byte
        unchanged, not just approximately."""

        baseline = IRFRelativeHistUnpolarized(_make_irf_hist(seed=2))
        full_range = EnergySelector(u.Quantity([0., 1e9], u.keV))
        cut = IRFRelativeHistUnpolarized(_make_irf_hist(seed=2), selections=full_range)

        np.testing.assert_array_equal(cut._tot_aeff.contents, baseline._tot_aeff.contents)

    def test_uniform_binning_exact_fraction_bin_aligned_cut(self):
        """Ei center is 550 keV; Epsilon edges are spaced by 0.2, so a
        [330, 550) keV cut is exactly eps in [-0.4, 0.0) -- two whole
        bins, no boundary approximation."""

        n_phi, n_theta, n_zeta, n_eps = 6, 4, 8, 8
        model = IRFRelativeHistUnpolarized(
            _make_uniform_irf_hist(n_eps=n_eps), copy=False,
            selections=EnergySelector(u.Quantity([330., 550.], u.keV)))

        expected = 2 * n_phi * n_theta * n_zeta
        np.testing.assert_allclose(model._tot_aeff.contents, expected)

    def test_uniform_binning_exact_fraction_sub_bin_cut(self):
        """[385, 550) keV -> eps in [-0.3, 0.0), i.e. half of the
        [-0.4,-0.2) bin plus the whole [-0.2,0) bin. Density is uniform
        (content=1 in every bin), so the exact answer is just
        density * width, regardless of the boundary falling mid-bin."""

        n_phi, n_theta, n_zeta = 6, 4, 8
        model = IRFRelativeHistUnpolarized(
            _make_uniform_irf_hist(), copy=False,
            selections=EnergySelector(u.Quantity([385., 550.], u.keV)))

        density_per_solid_angle_bin = 1 / 0.2  # content=1 per Epsilon bin of width 0.2
        expected = density_per_solid_angle_bin * 0.3 * n_phi * n_theta * n_zeta
        np.testing.assert_allclose(model._tot_aeff.contents, expected)

    def test_tuple_of_selectors_combines_via_union(self):
        """A tuple of EnergySelectors passed to selections= is OR'd
        together (each entry is an acceptable window), matching
        EnergySelector.union -- not intersected/AND'd."""

        a = EnergySelector(u.Quantity([100., 500.], u.keV))
        b = EnergySelector(u.Quantity([400., 900.], u.keV))  # overlaps a

        via_tuple = IRFRelativeHistUnpolarized(_make_irf_hist(seed=3), selections=(a, b))
        via_union = IRFRelativeHistUnpolarized(_make_irf_hist(seed=3), selections=a.union(b))
        via_intersect = IRFRelativeHistUnpolarized(_make_irf_hist(seed=3), selections=a.intersect(b))

        np.testing.assert_allclose(via_tuple._tot_aeff.contents, via_union._tot_aeff.contents)
        assert not np.allclose(via_tuple._tot_aeff.contents, via_intersect._tot_aeff.contents)

    def test_multi_range_selector_sums_disjoint_windows(self):
        """A single EnergySelector with two disjoint ranges should give
        the same total as summing each range's own single-range cut
        (they don't overlap, so no double-counting)."""

        n_phi, n_theta, n_zeta = 6, 4, 8

        range_a = EnergySelector(u.Quantity([330., 470.], u.keV))  # eps in [-0.4, -0.145...)
        range_b = EnergySelector(u.Quantity([600., 750.], u.keV))  # eps in [0.0909..., 0.3636...)
        both = range_a.union(range_b)

        model_a = IRFRelativeHistUnpolarized(_make_uniform_irf_hist(), copy=False, selections=range_a)
        model_b = IRFRelativeHistUnpolarized(_make_uniform_irf_hist(), copy=False, selections=range_b)
        model_both = IRFRelativeHistUnpolarized(_make_uniform_irf_hist(), copy=False, selections=both)

        np.testing.assert_allclose(model_both._tot_aeff.contents,
                                    model_a._tot_aeff.contents + model_b._tot_aeff.contents)

    def test_aeff_and_selections_together_interpolates_onto_aeff_grid(self):
        no_cut = IRFRelativeHistUnpolarized(_make_irf_hist(seed=4), aeff=_make_aeff_hist(seed=5))
        cut = IRFRelativeHistUnpolarized(
            _make_irf_hist(seed=4), aeff=_make_aeff_hist(seed=5),
            selections=EnergySelector(u.Quantity([150., 400.], u.keV)))

        assert cut._tot_aeff.contents.shape == no_cut._tot_aeff.contents.shape
        assert np.all(cut._tot_aeff.contents <= no_cut._tot_aeff.contents + 1e-9)
        assert cut._tot_aeff.contents.sum() < no_cut._tot_aeff.contents.sum()

    def test_diff_aeff_unaffected_by_selections(self):
        no_cut = IRFRelativeHistUnpolarized(_make_irf_hist(seed=6))
        cut = IRFRelativeHistUnpolarized(
            _make_irf_hist(seed=6),
            selections=EnergySelector(u.Quantity([150., 400.], u.keV)))

        np.testing.assert_array_equal(cut._diff_aeff.contents, no_cut._diff_aeff.contents)
