import pytest
import numpy as np
import astropy.units as u
from astropy.coordinates import Galactic, SkyCoord
from astropy.time import Time
from types import SimpleNamespace
import healpy as hp

from scoords import Attitude, SpacecraftFrame
from histpy import HealpixAxis, Histogram

from cosipy import SpacecraftHistory
from cosipy.event_selection import GoodTimeInterval


class DummySpacecraftHistory(SpacecraftHistory):

    def __init__(self, colatitude, earth_occ=None, attitude=None):

        self._colatitude = np.asarray(colatitude, dtype=float)
        if earth_occ is None:
            earth_occ = np.zeros_like(self._colatitude, dtype=bool)
        self._earth_occ = np.asarray(earth_occ, dtype=bool)
        self._attitude = attitude  # optional: scoords.Attitude for from_region_cut
        self._occ_call_idx = 0     # row counter for 2D earth_occ (from_region_cut)

    @property
    def intervals_tstart(self):
        return Time([60970.0, 60971.0, 60972.0, 60973.0, 60974.0],
                    format='mjd', scale='utc')

    @property
    def intervals_tstop(self):
        return Time([60971.0, 60972.0, 60973.0, 60974.0, 60975.0],
                    format='mjd', scale='utc')

    @property
    def attitude(self):
        if self._attitude is not None:
            return self._attitude
        return SimpleNamespace(frame=Galactic())

    @property
    def location(self):
        return Galactic()

    def get_target_in_sc_frame(self, source):
        return SkyCoord(lon=np.zeros_like(self._colatitude),
                        lat=np.pi/2 - self._colatitude,
                        unit='rad', frame=SpacecraftFrame())

    def get_earth_occ(self, source):
        # 2D (n_on_pix, npoints): return one row per call (from_region_cut)
        if self._earth_occ.ndim == 2:
            row = self._earth_occ[self._occ_call_idx]
            self._occ_call_idx += 1
            return row
        # 1D (npoints,): return as-is (from_pointing_cut)
        return self._earth_occ


def _make_region(nside=4):
    """Boolean Histogram covering the GC pixel and its 8 neighbours."""
    ax = HealpixAxis(nside=nside, scheme='ring', coordsys='galactic', label='lb')
    gc_pix = ax.ang2pix(0.0, 0.0, lonlat=True)
    neighbors = hp.get_all_neighbours(nside, gc_pix, nest=False)
    on_pixels = np.append(neighbors[neighbors >= 0], gc_pix)
    region = Histogram(ax, contents=np.zeros(ax.npix, dtype=bool))
    region.contents[on_pixels] = True
    return region


def _make_attitude(lons_deg, lats_deg):
    """Minimal Attitude array from Galactic z-pointing directions."""
    lons = np.asarray(lons_deg, dtype=float)
    lats = np.asarray(lats_deg, dtype=float)
    z_pts = SkyCoord(l=lons * u.deg, b=lats * u.deg, frame=Galactic())
    x_pts = SkyCoord(l=(lons + 90) * u.deg, b=np.zeros_like(lons) * u.deg,
                     frame=Galactic())
    return Attitude.from_axes(x=x_pts, z=z_pts, frame=Galactic())


def _make_sc_region(region, pattern):
    """
    Build a DummySpacecraftHistory whose z-pointing follows `pattern`.

    Parameters
    ----------
    pattern : list of bool, length 5
        True = pointing inside the region, False = outside.
        Internally padded to 6 edge-points (one extra 'out' at the end).
    """
    ax = region.axis
    gc_pix = ax.ang2pix(0.0, 0.0, lonlat=True)
    lon_in,  lat_in  = ax.pix2ang(int(gc_pix), lonlat=True)
    lon_out, lat_out = ax.pix2ang(0,            lonlat=True)

    # attitude has npoints = nintervals + 1 edges; pad one extra point at the end
    lons = [lon_in if p else lon_out for p in pattern] + [lon_out]
    lats = [lat_in if p else lat_out for p in pattern] + [lat_out]
    return DummySpacecraftHistory(colatitude=[], attitude=_make_attitude(lons, lats))


def test_GTI(tmp_path):
    # test with 1 range
    gti = GoodTimeInterval(Time(60970.0, format='mjd', scale='utc'),
                           Time(60975.0, format='mjd', scale='utc'))

    # A single start/stop pair should produce exactly one GTI interval.
    assert len(gti) == 1

    # test with 2 ranges
    tstarts = Time([60970.0, 60980.0], format='mjd', scale='utc')
    tstops  = Time([60975.0, 60985.0], format='mjd', scale='utc')

    gti = GoodTimeInterval(tstarts, tstops)

    assert len(gti) == 2

    # check the values
    tstarts = gti.tstart_list
    tstops  = gti.tstop_list

    for i, (tstart, tstop) in enumerate(gti):
        # Iteration, list access, and direct indexing should all agree.
        assert tstart == tstarts[i] == gti[i][0]
        assert tstop  == tstops[i]  == gti[i][1]

    # save file
    gti.save_as_fits(tmp_path / 'gti.fits')
    gti_from_fits = GoodTimeInterval.from_fits(tmp_path / 'gti.fits')

    # FITS serialization should preserve the GTI boundaries exactly.
    assert np.all(tstarts == gti_from_fits.tstart_list)
    assert np.all(tstops  == gti_from_fits.tstop_list)

    # intersection

    # GTI1
    tstarts_1 = Time([60970.0, 60980.0], format='mjd', scale='utc')
    tstops_1  = Time([60975.0, 60985.0], format='mjd', scale='utc')
    gti1 = GoodTimeInterval(tstarts_1, tstops_1)

    # GTI2
    tstarts_2 = Time([60972.0, 60979.0], format='mjd', scale='utc')
    tstops_2  = Time([60977.0, 60983.0], format='mjd', scale='utc')
    gti2 = GoodTimeInterval(tstarts_2, tstops_2)

    # GTI3
    tstarts_3 = Time([60970.0], format='mjd', scale='utc')
    tstops_3  = Time([60990.0], format='mjd', scale='utc')
    gti3 = GoodTimeInterval(tstarts_3, tstops_3)

    # Intersection
    gti_intersection = GoodTimeInterval.intersection(gti1, gti2, gti3)

    # Intersections should keep only the time ranges shared by all GTIs.
    assert np.all(gti_intersection.tstart_list == Time([60972.0, 60980.0], format='mjd', scale='utc'))
    assert np.all(gti_intersection.tstop_list  == Time([60975.0, 60983.0], format='mjd', scale='utc'))

    # Intersection with no components
    # GTI4
    tstarts_4 = Time([60950.0], format='mjd', scale='utc')
    tstops_4  = Time([60960.0], format='mjd', scale='utc')
    gti4 = GoodTimeInterval(tstarts_4, tstops_4)

    gti_intersection_no_components = GoodTimeInterval.intersection(gti1, gti4)

    # Non-overlapping GTIs should produce an empty intersection.
    assert len(gti_intersection_no_components) == 0


def test_gti_contains_single_interval_boundaries():
    gti = GoodTimeInterval(
        Time([100.0], format='unix', scale='utc'),
        Time([102.0], format='unix', scale='utc'),
    )

    times = np.array([99.0, 100.0, 101.0, 102.0])

    # Half-open convention: start included, stop excluded.
    assert np.all(gti.contains(times) == [False, True, True, False])


def test_gti_contains_multiple_intervals_unsorted_times():
    gti = GoodTimeInterval(
        Time([100.0, 200.0], format='unix', scale='utc'),
        Time([102.0, 203.0], format='unix', scale='utc'),
    )

    times = np.array([201.0, 99.0, 101.0, 203.0, 200.0])

    assert np.all(gti.contains(times) == [True, False, True, False, True])


def test_gti_contains_empty():
    empty_time = Time([], format='unix', scale='utc')
    gti = GoodTimeInterval(empty_time, empty_time.copy())

    times = np.array([100.0, 101.0])

    assert np.all(gti.contains(times) == [False, False])


def test_gti_contains_astropy_time_input():
    gti = GoodTimeInterval(
        Time([60000.0, 60002.0], format='mjd', scale='utc'),
        Time([60001.0, 60003.0], format='mjd', scale='utc'),
    )

    times = Time([60000.5, 60001.5, 60002.5, 60003.0],
                 format='mjd', scale='utc')

    assert np.all(gti.contains(times) == [True, False, True, False])


def test_gti_from_pointing_cut():
    sc_history = DummySpacecraftHistory(
        colatitude=np.deg2rad([80.0, 40.0, 20.0, 70.0, 50.0, 10.0])
    )
    target = SkyCoord(l=0*u.deg, b=0*u.deg, frame='galactic')

    gti = GoodTimeInterval.from_pointing_cut(target, sc_history, 60*u.deg)

    # Adjacent in-FoV bins should be merged into GTI ranges.
    assert np.all(gti.tstart_list == Time([60971.0, 60974.0], format='mjd', scale='utc'))
    assert np.all(gti.tstop_list  == Time([60973.0, 60975.0], format='mjd', scale='utc'))


def test_gti_from_pointing_cut_with_earth_occultation():
    sc_history = DummySpacecraftHistory(
        colatitude=np.deg2rad([80.0, 40.0, 20.0, 70.0, 50.0, 10.0]),
        earth_occ=[False, True, False, False, False, False],
    )
    target = SkyCoord(l=0*u.deg, b=0*u.deg, frame='galactic')

    gti = GoodTimeInterval.from_pointing_cut(target, sc_history, 60*u.deg, earth_occ=True)

    # Earth-occulted bins should be removed before GTI ranges are built.
    assert np.all(gti.tstart_list == Time([60972.0, 60974.0], format='mjd', scale='utc'))
    assert np.all(gti.tstop_list  == Time([60973.0, 60975.0], format='mjd', scale='utc'))


def test_gti_from_pointing_cut_empty():
    sc_history = DummySpacecraftHistory(
        colatitude=np.deg2rad([80.0, 70.0, 65.0, 70.0, 80.0, 85.0])
    )
    target = SkyCoord(l=0*u.deg, b=0*u.deg, frame='galactic')

    gti = GoodTimeInterval.from_pointing_cut(target, sc_history, 60*u.deg)

    # If no bin passes the pointing cut, the GTI should be empty.
    assert len(gti) == 0


def test_from_region_cut_basic():
    """Bins [F,T,T,F,T] produce two GTI intervals."""
    region = _make_region()
    sc = _make_sc_region(region, [False, True, True, False, True])

    gti = GoodTimeInterval.from_region_cut(region, sc)

    assert len(gti) == 2
    assert np.all(gti.tstart_list == Time([60971., 60974.], format='mjd', scale='utc'))
    assert np.all(gti.tstop_list  == Time([60973., 60975.], format='mjd', scale='utc'))


def test_from_region_cut_empty():
    """No bin inside the region -> empty GTI."""
    region = _make_region()
    sc = _make_sc_region(region, [False, False, False, False, False])

    gti = GoodTimeInterval.from_region_cut(region, sc)

    assert len(gti) == 0


def test_from_region_cut_all_in():
    """All bins inside the region -> single GTI covering the full range."""
    region = _make_region()
    sc = _make_sc_region(region, [True, True, True, True, True])

    gti = GoodTimeInterval.from_region_cut(region, sc)

    assert len(gti) == 1
    assert gti.tstart_list[0] == Time(60970., format='mjd', scale='utc')
    assert gti.tstop_list[0]  == Time(60975., format='mjd', scale='utc')


def test_from_region_cut_earth_occ_all_vs_any():
    """
    Verify that earth_occ_mode='all' and 'any' produce different results.
 
    Setup: all 5 bins point into the region.
    - bin 1: all on-region pixels occulted  -> excluded by both 'all' and 'any'
    - bin 3: only pixel 0 occulted          -> excluded by 'all', kept by 'any'
 
    Expected:
      mode='all': [T,F,T,F,T] -> 3 intervals
      mode='any': [T,F,T,T,T] -> bin 2,3,4 merge -> 2 intervals
    """
    region = _make_region()
    n_on = int(np.sum(np.asarray(region[:], dtype=bool)))
 
    # earth_occ 2D: shape (n_on, npoints=6)
    earth_occ = np.zeros((n_on, 6), dtype=bool)
    earth_occ[:, 1] = True   # bin 1: all pixels occulted
    earth_occ[0, 3] = True   # bin 3: only pixel 0 occulted
 
    sc_all = _make_sc_region(region, [True, True, True, True, True])
    sc_all._earth_occ = earth_occ
 
    sc_any = _make_sc_region(region, [True, True, True, True, True])
    sc_any._earth_occ = earth_occ
 
    gti_all = GoodTimeInterval.from_region_cut(region, sc_all, earth_occ=True, earth_occ_mode='all')
    gti_any = GoodTimeInterval.from_region_cut(region, sc_any, earth_occ=True, earth_occ_mode='any')
 
    # mode='all': bin 1 and bin 3 both excluded -> 3 separate intervals
    assert len(gti_all) == 3
    assert np.all(gti_all.tstart_list == Time([60970., 60972., 60974.], format='mjd', scale='utc'))
    assert np.all(gti_all.tstop_list  == Time([60971., 60973., 60975.], format='mjd', scale='utc'))
 
    # mode='any': only bin 1 excluded; bin 3 kept -> bin 2,3,4 merge into one interval
    assert len(gti_any) == 2
    assert np.all(gti_any.tstart_list == Time([60970., 60972.], format='mjd', scale='utc'))
    assert np.all(gti_any.tstop_list  == Time([60971., 60975.], format='mjd', scale='utc'))


def test_from_region_cut_invalid_earth_occ_mode():
    """Invalid earth_occ_mode raises ValueError."""
    region = _make_region()
    sc = _make_sc_region(region, [True, True, True, True, True])

    with pytest.raises(ValueError, match="earth_occ_mode"):
        GoodTimeInterval.from_region_cut(region, sc, earth_occ=True, earth_occ_mode='invalid')
