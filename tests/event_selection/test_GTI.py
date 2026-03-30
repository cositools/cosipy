import pytest
import numpy as np
import astropy.units as u
from astropy.coordinates import Galactic, SkyCoord
from astropy.time import Time
from types import SimpleNamespace

from scoords import SpacecraftFrame

from cosipy import SpacecraftHistory
from cosipy.event_selection import GoodTimeInterval


class DummySpacecraftHistory(SpacecraftHistory):

    def __init__(self, colatitude, earth_occ=None):

        self._colatitude = np.asarray(colatitude, dtype=float)
        if earth_occ is None:
            earth_occ = np.zeros_like(self._colatitude, dtype=bool)
        self._earth_occ = np.asarray(earth_occ, dtype=bool)

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
        return SimpleNamespace(frame=Galactic())

    @property
    def location(self):
        return Galactic()

    def get_target_in_sc_frame(self, source):
        return SkyCoord(lon = np.zeros_like(self._colatitude), lat = np.pi/2 - self._colatitude, unit = 'rad', frame = SpacecraftFrame())

    def get_earth_occ(self, source):
        return self._earth_occ

def test_GTI(tmp_path):
    # test with 1 range
    gti = GoodTimeInterval(Time(60970.0, format='mjd', scale = 'utc'),
                           Time(60975.0, format='mjd', scale = 'utc'))

    # A single start/stop pair should produce exactly one GTI interval.
    assert len(gti) == 1
    
    # test with 2 ranges
    tstarts = Time([60970.0, 60980.0], format='mjd', scale = 'utc')
    tstops  = Time([60975.0, 60985.0], format='mjd', scale = 'utc')

    gti = GoodTimeInterval(tstarts, tstops)

    assert len(gti) == 2
    
    # check the values
    tstarts = gti.tstart_list
    tstops  = gti.tstop_list

    for i, (tstart, tstop) in enumerate(gti):
        # Iteration, list access, and direct indexing should all agree.
        assert tstart == tstarts[i] == gti[i][0]
        assert tstop  == tstops[i] == gti[i][1]

    # save file
    gti.save_as_fits(tmp_path / 'gti.fits')
    gti_from_fits = GoodTimeInterval.from_fits(tmp_path / 'gti.fits')

    # FITS serialization should preserve the GTI boundaries exactly.
    assert np.all(tstarts == gti_from_fits.tstart_list)
    assert np.all(tstops  == gti_from_fits.tstop_list)

    # intersection

    #GTI1
    tstarts_1 = Time([60970.0, 60980.0], format='mjd', scale = 'utc')
    tstops_1  = Time([60975.0, 60985.0], format='mjd', scale = 'utc')
    
    gti1 = GoodTimeInterval(tstarts_1, tstops_1)
    
    #GTI2
    tstarts_2 = Time([60972.0, 60979.0], format='mjd', scale = 'utc')
    tstops_2  = Time([60977.0, 60983.0], format='mjd', scale = 'utc')
    
    gti2 = GoodTimeInterval(tstarts_2, tstops_2)
    
    #GTI3
    tstarts_3 = Time([60970.0], format='mjd', scale = 'utc')
    tstops_3  = Time([60990.0], format='mjd', scale = 'utc')
    
    gti3 = GoodTimeInterval(tstarts_3, tstops_3)
    
    #Intersection
    gti_intersection = GoodTimeInterval.intersection(gti1, gti2, gti3)

    # Intersections should keep only the time ranges shared by all GTIs.
    assert np.all(gti_intersection.tstart_list == Time([60972.0, 60980.0], format='mjd', scale = 'utc'))
    assert np.all(gti_intersection.tstop_list  == Time([60975.0, 60983.0], format='mjd', scale = 'utc'))

    #Intersection with no components
    #GTI4
    tstarts_4 = Time([60950.0], format='mjd', scale = 'utc')
    tstops_4  = Time([60960.0], format='mjd', scale = 'utc')
    
    gti4 = GoodTimeInterval(tstarts_4, tstops_4)

    gti_intersection_no_components = GoodTimeInterval.intersection(gti1, gti4)

    # Non-overlapping GTIs should produce an empty intersection.
    assert len(gti_intersection_no_components) == 0


def test_gti_from_pointing_cut():
    sc_history = DummySpacecraftHistory(
        colatitude=np.deg2rad([80.0, 40.0, 20.0, 70.0, 50.0, 10.0])
    )
    target = SkyCoord(l=0*u.deg, b=0*u.deg, frame='galactic')

    gti = GoodTimeInterval.from_pointing_cut(target, sc_history, 60*u.deg)

    # Adjacent in-FoV bins should be merged into GTI ranges.
    assert np.all(gti.tstart_list == Time([60971.0, 60974.0], format='mjd', scale='utc'))
    assert np.all(gti.tstop_list == Time([60973.0, 60975.0], format='mjd', scale='utc'))


def test_gti_from_pointing_cut_with_earth_occultation():
    sc_history = DummySpacecraftHistory(
        colatitude=np.deg2rad([80.0, 40.0, 20.0, 70.0, 50.0, 10.0]),
        earth_occ=[False, True, False, False, False, False],
    )
    target = SkyCoord(l=0*u.deg, b=0*u.deg, frame='galactic')

    gti = GoodTimeInterval.from_pointing_cut(target, sc_history, 60*u.deg, earth_occ=True)

    # Earth-occulted bins should be removed before GTI ranges are built.
    assert np.all(gti.tstart_list == Time([60972.0, 60974.0], format='mjd', scale='utc'))
    assert np.all(gti.tstop_list == Time([60973.0, 60975.0], format='mjd', scale='utc'))


def test_gti_from_pointing_cut_empty():
    sc_history = DummySpacecraftHistory(
        colatitude=np.deg2rad([80.0, 70.0, 65.0, 70.0, 80.0, 85.0])
    )
    target = SkyCoord(l=0*u.deg, b=0*u.deg, frame='galactic')

    gti = GoodTimeInterval.from_pointing_cut(target, sc_history, 60*u.deg)

    # If no bin passes the pointing cut, the GTI should be empty.
    assert len(gti) == 0
