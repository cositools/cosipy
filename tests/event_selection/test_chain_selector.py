import numpy as np
import astropy.units as u
from astropy.time import Time

from cosipy.data_io.EmCDSUnbinnedData import TimeTagEmCDSDistanceEventDataInSCFrameFromArrays
from cosipy.event_selection.chain_selector import ChainEventSelectors
from cosipy.event_selection.distance_selection import DistanceSelector
from cosipy.event_selection.time_selection import TimeSelector
from cosipy.util.iterables import asarray


def test_chain_distance_and_time_selectors():
    n = 6

    jd1 = np.full(n, 2460000.0)
    jd2 = np.linspace(0, 0.5, n)  # time-ordered, as required by TimeSelector
    energy_keV = np.linspace(100, 600, n)
    scatt_angle_rad = np.linspace(0.1, 1.0, n)
    scatt_lon_rad = np.linspace(0, 1, n)
    scatt_lat_rad = np.linspace(-0.5, 0.5, n)
    distance_cm = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

    data = TimeTagEmCDSDistanceEventDataInSCFrameFromArrays(
        jd1, jd2, energy_keV, scatt_lon_rad, scatt_lat_rad, scatt_angle_rad, distance_cm)

    tstart = Time(jd1[0], 0.1, format='jd')
    tstop = Time(jd1[0], 0.4, format='jd')

    time_selector = TimeSelector(tstart=tstart, tstop=tstop)
    distance_selector = DistanceSelector(min_distance=2. * u.cm)

    # Same construction order reported to crash: distance selector first,
    # time selector second.
    selector = ChainEventSelectors(distance_selector, time_selector)

    mask = asarray(selector.select(data), dtype=bool)

    expected = (jd2 >= 0.1) & (jd2 < 0.4) & (distance_cm >= 2.)

    assert np.array_equal(mask, expected)


def test_chain_selector_matches_individual_selectors():
    # Chaining should be equivalent to combining each selector's own mask
    # with a logical AND.
    n = 8

    jd1 = np.full(n, 2460000.0)
    jd2 = np.linspace(0, 0.7, n)
    energy_keV = np.linspace(100, 800, n)
    scatt_angle_rad = np.linspace(0.1, 1.0, n)
    scatt_lon_rad = np.linspace(0, 1, n)
    scatt_lat_rad = np.linspace(-0.5, 0.5, n)
    distance_cm = np.linspace(0, 7, n)

    data = TimeTagEmCDSDistanceEventDataInSCFrameFromArrays(
        jd1, jd2, energy_keV, scatt_lon_rad, scatt_lat_rad, scatt_angle_rad, distance_cm)

    time_selector = TimeSelector(tstart=Time(jd1[0], 0.2, format='jd'))
    distance_selector = DistanceSelector(min_distance=1. * u.cm, max_distance=5. * u.cm)

    time_mask = asarray(time_selector.select(data), dtype=bool)
    distance_mask = asarray(distance_selector.select(data), dtype=bool)

    chained_mask = asarray(ChainEventSelectors(time_selector, distance_selector).select(data), dtype=bool)

    assert np.array_equal(chained_mask, time_mask & distance_mask)
