from typing import Iterable

import numpy as np
import astropy.units as u

from cosipy.data_io.EmCDSUnbinnedData import TimeTagEmCDSDistanceEventInSCFrame
from cosipy.event_selection.distance_selection import DistanceSelector
from cosipy.interfaces.data_interface import TimeTagEmCDSDistanceEventDataInSCFrameInterface
from cosipy.util.iterables import asarray

# Dummy events
distances_cm = np.array([0., 1., 2., 3., 4., 5.])


class DummyDistanceEventData(TimeTagEmCDSDistanceEventDataInSCFrameInterface):

    @property
    def distance_cm(self) -> Iterable[float]:
        return distances_cm


events = DummyDistanceEventData()


def test_distance_selector_default_keeps_everything():
    selector = DistanceSelector()

    mask = asarray(selector.select(events), dtype=bool)

    assert np.all(mask)


def test_distance_selector_bounds():
    selector = DistanceSelector(min_distance=1. * u.cm, max_distance=4. * u.cm)

    mask = asarray(selector.select(events), dtype=bool)

    # min is inclusive, max is exclusive
    expected = np.array([False, True, True, True, False, False])

    assert np.array_equal(mask, expected)


def test_distance_selector_min_only():
    selector = DistanceSelector(min_distance=3. * u.cm)

    mask = asarray(selector.select(events), dtype=bool)

    expected = np.array([False, False, False, True, True, True])

    assert np.array_equal(mask, expected)


def test_distance_selector_max_only():
    selector = DistanceSelector(max_distance=3. * u.cm)

    mask = asarray(selector.select(events), dtype=bool)

    expected = np.array([True, True, True, False, False, False])

    assert np.array_equal(mask, expected)


def test_distance_selector_single_event():
    selector = DistanceSelector(min_distance=1. * u.cm, max_distance=4. * u.cm)

    inside = TimeTagEmCDSDistanceEventInSCFrame(0., 0., 100., 0.1, 0., 0., distance_cm=2.5, event_id=1)
    outside = TimeTagEmCDSDistanceEventInSCFrame(0., 0., 100., 0.1, 0., 0., distance_cm=10., event_id=2)

    assert selector.select(inside) == True
    assert selector.select(outside) == False
