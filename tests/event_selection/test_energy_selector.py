from typing import Iterable

import numpy as np
import astropy.units as u
import pytest

from cosipy.event_selection.energy_selection import EnergySelector
from cosipy.event_selection.distance_selection import DistanceSelector
from cosipy.interfaces.data_interface import EventDataWithEnergyInterface
from cosipy.interfaces.event import EventWithEnergyInterface
from cosipy.util.iterables import asarray

# Dummy events
energies_keV = np.array([50., 100., 150., 200., 250., 300.])


class DummyEnergyEventData(EventDataWithEnergyInterface):

    @property
    def energy_keV(self) -> Iterable[float]:
        return energies_keV


class DummyEnergyEvent(EventWithEnergyInterface):

    def __init__(self, energy_keV, id=None):
        self._energy_keV = energy_keV
        self._id = id

    @property
    def energy_keV(self) -> float:
        return self._energy_keV

    @property
    def id(self):
        return self._id


events = DummyEnergyEventData()


def test_energy_selector_default_keeps_everything():
    selector = EnergySelector()

    mask = asarray(selector.select(events), dtype=bool)

    assert np.all(mask)


def test_energy_selector_single_range_shape_2():
    selector = EnergySelector(u.Quantity([100., 250.], u.keV))

    mask = asarray(selector.select(events), dtype=bool)

    # min is inclusive, max is exclusive
    expected = np.array([False, True, True, True, False, False])

    assert np.array_equal(mask, expected)


def test_energy_selector_multiple_disjoint_ranges():
    selector = EnergySelector(u.Quantity([[0., 120.], [180., 260.]], u.keV))

    mask = asarray(selector.select(events), dtype=bool)

    # kept if inside ANY range
    expected = np.array([True, True, False, True, True, False])

    assert np.array_equal(mask, expected)


def test_energy_selector_merges_overlapping_ranges_on_construction():
    selector = EnergySelector(u.Quantity([[0., 150.], [100., 300.]], u.keV))

    np.testing.assert_allclose(selector.energy_ranges_keV, [[0., 300.]])


def test_energy_selector_merges_adjacent_ranges_on_construction():
    selector = EnergySelector(u.Quantity([[0., 100.], [100., 200.]], u.keV))

    np.testing.assert_allclose(selector.energy_ranges_keV, [[0., 200.]])


def test_energy_selector_drops_empty_ranges_on_construction():
    selector = EnergySelector(u.Quantity([[100., 100.], [200., 100.], [50., 150.]], u.keV))

    np.testing.assert_allclose(selector.energy_ranges_keV, [[50., 150.]])


def test_energy_ranges_and_energy_ranges_keV_are_consistent():
    selector = EnergySelector(u.Quantity([[50., 150.], [200., 300.]], u.keV))

    np.testing.assert_allclose(selector.energy_ranges.to_value(u.keV), selector.energy_ranges_keV)
    assert selector.energy_ranges.unit == u.keV


def test_union_combines_ranges():
    a = EnergySelector(u.Quantity([0., 100.], u.keV))
    b = EnergySelector(u.Quantity([200., 300.], u.keV))

    combined = a.union(b)

    np.testing.assert_allclose(combined.energy_ranges_keV, [[0., 100.], [200., 300.]])

    mask = asarray(combined.select(events), dtype=bool)
    expected = np.array([True, False, False, True, True, False])
    assert np.array_equal(mask, expected)


def test_union_merges_overlapping_result():
    a = EnergySelector(u.Quantity([0., 150.], u.keV))
    b = EnergySelector(u.Quantity([100., 300.], u.keV))

    combined = a.union(b)

    np.testing.assert_allclose(combined.energy_ranges_keV, [[0., 300.]])


def test_intersect_narrows_ranges():
    a = EnergySelector(u.Quantity([0., 200.], u.keV))
    b = EnergySelector(u.Quantity([100., 300.], u.keV))

    combined = a.intersect(b)

    np.testing.assert_allclose(combined.energy_ranges_keV, [[100., 200.]])

    mask = asarray(combined.select(events), dtype=bool)
    expected = np.array([False, True, True, False, False, False])
    assert np.array_equal(mask, expected)


def test_intersect_disjoint_ranges_selects_nothing():
    a = EnergySelector(u.Quantity([0., 100.], u.keV))
    b = EnergySelector(u.Quantity([200., 300.], u.keV))

    combined = a.intersect(b)

    assert combined.energy_ranges_keV.shape == (0, 2)

    mask = asarray(combined.select(events), dtype=bool)
    assert not np.any(mask)


def test_except_removes_overlapping_range():
    a = EnergySelector(u.Quantity([0., 300.], u.keV))
    b = EnergySelector(u.Quantity([100., 200.], u.keV))

    result = a.except_(b)

    np.testing.assert_allclose(result.energy_ranges_keV, [[0., 100.], [200., 300.]])


def test_except_disjoint_range_is_noop():
    a = EnergySelector(u.Quantity([0., 100.], u.keV))
    b = EnergySelector(u.Quantity([200., 300.], u.keV))

    result = a.except_(b)

    np.testing.assert_allclose(result.energy_ranges_keV, a.energy_ranges_keV)


def test_except_superset_range_empties_selector():
    a = EnergySelector(u.Quantity([100., 200.], u.keV))
    b = EnergySelector(u.Quantity([0., 300.], u.keV))

    result = a.except_(b)

    assert result.energy_ranges_keV.shape == (0, 2)


@pytest.mark.parametrize("method", ["union", "intersect", "except_"])
def test_combinators_raise_on_non_energy_selector(method):
    a = EnergySelector()
    not_a_selector = DistanceSelector()

    with pytest.raises(TypeError):
        getattr(a, method)(not_a_selector)


def test_energy_selector_single_event():
    selector = EnergySelector(u.Quantity([100., 250.], u.keV))

    inside = DummyEnergyEvent(energy_keV=150., id=1)
    outside = DummyEnergyEvent(energy_keV=10., id=2)

    assert selector.select(inside) == True
    assert selector.select(outside) == False
