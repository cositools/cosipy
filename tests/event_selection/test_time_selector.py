from typing import Iterable

import numpy as np
import pytest
from astropy.time import Time
from astropy.utils.metadata.utils import dtype

from cosipy.event_selection import GoodTimeInterval
from cosipy.event_selection.time_selection import TimeSelector
from cosipy.interfaces import TimeTagEventInterface, TimeTagEventDataInterface
from cosipy.util.iterables import asarray

# Dummy events
times = Time(60000 + np.arange(0,15), format = 'jd')

class DummyTimeTagEventData(TimeTagEventDataInterface):

    @property
    def jd1(self) -> Iterable[float]:
        return times.jd1

    @property
    def jd2(self) -> Iterable[float]:
        return times.jd2

class DummyTimeTagEvent(TimeTagEventInterface):

    def __init__(self, jd1, jd2):
        self._jd1 = jd1
        self._jd2 = jd2

    @property
    def jd1(self) -> float:
        return self._jd1

    @property
    def jd2(self) -> float:
        return self._jd2

class DummyTimeTagEventDataIterable(TimeTagEventDataInterface):

    def __iter__(self):
        for jd2, jd1 in zip(times.jd1, times.jd2):
            yield DummyTimeTagEvent(jd1, jd2)

events = DummyTimeTagEventData()
events_iter = DummyTimeTagEventDataIterable()

def test_time_selector():

    tstart = Time(60000. + np.asarray([0, 5]), format='jd')
    tstop = Time(60000. + np.asarray([1, 10]), format='jd')
    selected = np.asarray(
        [True, False, False, False, False, True, True, True, True, True, False, False, False, False, False])

    # From array
    selector = TimeSelector(tstart, tstop)

    mask = selector.select(events) # Returned as array

    assert np.all(mask == selected)

    # From iter. early_stop by default
    selector = TimeSelector(tstart, tstop, batch_size=2)

    mask = asarray(selector.select(events_iter), dtype=bool)  # Yield another iterator

    assert np.all(mask == selected[:12])

    # From iter. no early_stop
    selector = TimeSelector(tstart, tstop, batch_size=2)

    mask = asarray(selector.select(events_iter, early_stop=False), dtype=bool) # Yield another iterator

    assert np.all(mask == selected)

    # From GTI
    gti = GoodTimeInterval(tstart, tstop)
    selector = TimeSelector.from_gti(gti, batch_size=2)

    mask = asarray(selector.select(events_iter), dtype=bool)  # Yield another iterator

    assert np.all(mask == selected[:12])

    # Only tstart
    tstart = Time(60000. + 5, format='jd')
    tstop = None
    selected = np.asarray(
        [False, False, False, False, False, True, True, True, True, True, True, True, True, True, True])

    selector = TimeSelector(tstart, tstop)

    mask = selector.select(events) # Returned as array

    assert np.all(mask == selected)

    # Only tstop
    tstart = None
    tstop = Time(60000. + 5, format='jd')
    selected = np.asarray(
        [True, True, True, True, True, False, False, False, False, False, False, False, False, False, False])

    selector = TimeSelector(tstart, tstop)

    mask = selector.select(events) # Returned as array

    assert np.all(mask == selected)

    # All selected
    tstart = None
    tstop = None
    selected = np.asarray(
        [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True])

    selector = TimeSelector(tstart, tstop)

    mask = selector.select(events) # Returned as array

    assert np.all(mask == selected)

    # Bad init

    tstart = Time(60000. + np.asarray([0, 5]), format='jd')
    tstop = Time(60000. + np.asarray([1, 10]), format='jd')

    # tstart and tstop must both be scalar or both be list.
    with pytest.raises(ValueError):
        TimeSelector(tstart[0], tstop)

    with pytest.raises(ValueError):
        TimeSelector(tstart, tstop[0])

    # tstart and tstop must have same length.
    with pytest.raises(ValueError):
        TimeSelector(tstart, tstop[:-1])

    # If one is None, the other must be a scalar
    with pytest.raises(ValueError):
        TimeSelector(None, tstop)

    with pytest.raises(ValueError):
        TimeSelector(tstart, None)



