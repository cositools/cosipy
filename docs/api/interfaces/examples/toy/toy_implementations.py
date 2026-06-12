import itertools
from typing import Dict, Iterator, Iterable, Optional, Type, Union

from astromodels.sources import Source
import astropy.units as u
from astropy.time import Time
from astropy.units import Quantity

from cosipy.interfaces.background_interface import BackgroundDensityInterface
from cosipy.interfaces.data_interface import DataInterface, TimeTagEventDataInterface
from cosipy.interfaces.event_selection import EventSelectorInterface

from cosipy.interfaces import (BinnedDataInterface,
                               BinnedBackgroundInterface,
                               BinnedThreeMLModelFoldingInterface,
                               BinnedThreeMLSourceResponseInterface,
                               UnbinnedThreeMLSourceResponseInterface,
                               UnbinnedThreeMLModelFoldingInterface,
                               EventInterface,
                               ThreeMLSourceResponseInterface,
                               TimeTagEventInterface)

from histpy import Axis, Axes, Histogram
import numpy as np
from scipy.stats import norm, uniform

from threeML import Constant, PointSource, Model, JointLikelihood, DataList

from matplotlib import pyplot as plt

import copy

"""
This is an example on how to use the new interfaces.

To keep things simple, example itself is a toy model.
It a 1D model, with a Gaussian signal on top of a flat
uniform background. You can execute it until the end
to see a plot on how it looks like.

It looks nothing like COSI data, but
shows how generic the interfaces can be. 
"""

# ======== Create toy interfaces for this model ===========
class ToyEvent(TimeTagEventInterface, EventInterface):
    """
    Unit-less 1D data of a measurement called "x" (could be anything)
    """

    data_space_units = u.s

    def __init__(self, index:int, x:float, time:Time):
        self._id = index
        self._x = x
        self._jd1 = time.jd1
        self._jd2 = time.jd2

    @property
    def id(self):
        return self._id

    @property
    def x(self):
        return self._x

    @property
    def jd1(self):
        return self._jd1

    @property
    def jd2(self):
        return self._jd2

class ToyData(DataInterface):
    pass

class ToyEventDataStream(TimeTagEventDataInterface, ToyData):
    # This simulates reading event from file
    # Check that they are not being read twice

    def __init__(self, nevents_signal, nevents_bkg, min_value, max_value, tstart, tstop):

        rng = np.random.default_rng()

        signal = rng.normal(size=nevents_signal)
        bkg = rng.uniform(min_value, max_value, size=nevents_bkg)

        self._x = np.append(signal, bkg)

        self._tstart = tstart
        self._tstop = tstop

        dt = np.random.uniform(size=self._x.size)
        dt_sort = np.argsort(dt)
        self._x = self._x[dt_sort]
        dt = dt[dt_sort]

        self._timestamps = self._tstart + dt * u.day

    def __iter__(self) -> Iterator[ToyEvent]:
        print("Loading events!")
        for n,(x,t) in enumerate(zip(self._x, self._timestamps)):
            yield ToyEvent(n,x,t)

class ToyEventData(TimeTagEventDataInterface, ToyData):
    # Random data. Normal signal on top of uniform bkg

    event_type = ToyEvent

    def __init__(self, loader:ToyEventDataStream, selector:EventSelectorInterface = None):

        self._loader = [e for e,select in zip(loader, selector.select(loader)) if select]
        self._cached_iter = None
        self._nevents = None  # After selection

    def __iter__(self) -> Iterator[ToyEvent]:

        if self._cached_iter is None:
            # First call. Split. Keep one and return the other
            self._loader, self._cached_iter = itertools.tee(self._loader)
            return self._cached_iter
        else:
            # Following calls: tee the loader again
            self._loader, new_iter = itertools.tee(self._loader)
            return new_iter

    @property
    def nevents(self) -> int:
        if self._nevents is None:
            # Not cached yet
            self._nevents = sum(1 for _ in self)

        return self._nevents

    @property
    def x(self):
        return np.asarray([e.x for e in self])

    @property
    def jd1(self) -> Iterable[float]:
        return np.asarray([e.jd1 for e in self])

    @property
    def jd2(self) -> Iterable[float]:
        return np.asarray([e.jd2 for e in self])

class ToyBinnedData(BinnedDataInterface, ToyData):

    def __init__(self, data:Histogram):

        if data.ndim != 1:
            raise ValueError("ToyBinnedData only take a 1D histogram")

        if data.axis.label != 'x':
            raise ValueError("ToyBinnedData requires an axis labeled 'x'")

        self._data = data

    @property
    def data(self) -> Histogram:
        return self._data

    @property
    def axes(self) -> Axes:
        return self._data.axes

    def fill(self, event_data:ToyEventData):

        x = np.fromiter([e.x for e in event_data], dtype = float)

        self._data.fill(x)

class ToyBkg(BinnedBackgroundInterface, BackgroundDensityInterface):
    """
    Models a uniform background

    # Since the interfaces are Protocols, they don't *have*
    # to derive from the base class, but doing some helps
    # code readability, especially if you use an IDE.
    """

    event_data_type = ToyEventData

    def __init__(self, data: ToyEventData, duration:Quantity, axis:Axis):

        self._data = data
        self._duration = duration.to_value(u.s)
        self._unit_expectation = Histogram(axis)
        self._unit_expectation[:] = self._duration / self._unit_expectation.nbins
        self._norm = 1 # Hz

        self._unit_expectation_density = self._duration / (axis.hi_lim - axis.lo_lim)

    def set_parameters(self, **parameters:u.Quantity) -> None:
        self._norm = parameters['norm'].to_value(u.Hz)

    def expected_counts(self) -> float:
        return self._norm * self._duration

    def expectation_density(self, start:Optional[int] = None, stop:Optional[int] = None) -> Iterable[float]:

        for _ in itertools.islice(self._data, start, stop):
            yield self._norm * self._unit_expectation_density

    @property
    def parameters(self) -> Dict[str, u.Quantity]:
        return {'norm': u.Quantity(self._norm, u.Hz)}

    def expectation(self, copy = True) -> Histogram:

        # Always a copy
        return self._unit_expectation * self._norm

class ToyPointSourceResponse(BinnedThreeMLSourceResponseInterface, UnbinnedThreeMLSourceResponseInterface):
    """
    This models a Gaussian signal in 1D, centered at 0 and with std = 1.
    The normalization --the "flux"-- is the only free parameters
    """

    event_data_type = ToyEventData

    def __init__(self, data: Union[ToyEventData, ToyBinnedData], duration:Quantity):
        self._data = data
        self._source = None
        self._duration = duration.to_value(u.s)
        self._unit_expectation = Histogram(self.axes,
                                           contents= self._duration * np.diff(norm.cdf(self.axes[0].edges)))

    @property
    def axes(self):
        return self._data.axes

    def expected_counts(self) -> float:

        if self._source is None:
            raise RuntimeError("Set a source first")

        # Get the latest values of the flux
        # Remember that _model can be modified externally between calls.
        # This response doesn't have effective area or energy sensitivity. We're just using K as a rate
        ns_events = self._duration * self._source.spectrum.main.shape.k.value
        return ns_events

    def event_probability(self, start:Optional[int] = None, stop:Optional[int] = None) -> Iterable[float]:

        cache = norm.pdf([event.x for event in itertools.islice(self._data, start, stop)])

        for prob in cache:
            yield prob

        # Alternative version without cache (slower)
        # for event in itertools.islice(self._data, start, stop):
        #     yield norm.pdf(event.x)

    def set_source(self, source: Source):

        if not isinstance(source, PointSource):
            raise TypeError("I only know how to handle point sources!")

        self._source = source

    def expectation(self, copy = True) -> Histogram:

        if self._source is None:
            raise RuntimeError("Set a source first")

        # Get the latest values of the flux
        # Remember that _model can be modified externally between calls.
        # Always copies
        return self._unit_expectation * self._source.spectrum.main.shape.k.value

    def copy(self) -> "ToyPointSourceResponse":
        # We are not caching any results, so it's safe to do shallow copy without
        # re-initializing any member.
        return copy.copy(self)

class ToyModelFolding(BinnedThreeMLModelFoldingInterface, UnbinnedThreeMLModelFoldingInterface):

    event_data_type = ToyEventData

    def __init__(self, data:Union[ToyBinnedData,ToyEventData], psr: ToyPointSourceResponse):

        self._data = data
        self._model = None

        self._psr = psr
        self._psr_copies = {}

    @property
    def axes(self):
        return self._psr.axes

    def expected_counts(self) -> float:

        ncounts = 0

        for source_name,psr in self._psr_copies.items():
            ncounts += psr.expected_counts()

        return ncounts

    def expectation_density(self, start:Optional[int] = None, stop:Optional[int] = None) -> Iterable[float]:

        self._cache_psr_copies()

        if not self._psr_copies:
            for _ in itertools.islice(self._data, start, stop):
                yield 0
        else:
            for expectation in zip(*[p.expectation_density() for p in self._psr_copies.values()]):
                yield np.sum(expectation)

    def set_model(self, model: Model):

        self._model = model

    def _cache_psr_copies(self):

        new_psr_copies = {}

        for name,source in self._model.sources.items():

            if name in self._psr_copies:
                # Use cache
                new_psr_copies[name] = self._psr_copies[name]

            psr_copy = self._psr.copy()
            psr_copy.set_source(source)

            new_psr_copies[name] = psr_copy

        self._psr_copies = new_psr_copies

    def expectation(self, copy = True) -> Histogram:

        self._cache_psr_copies()

        expectation = Histogram(self.axes)

        for source_name,psr in self._psr_copies.items():
            expectation += psr.expectation(copy = False)

        # Always a copy
        return expectation

# ======= Actual code. This is how the "tutorial" will look like ================