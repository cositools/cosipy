import operator
from typing import Protocol, runtime_checkable, Dict, Any, Generator, Iterable, Optional, Union, Iterator, ClassVar, \
    Type, Tuple

import histpy
import numpy as np
from histpy import Axes

from cosipy.interfaces import BinnedDataInterface, EventDataInterface, DataInterface, EventInterface
from cosipy.util.iterables import asarray

__all__ = [
    "ExpectationDensityInterface",
           "BinnedExpectationInterface"
           ]


@runtime_checkable
class ExpectationInterface(Protocol):
    pass

@runtime_checkable
class BinnedExpectationInterface(ExpectationInterface, Protocol):

    @property
    def axes(self) -> histpy.Axes:
        """
        Axes of expectation
        """

    def expectation(self, copy: Optional[bool])->histpy.Histogram:
        """

        Parameters
        ----------
        copy:
            If True (default), it will return an array that the user if free to modify.
            Otherwise, it will result a reference, possible to the cache, that
            the user should not modify

        Returns
        -------

        """

@runtime_checkable
class ExpectationDensityInterface(ExpectationInterface, Protocol):

    # The event class that the instance handles
    event_data_type = EventDataInterface

    @property
    def event_type(self) -> Type[EventInterface]:
        return self.event_data_type.event_type

    def expected_counts(self) -> float:
        """
        Total expected counts
        """

    def event_probability(self) -> Iterable[float]:
        """
        Return the probability of obtaining the observed set of measurement of each event,
        given that the event was detected. It equals the expectation density times ncounts

        The units of the output the inverse of the phase space of the event_type data space.
        e.g. if the event measured energy in keV, the units of output of this function are implicitly 1/keV

        This is provided as a helper function assuming the child classes implemented expectation_density
        """

        # Guard to avoid infinite recursion in incomplete child classes
        cls = type(self)
        if (
                cls.event_probability is ExpectationDensityInterface.event_probability
                and
                cls.expectation_density is ExpectationDensityInterface.expectation_density):
            raise NotImplementedError("Implement event_probability and/or expectation_density")

        ncounts = self.expected_counts()
        return [expectation/ncounts for expectation in self.expectation_density()]

    def expectation_density(self) -> Iterable[float]:
        """
        Return the expected number of counts density from the start-th event
        to the stop-th event. This equals the event probabiliy times the number of events

        This is provided as a helper function assuming the child classes implemented event_probability

        Parameters
        ----------
        start
        stop

        Returns
        -------

        """
        # Guard to avoid infinite recursion in incomplete child classes
        cls = type(self)
        if (
                cls.event_probability is ExpectationDensityInterface.event_probability
                and
                cls.expectation_density is ExpectationDensityInterface.expectation_density):
            raise NotImplementedError("Implement event_probability and/or expectation_density")

        ncounts = self.expected_counts()
        return [prob*ncounts for prob in self.event_probability()]

class SumExpectationDensity(ExpectationDensityInterface):
    """
    Convenience class to sum multiple ExpectationDensityInterface implementation
    """

    def __init__(self, *expectations:Tuple[ExpectationDensityInterface, None], vectorize:bool = True):
        """
        Parameters
        ----------
        expectations: Other ExpectationDensityInterface implementations
        vectorize: If True (default), it will first cache all the individual expectations on numpy arrays, and then it will sum
        them up using numpy's method. The output will also be a numpy. If False, it will query one element from each
        expectation object a time and sum them up. The output in this case is an Generator.
        """
        # Allow None for convenience, we  should remove it
        self._expectations = tuple(ex for ex in expectations if ex is not None)

        self._event_type = expectations[0].event_type
        
        self._vectorize = vectorize

        for ex in expectations:
            if ex.event_type is not self._event_type:
                raise TypeError("All expectations should have the same event type")

    @property
    def event_type(self) -> Type[EventInterface]:
        """
        The event class that the implementation can handle
        """
        return self._event_type

    def expected_counts(self) -> float:
        """
        Total expected counts
        """
        return sum(ex.expected_counts() for ex in self._expectations)

    def _expectation_density_gen(self) -> Iterable[float]:
        for exdensity in zip(*[ex.expectation_density() for ex in self._expectations]):
            yield sum(exdensity)
    
    def expectation_density(self) -> Iterable[float]:
        if self._vectorize:
            if not self._expectations:
                return np.array([], dtype=np.float64)
            else:
                densities = [asarray(ex.expectation_density(), np.float64) for ex in self._expectations]
                return np.add.reduce(densities)
        else:
            return self._expectation_density_gen()



