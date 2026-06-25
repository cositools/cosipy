import itertools
from typing import Protocol, runtime_checkable, Iterator, Union, Iterable

from astropy.coordinates import Angle, SkyCoord
from astropy.units import  Quantity
import astropy.units as u

from scoords import SpacecraftFrame

from . import EventWithEnergyInterface

from .event import (
    EventInterface,
    TimeTagEventInterface,
    ComptonDataSpaceInSCFrameEventInterface,
    EmCDSEventInSCFrameInterface,
    TimeTagEmCDSEventInSCFrameInterface,
    EventWithScatteringAngleInterface,
)

from histpy import Histogram, Axes

from astropy.time import Time

__all__ = ["DataInterface",
           "EventDataInterface",
           "BinnedDataInterface",
           "TimeTagEventDataInterface",
           "EventDataWithEnergyInterface"
          ]

@runtime_checkable
class DataInterface(Protocol):
    pass

@runtime_checkable
class BinnedDataInterface(DataInterface, Protocol):
    @property
    def data(self) -> Histogram:...
    @property
    def axes(self) -> Axes:...
    def fill(self, event_data:'EventDataInterface'):
        """
        Bin the data.

        Parameters
        ----------
        event_data

        Returns
        -------

        """

@runtime_checkable
class EventDataInterface(DataInterface, Protocol):

    # Type returned by __iter__
    event_type = EventInterface

    def __iter__(self) -> Iterator[EventInterface]:
        """
        Return one Event at a time
        """

    def __getitem__(self, item: int) -> EventInterface:
        """
        Convenience method. Pretty slow in general. It's suggested that
        the implementations override it
        """
        return next(itertools.islice(self, item, None))

    @classmethod
    def fromiter(cls, events:Iterable[EventInterface], nevents = None):
        """
        Turns an iterable of events into a proper EventData

        For convenience. Specific implementation can do better job.

        It is not safe to call this method from an interface
        implementation that does not explicitly overload
        fromiter(). Call it from an interface instead.

        If you know it, specifying the length of the iterable can help
        optimize the code.

        """

        if not getattr(cls, "_is_protocol", False):
            raise RuntimeError("It is not safe to call fromiter() from an interface implementation that"
                               "does not explicitly overload fromiter(). Call it from an interface instead.")

        class EventDataIterableWrapper(cls):
            def __init__(self):
                self._nevents = nevents

            def __iter__(self) -> Iterator[cls.event_type]:
                return iter(events)

            @property
            def nevents(self) -> int:
                if nevents is None:
                    self._nevents = sum(1 for _ in iter(self))

                return self._nevents

        event_list = EventDataIterableWrapper()

        return event_list

    @classmethod
    def from_event(cls, event: EventInterface, repeat = 1):
        """
        Convert a single event to a proper EventData

        Parameters
        ----------
        event:
        repeat: Number of time to repeat photon. If None, it will
        repeat indefinitely (use with care)

        Returns
        -------
        tuple:  is_single? (bool), event_data

        """

        if not getattr(cls, "_is_protocol", False):
            raise RuntimeError("It is not safe to call from_event() from an interface implementation that"
                               "does not explicitly overload from_event(). Call it from an interface instead.")

        if not isinstance(event, cls.event_type):
            raise RuntimeError(
                f"Input event (type {type(event)}) is not an instance of {cls.event_type}")

        class SingleEventDataWrapper(cls):
            def __iter__(self) -> Iterator[cls.event_type]:
                for _ in range(repeat):
                    yield event

            def __getitem__(self, item):
                return event

            def nevents(self) -> int:
                return repeat

        event_list = SingleEventDataWrapper.__new__(SingleEventDataWrapper)

        return event_list

    @property
    def nevents(self) -> int:
        """
        Total number of events yielded by __iter__

        Convenience method. Pretty slow in general. It's suggested
        that the implementations override it

        """
        return sum(1 for _ in iter(self))

    @property
    def id(self) -> Iterable[int]:
        return [e.id for e in self]


def is_single_event(photon: Union[EventInterface, 'EventDataInterface']) -> bool:
    # Since these protocols are runtime checkable, it's not enough to
    # check the input is EventInterface, since the EventDataInterface
    # also returns True.
    if isinstance(photon, EventDataInterface):
        return False
    else:
        if not isinstance(photon, EventInterface):
            raise ValueError("Input must be either a EventInterface or a EventDataInterface.")

        return True


@runtime_checkable
class TimeTagEventDataInterface(EventDataInterface, Protocol):

    event_type = TimeTagEventInterface

    def __iter__(self) -> Iterator[TimeTagEventInterface]:...

    @property
    def jd1(self) -> Iterable[float]:
        return [e.jd1 for e in self]

    @property
    def jd2(self) -> Iterable[float]:
        return [e.jd2 for e in self]

    @property
    def time(self) -> Time:
        """
        Add fancy time
        """
        return Time(self.jd1, self.jd2, format = 'jd')

@runtime_checkable
class EventDataWithEnergyInterface(EventDataInterface, Protocol):

    event_type = EventWithEnergyInterface

    def __iter__(self) -> Iterator[EventWithEnergyInterface]:...

    @property
    def energy_keV(self) -> Iterable[float]:
        return [e.energy_keV for e in self]

    @property
    def energy(self) -> Quantity:
        """
        Add fancy energy quantity
        """
        return Quantity(self.energy_keV, u.keV)


@runtime_checkable
class EventDataWithScatteringAngleInterface(EventDataInterface, Protocol):

    event_type = EventWithScatteringAngleInterface

    def __iter__(self) -> Iterator[EventWithScatteringAngleInterface]:...

    @property
    def scattering_angle_rad(self) -> Iterable[float]:
        return [e.scattering_angle_rad for e in self]

    @property
    def scattering_angle(self) -> Angle:
        """
        Add fancy energy quantity
        """
        return Angle(self.scattering_angle_rad, u.rad)

@runtime_checkable
class ComptonDataSpaceInSCFrameEventDataInterface(EventDataWithScatteringAngleInterface, Protocol):

    event_type = ComptonDataSpaceInSCFrameEventInterface

    def __iter__(self) -> Iterator[ComptonDataSpaceInSCFrameEventInterface]:...

    @property
    def scattered_lon_rad_sc(self) -> Iterable[float]:
        return [e.scattered_lon_rad_sc for e in self]

    @property
    def scattered_lat_rad_sc(self) -> Iterable[float]:
        return [e.scattered_lat_rad_sc for e in self]

    @property
    def scattered_direction_sc(self) -> SkyCoord:
        """
        Add fancy energy quantity
        """
        return SkyCoord(self.scattered_lon_rad_sc,
                        self.scattered_lat_rad_sc,
                        unit = u.rad,
                        frame = SpacecraftFrame())

@runtime_checkable
class EmCDSEventDataInSCFrameInterface(EventDataWithEnergyInterface, ComptonDataSpaceInSCFrameEventDataInterface, Protocol):

    event_type = EmCDSEventInSCFrameInterface

    def __iter__(self) -> Iterator[EmCDSEventInSCFrameInterface]: ...

@runtime_checkable
class TimeTagEmCDSEventDataInSCFrameInterface(TimeTagEventDataInterface,
                                              EmCDSEventDataInSCFrameInterface,
                                              Protocol):

    event_type = TimeTagEmCDSEventInSCFrameInterface

    def __iter__(self) -> Iterator[TimeTagEmCDSEventInSCFrameInterface]:...
