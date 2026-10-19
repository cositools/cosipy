from abc import ABC, abstractmethod
from symtable import Class
from typing import Sequence, Union, Protocol, ClassVar

import numpy as np
from astropy.coordinates import Angle, SkyCoord, BaseCoordinateFrame
from scoords import SpacecraftFrame
from typing_extensions import runtime_checkable

from astropy.time import Time
from astropy.units import Quantity, Unit
import astropy.units as u

__all__ = [
    "EventInterface",
    "TimeTagEventInterface",
    "EventWithEnergyInterface",
    "EventWithScatteringAngleInterface",
    "ComptonDataSpaceInSCFrameEventInterface",
    "ComptonDataSpaceInGalFrameEventInterface",
    "EmCDSEventInSCFrameInterface",
    "TimeTagEmCDSEventInSCFrameInterface",
    "EmCDSEventInSCAndGalFrameInterface",
    "TimeTagEmCDSEventInSCAndGalFrameInterface",
    "TimeTagEmCDSDistanceEventInSCFrameInterface",
]

@runtime_checkable
class EventInterface(Protocol):
    """
    Derived classes implement all accessors
    """

    # This makes sure that all PDFs have the same units
    _data_space_units: ClassVar[Union[str, None]] = None

    @property
    def data_space_units(self) -> Union[Unit, None]:
        if self._data_space_units is None:
            return None

        return Unit(self._data_space_units)

    @property
    def id(self) -> int:
        """
        Typically set by the main data loader or source.

        No necessarily in sequential order
        """

@runtime_checkable
class TimeTagEventInterface(EventInterface, Protocol):

    _data_space_units = "s"

    @property
    def jd1(self) -> float:...

    @property
    def jd2(self) -> float:...

    @property
    def time(self) -> Time:
        """
        Add fancy time
        """
        return Time(self.jd1, self.jd2, format = 'jd')

@runtime_checkable
class EventWithEnergyInterface(EventInterface, Protocol):

    _data_space_units = "keV"

    @property
    def energy_keV(self) -> float:...

    @property
    def energy(self) -> Quantity:
        """
        Add fancy energy quantity
        """
        return Quantity(self.energy_keV, u.keV)

@runtime_checkable
class EventWithScatteringAngleInterface(EventInterface, Protocol):

    _data_space_units = "rad"

    @property
    def scattering_angle_rad(self) -> float: ...


    @property
    def scattering_angle(self) -> Angle:
        """
        Add fancy energy quantity
        """
        return Angle(self.scattering_angle_rad, u.rad)


@runtime_checkable
class ComptonDataSpaceInSCFrameEventInterface(EventWithScatteringAngleInterface, Protocol):

    _data_space_units = "rad sr"

    @property
    def scattered_lon_rad_sc(self) -> float: ...

    @property
    def scattered_lat_rad_sc(self) -> float: ...

    @property
    def scattered_direction_sc(self) -> SkyCoord:
        """
        Add fancy energy quantity
        """
        return SkyCoord(self.scattered_lon_rad_sc,
                        self.scattered_lat_rad_sc,
                        unit=u.rad,
                        frame=SpacecraftFrame())


@runtime_checkable
class ComptonDataSpaceInGalFrameEventInterface(EventWithScatteringAngleInterface, Protocol):

    _data_space_units = "rad sr"

    @property
    def scattered_lon_rad_gal(self) -> float: ...

    @property
    def scattered_lat_rad_gal(self) -> float: ...

    @property
    def scattered_direction_gal(self) -> SkyCoord:
        """
        Add fancy energy quantity
        """
        return SkyCoord(self.scattered_lon_rad_gal,
                        self.scattered_lat_rad_gal,
                        unit=u.rad,
                        frame="Galactic")

@runtime_checkable
class EmCDSEventInSCFrameInterface(EventWithEnergyInterface,
                                   ComptonDataSpaceInSCFrameEventInterface,
                                   Protocol):
    _data_space_units = "rad sr keV"

@runtime_checkable
class TimeTagEmCDSEventInSCFrameInterface(TimeTagEventInterface,
                                          EmCDSEventInSCFrameInterface,
                                          Protocol):
    _data_space_units = "rad sr keV s"

@runtime_checkable
class EmCDSEventInSCAndGalFrameInterface(EventWithEnergyInterface,
                                   ComptonDataSpaceInSCFrameEventInterface,
                                   ComptonDataSpaceInGalFrameEventInterface,      
                                   Protocol):
    _data_space_units = "rad2 sr2 keV"

@runtime_checkable
class TimeTagEmCDSEventInSCAndGalFrameInterface(TimeTagEventInterface,
                                          EmCDSEventInSCAndGalFrameInterface,
                                          Protocol):
    _data_space_units = "rad2 sr2 keV s"

@runtime_checkable
class TimeTagEmCDSDistanceEventInSCFrameInterface(TimeTagEmCDSEventInSCFrameInterface, Protocol):
    """
    A TimeTagEmCDSEventInSCFrameInterface event that also tracks the
    distance between the first two hits of the Compton event.
    """

    @property
    def distance_cm(self) -> float: ...

    @property
    def distance(self) -> Quantity:
        """
        Add fancy distance quantity
        """
        return Quantity(self.distance_cm, u.cm)

