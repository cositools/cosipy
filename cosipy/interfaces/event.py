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
    "TimeTagEmCDSEventInSCAndGalFrameInterface"
]

@runtime_checkable
class EventInterface(Protocol):
    """
    Derived classes implement all accessors
    """

    # This makes sure that all PDFs have the same units
    data_space_units = ClassVar[Union[u.Unit, None]]

    @property
    def id(self) -> int:
        """
        Typically set by the main data loader or source.

        No necessarily in sequential order
        """

@runtime_checkable
class TimeTagEventInterface(EventInterface, Protocol):

    data_space_units = u.s

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

    data_space_units = u.keV

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

    data_space_units = u.rad

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

    data_space_units = EventWithScatteringAngleInterface.data_space_units * u.sr

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

    data_space_units = EventWithScatteringAngleInterface.data_space_units * u.deg

    @property
    def scattered_lon_deg_gal(self) -> float: ...

    @property
    def scattered_lat_deg_gal(self) -> float: ...

    @property
    def scattered_direction_gal(self) -> SkyCoord:
        """
        Add fancy energy quantity
        """
        return SkyCoord(self.scattered_lon_deg_gal,
                        self.scattered_lat_deg_gal,
                        unit=u.deg,
                        frame="Galactic")

@runtime_checkable
class EmCDSEventInSCFrameInterface(EventWithEnergyInterface,
                                   ComptonDataSpaceInSCFrameEventInterface,
                                   Protocol):
    data_space_units = ComptonDataSpaceInSCFrameEventInterface.data_space_units * EventWithEnergyInterface.data_space_units

@runtime_checkable
class TimeTagEmCDSEventInSCFrameInterface(TimeTagEventInterface,
                                          EmCDSEventInSCFrameInterface,
                                          Protocol):
    data_space_units = EmCDSEventInSCFrameInterface.data_space_units * TimeTagEventInterface.data_space_units

@runtime_checkable
class EmCDSEventInSCAndGalFrameInterface(EventWithEnergyInterface,
                                   ComptonDataSpaceInSCFrameEventInterface,
                                   ComptonDataSpaceInGalFrameEventInterface,      
                                   Protocol):
    data_space_units = ComptonDataSpaceInSCFrameEventInterface.data_space_units * EventWithEnergyInterface.data_space_units * \
                       ComptonDataSpaceInGalFrameEventInterface.data_space_units

@runtime_checkable
class TimeTagEmCDSEventInSCAndGalFrameInterface(TimeTagEventInterface,
                                          EmCDSEventInSCAndGalFrameInterface,
                                          Protocol):
    data_space_units = EmCDSEventInSCAndGalFrameInterface.data_space_units * TimeTagEventInterface.data_space_units




