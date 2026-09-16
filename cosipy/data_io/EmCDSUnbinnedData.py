from pathlib import Path
from typing import Iterable, Iterator, Optional, Union, List

import numpy as np
from astropy.coordinates import Angle, SkyCoord, UnitSphericalRepresentation
from astropy.time import Time
from astropy.units import Quantity
from scoords import SpacecraftFrame

from cosipy.data_io.UnBinnedData import UnBinnedData
from cosipy.interfaces import (
    EventWithEnergyInterface, 
    EventDataInterface, 
    EventDataWithEnergyInterface
)
from cosipy.interfaces.data_interface import (
    TimeTagEmCDSEventDataInSCFrameInterface, 
    EmCDSEventDataInSCFrameInterface, 
    TimeTagEmCDSEventDataInSCAndGalFrameInterface, 
    EmCDSEventDataInSCAndGalFrameInterface
)
from cosipy.interfaces.event import  (
    TimeTagEmCDSEventInSCFrameInterface, 
    EmCDSEventInSCFrameInterface , 
    TimeTagEmCDSEventInSCAndGalFrameInterface, 
    EmCDSEventInSCAndGalFrameInterface
)
import astropy.units as u

from cosipy.interfaces.event_selection import EventSelectorInterface
from cosipy.util.iterables import asarray


class EmCDSEventInSCFrame(EmCDSEventInSCFrameInterface):

    _frame = SpacecraftFrame()

    def __init__(self, energy, scatt_angle, scatt_lon, scatt_lat, event_id = None):
        """
        Parameters
        ----------
        jd1: julian days
        jd2: julian days
        energy: keV
        scatt_angle: scattering angle radians
        scatt_lon: scattering longitude radians
        scatt_lat: scattering latitude radians
        """
        self._id = event_id
        self._energy = energy
        self._scatt_angle = scatt_angle
        self._scatt_lat = scatt_lat
        self._scatt_lon = scatt_lon

    @property
    def id(self) -> int:
        return self._id

    @property
    def frame(self):
        return self._frame

    @property
    def energy_keV(self) -> float:
        return self._energy

    @property
    def scattering_angle_rad(self) -> float:
        return self._scatt_angle

    @property
    def scattered_lon_rad_sc(self) -> float:
        return self._scatt_lon

    @property
    def scattered_lat_rad_sc(self) -> float:
        return self._scatt_lat


class EmCDSEventInSCAndGalFrame(EmCDSEventInSCAndGalFrameInterface):


    def __init__(self, energy, scatt_angle, scatt_lon_sc, scatt_lat_sc, 
                 scatt_lon_gal, scatt_lat_gal, event_id = None):
        """
        Parameters
        ----------
        jd1: julian days
        jd2: julian days
        energy: keV
        scatt_angle: scattering angle radians
        scatt_lon_sc: scattering longitude radians (SC frame)
        scatt_lat_sc: scattering latitude radians (SC frame)
        scatt_lon_gal: scattering longitude deg (Gal frame)
        scatt_lat_gal: scattering latitude deg (Gal frame)
        """
        self._id = event_id
        self._energy = energy
        self._scatt_angle = scatt_angle
        self._scatt_lat_sc = scatt_lat_sc
        self._scatt_lon_sc = scatt_lon_sc
        self._scatt_lat_gal = scatt_lat_gal
        self._scatt_lon_gal = scatt_lon_gal

    @property
    def id(self) -> int:
        return self._id

    @property
    def frame(self):
        return self._frame

    @property
    def energy_keV(self) -> float:
        return self._energy

    @property
    def scattering_angle_rad(self) -> float:
        return self._scatt_angle

    @property
    def scattered_lon_rad_sc(self) -> float:
        return self._scatt_lon_sc

    @property
    def scattered_lat_rad_sc(self) -> float:
        return self._scatt_lat_sc

    @property
    def scattered_lon_deg_gal(self) -> float:
        return self._scatt_lon_gal

    @property
    def scattered_lat_deg_gal(self) -> float:
        return self._scatt_lat_gal
        
class TimeTagEmCDSEventInSCFrame(EmCDSEventInSCFrame, TimeTagEmCDSEventInSCFrameInterface):

    def __init__(self, jd1, jd2, energy, scatt_angle, scatt_lon, scatt_lat, event_id=None):
        """
        Parameters
        ----------
        jd1: julian days
        jd2: julian days
        energy: keV
        scatt_angle: scattering angle radians
        scatt_lon: scattering longitude radians
        scatt_lat: scattering latitude radians
        """
        super().__init__(energy, scatt_angle, scatt_lon, scatt_lat, event_id)

        self._jd1 = jd1
        self._jd2 = jd2

    @property
    def jd1(self):
        return self._jd1

    @property
    def jd2(self):
        return self._jd2

class TimeTagEmCDSEventInSCAndGalFrame(EmCDSEventInSCAndGalFrame, TimeTagEmCDSEventInSCAndGalFrameInterface):

    def __init__(self, jd1, jd2, energy, scatt_angle, scatt_lon_sc, scatt_lat_sc, scatt_lon_gal, scatt_lat_gal, event_id=None):
        """
        Parameters
        ----------
        jd1: julian days
        jd2: julian days
        energy: keV
        scatt_angle: scattering angle radians
        scatt_lon_sc: scattering longitude radians (SC frame)
        scatt_lat_sc: scattering latitude radians (SC frame)
        scatt_lon_gal: scattering longitude deg (Gal frame)
        scatt_lat_gal: scattering latitude deg (Gal frame)
        """
        super().__init__(energy, scatt_angle, scatt_lon_sc, scatt_lat_sc, scatt_lon_gal, scatt_lat_gal, event_id)

        self._jd1 = jd1
        self._jd2 = jd2

    @property
    def jd1(self):
        return self._jd1

    @property
    def jd2(self):
        return self._jd2



        
class EmCDSEventDataInSCFrameFromArrays(EmCDSEventDataInSCFrameInterface):

    _frame = SpacecraftFrame()
    event_type = EmCDSEventInSCFrameInterface

    def __init__(self,
                   energy_keV: np.ndarray[float],
                   scattered_lon_rad_sc:  np.ndarray[float],
                   scattered_lat_rad_sc: np.ndarray[float],
                   scatt_angle_rad: np.ndarray[float],
                   event_id: Optional[np.ndarray[int]] = None,
                   selection: Optional[EventSelectorInterface] = None):
        """
        Initialize from bare numpy arrays. The user is responsible from getting the right units, coordinates and formats

        Parameters
        ----------
        energy_keV: energy [keV]
        scattered_lon_rad_sc: Longitude of the direction of the scattered photon in spacecraft coordinates [radian]
        scattered_lat_rad_sc:  Latitude of the direction of the scattered photon in spacecraft coordinates [radian]
        scatt_angle_rad: Compton scattering angle [radians]
        event_id: Event ID. Optional. Sequential is not provided
        selection: Optional. Apply an event selection.
        """

        # Check size
        self._energy, self._scatt_angle, self._scatt_lon, self._scatt_lat = np.broadcast_arrays(energy_keV, scatt_angle_rad, scattered_lon_rad_sc, scattered_lat_rad_sc)

        if event_id is None:
            self._id = np.arange(self._energy.size)
        else:
            self._id = np.asarray(event_id)

        self._nevents = self._id.size

        if selection is not None:
            mask = asarray(selection.select(self), dtype=bool)

            if mask.size < self._nevents:
                # The rest of the events are False implicitly
                mask = np.append(mask, np.full(self._nevents - mask.size, False))

            self._id = self._id[mask]
            self._energy = self._energy[mask]
            self._scatt_angle = self._scatt_angle[mask]
            self._scatt_lat = self._scatt_lat[mask]
            self._scatt_lon = self._scatt_lon[mask]

            self._nevents = self._id.size

    @classmethod
    def from_astropy(cls,
                 energy:Quantity,
                 scattering_angle:Angle,
                 scattered_direction:SkyCoord,
                 event_id:Optional[Iterable[int]] = None,
                 selection:Optional[EventSelectorInterface] = None):
        """
        Initialize from astropy objects, taking into account the units and formats

        Parameters
        ----------
        energy
        scattering_angle
        scattered_direction
        event_id
        selection
        """

        energy = energy.to_value(u.keV)
        scatt_angle = scattering_angle.to_value(u.rad)

        if not isinstance(scattered_direction.frame, SpacecraftFrame):
            raise ValueError("Coordinates need to be in SC frame")

        scattered_direction = scattered_direction.represent_as(UnitSphericalRepresentation)

        scatt_lat = scattered_direction.lat.rad
        scatt_lon = scattered_direction.lon.rad

        if event_id is not None:
            event_id = np.asarray(event_id)

        return cls(energy, scatt_lon, scatt_lat, scatt_angle, event_id, selection)


    def __getitem__(self, i: int) -> EmCDSEventInSCFrameInterface:
        return EmCDSEventInSCFrame(self._energy[i], self._scatt_angle[i], self._scatt_lon[i], self._scatt_lat[i],
                                          self._id[i])

    @property
    def nevents(self) -> int:
        return self._nevents

    def __iter__(self) -> Iterator[EmCDSEventInSCFrameInterface]:
        for id, energy, scatt_angle, scatt_lat, scatt_lon in zip(self._id, self._energy, self._scatt_angle, self._scatt_lat, self._scatt_lon):
            yield EmCDSEventInSCFrame(energy, scatt_angle, scatt_lon, scatt_lat, id)

    @property
    def frame(self) -> SpacecraftFrame:
        return self._frame

    @property
    def ids(self) -> Iterable[int]:
        return self._id

    @property
    def energy_keV(self) -> Iterable[float]:
        return self._energy

    @property
    def scattering_angle_rad(self) -> Iterable[float]:
        return self._scatt_angle

    @property
    def scattered_lon_rad_sc(self) -> Iterable[float]:
        return self._scatt_lon

    @property
    def scattered_lat_rad_sc(self) -> Iterable[float]:
        return self._scatt_lat


class EmCDSEventDataInSCAndGalFrameFromArrays(EmCDSEventDataInSCAndGalFrameInterface):

    _frame = "SpacecraftAndGalacticFrame"
    event_type = EmCDSEventInSCAndGalFrameInterface

    def __init__(self,
                   energy_keV: np.ndarray[float],
                   scattered_lon_rad_sc:  np.ndarray[float],
                   scattered_lat_rad_sc: np.ndarray[float],
                   scattered_lon_deg_gal:  np.ndarray[float],
                   scattered_lat_deg_gal: np.ndarray[float],
                   scatt_angle_rad: np.ndarray[float],
                   event_id: Optional[np.ndarray[int]] = None,
                   selection: Optional[EventSelectorInterface] = None):
        """
        Initialize from bare numpy arrays. The user is responsible from getting the right units, coordinates and formats

        Parameters
        ----------
        energy_keV: energy [keV]
        scattered_lon_rad_sc: Longitude of the direction of the scattered photon in spacecraft coordinates [radian]
        scattered_lat_rad_sc:  Latitude of the direction of the scattered photon in spacecraft coordinates [radian]
        scattered_lon_deg_gal: Longitude of the direction of the scattered photon in galactic coordinates [deg]
        scattered_lat_deg_gal:  Latitude of the direction of the scattered photon in galactic coordinates [deg]
        scatt_angle_rad: Compton scattering angle [radians]
        event_id: Event ID. Optional. Sequential is not provided
        selection: Optional. Apply an event selection.
        """

        # Check size
        self._energy, self._scatt_angle, self._scatt_lon_sc, self._scatt_lat_sc, self._scatt_lon_gal, self._scatt_lat_gal = np.broadcast_arrays(energy_keV, scatt_angle_rad, scattered_lon_rad_sc, scattered_lat_rad_sc, scattered_lon_deg_gal, scattered_lat_deg_gal)

        if event_id is None:
            self._id = np.arange(self._energy.size)
        else:
            self._id = np.asarray(event_id)

        self._nevents = self._id.size

        if selection is not None:
            mask = asarray(selection.select(self), dtype=bool)

            if mask.size < self._nevents:
                # The rest of the events are False implicitly
                mask = np.append(mask, np.full(self._nevents - mask.size, False))

            self._id = self._id[mask]
            self._energy = self._energy[mask]
            self._scatt_angle = self._scatt_angle[mask]
            self._scatt_lat_sc = self._scatt_lat_sc[mask]
            self._scatt_lon_sc = self._scatt_lon_sc[mask]
            self._scatt_lat_gal = self._scatt_lat_gal[mask]
            self._scatt_lon_gal = self._scatt_lon_gal[mask]
            
            self._nevents = self._id.size

    @classmethod
    def from_astropy(cls,
                 energy:Quantity,
                 scattering_angle:Angle,
                 scattered_direction_sc:SkyCoord,
                 scattered_direction_gal:SkyCoord,    
                 event_id:Optional[Iterable[int]] = None,
                 selection:Optional[EventSelectorInterface] = None):
        """
        Initialize from astropy objects, taking into account the units and formats

        Parameters
        ----------
        energy
        scattering_angle
        scattered_direction_sc
        scattered_direction_gal
        event_id
        selection
        """

        energy = energy.to_value(u.keV)
        scatt_angle = scattering_angle.to_value(u.rad)

        if not isinstance(scattered_direction_sc.frame, SpacecraftFrame):
            raise ValueError("Local PsiChi Coordinates need to be in SC frame")
        
        if not isinstance(scattered_direction_gal.frame, "Galactic"):
            raise ValueError("Galactic PsiChi Coordinates need to be in Galactic frame")

        
        scattered_direction_sc = scattered_direction_sc.represent_as(UnitSphericalRepresentation)
        scattered_direction_gal = scattered_direction_gal.represent_as(UnitSphericalRepresentation)

        scatt_lat_sc = scattered_direction_sc.lat.rad
        scatt_lon_sc = scattered_direction_sc.lon.rad
        scatt_lat_gal = scattered_direction_gal.lat.deg
        scatt_lon_gal = scattered_direction_gal.lon.deg

        if event_id is not None:
            event_id = np.asarray(event_id)

        return cls(energy, scatt_lon_rad, scatt_lat_rad, scatt_lon_gal, scatt_lat_gal, scatt_angle, event_id, selection)


    def __getitem__(self, i: int) -> EmCDSEventInSCAndGalFrameInterface:
        return EmCDSEventInSCAndGalFrame(self._energy[i], self._scatt_angle[i], self._scatt_lon_sc[i], self._scatt_lat_sc[i],
                                          self._scatt_lon_gal[i], self._scatt_lat_gal[i], self._id[i])

    @property
    def nevents(self) -> int:
        return self._nevents

    def __iter__(self) -> Iterator[EmCDSEventInSCAndGalFrameInterface]:
        for id, energy, scatt_angle, scatt_lat_sc, scatt_lon_sc, scatt_lat_gal, scatt_lon_gal in zip(self._id, self._energy, self._scatt_angle, self._scatt_lat_sc, self._scatt_lon_sc, self._scatt_lat_gal, self._scatt_lon_gal):
            yield EmCDSEventInSCAndGalFrame(energy, scatt_angle, scatt_lon_sc, scatt_lat_sc, scatt_lon_gal, scatt_lat_gal, id)

    @property
    def frame(self) -> str:
        return self._frame

    @property
    def ids(self) -> Iterable[int]:
        return self._id

    @property
    def energy_keV(self) -> Iterable[float]:
        return self._energy

    @property
    def scattering_angle_rad(self) -> Iterable[float]:
        return self._scatt_angle

    @property
    def scattered_lon_rad_sc(self) -> Iterable[float]:
        return self._scatt_lon_sc

    @property
    def scattered_lat_rad_sc(self) -> Iterable[float]:
        return self._scatt_lat_sc

    @property
    def scattered_lon_deg_gal(self) -> Iterable[float]:
        return self._scatt_lon_gal

    @property
    def scattered_lat_deg_gal(self) -> Iterable[float]:
        return self._scatt_lat_gal
        
class TimeTagEmCDSEventDataInSCFrameFromArrays(EmCDSEventDataInSCFrameFromArrays, TimeTagEmCDSEventDataInSCFrameInterface):

    event_type = TimeTagEmCDSEventInSCFrameInterface

    def __init__(self,
                   jd1: np.ndarray[float],
                   jd2: np.ndarray[float],
                   energy_keV: np.ndarray[float],
                   scattered_lon_rad_sc:  np.ndarray[float],
                   scattered_lat_rad_sc: np.ndarray[float],
                   scatt_angle_rad: np.ndarray[float],
                   event_id: Optional[np.ndarray[int]] = None,
                   selection: Optional[EventSelectorInterface] = None):
        """
        Initialize from bare numpy arrays. The user is responsible from getting the right units, coordinates and formats

        Parameters
        ----------
        jd1: Julian days. Internal astropy Time representation using two values for full precision.
        jd2: Julian days. Internal astropy Time representation using two values for full precision.
        energy_keV: energy [keV]
        scattered_lon_rad_sc: Longitude of the direction of the scattered photon in spacecraft coordinates [radian]
        scattered_lat_rad_sc:  Latitude of the direction of the scattered photon in spacecraft coordinates [radian]
        scatt_angle_rad: Compton scattering angle [radians]
        event_id: Event ID. Optional. Sequential is not provided
        selection: Optional. Apply an event selection.
        """

        # Check size
        self._jd1, self._jd2, energy, scatt_angle, scatt_lon, scatt_lat = np.broadcast_arrays(
            jd1, jd2, energy_keV, scatt_angle_rad, scattered_lon_rad_sc, scattered_lat_rad_sc)

        super().__init__(energy, scatt_lon, scatt_lat, scatt_angle, event_id)

        if selection is not None:
            mask = asarray(selection.select(self), dtype=bool)

            if mask.size < self._nevents:
                # The rest of the events are False implicitly
                mask = np.append(mask, np.full(self._nevents - mask.size, False))

            self._id = self._id[mask]
            self._jd1 = self._jd1[mask]
            self._jd2 = self._jd2[mask]
            self._energy = self._energy[mask]
            self._scatt_angle = self._scatt_angle[mask]
            self._scatt_lat = self._scatt_lat[mask]
            self._scatt_lon = self._scatt_lon[mask]

            self._nevents = self._id.size

    @classmethod
    def from_astropy(cls,
                 time:Time,
                 energy:Quantity,
                 scattering_angle:Angle,
                 scattered_direction:SkyCoord,
                 event_id:Optional[Iterable[int]] = None,
                 selection:Optional[EventSelectorInterface] = None):
        """
        Initialize from astropy objects, taking into account the units and formats

        Parameters
        ----------
        time
        energy
        scattering_angle
        scattered_direction
        event_id
        selection
        """

        jd1 = time.jd1
        jd2 = time.jd2
        energy = energy.to_value(u.keV)
        scatt_angle = scattering_angle.to_value(u.rad)

        if not isinstance(scattered_direction.frame, SpacecraftFrame):
            raise ValueError("Coordinates need to be in SC frame")

        scattered_direction = scattered_direction.represent_as(UnitSphericalRepresentation)

        scatt_lat = scattered_direction.lat.rad
        scatt_lon = scattered_direction.lon.rad

        if event_id is not None:
            event_id = np.asarray(event_id)

        return cls(jd1, jd2, energy, scatt_lon, scatt_lat, scatt_angle, event_id, selection)


    def __getitem__(self, i: int) -> TimeTagEmCDSEventInSCFrameInterface:
        return TimeTagEmCDSEventInSCFrame(self._jd1[i], self._jd2[i], self._energy[i], self._scatt_angle[i], self._scatt_lon[i], self._scatt_lat[i],
                                          self._id[i])

    def __iter__(self) -> Iterator[TimeTagEmCDSEventInSCFrameInterface]:
        for id, jd1, jd2, energy, scatt_angle, scatt_lat, scatt_lon in zip(self._id, self._jd1, self._jd2, self._energy, self._scatt_angle, self._scatt_lat, self._scatt_lon):
            yield TimeTagEmCDSEventInSCFrame(jd1, jd2, energy, scatt_angle, scatt_lon, scatt_lat, id)

    @property
    def jd1(self) -> Iterable[float]:
        return self._jd1

    @property
    def jd2(self) -> Iterable[float]:
        return self._jd2

class TimeTagEmCDSEventDataInSCAndGalFrameFromArrays(EmCDSEventDataInSCAndGalFrameFromArrays, TimeTagEmCDSEventDataInSCAndGalFrameInterface):

    event_type = TimeTagEmCDSEventInSCAndGalFrameInterface

    def __init__(self,
                   jd1: np.ndarray[float],
                   jd2: np.ndarray[float],
                   energy_keV: np.ndarray[float],
                   scattered_lon_rad_sc:  np.ndarray[float],
                   scattered_lat_rad_sc: np.ndarray[float],
                   scattered_lon_deg_gal:  np.ndarray[float],
                   scattered_lat_deg_gal: np.ndarray[float],
                   scatt_angle_rad: np.ndarray[float],
                   event_id: Optional[np.ndarray[int]] = None,
                   selection: Optional[EventSelectorInterface] = None):
        """
        Initialize from bare numpy arrays. The user is responsible from getting the right units, coordinates and formats

        Parameters
        ----------
        jd1: Julian days. Internal astropy Time representation using two values for full precision.
        jd2: Julian days. Internal astropy Time representation using two values for full precision.
        energy_keV: energy [keV]
        scattered_lon_rad_sc: Longitude of the direction of the scattered photon in spacecraft coordinates [radian]
        scattered_lat_rad_sc:  Latitude of the direction of the scattered photon in spacecraft coordinates [radian]
        scattered_lon_deg_gal: Longitude of the direction of the scattered photon in galactic coordinates [deg]
        scattered_lat_deg_gal:  Latitude of the direction of the scattered photon in galactic coordinates [deg]
        scatt_angle_rad: Compton scattering angle [radians]
        event_id: Event ID. Optional. Sequential is not provided
        selection: Optional. Apply an event selection.
        """

        # Check size
        self._jd1, self._jd2, energy, scatt_angle, scatt_lon_sc, scatt_lat_sc, scatt_lon_gal, scatt_lat_gal = np.broadcast_arrays(
            jd1, jd2, energy_keV, scatt_angle_rad, scattered_lon_rad_sc, scattered_lat_rad_sc, scattered_lon_deg_gal, scattered_lat_deg_gal)

        super().__init__(energy, scatt_lon_sc, scatt_lat_sc, scatt_lon_gal, scatt_lat_gal, scatt_angle, event_id)

        if selection is not None:
            mask = asarray(selection.select(self), dtype=bool)

            if mask.size < self._nevents:
                # The rest of the events are False implicitly
                mask = np.append(mask, np.full(self._nevents - mask.size, False))

            self._id = self._id[mask]
            self._jd1 = self._jd1[mask]
            self._jd2 = self._jd2[mask]
            self._energy = self._energy[mask]
            self._scatt_angle = self._scatt_angle[mask]
            self._scatt_lat_sc = self._scatt_lat_sc[mask]
            self._scatt_lon_sc = self._scatt_lon_sc[mask]
            self._scatt_lat_gal = self._scatt_lat_gal[mask]
            self._scatt_lon_gal = self._scatt_lon_gal[mask]
            
            self._nevents = self._id.size

    @classmethod
    def from_astropy(cls,
                 time:Time,
                 energy:Quantity,
                 scattering_angle:Angle,
                 scattered_direction_sc:SkyCoord,
                 scattered_direction_gal:SkyCoord,
                 event_id:Optional[Iterable[int]] = None,
                 selection:Optional[EventSelectorInterface] = None):
        """
        Initialize from astropy objects, taking into account the units and formats

        Parameters
        ----------
        time
        energy
        scattering_angle
        scattered_direction_sc
        scattered_direction_gal
        event_id
        selection
        """

        jd1 = time.jd1
        jd2 = time.jd2
        energy = energy.to_value(u.keV)
        scatt_angle = scattering_angle.to_value(u.rad)

        if not isinstance(scattered_direction_sc.frame, SpacecraftFrame):
            raise ValueError("Local PsiChi Coordinates need to be in SC frame")
        
        if not isinstance(scattered_direction_gal.frame, "Galactic"):
            raise ValueError("Galactic PsiChi Coordinates need to be in Galactic frame")

        scattered_direction_sc = scattered_direction_sc.represent_as(UnitSphericalRepresentation)
        scattered_direction_gal = scattered_direction_gal.represent_as(UnitSphericalRepresentation)

        scatt_lat_sc = scattered_direction_sc.lat.rad
        scatt_lon_sc = scattered_direction_sc.lon.rad
        scatt_lat_gal = scattered_direction_gal.lat.deg
        scatt_lon_gal = scattered_direction_gal.lon.deg

        if event_id is not None:
            event_id = np.asarray(event_id)

        return cls(jd1, jd2, energy, scatt_lon_sc, scatt_lat_sc, scatt_lon_gal, scatt_lat_gal, scatt_angle, event_id, selection)


    def __getitem__(self, i: int) -> TimeTagEmCDSEventInSCAndGalFrameInterface:
        return TimeTagEmCDSEventInSCAndGalFrame(self._jd1[i], self._jd2[i], self._energy[i], self._scatt_angle[i], self._scatt_lon_sc[i], self._scatt_lat_sc[i], self._scatt_lon_gal[i], self._scatt_lat_gal[i],
                                          self._id[i])

    def __iter__(self) -> Iterator[TimeTagEmCDSEventInSCAndGalFrameInterface]:
        for id, jd1, jd2, energy, scatt_angle, scatt_lat_sc, scatt_lon_sc, scatt_lat_gal, scatt_lon_gal in zip(self._id, self._jd1, self._jd2, self._energy, self._scatt_angle, self._scatt_lat_sc, self._scatt_lon_sc, self._scatt_lat_gal, self._scatt_lon_gal):
            yield TimeTagEmCDSEventInSCAndGalFrame(jd1, jd2, energy, scatt_angle, scatt_lon_sc, scatt_lat_sc, scatt_lon_gal, scatt_lat_gal, id)

    @property
    def jd1(self) -> Iterable[float]:
        return self._jd1

    @property
    def jd2(self) -> Iterable[float]:
        return self._jd2


        
class TimeTagEmCDSEventDataInSCFrameFromDC3Fits(TimeTagEmCDSEventDataInSCFrameFromArrays):

    def __init__(self, data_path: Union[Path, List[Path]],
                 selection:EventSelectorInterface = None):

        time = np.empty(0)
        energy = np.empty(0)
        phi = np.empty(0)
        psi = np.empty(0)
        chi = np.empty(0)

        if isinstance(data_path, (str, Path)):
            data_path = [Path(data_path)]

        for file in data_path:
            # get_dict_fits or hdf5 is really a static method, no config file needed
            # You can pass either hdf5 or fits file
            if "hdf5" in str(file).split("."):
                data_dict = UnBinnedData.get_dict_from_hdf5(None, str(file))
            elif "fits" in str(file).split("."):
                data_dict = UnBinnedData.get_dict_from_fits(None, str(file))
            else:
                raise ValueError("You should pass a fits or hdf5 file.")

            time = np.append(time, data_dict['TimeTags'])
            energy = np.append(energy, data_dict['Energies'])
            phi = np.append(phi, data_dict['Phi'])
            psi = np.append(psi, data_dict['Psi local'])
            chi = np.append(chi, data_dict['Chi local'])

        # Time sort
        tsort = np.argsort(time)

        time = time[tsort]
        energy = energy[tsort]
        phi = phi[tsort]
        psi = psi[tsort]
        chi = chi[tsort]

        time = Time(time, format='unix')

        # Psi is colatitude (latitude complementary angle)
        super().__init__(time.jd1, time.jd2, energy, chi, np.pi / 2 - psi, phi, selection = selection)



class TimeTagEmCDSEventDataInSCAndGalFrameFromDC3Fits(TimeTagEmCDSEventDataInSCAndGalFrameFromArrays):

    def __init__(self, data_path: Union[Path, List[Path]],
                 selection:EventSelectorInterface = None):

        time = np.empty(0)
        energy = np.empty(0)
        phi = np.empty(0)
        psi_local = np.empty(0)
        chi_local = np.empty(0)
        psi_gal = np.empty(0)
        chi_gal = np.empty(0)

        if isinstance(data_path, (str, Path)):
            data_path = [Path(data_path)]

        for file in data_path:
            # get_dict_fits or hdf5 is really a static method, no config file needed
            # You can pass either hdf5 or fits file
            if "hdf5" in str(file).split("."):
                data_dict = UnBinnedData.get_dict_from_hdf5(None, str(file))
            elif "fits" in str(file).split("."):
                data_dict = UnBinnedData.get_dict_from_fits(None, str(file))
            else:
                raise ValueError("You should pass a fits or hdf5 file.")
            
            time = np.append(time, data_dict['TimeTags'])
            energy = np.append(energy, data_dict['Energies'])
            phi = np.append(phi, data_dict['Phi'])
            psi_local = np.append(psi_local, data_dict['Psi local'])
            chi_local = np.append(chi_local, data_dict['Chi local'])
            psi_gal = np.append(psi_gal, data_dict['Psi galactic'])
            chi_gal = np.append(chi_gal, data_dict['Chi galactic'])
            
        # Time sort
        tsort = np.argsort(time)

        time = time[tsort]
        energy = energy[tsort]
        phi = phi[tsort]
        psi_local = psi_local[tsort]
        chi_local = chi_local[tsort]
        psi_gal = psi_gal[tsort]
        chi_gal = chi_gal[tsort]
        
        time = Time(time, format='unix')

        # Psi is colatitude (latitude complementary angle)
        super().__init__(time.jd1, time.jd2, energy, chi_local, np.pi / 2 - psi_local, psi_gal, chi_gal, phi, selection = selection)


