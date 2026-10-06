from pathlib import Path
from typing import Iterable, Iterator, Optional, Union, List

import numpy as np

from astropy.coordinates import Angle, SkyCoord, UnitSphericalRepresentation, Galactic
from astropy.time import Time
from astropy.units import Quantity
import astropy.units as u

from scoords import SpacecraftFrame

from cosipy.data_io.UnBinnedData import UnBinnedData
from cosipy.interfaces.data_interface import (
    TimeTagEmCDSEventDataInSCAndGalFrameInterface,
    EmCDSEventDataInSCAndGalFrameInterface,
    TimeTagEmCDSEventDataInSCFrameInterface,
    EmCDSEventDataInSCFrameInterface,
    TimeTagEmCDSDistanceEventDataInSCFrameInterface,
)
from cosipy.interfaces.event import (
    TimeTagEmCDSEventInSCAndGalFrameInterface,
    EmCDSEventInSCAndGalFrameInterface,
    TimeTagEmCDSEventInSCFrameInterface,
    EmCDSEventInSCFrameInterface,
    TimeTagEmCDSDistanceEventInSCFrameInterface,
)

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

class TimeTagEmCDSDistanceEventInSCFrame(TimeTagEmCDSEventInSCFrame, TimeTagEmCDSDistanceEventInSCFrameInterface):

    def __init__(self, jd1, jd2, energy, scatt_angle, scatt_lon, scatt_lat, distance_cm, event_id=None):
        """
        Parameters
        ----------
        jd1: julian days
        jd2: julian days
        energy: keV
        scatt_angle: scattering angle radians
        scatt_lon: scattering longitude radians
        scatt_lat: scattering latitude radians
        distance_cm: distance between the first two hits, cm
        """
        super().__init__(jd1, jd2, energy, scatt_angle, scatt_lon, scatt_lat, event_id)

        self._distance_cm = distance_cm

    @property
    def distance_cm(self) -> float:
        return self._distance_cm

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
        Initialize from bare numpy arrays. The user is responsible from
        getting the right units, coordinates and formats

        Parameters
        ----------
        energy_keV: energy [keV]
        scattered_lon_rad_sc: Longitude of the direction of the
          scattered photon in spacecraft coordinates [radian]
        scattered_lat_rad_sc: Latitude of the direction of the
          scattered photon in spacecraft coordinates [radian]
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

        self._apply_selection(selection, ["_energy", "_scatt_angle", "_scatt_lat", "_scatt_lon"])

    def _apply_selection(self, selection: Optional[EventSelectorInterface], array_attrs: Iterable[str]) -> None:
        """
        Evaluate an (optional) event selection against this object, and
        filter ``self._id`` and every array named in ``array_attrs`` in
        place to keep only the selected events.

        This is meant to be called at the end of ``__init__`` by
        implementations backed by plain numpy arrays, once all the arrays
        that make up an event (including ``self._id``) have been set.

        Parameters
        ----------
        selection: Event selection to apply. If None, nothing is done.
        array_attrs: Names of the (already set) numpy array attributes,
          other than ``self._id``, that should be filtered alongside it.
        """

        if selection is None:
            return

        mask = asarray(selection.select(self), dtype=bool)

        if mask.size < self._nevents:
            # The rest of the events are False implicitly
            mask = np.append(mask, np.full(self._nevents - mask.size, False))

        self._id = self._id[mask]

        for attr in array_attrs:
            setattr(self, attr, getattr(self, attr)[mask])

        self._nevents = self._id.size

    @classmethod
    def from_astropy(cls,
                 energy:Quantity,
                 scattering_angle:Angle,
                 scattered_direction:SkyCoord,
                 event_id:Optional[Iterable[int]] = None,
                 selection:Optional[EventSelectorInterface] = None):
        """
        Initialize from astropy objects, taking into account the units and
        formats

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
        """Initialize from bare numpy arrays. The user is responsible from
        getting the right units, coordinates and formats

        Parameters
        ----------
        jd1: Julian days. Internal astropy Time representation using
          two values for full precision.
        jd2: Julian days. Internal astropy Time representation using
          two values for full precision.
        energy_keV: energy [keV]
        scattered_lon_rad_sc: Longitude of the direction of the
          scattered photon in spacecraft coordinates [radian]
        scattered_lat_rad_sc: Latitude of the direction of the
          scattered photon in spacecraft coordinates [radian]
        scatt_angle_rad: Compton scattering angle [radians]
        event_id: Event ID. Optional. Sequential is not provided
        selection: Optional. Apply an event selection.

        """

        # Check size
        self._jd1, self._jd2, energy, scatt_angle, scatt_lon, scatt_lat = np.broadcast_arrays(
            jd1, jd2, energy_keV, scatt_angle_rad, scattered_lon_rad_sc, scattered_lat_rad_sc)

        super().__init__(energy, scatt_lon, scatt_lat, scatt_angle, event_id)

        self._apply_selection(selection, ["_jd1", "_jd2", "_energy", "_scatt_angle", "_scatt_lat", "_scatt_lon"])

    @classmethod
    def from_astropy(cls,
                     time:Time,
                     energy:Quantity,
                     scattering_angle:Angle,
                     scattered_direction:SkyCoord,
                     event_id:Optional[Iterable[int]] = None,
                    selection:Optional[EventSelectorInterface] = None):
        """
        Initialize from astropy objects, taking into account the units and
        formats

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

class TimeTagEmCDSDistanceEventDataInSCFrameFromArrays(TimeTagEmCDSEventDataInSCFrameFromArrays,
                                                        TimeTagEmCDSDistanceEventDataInSCFrameInterface):

    event_type = TimeTagEmCDSDistanceEventInSCFrameInterface

    def __init__(self,
                   jd1: np.ndarray[float],
                   jd2: np.ndarray[float],
                   energy_keV: np.ndarray[float],
                   scattered_lon_rad_sc:  np.ndarray[float],
                   scattered_lat_rad_sc: np.ndarray[float],
                   scatt_angle_rad: np.ndarray[float],
                   distance_cm: np.ndarray[float],
                   event_id: Optional[np.ndarray[int]] = None,
                   selection: Optional[EventSelectorInterface] = None):
        """Initialize from bare numpy arrays. The user is responsible from
        getting the right units, coordinates and formats

        Parameters
        ----------
        jd1: Julian days. Internal astropy Time representation using
          two values for full precision.
        jd2: Julian days. Internal astropy Time representation using
          two values for full precision.
        energy_keV: energy [keV]
        scattered_lon_rad_sc: Longitude of the direction of the
          scattered photon in spacecraft coordinates [radian]
        scattered_lat_rad_sc: Latitude of the direction of the
          scattered photon in spacecraft coordinates [radian]
        scatt_angle_rad: Compton scattering angle [radians]
        distance_cm: distance between the first two hits [cm]
        event_id: Event ID. Optional. Sequential is not provided
        selection: Optional. Apply an event selection.

        """

        # Check size
        self._jd1, self._jd2, energy, scatt_angle, scatt_lon, scatt_lat, self._distance_cm = np.broadcast_arrays(
            jd1, jd2, energy_keV, scatt_angle_rad, scattered_lon_rad_sc, scattered_lat_rad_sc, distance_cm)

        super().__init__(self._jd1, self._jd2, energy, scatt_lon, scatt_lat, scatt_angle, event_id)

        self._apply_selection(selection,
                               ["_jd1", "_jd2", "_energy", "_scatt_angle", "_scatt_lat", "_scatt_lon",
                                "_distance_cm"])

    @classmethod
    def from_astropy(cls,
                     time:Time,
                     energy:Quantity,
                     scattering_angle:Angle,
                     scattered_direction:SkyCoord,
                     distance:Quantity,
                     event_id:Optional[Iterable[int]] = None,
                    selection:Optional[EventSelectorInterface] = None):
        """
        Initialize from astropy objects, taking into account the units and
        formats

        Parameters
        ----------
        time
        energy
        scattering_angle
        scattered_direction
        distance: distance between the first two hits
        event_id
        selection

        """

        jd1 = time.jd1
        jd2 = time.jd2
        energy = energy.to_value(u.keV)
        scatt_angle = scattering_angle.to_value(u.rad)
        distance_cm = distance.to_value(u.cm)

        if not isinstance(scattered_direction.frame, SpacecraftFrame):
            raise ValueError("Coordinates need to be in SC frame")

        scattered_direction = scattered_direction.represent_as(UnitSphericalRepresentation)

        scatt_lat = scattered_direction.lat.rad
        scatt_lon = scattered_direction.lon.rad

        if event_id is not None:
            event_id = np.asarray(event_id)

        return cls(jd1, jd2, energy, scatt_lon, scatt_lat, scatt_angle, distance_cm, event_id, selection)

    def __getitem__(self, i: int) -> TimeTagEmCDSDistanceEventInSCFrameInterface:
        return TimeTagEmCDSDistanceEventInSCFrame(self._jd1[i], self._jd2[i], self._energy[i], self._scatt_angle[i],
                                                   self._scatt_lon[i], self._scatt_lat[i], self._distance_cm[i],
                                                   self._id[i])

    def __iter__(self) -> Iterator[TimeTagEmCDSDistanceEventInSCFrameInterface]:
        for id, jd1, jd2, energy, scatt_angle, scatt_lat, scatt_lon, distance_cm in zip(
                self._id, self._jd1, self._jd2, self._energy, self._scatt_angle, self._scatt_lat, self._scatt_lon,
                self._distance_cm):
            yield TimeTagEmCDSDistanceEventInSCFrame(jd1, jd2, energy, scatt_angle, scatt_lon, scatt_lat,
                                                     distance_cm, id)

    @property
    def distance_cm(self) -> Iterable[float]:
        return self._distance_cm

def _load_dc3_columns(data_path: Union[Path, List[Path]],
                       extra_columns: Iterable[str] = ()) -> dict:
    """
    Read the standard DC3 fits columns needed to build a
    TimeTagEmCDSEventDataInSCFrameFromArrays (plus any extra columns
    requested) from one or more fits or hdf5 files, and time-sort the result.

    Parameters
    ----------
    data_path: Single fits or hdf5 file, or list of files (concatenated)
    extra_columns: Names of any additional fits columns to read, e.g. 'Distance'

    Returns
    -------
    dict mapping each column name to its time-sorted, concatenated array
    """

    columns = ['TimeTags', 'Energies', 'Phi', 'Psi local', 'Chi local', *extra_columns]

    if isinstance(data_path, (str, Path)):
        data_path = [Path(data_path)]

    data = {column: np.empty(0) for column in columns}

    for file in data_path:
        # get_dict_from_fits/hdf5 is really a static method, no config file needed
        # You can pass either hdf5 or fits file
        if "hdf5" in str(file).split("."):
            data_dict = UnBinnedData.get_dict_from_hdf5(None, str(file))
        elif "fits" in str(file).split("."):
            data_dict = UnBinnedData.get_dict_from_fits(None, str(file))
        else:
            raise ValueError("You should pass a fits or hdf5 file.")

        for column in columns:
            data[column] = np.append(data[column], data_dict[column])

    # Time sort
    tsort = np.argsort(data['TimeTags'])

    for column in columns:
        data[column] = data[column][tsort]

    return data

class TimeTagEmCDSEventDataInSCFrameFromDC3Fits(TimeTagEmCDSEventDataInSCFrameFromArrays):

    def __init__(self, data_path: Union[Path, List[Path]],
                 selection:EventSelectorInterface = None):

        data = _load_dc3_columns(data_path)

        time = Time(data['TimeTags'], format='unix')
        energy = data['Energies']
        phi = data['Phi']
        psi = data['Psi local']
        chi = data['Chi local']

        # Psi is colatitude (latitude complementary angle)
        super().__init__(time.jd1, time.jd2, energy, chi, np.pi / 2 - psi, phi, selection = selection)

class TimeTagEmCDSDistanceEventDataInSCFrameFromDC3Fits(TimeTagEmCDSDistanceEventDataInSCFrameFromArrays):

    def __init__(self, data_path: Union[Path, List[Path]],
                 selection:EventSelectorInterface = None):

        data = _load_dc3_columns(data_path, extra_columns=['Distance'])

        time = Time(data['TimeTags'], format='unix')
        energy = data['Energies']
        phi = data['Phi']
        psi = data['Psi local']
        chi = data['Chi local']
        distance = data['Distance']

        # Psi is colatitude (latitude complementary angle)
        super().__init__(time.jd1, time.jd2, energy, chi, np.pi / 2 - psi, phi, distance, selection = selection)

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
        scatt_lon_gal: scattering longitude radians (Gal frame)
        scatt_lat_gal: scattering latitude radians (Gal frame)
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
    def scattered_lon_rad_gal(self) -> float:
        return self._scatt_lon_gal

    @property
    def scattered_lat_rad_gal(self) -> float:
        return self._scatt_lat_gal

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
        scatt_lon_gal: scattering longitude radians (Gal frame)
        scatt_lat_gal: scattering latitude radians (Gal frame)
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

class EmCDSEventDataInSCAndGalFrameFromArrays(EmCDSEventDataInSCAndGalFrameInterface):

    _frame = "SpacecraftAndGalacticFrame"
    event_type = EmCDSEventInSCAndGalFrameInterface

    def __init__(self,
                   energy_keV: np.ndarray[float],
                   scattered_lon_rad_sc:  np.ndarray[float],
                   scattered_lat_rad_sc: np.ndarray[float],
                   scattered_lon_rad_gal:  np.ndarray[float],
                   scattered_lat_rad_gal: np.ndarray[float],
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
        scattered_lon_rad_gal: Longitude of the direction of the scattered photon in galactic coordinates [radians]
        scattered_lat_rad_gal:  Latitude of the direction of the scattered photon in galactic coordinates [radians]
        scatt_angle_rad: Compton scattering angle [radians]
        event_id: Event ID. Optional. Sequential is not provided
        selection: Optional. Apply an event selection.
        """

        # Check size
        self._energy, self._scatt_angle, self._scatt_lon_sc, self._scatt_lat_sc, self._scatt_lon_gal, self._scatt_lat_gal = np.broadcast_arrays(energy_keV, scatt_angle_rad, scattered_lon_rad_sc, scattered_lat_rad_sc, scattered_lon_rad_gal, scattered_lat_rad_gal)

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
        
        if not isinstance(scattered_direction_gal.frame, Galactic):
            raise ValueError("Galactic PsiChi Coordinates need to be in Galactic frame")

        
        scattered_direction_sc = scattered_direction_sc.represent_as(UnitSphericalRepresentation)
        scattered_direction_gal = scattered_direction_gal.represent_as(UnitSphericalRepresentation)

        scatt_lat_sc = scattered_direction_sc.lat.rad
        scatt_lon_sc = scattered_direction_sc.lon.rad
        scatt_lat_gal = scattered_direction_gal.lat.rad
        scatt_lon_gal = scattered_direction_gal.lon.rad

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
    def scattered_lon_rad_gal(self) -> Iterable[float]:
        return self._scatt_lon_gal

    @property
    def scattered_lat_rad_gal(self) -> Iterable[float]:
        return self._scatt_lat_gal

class TimeTagEmCDSEventDataInSCAndGalFrameFromArrays(EmCDSEventDataInSCAndGalFrameFromArrays, TimeTagEmCDSEventDataInSCAndGalFrameInterface):

    event_type = TimeTagEmCDSEventInSCAndGalFrameInterface

    def __init__(self,
                   jd1: np.ndarray[float],
                   jd2: np.ndarray[float],
                   energy_keV: np.ndarray[float],
                   scattered_lon_rad_sc:  np.ndarray[float],
                   scattered_lat_rad_sc: np.ndarray[float],
                   scattered_lon_rad_gal:  np.ndarray[float],
                   scattered_lat_rad_gal: np.ndarray[float],
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
        scattered_lon_rad_gal: Longitude of the direction of the scattered photon in galactic coordinates [radians]
        scattered_lat_rad_gal:  Latitude of the direction of the scattered photon in galactic coordinates [radians]
        scatt_angle_rad: Compton scattering angle [radians]
        event_id: Event ID. Optional. Sequential is not provided
        selection: Optional. Apply an event selection.
        """

        # Check size
        self._jd1, self._jd2, energy, scatt_angle, scatt_lon_sc, scatt_lat_sc, scatt_lon_gal, scatt_lat_gal = np.broadcast_arrays(
            jd1, jd2, energy_keV, scatt_angle_rad, scattered_lon_rad_sc, scattered_lat_rad_sc, scattered_lon_rad_gal, scattered_lat_rad_gal)

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
        
        if not isinstance(scattered_direction_gal.frame, Galactic):
            raise ValueError("Galactic PsiChi Coordinates need to be in Galactic frame")

        scattered_direction_sc = scattered_direction_sc.represent_as(UnitSphericalRepresentation)
        scattered_direction_gal = scattered_direction_gal.represent_as(UnitSphericalRepresentation)

        scatt_lat_sc = scattered_direction_sc.lat.rad
        scatt_lon_sc = scattered_direction_sc.lon.rad
        scatt_lat_gal = scattered_direction_gal.lat.rad
        scatt_lon_gal = scattered_direction_gal.lon.rad

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

class TimeTagEmCDSEventDataInSCAndGalFrameFromDC3Fits(TimeTagEmCDSEventDataInSCAndGalFrameFromArrays):

    def __init__(self, data_path: Union[Path, List[Path]],
                 selection:EventSelectorInterface = None):

        data = _load_dc3_columns(data_path, extra_columns=['Psi galactic', 'Chi galactic'])

        time = Time(data['TimeTags'], format='unix')
        energy = data['Energies']
        phi = data['Phi']
        psi_local = data['Psi local']
        chi_local = data['Chi local']
        psi_gal = data['Psi galactic']
        chi_gal = data['Chi galactic']

        # Psi is colatitude (latitude complementary angle)
        super().__init__(time.jd1, time.jd2, energy, chi_local, np.pi / 2 - psi_local, np.deg2rad(chi_gal), np.deg2rad(psi_gal), phi, selection = selection)
