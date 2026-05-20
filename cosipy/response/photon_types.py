from typing import TypeVar, Generic, Iterator

import numpy as np

from cosipy.interfaces.photon_parameters import (
    PhotonWithDirectionAndEnergyInSCFrameInterface,
    PolarizedPhotonWithDirectionAndEnergyInSCFrameStereographicConventionInterface,
    PhotonWithEnergyInterface,
    PhotonListWithEnergyInterface,
    PhotonInterface,
    PhotonListWithDirectionAndEnergyInSCFrameInterface,
    PolarizedPhotonListWithDirectionAndEnergyInSCFrameStereographicConventionInterface,
    PhotonWithDirectionInSCFrameInterface,
    PhotonListWithDirectionInSCFrameInterface
)


T = TypeVar("T")

class PhotonWithEnergyGen(Generic[T]):

    def __init__(self, energy_keV: T):
        self._energy = energy_keV

    @property
    def energy_keV(self) -> T:
        return self._energy

class PhotonWithEnergy(PhotonWithEnergyGen[float],
                       PhotonWithEnergyInterface):...

class PhotonListWithEnergy(PhotonWithEnergyGen[np.ndarray[float]],
                           PhotonListWithEnergyInterface):

    def __iter__(self) -> Iterator[PhotonInterface]:
        for energy_keV in self.energy_keV:
            yield PhotonWithEnergy(energy_keV)

    def nphotons(self) -> int:
        return self._energy.size

    def __getitem__(self, item):
        return PhotonWithEnergy(self._energy[item])

class PhotonWithDirectionInSCFrameGen(Generic[T]):

    def __init__(self, direction_lon_radians: T, direction_lat_radians: T):

        self._lon = direction_lon_radians
        self._lat = direction_lat_radians

    @property
    def direction_lon_rad_sc(self) -> T:
        return self._lon

    @property
    def direction_lat_rad_sc(self) -> T:
        return self._lat

class PhotonWithDirectionInSCFrame(PhotonWithDirectionInSCFrameGen[float],
                                   PhotonWithDirectionInSCFrameInterface): ...

class PhotonListWithDirectionInSCFrame(PhotonWithDirectionInSCFrameGen[np.ndarray[float]], PhotonListWithDirectionInSCFrameInterface):

    def __iter__(self) -> Iterator[PhotonInterface]:
        for energy, lon, lat in zip(self.direction_lon_rad_sc, self.direction_lat_rad_sc):
            yield PhotonWithDirectionInSCFrame(lon, lat)

    def __getitem__(self, item):
        return PhotonWithDirectionInSCFrame(self.direction_lon_rad_sc[item], self.direction_lat_rad_sc[item])


class PhotonWithDirectionAndEnergyInSCFrameGen(PhotonWithDirectionInSCFrameGen[T], PhotonWithEnergyGen[T]):

    def __init__(self, direction_lon_radians: T, direction_lat_radians: T, energy_keV: T):
        PhotonWithDirectionInSCFrameGen.__init__(self, direction_lon_radians, direction_lat_radians)
        PhotonWithEnergyGen.__init__(self, energy_keV)

class PhotonWithDirectionAndEnergyInSCFrame(PhotonWithDirectionAndEnergyInSCFrameGen[float], PhotonWithDirectionAndEnergyInSCFrameInterface): ...

class PhotonListWithDirectionAndEnergyInSCFrame(PhotonWithDirectionAndEnergyInSCFrameGen[np.ndarray[float]], PhotonListWithDirectionAndEnergyInSCFrameInterface):

    def __iter__(self) -> Iterator[PhotonWithDirectionAndEnergyInSCFrame]:
        for energy, lon, lat in zip(self.energy_keV, self.direction_lon_rad_sc, self.direction_lat_rad_sc):
            yield PhotonWithDirectionAndEnergyInSCFrame(lon, lat, energy)

    def __getitem__(self, item):
        return PhotonWithDirectionAndEnergyInSCFrame(self.direction_lon_rad_sc[item], self.direction_lat_rad_sc[item], self._energy[item])

class PolarizedPhotonWithDirectionAndEnergyInSCFrameStereographicConventionGen(PhotonWithDirectionAndEnergyInSCFrameGen[T]):

    def __init__(self, direction_lon_radians: T, direction_lat_radians: T, energy_keV: T, polarization_angle_radians: T):

        super().__init__(direction_lon_radians, direction_lat_radians, energy_keV)

        self._pa = polarization_angle_radians

    @property
    def polarization_angle_rad_stereo(self) -> T:
        return self._pa


class PolarizedPhotonWithDirectionAndEnergyInSCFrameStereographicConvention(PolarizedPhotonWithDirectionAndEnergyInSCFrameStereographicConventionGen[float], PolarizedPhotonWithDirectionAndEnergyInSCFrameStereographicConventionInterface): ...

class PolarizedPhotonListWithDirectionAndEnergyInSCFrameStereographicConvention(PolarizedPhotonWithDirectionAndEnergyInSCFrameStereographicConventionGen[np.ndarray[float]], PolarizedPhotonListWithDirectionAndEnergyInSCFrameStereographicConventionInterface):

    def __iter__(self) -> Iterator[PhotonInterface]:
        for energy, lon, lat, pa in zip(self.energy_keV, self.direction_lon_rad_sc, self.direction_lat_rad_sc, self.polarization_angle_rad_stereo):
            yield PolarizedPhotonWithDirectionAndEnergyInSCFrameStereographicConvention(lon, lat, energy, pa)

    def __getitem__(self, item):
        return PolarizedPhotonWithDirectionAndEnergyInSCFrameStereographicConvention(self.direction_lon_rad_sc[item], self.direction_lat_rad_sc[item], self._energy[item], self.polarization_angle_rad_stereo[item])
