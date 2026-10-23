import itertools
from typing import Protocol, runtime_checkable, TypeVar, Iterable, Iterator, Union

from astropy.units import Quantity
from astropy import units as u
from astropy.coordinates import SkyCoord
from scoords import SpacecraftFrame

from cosipy.polarization import (
    PolarizationConvention,
    PolarizationAngle,
    StereographicConvention
)

T = TypeVar('T')

@runtime_checkable
class PhotonInterface(Protocol):
    """
    Derived classes have all access methods
    """

def is_single_photon(photon : Union[PhotonInterface, 'PhotonListInterface']) -> bool:
    # Since these protocols are runtime checkable, it's not enough to
    # check the input is PhotonInterface, since the
    # PhotonListInterface also returns True.
    if isinstance(photon, PhotonListInterface):
        return False
    else:
        if not isinstance(photon, PhotonInterface):
            raise ValueError("Input must be either a PhotonInterface or a PhotonListInterface.")

        return True


@runtime_checkable
class PhotonListInterface(Protocol):

    # Type returned by __iter__
    photon_type = PhotonInterface

    def __iter__(self) -> Iterator[PhotonInterface]:
        """
        Return one Event at a time
        """
    def __getitem__(self, item: int) -> PhotonInterface:
        """
        Convenience method. Pretty slow in general. It's suggested that
        the implementations override it
        """
        return next(itertools.islice(self, item, None))

    @classmethod
    def fromiter(cls, photons:Iterable[PhotonInterface], nphotons = None):
        """Turns an iterable of photon into a proper PhotonList

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

        class PhotonListIterableWrapper(cls):

            def __init__(self):
                self._nphotons = nphotons

            def __iter__(self) -> Iterator[cls.photon_type]:
                return iter(photons)

            @property
            def nphotons(self) -> int:
                if nphotons is None:
                    self._nphotons = sum(1 for _ in iter(self))

                return self._nphotons

        photon_list = PhotonListIterableWrapper()

        return photon_list

    @classmethod
    def from_photon(cls, photon: PhotonInterface, repeat = 1):
        """
        Convert a single photon to a proper PhotonList

        Parameters
        ----------
        photon:
        repeat: Number of time to repeat the photon.

        Returns
        -------
        tuple: is_single? (bool), photon_list
        """

        if not getattr(cls, "_is_protocol", False):
            raise RuntimeError("It is not safe to call from_photon() from an interface implementation that"
                               "does not explicitly overload from_photon(). Call it from an interface instead.")

        if not isinstance(photon, cls.photon_type):
            raise RuntimeError(
                f"Input photon (type {type(photon)}) is not an instance of {cls.photon_type}")

        class SinglePhotonListWrapper(cls):
            def __iter__(self) -> Iterator[cls.photon_type]:
                for _ in range(repeat):
                    yield photon

            def __getitem__(self, item):
                return photon

            def nphotons(self) -> int:
                return repeat

        photon_list = SinglePhotonListWrapper.__new__(SinglePhotonListWrapper)

        return photon_list

    @property
    def nphotons(self) -> int:
        """
        Total number of events yielded by __iter__

        Convenience method. Pretty slow in general. It's suggested that
        the implementations override it
        """
        return sum(1 for _ in iter(self))

@runtime_checkable
class PhotonWithEnergyInterface(PhotonInterface, Protocol):

    @property
    def energy_keV(self) -> float: ...

    @property
    def energy(self) -> Quantity:
        """
        Add fancy energy quantity
        """
        return Quantity(self.energy_keV, u.keV, copy=None)


@runtime_checkable
class PhotonListWithEnergyInterface(PhotonListInterface, Protocol):

    photon_type = PhotonWithEnergyInterface

    def __iter__(self) -> Iterator[PhotonWithEnergyInterface]:...

    @property
    def energy_keV(self) -> Iterable[float]:
        return [e.energy_keV for e in self]

    @property
    def energy(self) -> Quantity:
        """
        Add fancy energy quantity
        """
        return Quantity(self.energy_keV, u.keV, copy=False)


@runtime_checkable
class PhotonWithDirectionInterface(PhotonInterface, Protocol):

    @property
    def direction(self) -> SkyCoord:
        """
        Add fancy SkyCoord
        """

@runtime_checkable
class PhotonListWithDirectionInterface(PhotonListInterface, Protocol):

    photon_type = PhotonWithDirectionInterface

    def __iter__(self) -> Iterator[PhotonWithDirectionInterface]: ...

    @property
    def direction(self) -> SkyCoord:
        """
        Add fancy SkyCoord
        """

@runtime_checkable
class PhotonWithDirectionInSCFrameInterface(PhotonWithDirectionInterface, Protocol):

    @property
    def direction_lon_rad_sc(self) -> float: ...

    @property
    def direction_lat_rad_sc(self) -> float: ...

    @property
    def direction(self) -> SkyCoord:
        """
        Add fancy energy quantity
        """
        return SkyCoord(self.direction_lon_rad_sc,
                        self.direction_lat_rad_sc,
                        unit=u.rad,
                        frame=SpacecraftFrame())

@runtime_checkable
class PhotonListWithDirectionInSCFrameInterface(PhotonListWithDirectionInterface, Protocol):

    photon_type = PhotonWithDirectionInSCFrameInterface

    def __iter__(self) -> Iterator[PhotonWithDirectionInSCFrameInterface]: ...

    @property
    def direction_lon_rad_sc(self) -> Iterable[float]:
        return [e.direction_lon_rad_sc for e in self]

    @property
    def direction_lat_rad_sc(self) -> Iterable[float]:
        return [e.direction_lat_rad_sc for e in self]

    @property
    def direction(self) -> SkyCoord:
        """
        Add fancy energy quantity
        """
        return SkyCoord(self.direction_lon_rad_sc,
                        self.direction_lat_rad_sc,
                        unit=u.rad,
                        frame=SpacecraftFrame())


@runtime_checkable
class PhotonWithDirectionAndEnergyInSCFrameInterface(PhotonWithDirectionInSCFrameInterface,
                                                     PhotonWithEnergyInterface, Protocol):...

@runtime_checkable
class PhotonListWithDirectionAndEnergyInSCFrameInterface(PhotonListWithDirectionInSCFrameInterface,
                                                         PhotonListWithEnergyInterface, Protocol):...


@runtime_checkable
class PolarizedPhotonInterfaceGen(Protocol):

    @property
    def polarization_angle(self) -> PolarizationAngle:
        """
        This convenience function only makes sense for implementations
        that couple with PhotonWithDirectionInterface
        """
        raise NotImplementedError("This class does not implement the polarization_angle() convenience method.")


@runtime_checkable
class PolarizedPhotonInterface(PolarizedPhotonInterfaceGen, PhotonInterface, Protocol):...
@runtime_checkable
class PolarizedPhotonListInterface(PolarizedPhotonInterfaceGen, PhotonListInterface, Protocol):...

@runtime_checkable
class PolarizedPhotonStereographicConventionInSCInterfaceGen(PolarizedPhotonInterfaceGen, Protocol[T]):

    @property
    def polarization_angle_rad_stereo(self) -> T: ...

    @property
    def polarization_convention(self) -> PolarizationConvention:
        return StereographicConvention()

@runtime_checkable
class PolarizedPhotonStereographicConventionInSCInterface(PolarizedPhotonStereographicConventionInSCInterfaceGen[float], PolarizedPhotonInterface, Protocol):...
@runtime_checkable
class PolarizedPhotonListStereographicConventionInSCInterface(PolarizedPhotonStereographicConventionInSCInterfaceGen[Iterable[float]], PolarizedPhotonListInterface, Protocol):
    @property
    def polarization_angle_rad_stereo(self) -> Iterable[float]:
        return [e.polarization_angle_rad_stereo for e in self]


@runtime_checkable
class PolarizedPhotonWithDirectionAndEnergyInSCFrameStereographicConventionInterfaceGen(PolarizedPhotonStereographicConventionInSCInterfaceGen, Protocol[T]):

    @property
    def polarization_angle(self) -> PolarizationAngle:
        return PolarizationAngle(Quantity(self.polarization_angle_rad_stereo, u.rad, copy = None), self.direction, 'stereographic')

@runtime_checkable
class PolarizedPhotonWithDirectionAndEnergyInSCFrameStereographicConventionInterface(PolarizedPhotonWithDirectionAndEnergyInSCFrameStereographicConventionInterfaceGen[float], PhotonWithDirectionAndEnergyInSCFrameInterface, PolarizedPhotonStereographicConventionInSCInterface, Protocol):...
@runtime_checkable
class PolarizedPhotonListWithDirectionAndEnergyInSCFrameStereographicConventionInterface(PolarizedPhotonWithDirectionAndEnergyInSCFrameStereographicConventionInterfaceGen[Iterable[float]], PhotonListWithDirectionAndEnergyInSCFrameInterface, PolarizedPhotonListStereographicConventionInSCInterface, Protocol):...
