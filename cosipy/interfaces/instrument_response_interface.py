import itertools
import operator
from typing import Protocol, Union, Optional, Iterable, Tuple, runtime_checkable, ClassVar, Type

from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.units import Quantity
from histpy import Axes, Histogram

from astropy import units as u
from numba.core.event import broadcast
from scoords import Attitude

from cosipy.interfaces import BinnedDataInterface, ExpectationDensityInterface, BinnedExpectationInterface, \
    EventInterface, EventDataInterface
from cosipy.interfaces.data_interface import is_single_event
from cosipy.interfaces.photon_parameters import PhotonInterface, PhotonWithDirectionInSCFrameInterface, \
    PhotonListWithDirectionInterface, PhotonListInterface, PhotonListWithDirectionInSCFrameInterface, \
    PhotonWithDirectionInterface, is_single_photon, PhotonListWithDirectionAndEnergyInSCFrameInterface, \
    PolarizedPhotonWithDirectionAndEnergyInSCFrameStereographicConventionInterface, \
    PolarizedPhotonListWithDirectionAndEnergyInSCFrameStereographicConventionInterface
from cosipy.polarization import PolarizationAngle

__all__ = ["BinnedInstrumentResponseInterface"]

from cosipy.response.photon_types import PhotonListWithDirectionAndEnergyInSCFrame


class BinnedInstrumentResponseInterface(Protocol):

    def differential_effective_area(self,
                                    direction: SkyCoord,
                                    energy:u.Quantity,
                                    polarization:PolarizationAngle,
                                    attitude:Attitude,
                                    time: Optional[Time],
                                    weight: Union[Quantity, float],
                                    out: Quantity,
                                    add_inplace: bool) -> Quantity:
        """

        Parameters
        ----------
        direction:
            Photon incoming direction. If not in a SpacecraftFrame, then provide an attitude for the transformation
        energy:
            Photon energy
        polarization
            Photon polarization angle. If the coordinate frame of the polarization convention is not a
            SpacecraftFrame, then provide an attitude for the transformation
        attitude
            Attitude defining the orientation of the SC in an inertial coordinate system.
        time:
            For time-dependent response
        weight
            Optional. Weighting the result by a given weight. Providing the weight at this point as opposed to
            apply it to the output can result in greater efficiency.
        out
            Optional. Histogram to store the output. If possible, the implementation should try to avoid allocating
            new memory.
        add_inplace
            Optional. If True and a Histogram output was provided, the implementation should try to avoid allocating new
            memory and add --not set-- the result of this operation to the output.

        Returns
        -------
        The effective area times the event measurement probability distribution integrated on each of the bins
        of the provided axes. It has the shape (direction.shape, energy.shape, polarization.shape, axes.shape)
        """

@runtime_checkable
class InstrumentResponseFunctionInterface(Protocol):

    # The photon class and event class that the IRF implementation can handle
    photon_list_type = PhotonListInterface
    event_data_type = EventDataInterface

    @property
    def photon_type(self) -> Type[PhotonInterface]:
        return self.photon_list_type.photon_type

    @property
    def event_type(self) -> Type[EventInterface]:
        return self.event_data_type.event_type

    def event_probability(self, photons:Union[PhotonInterface, PhotonListInterface], events: Union[EventInterface, EventDataInterface]) -> Union[float, Iterable[float]]:
        """
        Return the probability density of measuring a given event given a photon.

        The units of the output the inverse of the phase space of the class event_type data space.
        e.g. if the event measured energy in keV, the units of output of this function are implicitly 1/keV

        If we receive a single photon and a single event, the output is a scalar. Otherwise, it's an iterable.
        If we receive multiple photons and multiple event, their number must match

        Implementation can define only _event_probability assuming both photons and event are lists, and let this
        default function handle single photons and single events.
        """

        single_photon = is_single_photon(photons)
        single_event = is_single_event(events)

        if single_photon and single_event:
            # Just one. Output is scalar
            photons = self.photon_list_type.from_photon(photons)
            events = self.event_data_type.from_event(events)
            return next(iter(self._event_probability(photons, events)))
        else:
            # Output is iterable
            if single_photon:
                # Single photon, multiple events
                photons = self.photon_list_type.from_photon(photons, repeat=events.nevents)
            elif single_event:
                # Single event, multiple photons
                events = self.event_data_type.from_event(events, repeat=photons.nphotons)

            return self._event_probability(photons, events)

    def _event_probability(self, photons: PhotonListInterface, events: EventDataInterface) -> Iterable[float]:
        """
        This allows implementation to only define the behaviour for list, and let the above function handle
        the case for a single photon and/or a single event.

        The number of photons should match the number of events.
        """


    def random_events(self, photons:Union[PhotonInterface, PhotonListInterface]) -> Union[EventInterface, EventDataInterface]:
        """
        Generate one random event per photon.

        If we receive a single photon, the output is a single event. Otherwise, it's an event data stream.

        Implementation can define only _random_events assuming multiple photons, and let this
        default function handle single photon case
        """

        single_photon = is_single_photon(photons)

        if single_photon:
            photons = self.photon_list_type.from_photon(photons)
            return next(iter(self._random_events(photons)))
        else:
            return self._random_events(photons)

    def _random_events(self, photons:PhotonListInterface) -> EventDataInterface:
        """
        This allows implementation to only define the behaviour for list, and let the above function handle
        the case for a single photon and/or a single event.
        """

@runtime_checkable
class FarFieldInstrumentResponseFunctionInterface(InstrumentResponseFunctionInterface, Protocol):

    photon_list_type = PhotonListWithDirectionInSCFrameInterface

    def effective_area_cm2(self, photons: Union[PhotonWithDirectionInSCFrameInterface, PhotonListWithDirectionInSCFrameInterface]) -> Union[float,Iterable[float]]:
        """
        If we receive a single photon, the output is a scalar. Otherwise, it's an iterable

        Implementation can define only _effective_area_cm2 assuming multiple photons, and let this
        default function handle single photon case
        """

        single_photon = is_single_photon(photons)

        if single_photon:
            photons = self.photon_list_type.from_photon(photons)
            return next(iter(self._effective_area_cm2(photons)))
        else:
            return self._effective_area_cm2(photons)

    def _effective_area_cm2(self, photons: PhotonListWithDirectionInSCFrameInterface) -> Iterable[float]:
        """
        This allows implementation to only define the behaviour for list, and let the above function handle
        the case for a single photon and/or a single event.
        """

    def differential_effective_area_cm2(self, photons: Union[PhotonWithDirectionInSCFrameInterface, PhotonListWithDirectionInSCFrameInterface], events: Union[EventInterface, EventDataInterface]) -> Union[float,Iterable[float]]:
        """
        Event probability multiplied by effective area

        The units of the output are cm2 times the inverse of the phase space of the class event_type data space.
        e.g. if the event measured energy in keV, the units of output of this function are implicitly cm2/keV

        If we receive a single photon and a single event, the output is a scalar. Otherwise, it's an iterable.
        If we receive multiple photons and multiple event, their number must match

        Implementation can define only _differential_effective_area_cm2 assuming both photons and event are lists, and let this
        default function handle single photons and single events.
        """

        single_photon = is_single_photon(photons)
        single_event = is_single_event(events)

        if single_photon and single_event:
            # Just one. Output is scalar
            photons = self.photon_list_type.from_photon(photons)
            events = self.event_data_type.from_event(events)
            return next(iter(self._differential_effective_area_cm2(photons, events)))
        else:
            # Output is iterable
            if single_photon:
                # Single photon, multiple events
                photons = self.photon_list_type.from_photon(photons, repeat=events.nevents)
            elif single_event:
                # Single event, multiple photons
                events = self.event_data_type.from_event(events, repeat=photons.nphotons)

            return self._differential_effective_area_cm2(photons, events)

    def _differential_effective_area_cm2(self, photons:PhotonListWithDirectionInSCFrameInterface, events: EventDataInterface) -> Iterable[float]:
        """
        Event probability multiplied by effective area

        This is provided as a helper function assuming the child classes implemented _event_probability

        The number of photons should match the number of events.
        """

        # Guard to avoid infinite recursion in incomplete child classes
        cls = type(self)
        if (cls._differential_effective_area_cm2 is FarFieldInstrumentResponseFunctionInterface._differential_effective_area_cm2
            and
            cls._event_probability is FarFieldInstrumentResponseFunctionInterface._event_probability):
            raise NotImplementedError("Implement _differential_effective_area_cm2 and/or _event_probability")

        return map(operator.mul, self._effective_area_cm2(photons), self._event_probability(photons, events))

    def _event_probability(self, photons:PhotonListWithDirectionInSCFrameInterface, events: EventDataInterface) -> Iterable[float]:
        """
        Return the probability density of measuring a given event given a photon.

        In the far field case it is the same as the differential_effective_area_cm2 divided by the effective area

        This is provided as a helper function assuming the child classes implemented differential_effective_area_cm2
        """

        # Guard to avoid infinite recursion in incomplete child classes
        cls = type(self)
        if (
                cls._differential_effective_area_cm2 is FarFieldInstrumentResponseFunctionInterface._differential_effective_area_cm2
                and
                cls._event_probability is FarFieldInstrumentResponseFunctionInterface._event_probability):
            raise NotImplementedError("Implement _differential_effective_area_cm2 and/or _event_probability")

        return map(operator.truediv, self._differential_effective_area_cm2(photons, events), self._effective_area_cm2(photons))

    def effective_area(self, photons: Union[PhotonWithDirectionInSCFrameInterface, PhotonListWithDirectionInSCFrameInterface]) -> Union[u.Quantity,Iterable[u.Quantity]]:
        """
        Convenience function. Implementation might optimize it
        """
        cm2 = u.cm*u.cm
        if isinstance(photons, PhotonInterface):
            return u.Quantity(self.effective_area_cm2(photons), cm2)
        else:
            return (u.Quantity(area_cm2, cm2) for area_cm2 in self.effective_area_cm2(photons))

    def differential_effective_area(self, photons: Union[PhotonWithDirectionInSCFrameInterface, PhotonListWithDirectionInSCFrameInterface], events: Union[EventInterface, EventDataInterface]) -> Union[u.Quantity,Iterable[u.Quantity]]:
        """
        Convenience function. Implementation might optimize it
        """
        cm2 = u.cm * u.cm

        single_photon = is_single_photon(photons)
        single_event = is_single_event(events)

        if single_photon and single_event:
            # Just one. Output is scalar
            photons = self.photon_list_type.from_photon(photons)
            events = self.event_data_type.from_event(events)
            return u.Quantity(next(iter(self._differential_effective_area_cm2(photons, events))), cm2)
        else:
            # Output is iterable
            if single_photon:
                # Single photon, multiple events
                photons = self.photon_list_type.from_photon(photons, repeat=events.nevents)
            elif single_event:
                # Single event, multiple photons
                events = self.event_data_type.from_event(events, repeat=photons.nphotons)

            return (u.Quantity(area_cm2, cm2) for area_cm2 in self._differential_effective_area_cm2(photons, events))

@runtime_checkable
class FarFieldSpectralInstrumentResponseFunctionInterface(FarFieldInstrumentResponseFunctionInterface, Protocol):

    photon_list_type = PhotonListWithDirectionAndEnergyInSCFrameInterface

@runtime_checkable
class FarFieldSpectralPolarizedInstrumentResponseFunctionInterface(FarFieldSpectralInstrumentResponseFunctionInterface, Protocol):

    photon_list_type = PolarizedPhotonListWithDirectionAndEnergyInSCFrameStereographicConventionInterface

















