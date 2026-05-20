from typing import Dict, Union, Type, Iterable

import numpy as np

from astropy.coordinates import (
    SkyCoord,
    CartesianRepresentation,
    UnitSphericalRepresentation
)
from astropy.time import Time
from astropy import units as u

from histpy import Histogram

from scoords import SpacecraftFrame

from cosipy import SpacecraftHistory
from cosipy.interfaces import (
    BinnedBackgroundInterface,
    BackgroundDensityInterface,
    BackgroundInterface,
    EventInterface
)

from cosipy.interfaces.data_interface import TimeTagEmCDSEventDataInSCFrameInterface

from cosipy.interfaces.event import TimeTagEmCDSEventInSCFrameInterface
from cosipy.util.iterables import itertools_batched


__all__ = ["FreeNormBinnedBackground"]


class FreeNormBackground(BackgroundInterface):
    """
    This must translate to/from regular parameters
    with arbitrary type from/to 3ML parameters

    Default to "bkg_norm" is there was a single unlabeled component
    """

    _default_label = 'bkg_norm'

    def __init__(self,
                 distribution:Union[Histogram, Dict[str, Histogram]],
                 sc_history:SpacecraftHistory,
                 copy = True):
        """

        Parameters
        ----------
        distribution
        sc_history
        copy: copy hist distribution
        """

        if isinstance(distribution, Histogram):
            # Single component
            self._distributions = {self._default_label: distribution}
            self._norms = np.ones(1) # Hz. Each component
            self._norm = 1 # Hz. Total
            self._single_component = True
        else:
            # Multiple label components.
            self._distributions = distribution
            self._norms = np.ones(self.ncomponents) # Hz Each component
            self._norm = np.sum(self._norms) # Hz. Total
            self._single_component = False

        self._labels = tuple(self._distributions.keys())

        self._livetime = sc_history.cumulative_livetime().to_value(u.s)

        # These will be densified anyway since _expectation is dense
        # And histpy doesn't yet handle this operation efficiently
        # See Histogram._inplace_operation_handle_sparse()
        # Do it once and for all
        for label, bkg in self._distributions.items():
            self._distributions[label] = bkg.to_dense(copy=copy)

        # Normalize
        # Unit: second
        for label,dist in self._distributions.items():
            dist_norm = np.sum(dist)
            dist /= dist_norm

        if self.ncomponents == 0:
            raise ValueError("You need to input at least one components")

        self._axes = None
        for bkg in self._distributions.values():
            if self._axes is None:
                self._axes = bkg.axes
            else:
                if self._axes != bkg.axes:
                    raise ValueError("All background components mus have the same axes")

    @property
    def norm(self):
        """
        Sum of all rates
        """

        return u.Quantity(self._norm, u.Hz)

    @property
    def norms(self):
        if self._single_component:
            return {self._default_label: u.Quantity(self._norms[0], u.Hz)}
        else:
            return {l:u.Quantity(n, u.Hz, copy = None) for l,n in zip(self.labels,self._norms)}

    @property
    def ncomponents(self):
        return len(self._distributions)

    @property
    def axes(self):
        return self._axes

    @property
    def labels(self):
        return self._labels

    def set_norm(self, norm: Union[u.Quantity, Dict[str, u.Quantity]]):

        if self._single_component:
            if isinstance(norm, dict):
                self._norms[0] = norm[self._default_label].to_value(u.Hz)
            else:
                self._norms[0] = norm.to_value(u.Hz)
        else:
            # Multiple
            if not isinstance(norm, dict):
                raise TypeError("This a multi-component background. Provide labeled norm values in a dictionary")

            for label,norm_i in norm.items():
                if label not in self.labels:
                    raise ValueError(f"Norm {label} not in {self.labels}")

                self._norms[self.labels.index(label)] = norm_i.to_value(u.Hz)

        self._norm = sum(n for n in self._norms)

    def set_parameters(self, **parameters:u.Quantity) -> None:
        """
        Same keys as background components
        """

        self.set_norm(parameters)

    @property
    def parameters(self) -> Dict[str, u.Quantity]:
        return self.norms

class FreeNormBinnedBackground(FreeNormBackground, BinnedBackgroundInterface):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Cache
        self._expectation = None
        self._last_norm_values = None

    def expectation(self, copy:bool = True)->Histogram:
        """

        Parameters
        ----------
        copy:
            If True, it will return an array that the user if free to modify.
            Otherwise, it will result a reference, possible to the cache, that
            the user should not modify

        Returns
        -------

        """

        # Check if we can use the cache
        if self._expectation is None:
            # First call. Initialize
            self._expectation = Histogram(self.axes)

        elif self.norms == self._last_norm_values:
            # No changes. Use cache
            if copy:
                return self._expectation.copy()
            else:
                return self._expectation

        else:
            # First call or norms have change. Recalculate
            self._expectation.clear()

        # Compute expectation
        for label,bkg in self._distributions.items():
            norm = self._norms[self.labels.index(label)]
            self._expectation += bkg * norm * self._livetime

        # Cache. Regular copy is enough since norm values are float en not mutable
        self._last_norm_values = self.norms.copy()

        if copy:
            return self._expectation.copy()
        else:
            return self._expectation


class FreeNormBackgroundInterpolatedDensityTimeTagEmCDS(FreeNormBackground, BackgroundDensityInterface):

    @property
    def event_type(self) -> Type[EventInterface]:
        return TimeTagEmCDSEventInSCFrameInterface

    def __init__(self,
                 data:TimeTagEmCDSEventDataInSCFrameInterface,
                 distribution:Union[Histogram, Dict[str, Histogram]],
                 sc_history:SpacecraftHistory,
                 copy=True,
                 batch_size = 100000,
                 *args,
                 **kwargs):

        super().__init__(distribution, sc_history,
                         copy=copy, *args, **kwargs)

        # We need the density per phase space for the specific measurement units TimeTagEmCDSEventInSCFrameInterface
        # Energy: keV
        # Phi: rad
        # PsiChi: sr (for the phase space. The axis is a HealpixAxis)
        # Time: seconds (taken into account by the norm (a rate) unit)

        psichi_frame = None

        for label,dist in self._distributions.items():

            dist = self._distributions[label] = dist.project('Em', 'Phi', 'PsiChi')

            dist.axes['Em'] = dist.axes['Em'].to(u.keV).to(None, copy=False, update=False)
            dist.axes['Phi'] = dist.axes['Phi'].to(u.rad).to(None, copy=False, update=False)

            energy_phase_space = dist.axes['Em'].widths
            phi_phase_space = dist.axes['Phi'].widths
            psichi_phase_space = dist.axes['PsiChi'].pixarea().to_value(u.sr)

            if psichi_frame is None:
                psichi_frame = dist.axes['PsiChi'].coordsys
            else:
                if psichi_frame != dist.axes['PsiChi'].coordsys:
                    raise ValueError("All PsiChi axes must be in the same frame")

            dist /= dist.axes.expand_dims(energy_phase_space, 'Em')
            dist /= dist.axes.expand_dims(phi_phase_space, 'Phi')
            dist /= psichi_phase_space

        # Compute the probabilities once and for all
        # TODO: account for livetime
        self._prob = [[] for _ in range(self.ncomponents)]

        for events_chunk in itertools_batched(data, batch_size):

            jd1, jd2, energy,phi, psichi_lon, psichi_lat  = np.asarray([[
                event.jd1,
                event.jd2,
                event.energy_keV,
                                                               event.scattering_angle_rad,
                                                               event.scattered_lon_rad_sc,
                                                               event.scattered_lat_rad_sc]
                                               for event in events_chunk], dtype=float).transpose()

            times = Time(jd1, jd2, format = 'jd')

            # Transform local to inertial
            sc_psichi_coord = SkyCoord(psichi_lon, psichi_lat, unit=u.rad, frame=SpacecraftFrame())
            sc_psichi_vec = sc_psichi_coord.cartesian.xyz.value
            attitudes = sc_history.interp_attitude(times).transform_to(psichi_frame)
            inertial_psichi_vec = attitudes.rot.apply(sc_psichi_vec.transpose())
            inertial_psichi_sph = UnitSphericalRepresentation.from_cartesian(CartesianRepresentation(*inertial_psichi_vec.transpose()))
            inertial_psichi_coord = SkyCoord(inertial_psichi_sph, frame = psichi_frame)

            for label,dist in self._distributions.items():
                prob = dist.interp(energy, phi, inertial_psichi_coord)
                self._prob[self.labels.index(label)].extend(prob)

        self._prob = np.asarray(self._prob)

    def expected_counts(self) -> float:
        """
        Total expected counts
        """
        return self._livetime * self._norm

    def expectation_density(self) -> Iterable[float]:
        """
        Return the expected number of counts density from the start-th
        event to the stop-th event. This equals the event probabiliy
        times the number of events

        """

        # Multiply each probability by the norm, and then sum
        return np.tensordot(self._prob, self._norms, axes = (0,0))
