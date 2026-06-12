from typing import Iterable, Tuple

import numpy as np
from astropy.coordinates import SkyCoord

from astropy.units import Quantity
from astropy import units as u

from scoords import SpacecraftFrame

from cosipy.response import FullDetectorResponse
from cosipy.util.iterables import itertools_batched

from cosipy.interfaces.data_interface import EmCDSEventDataInSCFrameInterface
from cosipy.interfaces.event import EmCDSEventInSCFrameInterface
from cosipy.interfaces.instrument_response_interface import FarFieldSpectralInstrumentResponseFunctionInterface
from cosipy.interfaces.photon_parameters import PhotonWithDirectionAndEnergyInSCFrameInterface


class UnpolarizedDC3InterpolatedFarFieldInstrumentResponseFunction(FarFieldSpectralInstrumentResponseFunctionInterface):

    event_data_type = EmCDSEventDataInSCFrameInterface

    def __init__(self, response: FullDetectorResponse,
                 batch_size = 100000):

        # Get the differential effective area, which is still
        # integrated on each bin at this point
        # FarFieldInstrumentResponseFunctionInterface uses cm2
        # First convert and then drop the units
        self._diff_area = response.to_dr().project('NuLambda', 'Ei', 'Em', 'Phi', 'PsiChi').to(u.cm * u.cm, copy=False).to(None, copy = False, update = False)

        # Now fix units for the axes
        # PhotonWithDirectionAndEnergyInSCFrameInterface has energy in keV
        # EmCDSEventInSCFrameInterface has energy in keV, phi in rad
        # NuLambda and PsiChi don't have units since these are
        # HealpixAxis. They take SkyCoords
        # Copy the axes the first time since they are shared with the
        # response:FullDetectorResponse input
        self._diff_area.axes['Ei'] = self._diff_area.axes['Ei'].to(u.keV).to(None, copy = False, update = False)
        self._diff_area.axes['Em'] = self._diff_area.axes['Em'].to(u.keV).to(None, copy = False, update = False)
        self._diff_area.axes['Phi'] = self._diff_area.axes['Phi'].to(u.rad).to(None, copy = False, update = False)

        # Integrate to get the total effective area
        self._area = self._diff_area.project('NuLambda', 'Ei')

        # Now make it differential by dividing by the phasespace
        # EmCDSEventInSCFrameInterface energy and phi units have
        # already been taken care off. Only PsiChi remains, which is a
        # direction in the sphere, therefore per steradians
        energy_phase_space =  self._diff_area.axes['Ei'].widths
        phi_phase_space = self._diff_area.axes['Phi'].widths
        psichi_phase_space = self._diff_area.axes['PsiChi'].pixarea().to_value(u.sr)

        self._diff_area /= self._diff_area.axes.expand_dims(energy_phase_space, 'Em')
        self._diff_area /= self._diff_area.axes.expand_dims(phi_phase_space, 'Phi')
        self._diff_area /= psichi_phase_space

        self._batch_size = batch_size

    def effective_area_cm2(self, photons: Iterable[PhotonWithDirectionAndEnergyInSCFrameInterface]) -> Iterable[float]:
        """

        """

        for photon_chunk in itertools_batched(photons, self._batch_size):

            lon, lat, energy_keV = np.asarray([[photon.direction_lon_rad_sc,
                                             photon.direction_lat_rad_sc,
                                             photon.energy_keV] for photon in photon_chunk], dtype=float).transpose()

            direction = SkyCoord(lon, lat, unit = u.rad, frame = SpacecraftFrame())

            for area_eff in self._area.interp(direction, energy_keV):
                yield area_eff

    def differential_effective_area_cm2(self, query: Iterable[Tuple[PhotonWithDirectionAndEnergyInSCFrameInterface, EmCDSEventInSCFrameInterface]]) -> Iterable[float]:
        """
        Return the differential effective area (probability density of
        measuring a given event given a photon times the effective
        area)

        """

        for query_chunk in itertools_batched(query, self._batch_size):

            # Psi is colatitude (complementary angle)
            lon_ph, lat_ph, energy_i_keV, energy_m_keV, phi_rad, psi_comp, chi  = \
                np.asarray([[photon.direction_lon_rad_sc,
                             photon.direction_lat_rad_sc,
                             photon.energy_keV,
                             event.energy_keV,
                             event.scattering_angle_rad,
                             event.scattered_lat_rad_sc,
                             event.scattered_lon_rad_sc,
                            ] for photon,event in query_chunk], dtype=float).transpose()

            direction_ph = SkyCoord(lon_ph, lat_ph, unit = u.rad, frame = SpacecraftFrame())
            psichi = SkyCoord(chi, psi_comp, unit=u.rad, frame=SpacecraftFrame())

            for diff_area in self._diff_area.interp(direction_ph, energy_i_keV, energy_m_keV, phi_rad, psichi):
                yield diff_area
