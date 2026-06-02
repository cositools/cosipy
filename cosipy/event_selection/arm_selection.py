#!/usr/bin/env python
# coding: utf-8

# In[1]:


import logging

logger = logging.getLogger(__name__)

from typing import Iterable

import numpy as np
import astropy.units as u
from astropy.coordinates import Angle, SkyCoord
from scoords import SpacecraftFrame

from cosipy.interfaces.data_interface import (
    ComptonDataSpaceInSCFrameEventDataInterface,
)
from cosipy.interfaces.event_selection import EventSelectorInterface
from cosipy.polarization import PolarizationConvention
from cosipy.response.relative_coordinates import RelativeCDSCoordinates


class ARMSelector(EventSelectorInterface):
    """
    ARM event selector.

    Selects Compton events satisfying:

        |ARM| <= arm_cut

    where:

        ARM = angular_distance(source_direction, scattered_direction) - Phi

    In this implementation:

        - scattered_direction is obtained from the input event-data object via
          events.scattered_direction_sc
        - Phi is obtained from the input event-data object via
          events.scattering_angle_rad
        - the angular distance is computed using RelativeCDSCoordinates

    This follows the EventSelectorInterface pattern: _select() accepts an
    EventDataInterface object and returns one Boolean value per event.

    Notes
    -----
    The input event data must satisfy the
    ComptonDataSpaceInSCFrameEventDataInterface protocol.

    The target coordinate is expected to be in the spacecraft frame for this
    selector version, because the event scattered directions are also in the
    spacecraft frame.
    """

    event_data_type = ComptonDataSpaceInSCFrameEventDataInterface

    def __init__(
        self,
        target_coord: SkyCoord,
        arm_cut: u.Quantity,
        polarization_convention="RelativeX",
    ):
        """
        Parameters
        ----------
        target_coord : astropy.coordinates.SkyCoord
            Source direction. For this selector version, this should be given
            in SpacecraftFrame, because events.scattered_direction_sc is in
            SpacecraftFrame.

        arm_cut : astropy.units.Quantity
            Maximum absolute ARM value accepted by the selector.
            Example: 13.0 * u.deg

        polarization_convention : str or PolarizationConvention, optional
            Polarization convention used by RelativeCDSCoordinates.
            Default is "RelativeX", matching the MEGAlib-style relative
            coordinate convention.
        """

        if not isinstance(target_coord, SkyCoord):
            raise TypeError(
                "target_coord must be an astropy.coordinates.SkyCoord object."
            )

        if not isinstance(target_coord.frame, SpacecraftFrame):
            raise ValueError(
                "target_coord must be in SpacecraftFrame for this "
                "ARMSelector version. The input events provide scattered "
                "directions in spacecraft coordinates, so the source direction "
                "must be in the same frame."
            )

        self._target_coord = target_coord
        self._arm_cut = Angle(arm_cut).to(u.deg)

        if isinstance(polarization_convention, str):
            polarization_convention = PolarizationConvention.get_convention(
                polarization_convention
            )
        elif not isinstance(polarization_convention, PolarizationConvention):
            raise TypeError(
                "polarization_convention must be either a string registered "
                "with PolarizationConvention or a PolarizationConvention object."
            )

        self._polarization_convention = polarization_convention

        self._relative_cds = RelativeCDSCoordinates(
            source_direction=self._target_coord,
            pol_convention=self._polarization_convention,
        )

    @property
    def target_coord(self):
        """
        Source direction used for the ARM calculation.
        """

        return self._target_coord

    @property
    def arm_cut(self):
        """
        Maximum accepted absolute ARM value.
        """

        return self._arm_cut

    @property
    def polarization_convention(self):
        """
        Polarization convention used by RelativeCDSCoordinates.
        """

        return self._polarization_convention

    def _compute_arm(
        self,
        events: ComptonDataSpaceInSCFrameEventDataInterface,
    ):
        """
        Compute ARM values for Compton event data.

        Parameters
        ----------
        events : ComptonDataSpaceInSCFrameEventDataInterface
            Event data object providing:

                - events.scattered_direction_sc
                - events.scattering_angle_rad

        Returns
        -------
        astropy.coordinates.Angle
            ARM values in degrees.
        """

        scattered_direction = events.scattered_direction_sc
        scattering_angle = Angle(events.scattering_angle_rad, unit=u.rad)

        angular_distance, _ = self._relative_cds.to_relative(
            scattered_direction
        )

        arm = angular_distance - scattering_angle

        return arm.to(u.deg)

    def _select(
        self,
        events: ComptonDataSpaceInSCFrameEventDataInterface,
        early_stop: bool = True,
    ) -> Iterable[bool]:
        """
        Return Boolean event-selection mask.

        Parameters
        ----------
        events : ComptonDataSpaceInSCFrameEventDataInterface
            Compton event data in spacecraft frame.

        early_stop : bool, optional
            Kept for compatibility with EventSelectorInterface. This selector
            does not use early stopping because ARM selection is independent
            for each event.

        Returns
        -------
        numpy.ndarray
            Boolean mask. True means the event is accepted. False means the
            event is rejected.
        """

        arm = self._compute_arm(events)

        return np.abs(arm.to_value(u.deg)) <= self._arm_cut.to_value(u.deg)


# In[ ]:




