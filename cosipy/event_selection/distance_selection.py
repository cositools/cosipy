from typing import Iterable

import numpy as np
import astropy.units as u
from astropy.units import Quantity

from cosipy.interfaces.data_interface import TimeTagEmCDSDistanceEventDataInSCFrameInterface
from cosipy.interfaces.event_selection import EventSelectorInterface
from cosipy.util.iterables import asarray


class DistanceSelector(EventSelectorInterface):

    event_data_type = TimeTagEmCDSDistanceEventDataInSCFrameInterface

    def __init__(self, min_distance: Quantity = None, max_distance: Quantity = None):
        """
        Selects events whose distance between the first two hits falls
        within [min_distance, max_distance).

        Parameters
        ----------
        min_distance: Minimum distance [inclusive]. Default: 0 cm
        max_distance: Maximum distance [exclusive]. Default: infinity
        """

        if min_distance is None:
            min_distance = 0 * u.cm

        if max_distance is None:
            max_distance = np.inf * u.cm

        self._min_distance_cm = min_distance.to_value(u.cm)
        self._max_distance_cm = max_distance.to_value(u.cm)

    def _select(self, events: TimeTagEmCDSDistanceEventDataInSCFrameInterface,
                early_stop: bool = True) -> Iterable[bool]:

        distance_cm = asarray(events.distance_cm, dtype=np.float64, force_dtype=False)

        return (distance_cm >= self._min_distance_cm) & (distance_cm < self._max_distance_cm)
