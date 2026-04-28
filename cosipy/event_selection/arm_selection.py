import logging
logger = logging.getLogger(__name__)

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord

from cosipy.interfaces.event_selection import EventSelectorInterface


class ARMSelector(EventSelectorInterface):
    """
    ARM event selector.

    Selects events satisfying:

        |ARM| <= arm_cut

    with:

        ARM = angular_separation(source_direction, event_axis) - Phi

    Current validated FITS-table implementation:
        - event axis: Xpointings (glon,glat)
        - pointings unit: radians
        - Phi unit: radians
        - source coordinate: SkyCoord

    Parameters
    ----------
    spacecraft_history
        Spacecraft history object. Stored for compatibility with the requested
        ARMSelector interface. In this validated FITS-table implementation,
        event pointings are already available in Galactic coordinates.

    target_coord : astropy.coordinates.SkyCoord
        Source position.

    arm_cut : astropy.units.Quantity
        Symmetric ARM half-width, for example 13.0 * u.deg.

    pointing_col : str
        FITS/event-table column containing event pointing directions.
        Validated value: "Xpointings (glon,glat)"

    phi_col : str
        FITS/event-table column containing Phi.
        Validated value: "Phi"

    batch_size : int or None
        Optional batch size. Stored for future extension.
    """

    def __init__(
        self,
        spacecraft_history,
        target_coord: SkyCoord,
        arm_cut: u.Quantity,
        pointing_col: str = "Xpointings (glon,glat)",
        phi_col: str = "Phi",
        batch_size: int = None,
    ):
        self._spacecraft_history = spacecraft_history
        self._target_coord = target_coord.galactic
        self._arm_cut = arm_cut.to(u.deg)

        self._pointing_col = pointing_col
        self._phi_col = phi_col
        self._batch_size = batch_size

        # Precompute source quantities in radians for fast vectorized selection.
        self._src_l_rad = self._target_coord.l.to_value(u.rad)
        self._src_b_rad = self._target_coord.b.to_value(u.rad)

        self._sin_src_b = np.sin(self._src_b_rad)
        self._cos_src_b = np.cos(self._src_b_rad)

    @property
    def arm_cut(self):
        return self._arm_cut

    @property
    def target_coord(self):
        return self._target_coord

    @property
    def spacecraft_history(self):
        return self._spacecraft_history

    def _compute_arm(self, events):
        """
        Compute ARM values.

        Parameters
        ----------
        events : table-like
            Event table containing:
                self._pointing_col
                self._phi_col

        Returns
        -------
        arm : astropy.units.Quantity
            ARM values in degrees.
        """

        pointing_rad = np.asarray(events[self._pointing_col])
        phi_rad = np.asarray(events[self._phi_col])

        if pointing_rad.ndim != 2 or pointing_rad.shape[1] != 2:
            raise ValueError(
                f"Expected pointing column with shape (N, 2), "
                f"got {pointing_rad.shape}"
            )

        event_l_rad = pointing_rad[:, 0].astype(np.float64)
        event_b_rad = pointing_rad[:, 1].astype(np.float64)

        cos_sep = (
            self._sin_src_b * np.sin(event_b_rad)
            + self._cos_src_b
            * np.cos(event_b_rad)
            * np.cos(event_l_rad - self._src_l_rad)
        )

        cos_sep = np.clip(cos_sep, -1.0, 1.0)

        sep_rad = np.arccos(cos_sep)
        arm_rad = sep_rad - phi_rad.astype(np.float64)

        return np.rad2deg(arm_rad) * u.deg

    def _select(self, events):
        """
        Return boolean event-selection mask.

        Returns
        -------
        mask : numpy.ndarray
            True for events inside the ARM cut.
            False for events outside the ARM cut.
        """

        arm = self._compute_arm(events)

        return np.abs(arm.to_value(u.deg)) <= self._arm_cut.to_value(u.deg)
