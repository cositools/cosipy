import logging

logger = logging.getLogger(__name__)

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord

from cosipy.interfaces.event_selection import EventSelectorInterface


class ARMSelector(EventSelectorInterface):
    """
    ARM event selector.

    Selects Compton events satisfying:

        |ARM| <= arm_cut

    where:

        ARM = angular_separation(source_direction, event_axis) - Phi

    Current validated FITS-table implementation
    -------------------------------------------
    This implementation uses the event-axis columns:

        - Chi galactic
        - Psi galactic
        - Phi

    Units assumed from the validated FITS files:

        - Chi galactic: radians
        - Psi galactic: radians
        - Phi: radians

    The source position is provided as an astropy SkyCoord and is internally
    converted to Galactic coordinates. The ARM values are returned in degrees.

    Parameters
    ----------
    spacecraft_history
        Spacecraft history object. Stored for compatibility with the requested
        ARMSelector interface. In this validated FITS-column implementation,
        the event-axis direction is already available in Galactic coordinates
        through Chi/Psi columns.

    target_coord : astropy.coordinates.SkyCoord
        Source position.

    arm_cut : astropy.units.Quantity
        Symmetric ARM half-width, for example 13.0 * u.deg.

    chi_col : str
        FITS/event-table column containing the Galactic longitude-like event
        axis angle. Validated value: "Chi galactic".

    psi_col : str
        FITS/event-table column containing the Galactic latitude-like event
        axis angle. Validated value: "Psi galactic".

    phi_col : str
        FITS/event-table column containing the Compton scatter angle Phi.
        Validated value: "Phi".

    batch_size : int or None
        Optional batch size. Stored for future extension.
    """

    def __init__(
        self,
        spacecraft_history,
        target_coord: SkyCoord,
        arm_cut: u.Quantity,
        chi_col: str = "Chi galactic",
        psi_col: str = "Psi galactic",
        phi_col: str = "Phi",
        batch_size: int = None,
    ):
        self._spacecraft_history = spacecraft_history
        self._target_coord = target_coord.galactic
        self._arm_cut = arm_cut.to(u.deg)

        self._chi_col = chi_col
        self._psi_col = psi_col
        self._phi_col = phi_col
        self._batch_size = batch_size

        # Precompute source quantities in radians for fast vectorized selection.
        self._src_l_rad = self._target_coord.l.to_value(u.rad)
        self._src_b_rad = self._target_coord.b.to_value(u.rad)

        self._sin_src_b = np.sin(self._src_b_rad)
        self._cos_src_b = np.cos(self._src_b_rad)

    @property
    def arm_cut(self):
        """
        ARM selection half-width.

        Returns
        -------
        astropy.units.Quantity
            ARM cut in degrees.
        """
        return self._arm_cut

    @property
    def target_coord(self):
        """
        Target/source coordinate.

        Returns
        -------
        astropy.coordinates.SkyCoord
            Source coordinate in the Galactic frame.
        """
        return self._target_coord

    @property
    def spacecraft_history(self):
        """
        Spacecraft history input.

        Returns
        -------
        object
            Stored spacecraft history object.
        """
        return self._spacecraft_history

    @property
    def chi_col(self):
        """
        Name of the Chi Galactic event-axis column.

        Returns
        -------
        str
            Chi column name.
        """
        return self._chi_col

    @property
    def psi_col(self):
        """
        Name of the Psi Galactic event-axis column.

        Returns
        -------
        str
            Psi column name.
        """
        return self._psi_col

    @property
    def phi_col(self):
        """
        Name of the Phi column.

        Returns
        -------
        str
            Phi column name.
        """
        return self._phi_col

    def _compute_arm(self, events):
        """
        Compute ARM values.

        Parameters
        ----------
        events : table-like
            Event table containing:
                self._chi_col
                self._psi_col
                self._phi_col

        Returns
        -------
        astropy.units.Quantity
            ARM values in degrees.
        """

        chi_rad = np.asarray(events[self._chi_col], dtype=np.float64)
        psi_rad = np.asarray(events[self._psi_col], dtype=np.float64)
        phi_rad = np.asarray(events[self._phi_col], dtype=np.float64)

        if chi_rad.shape != psi_rad.shape:
            raise ValueError(
                "Chi and Psi columns must have the same shape, "
                f"got {chi_rad.shape} and {psi_rad.shape}."
            )

        if chi_rad.shape != phi_rad.shape:
            raise ValueError(
                "Chi/Psi and Phi columns must have the same shape, "
                f"got {chi_rad.shape} and {phi_rad.shape}."
            )

        cos_sep = (
            self._sin_src_b * np.sin(psi_rad)
            + self._cos_src_b
            * np.cos(psi_rad)
            * np.cos(chi_rad - self._src_l_rad)
        )

        cos_sep = np.clip(cos_sep, -1.0, 1.0)

        sep_rad = np.arccos(cos_sep)
        arm_rad = sep_rad - phi_rad

        return np.rad2deg(arm_rad) * u.deg

    def _select(self, events):
        """
        Return boolean event-selection mask.

        Parameters
        ----------
        events : table-like
            Event table containing:
                self._chi_col
                self._psi_col
                self._phi_col

        Returns
        -------
        numpy.ndarray
            Boolean mask. True for events inside the ARM cut and False for
            events outside the ARM cut.
        """

        arm = self._compute_arm(events)

        return np.abs(arm.to_value(u.deg)) <= self._arm_cut.to_value(u.deg)
