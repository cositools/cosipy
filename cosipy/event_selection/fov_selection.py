import logging
logger = logging.getLogger(__name__)

from typing import Iterable

import numpy as np
import astropy.units as u
from astropy.units import Quantity

from cosipy.interfaces.data_interface import ComptonDataSpaceInSCFrameEventDataInterface
from cosipy.interfaces.event_selection import EventSelectorInterface
from cosipy.util.iterables import asarray


class FOVSelector(EventSelectorInterface):

    event_data_type = ComptonDataSpaceInSCFrameEventDataInterface

    def __init__(self, cap_angle: Quantity = None, cutvalue: float = None, plotffov: bool = False):
        """
        Selects events based on how much of their Compton circle falls within a cap
        around the spacecraft zenith (the +z axis of the spacecraft frame).

        This is the spacecraft-frame analogue of the Earth horizon cut (see
        ``EHSelector`` in ``earth_horizon_selection.py``): there, the cap is the
        Earth-free region around the (time-dependent) Earth zenith; here, the cap is a
        fixed region of given angular radius around the spacecraft's own zenith. Unlike
        the Earth horizon cut, no orientation history or event time is needed, since the
        spacecraft zenith does not depend on time.

        Parameters
        ----------
        cap_angle: Quantity
            Angular radius of the field-of-view cap around the spacecraft zenith.
        cutvalue: float
            Minimum fraction of the Compton circle that must fall inside the cap, in [0,1].
        plotffov: bool, default False
            If you want to plot the distribution of the in-cap fraction.
        """
        if cap_angle is None:
            logger.error("You must give a value for the FOV cap angle.")
            raise ValueError

        if cutvalue is None:
            logger.error("You must give a value for the FOV cut.")
            raise ValueError

        if cutvalue > 1 or cutvalue < 0:
            logger.error("The cut value must be between 0 and 1.")
            raise ValueError

        self._cos_cap_angle = np.cos(cap_angle.to_value(u.rad))
        self._cutvalue = cutvalue
        self._plotffov = plotffov

    def _select(self, events: ComptonDataSpaceInSCFrameEventDataInterface,
                early_stop: bool = True) -> Iterable[bool]:

        phi = asarray(events.scattering_angle_rad, dtype=np.float64, force_dtype=False)
        lat_sc = asarray(events.scattered_lat_rad_sc, dtype=np.float64, force_dtype=False)

        # Angle between the psichi direction (the Compton circle's axis) and the
        # spacecraft zenith: colatitude from the +z axis, i.e. pi/2 - latitude.
        costheta_psichi = np.sin(lat_sc)

        ffov = self.calculate_cap_fraction(costheta_psichi, phi, self._cos_cap_angle)

        result = (ffov >= self._cutvalue)

        if self._plotffov:
            import matplotlib.pyplot as plt
            bins = np.arange(0, 1, 0.01)
            fig, ax = plt.subplots()
            count, _, _ = ax.hist(ffov, histtype="step", label="test", bins=bins, density=True)
            ax.set_xlabel("Fraction of the Compton circle within the FOV cap")
            ax.set_ylabel("a.u")
            ax.set_yscale("log")
            plt.show()
            print(list(count))

        return result

    @staticmethod
    def calculate_cap_fraction(costheta_psichi, theta_phi, costheta_cap):
        """
        Calculates the fraction of the Compton circle that is within a cap of a given
        angular radius around a fixed axis, using the Spherical Law of Cosines.

        Same geometry as ``EHSelector.calculate_sky_fraction``, applied to a fixed cap
        around the spacecraft zenith instead of the (time-dependent) Earth-free region
        around the Earth zenith -- see that function's docstring for the derivation.

        Parameters
        ----------
        costheta_psichi: float or array-like
            cos(angle) between the scattered direction (Compton circle axis) and the
            cap axis (the spacecraft zenith).
        theta_phi: float or array-like
            Compton scattering angle / circle half-opening angle (radians).
        costheta_cap: float
            cos(angular radius) of the cap around the cap axis.

        Returns
        -------
        f_fov: float or array-like
            Fraction of the Compton circle inside the cap, in [0, 1].
        """
        sin_term = np.sqrt(1 - costheta_psichi ** 2) * np.sin(theta_phi)

        safe_sin_term = np.where(sin_term == 0, 1e-9, sin_term)

        cos_alpha_cut = (costheta_cap - costheta_psichi * np.cos(theta_phi)) / safe_sin_term
        cos_alpha_cut = np.clip(cos_alpha_cut, -1.0, 1.0)

        alpha_cut = np.arccos(cos_alpha_cut)
        f_fov = alpha_cut / np.pi

        cos_sum = costheta_psichi * np.cos(theta_phi) - safe_sin_term * np.sin(theta_phi)
        condition = cos_sum >= costheta_cap

        if np.isscalar(costheta_psichi):
            if sin_term == 0:
                f_fov = 1.0 if condition else 0.0
        else:
            f_fov = np.where(sin_term == 0, np.where(condition, 1.0, 0.0), f_fov)

        return f_fov
