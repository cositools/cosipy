import logging
logger = logging.getLogger(__name__)

import itertools
from typing import Union, Iterable

import numpy as np
from astropy.coordinates import SkyCoord, Galactic, GCRS, CartesianRepresentation

from cosipy.interfaces import TimeTagEmCDSEventInSCFrameInterface
from cosipy.interfaces.event_selection import EventSelectorInterface
from cosipy.util.iterables import itertools_batched, asarray
from cosipy.spacecraftfile import SpacecraftHistory

class EHSelector(EventSelectorInterface):
    def __init__(self, ori:SpacecraftHistory, cutvalue:float = None, batch_size:int = None):
        """
        Assumes events are time-ordered
        
        Selects events based on Earth Horizon cut
        
        Parameters
        ----------
        ori: SpacecraftHistory
            SpacecraftHistory object
        cutvalue: float
            Value of the earth horizon cut to apply [0,1].
        batch_size: int, default None
            Number of events to process at once
            If None, all values are processed in a single batch.
            This parameter only affects iteration when the expectation density
            is provided as an iterator that is not a numpy array. If it is already
            an array batching is not applied.
        """
        if cutvalue is None:
            logger.error("You must give a value for the EH cut.")
            raise ValueError

        if cutvalue > 1 or cutvalue < 0 :
            logger.error("The cut value must be between 0 and 1.")
            raise ValueError
    
    def _select(self, events:TimeTagEmCDSEventInSCFrameInterface, early_stop:bool = True) -> Iterable[bool]:
        
        def process_chunk(jd1: np.ndarray, phi: np.ndarray):

            costheta, theta_max = self.Angle_PsiChi_Ez(jd1)
            fsky = self.calculate_sky_fraction(costheta, phi, theta_max)

            result = ( fsky >= self.cutvalue)

            # Stop further loading of event
            stop = early_stop

            return result, stop

        def process_in_chunks(events):

            for chunk in itertools_batched(events, self._batch_size):

                jd1 = []
                phi = []

                for event in chunk:
                    jd1.append(event.jd1)
                    phi.append(event.events.scattering_angle_rad)

                # Cache in memory
                jd1 = asarray(jd1, dtype=np.float64, force_dtype=False)
                phi = asarray(phi, dtype=np.float64, force_dtype=False)

                result, stop = process_chunk(jd1, phi)

                yield from result

                if stop:
                    return

        if (self._batch_size is None) or (isinstance(events.jd1, np.ndarray) and isinstance(events.scattering_angle_rad, np.ndarray)):
            results, _ = process_chunk(events.jd1, events.scattering_angle_rad)
            return results
        else:
            return process_in_chunks(events)

       
    def Angle_PsiChi_Ez(self, jd1):
        """
        Calculates the angle between PsiChi and the Earth zenith for each event.
    
        Parameters:
        -----------
        events : TimeTagEventDataInterface
        
        Returns:
        --------
        costheta : array-like
            Angle in rad.
        thetamax : array-like
            Max angle between Earth zenith and Earth horizon
        """

        #Get orientation info
        self.ori.cache_earth_occ = True
    
        #compute once the Earth occ for a random source in order to cache min_angle_cos and ez_cart
        randomsource = SkyCoord(0*u.deg, 0*u.deg, frame ="galactic")
        self.ori.get_earth_occ(randomsource)
    
    
        # Ensure the events time is sorted and convert it to a NumPy array
        ori_times = self.ori.obstime.value
        event_times = jd1
    
        # Find the closest index ahead of each event time
        idx = np.searchsorted(ori_times, event_times, side='left')
        idx = np.clip(idx, 1, len(ori_times) - 1)
        left_idx = idx - 1
        right_idx = idx
    
        # Choose whichever side is closer in time
        closer_than_right = np.abs(event_times - ori_times[left_idx]) < np.abs(event_times - ori_times[right_idx])
        nearest_idx = np.where(closer_than_right, left_idx, right_idx)
    
        # Get the Earth zenith and max angle for each event
        # ori._ez_cart has a shape of (3, N). Slicing the columns via [:, nearest_idx] 
        # and transposing (.T) instantly yields a clean (N, 3) vector array.
        earth_zenith_vector = self.ori._ez_cart[:, nearest_idx].T 
        max_ang = self.ori._min_angle_cos[nearest_idx]
    
        # Standard spherical to cartesian unit vectors (Galactic Frame)
        x_gal = np.cos(np.radians(self.cosi_dataset["Psi galactic"])) * np.cos(np.radians(self.cosi_dataset["Chi galactic"]))
        y_gal = np.cos(np.radians(self.cosi_dataset["Psi galactic"])) * np.sin(np.radians(self.cosi_dataset["Chi galactic"]))
        z_gal = np.sin(np.radians(self.cosi_dataset["Psi galactic"]))
        source_vector_gal = np.vstack([x_gal, y_gal, z_gal]).T 
    
        # Compute the 3x3 rotation matrix from Galactic to GCRS (ICRS/Equatorial aligned)
        # This is a trick to not use directly astropy.transform_to for every event
        # because this would be very slow for millions of events
        cart_axes = CartesianRepresentation(x=[1, 0, 0], y=[0, 1, 0], z=[0, 0, 1])
        dummy_gal = Galactic(cart_axes)
        axes_gcrs = dummy_gal.transform_to(GCRS(obstime="J2000"))
    
        R = np.vstack([
            axes_gcrs.cartesian.x.value,
            axes_gcrs.cartesian.y.value,
            axes_gcrs.cartesian.z.value
            ])
    
        # Rotate Galactic vectors to GCRS unit vectors
        source_vector_gcrs = np.dot(source_vector_gal, R.T) 
    
        # Einstein Summation: Dot product of two unit vectors in GCRS frame
        costheta = np.einsum('ij,ij->i', source_vector_gcrs, earth_zenith_vector)
    
        # Safety clip for floating-point precision edge cases
        costheta = np.clip(costheta, -1.0, 1.0)
        theta_max = np.clip(max_ang, -1.0, 1.0)

    
        return np.arccos(costheta), np.arccos(theta_max)


    @classmethod
    def calculate_sky_fraction(theta_psichi, theta_phi, theta_max):
        """
        Calculates the fraction of the Compton cone that is above the Earth horizon
        using the Spherical Law of Cosines.

        Mathematical Logic:
        ------------------
        1. Define a spherical triangle on the sky with vertices at:
           - Z: Earth Zenith
           - C: Center of the Compton scattering circle (cone axis)
           - I: Intersection point where the Compton circle crosses the horizon boundary

        2. Define the angular lengths of the sides of this spherical triangle:
           - Side connecting Z and I = theta_max (Max angle from Zenith to horizon)
           - Side connecting Z and C = theta_psichi (Angle between cone axis and Zenith)
           - Side connecting C and I = theta_phi (Compton scattering cone half-angle)

        3. Solve for the interior angle at vertex C (alpha_cut) using the Spherical 
           Law of Cosines for sides:
       
           cos(theta_max) = cos(theta_psichi)*cos(theta_phi) + sin(theta_psichi)*sin(theta_phi)*cos(alpha_cut)

        4. Isolate alpha_cut, which represents the half-angle of the Compton circle arc
           that resides safely in the unoccluded sky (above the horizon):
       
           cos_alpha_cut = (cos(theta_max) - cos(theta_psichi)*cos(theta_phi)) / (sin(theta_psichi)*sin(theta_phi))
           alpha_cut = arccos(cos_alpha_cut)

        5. Convert the unoccluded half-angle into the total sky fraction (f_sky):
           - The total unoccluded arc fraction of the circle is (2 * alpha_cut) / (2 * pi)
           - f_sky = alpha_cut / pi

        Boundary Conditions & Clipping:
        -------------------------------
        - If the cone is completely above the horizon: cos_alpha_cut scales below -1.0. 
          Clipping to -1.0 yields alpha_cut = pi, meaning f_sky = 1.0 (100% sky exposure).
        - If the cone is completely below the horizon: cos_alpha_cut scales above 1.0. 
          Clipping to 1.0 yields alpha_cut = 0.0, meaning f_sky = 0.0 (100% occulted).
    
        Parameters:
        -----------
        theta_psichi : float or array-like
            Angle between the scattered direction (cone axis) and Earth zenith (radians).
        theta_phi : float or array-like
            Compton scattering angle / cone half-opening angle (radians).
        theta_max : float
            Maximum angle from Earth zenith to the horizon boundary (radians).
        
        Returns:
        --------
        f_sky : float or array-like
            Fraction of the cone in the sky [0.0, 1.0].
        """
        # Avoid division by zero for perfectly on-axis events
        sin_term = np.sin(theta_psichi) * np.sin(theta_phi)
    
        # Handle the edge case where sin_term is 0 (e.g., theta_psichi=0 or theta_phi=0)
        # If sin_term is 0, the cone is either entirely in the sky or entirely in the mud.
        safe_sin_term = np.where(sin_term == 0, 1e-9, sin_term)
    
        # Calculate cos(alpha_cut) using the spherical law of cosines
        cos_alpha_cut = (np.cos(theta_max) - np.cos(theta_psichi) * np.cos(theta_phi)) / safe_sin_term
    
        # Clip values to handle cases where the cone is entirely above or below the horizon
        cos_alpha_cut = np.clip(cos_alpha_cut, -1.0, 1.0)
    
        # Calculate the angle and the resulting sky fraction
        alpha_cut = np.arccos(cos_alpha_cut)
        f_sky = alpha_cut / np.pi
    
        # Clean up the perfectly on-axis edge cases manually if needed
        # (If centered on zenith and within horizon, f_sky should be 1.0)
        if np.isscalar(theta_psichi):
            if sin_term == 0:
                f_sky = 1.0 if (theta_psichi + theta_phi) <= theta_max else 0.0
        else:
            f_sky = np.where(sin_term == 0, np.where((theta_psichi + theta_phi) <= theta_max, 1.0, 0.0), f_sky)
        
        return f_sky
