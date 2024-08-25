import numpy as np
from astropy.coordinates import SkyCoord, Angle
import astropy.units as u

from .conventions import PolarizationConvention

class PolarizationAngle:

    def __init__(self, angle, skycoord ,
                 convention = 'iau',
                 *args, **kwargs):
    """
    Defines a polarization angle in the context of a source direction and
    polarization angle convention.

    Parameters:
    angle : :py:class:`astropy.coordinates.Angle
        Polarization angle
    skycoord : :py:class:`astropy.coordinates.SkyCoord`
        Source direction
    convention : PolarizationConvention
        Convention the defined the polarization basis and direction in 
        the polarization plane (for which the source direction is normal)
    *args, **kwargs
        Passed to convention class.
    """
        
        # Ensure pa is an Angle object
        self._angle = Angle(angle)

        self._convention = PolarizationConvention.get_convention(convention,
                                                                 *args, **kwargs)

        self._skycoord = skycoord

    @property
    def pa(self):
        return self._angle

    @property
    def convention(self):
        return self._convention

    @property
    def skycoord(self):
        return self._skycoord

    def transform_to(self, convention):

        convention1 = self.convention
        convention2 = convention
        pa = self.pa
        source_direction = self.skycoord
        
        # Transform the polarization angle from one convention to another
        transformed_pa = convention1.transform(pa, convention2, source_direction)

        # Normalize the angle to be between 0 and pi
        if transformed_pa < 0:
            transformed_pa += Angle(np.pi, unit=u.rad)

        return PolarizationConvention(transformed_pa,
                                      self,skycoord,
                                      convention = convention2)

        
