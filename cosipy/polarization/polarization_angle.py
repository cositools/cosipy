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

    def __repr__(self):
        return f"<PolarizationAngle({self._angle.degree} deg at {self._skycoord} using convention {self._convention})>"
        
    @property
    def angle(self):
        return self._angle

    @property
    def convention(self):
        return self._convention

    @property
    def skycoord(self):
        return self._skycoord

    @property
    def vector(self):
        """
        Direction of the electric field vector
        """

        # Get the projection vectors for the source direction in the current convention
        px, py = self.convention.get_basis(self.skycoord)

        px = px.cartesian.xyz
        py = py.cartesian.xyz

        # Calculate the cosine and sine of the polarization angle
        cos_pa = np.cos(self.angle.radian)
        sin_pa = np.sin(self.angle.radian)

        # Calculate the polarization vector
        pol_vec = px * cos_pa + py * sin_pa
                
        return SkyCoord(*pol_vec,
                        representation_type = 'cartesian',
                        frame = self.convention.frame)
    
    def transform_to(self, convention, *args, **kwargs):

        # Standarize convention 2
        convention2 = PolarizationConvention.get_convention(convention, *args, **kwargs)
        
        # Calculate the polarization vector in the current convention
        pol_vec = self.vector.transform_to(convention2.frame).cartesian.xyz

        # Get the projection vectors for the source direction in the new convention
        (px2, py2) = convention2.get_basis(self.skycoord)

        px2 = px2.cartesian.xyz
        py2 = py2.cartesian.xyz
        
        # Compute the dot products for the transformation
        a = np.sum(pol_vec * px2, axis=0)
        b = np.sum(pol_vec * py2, axis=0)

        # Calculate the new polarization angle in the new convention
        pa_2 = Angle(np.arctan2(b, a), unit=u.rad)

        # Normalize the angle to be between 0 and pi
        if pa_2 < 0:
            pa_2 += Angle(np.pi, unit=u.rad)

        return PolarizationAngle(pa_2,
                                 self.skycoord,
                                 convention = convention2)

        
