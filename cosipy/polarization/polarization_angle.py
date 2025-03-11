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

    @classmethod
    def from_scattering_direction(cls, psichi, source_coord, convention):
        """
        Calculate the azimuthal scattering angle of a scattered photon.
        
        Parameters
        ----------
        psichi : astropy.coordinates.SkyCoord
            Scattered photon direction
        source_coord : astropy.coordinates.SkyCoord
            Source direction
        convention :
            cosipy.polarization.PolarizationConvention

        Returns
        -------
        azimuthal_scattering_angle : cosipy.polarization.PolarizationAngle
            Azimuthal scattering angle
        """

        if not psichi.frame.name == source_coord.frame.name:
            raise RuntimeError('psichi and source_coord must have the same frame')

        reference_coord = convention.get_basis(source_coord)[0]

        source_vector_cartesian = [source_coord.cartesian.x.value,
                                   source_coord.cartesian.y.value, 
                                   source_coord.cartesian.z.value]

        reference_vector_cartesian = [reference_coord.cartesian.x.value, 
                                      reference_coord.cartesian.y.value,
                                      reference_coord.cartesian.z.value]

        if psichi.frame.name == 'spacecraftframe':

            psi = (np.pi/2) - psichi.lat.rad
            chi = psichi.lon.rad

        elif psichi.frame.name == 'galactic':

            psi = (np.pi/2) - psichi.b.rad
            chi = psichi.l.rad

        elif psichi.frame.name == 'icrs':

            psi = (np.pi/2) - psichi.dec.rad
            chi = psichi.ra.rad

        else:
            raise RuntimeError('Unsupported frame "' + psichi.frame.name + '"')

        # Convert scattered photon vector from spherical to Cartesian coordinates
        scattered_photon_vector = [np.sin(psi) * np.cos(chi), np.sin(psi) * np.sin(chi), np.cos(psi)]

        # Project scattered photon vector onto plane perpendicular to source direction
        d = np.dot(scattered_photon_vector, source_vector_cartesian) / np.dot(source_vector_cartesian, source_vector_cartesian)
        projection = [scattered_photon_vector[0] - (d * source_vector_cartesian[0]), 
                      scattered_photon_vector[1] - (d * source_vector_cartesian[1]), 
                      scattered_photon_vector[2] - (d * source_vector_cartesian[2])]

        # Calculate angle between scattered photon vector & reference vector on plane perpendicular to source direction
        cross_product = np.cross(projection, reference_vector_cartesian)
        if np.dot(source_vector_cartesian, cross_product) < 0:
            sign = -1
        else:
            sign = 1
        normalization = np.sqrt(np.dot(projection, projection)) * np.sqrt(np.dot(reference_vector_cartesian, reference_vector_cartesian))
    
        angle = Angle(sign * np.arccos(np.dot(projection, reference_vector_cartesian) / normalization), unit=u.rad)

        azimuthal_scattering_angle = cls(angle, source_coord, convention=convention)

        return azimuthal_scattering_angle


        
