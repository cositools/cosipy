import numpy as np
from astropy.coordinates import SkyCoord, Angle
import astropy.units as u
from scoords import SpacecraftFrame

from .conventions import PolarizationConvention

class PolarizationAngle:

    def __init__(self, angle, source,
                 convention = 'iau',
                 *args, **kwargs):
        """
        Defines a polarization angle in the context of a source direction and
        polarization angle convention.

        Parameters:
        angle : :py:class:`astropy.coordinates.Angle
            Polarization angle
        source : :py:class:`astropy.coordinates.SkyCoord`
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

        if source.size > 1:
            raise ValueError("Only single source location is allowed")
        elif source.ndim > 0:
            source = np.ravel(source)[0]

        self._source = source

    def __repr__(self):
        return f"<PolarizationAngle({self._angle.degree} deg at {self._source} using convention {self._convention})>"

    @property
    def angle(self):
        return self._angle

    @property
    def convention(self):
        return self._convention

    @property
    def source(self):
        return self._source

    @property
    def vector(self):
        """
        Direction of the electric field vector
        """

        # Get the projection vectors for the source direction in the
        # current convention
        px, py = self._convention.get_basis(self._source)

        px = px.cartesian.xyz
        py = py.cartesian.xyz

        # Calculate the cosine and sine of the polarization angle
        cos_pa = np.cos(self._angle.rad)
        sin_pa = np.sin(self._angle.rad)

        # Calculate the polarization vector
        pol_vec = np.outer(px, cos_pa) + np.outer(py, sin_pa)

        v = SkyCoord(*pol_vec,
                     representation_type = 'cartesian',
                     frame = self._convention.frame)

        # do not return a vector of pol_vecs for a scalar Angle
        if self._angle.ndim == 0:
            v = v[0]

        return v

    def transform_to(self, convention, *args, **kwargs):

        # Standarize convention
        convention = PolarizationConvention.get_convention(convention, *args, **kwargs)

        # Get the projection vectors for the source direction in the new convention
        px, py = convention.get_basis(self._source)

        px = px.cartesian.xyz
        py = py.cartesian.xyz

        # Calculate the polarization vector in the current convention
        pol_vec = self.vector.transform_to(convention.frame).cartesian.xyz

        # Compute the dot products for the transformation
        a = np.dot(pol_vec.T, px)
        b = np.dot(pol_vec.T, py)

        # Calculate the new polarization angle in the new convention
        pa = Angle(np.arctan2(b, a), unit=u.rad)

        # Normalize the angle to be between 0 and pi
        pa = np.where(pa < 0, pa + Angle(np.pi, unit=u.rad), pa)

        return PolarizationAngle(pa,
                                 self._source,
                                 convention = convention)

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

        source_coord = source_coord.transform_to(convention.frame)
        psichi = psichi.transform_to(convention.frame)

        reference_coord = convention.get_basis(source_coord)[0]

        source_vector_cartesian = source_coord.cartesian.xyz.value
        reference_vector_cartesian = reference_coord.cartesian.xyz.value
        scattered_photon_vector = psichi.cartesian.xyz.value.T

        # Project scattered photon vector onto plane perpendicular to
        # source direction
        d = np.dot(scattered_photon_vector, source_vector_cartesian) / np.dot(source_vector_cartesian, source_vector_cartesian)
        projection = scattered_photon_vector - np.outer(d, source_vector_cartesian)

        # Calculate angle between scattered photon vector & reference
        # vector on plane perpendicular to source direction
        cross_product = np.cross(projection, reference_vector_cartesian)
        sign = np.where(np.dot(cross_product, source_vector_cartesian) < 0, -1, 1)

        normalization = np.linalg.norm(projection, axis=-1) * np.linalg.norm(reference_vector_cartesian)

        dot_product = np.dot(projection, reference_vector_cartesian) / normalization

        dot_product = np.where((dot_product < -1.) & np.isclose(dot_product, -1.), -1., dot_product)
        dot_product = np.where((dot_product >  1.) & np.isclose(dot_product,  1.),  1., dot_product)

        angle = Angle(sign * np.arccos(dot_product), unit=u.rad)

        azimuthal_scattering_angle = cls(angle, source_coord, convention=convention)

        return azimuthal_scattering_angle
