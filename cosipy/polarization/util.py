import numpy as np
from astropy.coordinates import SkyCoord, Angle
import astropy.units as u

# Base class for polarization conventions
class PolarizationConvention:
    def __init__(self, ref_vector: SkyCoord = None):
        # Set the reference vector, defaulting to celestial north if not provided
        if ref_vector is None:
            self.ref_vector = SkyCoord(ra=0 * u.deg, dec=90 * u.deg, frame="icrs")
        else:
            self.ref_vector = ref_vector

    def transform(self, pa_1, convention2, source_direction: SkyCoord):
       # Ensure pa_1 is an Angle object
        pa_1 = Angle(pa_1)

        # Get the projection vectors for the source direction in the current convention
        (px1, py1) = self.project(source_direction)

        # Calculate the cosine and sine of the polarization angle
        cos_pa_1 = np.cos(pa_1.radian)
        sin_pa_1 = np.sin(pa_1.radian)

        # Calculate the polarization vector in the current convention
        pol_vec = px1 * cos_pa_1 + py1 * sin_pa_1

        # Get the projection vectors for the source direction in the new convention
        (px2, py2) = convention2.project(source_direction)

        # Compute the dot products for the transformation
        a = np.sum(pol_vec * px2, axis=0)
        b = np.sum(pol_vec * py2, axis=0)

        # Calculate the new polarization angle in the new convention
        pa_2 = Angle(np.arctan2(b, a), unit=u.rad)
        return pa_2


# Orthographic projection convention
class OrthographicConvention(PolarizationConvention):
    def project(self, source_direction: SkyCoord):
        # Extract Cartesian coordinates for the source direction and reference vector
        x, y, z = source_direction.cartesian.xyz
        
        ref_x, ref_y, ref_z = self.ref_vector.cartesian.xyz

        # Calculate the dot product between the source direction and the reference vector
        dot_product = x * ref_x + y * ref_y + z * ref_z

        # calculate the norm of the cource vector
        norm_source = np.linalg.norm([x, y, z])

        # Project the reference vector onto the plane perpendicular to the source direction
        px_x = ref_x - dot_product * x / (norm_source)**2
        px_y = ref_y - dot_product * y / (norm_source)**2
        px_z = ref_z - dot_product * z / (norm_source)**2

        # Combine the components into the projection vector px
        px = np.array([px_x, px_y, px_z])

        # Normalize the projection vector
        norm = np.linalg.norm(px, axis=0)
        px /= norm

        # Calculate the perpendicular vector py using the cross product
        py = np.cross([x, y, z], px, axis=0)

        return px, py

# Stereographic projection convention
class StereographicConvention(PolarizationConvention):
    def project(self, source_direction: SkyCoord):
        # Extract Cartesian coordinates for the source direction
        x, y, z = source_direction.cartesian.xyz

        # Calculate the projection of the reference vector in stereographic coordinates
        px_x = 1 - (x**2 - y**2) / (z + 1) ** 2
        px_y = -2 * x * y / (z + 1) ** 2
        px_z = -2 * x / (z + 1)

        # Combine the components into the projection vector px
        px = np.array([px_x, px_y, px_z])

        # Normalize the projection vector
        norm = np.linalg.norm(px, axis=0)
        px /= norm

        # Calculate the perpendicular vector py using the cross product
        py = np.cross([x, y, z], px, axis=0)
        return px, py


def pa_transformation(pa, convention1, convention2, source_direction: SkyCoord):
    # Ensure pa is an Angle object
    pa = Angle(pa)
      # Transform the polarization angle from one convention to another
    transformed_pa = convention1.transform(pa, convention2, source_direction)
    
    # Normalize the angle to be between 0 and pi
    if transformed_pa < 0:
        transformed_pa += Angle(np.pi, unit=u.rad)
    
    return transformed_pa