import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from scoords import Attitude, SpacecraftFrame

# Generate theta and phi for meshgrid
def generate_meshgrid():
    theta = np.linspace(0, 0.5 * np.pi, 6)
    phi = np.linspace(-np.pi / 2, np.pi / 2, 13)
    theta, phi = np.meshgrid(theta, phi)

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    return x, y, z

# Orthographic projection function
def orthographic(x, y, z, ref_vector):
    ref_x, ref_y, ref_z = ref_vector
    dot_product = x * ref_x + y * ref_y + z * ref_z
    px_x = ref_x - dot_product * x
    px_y = ref_y - dot_product * y
    px_z = ref_z - dot_product * z

    # Normalize the projection vector
    norm = np.sqrt(px_x**2 + px_y**2 + px_z**2)
    px_x /= norm
    px_y /= norm
    px_z /= norm

    # Compute the perpendicular vector
    py_x, py_y, py_z = np.cross([x, y, z], [px_x, px_y, px_z], axis=0)

    return (px_x, px_y, px_z), (py_x, py_y, py_z)

# Stereographic projection function
def stereographic(x, y, z, ref_vector):
    ref_x, ref_y, ref_z = ref_vector
    norm_ref = np.sqrt(ref_x**2 + ref_y**2 + ref_z**2)
    ref_x /= norm_ref
    ref_y /= norm_ref
    ref_z /= norm_ref

    denom = (z + 1)
    px_x = 1 - (x**2 - y**2) / denom**2
    px_y = -2 * x * y / denom**2
    px_z = -2 * x / denom

    norm = np.sqrt(px_x**2 + px_y**2 + px_z**2)
    px_x /= norm
    px_y /= norm
    px_z /= norm

    py_x, py_y, py_z = np.cross([x, y, z], [px_x, px_y, px_z], axis=0)
    return (px_x, px_y, px_z), (py_x, py_y, py_z)

# Function to transform polarization angle between projections
def transform_pa(pa_1, convention1, convention2, ref_vector, x, y, z):
    (px1_x, px1_y, px1_z), (py1_x, py1_y, py1_z) = convention1(x, y, z, ref_vector)

    cos_pa_1 = np.cos(pa_1)
    sin_pa_1 = np.sin(pa_1)

    pol_vec_x = px1_x * cos_pa_1 + py1_x * sin_pa_1
    pol_vec_y = px1_y * cos_pa_1 + py1_y * sin_pa_1
    pol_vec_z = px1_z * cos_pa_1 + py1_z * sin_pa_1

    (px2_x, px2_y, px2_z), (py2_x, py2_y, py2_z) = convention2(x, y, z, ref_vector)

    a = pol_vec_x * px2_x + pol_vec_y * px2_y + pol_vec_z * px2_z
    b = pol_vec_x * py2_x + pol_vec_y * py2_y + pol_vec_z * py2_z

    pa_2 = np.arctan2(b, a)
    return pa_2

# Function to transform polarization angle from SC to celestial coordinates
def transform_pa_sc_to_celestial(pa_sc, attitude):
    sc_coord = SkyCoord(
        lon=pa_sc * u.rad, lat=0 * u.rad, frame=SpacecraftFrame(attitude=attitude)
    )

    # Transform to Celestial coordinates
    celestial_coord = sc_coord.transform_to('icrs')

    # Extract the polarization angle (PA) in celestial coordinates
    pa_celestial = celestial_coord.ra

    return pa_celestial.rad  # Return PA in radians

# Function to transform polarization angle from celestial to SC coordinates
def transform_pa_celestial_to_sc(pa_celestial, attitude):
    celestial_coord = SkyCoord(
        ra=pa_celestial * u.rad, dec=0 * u.rad, frame='icrs'
    )

    # Transform to SpacecraftFrame
    sc_coord = celestial_coord.transform_to(SpacecraftFrame(attitude=attitude))

    # Extract the polarization angle (PA) in spacecraft coordinates
    pa_sc = sc_coord.lon

    return pa_sc.rad  # Return PA in radians