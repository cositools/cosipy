import numpy as np
import pytest
from astropy.coordinates import SkyCoord, Angle
import astropy.units as u
from cosipy.polarization.util import OrthographicConvention, StereographicConvention, pa_transformation

# Define common test data
source_direction = SkyCoord(ra=83.637375*u.deg, dec=22.014472*u.deg, frame='icrs')  # Crab
default_ref_vector = SkyCoord(ra=0*u.deg, dec=90*u.deg, frame='icrs')  # Celestial north
custom_ref_vector = SkyCoord(ra=10*u.deg, dec=80*u.deg, frame='icrs')

def test_orthographic_projection_default():
    ortho_convention = OrthographicConvention()
    px, py = ortho_convention.project(source_direction)
    
    assert px is not None
    assert py is not None
    assert px.shape == (3,)
    assert py.shape == (3,)

def test_stereographic_projection_default():
    stereo_convention = StereographicConvention()
    px, py = stereo_convention.project(source_direction)
    
    assert px is not None
    assert py is not None
    assert px.shape == (3,)
    assert py.shape == (3,)

def test_pa_transformation():
    ortho_convention = OrthographicConvention(ref_vector=custom_ref_vector)
    stereo_convention = StereographicConvention(ref_vector=custom_ref_vector)
    
    pa_initial = Angle(45 * u.deg)  # Initial polarization angle
    pa_transformed = pa_transformation(pa_initial, ortho_convention, stereo_convention, source_direction)
    
    assert isinstance(pa_transformed, Angle)  # Ensure the result is an Angle object
    assert 0 <= pa_transformed.radian <= np.pi

def test_default_ref_vector():
    ortho_convention_default = OrthographicConvention()
    stereo_convention_default = StereographicConvention()

    pa_initial = Angle(36 * u.deg)  # Initial polarization angle
    pa_transformed = pa_transformation(pa_initial, ortho_convention_default, stereo_convention_default, source_direction)

    assert isinstance(pa_transformed, Angle)  # Ensure the result is an Angle object
    assert 0 <= pa_transformed.radian <= np.pi