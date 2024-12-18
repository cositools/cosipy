import numpy as np
import pytest
from astropy.coordinates import SkyCoord, Angle
import astropy.units as u
from cosipy.polarization import OrthographicConvention, StereographicConvention
from scoords import SpacecraftFrame

# Define common test data
source_direction = SkyCoord(ra=-90*u.deg, dec=0*u.deg, frame='icrs') # -y

def test_orthographic_projection_default():

    ortho_convention = OrthographicConvention()
    px, py = ortho_convention.get_basis(source_direction)

    assert np.isclose(px.separation(SkyCoord(ra = 0*u.deg, dec = 90*u.deg)), 0)
    assert np.isclose(py.separation(SkyCoord(ra = 0*u.deg, dec = 0*u.deg)), 0)

def test_stereographic_projection_default():
    
    stereo_convention = StereographicConvention()
    px, py = stereo_convention.get_basis(source_direction)
    
    assert np.isclose(px.separation(SkyCoord(lon = 0*u.deg, lat = 0*u.deg,
                                             frame = SpacecraftFrame())), 0)
    assert np.isclose(py.separation(SkyCoord(lon = 0*u.deg, lat = -90*u.deg,
                                             frame = SpacecraftFrame())), 0)

