from cosipy.polarization import OrthographicConvention, StereographicConvention, PolarizationAngle

import numpy as np
import pytest
from astropy.coordinates import SkyCoord, Angle
import astropy.units as u
from cosipy.polarization import OrthographicConvention, StereographicConvention
from scoords import SpacecraftFrame, Attitude

# Define common test data
source_direction = SkyCoord(ra = -36*u.deg, dec = 30*u.deg, frame = 'icrs')

def test_pa_transformation():

    pa = PolarizationAngle(20*u.deg, source_direction, convention = 'IAU')

    pa2 = pa.transform_to(StereographicConvention(attitude = Attitude.identity()))
    
    assert np.isclose(pa2.angle, 56*u.deg)

    pa2 = pa.transform_to('RelativeZ', attitude = Attitude.identity())

    assert np.isclose(pa2.angle, 110*u.deg)

    pa2 = pa.transform_to('RelativeZ', attitude = Attitude.from_rotvec([0,0,10]*u.deg))

    assert np.isclose(pa2.angle, 110*u.deg)

    pa2 = pa.transform_to('RelativeZ', attitude = Attitude.from_rotvec([30,0,0]*u.deg))

    assert np.isclose(pa2.angle, 143.852403963*u.deg)

    
    pa2 = pa.transform_to('RelativeX', attitude = Attitude.identity())
    
    assert np.isclose(pa2.angle, 165.46460289540*u.deg)
    
    
    
