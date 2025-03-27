from cosipy.polarization import OrthographicConvention, StereographicConvention, PolarizationAngle

import numpy as np
import pytest
from astropy.coordinates import SkyCoord, Angle
import astropy.units as u
from cosipy.polarization import OrthographicConvention, StereographicConvention
from scoords import SpacecraftFrame, Attitude
from cosipy.polarization.conventions import MEGAlibRelativeX, IAUPolarizationConvention

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
    
def test_from_scattering_direction():

    psichi = SkyCoord(lat=np.pi/8, lon=np.pi/6, unit=u.rad, frame=SpacecraftFrame(attitude = Attitude.identity()))
    pa = PolarizationAngle.from_scattering_direction(psichi, source_direction.transform_to(SpacecraftFrame(attitude = Attitude.identity())), MEGAlibRelativeX(attitude = Attitude.identity()))

    assert np.isclose(pa.angle.deg, -134.186)

    pa2 = PolarizationAngle.from_scattering_direction(psichi.transform_to('galactic'), source_direction.transform_to('galactic'), IAUPolarizationConvention())

    assert np.isclose(pa2.angle.deg, 980.349)

    pa3 = PolarizationAngle.from_scattering_direction(psichi.transform_to('icrs'), source_direction, IAUPolarizationConvention())

    assert np.isclose(pa3.angle.deg, 80.349)
