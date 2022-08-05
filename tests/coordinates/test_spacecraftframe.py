from cosipy.coordinates import Attitude, SpacecraftFrame

from astropy.coordinates import SkyCoord
import astropy.units as u

import pytest
from pytest import approx

def test_transform_to():

    # Following the same conventions as GBM. Tested against
    # https://fermi.gsfc.nasa.gov/ssc/data/analysis/gbm/gbm_data_tools/gdt-docs/api/api-utils.html?highlight=quaternion#gbm.coords.spacecraft_to_radec

    # From SC to RA/Dec
    attitude = Attitude.from_quat([0.08, 0.17, 0.25, 0.94])

    frame = SpacecraftFrame(attitude = attitude)
    
    c = SkyCoord(lon = 40*u.deg, lat = 80*u.deg, frame = frame)

    ci = c.transform_to('icrs')

    assert ci.ra.deg == approx(13.1349599)
    assert ci.dec.deg == approx(64.53362198)

    # From RA/Dec to SC
    cs = ci.transform_to(frame)

    assert cs.lon.deg == approx(c.lon.deg)
    assert cs.lat.deg == approx(c.lat.deg)

def test_exeptions():

    # Frame without attitude
    c = SkyCoord(lon = 40*u.deg, lat = 80*u.deg, frame = SpacecraftFrame())
    
    with pytest.raises(RuntimeError):

        c.transform_to('icrs')

        
    c = SkyCoord(ra = 40*u.deg, dec = 80*u.deg, frame = 'icrs')
    
    with pytest.raises(RuntimeError):

        c.transform_to(SpacecraftFrame())
