from cosipy.coordinates import Attitude

from scipy.spatial.transform import Rotation

from astropy.coordinates import SkyCoord, CartesianRepresentation
import astropy.units as u

from pytest import approx

import numpy as np
from numpy import array_equal as arr_eq

def test_init():

    # Default to 'icrs' frame
    a = Attitude(Rotation.identity())

    assert a.frame == 'icrs'

    assert arr_eq(a.rot.as_matrix(), [[1,0,0],[0,1,0],[0,0,1]])    

def test_from_as_axes():

    a = Attitude.from_axes(x = SkyCoord(ra = 10*u.deg, dec = 0*u.deg),
                           y = SkyCoord(ra = 100*u.deg, dec = 0*u.deg),
                           frame = 'icrs')

    a_rotvec = a.as_rotvec()
    
    assert a_rotvec[0].to_value('deg') == approx(0)
    assert a_rotvec[1].to_value('deg') == approx(0)
    assert a_rotvec[2].to_value('deg') == approx(10)

    x,y,z = a.as_axes()

    assert x.ra.deg == approx(10)
    assert x.dec.deg == approx(0)
    assert y.ra.deg == approx(100)
    assert y.dec.deg == approx(0)
    # z.ra undetermined
    assert z.dec.deg == approx(90)
    

def test_transform_to():

    a = Attitude(Rotation.identity(), frame = 'galactic')

    x,y,z = a.transform_to('icrs').as_axes()

    
    
