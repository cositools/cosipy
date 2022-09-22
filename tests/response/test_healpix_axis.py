from cosipy.response import HealpixAxis

from astropy.coordinates import SkyCoord
import astropy.units as u

import numpy as np

def test_healpix_axis():

    axis = HealpixAxis(nside = 128,
                       coordsys = 'icrs')

    assert axis.find_bin(SkyCoord(ra = 1*u.deg, dec = 89.999*u.deg)) == 0

    pix, weights = axis.interp_weights(SkyCoord(ra = 0*u.deg, dec = 90*u.deg))

    assert np.array_equal(np.sort(pix), [0,1,2,3])
    
    assert np.allclose(weights, 0.25)
