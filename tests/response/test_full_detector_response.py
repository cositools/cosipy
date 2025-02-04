import numpy as np
from numpy import array_equal as arr_eq

from scoords import SpacecraftFrame
from astropy.coordinates import SkyCoord
import astropy.units as u

from histpy import Histogram, HealpixAxis, Axis

from cosipy import test_data
from cosipy.response import FullDetectorResponse

response_path = test_data.path / "test_full_detector_response_dense.h5"

def test_open():

    with FullDetectorResponse.open(response_path) as response:

        assert response.filename == response_path

        assert response.ndim == 5

        assert arr_eq(response.axes.labels,
                      ['NuLambda', 'Ei', 'Em', 'Phi', 'PsiChi'])

        assert response.unit.is_equivalent('m2')

def test_write_h5(tmp_path):
    """
    Tests storing a Histogram as an HDF5 response
    """

    tmp_rsp = tmp_path / 'tmp_rsp.h5'

    drm = Histogram([HealpixAxis(nside=1, label='NuLambda'),
                     Axis(np.geomspace(200, 5000, 11) * u.keV, label='Ei'),
                     Axis(np.geomspace(200, 5000, 11) * u.keV, label='Em'),
                     Axis(np.linspace(0, 180, 11) * u.deg, label='Phi'),
                     HealpixAxis(nside=1, label='PsiChi')],
                    contents=np.ones([12, 10, 10, 10, 12]),
                    unit='cm2')

    FullDetectorResponse._write_h5(drm, tmp_rsp)

    with FullDetectorResponse.open(tmp_rsp) as rsp:

        assert arr_eq(rsp[0].project("Ei").contents.to_value('cm2'), (10*10*12)*np.ones(10))


def test_get_item():

    with FullDetectorResponse.open(response_path) as response:

        drm = response[0]

        assert drm.ndim == 4

        assert arr_eq(drm.axes.labels,
                      ['Ei', 'Em', 'Phi', 'PsiChi'])

        assert drm.unit.is_equivalent('m2')

def test_get_interp_response():

    with FullDetectorResponse.open(response_path) as response:

        drm = response.get_interp_response(SkyCoord(lon = 0*u.deg,
                                                    lat = 0*u.deg,
                                                    frame = SpacecraftFrame()))

        assert drm.ndim == 4

        assert arr_eq(drm.axes.labels,
                      ['Ei', 'Em', 'Phi', 'PsiChi'])

        assert drm.unit.is_equivalent('m2')
        

    

        
        
    
