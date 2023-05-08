import numpy as np
from numpy import array_equal as arr_eq

from scoords import SpacecraftFrame
from astropy.coordinates import SkyCoord
import astropy.units as u

from cosipy import test_data
from cosipy.response import FullDetectorResponse

response_path = test_data.path / "test_full_detector_response.h5"

def test_open():

    with FullDetectorResponse.open(response_path) as response:

        assert response.filename == response_path

        assert response.ndim == 7

        assert arr_eq(response.axes.labels,
                      ['NuLambda', 'Ei', 'Em', 'Phi', 'PsiChi', 'SigmaTau', 'Dist'])

        assert response.unit.is_equivalent('m2')

    
def test_get_item():

    with FullDetectorResponse.open(response_path) as response:

        drm = response[0]

        assert drm.ndim == 6

        assert arr_eq(drm.axes.labels,
                      ['Ei', 'Em', 'Phi', 'PsiChi', 'SigmaTau', 'Dist'])

        assert drm.unit.is_equivalent('m2')

def test_get_interp_response():

    with FullDetectorResponse.open(response_path) as response:

        drm = response.get_interp_response(SkyCoord(lon = 0*u.deg,
                                                    lat = 0*u.deg,
                                                    frame = SpacecraftFrame()))

        assert drm.ndim == 6

        assert arr_eq(drm.axes.labels,
                      ['Ei', 'Em', 'Phi', 'PsiChi', 'SigmaTau', 'Dist'])

        assert drm.unit.is_equivalent('m2')
        

    

        
        
    
