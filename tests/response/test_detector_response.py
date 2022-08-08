import numpy as np
from numpy import array_equal as arr_eq

from cosipy.coordinates import SpacecraftFrame
from astropy.coordinates import SkyCoord
import astropy.units as u

from cosipy import test_data
from cosipy.response import FullDetectorResponse

from pytest import approx

response_path = test_data.path / "test_full_detector_response.h5"

def test_get_effective_area():
    
    with FullDetectorResponse.open(response_path) as response:

        drm = response[0]

        assert drm.ndim == 6
        
        assert arr_eq(drm.axes.labels,
                      ['Ei', 'Em', 'Phi', 'PsiChi', 'SigmaTau', 'Dist'])
        
        assert drm.unit.is_equivalent('m2')

        assert drm.get_effective_area(511*u.keV).to_value('cm2') == approx(1.70905119244)

        aeff = drm.get_effective_area()
        
        assert aeff.ndim == 1
        
        assert arr_eq(aeff.axes.labels, ['Ei'])
        
        assert aeff.unit.is_equivalent('m2')

        assert drm.axes['Ei'] == aeff.axis
        
        print(aeff.contents)

        assert np.allclose(aeff.contents.to_value('cm2'),
                           [0.06106539, 0.46578213, 1.19104624, 1.67014663, 2.07199291,
                            2.24013146, 2.09571064, 1.71057496, 1.17616575, 0.80718952])
        
        
def test_spectral_response():
        
    with FullDetectorResponse.open(response_path) as response:
        
        drm = response[0]

        spec = drm.get_spectral_response().to_dense()

        assert np.allclose(spec[0].to_value('cm2'),
                           [0.05987875, 0.00118664, 0.        , 0.        , 0.        ,
                            0.        , 0.        , 0.        , 0.        , 0.        ])
        assert np.allclose(spec[5].to_value('cm2'),
                           [0.00760535, 0.04997702, 0.16718662, 0.06108004, 0.58040561,
                            1.37079256, 0.00308426, 0.        , 0.        , 0.        ])
        assert np.allclose(spec[9].to_value('cm2'),
                           [0.00543126, 0.03039386, 0.07430311, 0.05662412, 0.12452137,
                            0.12412425, 0.1377126 , 0.11414493, 0.10317496, 0.03675906])
        
def test_get_dispersion_matrix():

    with FullDetectorResponse.open(response_path) as response:
        
        drm = response[0]
        
        rmf = drm.get_dispersion_matrix()

        assert np.allclose(rmf.project('Ei').to_dense().contents, 1)
    





        
