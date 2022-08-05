import numpy as np
from numpy import array_equal as arr_eq

import astropy.units as u

from cosipy import test_data
from cosipy.response import FullDetectorResponse

response_path = test_data.path / "test_full_detector_response.h5"

def test_open():

    with FullDetectorResponse(response_path) as response:

        assert response.filename == response_path

        assert response.ndim == 7

        assert arr_eq(response.axes.labels,
                      ['NuLambda', 'Ei', 'Em', 'Phi', 'PsiChi', 'SigmaTau', 'Dist'])

        assert response.unit.is_equivalent('m2')

    
