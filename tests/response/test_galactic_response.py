import numpy as np
from numpy import array_equal as arr_eq

from astropy.coordinates import SkyCoord
import astropy.units as u

from cosipy import test_data
from cosipy.response import GalacticResponse

response_path = test_data.path / "test_precomputed_response.h5"

def test_open():

    with GalacticResponse.open(response_path) as response:

        assert response.filename == response_path

        assert response.dtype == np.float64

        assert response.ndim == 5

        assert response.shape == tuple(response.axes.nbins)

        assert arr_eq(response.eff_area_correction,
                      np.ones(response.axes['Ei'].nbins, response.dtype))

        assert arr_eq(response.axes.labels,
                      ['NuLambda', 'Ei', 'Em', 'Phi', 'PsiChi'])

        assert response.unit.is_equivalent('cm2 s')

def test_get_item():

    with GalacticResponse.open(response_path) as response:

        # test reading single NuLambda slice of response

        drm = response[0]

        assert drm.ndim == 4

        assert arr_eq(drm.axes.labels,
                      ['Ei', 'Em', 'Phi', 'PsiChi'])

        assert drm.unit.is_equivalent('cm2 s')

        # test reading entire response at once

        rsp = response.to_dr()

        assert rsp.ndim == 5

        assert arr_eq(rsp.axes.labels,
                      ['NuLambda', 'Ei', 'Em', 'Phi', 'PsiChi'])

        assert rsp.unit.is_equivalent('cm2 s')

def test_get_point_source_response():

    with GalacticResponse.open(response_path) as response:

        drm = response.get_point_source_response(SkyCoord(l = 0*u.deg,
                                                          b = 0*u.deg,
                                                          frame = "galactic"))

        assert drm.ndim == 4

        assert arr_eq(drm.axes.labels,
                      ['Ei', 'Em', 'Phi', 'PsiChi'])

        assert drm.unit.is_equivalent('cm2 s')

def test_get_counts():

    with GalacticResponse.open(response_path) as response:

        data = response.get_counts(2)

        assert data.shape == tuple(response.axes.nbins[1:])

        data = response.get_counts(2, em_slice = slice(2,3))

        assert data.shape == (response.axes.nbins[1],
                              1,
                              response.axes.nbins[3],
                              response.axes.nbins[4])
