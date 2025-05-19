import numpy as np
from numpy import array_equal as arr_eq

from scoords import SpacecraftFrame
from astropy.coordinates import SkyCoord
import astropy.units as u

from cosipy import test_data
from cosipy.response import FullDetectorResponse
from cosipy.response.FullDetectorResponse import cosi_response

from pytest import approx

response_path = test_data.path/"test_full_detector_response.h5"

def test_get_effective_area():

    with FullDetectorResponse.open(response_path) as response:

        drm = response[0]

        assert drm.unit.is_equivalent('m2')

        assert drm.get_effective_area(511*u.keV).to_value('cm2') == approx(58.425129112973515)

        aeff = drm.get_effective_area()

        assert aeff.ndim == 1

        assert arr_eq(aeff.axes.labels, ['Ei'])

        assert aeff.unit.is_equivalent('m2')

        assert drm.axes['Ei'] == aeff.axis

        print(aeff.contents)

        assert np.allclose(aeff.contents.to_value('cm2'),
                           [ 9.08694864, 35.97844922, 56.56536617, 58.62912692, 53.78421084,
                            46.68086907, 37.56363809, 25.57306065, 18.39937526, 10.23204873])


def test_spectral_response():

    with FullDetectorResponse.open(response_path) as response:

        drm = response[0]

        spec = drm.get_spectral_response().to_dense()

        assert np.allclose(spec[0].to_value('cm2'),
                           [8.92579524, 0.16115340, 0.        , 0.        , 0.        ,
                            0.        , 0.        , 0.        , 0.        , 0.        ])
        assert np.allclose(spec[5].to_value('cm2'),
                           [0.333909932, 2.01132240,    3.42465851,   3.72364539, 7.76190391,
                            29.39873340, 0.0266955763,  0.,           0.,         0.])

        assert np.allclose(spec[9].to_value('cm2'),
                           [0.09259213, 0.45248619, 0.86409322, 1.13900893, 0.5104818,
                            0.4244306,  0.94009569, 2.3514951,  3.09151486, 0.36585021])


def test_spectral_readwrite(tmp_path):

    with FullDetectorResponse.open(response_path) as response:

        drm = response[0]
        spec = drm.get_spectral_response().to_dense()
        aeff = drm.get_effective_area()

        drm.write(tmp_path / "drmfile", overwrite=True)

        drm2 = drm.open(tmp_path / "drmfile")
        spec2 = drm.get_spectral_response().to_dense()
        aeff2 = drm.get_effective_area()

        assert spec == spec2 and aeff == aeff2

def test_get_dispersion_matrix():

    with FullDetectorResponse.open(response_path) as response:

        drm = response[0]

        rmf = drm.get_dispersion_matrix()

        assert np.allclose(rmf.project('Ei').to_dense().contents, 1)

def test_cosi_response(tmp_path):

    # Just check if it runs without errors

    cosi_response(['dump','header', str(response_path)])

    cosi_response(['dump', 'aeff', str(response_path),
                   '--lat', '90deg', '--lon', '0deg'])

    cosi_response(['plot', 'aeff', str(response_path),
                   '--lat', '90deg', '--lon', '0deg',
                   '-o', str(tmp_path/'test_plot_aeff.png')])

    cosi_response(['dump', 'expectation', str(response_path),
                   '-c', str(test_data.path/'cosi-response-config-example.yaml'),
                   '--config-override', 'sources:source (point source):spectrum:main:Powerlaw:index:value=-3'])

    cosi_response(['plot', 'expectation', str(response_path),
                   '-c', str(test_data.path/'cosi-response-config-example.yaml'),
                   '-o', str(tmp_path/'test_plot_expectation.png')])

    cosi_response(['plot', 'dispersion' ,str(response_path),
                   '--lat', '90deg', '--lon', '0deg',
                   '-o', str(tmp_path/'test_plot_dispersion.png')])
