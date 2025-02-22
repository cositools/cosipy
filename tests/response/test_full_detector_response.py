import numpy as np
from numpy import array_equal as arr_eq

from scoords import SpacecraftFrame
from astropy.coordinates import SkyCoord
import astropy.units as u

from cosipy import test_data
from cosipy.response import FullDetectorResponse
from cosipy.spacecraftfile import SpacecraftFile

response_path = test_data.path / "test_full_detector_response_dense.h5"
orientation_path = test_data.path / "20280301_first_10sec.ori"

def test_open():

    with FullDetectorResponse.open(response_path) as response:

        assert response.filename == response_path

        assert response.ndim == 5

        assert arr_eq(response.axes.labels,
                      ['NuLambda', 'Ei', 'Em', 'Phi', 'PsiChi'])

        assert response.unit.is_equivalent('m2')

    
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
        
def test_get_extended_source_response():    
    
    orientation = SpacecraftFile.parse_from_file(orientation_path)

    with FullDetectorResponse.open(response_path) as response:

        extended_source_response = response.get_extended_source_response(orientation, 
                                                                         coordsys = 'galactic', 
                                                                         nside_image = None, 
                                                                         nside_scatt_map = None, 
                                                                         Earth_occ = True)

        assert extended_source_response.ndim == 5

        assert arr_eq(extended_source_response.axes.labels,
                      ['NuLambda', 'Ei', 'Em', 'Phi', 'PsiChi'])

        assert extended_source_response.unit.is_equivalent('cm2 s')

def test_merge_psr_to_extended_source_response(tmp_path):

    orientation = SpacecraftFile.parse_from_file(orientation_path)

    with FullDetectorResponse.open(response_path) as response:

        for ipix_image in range(response.axes['NuLambda'].npix):

            psr = response.get_point_source_response_per_image_pixel(ipix_image, orientation, 
                                                                     coordsys='galactic',
                                                                     nside_image=None,
                                                                     nside_scatt_map=None,
                                                                     Earth_occ=True)

            psr.write(tmp_path / f"psr_{ipix_image:08}.h5")


        extended_source_response = response.merge_psr_to_extended_source_response(str(tmp_path / "psr_"), 
                                                                                  coordsys = 'galactic', 
                                                                                  nside_image = None)

        assert extended_source_response.ndim == 5

        assert arr_eq(extended_source_response.axes.labels,
                      ['NuLambda', 'Ei', 'Em', 'Phi', 'PsiChi'])

        assert extended_source_response.unit.is_equivalent('cm2 s')
