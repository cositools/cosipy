import numpy as np
from numpy import array_equal as arr_eq

from scoords import SpacecraftFrame
from astropy.coordinates import SkyCoord
import astropy.units as u

from histpy import Histogram, HealpixAxis, Axis

from cosipy import test_data
from cosipy.response import FullDetectorResponse
from cosipy.spacecraftfile import SpacecraftFile

response_path = test_data.path / "test_full_detector_response.h5"
orientation_path = test_data.path / "20280301_first_10sec.fits"

def test_open():

    with FullDetectorResponse.open(response_path, dtype=np.float32) as response:

        assert response.filename == response_path

        assert response.ndim == 5

        assert response.shape == tuple(response.axes.nbins)

        assert response.eff_area_correction.dtype == np.float32
        assert len(response.eff_area_correction) == response.axes['Ei'].nbins

        assert arr_eq(response.axes.labels,
                      ['NuLambda', 'Ei', 'Em', 'Phi', 'PsiChi'])

        assert response.unit.is_equivalent('m2')

        hdr = response.headers

        for tag in ('Version', 'NM', 'OD', 'TS', 'SA', 'SP', 'BE', 'CE'):
            assert tag in hdr

def test_get_item():

    with FullDetectorResponse.open(response_path) as response:

        drm = response[0]

        assert drm.ndim == 4

        assert arr_eq(drm.axes.labels,
                      ['Ei', 'Em', 'Phi', 'PsiChi'])

        assert drm.unit.is_equivalent('m2')

    with FullDetectorResponse.open(response_path, dtype=np.float32, cache_size = 100) as response:

        drm = response[0]

def test_get_counts():

    with FullDetectorResponse.open(response_path) as response:

        data = response.get_counts(2)

        assert data.shape == tuple(response.axes.nbins[1:])

        data = response.get_counts(2, em_slice = slice(2,3))

        assert data.shape == (response.axes.nbins[1],
                              1,
                              response.axes.nbins[3],
                              response.axes.nbins[4])

def test_get_interp_response():

    with FullDetectorResponse.open(response_path) as response:

        drm = response.get_interp_response(SkyCoord(lon = 0*u.deg,
                                                    lat = 0*u.deg,
                                                    frame = SpacecraftFrame()))

        assert drm.ndim == 4

        assert arr_eq(drm.axes.labels,
                      ['Ei', 'Em', 'Phi', 'PsiChi'])

        assert drm.unit.is_equivalent('m2')

def test_get_point_source_response():

    orientation = SpacecraftFile.open(orientation_path)
    coord = SkyCoord(l=0,b=0,unit=u.deg,frame="galactic")

    with FullDetectorResponse.open(response_path) as response:

        # test call with dwell_map
        src_path = orientation.get_target_in_sc_frame(coord)
        exp_map = orientation.get_dwell_map(response, src_path)

        psr = response.get_point_source_response(exposure_map = exp_map)

        # test call with source + scatt_map
        scatt_map = orientation.get_scatt_map(nside=16,
                                              target_coord=coord)

        psr = response.get_point_source_response(coord=coord,
                                                 scatt_map=scatt_map)

        # test stripping extra dimensions from SkyCoord
        coord = SkyCoord(l=[0],b=[0],unit=u.deg,frame="galactic")
        psr = response.get_point_source_response(coord=coord,
                                                 scatt_map=scatt_map)
def test_get_extended_source_response():

    orientation = SpacecraftFile.open(orientation_path)

    with FullDetectorResponse.open(response_path) as response:

        extended_source_response = response.get_extended_source_response(orientation,
                                                                         coordsys = 'galactic',
                                                                         nside_image = None,
                                                                         nside_scatt_map = None,
                                                                         earth_occ = True)

        assert extended_source_response.ndim == 5

        assert arr_eq(extended_source_response.axes.labels,
                      ['NuLambda', 'Ei', 'Em', 'Phi', 'PsiChi'])

        assert extended_source_response.unit.is_equivalent('cm2 s')

def test_merge_psr_to_extended_source_response(tmp_path):

    orientation = SpacecraftFile.open(orientation_path)

    with FullDetectorResponse.open(response_path) as response:

        for ipix_image in range(response.axes['NuLambda'].npix):

            psr = response.get_point_source_response_per_image_pixel(ipix_image, orientation,
                                                                     coordsys='galactic',
                                                                     nside_image=None,
                                                                     nside_scatt_map=None,
                                                                     earth_occ=True)

            psr.write(tmp_path / f"psr_{ipix_image:08}.h5")


        extended_source_response = response.merge_psr_to_extended_source_response(str(tmp_path / "psr_"),
                                                                                  coordsys = 'galactic',
                                                                                  nside_image = None)

        assert extended_source_response.ndim == 5

        assert arr_eq(extended_source_response.axes.labels,
                      ['NuLambda', 'Ei', 'Em', 'Phi', 'PsiChi'])

        assert extended_source_response.unit.is_equivalent('cm2 s')
