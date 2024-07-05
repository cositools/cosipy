import astropy.units as u

from cosipy import test_data
from cosipy.response import FullDetectorResponse
from cosipy.spacecraftfile import SpacecraftFile
from cosipy.image_deconvolution import SpacecraftAttitudeExposureTable

from cosipy.image_deconvolution import CoordsysConversionMatrix

def test_coordsys_conversion_matrix_time():

    full_detector_response = FullDetectorResponse.open(test_data.path / "test_full_detector_response.h5")

    ori = SpacecraftFile.parse_from_file(test_data.path / "20280301_first_10sec.ori")

    ccm = CoordsysConversionMatrix.time_binning_ccm(full_detector_response, ori, [ori.get_time()[0].value, ori.get_time()[-1].value] * u.s)

    assert ccm.binning_method == 'Time'

    ccm_test = CoordsysConversionMatrix.open(test_data.path / "image_deconvolution/ccm_time_test.hdf5")

    assert ccm.axes     == ccm_test.axes
    assert ccm.contents == ccm_test.contents
    assert ccm.unit     == ccm_test.unit

def test_coordsys_conversion_matrix_scatt():

    full_detector_response = FullDetectorResponse.open(test_data.path / "test_full_detector_response.h5")

    exposure_table = SpacecraftAttitudeExposureTable.from_fits(test_data.path / "image_deconvolution/exposure_table_test_nside1_ring.fits") 

    ccm = CoordsysConversionMatrix.spacecraft_attitude_binning_ccm(full_detector_response, exposure_table, use_averaged_pointing = False)

    assert ccm.binning_method == 'ScAtt'

    ccm_test = CoordsysConversionMatrix.open(test_data.path / "image_deconvolution/ccm_scatt_use_averaged_pointing_False_test.hdf5")

    assert ccm.axes     == ccm_test.axes
    assert ccm.contents == ccm_test.contents
    assert ccm.unit     == ccm_test.unit

    ccm = CoordsysConversionMatrix.spacecraft_attitude_binning_ccm(full_detector_response, exposure_table, use_averaged_pointing = True)

    assert ccm.binning_method == 'ScAtt'

    ccm_test = CoordsysConversionMatrix.open(test_data.path / "image_deconvolution/ccm_scatt_use_averaged_pointing_True_test.hdf5")

    assert ccm.axes     == ccm_test.axes
    assert ccm.contents == ccm_test.contents
    assert ccm.unit     == ccm_test.unit
