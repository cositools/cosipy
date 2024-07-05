import pytest
from histpy import Histogram

from cosipy.image_deconvolution import ImageDeconvolutionDataInterfaceBase, DataIF_COSI_DC2, CoordsysConversionMatrix, AllSkyImageModel
from cosipy.response import FullDetectorResponse
from cosipy import test_data

def test_dataIF_COSI_DC2_miniDC2_format():

    event_binned_data = Histogram.open(test_data.path / "test_event_histogram_localCDS.hdf5")
    dict_bkg_binned_data = {"bkg": Histogram.open(test_data.path / "test_event_histogram_localCDS.hdf5")}
    rsp = FullDetectorResponse.open(test_data.path / "test_full_detector_response.h5")
    ccm = CoordsysConversionMatrix.open(test_data.path / "image_deconvolution/ccm_time_test.hdf5")

    data = DataIF_COSI_DC2.load(name = 'testdata_miniDC2', 
                                event_binned_data = event_binned_data, 
                                dict_bkg_binned_data = dict_bkg_binned_data, 
                                rsp = rsp, 
                                coordsys_conv_matrix = ccm, 
                                is_miniDC2_format = True)

def test_dataIF_COSI_DC2_localCDS_scatt():

    event_binned_data = Histogram.open(test_data.path / "image_deconvolution" / "test_event_histogram_localCDS_scatt.h5")
    dict_bkg_binned_data = {"bkg": Histogram.open(test_data.path / "image_deconvolution" / "test_event_histogram_localCDS_scatt.h5")}
    rsp = FullDetectorResponse.open(test_data.path / "test_full_detector_response.h5")
    ccm = CoordsysConversionMatrix.open(test_data.path / "image_deconvolution" / 'ccm_scatt_use_averaged_pointing_True_test.hdf5')

    data = DataIF_COSI_DC2.load(name = "testdata_localCDS_scatt", 
                                event_binned_data = event_binned_data, 
                                dict_bkg_binned_data = dict_bkg_binned_data, 
                                rsp = rsp, 
                                coordsys_conv_matrix = ccm, 
                                is_miniDC2_format = True)

    model = AllSkyImageModel(rsp.axes['NuLambda'].nside, rsp.axes['Ei'].edges)
    model[:] = 1.0 * model.unit

    expectation = data.calc_expectation(model = model, dict_bkg_norm = {"bkg": 1.0})

    loglikelihood = data.calc_loglikelihood(expectation)

    data.calc_T_product(expectation)

    data.calc_bkg_model_product("bkg", expectation)

def test_dataIF_COSI_DC2_galacticCDS():

    event_binned_data = Histogram.open(test_data.path / "test_event_histogram_galacticCDS.hdf5").project(["Em", "Phi", "PsiChi"])
    dict_bkg_binned_data = {"bkg": Histogram.open(test_data.path / "test_event_histogram_galacticCDS.hdf5").project(["Em", "Phi", "PsiChi"])}
    precomputed_response = Histogram.open(test_data.path / "test_precomputed_response.h5")

    data = DataIF_COSI_DC2.load(name = "testdata_galacticCDS", 
                                event_binned_data = event_binned_data, 
                                dict_bkg_binned_data = dict_bkg_binned_data, 
                                rsp = precomputed_response, 
                                coordsys_conv_matrix = None, 
                                is_miniDC2_format = False)

    model = AllSkyImageModel(precomputed_response.axes['NuLambda'].nside, precomputed_response.axes['Ei'].edges)
    model[:] = 1.0 * model.unit

    expectation = data.calc_expectation(model = model, dict_bkg_norm = {"bkg": 1.0})

    loglikelihood = data.calc_loglikelihood(expectation)

    data.calc_T_product(expectation)

    data.calc_bkg_model_product("bkg", expectation)
