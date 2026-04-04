import pytest
from histpy import Histogram
import numpy as np
import astropy.units as u

from cosipy.image_deconvolution import DataIF_COSI_DC2, AllSkyImageModel, DataInterfaceCollection
from cosipy.response import FullDetectorResponse
from cosipy import test_data

def test_dataIF_collection():

    event_binned_data = Histogram.open(test_data.path / "test_event_histogram_galacticCDS.hdf5").project(["Em", "Phi", "PsiChi"])
    dict_bkg_binned_data = {"bkg": Histogram.open(test_data.path / "test_event_histogram_galacticCDS.hdf5").project(["Em", "Phi", "PsiChi"])}
    precomputed_response = Histogram.open(test_data.path / "test_precomputed_response.h5")

    data = DataIF_COSI_DC2.load(name = "testdata_galacticCDS",
                                event_binned_data = event_binned_data,
                                dict_bkg_binned_data = dict_bkg_binned_data,
                                rsp = precomputed_response,
                                coordsys_conv_matrix = None)

    data_collection = DataInterfaceCollection([data])

    model = AllSkyImageModel(precomputed_response.axes['NuLambda'].nside, precomputed_response.axes['Ei'].edges)
    model[:] = 1.0 * model.unit

    expectation_list = data_collection.calc_expectation_list(model = model, dict_bkg_norm = {"bkg": 1.0})

    expectation_list_src = data_collection.calc_source_expectation_list(model = model)

    expectation_list_bkg = data_collection.calc_bkg_expectation_list(dict_bkg_norm = {"bkg": 1.0})

    assert np.all(expectation_list[0].contents == data_collection.combine_expectation_list(expectation_list_src, expectation_list_bkg)[0].contents)

    log_likelihood_list = data_collection.calc_total_log_likelihood(expectation_list)

    assert np.isclose(log_likelihood_list, -8330.6845369209)

    assert np.isclose(data_collection.calc_summed_bkg_model("bkg"), 4676.0)

    assert np.isclose(data_collection.calc_summed_bkg_model_product("bkg", expectation_list), 319612.9093017483)

    assert np.isclose(np.sum(data_collection.calc_summed_exposure_map()), 22580.773319674616 * u.Unit("cm2 s sr"))
