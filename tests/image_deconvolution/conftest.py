import pytest

import numpy as np
from histpy import Histogram, Axis, Axes

from cosipy import test_data
from cosipy.image_deconvolution import DataIF_COSI_DC2, AllSkyImageModel

@pytest.fixture
def dataset():

    event_binned_data = Histogram.open(test_data.path / "test_event_histogram_galacticCDS.hdf5").project(["Em", "Phi", "PsiChi"])
    dict_bkg_binned_data = {"bkg": Histogram.open(test_data.path / "test_event_histogram_galacticCDS.hdf5").project(["Em", "Phi", "PsiChi"]) * 0.1}
    precomputed_response = Histogram.open(test_data.path / "test_precomputed_response.h5")

    data = DataIF_COSI_DC2.load(name = "testdata_galacticCDS",
                                event_binned_data = event_binned_data,
                                dict_bkg_binned_data = dict_bkg_binned_data,
                                rsp = precomputed_response,
                                coordsys_conv_matrix = None)

    return [data]

@pytest.fixture
def model(dataset):

    model = AllSkyImageModel(dataset[0].model_axes['lb'].nside, dataset[0].model_axes['Ei'].edges)
    model[:] = 1.0 * model.unit

    return model

@pytest.fixture
def mask(dataset):

    axes = Axes([dataset[0].model_axes['lb'], dataset[0].model_axes['Ei']])

    mask = Histogram(axes, contents = np.ones(axes.nbins, dtype = bool))

    mask[0,0] = False

    return mask
