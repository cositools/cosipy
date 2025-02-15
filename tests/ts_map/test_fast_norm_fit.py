from cosipy import test_data
from cosipy import FastTSMap
from histpy import Histogram
import numpy as np
from cosipy.ts_map import FastNormFit as fnf


def test_solve():

    # read the signal+background
    src_bkg_path = test_data.path / "ts_map_src_bkg.h5"
    src_bkg = Histogram.open(src_bkg_path)

    # read the background
    bkg_path = test_data.path / "ts_map_bkg.h5"
    bkg = Histogram.open(bkg_path)

    # get the cds arrays of src_bkg and bkg
    src_bkg_cds_array = FastTSMap.get_cds_array(src_bkg, [0,10])
    bkg_cds_array = FastTSMap.get_cds_array(bkg, [0,10])

    # read the cds array of expectation
    ei_path = test_data.path / "ei_cds_array.npy"
    ei_cds_array = np.load(ei_path)


    # calculate the ts value
    fit = fnf(max_iter=1000)
    result = fit.solve(src_bkg_cds_array, bkg_cds_array, ei_cds_array)

    assert result[0] == 187.3360310655543

    assert result[1] == 0.02119470713546078

    assert result[2] == 0.0055665881497504646