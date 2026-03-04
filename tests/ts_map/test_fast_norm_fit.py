from cosipy import test_data
from cosipy import FastTSMap
from histpy import Histogram
import numpy as np
from cosipy.ts_map import FastNormFit as fnf


def test_solve():

    # read the signal+background CDS
    src_bkg_path = test_data.path / "ts_map_src_bkg.h5"
    src_bkg = Histogram.open(src_bkg_path).todense().project(["Em", "Phi", "PsiChi"])

    # read the background CDS
    bkg_path = test_data.path / "ts_map_bkg.h5"
    bkg = Histogram.open(bkg_path).todense().project(["Em", "Phi", "PsiChi"])

    # get the cds arrays of src_bkg and bkg
    src_bkg_cds_array = FastTSMap._get_cds_array(src_bkg, slice(0,10))
    bkg_cds_array = FastTSMap._get_cds_array(bkg, slice(0,10))

    # read the cds array of expectation
    ei_path = test_data.path / "ei_cds_array.npy"
    ei_cds_array = np.load(ei_path)

    ei_sum = np.sum(ei_cds_array)

    # remove empty/invalid CDS cells
    valid = np.logical_and(src_bkg_cds_array > 0,
                           bkg_cds_array > 0)
    src_bkg_cds_array = src_bkg_cds_array[valid]
    bkg_cds_array = bkg_cds_array[valid]
    ei_cds_array = ei_cds_array[valid]

    # calculate the ts value
    fit = fnf(max_iter=1000)
    result = fit.solve(src_bkg_cds_array, bkg_cds_array, ei_cds_array, ei_sum)

    assert np.isclose(result[0], 187.3360310655543)

    assert np.isclose(result[1], 0.02119470713546078)

    assert np.isclose(result[2], 0.0055665881497504646)

    ##############################

    # raise bkg to a very high value to test under-fluctuation code
    src_bkg_cds_array = FastTSMap._get_cds_array(src_bkg, slice(0,10))
    bkg_cds_array = FastTSMap._get_cds_array(bkg*10000, slice(0,10))

    # read the cds array of expectation
    ei_path = test_data.path / "ei_cds_array.npy"
    ei_cds_array = np.load(ei_path)

    ei_sum = np.sum(ei_cds_array)

    # remove empty/invalid CDS cells
    valid = np.logical_and(src_bkg_cds_array > 0,
                           bkg_cds_array > 0)
    src_bkg_cds_array = src_bkg_cds_array[valid]
    bkg_cds_array = bkg_cds_array[valid]
    ei_cds_array = ei_cds_array[valid]

    # calculate the ts value
    fit = fnf(max_iter=1000)
    result = fit.solve(src_bkg_cds_array, bkg_cds_array, ei_cds_array, ei_sum)

    assert np.isclose(result[0], 0)

    assert np.isclose(result[1], 0)

    assert np.isclose(result[2], 0.0009080550103295215)

    fit = fnf(max_iter=1000, allow_negative = True)
    result = fit.solve(src_bkg_cds_array, bkg_cds_array, ei_cds_array, ei_sum)

    -181.40126466889595, -0.3298980580267179, np.float64(0.02449399381236294)

    assert np.isclose(result[0], -181.40126466889595)


    assert np.isclose(result[1], -0.3298980580267179)


    assert np.isclose(result[2], 0.02449399381236294)
