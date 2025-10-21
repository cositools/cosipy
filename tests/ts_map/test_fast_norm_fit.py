from cosipy import test_data
from cosipy import FastTSMap
from histpy import Histogram
import numpy as np
from cosipy.ts_map import FastNormFit as fnf
import pytest


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


def test_solve():


    # calculate the ts value
    fit = fnf(max_iter=1000)
    result = fit.solve(src_bkg_cds_array, bkg_cds_array, ei_cds_array)
    assert result[0] == 187.3360310655543
    assert result[1] == 0.02119470713546078
    assert result[2] == 0.0055665881497504646
    
    # test Underfluctuation with allow_negative = True
    fit = fnf(max_iter=1000, allow_negative = True)
    result = fit.solve(src_bkg_cds_array-0.03, bkg_cds_array-0.03, ei_cds_array)
    assert np.allclose(result[0], -14.881883359545997)
    assert np.allclose(result[1],  -0.0033011907532943985)
    assert np.allclose(result[2], 0.0008557396828509948)
    assert result[3] is False
    
    # test Underfluctuation with allow_negative = False
    fit = fnf(max_iter=1000, allow_negative=False)
    result = fit.solve(src_bkg_cds_array-100, bkg_cds_array, ei_cds_array)
    assert np.allclose(result[0], 0)
    assert np.allclose(result[1], 0)
    assert np.allclose(result[2], 0.0007305308374779704)
    assert result[3] is False
    
    
    # test Underfluctuation with allow_negative = False, but ddts !=0
    # for line: norm_err = -(sqrt(dts0*dts0 - 2*ddts0) + dts0) / ddts0
    fit = fnf(max_iter=1000, allow_negative=False)
    result = fit.solve(src_bkg_cds_array-0.3, bkg_cds_array-0.3, ei_cds_array)
    assert np.allclose(result[0], 0)
    assert np.allclose(result[1], 0)
    assert np.allclose(result[2], 0.0007297521305655876)
    assert result[3] is False
    

    # test when ddts > 0, ts<0
    # for line: if ts < -self.zero_ts_tol:
    fit = fnf(max_iter=1000, allow_negative=False, zero_ts_tol = 0.1)
    result = fit.solve(src_bkg_cds_array-0.1, bkg_cds_array+0.1, ei_cds_array)
    assert np.allclose(result[0], 0)
    assert np.allclose(result[1], 0.00020618010210669564)
    assert np.allclose(result[2], 0.0036407149958900074)
    assert result[3] is False
    
    # test when ddts > 0, ts<0
    # for line elif: ts < 0:
    fit = fnf(max_iter=1000, allow_negative=False)
    result = fit.solve(src_bkg_cds_array-0.1, bkg_cds_array+0.1, ei_cds_array)
    assert np.allclose(result[0], -0.013547902882134743)
    assert np.allclose(result[1], 0.00020618010210669564)
    assert np.allclose(result[2], 0.0036407149958900074)
    assert result[3] is True
    

def test_dts():
    
    # only run the remaining part

    with pytest.raises(ValueError):
        fnf.dts(data = src_bkg_cds_array, 
                bkg = bkg_cds_array, 
                unit_excess = ei_cds_array, 
                norm = 0, 
                order=0)
        