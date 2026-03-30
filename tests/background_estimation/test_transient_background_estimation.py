from histpy import Histogram
from cosipy import TransientBackgroundEstimation, test_data
import numpy as np
import pytest
from pathlib import Path


def test_transient_background():
    
    bkg_file = test_data.path / "GRB_bn110605183_with_bkg.hdf5"
    
    data = Histogram.open(bkg_file)
    
    estimator = TransientBackgroundEstimation(data)
    
    estimator.plot_lightcurve(plot_limits = [1.841896e9 + 400, 1.841896e9+450])
    
    # only one burst window error
    with pytest.raises(ValueError):
        estimator.add_burst_windows([1.8418964e9+4, 1.8418964e9+48], [1.8418964e9+4, 1.8418964e9+48])
        
    # burst window start > end error 
    with pytest.raises(ValueError):
        estimator.add_burst_windows([1.8418964e9+48, 1.8418964e9+4])
    
    estimator.add_burst_windows([1.8418964e9+4, 1.8418964e9+48])
    
    estimator.add_bkg_windows([1.841896e9+200, 1.841896e9+300], [1.841896e9+500, 1.841896e9+600])
    
    estimator.plot_lightcurve(burst_windows=True, bkg_windows=True)
    
    # error for currently unsupported scaling by fitting
    with pytest.raises(NotImplementedError):
        
        _ = estimator.make_background_model(scaling = "fitting")
        
    # error for scaling method other than duration and fitting
    with pytest.raises(ValueError):
        
        _ = estimator.make_background_model(scaling = "test")
    
    save_path = Path("")/"test_data.hdf5"
    
    bkg_model = estimator.make_background_model(save_path = save_path)
 
    save_path.unlink()
    
    psichi_array = np.array([4.62, 3.52, 3.96, 3.52, 5.06, 3.74, 2.86, 3.08, 2.86, 3.08, 3.52,
                               2.2 , 4.84, 4.62, 3.52, 3.08, 4.18, 3.52, 3.74, 2.86, 2.64, 3.52,
                               4.4 , 5.06, 4.62, 3.96, 2.2 , 3.08, 3.52, 3.74, 4.62, 3.08, 3.52,
                               5.5 , 3.08, 2.42, 3.08, 4.4 , 4.62, 2.64, 1.98, 5.06, 4.62, 2.64,
                               2.64, 3.52, 4.62, 7.26, 4.18, 2.86, 1.76, 3.08, 5.72, 4.4 , 3.74,
                               3.96, 5.06, 4.18, 4.18, 1.54, 3.08, 5.94, 2.86, 3.96, 4.84, 2.2 ,
                               3.74, 2.2 , 5.06, 1.98, 2.64, 5.28, 1.76, 4.84, 3.52, 3.08, 3.74,
                               2.2 , 2.86, 4.84, 4.62, 2.42, 3.74, 5.28, 1.76, 3.74, 3.08, 2.42,
                               1.76, 4.18, 2.64, 2.2 , 5.06, 2.64, 5.94, 3.96, 5.5 , 1.98, 2.42,
                               2.86, 4.84, 2.86, 5.06, 4.62, 1.98, 2.42, 4.62, 1.76, 3.08, 3.08,
                               2.42, 3.52, 5.06, 4.84, 3.96, 1.76, 2.42, 2.64, 2.64, 1.98, 2.2 ,
                               1.98, 4.62, 2.42, 3.08, 3.52, 4.62, 2.42, 3.3 , 1.98, 4.84, 5.06,
                               2.42, 3.74, 2.42, 2.2 , 2.42, 4.4 , 5.06, 2.64, 4.18, 4.62, 3.08,
                               1.76, 3.52, 5.06, 3.74, 4.62, 4.84, 2.64, 3.96, 3.3 , 1.76, 3.3 ,
                               3.08, 2.64, 4.4 , 3.08, 3.74, 4.62, 0.88, 3.08, 3.3 , 5.5 , 2.86,
                               3.52, 4.18, 2.86, 0.44, 3.96, 3.96, 4.62, 4.4 , 3.52, 4.4 , 3.08,
                               2.86, 3.96, 3.74, 2.64, 3.08, 5.06, 1.98, 1.98, 2.64, 2.2 , 4.4 ,
                               2.42, 3.96, 4.4 , 5.28, 2.2 , 1.54, 2.64, 5.06, 3.3 , 2.86, 4.62,
                               4.62, 1.76, 2.86, 4.18, 5.06, 3.08, 3.08, 4.18, 3.3 , 3.3 , 1.32,
                               3.96, 4.4 , 4.62, 4.84, 2.64, 2.64, 3.3 , 0.88, 3.3 , 5.28, 3.52,
                               4.84, 3.96, 4.18, 2.86, 1.1 , 3.52, 3.74, 2.64, 3.96, 4.18, 3.52,
                               2.42, 0.44, 2.2 , 3.3 , 3.08, 3.3 , 2.42, 3.96, 3.52, 1.76, 5.06,
                               4.4 , 2.2 , 2.42, 3.74, 1.98, 2.42, 2.64, 4.18, 3.74, 1.76, 3.96,
                               4.62, 4.84, 2.2 , 1.98, 2.86, 3.52, 2.42, 4.4 , 2.86, 3.96, 2.2 ,
                               0.88, 3.3 , 3.96, 3.74, 3.08, 3.08, 4.62, 4.84, 0.88, 4.4 , 3.74,
                               5.06, 3.96, 2.2 , 3.08, 5.06, 0.88, 4.62, 4.62, 5.06, 5.28, 3.08,
                               3.96, 1.98, 1.76, 2.86, 3.96, 3.96, 5.28, 3.96, 3.96, 2.2 , 2.2 ,
                               3.3 , 3.96, 4.4 , 3.96, 4.84, 4.18, 3.3 , 4.18, 3.74, 4.18, 2.64,
                               3.3 , 6.6 , 3.74, 2.64, 2.2 , 4.84, 4.4 , 3.96, 3.52, 5.5 , 3.3 ,
                               1.54, 3.74, 2.64, 4.18, 4.4 , 2.86, 6.16, 3.96, 3.3 , 2.64, 6.16,
                               5.06, 2.2 , 2.64, 3.3 , 4.62, 3.08, 1.76, 4.18, 4.18, 3.96, 3.96,
                               4.62, 5.06, 3.3 , 2.86, 4.62, 2.86, 4.84, 4.84, 4.62, 6.16, 4.4 ,
                               2.64, 4.18, 4.62, 3.08, 4.4 , 4.62, 5.72, 3.96, 1.54, 5.06, 4.62,
                               2.42, 5.06, 5.06, 2.86, 3.08, 1.54, 5.06, 5.28, 7.04, 5.72, 4.84,
                               5.5 , 5.28, 3.52, 5.5 , 4.84, 4.18, 4.18, 7.48, 5.5 , 5.06, 4.84,
                               4.84, 6.6 , 6.38, 3.96, 5.94, 4.18, 3.74, 5.06, 4.84, 7.04, 3.52,
                               3.74, 5.94, 4.62, 4.18, 1.98, 4.62, 5.94, 2.64, 6.82, 3.52, 5.06,
                               2.86, 1.54, 4.84, 3.08, 3.08, 5.72, 3.74, 3.74, 3.3 , 0.44, 6.16,
                               5.28, 2.86, 4.62, 3.52, 5.72, 3.96, 1.98, 3.96, 5.5 , 5.28, 5.28,
                               5.28, 4.84, 2.86, 2.42, 3.74, 3.96, 3.08, 3.52, 4.18, 4.4 , 3.52,
                               3.74, 3.52, 5.06, 4.18, 4.62, 3.74, 4.62, 2.2 , 2.42, 5.06, 3.74,
                               3.96, 3.96, 5.06, 2.42, 1.76, 2.64, 1.76, 3.96, 2.42, 3.52, 3.74,
                               4.62, 2.2 , 1.1 , 2.64, 4.18, 5.72, 2.86, 1.76, 4.84, 3.3 , 0.88,
                               2.64, 2.86, 2.42, 4.62, 4.4 , 4.18, 3.74, 1.32, 2.2 , 4.18, 3.74,
                               4.18, 4.18, 3.52, 3.74, 1.32, 3.08, 3.96, 3.52, 2.64, 2.86, 4.18,
                               4.18, 1.54, 3.52, 6.38, 1.98, 3.3 , 3.08, 5.06, 2.64, 1.54, 2.42,
                               3.96, 1.98, 3.08, 2.64, 4.4 , 3.52, 1.32, 4.18, 3.96, 1.54, 3.74,
                               3.96, 3.08, 0.44, 2.42, 3.96, 3.96, 2.42, 2.64, 4.18, 3.96, 1.98,
                               1.1 , 2.42, 2.64, 3.52, 3.74, 1.54, 2.2 , 2.64, 1.32, 3.52, 3.3 ,
                               4.18, 2.86, 5.28, 3.74, 3.96, 0.44, 3.52, 3.3 , 3.96, 3.52, 4.4 ,
                               3.96, 2.64, 0.66, 4.18, 4.62, 4.4 , 4.62, 4.4 , 3.52, 3.74, 2.64,
                               3.96, 3.52, 1.32, 1.54, 3.3 , 3.08, 1.32, 2.86, 3.52, 2.64, 3.3 ,
                               1.98, 5.28, 2.42, 2.64, 2.2 , 3.74, 3.3 , 2.42, 3.3 , 3.08, 3.08,
                               1.98, 1.76, 3.52, 3.08, 2.64, 3.3 , 4.4 , 1.98, 2.86, 2.2 , 2.42,
                               2.2 , 1.76, 3.52, 3.52, 1.76, 2.42, 1.1 , 1.76, 3.74, 3.3 , 3.08,
                               2.86, 4.84, 4.18, 1.76, 2.42, 4.18, 3.74, 5.94, 4.62, 3.3 , 1.98,
                               1.76, 3.3 , 4.84, 3.3 , 5.28, 3.08, 3.74, 4.18, 2.42, 2.2 , 3.08,
                               2.42, 1.54, 3.52, 3.52, 0.66, 1.1 , 3.3 , 2.64, 1.32, 2.86, 3.74,
                               2.64, 1.98, 3.08, 3.3 , 4.84, 3.52, 2.2 , 3.52, 3.08, 1.1 , 2.42,
                               3.96, 4.62, 4.18, 2.64, 3.74, 4.62, 1.76, 1.98, 2.2 , 3.74, 3.74,
                               2.42, 3.96, 0.88, 2.64, 3.96, 2.2 , 2.86, 3.08, 2.86, 2.2 , 1.76,
                               3.08, 3.08, 5.94, 2.42, 3.52, 0.88, 0.88, 3.74, 3.08, 4.18, 1.76,
                               2.64, 1.98, 2.2 , 3.52, 1.98, 3.74, 2.64, 1.1 , 2.86, 3.74, 2.64,
                               4.18, 3.08, 1.98, 2.2 , 5.28, 2.42, 3.3 , 1.98, 1.32, 2.42, 5.28,
                               1.76, 5.06, 2.2 , 1.54, 2.86, 2.64, 4.4 , 3.96, 2.42, 3.52, 2.86,
                               3.96, 1.98, 3.3 , 3.52, 4.4 , 5.28, 2.42, 2.42, 1.76, 2.64, 3.08,
                               2.64, 1.76, 3.08, 2.42, 1.32, 1.32, 2.86, 3.52, 2.64, 2.2 , 2.64,
                               3.3 , 1.32, 1.98, 1.76, 2.86, 1.76, 1.76, 3.08, 3.74, 2.86, 4.62,
                               5.94, 2.86, 1.98, 4.84, 2.86, 3.08, 4.18, 3.3 , 3.3 , 2.2 , 3.96,
                               2.42, 2.86, 1.54, 2.42, 2.42, 1.54, 2.2 , 2.64, 1.98])
    
    assert np.allclose(bkg_model.project("PsiChi").contents.todense(),psichi_array)
    
    assert np.allclose([44.0], estimator.burst_durations)
    
    assert np.allclose([[1841896404.0, 1841896448.0]], estimator.burst_windows)
    
    assert np.allclose([[1841896200.0, 1841896300.0], [1841896500.0, 1841896600.0]], estimator.bkg_windows)
    
    assert np.allclose([100.0, 100.0], estimator.bkg_durations)
    
def test_data_type_error():
    
    data = np.array([10])

    with pytest.raises(TypeError):
        
        estimator = TransientBackgroundEstimation(data)
        
        
def test_window_error():
    
    bkg_file = test_data.path / "GRB_bn110605183_with_bkg.hdf5"
    
    data = Histogram.open(bkg_file)
    
    estimator = TransientBackgroundEstimation(data)
    
    # making background error when the background window not defined
    with pytest.raises(ValueError):
        
        _ = estimator.make_background_model()
        
    estimator.add_bkg_windows([1.841896e9+200, 1.841896e9+300], [1.841896e9+500, 1.841896e9+600])
    
    # making background error when the burst window not defined
    with pytest.raises(ValueError):
        
        _ = estimator.make_background_model()
        
def test_Time_axis_error():
    
    bkg_file = test_data.path / "GRB_bn110605183_with_bkg_no_time_axis.hdf5"  
    data = Histogram.open(bkg_file)
    
    # error when there is no Time axis in the data
    with pytest.raises(ValueError):
        
        estimator = TransientBackgroundEstimation(data)
        
    bkg_file = test_data.path / "GRB_bn110605183_with_bkg_time_axis_idx_not_zero.hdf5"
    data = Histogram.open(bkg_file)
    
    # error when there is the Time axis doesn't have index of 0
    with pytest.raises(ValueError):
        
        estimator = TransientBackgroundEstimation(data)
        
        
def test_slice_timetage_error():
    
    bkg_file = test_data.path / "GRB_bn110605183_with_bkg.hdf5"
    
    data = Histogram.open(bkg_file)
    
    estimator = TransientBackgroundEstimation(data)
    
    # error when slicing window start > end
    with pytest.raises(ValueError):
        
        estimator.slice_by_timetags(1.8418964e9+48, 1.8418964e9+4)
        
    # error when the scaling start is smaller than the minumum time tag
    with pytest.raises(ValueError):
        
        estimator.slice_by_timetags(1841896154.0-10, 1.8418964e9+48)