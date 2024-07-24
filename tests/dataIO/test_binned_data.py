# Imports:
from cosipy import BinnedData
from cosipy import test_data
import os, sys
import numpy as np
import pytest
from pathlib import Path

# Need to change the backend, 
# otherwise testing plots can take long time (particularly with ssh):
import matplotlib
matplotlib.use('Svg')

def test_binned_data(tmp_path):
    
    # Load file with dataIO class.
    # Note: this is the first 10 seconds of the crab sim from mini-DC2.
    yaml = os.path.join(test_data.path,"inputs_crab.yaml")
    analysis = BinnedData(yaml)
    test_filename = analysis.data_file
    analysis.data_file = os.path.join(test_data.path,analysis.data_file)
    analysis.ori_file = os.path.join(test_data.path,analysis.ori_file)

    # Read in test file and select first 0.1 seconds to reduce number of photons:
    analysis.read_tra()
    analysis.tmax = 1835478000.1
    analysis.select_data(output_name=tmp_path/"temp_unbinned")
    
    # Test default binning (which is in Galactic coordinates).
    # Use small number of bins to speed things up. 
    analysis.tmax = 1835478010.0 # to match time bin size of 5 seconds.  
    analysis.nside = 1
    analysis.phi_pix_size = 90
    analysis.get_binned_data(unbinned_data=tmp_path/"temp_unbinned.hdf5", output_name=tmp_path/"temp_binned_data",\
            make_binning_plots=True, show_plots=False)
    os.system("rm *.pdf *.png")

    assert analysis.binned_data.axes["Em"].unit == "keV" 
    assert analysis.binned_data.axes["Time"].unit == "s"
    assert analysis.binned_data.axes["Phi"].unit == "deg"
    assert analysis.binned_data.axes["PsiChi"].unit == None

    # Test binning in local coordinates and chunks:
    analysis.get_binned_data(psichi_binning="local",event_range=[0,3])
    assert np.sum(analysis.time_hist) == 3

    # Test passing a list for time binning:
    analysis.time_bins = [1835478000,1835478005,1835478010]
    analysis.get_binned_data()
    
    # Pass problematic list to exit code:
    with pytest.raises(SystemExit) as pytest_wrapped_exp:
        analysis.time_bins = [0,1]
        analysis.get_binned_data()
    assert pytest_wrapped_exp.type == SystemExit

    # Test loading binned data:
    analysis.load_binned_data_from_hdf5(tmp_path/"temp_binned_data.hdf5")

    # Test plots:
    analysis.make_basic_plot([1,1],[1,1],plt_scale="semilogx",x_error=[1,1])
    analysis.get_raw_spectrum(binned_data=tmp_path/"temp_binned_data.hdf5",\
            output_name=tmp_path/"temp_spec", show_plots=False)
    analysis.get_raw_spectrum(time_rate=True, show_plots=False)
    analysis.get_raw_lightcurve(binned_data=tmp_path/"temp_binned_data.hdf5",\
            output_name=tmp_path/"temp_lc", show_plots=False)
    analysis.plot_psichi_map_slices(0,0,tmp_path/"temp_psichi",coords=[0,0], show_plots=False)
