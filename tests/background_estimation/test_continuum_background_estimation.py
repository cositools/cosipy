import pytest
from cosipy.background_estimation import ContinuumEstimation
from cosipy import test_data

def test_continuum_background_estimation():
   
    
    instance = ContinuumEstimation() 
     
    # Test main method:
    data_yaml = test_data.path / "inputs_crab_continuum_bg_estimation_testing.yaml"
    data_file = test_data.path / "crab_bkg_binned_data_for_continuum_bg_testing.hdf5"
    psr_file = test_data.path / "test_precomputed_response.h5"
    psr = instance.load_psr_from_file(psr_file)
    
    instance.continuum_bg_estimation(data_file, data_yaml, psr, e_loop=(1,2), s_loop=(1,2))
