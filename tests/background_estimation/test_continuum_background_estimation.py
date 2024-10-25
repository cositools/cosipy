import pytest
from cosipy.background_estimation import ContinuumEstimation
from cosipy import test_data

def test_line_background_estimation():
   
    instance = ContinuumEstimation() 
     
    # Test main method:
    data_file = test_data.path / "bkg_pl.h5"
    data_yaml = test_data.path / "inputs_crab.yaml"
    psr_file = test_data.path / "test_precomputed_response.h5"
    #psr_file = "crab_psr.h5"
    #instance.continuum_bg_estimation(data_file, data_yaml, psr_file, "estimated_bg", e_loop=(2,3), s_loop=(4,5))
