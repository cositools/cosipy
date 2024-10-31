import pytest
from cosipy.background_estimation import ContinuumEstimation
from cosipy import test_data

def test_continuum_background_estimation():
   
    instance = ContinuumEstimation() 
     
    # Test main method:
    #data_file = test_data.path / "bkg_pl.h5"
    data_file = "/project/majello/astrohe/ckarwin/COSI/COSIpy_Development/Continuum_BG_Estimation/Run_8/crab_bkg_binned_data_galactic.hdf5"
    data_yaml = test_data.path / "inputs_crab.yaml"
    #psr_file = test_data.path / "test_precomputed_response.h5"
    psr_file = "/project/majello/astrohe/ckarwin/COSI/COSIpy_Development/Continuum_BG_Estimation/Run_8/crab_psr.h5"
    psr = instance.load_psr_from_file(psr_file)
    instance.continuum_bg_estimation(data_file, data_yaml, psr, e_loop=(2,3), s_loop=(4,5))
