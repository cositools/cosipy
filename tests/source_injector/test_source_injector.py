from cosipy import SpacecraftFile, SourceInjector
from astropy.coordinates import SkyCoord
from threeML import Powerlaw
from pathlib import Path
import os
from cosipy import test_data
import numpy as np
import astropy.units as u
from histpy import Histogram
import pytest

def test_inject_point_source():

    # defind the response and orientation
    response_path = test_data.path / "test_full_detector_response.h5"
    orientation_path = test_data.path / "20280301_2s.ori"
    ori = SpacecraftFile.parse_from_file(orientation_path)
    
    # powerlaw model 
    index = -2.2
    K = 17 / u.cm / u.cm / u.s / u.keV
    piv = 1 * u.keV
    spectrum = Powerlaw()
    spectrum.index.value = index
    spectrum.K.value = K.value
    spectrum.piv.value= piv.value 
    spectrum.K.unit = K.unit
    spectrum.piv.unit = piv.unit
    
    # Define an injector by the response
    injector = SourceInjector(response_path = response_path)
    
    # Define the coordinate of the point source
    source_coord = SkyCoord(l = 50, b = -45, frame = "galactic", unit = "deg")
    
    # Get the data of the injected source
    injected_crab_signal = injector.inject_point_source(spectrum = spectrum, coordinate = source_coord, 
                                                        orientation = ori, source_name = "point_source",
                                                        make_spectrum_plot = False, data_save_path = None,
                                                        project_axes = None)

    results = injected_crab_signal.project("Em").to_dense().contents

    assert isinstance(results, u.quantity.Quantity) == True
    
    assert np.allclose(results.value,
                       [2.18846305e-03, 9.45773119e-03, 1.34892237e-02, 2.78741695e-03, 1.08413769e-02, 
                        6.28299687e-03, 3.63716712e-03, 1.43443841e-03,3.79135752e-04, 2.10058977e-05])



def test_inject_point_source_galactic():

    # defind the response and orientation
    response_path = test_data.path / "test_precomputed_response.h5"
    
    # powerlaw model 
    index = -2.2
    K = 17 / u.cm / u.cm / u.s / u.keV
    piv = 1 * u.keV
    spectrum = Powerlaw()
    spectrum.index.value = index
    spectrum.K.value = K.value
    spectrum.piv.value= piv.value 
    spectrum.K.unit = K.unit
    spectrum.piv.unit = piv.unit
    
    # Define an injector by the response
    injector = SourceInjector(response_path = response_path, response_frame = "galactic")
    
    # Define the coordinate of the point source
    source_coord = SkyCoord(l = 50, b = -45, frame = "galactic", unit = "deg")
    
    # Get the data of the injected source
    injected_crab_signal = injector.inject_point_source(spectrum = spectrum, coordinate = source_coord, 
                                                        source_name = "point_source",
                                                        make_spectrum_plot = True, data_save_path = None,
                                                        project_axes = None)

    results = injected_crab_signal.project("Em").to_dense().contents

    assert isinstance(results, u.quantity.Quantity) == True
    
    assert np.allclose(results.value,
                       [4.02116790e-03, 1.80171140e-02, 2.55344563e-02, 5.45316809e-03, 2.19219388e-02, 
                        1.50895341e-02, 9.97883729e-03, 4.16116828e-03, 1.02528085e-03, 6.26208604e-05])
    

def test_inject_point_source_saving():

    # defind the response and orientation
    response_path = test_data.path / "test_precomputed_response.h5"
    
    # powerlaw model 
    index = -2.2
    K = 17 / u.cm / u.cm / u.s / u.keV
    piv = 1 * u.keV
    spectrum = Powerlaw()
    spectrum.index.value = index
    spectrum.K.value = K.value
    spectrum.piv.value= piv.value 
    spectrum.K.unit = K.unit
    spectrum.piv.unit = piv.unit
    
    # Define an injector by the response
    injector = SourceInjector(response_path = response_path, response_frame = "galactic")
    
    # Define the coordinate of the point source
    source_coord = SkyCoord(l = 50, b = -45, frame = "galactic", unit = "deg")
    
    # Get the data of the injected source
    injected_crab_signal = injector.inject_point_source(spectrum = spectrum, coordinate = source_coord, 
                                                        source_name = "point_source",
                                                        make_spectrum_plot = False, data_save_path = Path("./galactic_rsp.h5"),
                                                        project_axes = "Em")
    
    hist= Histogram.open(Path("./galactic_rsp.h5"))
    
    os.remove(Path("./galactic_rsp.h5"))
    
    assert np.allclose(hist[:].value, 
                       [4.02116790e-03, 1.80171140e-02, 2.55344563e-02, 5.45316809e-03, 2.19219388e-02, 
                        1.50895341e-02, 9.97883729e-03, 4.16116828e-03, 1.02528085e-03, 6.26208604e-05])


def test_response_frame_error():

    # defind the response and orientation
    response_path = test_data.path / "test_precomputed_response.h5"
    
    with pytest.raises(ValueError):
        injector = SourceInjector(response_path = response_path, response_frame = "some_frame")
        

def test_orientation_error():

    # defind the response and orientation
    response_path = test_data.path / "test_full_detector_response.h5"
    
    # powerlaw model 
    index = -2.2
    K = 17 / u.cm / u.cm / u.s / u.keV
    piv = 1 * u.keV
    spectrum = Powerlaw()
    spectrum.index.value = index
    spectrum.K.value = K.value
    spectrum.piv.value= piv.value 
    spectrum.K.unit = K.unit
    spectrum.piv.unit = piv.unit
    
    # Define an injector by the response
    injector = SourceInjector(response_path = response_path)
    
    # Define the coordinate of the point source
    source_coord = SkyCoord(l = 50, b = -45, frame = "galactic", unit = "deg")

    with pytest.raises(TypeError):
    
        # Get the data of the injected source
        injected_crab_signal = injector.inject_point_source(spectrum = spectrum, coordinate = source_coord, 
                                                            source_name = "point_source",
                                                            make_spectrum_plot = False, data_save_path = None,
                                                            project_axes = None)





