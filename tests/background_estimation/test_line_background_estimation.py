import pytest
from histpy import Histogram
import astropy.units as u
import numpy as np

from cosipy import LineBackgroundEstimation, test_data

def test_line_background_estimation():
    
    # prepare data
    data_path = test_data.path / "test_event_histogram_galacticCDS.hdf5"

    data = Histogram.open(data_path)
    data = data.project(['Em', 'Phi', 'PsiChi'])

    # prepare model
    def bkg_model(x, a, b):
        pivot = 1000.0
        return a * (x/pivot)**b

    # instantiate the line background class
    instance = LineBackgroundEstimation(data)
    
    # set background spectrum model
    instance.set_bkg_energy_spectrum_model(bkg_model, [1.0, -3.0])

    # set mask
    instance.set_mask((0.0, 1000.0) * u.keV, (3000.0, 5000.0) * u.keV)

    # run fitting
    m = instance.fit_energy_spectrum()

    # run plotting
    ax, _ = instance.plot_energy_spectrum()
    
    # set range for source region
    source_range = (2000.0, 2500.0)  * u.keV

    # generate background model
    
    ## Case 1: a single extracting region 
    background_region = (1120.0, 1650.0) * u.keV
    
    bkg_model_histogram = instance.generate_bkg_model_histogram(source_range, [background_region])

    ### check sum
    assert np.sum(bkg_model_histogram) == 41.61181341324655

    ## Case 2: a single extracting region broader than the actual bin width
    background_region = (1119.0, 1651.0) * u.keV
    
    bkg_model_histogram = instance.generate_bkg_model_histogram(source_range, [background_region])

    ## Case 3: a extracting region is too narrow. Check error
    background_region = (1121.0, 1121.001) * u.keV
    
    with pytest.raises(ValueError) as e_info:
        bkg_model_histogram = instance.generate_bkg_model_histogram(source_range, [background_region])

    ## Case 4: two extracting regions
    background_region_1 = (1120.0, 1650.0) * u.keV #background counts estimation before the line
    background_region_2 = (3450.0, 5000.0) * u.keV #background counts estimation before the line
    
    bkg_model_histogram = instance.generate_bkg_model_histogram(source_range, [background_region_1, background_region_2])
