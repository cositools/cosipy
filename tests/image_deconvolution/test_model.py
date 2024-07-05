import pytest
import astropy.units as u
import numpy as np
from astromodels import Gaussian, Gaussian_on_sphere, ExtendedSource
from histpy import Histogram

from cosipy.image_deconvolution import AllSkyImageModel
from cosipy import test_data

def test_allskyimage():

    model = AllSkyImageModel(nside = 1, energy_edges = [100, 1000] * u.keV)
    
    # different energy unit
    model = AllSkyImageModel(nside = 1, energy_edges = [0.1, 1.0] * u.MeV)

    # open a file
    model = AllSkyImageModel.open(test_data.path / "image_deconvolution/all_sky_image_model_test_nside1.hdf5")

    # set values from parameter
    parameter = {"algorithm": "flat",
                 "parameter": {"value": [1.0],
                               "unit": "cm-2 s-1 sr-1"}}

    model.set_values_from_parameters(parameter)

    # instatiation from parameter
    parameter = {"nside": 1,
                 "energy_edges": {"value": [100.,  200.],
                                  "unit": "keV"},
                 "scheme": "ring",
                 "coordinate": "galactic",
                 "unit": "cm-2 s-1 sr-1"}

    model = AllSkyImageModel.instantiate_from_parameters(parameter)
    
    # smoothing
    model.smoothing(fwhm = 10.0 * u.deg)
    model.smoothing(sigma = 10.0 * u.deg)
    
    # mask
    mask = Histogram(model.axes, contents = np.zeros(model.axes.nbins, dtype = bool))

    model.mask_pixels(mask = mask, fill_value = 0)

    # set values from astromodels
    
    ### spectrum
    F = 4e-2 / u.cm / u.cm / u.s  
    mu = 511*u.keV
    sigma = 0.85*u.keV
    spectrum = Gaussian()
    spectrum.F.value = F.value
    spectrum.F.unit = F.unit
    spectrum.mu.value = mu.value
    spectrum.mu.unit = mu.unit
    spectrum.sigma.value = sigma.value
    spectrum.sigma.unit = sigma.unit
    
    ### morphology:
    morphology = Gaussian_on_sphere(lon0 = 359.75, lat0 = -1.25, sigma = 5)
    
    ### define source:
    src = ExtendedSource('gaussian', spectral_shape=spectrum, spatial_shape=morphology)
    
    ### set values 
    model.set_values_from_extendedmodel(src)
