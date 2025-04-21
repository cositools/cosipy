from cosipy import test_data
from cosipy.response import ExtendedSourceResponse
from cosipy.image_deconvolution import AllSkyImageModel

from astromodels import Gaussian, Gaussian_on_sphere, ExtendedSource
from astromodels import *
import astropy.units as u 

extended_response_path = test_data.path/"test_precomputed_response.h5"

def test_open():
    resp = ExtendedSourceResponse.open(extended_response_path)

def test_get_expectation():

    resp = ExtendedSourceResponse.open(extended_response_path)

    nside = resp.axes['NuLambda'].nside
    energy_edges = resp.axes['Ei'].edges

    allsky_imagemodel = AllSkyImageModel(nside = nside, 
                                         energy_edges = energy_edges,
                                         label_image = 'NuLambda')

    hist = resp.get_expectation(allsky_imagemodel)

    assert isinstance(hist[:], u.quantity.Quantity) == True

def test_get_expectation_from_astromodel():

    resp = ExtendedSourceResponse.open(extended_response_path)

    # Define spectrum:
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

    # Define morphology:
    morphology = Gaussian_on_sphere(lon0 = 0, lat0 = 0, sigma = 5)

    # Define source:
    extended_model = ExtendedSource('gaussian', spectral_shape=spectrum, spatial_shape=morphology)
    
    # Calculate the expectation
    hist = resp.get_expectation_from_astromodel(extended_model)

    assert isinstance(hist[:], u.quantity.Quantity) == True

def test_get_expectation_from_astromodel_3d():

    resp = ExtendedSourceResponse.open(extended_response_path)

    # Upload source model:
    extended_model = load_model(test_data.path/'galprop_model.yaml')
    extended_model.galprop_source.spatial_shape._fitsfile = test_data.path/'ics_isotropic_healpix_54_0780000f.gz' 
    extended_model.galprop_source.spatial_shape.set_version(54)
    
    # Calculate the expectation
    hist = resp.get_expectation_from_astromodel(extended_model.galprop_source)

    assert isinstance(hist[:], u.quantity.Quantity) == True
