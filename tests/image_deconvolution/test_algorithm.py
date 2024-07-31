import pytest
from yayc import Configurator

from cosipy.image_deconvolution import RichardsonLucySimple, RichardsonLucy

def test_RicharsonLucySimple(dataset, model, mask):

    parameter = Configurator({"iteration_max": 2,
                              "minimum_flux": {"value": 0.0, "unit": "cm-2 s-1 sr-1"},
                              "background_normalization_optimization": True})

    algorithm = RichardsonLucySimple(initial_model = model, 
                                     dataset = dataset, 
                                     mask = mask, 
                                     parameter = parameter)

    algorithm.initialization()

    algorithm.iteration()
    algorithm.iteration()

    algorithm.finalization()

def test_RicharsonLucy(dataset, model, mask, tmp_path):

    parameter = Configurator({"iteration_max": 2,
                              "minimum_flux": {"value": 0.0, "unit": "cm-2 s-1 sr-1"},
                              "acceleration": True,
                              "alpha_max": 10.0,
                              "response_weighting": True,
                              "response_weighting_index": 0.5,
                              "smoothing": True,
                              "smoothing_FWHM": {"value": 2.0, "unit": "deg"},
                              "background_normalization_optimization": True,
                              "background_normalization_range": {"bkg": [0.9, 1.1]},
                              "save_results": True,
                              "save_results_directory": f"{str(tmp_path)}/results"})

    # w/ acceleration
    algorithm = RichardsonLucy(initial_model = model, 
                               dataset = dataset, 
                               mask = mask, 
                               parameter = parameter)

    algorithm.initialization()

    algorithm.iteration()
    algorithm.iteration()

    algorithm.finalization()

    # wo/ acceleration and overwrite the directory
    parameter["acceleration"] = False

    algorithm = RichardsonLucy(initial_model = model, 
                               dataset = dataset, 
                               mask = mask, 
                               parameter = parameter)

    algorithm.initialization()

    algorithm.iteration()
    algorithm.iteration()

    algorithm.finalization()
