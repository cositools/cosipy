import pytest
from yayc import Configurator

from cosipy.image_deconvolution import RichardsonLucySimple, RichardsonLucy

def test_RicharsonLucySimple(dataset, model, mask, tmp_path):

    parameter = Configurator({"iteration_max": 2,
                              "minimum_flux": {"value": 0.0, "unit": "cm-2 s-1 sr-1"},
                              "background_normalization_optimization": {"activate": True, 
                                                                        "range": {"bkg": [0.9, 1.1]}},
                              "save_results": {"activate": True, "directory": f"{str(tmp_path)}", "only_final_result": True}
                              })

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
                              "acceleration": {"activate": True, "alpha_max": 10.0},
                              "response_weighting": {"activate": True, "index": 0.5},
                              "smoothing": {"activate": True, "FWHM": {"value": 2.0, "unit": "deg"}},
                              "background_normalization_optimization": {"activate": True, 
                                                                        "range": {"bkg": [0.9, 1.1]}},
                              "save_results": {"activate": True, "directory": f"{str(tmp_path)}", "only_final_result": True}
                              })

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
    parameter["acceleration:activate"] = False

    algorithm = RichardsonLucy(initial_model = model, 
                               dataset = dataset, 
                               mask = mask, 
                               parameter = parameter)

    algorithm.initialization()

    algorithm.iteration()
    algorithm.iteration()

    algorithm.finalization()
