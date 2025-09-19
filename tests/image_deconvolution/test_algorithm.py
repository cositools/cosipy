import pytest
import numpy as np
from yayc import Configurator

from cosipy.image_deconvolution import RichardsonLucySimple, RichardsonLucy, MAP_RichardsonLucy

def test_RicharsonLucySimple(dataset, model, mask, tmp_path):

    num_iteration = 3

    parameter = Configurator({"iteration_max": num_iteration,
                              "response_weighting": {"activate": True, "index": 0.5},
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

    for i in range(num_iteration):
        stop = algorithm.iteration()
        if stop:
            break

    algorithm.finalization()

def test_RicharsonLucy(dataset, model, mask, tmp_path):

    num_iteration = 3

    parameter = Configurator({"iteration_max": num_iteration,
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

    for i in range(num_iteration):
        stop = algorithm.iteration()
        if stop:
            break

    algorithm.finalization()

    assert np.isclose(algorithm.results[-1]['log-likelihood'][0], 5495.120521335304)

    # wo/ acceleration and overwrite the directory
    parameter["acceleration:activate"] = False

    algorithm = RichardsonLucy(initial_model = model,
                               dataset = dataset,
                               mask = mask,
                               parameter = parameter)

    algorithm.initialization()

    for i in range(num_iteration):
        stop = algorithm.iteration()
        if stop:
            break

    algorithm.finalization()

    assert np.isclose(algorithm.results[-1]['log-likelihood'][0], 5270.562770130176)

def test_MAP_RichardsonLucy(dataset, model, mask, tmp_path):

    num_iteration = 10

    parameter = Configurator({"iteration_max": num_iteration,
                              "minimum_flux": {"value": 0.0, "unit": "cm-2 s-1 sr-1"},
                              "response_weighting": {"activate": True, "index": 0.5},
                              "background_normalization_optimization": {"activate": True,
                                                                        "range": {"bkg": [0.9, 1.1]}},
                              "stopping_criteria": {"statistics": "log-posterior",
                                                    "threshold": 1e-2},
                              "prior": {"TSV"  :{"coefficient": 1.e-10},
                                        "gamma":{"model":{"theta": {"value": np.inf, "unit": "cm-2 s-1 sr-1"},
                                                          "k": {"value": 0.999}},
                                                 "background": {"theta": {"value": np.inf}, "k": {"value": 1.0}}
                                                 }
                                        },
                              "save_results": {"activate": True, "directory": f"{str(tmp_path)}", "only_final_result": True}
                              })

    # first run
    algorithm = MAP_RichardsonLucy(initial_model = model,
                                   dataset = dataset,
                                   mask = mask,
                                   parameter = parameter)

    algorithm.initialization()

    for i in range(num_iteration):
        stop = algorithm.iteration()
        if stop:
            break

    algorithm.finalization()

    assert np.isclose(algorithm.results[-1]['log-posterior'], 6567.857548203495)

    # background fixed
    parameter["background_normalization_optimization:activate"] = False

    algorithm = MAP_RichardsonLucy(initial_model = model,
                                   dataset = dataset,
                                   mask = mask,
                                   parameter = parameter)

    algorithm.initialization()

    for i in range(num_iteration):
        stop = algorithm.iteration()
        if stop:
            break

    algorithm.finalization()

    assert np.isclose(algorithm.results[-1]['log-posterior'], 6202.336733778631)

    # with large threshold
    parameter["stopping_criteria:threshold"] = 1e10

    algorithm = MAP_RichardsonLucy(initial_model = model,
                                   dataset = dataset,
                                   mask = mask,
                                   parameter = parameter)

    algorithm.initialization()

    for i in range(num_iteration):
        stop = algorithm.iteration()
        if stop:
            break

    algorithm.finalization()

    assert len(algorithm.results) == 2

    # with log-likelihood
    parameter["stopping_criteria:statistics"] = "log-likelihood"

    algorithm = MAP_RichardsonLucy(initial_model = model,
                                   dataset = dataset,
                                   mask = mask,
                                   parameter = parameter)

    algorithm.initialization()

    for i in range(num_iteration):
        stop = algorithm.iteration()
        if stop:
            break

    algorithm.finalization()

    assert np.isclose(algorithm.results[-1]['log-likelihood'][0], 3931.3528198012773)

    # without gamma prior
    parameter["stopping_criteria:statistics"] = 'log-posterior'
    parameter["prior"] = {"TSV":{"coefficient": 1.e-10}}

    algorithm = MAP_RichardsonLucy(initial_model = model,
                                   dataset = dataset,
                                   mask = mask,
                                   parameter = parameter)

    algorithm.initialization()

    for i in range(num_iteration):
        stop = algorithm.iteration()
        if stop:
            break

    algorithm.finalization()

    assert np.isclose(algorithm.results[-1]['log-posterior'], 3931.29811442966)

    # wrong statistics
    parameter["stopping_criteria:statistics"] = "likelihooooooooood!!!"

    with pytest.raises(ValueError) as e_info:
        algorithm = MAP_RichardsonLucy(initial_model = model,
                                       dataset = dataset,
                                       mask = mask,
                                       parameter = parameter)
