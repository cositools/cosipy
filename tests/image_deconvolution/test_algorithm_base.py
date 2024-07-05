import pytest
from yayc import Configurator

from cosipy.image_deconvolution import DeconvolutionAlgorithmBase

def test_deconvolution_algorithm_base(dataset, model, mask):

    DeconvolutionAlgorithmBase.__abstractmethods__ = set()

    parameter = Configurator({})

    algorithm = DeconvolutionAlgorithmBase(initial_model = model, 
                                           dataset = dataset, 
                                           mask = mask, 
                                           parameter = parameter)

    with pytest.raises(RuntimeError) as e_info:
        algorithm.initialization()
    assert e_info.type is NotImplementedError

    with pytest.raises(RuntimeError) as e_info:
        algorithm.pre_processing()
    assert e_info.type is NotImplementedError

    with pytest.raises(RuntimeError) as e_info:
        algorithm.Estep()
    assert e_info.type is NotImplementedError

    with pytest.raises(RuntimeError) as e_info:
        algorithm.Mstep()
    assert e_info.type is NotImplementedError

    with pytest.raises(RuntimeError) as e_info:
        algorithm.post_processing()
    assert e_info.type is NotImplementedError

    with pytest.raises(RuntimeError) as e_info:
        algorithm.register_result()
    assert e_info.type is NotImplementedError

    with pytest.raises(RuntimeError) as e_info:
        algorithm.check_stopping_criteria()
    assert e_info.type is NotImplementedError

    with pytest.raises(RuntimeError) as e_info:
        algorithm.finalization()
    assert e_info.type is NotImplementedError
