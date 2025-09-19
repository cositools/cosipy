import pytest
import numpy as np

from cosipy.image_deconvolution.prior_base import PriorBase

def test_PriorBase():
    PriorBase.__abstractmethods__ = set()
    
    # no class is allowered
    with pytest.raises(TypeError) as e_info:
        coefficient = 10
        test_model = np.zeros(2)
        prior = PriorBase(coefficient, test_model)
    
    # As a test, np.ndarray is added
    PriorBase.usable_model_classes.append(np.ndarray)

    coefficient = 10
    test_model = np.zeros(2)
    prior = PriorBase(coefficient, test_model)
    
    # other function tests
    with pytest.raises(RuntimeError) as e_info:
        prior.log_prior(test_model)
    assert e_info.type is NotImplementedError

    with pytest.raises(RuntimeError) as e_info:
        prior.grad_log_prior(test_model)
    assert e_info.type is NotImplementedError
