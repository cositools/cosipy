import pytest
import numpy as np

from cosipy.image_deconvolution.algorithms.prior_base import PriorBase


def test_PriorBase():

    PriorBase.__abstractmethods__ = set()

    # Instantiation should fail when model class is not in usable_model_classes
    with pytest.raises(TypeError):
        PriorBase({'coefficient': 10}, np.zeros(2))

    # Allow np.ndarray for testing
    PriorBase.usable_model_classes.append(np.ndarray)

    prior = PriorBase({'coefficient': 10}, np.zeros(2))

    # Abstract methods should raise NotImplementedError
    with pytest.raises(NotImplementedError):
        prior.log_prior(np.zeros(2))

    with pytest.raises(NotImplementedError):
        prior.grad_log_prior(np.zeros(2))
