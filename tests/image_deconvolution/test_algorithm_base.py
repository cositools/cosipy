import pytest
from yayc import Configurator

from cosipy.image_deconvolution import DeconvolutionAlgorithmBase


def test_deconvolution_algorithm_base(dataset, model, mask):
    """All abstract methods should raise NotImplementedError."""

    DeconvolutionAlgorithmBase.__abstractmethods__ = set()

    algorithm = DeconvolutionAlgorithmBase(
        initial_model = model,
        dataset       = dataset,
        mask          = mask,
        parameter     = Configurator({}),
    )

    for method in [
        algorithm.initialization,
        algorithm.pre_processing,
        algorithm.processing_core,
        algorithm.post_processing,
        algorithm.register_result,
        algorithm.check_stopping_criteria,
        algorithm.finalization,
    ]:
        with pytest.raises(NotImplementedError):
            method()
