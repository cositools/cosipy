import pytest

from cosipy.image_deconvolution import ModelBase

def test_model_base():
    ModelBase.__abstractmethods__ = set()

    model_base = ModelBase([0,1])

    with pytest.raises(RuntimeError) as e_info:
        ModelBase.instantiate_from_parameters({})
    assert e_info.type is NotImplementedError

    with pytest.raises(RuntimeError) as e_info:
        model_base.set_values_from_parameters({})
    assert e_info.type is NotImplementedError 
