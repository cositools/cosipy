import pytest

from cosipy.image_deconvolution import ImageDeconvolutionDataInterfaceBase

def test_ImageDeconvolutionDataInterfaceBase():
    ImageDeconvolutionDataInterfaceBase.__abstractmethods__ = set()

    data = ImageDeconvolutionDataInterfaceBase("dataname")

    assert data.name == "dataname"
    assert data.event is None
    assert data.exposure_map is None
    assert data.model_axes is None
    assert data.data_axes is None

    data._bkg_models = {"test": None}
    assert data.keys_bkg_models() == ["test"]
    assert data.bkg_model("test") is None

    with pytest.raises(RuntimeError) as e_info:
        data.calc_expectation(model = None, dict_bkg_norm = {"test": 1})
    assert e_info.type is NotImplementedError

    with pytest.raises(RuntimeError) as e_info:
        data.calc_T_product(dataspace_histogram = None)
    assert e_info.type is NotImplementedError

    with pytest.raises(RuntimeError) as e_info:
        data.calc_bkg_model_product(key = "test", dataspace_histogram = None)
    assert e_info.type is NotImplementedError

    with pytest.raises(RuntimeError) as e_info:
        data.calc_log_likelihood(expectation = None)
    assert e_info.type is NotImplementedError
