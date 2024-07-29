import pytest

from cosipy.image_deconvolution import ImageDeconvolution
from cosipy import test_data

@pytest.fixture
def parameter_filepath():

    return test_data.path / "image_deconvolution" / "imagedeconvolution_parfile_test.yml"

def test_image_deconvolution(dataset, model, mask, parameter_filepath):

    image_deconvolution = ImageDeconvolution()

    # set dataloader to image_deconvolution
    image_deconvolution.set_dataset(dataset)
    
    # set a parameter file for the image deconvolution
    image_deconvolution.read_parameterfile(parameter_filepath)

    # set a mask
    image_deconvolution.set_mask(mask)

    # initialization
    assert image_deconvolution.initial_model is None

    image_deconvolution.initialize()

    assert image_deconvolution.initial_model is not None
    
    # run
    image_deconvolution.run_deconvolution()

    # get results
    results = image_deconvolution.results

def test_image_deconvolution_override_parameter(dataset, model, mask, parameter_filepath):

    image_deconvolution = ImageDeconvolution()

    # set dataloader to image_deconvolution
    image_deconvolution.set_dataset(dataset)
    
    # set a parameter file for the image deconvolution
    image_deconvolution.read_parameterfile(parameter_filepath)

    # set a mask
    image_deconvolution.set_mask(mask)

    # override a parameter
    image_deconvolution.override_parameter("deconvolution:parameter:iteration_max = 1")

    # override a wrong parameter 
    ## model
    image_deconvolution.read_parameterfile(parameter_filepath)

    image_deconvolution.override_parameter('model_definition:class = "WrongName"')

    with pytest.raises(ValueError) as e_info:
        image_deconvolution.initialize()

    ## algorithm
    image_deconvolution.read_parameterfile(parameter_filepath)

    image_deconvolution.override_parameter('deconvolution:algorithm = "WrongName"')

    with pytest.raises(ValueError) as e_info:
        image_deconvolution.initialize()
