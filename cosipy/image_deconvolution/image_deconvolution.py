from tqdm.autonotebook import tqdm

import logging
logger = logging.getLogger(__name__)

from yayc import Configurator

from .allskyimage import AllSkyImageModel

from .RichardsonLucy import RichardsonLucy
from .RichardsonLucySimple import RichardsonLucySimple

class ImageDeconvolution:
    """
    A class to reconstruct all-sky images from COSI data based on image deconvolution methods.
    """
    model_classes = {"AllSkyImage": AllSkyImageModel}
    deconvolution_algorithm_classes = {"RL": RichardsonLucy, "RLsimple": RichardsonLucySimple}

    def __init__(self):
        self._dataset = None
        self._initial_model = None
        self._mask = None
        self._parameter = None
        self._model_class = None
        self._deconvolution_class = None

    def set_dataset(self, dataset):
        """
        Set dataset

        Parameters
        ----------
        dataset : list of :py:class:`cosipy.image_deconvolution.ImageDeconvolutionDataInterfaceBase` or its subclass
            Each component contaning an event histogram, a background model, a response matrix, and a coordsys_conversion_matrix.
        """

        self._dataset = dataset
        
        logger.debug(f"dataset for image deconvolution was set -> {self._dataset}")

    def set_mask(self, mask):
        """
        Set dataset

        Parameters
        ----------
        mask: :py:class:`histpy.Histogram`
            A mask which will be applied to a model 
        """

        self._mask = mask

    def read_parameterfile(self, parameter_filepath):
        """
        Read parameters from a yaml file.

        Parameters
        ----------
        parameter_filepath : str or pathlib.Path
            Path of parameter file.
        """

        self._parameter = Configurator.open(parameter_filepath)

        logger.debug(f"parameter file for image deconvolution was set -> {parameter_filepath}")

    @property
    def dataset(self):
        """
        Return the dataset.
        """
        return self._dataset

    @property
    def parameter(self):
        """
        Return the registered parameter.
        """
        return self._parameter

    def override_parameter(self, *args):
        """
        Override parameter

        Parameters
        ----------
        *args
            new parameter

        Examples
        --------
        >>> image_deconvolution.override_parameter("deconvolution:parameter_RL:iteration = 30")
        """
        self._parameter.override(args)

    @property
    def initial_model(self):
        """
        Return the initial model.
        """
        if self._initial_model is None:
            logger.warning("Need to initialize model in the image_deconvolution instance!")

        return self._initial_model

    @property
    def mask(self):
        """
        Return the mask.
        """
        return self._mask

    @property
    def results(self):
        """
        Return the results.
        """
        return self._deconvolution.results

    def initialize(self):
        """
        Initialize an initial model and an image deconvolution algorithm.
        It is mandatory to execute this method before running the image deconvolution.
        """

        logger.info("#### Initialization Starts ####")
        
        self.model_initialization()        

        self.register_deconvolution_algorithm()        

        logger.info("#### Initialization Finished ####")

    def model_initialization(self):
        """
        Create an instance of the model class and set initial values of it.

        Returns
        -------
        bool 
            whether the instantiation and initialization are successfully done.
        """
        # set self._model_class
        model_name = self.parameter['model_definition']['class']

        if not model_name in self.model_classes.keys():
            logger.error(f'The model class "{model_name}" does not exist!')
            raise ValueError

        self._model_class = self.model_classes[model_name]

        # instantiate the model class
        logger.info(f"<< Instantiating the model class {model_name} >>")
        parameter_model_property = Configurator(self.parameter['model_definition']['property'])
        self._initial_model = self._model_class.instantiate_from_parameters(parameter_model_property)

        logger.info("---- parameters ----")
        logger.info(parameter_model_property.dump())

        # setting initial values
        logger.info("<< Setting initial values of the created model object >>")
        parameter_model_initialization = Configurator(self.parameter['model_definition']['initialization'])
        self._initial_model.set_values_from_parameters(parameter_model_initialization)

        # applying a mask to the model if needed
        if self.mask is not None:
            self._initial_model = self._initial_model.mask_pixels(self.mask, 0)

        # axes check
        if not self._check_model_response_consistency():
            logger.error("The model axes mismatches with the reponse in the dataset!")
            raise ValueError

        logger.info("---- parameters ----")
        logger.info(parameter_model_initialization.dump())

    def register_deconvolution_algorithm(self):
        """
        Register the deconvolution algorithm

        Returns
        -------
        bool 
            whether the deconvolution algorithm is successfully registered.
        """
        logger.info("<< Registering the deconvolution algorithm >>")
        parameter_deconvolution = Configurator(self.parameter['deconvolution'])

        algorithm_name = parameter_deconvolution['algorithm']
        algorithm_parameter = Configurator(parameter_deconvolution['parameter'])

        if not algorithm_name in self.deconvolution_algorithm_classes.keys():
            logger.error(f'The algorithm "{algorithm_name}" does not exist!')
            raise ValueError

        self._deconvolution_class = self.deconvolution_algorithm_classes[algorithm_name]
        self._deconvolution = self._deconvolution_class(initial_model = self.initial_model, 
                                                        dataset = self.dataset, 
                                                        mask = self.mask, 
                                                        parameter = algorithm_parameter)

        logger.info("---- parameters ----")
        logger.info(parameter_deconvolution.dump()) 

    def run_deconvolution(self):
        """
        Perform the image deconvolution. Make sure that the initialize method has been conducted.
        
        Returns
        -------
        list
            List containing results (reconstructed image, likelihood etc) at each iteration. 
        """
        logger.info("#### Image Deconvolution Starts ####")
       
        logger.info(f"<< Initialization >>")
        self._deconvolution.initialization()
        
        stop_iteration = False
        for i in tqdm(range(self._deconvolution.iteration_max)):
            if stop_iteration:
                break
            stop_iteration = self._deconvolution.iteration()

        logger.info(f"<< Finalization >>")
        self._deconvolution.finalization()

        logger.info("#### Image Deconvolution Finished ####")

    def _check_model_response_consistency(self):
        """
        Check if the model axes is consistent with the dataset

        Returns
        -------
        bool 
            whether the axes of dataset are consistent with the model.
        """
        
        for data in self.dataset:
            if data.model_axes != self.initial_model.axes:
                return False
        return True
