import warnings
import astropy.units as u

from cosipy.config import Configurator

from .modelmap import ModelMap
# import image deconvolution algorithms
from .RichardsonLucy import RichardsonLucy

class ImageDeconvolution:
    """
    A class to reconstruct all-sky images from COSI data based on image deconvolution methods.
    """

    def __init__(self):
        self._initial_model_map = None

    def set_data(self, data):
        """
        Set COSI dataset

        Parameters
        ----------
        data : :py:class:`cosipy.image_deconvolution.DataLoader`
            Data loader contaning an event histogram, a background model, a response matrix, and a coordsys_conversion_matrix.

        Notes
        -----
        cosipy.image_deconvolution.DataLoader may be removed in the future once the formats of event/background/response are fixed.
        In this case, this method will be also modified in the future.
        """

        self._data = data
        
        print("data for image deconvolution was set -> ", data)

    def read_parameterfile(self, parameter_filepath):
        """
        Read parameters from a yaml file.

        Parameters
        ----------
        parameter_filepath : str or pathlib.Path
            Path of parameter file.
        """

        self._parameter = Configurator.open(parameter_filepath)

        print("parameter file for image deconvolution was set -> ", parameter_filepath)

    @property
    def data(self):
        """
        Return the set data.
        """
        return self._data

    @property
    def parameter(self):
        """
        Return the set parameter.
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
    def initial_model_map(self):
        """
        Return the initial model map.
        """
        if self._initial_model_map is None:
            warnings.warn("Need to initialize model map in the image_deconvolution instance")

        return self._initial_model_map

    def _check_model_response_consistency(self):
        """
        Check whether the axes of model map are consistent with those of the response matrix.

        Returns
        -------
        bool
            If True, their axes are consistent with each other.

        Notes
        -----
        It will be implemented in the future. Currently it always returns true.
        """
        #self._initial_model_map.axes["Ei"].axis_scale = self._data.image_response_dense_projected.axes["Ei"].axis_scale

        #return self._initial_model_map.axes["lb"] == self._data.image_response_dense_projected.axes["lb"] \
        #       and self._initial_model_map.axes["Ei"] == self._data.image_response_dense_projected.axes["Ei"]
        return True

    def initialize(self):
        """
        Initialize an image_deconvolution instance. It is mandatory to execute this method before running the image deconvolution.

        This method has three steps:
        1. generate a model map with properties (nside, energy bins, etc.) given in the parameter file.
        2. initialize a model map following an initial condition given in the parameter file
        3. load parameters for the image deconvolution
        """

        print("#### Initialization ####")

        ### check self._data ###
        ### this part will be removed in the future ###
        if self._data.response_on_memory == False:

            warnings.warn("In the image deconvolution, the option to not load the response on memory is currently not supported. Performing DataLoader.load_full_detector_response_on_memory().")
            self._data.load_full_detector_response_on_memory()

        if self._data.image_response_dense_projected is None:

            warnings.warn("The image_response_dense_projected has not been calculated. Performing DataLoader.calc_image_response_projected().")
            self._data.calc_image_response_projected()
        
        print("1. generating a model map") 
        parameter_model_property = Configurator(self._parameter['model_property'])
        self._initial_model_map = ModelMap(nside = parameter_model_property['nside'],
                                           energy_edges = parameter_model_property['energy_edges'] * u.keV, 
                                           scheme = parameter_model_property['scheme'], 
                                           coordsys = parameter_model_property['coordinate'])

        print("---- parameters ----")
        print(parameter_model_property.dump())
        
        print("2. initializing the model map ...")
        parameter_model_initialization = Configurator(self._parameter['model_initialization'])

        algorithm_name = parameter_model_initialization['algorithm']

        self._initial_model_map.set_values_from_parameters(algorithm_name, 
                                                           parameter_model_initialization['parameter_'+algorithm_name])

        if not self._check_model_response_consistency():
            return

        print("---- parameters ----")
        print(parameter_model_initialization.dump())

        print("3. registering the deconvolution algorithm ...")
        parameter_deconvolution = Configurator(self._parameter['deconvolution'])
        self._deconvolution = self.register_deconvolution_algorithm(parameter_deconvolution)

        print("---- parameters ----")
        print(parameter_deconvolution.dump())

        print("#### Done ####")
        print("")

    def register_deconvolution_algorithm(self, parameter_deconvolution):
        """
        Register parameters for image deconvolution on a deconvolution instance.

        Parameters
        ----------
        parameter_deconvolution : :py:class:`cosipy.config.Configurator`
            Parameters for the image deconvolution methods.

        Notes
        -----
        Currently only RichardsonLucy algorithm is implemented.

        ***An example of parameters for RL algorithm***
        algorithm: "RL"
        parameter_RL:
            iteration: 10 
            # number of iterations
            acceleration: True 
            # whether the accelerated ML-EM algorithm (Knoedlseder+99) is used
            alpha_max: 10.0 
            # the maximum value for the acceleration alpha parameter
            save_results_each_iteration: False 
            # whether a updated model map, detal map, likelihood etc. are save at the end of each iteration
            response_weighting: True 
            # whether a factor $w_j = (\sum_{i} R_{ij})^{\beta}$ for weighting the delta image is introduced 
            # see Knoedlseder+05, Siegert+20
            response_weighting_index: 0.5 
            # $\beta$ in the above equation
            smoothing: True 
            # whether a Gaussian filter is used (see Knoedlseder+05, Siegert+20)
            smoothing_FWHM: 2.0 #deg 
            # the FWHM of the Gaussian in the filter 
            background_normalization_fitting: False 
            # whether the background normalization is optimized at each iteration. 
            # As for now, the same single background normalization factor is used in all of the time bins
            background_normalization_range: [0.01, 10.0]
            # the range of the normalization factor. it should be positive.
        """

        algorithm_name = parameter_deconvolution['algorithm']

        if algorithm_name == 'RL':
            parameter_RL = Configurator(parameter_deconvolution['parameter_RL'])
            _deconvolution = RichardsonLucy(self._initial_model_map, self._data, parameter_RL)
#        elif algorithm_name == 'MaxEnt':
#            parameter = self.parameter['deconvolution']['parameter_MaxEnt']
#            self.deconvolution == ...

        return _deconvolution

    def run_deconvolution(self):
        """
        Perform the image deconvolution. Make sure that the initialize method has been conducted.
        
        Returns
        -------
        list
            List containing results (reconstructed image, likelihood etc) at each iteration. 
        """
        print("#### Deconvolution Starts ####")
        
        all_result = []
        for result in self._deconvolution.iteration():
            all_result.append(result)
            ### can perform intermediate check ###
            #...
            ###

        print("#### Done ####")
        print("")
        return all_result

#    def analyze_result(self):
#        pass
