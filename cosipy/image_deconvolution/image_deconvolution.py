import astropy.units as u
import numpy as np

from histpy import Histogram, Axes
from mhealpy import HealpixMap

from astropy.coordinates import SkyCoord
from scoords import SpacecraftFrame, Attitude
from cosipy.response import FullDetectorResponse
from cosipy.config import Configurator

from .modelmap import ModelMap
from .RichardsonLucy import RichardsonLucy 
from .RichardsonLucy_memorysave import RichardsonLucy_memorysave

class ImageDeconvolution:
    _initial_model_map = None

    def __init__(self):
        self._use_sparse = False #not sure whether this implementation is good. Maybe it should be written in the parameter file?

    def set_data(self, data):

        self._data = data

        print("data for image deconvolution was set -> ", data)

    def read_parameterfile(self, parameter_filepath):

        self._parameter = Configurator.open(parameter_filepath)

        print("parameter file for image deconvolution was set -> ", parameter_filepath)

    @property
    def data(self):
        return self._data

    @property
    def parameter(self):
        return self._parameter

    def override_parameter(self, *args):
        self._parameter.override(args)

    @property
    def initial_model_map(self):
        if self._initial_model_map is None:
            print("warining: need the initialization")

        return self._initial_model_map

    @property
    def use_sparse(self):
        return self._use_sparse

    @use_sparse.setter
    def use_sparse(self, use_sparse):
        self._use_sparse = use_sparse

    def _check_model_response_consistency(self):
#        self._initial_model_map.axes["Ei"].axis_scale = self._data.image_response_dense.axes["Ei"].axis_scale
        self._initial_model_map.axes["Ei"].axis_scale = self._data.image_response_dense_projected.axes["Ei"].axis_scale

#        return self._initial_model_map.axes["lb"] == self._data.image_response_dense.axes["lb"] \
#               and self._initial_model_map.axes["Ei"] == self._data.image_response_dense.axes["Ei"]
        return self._initial_model_map.axes["lb"] == self._data.image_response_dense_projected.axes["lb"] \
               and self._initial_model_map.axes["Ei"] == self._data.image_response_dense_projected.axes["Ei"]

    def initialize(self):
        print("#### Initialization ####")
        
        print("1. generating a model map") 
        parameter_model_property = Configurator(self._parameter['model_property'])
        self._initial_model_map = ModelMap(self._data, parameter_model_property)

        print("---- parameters ----")
        print(parameter_model_property.dump())
        
        print("2. initializing the model map ...")
        parameter_model_initialization = Configurator(self._parameter['model_initialization'])
        self._initial_model_map.initialize(self._data, parameter_model_initialization)

        if not self._check_model_response_consistency():
            return

        print("---- parameters ----")
        print(parameter_model_initialization.dump())

        print("3. resistering the deconvolution algorithm ...")
        parameter_deconvolution = Configurator(self._parameter['deconvolution'])
        self._deconvolution = self.resister_deconvolution_algorithm(parameter_deconvolution)

        print("---- parameters ----")
        print(parameter_deconvolution.dump())

        print("#### Done ####")
        print("")

    def resister_deconvolution_algorithm(self, parameter_deconvolution):

        algorithm_name = parameter_deconvolution['algorithm']

        if algorithm_name == 'RL':
            parameter_RL = Configurator(parameter_deconvolution['parameter_RL'])
            _deconvolution = RichardsonLucy(self._initial_model_map, self._data, parameter_RL)
#        elif algorithm_name == 'MaxEnt':
#            parameter = self.parameter['deconvolution']['parameter_MaxEnt']
#            self.deconvolution == ...
        elif algorithm_name == 'RL_memsave':
            parameter_RL = Configurator(parameter_deconvolution['parameter_RL_memsave'])
            _deconvolution = RichardsonLucy_memorysave(self._initial_model_map, self._data, parameter_RL)

        _deconvolution.use_sparse = self._use_sparse #not sure whether this implementation is good. Maybe it should be written in the parameter file?

        return _deconvolution

    def run_deconvolution(self):
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

    def analyze_result(self):
        pass
