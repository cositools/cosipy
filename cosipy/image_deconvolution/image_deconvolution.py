import astropy.units as u
import numpy as np
import yaml

from histpy import Histogram, Axes
from mhealpy import HealpixMap

from astropy.coordinates import SkyCoord
from cosipy.coordinates import SpacecraftFrame, Attitude
from cosipy.response import FullDetectorResponse, HealpixAxis

from .modelmap import ModelMap
from .RichardsonLucy import RichardsonLucy 

class ImageDeconvolution:
    _initial_model_map = None

    def __init__(self):
        self._use_sparse = False #not sure whether this implementation is good. Maybe it should be written in the parameter file?

    def set_data(self, data): # to be replaced after the dataIO library is created.

        self._data = data

        print("data for image deconvolution was set -> ", data)

    def read_parameterfile(self, parameter_filepath):

        with open(parameter_filepath, "r") as f:
            self._parameter = yaml.safe_load(f)

        print("parameter file for image deconvolution was set -> ", parameter_filepath)

    @property
    def data(self):
        return self._data

    @property
    def parameter(self):
        return self._parameter

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

    def initialize(self):
        print("#### Initialization ####")
        
        print("1. generating a model map") 
        parameter_model_property = self._parameter['model_property']
        self._initial_model_map = ModelMap(self._data, parameter_model_property)

        print("---- parameters ----")
        print(yaml.dump(parameter_model_property))
        
        print("2. initializing the model map ...")
        parameter_model_initialization = self._parameter['model_initialization']
        self._initial_model_map.initialize(self._data, parameter_model_initialization)

        print("---- parameters ----")
        print(yaml.dump(parameter_model_initialization))

        print("3. resistering the deconvolution algorithm ...")
        parameter_deconvolution = self._parameter['deconvolution']
        self._deconvolution = self.resister_deconvolution_algorithm(parameter_deconvolution)

        print("---- parameters ----")
        print(yaml.dump(parameter_deconvolution))

        print("#### Done ####")
        print("")

    def resister_deconvolution_algorithm(self, parameter_deconvolution):

        algorithm_name = parameter_deconvolution['algorithm']

        if algorithm_name == 'RL':
            parameter_RL = parameter_deconvolution['parameter_RL']
            _deconvolution = RichardsonLucy(self._initial_model_map, self._data, parameter_RL)
#        elif algorithm_name == 'MaxEnt':
#            parameter = self.parameter['deconvolution']['parameter_MaxEnt']
#            self.deconvolution == ...

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
