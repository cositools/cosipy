import astropy.units as u
import numpy as np

from histpy import Histogram, Axes

class ModelMap(Histogram):
    def __init__(self, data, parameter):
        spherical_axis = data.image_response_mul_time.axes["NuLambda"]
        energy_axis = data.image_response_mul_time.axes["Ei"]

        axes = Axes([spherical_axis, energy_axis])

        Histogram.__init__(self, axes, unit = 1 / u.s / u.cm / u.cm / u.sr) # unit might be specified in the input parameter.

    def initialize(self, data, parameter):
        algorithm_name = parameter['algorithm']

        if algorithm_name == "flat":
            self[:] = parameter['parameter_flat']['value'] * self.unit
    #    elif algorithm_name == ... 
    #       ...
