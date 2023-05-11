import astropy.units as u
import numpy as np

from histpy import Histogram, Axes, Axis, HealpixAxis

class ModelMap(Histogram):
    def __init__(self, data, parameter):
        image_axis = HealpixAxis(nside = parameter["nside"],
                                     scheme = parameter["scheme"],
                                     coordsys = parameter["coordinate"],
                                     label = "lb")
        energy_axis = Axis(edges = parameter["energy_edges"] * u.keV, label = "Ei")

        axes = Axes([image_axis, energy_axis])

        Histogram.__init__(self, axes, unit = 1 / u.s / u.cm / u.cm / u.sr) # unit might be specified in the input parameter.

    def initialize(self, data, parameter):
        algorithm_name = parameter['algorithm']

        if algorithm_name == "flat":
            for idx, value in enumerate(parameter['parameter_flat']['values']):
                self[:,idx:idx+1] = value * self.unit
    #    elif algorithm_name == ... 
    #       ...
