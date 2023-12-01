import astropy.units as u
import numpy as np

from histpy import Histogram, Axes, Axis, HealpixAxis

class ModelMap(Histogram):

    def __init__(self,
                 nside,
                 energy_edges,
                 scheme = 'ring',
                 coordsys = 'galactic',
                 ):

        if energy_edges.unit != u.keV:
            print("Warning (ModelMap): the unit of energy_edges is not keV!")

        image_axis = HealpixAxis(nside = nside,
                                 scheme = scheme,
                                 coordsys = coordsys,
                                 label = "lb")

        energy_axis = Axis(edges = energy_edges, label = "Ei", scale = "log")

        axes = Axes([image_axis, energy_axis])

        super().__init__(axes, sparse = False, unit = 1 / u.s / u.cm**2 / u.sr) # unit might be specified in the input parameter.

    def set_values_from_parameters(self, algorithm_name, parameter):

        if algorithm_name == "flat":
            for idx, value in enumerate(parameter['values']):
                self[:,idx:idx+1] = value * self.unit
    #    elif algorithm_name == ... 
    #       ...
