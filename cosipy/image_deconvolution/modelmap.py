import astropy.units as u
import numpy as np

from histpy import Histogram, Axes, Axis, HealpixAxis
import astropy.units as u

from cosipy.threeml.custom_functions import get_integrated_spectral_model

class ModelMap(Histogram):

    def __init__(self,
                 nside,
                 energy_edges,
                 scheme = 'ring',
                 coordsys = 'galactic',
                 label_image = 'lb',
                 label_energy = 'Ei'
                 ):

        if energy_edges.unit != u.keV:
            print("Warning (ModelMap): the unit of energy_edges is not keV!")

        self.image_axis = HealpixAxis(nside = nside,
                                      scheme = scheme,
                                      coordsys = coordsys,
                                      label = label_image)

        self.energy_axis = Axis(edges = energy_edges, label = label_energy, scale = "log")

        axes = Axes([self.image_axis, self.energy_axis])

        super().__init__(axes, sparse = False, unit = 1 / u.s / u.cm**2 / u.sr) # unit might be specified in the input parameter.

    def set_values_from_parameters(self, algorithm_name, parameter):

        if algorithm_name == "flat":
            for idx, value in enumerate(parameter['values']):
                self[:,idx:idx+1] = value * self.unit
    #    elif algorithm_name == ... 
    #       ...

    def set_values_from_extendedmodel(self, extendedmodel):

        integrated_flux = get_integrated_spectral_model(spectrum = extendedmodel.spectrum.main.shape,
                                                        eaxis = self.energy_axis)
        
        npix = self.image_axis.npix
        coords = self.image_axis.pix2skycoord(np.arange(npix))

        normalized_map = extendedmodel.spatial_shape(coords.l.deg, coords.b.deg) / u.sr

        self[:] = np.tensordot(normalized_map, integrated_flux.contents, axes = 0) 
