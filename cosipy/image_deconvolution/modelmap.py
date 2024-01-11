import astropy.units as u
import numpy as np
import warnings

from histpy import Histogram, Axes, Axis, HealpixAxis

from cosipy.threeml.custom_functions import get_integrated_spectral_model

class ModelMap(Histogram):
    """
    Photon flux maps in given energy bands. 2-dimensional histogram.

    Args:
        nside (int): Healpix NSIDE parameter.
        energy_edges (array): Bin edges for energies. We recommend to use a Quantity array with the unit of keV.
        scheme (str, optional): Healpix scheme. Either 'ring', 'nested'. The default is 'ring'.
        coordsys (BaseFrameRepresentation or str, optional): Instrinsic coordinates of the map. The default is 'galactic'.
        label_image (str, optional): The label name of the healpix axis. The default is 'lb'.
        label_energy (str, optional): The label name of the energy axis. The default is 'Ei'.
    """

    def __init__(self,
                 nside,
                 energy_edges,
                 scheme = 'ring',
                 coordsys = 'galactic',
                 label_image = 'lb',
                 label_energy = 'Ei'
                 ):

        if energy_edges.unit != u.keV:

            warnings.warn(f"The unit of the given energy_edges is {energy_edges.unit}. It is converted to keV.")
            energy_edges = energy_edges.to('keV')

        self.image_axis = HealpixAxis(nside = nside,
                                      scheme = scheme,
                                      coordsys = coordsys,
                                      label = label_image)

        self.energy_axis = Axis(edges = energy_edges, label = label_energy, scale = "log")

        axes = Axes([self.image_axis, self.energy_axis])

        super().__init__(axes, sparse = False, unit = 1 / u.s / u.cm**2 / u.sr) # unit might be specified in the input parameter.

    def set_values_from_parameters(self, algorithm_name, parameter):
        """
        Set the values of this model map accordinng to the specified algorithm. 

        Args:
            algorithm_name (str): Algorithm name to fill the values.  
            parameter (dict): Parameters for the specified algorithm.

        Notes:
            Currently algorithm_name can be only 'flat'. All of the pixel values in each energy bins will set to the given value.
            parameter should be {'values': [ flux value at 1st energy bin (without unit), flux value at 2nd energy bin, ...]}
        """

        if algorithm_name == "flat":
            for idx, value in enumerate(parameter['values']):
                self[:,idx:idx+1] = value * self.unit
    #    elif algorithm_name == ... 
    #       ...

    def set_values_from_extendedmodel(self, extendedmodel):
        """
        Set the values of this model map accordinng to the given astromodels.ExtendedSource.

        Args:
            extendedmodel (astromodels.ExtendedSource): the extended source model.
        """

        integrated_flux = get_integrated_spectral_model(spectrum = extendedmodel.spectrum.main.shape,
                                                        eaxis = self.energy_axis)
        
        npix = self.image_axis.npix
        coords = self.image_axis.pix2skycoord(np.arange(npix))

        normalized_map = extendedmodel.spatial_shape(coords.l.deg, coords.b.deg) / u.sr

        self[:] = np.tensordot(normalized_map, integrated_flux.contents, axes = 0) 
