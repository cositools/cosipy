import warnings
import astropy.units as u
import numpy as np
import healpy as hp
import copy

from histpy import Histogram, Axes, Axis, HealpixAxis

from cosipy.threeml.custom_functions import get_integrated_spectral_model

class ModelMap(Histogram):
    """
    Photon flux maps in given energy bands. 2-dimensional histogram.

    Attributes
    ----------
    nside : int
        Healpix NSIDE parameter.
    energy_edges : :py:class:`np.array`
        Bin edges for energies. We recommend to use a Quantity array with the unit of keV.
    scheme : str, default 'ring'
        Healpix scheme. Either 'ring', 'nested'.
    coordsys : str or :py:class:`astropy.coordinates.BaseRepresentation`, default is 'galactic'
        Instrinsic coordinates of the map. The default is 'galactic'.
    label_image : str, default 'lb'
        The label name of the healpix axis.
    label_energy : str, default 'Ei'
        The label name of the energy axis. The default is 'Ei'.
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

    @classmethod
    def open(cls, filename, name = 'hist'):

        hist = Histogram.open(filename, name)

        modelmap = ModelMap(nside = hist.axes[0].nside, 
                            energy_edges = hist.axes[1].edges,
                            scheme = hist.axes[0].scheme, 
                            coordsys = hist.axes[0].coordsys.name, 
                            label_image = hist.axes[0].label, 
                            label_energy = hist.axes[1].label)

        modelmap[:] = hist.contents

        del hist
        return modelmap

    def set_values_from_parameters(self, algorithm_name, parameter):
        """
        Set the values of this model map accordinng to the specified algorithm. 

        Parameters
        ----------
        algorithm_name : str
            Algorithm name to fill the values.  
        parameter : py:class:`cosipy.config.Configurator`
            Parameters for the specified algorithm.

        Notes
        -----
        Currently algorithm_name can be only 'flat'. All of the pixel values in each energy bins will set to the given value.
        parameter should be {'values': [ flux value at 1st energy bin (without unit), flux value at 2nd energy bin, ...]}.
        """

        if algorithm_name == "flat":
            for idx, value in enumerate(parameter['values']):
                self[:,idx:idx+1] = value * self.unit
    #    elif algorithm_name == ... 
    #       ...

    def set_values_from_extendedmodel(self, extendedmodel):
        """
        Set the values of this model map accordinng to the given astromodels.ExtendedSource.

        Parameters
        ----------
        extendedmodel : :py:class:`astromodels.ExtendedSource`
            Extended source model
        """

        integrated_flux = get_integrated_spectral_model(spectrum = extendedmodel.spectrum.main.shape,
                                                        eaxis = self.energy_axis)
        
        npix = self.image_axis.npix
        coords = self.image_axis.pix2skycoord(np.arange(npix))

        normalized_map = extendedmodel.spatial_shape(coords.l.deg, coords.b.deg) / u.sr

        self[:] = np.tensordot(normalized_map, integrated_flux.contents, axes = 0) 

    def mask_pixels(self, mask, fill_value = 0):
        """
        Mask pixels

        Parameters
        ----------
        mask: :py:class:`histpy.histogram.Histogram`
        fill_value: float or :py:class:`astropy.units.quantity.Quantity`
        """

        if not isinstance(fill_value, u.quantity.Quantity):
            fill_value *= self.contents.unit

        self[:] = np.where(mask.contents, self.contents, fill_value)

    def smoothing(self, fwhm = 0.0 * u.deg, sigma = None):
        """
        Smooth a map with a Gaussian filter

        Parameters
        ----------
        fwhm: :py:class:`astropy.units.quantity.Quantity`
            The FWHM of the Gaussian (with a unit of deg or rad).
        sigma: :py:class:`astropy.units.quantity.Quantity`
            The sigma of the Gaussian (with a unit of deg or rad). Override fwhm.
        """

        if sigma is not None:
            fwhm = 2.354820 * sigma

        modelmap_new = copy.deepcopy(self)
        
        for i in range(self.axes['Ei'].nbins):
            modelmap_new[:,i] = hp.smoothing(self[:,i].value, fwhm = fwhm.to('rad').value) * self.unit

        return modelmap_new
