import astropy.units as u
import numpy as np
import healpy as hp
import copy

import logging
logger = logging.getLogger(__name__)

from histpy import Histogram, Axes, Axis, HealpixAxis

from cosipy.response.functions import get_integrated_spectral_model

from .model_base import ModelBase

class AllSkyImageModel(ModelBase):
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
                 label_energy = 'Ei',
                 unit = '1/(s*cm*cm*sr)'
                 ):

        if energy_edges.unit != u.keV:

            logger.warning(f"The unit of the given energy_edges is {energy_edges.unit}. It will be converted to keV.")
            energy_edges = energy_edges.to('keV')

        image_axis = HealpixAxis(nside = nside,
                                 scheme = scheme,
                                 coordsys = coordsys,
                                 label = label_image)

        energy_axis = Axis(edges = energy_edges, label = label_energy, scale = "log")

        axes = Axes([image_axis, energy_axis])

        super().__init__(axes, sparse = False, unit = unit)

    @classmethod
    def open(cls, filename, name = 'hist'):
        """
        Open a file

        Parameters
        ----------
        filename: str
        
        Returns
        -------
        py:class:`AllSkyImageModel`
        """

        hist = Histogram.open(filename, name)

        allskyimage = AllSkyImageModel(nside = hist.axes[0].nside, 
                                    energy_edges = hist.axes[1].edges,
                                    scheme = hist.axes[0].scheme, 
                                    coordsys = hist.axes[0].coordsys.name, 
                                    label_image = hist.axes[0].label, 
                                    label_energy = hist.axes[1].label,
                                    unit = hist.unit)

        allskyimage[:] = hist.contents

        del hist
        return allskyimage

    @classmethod
    def instantiate_from_parameters(cls, parameter):
        """
        Generate an instance of this class

        Parameters
        ----------
        parameter : py:class:`yayc.Configurator`
        
        Returns
        -------
        py:class:`AllSkyImageModel`

        Notes
        -----
        The parameters should be given like this:

        nside: 8
        energy_edges:
            value: [100.,  200.,  500., 1000., 2000., 5000.]
            unit: "keV"
        scheme: "ring"
        coordinate: "galactic"
        unit: "cm-2 s-1 sr-1"

        """

        new = cls(nside = parameter['nside'],
                  energy_edges = parameter['energy_edges']['value'] * u.Unit(parameter['energy_edges']['unit']), 
                  scheme = parameter['scheme'], 
                  coordsys = parameter['coordinate'],
                  unit = u.Unit(parameter['unit']))

        return new

    def set_values_from_parameters(self, parameter):
        """
        Set the values accordinng to the specified algorithm. 

        Parameters
        ----------
        parameter : py:class:`yayc.Configurator`
            Parameters for the specified algorithm.

        Notes
        -----
        Currently algorithm_name can be only 'flat'. All of the pixel values in each energy bins will set to the given value.
        parameter should be {'values': [ flux value at 1st energy bin (without unit), flux value at 2nd energy bin, ...]}.

        An example of contents in parameter is like this:

        algorithm: "flat"
        parameter:
            value: [1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2]
            unit: "cm-2 s-1 sr-1"

        """

        algorithm_name = parameter['algorithm']
        algorithm_parameter = parameter['parameter']

        if algorithm_name == "flat":
            unit = u.Unit(algorithm_parameter['unit'])
            for idx, value in enumerate(algorithm_parameter['value']):
                self[:,idx] = value * unit
    #    elif algorithm_name == ... 
    #       ...

    def set_values_from_extendedmodel(self, extendedmodel):
        """
        Set the values accordinng to the given astromodels.ExtendedSource.

        Parameters
        ----------
        extendedmodel : :py:class:`astromodels.ExtendedSource`
            Extended source model
        """

        integrated_flux = get_integrated_spectral_model(spectrum = extendedmodel.spectrum.main.shape,
                                                        energy_axis = self.axes[1])
        
        npix = self.axes[0].npix
        coords = self.axes[0].pix2skycoord(np.arange(npix))

        normalized_map = extendedmodel.spatial_shape(coords.l.deg, coords.b.deg) / u.sr

        self[:] = np.tensordot(normalized_map, integrated_flux.contents, axes = 0) 

    def smoothing(self, fwhm = None, sigma = None):
        """
        Smooth a map with a Gaussian filter

        Parameters
        ----------
        fwhm: :py:class:`astropy.units.quantity.Quantity`
            The FWHM of the Gaussian (with a unit of deg or rad). Default: 0 deg
        sigma: :py:class:`astropy.units.quantity.Quantity`
            The sigma of the Gaussian (with a unit of deg or rad). Override fwhm.
        """

        if fwhm is None:
            fwhm = 0.0 * u.deg
        
        if sigma is not None:
            fwhm = 2.354820 * sigma

        allskyimage_new = copy.deepcopy(self)
        
        for i in range(self.axes['Ei'].nbins):
            allskyimage_new[:,i] = hp.smoothing(self[:,i].value, fwhm = fwhm.to('rad').value) * self.unit

        return allskyimage_new
