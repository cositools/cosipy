import logging
logger = logging.getLogger(__name__)

import numpy as np

from copy import deepcopy

from histpy import Histogram, Axes, Axis

import astropy.units as u

class DetectorResponse(Histogram):
    """
    Handles the multi-dimensional matrix that describes the
    response of the instrument for a particular :py:class:`.SpacecraftFrame` coordinate
    location.

    Parameters
    ----------
    axes : :py:class:`histpy.Axes`
        Binning information for each variable. The following labels are expected:\n
        - ``Ei``: Real energy
        - ``Em``: Measured energy
        - ``Phi``: Compton angle. Optional.
        - ``PsiChi``:  Location in the Compton Data Space (HEALPix pixel). Optional.
        - ``SigmaTau``: Electron recoil angle (HEALPix pixel). Optional.
        - ``Dist``: Distance from first interaction. Optional.
    contents : array, :py:class:`astropy.units.Quantity` or :py:class:`sparse.SparseArray`
        Array containing the differential effective area.
    unit : :py:class:`astropy.units.Unit`, optional
        Physical area units, if not specified as part of ``contents``
    """

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self._spec = None
        self._aeff = None
    
    def get_spectral_response(self, copy = True):
        """
        Reduced detector response, projected along the real and measured energy axes only.
        The Compton Data Space axes are not included.

        Parameters
        ----------
        copy : bool
            If true, a copy of the cached spectral response will be returned.
        
        Returns
        -------
        :py:class:`DetectorResponse`
        """

        # Cache the spectral response
        if self._spec is None:
            spec = self.project(['Ei','Em'])
            self._spec = DetectorResponse(spec.axes,
                                          contents = spec.contents,
                                          unit = spec.unit)

        if copy:
            return deepcopy(self._spec)
        else:
            return self._spec
        
    def get_effective_area(self, energy = None, copy = True):
        """
        Compute the effective area at a given energy. If no energy is specified, the
        output is a histogram for the effective area at each energy bin.

        Parameters
        ----------
        energy : optional, :py:class:`astropy.units.Quantity`
            Energy/energies at which to interpolate the linearly effective area
        copy : bool
            If true, a copy of the cached effective will be returned.
        
        Returns
        -------
        :py:class:`astropy.units.Quantity` or :py:class:`histpy.Histogram`
        """
        
        if self._aeff is None:
            self._aeff = self.get_spectral_response(copy = False).project('Ei').to_dense()

        if energy is None:
            if copy:
                return deepcopy(self._aeff)
            else:
                return self._aeff
        else:
            return self._aeff.interp(energy)

    def get_dispersion_matrix(self):
        """
        Compute the energy dispersion matrix, also known as migration matrix. This holds the
        probability of an event with real energy ``Ei`` to be reconstructed with an measured
        energy ``Em``.

        Returns
        -------
        :py:class:`histpy.Histogram`
        """
        
        # Get spectral response and effective area normalization
        spec = self.get_spectral_response(copy = False)
        norm = self.get_effective_area().full_contents

        # Hack the under/overflow bins to supress 0/0 wearning
        norm[0] = 1*norm.unit if norm[0] == 0 else norm[0]
        norm[-1] = 1*norm.unit if norm[-1] == 0 else norm[-1]

        # Avoid another 0/0 is the effective area is null for some bins
        if np.any(norm == 0):
            norm[norm == 0] = 1*norm.unit

            logger.warn("Null effective area, cannot properly compute dispersion matrix.")
        
        # "Broadcast" such that it has the compatible dimensions with the 2D matrix
        norm = spec.expand_dims(norm, 'Ei')
        
        # Normalize column-by-column
        return (spec / norm)

    @property
    def photon_energy_axis(self):
        """
        Real energy bins (``Ei``).

        Returns
        -------
        :py:class:`histpy.Axes`
        """
        
        return self.axes['Ei']

    
    @property
    def measured_energy_axis(self):
        """
        Measured energy bins (``Em``).

        Returns
        -------
        :py:class:`histpy.Axes`        
        """
        
        return self.axes['Em']
        

        
    
