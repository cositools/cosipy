from histpy import Histogram, Axes, Axis

import numpy as np
import astropy.units as u
from astropy.units import Quantity
from scipy import integrate

from threeML import DiracDelta, Constant, Line, Quadratic, Cubic, Quartic, StepFunction, StepFunctionUpper, Cosine_Prior, Uniform_prior, PhAbs, Gaussian

from .functions import get_integrated_spectral_model

class PointSourceResponse(Histogram):
    """
    Handles the multi-dimensional matrix that describes the expected
    response of the instrument for a particular point in the sky.

    Parameters
    ----------
    axes : :py:class:`histpy.Axes`
        Binning information for each variable. The following labels are expected:\n
        - ``Ei``: Real energy
        - ``Em``: Measured energy. Optional
        - ``Phi``: Compton angle. Optional.
        - ``PsiChi``:  Location in the Compton Data Space (HEALPix pixel). Optional.
        - ``SigmaTau``: Electron recoil angle (HEALPix pixel). Optional.
        - ``Dist``: Distance from first interaction. Optional.
    contents : array, :py:class:`astropy.units.Quantity` or :py:class:`sparse.SparseArray`
        Array containing the differential effective area convolved with wht source exposure.
    unit : :py:class:`astropy.units.Unit`, optional
        Physical units, if not specified as part of ``contents``. Units of ``area*time``
        are expected.
    """
    
    @property
    def photon_energy_axis(self):
        """
        Real energy bins (``Ei``).

        Returns
        -------
        :py:class:`histpy.Axes`
        """
        
        return self.axes['Ei']
       
    def get_expectation(self, spectrum):
        """
        Convolve the response with a spectral hypothesis to obtain the expected
        excess counts from the source.

        Parameters
        ----------
        spectrum : :py:class:`threeML.Model`
            Spectral hypothesis.

        Returns
        -------
        :py:class:`histpy.Histogram`
             Histogram with the expected counts on each analysis bin
        """
        
        energy_axis = self.photon_energy_axis

        flux = get_integrated_spectral_model(spectrum, energy_axis)
        
        expectation = np.tensordot(self.contents, flux.contents, axes = ([0], [0]))
        
        # Note: np.tensordot loses unit if we use a sparse matrix as it input.
        if self.is_sparse:
            expectation *= self.unit * flux.unit

        hist = Histogram(self.axes[1:], contents = expectation)
        
        return hist
