from histpy import Histogram

import numpy as np
import astropy.units as u
from scoords import SpacecraftFrame, Attitude

from .functions import get_integrated_spectral_model

import logging
logger = logging.getLogger(__name__)

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

    def get_expectation(self, spectrum, polarization=None, flux=None):
        """
        Convolve the response with a spectral (and optionally, polarization) hypothesis to obtain the expected
        excess counts from the source.

        Parameters
        ----------
        spectrum : :py:class:`threeML.Model`
            Spectral hypothesis.
        polarization : 'astromodels.core.polarization.LinearPolarization', optional
            Polarization angle and degree. The angle is assumed to have same convention as point source response.
        flux : 1D Histogram, optional
            Pre-computed integrated flux of spectrum for each bin on Ei axis

        Returns
        -------
        :py:class:`histpy.Histogram`
             Histogram with the expected counts on each analysis bin
        """

        if polarization is None:

            if 'Pol' in self.axes.labels:

                raise RuntimeError("Must include polarization in point source response if using polarization response")

            contents = self.contents
            axes = self.axes[1:]

        else:

            if not 'Pol' in self.axes.labels:

                raise RuntimeError("Response must have polarization angle axis to include polarization in point source response")

            polarization_angle = polarization.angle.value
            polarization_level = polarization.degree.value / 100.

            if polarization_angle == 180.:
                polarization_angle = 0.

            pol_axis = self.axes['Pol']

            # unpolarized weights
            weights = np.full(pol_axis.nbins, (1. - polarization_level) / pol_axis.nbins)
            
            # add polarized weights
            polarization_bin_index = pol_axis.find_bin(polarization_angle * u.deg)
            weights[polarization_bin_index] += polarization_level
            weights *= self.axes['Pol'].nbins

            contents = np.tensordot(weights, self.contents, axes=(0, self.axes.label_to_index('Pol')))

            axes = self.axes['Em', 'Phi', 'PsiChi']

        if flux is None:
            energy_axis = self.photon_energy_axis
            flux = get_integrated_spectral_model(spectrum, energy_axis)
            
        expectation = np.tensordot(contents, flux.contents, axes=(0, 0))

        # if self is sparse, expectation will be a SparseArray with
        # no units, so set the result's unit explicitly
        hist = Histogram(axes, contents = expectation,
                         unit = self.unit * flux.unit,
                         copy_contents = False)

        if not hist.unit == u.dimensionless_unscaled:
            raise RuntimeError(f"Expectation should be dimensionless, but has units of {(hist.unit)}.")
        
        return hist
