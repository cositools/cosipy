from histpy import Histogram#, Axes, Axis

import numpy as np
import astropy.units as u
#from astropy.units import Quantity
#from scipy import integrate
from scoords import SpacecraftFrame, Attitude

#from threeML import DiracDelta, Constant, Line, Quadratic, Cubic, Quartic, StepFunction, StepFunctionUpper, Cosine_Prior, Uniform_prior, PhAbs, Gaussian

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
       
    def get_expectation(self, spectrum, polarization=None):
        """
        Convolve the response with a spectral (and optionally, polarization) hypothesis to obtain the expected
        excess counts from the source.

        Parameters
        ----------
        spectrum : :py:class:`threeML.Model`
            Spectral hypothesis.
        polarization : 'astromodels.core.polarization.LinearPolarization', optional
            Polarization angle and degree. The angle is assumed to have same convention as point source response.
        
        Returns
        -------
        :py:class:`histpy.Histogram`
             Histogram with the expected counts on each analysis bin
        """

        if polarization is None:

            if 'Pol' in self.axes.labels:

                raise RuntimeError("Must include polarization in point source response if using polarization response")

            else:

                contents = self.contents
                axes = self.axes[1:]

        else:

            polarization_angle = polarization.angle.value
            polarization_level = polarization.degree.value / 100.

            if not 'Pol' in self.axes.labels:
                
                raise RuntimeError("Response must have polarization angle axis to include polarization in point source response")

            if polarization_angle == 180.:
                polarization_angle = 0.

            polarization_angle_components = []

            for i in range(self.axes['Pol'].nbins):

                polarization_angle_components.append(self.slice[{'Pol':slice(i,i+1)}].project('Ei', 'Em', 'Phi', 'PsiChi'))

                if polarization_angle >= self.axes['Pol'].edges.to_value(u.deg)[i] and polarization_angle < self.axes['Pol'].edges.to_value(u.deg)[i+1]:
                    polarized_component = polarization_angle_components[i].contents

            unpolarized_component = polarization_angle_components[0].contents

            for i in range(len(polarization_angle_components) - 1):

                unpolarized_component += polarization_angle_components[i+1].contents

            polarized_component /= np.sum(polarized_component.value)
            unpolarized_component /= np.sum(unpolarized_component.value)

            polarization_hist = (polarization_level * polarized_component) + ((1 - polarization_level) * unpolarized_component)
            polarization_hist *= np.sum(self.contents) / np.sum(polarization_hist)

            contents = polarization_hist
            axes = self.project('Ei', 'Em', 'Phi', 'PsiChi').axes[1:]

        energy_axis = self.photon_energy_axis

        flux = get_integrated_spectral_model(spectrum, energy_axis)
        
        expectation = np.tensordot(contents, flux.contents, axes=([0], [0]))
        
        # Note: np.tensordot loses unit if we use a sparse matrix as it input.
        if self.is_sparse:
            expectation *= self.unit * flux.unit

        hist = Histogram(axes, contents=expectation)

        if not hist.unit == u.dimensionless_unscaled:
            raise RuntimeError("Expectation should be dimensionless, but has units of " + str(hist.unit) + ".")

        return hist
