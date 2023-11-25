from histpy import Histogram, Axes, Axis

import astropy.units as u

from astropy.units import Quantity

from scipy import integrate

from threeML import DiracDelta, Constant, Line, Quadratic, Cubic, Quartic, StepFunction, StepFunctionUpper, Cosine_Prior, Uniform_prior, PhAbs, Gaussian


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
        
        eaxis = self.photon_energy_axis
        
        spectrum_unit = None

        for item in spectrum.parameters:
            if getattr(spectrum, item).is_normalization == True:
                spectrum_unit = getattr(spectrum, item).unit
                break
                
        if spectrum_unit == None:
            if isinstance(spectrum, Constant):
                spectrum_unit = spectrum.k.unit
            elif isinstance(spectrum, Line) or isinstance(spectrum, Quadratic) or isinstance(spectrum, Cubic) or isinstance(spectrum, Quartic):
                spectrum_unit = spectrum.a.unit
            elif isinstance(spectrum, StepFunction) or isinstance(spectrum, StepFunctionUpper) or isinstance(spectrum, Cosine_Prior) or isinstance(spectrum, Uniform_prior) or isinstance(spectrum, DiracDelta): 
                spectrum_unit = spectrum.value.unit
            elif isinstance(spectrum, PhAbs):
                spectrum_unit = u.dimensionless_unscaled
            elif isinstance(spectrum, Gaussian):
                spectrum_unit = spectrum.F.unit / spectrum.sigma.unit 
            else:
                try:
                    spectrum_unit = spectrum.K.unit
                except:
                    raise RuntimeError("Spectrum not yet supported because units of spectrum are unknown.")
                    
        if isinstance(spectrum, DiracDelta):
            flux = Quantity([spectrum.value.value * spectrum_unit * lo_lim.unit if spectrum.zero_point.value >= lo_lim/lo_lim.unit and spectrum.zero_point.value <= hi_lim/hi_lim.unit else 0 * spectrum_unit * lo_lim.unit
                             for lo_lim,hi_lim
                             in zip(eaxis.lower_bounds, eaxis.upper_bounds)])
        else:
            flux = Quantity([integrate.quad(spectrum, lo_lim/lo_lim.unit, hi_lim/hi_lim.unit)[0] * spectrum_unit * lo_lim.unit
                             for lo_lim,hi_lim
                             in zip(eaxis.lower_bounds, eaxis.upper_bounds)])
        
        flux = self.expand_dims(flux.value, 'Ei') * flux.unit

        expectation = self * flux
        
        return expectation
