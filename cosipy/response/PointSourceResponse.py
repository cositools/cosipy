from histpy import Histogram, Axes, Axis

import astropy.units as u

from astropy.units import Quantity

from scipy import integrate

from threeML import Blackbody, ModifiedBlackbody, NonDissipativePhotosphere, NonDissipativePhotosphere_Deep, StepFunction, StepFunctionUpper, Sin, DiracDelta, Log_parabola, Exponential_cutoff, PhAbs, TbAbs, WAbs, ZDust, Constant, Line, Quadratic, Cubic, Quartic, Powerlaw, Powerlaw_flux, Powerlaw_Eflux, Cutoff_powerlaw, Cutoff_powerlaw_Ep, Inverse_cutoff_powerlaw, Super_cutoff_powerlaw, SmoothlyBrokenPowerLaw, Broken_powerlaw, Band, Band_grbm, Band_Calderone, DMFitFunction, DMSpectra, Gaussian, Truncated_gaussian, Cauchy, Cosine_Prior, Log_normal, Uniform_prior, Log_uniform_prior

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
        # Dirty workaround for inconsistent naming conventions in astromodels (only until someone implements something more clever). 
        # Idea: Since most spectral functions allow axes to the spectrum unit via spectrum.K.unit, only check for the non-default cases.
        # Then assume that you are in the default case and throw a RuntimeError when something goes wrong. 
        #
        #
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Attention !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # When using a 'new' spectral function, make sure that K.unit gets you the information you need to calculate the flux!
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #
        #

        if isinstance(spectrum, Constant):
            spectrum_unit = spectrum.k.unit
        elif isinstance(spectrum, Line) or isinstance(spectrum, Quadratic) or isinstance(spectrum, Cubic) or isinstance(spectrum, Quartic):
            spectrum_unit = spectrum.a.unit
        elif isinstance(spectrum, Powerlaw_flux) or isinstance(spectrum, Powerlaw_Eflux) or isinstance(spectrum, Log_normal) or isinstance(spectrum, Band_Calderone)\
        or isinstance(spectrum, Gaussian) or isinstance(spectrum, Truncated_gaussian):
            spectrum_unit = spectrum.F.unit
        elif isinstance(spectrum, StepFunction) or isinstance(spectrum, StepFunctionUpper) or isinstance(spectrum, Cosine_Prior) or isinstance(spectrum, Uniform_prior): 
            spectrum_unit = spectrum.value.unit
        elif isinstance(spectrum, PhAbs) or isinstance(spectrum, TbAbs) or isinstance(spectrum, WAbs):
            spectrum_unit = u.dimensionless_unscaled
        elif isinstance(spectrum, ZDust):
            spectrum_unit = spectrum.e_bmv.unit
        elif isinstance(spectrum, DiracDelta) or isinstance(spectrum, DMFitFunction) or isinstance(spectrum, DMSpectra):
            raise RuntimeError("Spectrum not yet supported")
        else:
            try:
                spectrum_unit = spectrum.K.unit
            except:
                raise RuntimeError("Spectrum not yet supported")
            
        flux = Quantity([integrate.quad(spectrum, lo_lim/lo_lim.unit, hi_lim/hi_lim.unit)[0] * spectrum_unit * lo_lim.unit
                         for lo_lim,hi_lim
                         in zip(eaxis.lower_bounds, eaxis.upper_bounds)])
        
        flux = self.expand_dims(flux.value, 'Ei') * flux.unit

        expectation = self * flux
        
        return expectation

    
    
