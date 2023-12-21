from astromodels.functions.function import Function1D, FunctionMeta, ModelAssertionViolation
import astromodels.functions.numba_functions as nb_func
from threeML import Band, DiracDelta, Constant, Line, Quadratic, Cubic, Quartic, StepFunction, StepFunctionUpper, Cosine_Prior, Uniform_prior, PhAbs, Gaussian
import astropy.units as astropy_units
from astropy.units import Quantity
from past.utils import old_div
from scipy.special import gammainc, expi
from scipy import integrate
import numpy as np
import math

from histpy import Histogram, Axes, Axis

class Band_Eflux(Function1D, metaclass=FunctionMeta):
    r"""
    description :
        Band model from Band et al., 1993 where the normalization is the flux defined between a and b
    latex : $ A \begin{cases} x^{\alpha} \exp{(-\frac{x}{E0})} & x \leq (\alpha-\beta) E0 \\ x^{\beta} \exp (\beta-\alpha)\left[(\alpha-\beta) E0\right]^{\alpha-\beta} & x>(\alpha-\beta) E0 \end{cases} $
    parameters :
        K :
            desc : Normalization (flux between a and b)
            initial value : 1.e-5
            min : 1e-50
            is_normalization : True
            transformation : log10
        E0 :
            desc : $\frac{xp}{2+\alpha}$ where xp is peak in the x * x * N (nuFnu if x is an energy)
            initial value : 500
            min : 1
            transformation : log10
        alpha :
            desc : low-energy photon index
            initial value : -1.0
            min : -1.5
            max : 3
        beta :
            desc : high-energy photon index
            initial value : -2.0
            min : -5.0
            max : -1.6
        a :
            desc : lower energy integral bound (keV)
            initial value : 10
            min : 0
            fix: yes
        b :
            desc : upper energy integral bound (keV)
            initial value : 1000
            min : 0
            fix: yes
    """
    
    def _set_units(self, x_unit, y_unit):
        # The normalization has the unit of x * y
        self.K.unit = y_unit * x_unit

        # The break point has always the same dimension as the x variable
        self.E0.unit = x_unit

        # alpha and beta are dimensionless
        self.alpha.unit = astropy_units.dimensionless_unscaled
        self.beta.unit = astropy_units.dimensionless_unscaled

        # a and b have the same units of x
        self.a.unit = x_unit
        self.b.unit = x_unit

    def evaluate(self, x, K, E0, alpha, beta, a, b):
        if alpha < beta:
            raise ModelAssertionViolation("Alpha cannot be less than beta")

        if isinstance(x, astropy_units.Quantity):
            alpha_ = alpha.value
            beta_ = beta.value
            K_ = K.value
            E0_ = E0.value
            a_ = a.value
            b_ = b.value
            x_ = x.value

            unit_ = self.y_unit

        else:
            unit_ = 1.0
            alpha_, beta_, K_, E0_, a_, b_, x_ = alpha, beta, K, E0, a, b, x
            
        spectrum_ = Band(alpha=alpha_,
                         beta=beta_,
                         K=1.0,
                         xp=E0_*(2 + alpha_),
                         piv=1.0)
        A_ = K_ / integrate.quad(spectrum_, a_, b_)[0]

        return nb_func.band_eval(x_, A_, alpha_, beta_, E0_, 1.0) * unit_

def get_integrated_spectral_model(spectrum, eaxis):
    """
    Get the photon fluxes integrated over given energy bins with an input astropy spectral model
        
    Parameters
    ----------
    spectrum: astromodels (one-dimensional function)
    eaxis: histpy.Axis
    
    Returns
    -------
    flux: histpy.Histogram 
    """

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
    
    flux = Histogram(eaxis, contents = flux)

    return flux
