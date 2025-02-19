import numpy as np
import astropy.units as u
from astropy.units import Quantity
from astropy.coordinates import Galactic
from scipy import integrate

from histpy import Histogram, Axes, Axis, HealpixAxis

from threeML import Band, DiracDelta, Constant, Line, Quadratic, Cubic, Quartic, StepFunction, StepFunctionUpper, Cosine_Prior, Uniform_prior, PhAbs, Gaussian

def get_integrated_spectral_model(spectrum, energy_axis):
    """
    Get the photon fluxes integrated over given energy bins with an input astropy spectral model
        
    Parameters
    ----------
    spectrum : astromodels.functions
        One-dimensional spectral function from astromodels.
    energy_axis : histpy.Axis
        Energy axis defining the energy bins for integration.

    Returns
    -------
    flux : histpy.Histogram
        Histogram of integrated photon fluxes for each energy bin.

    Raises
    ------
    RuntimeError
        If the spectrum is not supported or its units are unknown.

    Notes
    -----
    This function determines the unit of the spectrum, performs the integration
    over each energy bin, and returns the result as a Histogram object.
    """

    from cosipy.threeml import Band_Eflux

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
        elif isinstance(spectrum, Band_Eflux):
            spectrum_unit = spectrum.K.unit / spectrum.a.unit
        else:
            try:
                spectrum_unit = spectrum.K.unit
            except:
                raise RuntimeError("Spectrum not yet supported because units of spectrum are unknown.")
                
    if isinstance(spectrum, DiracDelta):
        flux = Quantity([spectrum.value.value * spectrum_unit * lo_lim.unit if spectrum.zero_point.value >= lo_lim/lo_lim.unit and spectrum.zero_point.value <= hi_lim/hi_lim.unit else 0 * spectrum_unit * lo_lim.unit
                         for lo_lim,hi_lim
                         in zip(energy_axis.lower_bounds, energy_axis.upper_bounds)])
    else:
        flux = Quantity([integrate.quad(spectrum, lo_lim/lo_lim.unit, hi_lim/hi_lim.unit)[0] * spectrum_unit * lo_lim.unit
                         for lo_lim,hi_lim
                         in zip(energy_axis.lower_bounds, energy_axis.upper_bounds)])
    
    flux = Histogram(energy_axis, contents = flux)

    return flux

def get_integrated_extended_model(extendedmodel, image_axis, energy_axis):
    """
    Calculate the integrated flux map for an extended source model.

    Parameters
    ----------
    extendedmodel : astromodels.ExtendedSource
        An astromodels extended source model object. This model represents
        the spatial and spectral distribution of an extended astronomical source.
    image_axis : histpy.HealpixAxis
        Spatial axis for the image.
    energy_axis : histpy.Axis
        Energy axis defining the energy bins.

    Returns
    -------
    flux_map : histpy.Histogram
        2D histogram representing the integrated flux map.

    Notes
    -----
    This function first integrates the spectral model over the energy bins,
    then combines it with the spatial distribution to create a 2D flux map.
    """
    
    if not isinstance(image_axis.coordsys, Galactic):
        raise ValueError

    integrated_flux = get_integrated_spectral_model(spectrum = extendedmodel.spectrum.main.shape, energy_axis = energy_axis)

    npix = image_axis.npix
    coords = image_axis.pix2skycoord(np.arange(npix))

    normalized_map = extendedmodel.spatial_shape(coords.l.deg, coords.b.deg) / u.sr

    flux_map = Histogram([image_axis, energy_axis], contents = np.tensordot(normalized_map, integrated_flux.contents, axes = 0))

    return flux_map
    
