import numpy as np

import astropy.units as u
from astropy.coordinates import Galactic

from histpy import Histogram

from threeML import (
    Band,
    DiracDelta,
    Constant,
    Line,
    Quadratic,
    Cubic,
    Quartic,
    StepFunction,
    StepFunctionUpper,
    Cosine_Prior,
    Uniform_prior,
    PhAbs,
    Gaussian
)

def get_integrated_spectral_model(spectrum, energy_axis):
    """Get the photon fluxes integrated over given energy bins with an
    input astropy spectral model

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
    This function determines the unit of the spectrum, performs the
    integration over each energy bin, and returns the result as a
    Histogram object.

    """

    from cosipy.response.integrals import get_integral_values

    spectrum_unit = get_spectrum_unit(spectrum)

    flux_values = get_integral_values(spectrum, energy_axis.edges.value)

    flux = Histogram(energy_axis,
                     contents = flux_values,
                     unit = spectrum_unit * energy_axis.unit,
                     copy_contents = False)

    return flux


def get_spectrum_unit(spectrum):
    """
    Get the unit of the spectral model.

    Parameters
    ----------
    spectrum : astromodels.functions
        One-dimensional spectral function from astromodels.

    Returns:
    astropy.unit for the spectrum

    """

    from cosipy.threeml import Band_Eflux

    spectrum_unit = None
    for param in spectrum.parameters.values():
        if param.is_normalization:
            spectrum_unit = param.unit
            break

    if spectrum_unit is None:
        match spectrum:
            case Constant():
                spectrum_unit = spectrum.k.unit
            case Line() | Quadratic() | Cubic() | Quartic():
                spectrum_unit = spectrum.a.unit
            case StepFunction() | StepFunctionUpper() | Cosine_Prior() | Uniform_prior() | DiracDelta():
                spectrum_unit = spectrum.value.unit
            case PhAbs():
                spectrum_unit = u.dimensionless_unscaled
            case Gaussian():
                spectrum_unit = spectrum.F.unit / spectrum.sigma.unit
            case Band_Eflux():
                spectrum_unit = spectrum.K.unit / spectrum.a.unit
            case _:
                spectrum_unit = None
                for pname in ('K', 'k'):
                    if pname in spectrum.parameters:
                        spectrum_unit = spectrum.parameters[pname].unit

                if spectrum_unit is None:
                    raise RuntimeError("Spectrum not yet supported because units are unknown.")

    return spectrum_unit


def get_integrated_extended_model(extendedmodel, image_axis, energy_axis):
    """Calculate the integrated flux map for an extended source model.

    Parameters
    ----------
    extendedmodel : astromodels.ExtendedSource
        An astromodels extended source model object. This model
        represents the spatial and spectral distribution of an
        extended astronomical source.
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
    This function first integrates the spectral model over the energy
    bins, then combines it with the spatial distribution to create a
    2D flux map.

    """

    if not isinstance(image_axis.coordsys, Galactic):
        raise ValueError

    integrated_flux = \
        get_integrated_spectral_model(spectrum = extendedmodel.spectrum.main.shape,
                                      energy_axis = energy_axis)

    l, b = image_axis.pix2ang(np.arange(image_axis.npix), lonlat=True)
    normalized_map = extendedmodel.spatial_shape(l, b) / u.sr

    flux = np.tensordot(normalized_map, integrated_flux.contents, axes = 0)

    flux_map = Histogram((image_axis, energy_axis),
                         contents = flux,
                         copy_contents = False)

    return flux_map
