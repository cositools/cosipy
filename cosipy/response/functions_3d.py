import numpy as np

from scipy import integrate
from scipy.interpolate import interp1d

import astropy.units as u
from astropy.coordinates import Galactic

from histpy import Histogram

import logging
logger = logging.getLogger(__name__)

def get_integrated_extended_model_3d(extendedmodel, image_axis, energy_axis):

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
    """

    from cosipy.threeml.custom_functions import GalpropHealpixModel

    if not isinstance(image_axis.coordsys, Galactic):
        raise ValueError

    l, b = image_axis.pix2ang(np.arange(image_axis.npix), lonlat=True)

    if isinstance(extendedmodel.spatial_shape, GalpropHealpixModel):

        # The norm is updated internally by 3ML for each likelihood call
        norm = extendedmodel.spatial_shape.K.value

        # Make sure the dummy spectral parameter is fixed
        extendedmodel.spectrum.main.Constant.k.free = False

        # First get differential intensity from GALPROP model (ph/cm2/s/sr/MeV),
        # and then integrate over energy bins. We attach the integrated flux
        # to the extended model instance so that it only needs to be calculated once.
        if not isinstance(extendedmodel.spatial_shape._result, np.ndarray):

            intensity = (1/norm)*extendedmodel.spatial_shape.evaluate(l, b, energy_axis.edges.to(u.MeV), norm)

            # Integrate over energy bins for each sky position
            extendedmodel.spatial_shape.intg_flux = np.zeros((intensity.shape[0], intensity.shape[1]-1))

            # Convert units outside loop to optimize speed
            energy_edges = energy_axis.edges.to_value(u.MeV)

            # Integrate spectrum over energy bins for each spatial pixel
            logger.info("Integrating intensity over energy bins...")
            for j in range(len(intensity)):

                interp_func = interp1d(energy_edges, intensity[j], bounds_error=False, fill_value='extrapolate')

                extendedmodel.spatial_shape.intg_flux[j] = \
                    np.array([integrate.quad(interp_func, lo_lim, hi_lim)[0]
                              for lo_lim, hi_lim
                              in zip(energy_edges[:-1], energy_edges[1:])])

        flux = norm * extendedmodel.spatial_shape.intg_flux

        flux_map = Histogram((image_axis, energy_axis), \
                             contents = flux, \
                             unit = (u.s * u.cm**2 * u.sr) ** (-1),
                             copy_contents = False)

    return flux_map
