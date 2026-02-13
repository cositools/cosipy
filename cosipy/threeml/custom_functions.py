import numpy as np
from scipy.interpolate import interp1d

import astropy.units as u
from astropy.io import fits
from astropy.coordinates import BaseCoordinateFrame, Galactic, SkyCoord

from astromodels.functions.function import (
    Function1D,
    Function2D,
    Function3D,
    FunctionMeta,
    ModelAssertionViolation,
)

import healpy as hp

import logging
logger = logging.getLogger(__name__)

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
            is_normalization : False
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

    def _setup(self):
        self._params = np.full(5, np.nan)

    def _set_units(self, x_unit, y_unit):
        # The normalization has the unit of x * y
        self.K.unit = y_unit * x_unit

        # The break point has always the same dimension as the x variable
        self.E0.unit = x_unit

        # alpha and beta are dimensionless
        self.alpha.unit = u.dimensionless_unscaled
        self.beta.unit = u.dimensionless_unscaled

        # a and b have the same units of x
        self.a.unit = x_unit
        self.b.unit = x_unit

    def get_normalization(self, a, b, alpha, beta, E0):
        """
        Compute normalization constant for function.
        """

        from cosipy.response.integrals import get_integral_values
        from astromodels import Band_grbm

        # Cache the normalizing integral so we can reuse its value
        # instead of recomputing it if its parameters have not
        # changed. We must test for change in all the parameters of
        # spectrum.

        params = np.array([a, b, alpha, beta, E0])
        if not np.array_equal(self._params, params):
            self._params = params

            spectrum = Band_grbm(alpha=alpha,
                                 beta=beta,
                                 K=1.0,
                                 xc=E0,
                                 piv=1.0)

            self._integral = get_integral_values(spectrum, [a, b])[0]

        return self._integral

    def evaluate(self, x, K, E0, alpha, beta, a, b):

        import astromodels.functions.numba_functions as nb_func

        if alpha < beta:
            raise ModelAssertionViolation("Alpha cannot be less than beta")

        if isinstance(x, u.Quantity):
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

        A_ = K_ / self.get_normalization(a_, b_, alpha_, beta_, E0_)

        # accelerated eval uses function of Band_grbm(), not Band()
        return nb_func.band_eval(x_, A_, alpha_, beta_, E0_, 1.0) * unit_


class SpecFromDat(Function1D, metaclass=FunctionMeta):
        r"""
        description :
            A  spectrum loaded from a dat file
        parameters :
            K :
                desc : Normalization factor
                initial value : 1.0
                is_normalization : True
                min: 0.0
                max: 1e6
                delta: 1.0
                units: ph/cm2/s/kev
        properties:
            dat:
                desc: the data file to load
                initial value: test.dat
                defer: True
                units:
                    energy: keV
                    flux: ph/cm2/s/kev
        """

        def _setup(self):
            self._dat_file = None

        def _set_units(self, x_unit, y_unit):

            self.K.unit = y_unit

        def evaluate(self, x, K):

            if self.dat.value != self._dat_file:

                # data file property changed -- reload and rebuild
                # interpolator function

                self._dat_file = self.dat.value

                data = np.genfromtxt(self.dat.value,comments = "#",
                                     usecols = (1,2),
                                     skip_footer=1,
                                     skip_header=5)
                dataEn = data[:,0]
                dataFlux = data[:,1]

                # Calculate the widths of the energy bins
                ewidths = np.diff(dataEn, append=dataEn[-1])

                # Normalize dataFlux using the energy bin widths
                dataFlux /= np.sum(dataFlux * ewidths)

                self._fun = interp1d(dataEn, dataFlux, fill_value=0, bounds_error=False)

            return K * self._fun(x)


class GalpropHealpixModel(Function3D, metaclass=FunctionMeta):

    r"""
    description :
        A custom 3D function that reads a GALPROP HEALPix map and
        interpolates over energy for a given set of sky positions in
        Galactic coordinates (default is all-sky). The intensity is
        interpolated from the GALPROP spectra stored in the HEALPix
        map, and scaled by a normalization constant K.

        This class is compatible with healpix outputs from GALPROP v54 and
        v57 (default). The GALPROP maps should be defined in Galactic
        coordinates and specify the intensity in units of ph/cm2/s/sr/MeV,
        with energy given in MeV.

        When calling the function, energies are assumed to be in MeV,
        coordinates in degrees (galactic frame), and fluxes are returned
        in 1/(cm2 MeV s sr).

    latex : $ K \times \ \mathrm{GALPROP_map(l,b,E)}$

    parameters :
        K :
            desc : Normalization factor (unitless)
            initial value : 1.0
            min : 0
            max : 1e3
            delta : 0.01
            is_normalization : True
    """

    def _setup(self):
        self._file_loaded = False
        self._fitsfile = None
        self._frame = Galactic().name
        self._result = None
        self._gal_version = 57

    def set_frame(self, new_frame):

        """
        Set a new frame for the coordinates (the default is Galactic)

        :param new_frame: a coordinate frame from astropy
        :return: (none)
        """
        assert isinstance(new_frame, BaseCoordinateFrame)

        self._frame = new_frame.name

    def set_version(self,v):

        """
        Set GALPROP version for input skymap.

        "param v: version number, either 57 (default) or 54.
        """

        if not v in [54,57]:
            raise ValueError("GALPROP version must be 54 or 57.")

        self._gal_version = v

    def load_file(self, fits_path):

        self._fitsfile = fits_path
        self._file_loaded = True
        logger.info(f"loading GALPROP model: {self._fitsfile}")

        with fits.open(fits_path) as hdul:
            skymap_hdu = hdul['SKYMAP']
            energy_hdu = hdul['ENERGIES']

            if self._gal_version == 57:
                self.table = np.stack([skymap_hdu.data[col] for col in skymap_hdu.columns.names], axis=1)
                self.energy = energy_hdu.data['ENERGY'] * u.MeV # in MeV

            if self._gal_version == 54:
                self.table = np.stack([skymap_hdu.data[s] for s in range(skymap_hdu.data.shape[0])], axis=1)[0]
                self.energy = energy_hdu.data['MeV'] * u.MeV # in MeV

        self.n_pixels, self.n_energies = self.table.shape
        self.nside = hp.npix2nside(self.n_pixels)

    def _set_units(self, x_unit, y_unit, z_unit, w_unit):

        self.K.unit = u.dimensionless_unscaled

    def evaluate(self, x, y, z, K):

        if x.shape != y.shape:
            raise ValueError("x and y must have the same shape")

        if self._fitsfile is None:
            raise RuntimeError("Need to either specify or load a fits file")

        if not self._file_loaded:
            self.load_file(self._fitsfile)

        if self._frame != "galactic":
            logger.info(f"Converting input coords from {self._frame} to galactic")
            _coord = SkyCoord(ra=x, dec=y, frame=self._frame, unit="deg")
            x = _coord.transform_to("galactic").l.deg
            y = _coord.transform_to("galactic").b.deg

        theta = np.radians(90.0 - y)
        phi = np.radians(x)
        pix = hp.ang2pix(self.nside, theta, phi)

        # Get interpolated function.
        logger.info("Interpolating GALPROP map...")
        self._result = np.zeros((x.size, z.size))
        for i, p in enumerate(pix):
            spectrum = self.table[p]
            interp_func = interp1d(self.energy, spectrum, bounds_error=False, fill_value='extrapolate')
            self._result[i] = interp_func(z)

        return K * self._result * ((u.MeV * u.s * u.cm**2 * u.sr) ** (-1))

    def to_dict(self, minimal=False):

        data = super(Function3D, self).to_dict(minimal)

        if not minimal:

            data['extra_setup'] = {"_fitsfile": self._fitsfile, "_frame": self._frame}

        return data

    def get_total_spatial_integral(self, z, avg_int=False, nside=None):

        """
        Returns the total integral over the spatial components.

        :return: an array of values of the integral (same dimension as z).
        """

        # access with results.optimized_model["galprop_source"].spatial_shape.nside

        if nside is not None:
            # Get spatial grid from nside
            n_pixels = hp.nside2npix(nside)
            ipix = np.arange(n_pixels)
            x, y = hp.pix2ang(nside, ipix, lonlat=True)
            logger.info(f"using nside={nside} from user input in evaluate method")

        else:
            # Get spatial grid from GALPROP map:
            self.load_file(self._fitsfile)
            ipix = np.arange(self.n_pixels)
            x, y = hp.pix2ang(self.nside, ipix, lonlat=True)
            logger.info(f"using nside={self.nside} from GALPROP map in evaluate method")

        intensity_3d = self.evaluate(x, y, z, self.K.value)

        # We are calculating the average intensity (and not the total in)
        intensity_2d = np.sum(intensity_3d, axis=0)

        if avg_int:
            intensity_2d /= len(intensity_3d) # return average intensity

        return intensity_2d
