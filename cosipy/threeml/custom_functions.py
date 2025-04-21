from astromodels.functions.function import Function1D, FunctionMeta, ModelAssertionViolation, Function2D, Function3D
import astromodels.functions.numba_functions as nb_func
from astromodels.utils.angular_distance import angular_distance
from threeML import Band, DiracDelta, Constant, Line, Quadratic, Cubic, Quartic, StepFunction, StepFunctionUpper, Cosine_Prior, Uniform_prior, PhAbs, Gaussian
import astropy.units as astropy_units
from astropy.units import Quantity
from past.utils import old_div
from scipy.special import gammainc, expi
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy import integrate
import numpy as np
import math
import astropy.units as u
from astropy.io import fits
import healpy as hp
from histpy import Histogram, Axes, Axis
from astropy.coordinates import BaseCoordinateFrame, ICRS, Galactic, SkyCoord

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

class SpecFromDat(Function1D, metaclass=FunctionMeta):
        r"""
        description :
            A  spectrum loaded from a dat file
        parameters :
            K :
                desc : Normalization
                initial value : 1.0
                is_normalization : True
                transformation : log10
                min : 1e-30
                max : 1e3
                delta : 0.1
                units: ph/cm2/s
        properties:
            dat:
                desc: the data file to load
                initial value: test.dat
                defer: True
                units: 
                    energy: keV
                    flux: ph/cm2/s/kev
        """            
        def _set_units(self, x_unit, y_unit):
            
            self.K.unit = y_unit

        def evaluate(self, x, K):
            dataFlux = np.genfromtxt(self.dat.value,comments = "#",usecols = (2),skip_footer=1,skip_header=5)
            dataEn = np.genfromtxt(self.dat.value,comments = "#",usecols = (1),skip_footer=1,skip_header=5)
            
            # Calculate the widths of the energy bins
            ewidths = np.diff(dataEn, append=dataEn[-1])

            # Normalize dataFlux using the energy bin widths
            dataFlux = dataFlux  / np.sum(dataFlux * ewidths)
            
            fun = interp1d(dataEn,dataFlux,fill_value=0,bounds_error=False)
            
            if self._x_unit != None:
                dataEn *= self._x_unit

            result = np.zeros(x.shape) * K * 0

            for i in range(len(x)): result[i] += K*fun(x[i])
            return result


class Wide_Asymm_Gaussian_on_sphere(Function2D, metaclass=FunctionMeta):
    r"""
    description :

        A bidimensional Gaussian function on a sphere (in spherical coordinates)

        see https://en.wikipedia.org/wiki/Gaussian_function#Two-dimensional_Gaussian_function

    parameters :

        lon0 :

            desc : Longitude of the center of the source
            initial value : 0.0
            min : 0.0
            max : 360.0

        lat0 :

            desc : Latitude of the center of the source
            initial value : 0.0
            min : -90.0
            max : 90.0

        a :

            desc : Standard deviation of the Gaussian distribution (major axis)
            initial value : 10
            min : 0
            max : 90

        e :

            desc : Excentricity of Gaussian ellipse, e^2 = 1 - (b/a)^2, where b is the standard deviation of the Gaussian distribution (minor axis)
            initial value : 0.9
            min : 0
            max : 1

        theta :

            desc : inclination of major axis to a line of constant latitude
            initial value : 10.
            min : -90.0
            max : 90.0

    """
    def _set_units(self, x_unit, y_unit, z_unit):

        # lon0 and lat0 and a have most probably all units of degrees. However,
        # let's set them up here just to save for the possibility of using the
        # formula with other units (although it is probably never going to happen)

        self.lon0.unit = x_unit
        self.lat0.unit = y_unit
        self.a.unit = x_unit
        self.e.unit = u.dimensionless_unscaled
        self.theta.unit = u.degree

    def evaluate(self, x, y, lon0, lat0, a, e, theta):

        lon, lat = x, y

        b = a * np.sqrt(1.0 - e**2)

        dX = np.atleast_1d(angular_distance(lon0, lat0, lon, lat0))
        dY = np.atleast_1d(angular_distance(lon0, lat0, lon0, lat))

        dlon = lon - lon0
        if isinstance(dlon, u.Quantity):
            dlon = (dlon.to(u.degree)).value

        idx = np.logical_and(
            np.logical_or(dlon < 0, dlon > 180),
            np.logical_or(dlon > -180, dlon < -360),
        )
        dX[idx] = -dX[idx]

        idx = lat < lat0
        dY[idx] = -dY[idx]

        if isinstance(theta, u.Quantity):
            phi = (theta.to(u.degree)).value + 90.0
        else:
            phi = theta + 90.0

        cos2_phi = np.power(np.cos(phi * np.pi / 180.0), 2)
        sin2_phi = np.power(np.sin(phi * np.pi / 180.0), 2)

        sin_2phi = np.sin(2.0 * phi * np.pi / 180.0)

        A = old_div(cos2_phi, (2.0 * b**2)) + old_div(sin2_phi, (2.0 * a**2))
        
        B = old_div(-sin_2phi, (4.0 * b**2)) + old_div(sin_2phi, (4.0 * a**2))

        C = old_div(sin2_phi, (2.0 * b**2)) + old_div(cos2_phi, (2.0 * a**2))

        E = -A * np.power(dX, 2) + 2.0 * B * dX * dY - C * np.power(dY, 2)

        return np.power(old_div(180, np.pi), 2) * 1.0 / (2 * np.pi * a * b) * np.exp(E)

    def get_boundaries(self):

        # Truncate the gaussian at 2 times the max of sigma allowed

        min_lat = max(-90.0, self.lat0.value - 2 * self.a.max_value)
        max_lat = min(90.0, self.lat0.value + 2 * self.a.max_value)

        max_abs_lat = max(np.absolute(min_lat), np.absolute(max_lat))

        if (
            max_abs_lat > 89.0
            or 2 * self.a.max_value / np.cos(max_abs_lat * np.pi / 180.0) >= 180.0
        ):

            min_lon = 0.0
            max_lon = 360.0

        else:

            min_lon = self.lon0.value - 2 * self.a.max_value / np.cos(
                max_abs_lat * np.pi / 180.0
            )
            max_lon = self.lon0.value + 2 * self.a.max_value / np.cos(
                max_abs_lat * np.pi / 180.0
            )

            if min_lon < 0.0:

                min_lon = min_lon + 360.0

            elif max_lon > 360.0:

                max_lon = max_lon - 360.0

        return (min_lon, max_lon), (min_lat, max_lat)

    def get_total_spatial_integral(self, z=None):
        """
        Returns the total integral (for 2D functions) or the integral over the spatial components (for 3D functions).
        needs to be implemented in subclasses.

        :return: an array of values of the integral (same dimension as z).
        """

        if isinstance(z, u.Quantity):
            z = z.value
        return np.ones_like(z)        

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
        Set a new frame for the coordinates (the default is ICRS J2000)

        :param new_frame: a coordinate frame from astropy
        :return: (none)
        """
        assert isinstance(new_frame, BaseCoordinateFrame)

        self._frame = new_frame.name

    def set_version(self,v):
        
        """
        Set GALPROP version for input skymap. 

        "param v: version number, either 56 (default) or 54.
        """
        
        if not v in [54,57]:
            raise ValueError("GALPROP version must be 54 or 56.")

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

        if self._fitsfile == None:
            raise RuntimeError("Need to either specify or load a fits file")

        if self._file_loaded == False:
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

        if nside != None:
            # Get spatial grid from nside
            n_pixels = hp.nside2npix(nside) 
            ipix = np.arange(n_pixels)
            coords = hp.pix2ang(nside, ipix, lonlat=True)
            logger.info(f"using nside={nside} from user input in evaluate method")

        else:
            # Get spatial grid from GALPROP map:
            self.load_file(self._fitsfile)
            ipix = np.arange(self.n_pixels)
            coords = hp.pix2ang(self.nside, ipix, lonlat=True)
            logger.info(f"using nside={self.nside} from GALPROP map in evaluate method")

        x = coords[0]
        y = coords[1]
        
        intensity_3d = self.evaluate(x, y, z, self.K.value)

        # We are calculating the average intensity (and not the total in)
        intensity_2d = np.sum(intensity_3d,axis=0)

        if avg_int == True:
            intensity_2d /= len(intensity_3d) # return average intensity

        return intensity_2d

