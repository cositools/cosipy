from astromodels.functions.function import Function1D, FunctionMeta, ModelAssertionViolation,Function2D
import astromodels.functions.numba_functions as nb_func
from astromodels.utils.angular_distance import angular_distance
from threeML import Band, DiracDelta, Constant, Line, Quadratic, Cubic, Quartic, StepFunction, StepFunctionUpper, Cosine_Prior, Uniform_prior, PhAbs, Gaussian
import astropy.units as astropy_units
from astropy.units import Quantity
from past.utils import old_div
from scipy.special import gammainc, expi
from scipy.interpolate import interp1d
from scipy import integrate
import numpy as np
import math
import astropy.units as u
from astropy.constants import h
import pandas as pd
import matplotlib.pyplot as plt

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
        

class SpectrumFileProcessor:
    def __init__(self, input_file, reformatted_file, energy_col=0, flux_col=1, convert_data=True):
        """
        Initialize the SpectrumFileProcessor class with default columns for energy and flux.
        
        Parameters:
        - input_file: Path to the input .dat file.
        - reformatted_file: Path to save the reformatted data.
        - energy_col: Index of the column containing energy or frequency (default: 0).
        - flux_col: Index of the column containing flux (default: 1).
        - convert_data: Boolean flag indicating if conversion is needed (default: True).
        """
        self.input_file = input_file
        self.reformatted_file = reformatted_file
        self.data = None
        self.df_filtered = None
        self.energy_col = energy_col  # Default to column 0 for energy
        self.flux_col = flux_col      # Default to column 1 for flux
        self.convert_data = convert_data  # Whether conversion is necessary

    def load_data(self):
        """Load the data from the .dat file."""
        try:
            self.data = np.loadtxt(self.input_file)
            print(f"Data loaded successfully from {self.input_file}")
        except Exception as e:
            print(f"Error loading file {self.input_file}: {e}")
            raise

    def process_data(self):
        """Convert frequency to energy in keV and flux to ph/cm²/sec/keV."""
        energy_hz = self.data[:, self.energy_col] * u.Hz   # Column defined by user
        flux_ergs = self.data[:, self.flux_col] * u.erg / (u.cm**2 * u.s)  # Column defined by user
        
        # Convert frequency to energy in keV
        energy_keV = (h * energy_hz).to(u.keV)

        # Convert flux from ergs to keV
        flux_keV = flux_ergs.to(u.keV / (u.cm**2 * u.s))

        # Convert flux to ph/cm²/sec/keV
        flux_ph = (flux_keV / energy_keV**2)
        
        # Create a DataFrame to store the converted data
        df = pd.DataFrame({
            'Energy (keV)': energy_keV.value,
            'Flux (ph/cm²/sec/keV)': flux_ph.value
        })
        
        # Filter out rows with energy less than 100 keV and more than 10000 keV
        self.df_filtered = df[(df['Energy (keV)'] >= 100) & (df['Energy (keV)'] <= 10000)]
        return self.df_filtered
    
    def integrate_flux(self):
        """Perform numerical integration of the flux over the energy range using the trapezoidal rule."""
        if self.df_filtered is None or self.df_filtered.empty:
            raise ValueError("No data to integrate. Please run process_data() first.")
        
        energy = self.df_filtered['Energy (keV)'].values
        flux = self.df_filtered['Flux (ph/cm²/sec/keV)'].values

        integrated_flux = np.trapz(flux, energy)
        print(f"Integrated Flux (Normalization Constant): {integrated_flux} ph/cm²/sec")

        return integrated_flux

    def plot_spectrum(self):
        """Generate a plot of energy vs flux with a log-log scale."""
        if self.df_filtered is not None:
            plt.figure(figsize=(10, 6))
            plt.plot(self.df_filtered['Energy (keV)'], self.df_filtered['Flux (ph/cm²/sec/keV)'], marker="o", linestyle="-")
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel("Energy (keV)")
            plt.ylabel("Flux (ph/cm²/sec/keV)")
            plt.title("Energy Flux Data")
            plt.grid(True)
            plt.show()
        else:
            print("No data available for plotting. Please run process_data first.")

    def reformat_data(self):
        """Reformat the data for the photon spectrum file."""
        try:
            formatted_lines = []
            formatted_lines.append("-Ps photon spectrum file")
            formatted_lines.append("#")
            formatted_lines.append("# Format: DP <energy in keV> <shape of O-Ps [XX/keV]>")
            formatted_lines.append("")
            formatted_lines.append("IP LIN")
            formatted_lines.append("")
            
            # Iterate through the DataFrame to format each line as required
            for index, row in self.df_filtered.iterrows():
                energy = row['Energy (keV)']
                flux = row['Flux (ph/cm²/sec/keV)']
                formatted_line = f"DP\t{energy:.5e}\t{flux:.5e}"  # Add DP and reformat the line
                formatted_lines.append(formatted_line)
            
            # Append the closing 'EN' line
            formatted_lines.append("EN")
            
            # Save the new formatted data to a file
            with open(self.reformatted_file, 'w') as f:
                f.write("\n".join(formatted_lines))
            
            print(f"Reformatted data saved to {self.reformatted_file}")
        
        except Exception as e:
            print(f"Error during reformatting: {e}")
            raise