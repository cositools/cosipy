import numpy as np
from astropy.coordinates import Angle
import astropy.units as u
from astropy.stats import poisson_conf_interval
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from cosipy.polarization import PolarizationAngle
from cosipy.polarization.conventions import MEGAlibRelativeX, IAUPolarizationConvention
from scoords import SpacecraftFrame

def calculate_uncertainties(counts):
    """
    Calculate the Poisson/Gaussian uncertainties for a list of binned counts.
        
    Parameters
    ----------
    counts : list
        List of counts in each bin

    Returns
    -------
    uncertainties : np.ndarray
        Lower & upper uncertainties for each bin
    """
    
    uncertainties_low = []
    uncertainties_high = []
    for i in range(len(counts)):
        if counts[i] <= 5:
            poisson_uncertainty = poisson_conf_interval(counts[i], interval="frequentist-confidence", sigma=1)
            uncertainties_low.append(counts[i] - poisson_uncertainty[0])
            uncertainties_high.append(poisson_uncertainty[1] - counts[i])
        else:
            gaussian_uncertainty = np.sqrt(counts[i])
            uncertainties_low.append(gaussian_uncertainty)
            uncertainties_high.append(gaussian_uncertainty)

    uncertainties = np.array([uncertainties_low, uncertainties_high])
        
    return uncertainties

class PolarizationASAD():
    """
    Azimuthal scattering angle distribution (ASAD) method to fit polarization.

    Parameters
    ----------
    source_vector : astropy.coordinates.sky_coordinate.SkyCoord
        Source direction
    fit_convention : cosipy.polarization.conventions.PolarizationConvention, optional
        Polarization reference convention to use for fit. Default is RelativeX
    """

    def __init__(self, source_vector, fit_convention=None):
        
        if fit_convention == None:
            self._convention = MEGAlibRelativeX(attitude=source_vector.attitude)
        else:
            self._convention = fit_convention

        reference_vector = self._convention.get_basis(source_vector)[0] #px

        if isinstance(source_vector.frame, SpacecraftFrame):
            self._source_vector = source_vector
        else:
            self._source_vector = source_vector.transform_to(SpacecraftFrame(attitude=source_vector.attitude))
        
        if isinstance(reference_vector.frame, SpacecraftFrame):
            self._reference_vector = reference_vector
        else:
            self._reference_vector = reference_vector.transform_to(SpacecraftFrame(attitude=source_vector.attitude))

        self._source_vector_cartesian = [self._source_vector.cartesian.x.value,
                                         self._source_vector.cartesian.y.value, 
                                         self._source_vector.cartesian.z.value]
        self._reference_vector_cartesian = [self._reference_vector.cartesian.x.value, 
                                            self._reference_vector.cartesian.y.value, 
                                            self._reference_vector.cartesian.z.value]

    def calculate_azimuthal_scattering_angle(self, psi, chi):
        """
        Calculate the azimuthal scattering angle of a scattered photon.
        
        Parameters
        ----------
        psi : float
            Polar angle (radians) of scattered photon in local coordinates
        chi : float
            Azimuthal angle (radians) of scattered photon in local coordinates

        Returns
        -------
        azimuthal_angle : astropy.coordinates.Angle
            Azimuthal scattering angle defined with respect to given reference vector
        """
        
        # Convert scattered photon vector from spherical to Cartesian coordinates
        scattered_photon_vector = [np.sin(psi) * np.cos(chi), np.sin(psi) * np.sin(chi), np.cos(psi)]

        # Project scattered photon vector onto plane perpendicular to source direction
        d = np.dot(scattered_photon_vector, self._source_vector_cartesian) / np.dot(self._source_vector_cartesian, self._source_vector_cartesian)
        projection = [scattered_photon_vector[0] - (d * self._source_vector_cartesian[0]), 
                      scattered_photon_vector[1] - (d * self._source_vector_cartesian[1]), 
                      scattered_photon_vector[2] - (d * self._source_vector_cartesian[2])]

        # Calculate angle between scattered photon vector & reference vector on plane perpendicular to source direction
        cross_product = np.cross(projection, self._reference_vector_cartesian)
        if np.dot(self._source_vector_cartesian, cross_product) < 0:
            sign = -1
        else:
            sign = 1
        normalization = np.sqrt(np.dot(projection, projection)) * np.sqrt(np.dot(self._reference_vector_cartesian, self._reference_vector_cartesian))
    
        azimuthal_angle = Angle(sign * np.arccos(np.dot(projection, self._reference_vector_cartesian) / normalization), unit=u.rad)
    
        return azimuthal_angle

    def calculate_azimuthal_scattering_angles(self, unbinned_data):
        """
        Calculate the azimuthal scattering angles for all events in a dataset.
        
        Parameters
        ----------
        unbinned_data : dict
            Unbinned data including polar and azimuthal angles (radians) of scattered photon in local coordinates

        Returns
        -------
        azimuthal_angles : list
            Azimuthal scattering angles. Each angle must be an astropy.coordinates.Angle object
        """

        azimuthal_angles = []

        for i in range(len(unbinned_data['Psi local'])):
            azimuthal_angle = self.calculate_azimuthal_scattering_angle(unbinned_data['Psi local'][i], unbinned_data['Chi local'][i])
            azimuthal_angles.append(azimuthal_angle)

        return azimuthal_angles

    def create_asad(self, azimuthal_angles, bins=20):
        """
        Create ASAD and calculate uncertainties.
        
        Parameters
        ----------
        azimuthal_angles : list
            Azimuthal scattering angles (radians)
        bin_edges : np.array
            Edges of azimuthal scattering angle bins (radians)

        Returns
        -------
        asad : dict
            Counts and Gaussian/Poisson errors in each bin
        """

        if isinstance(bins, int):
            bin_edges = Angle(np.linspace(-np.pi, np.pi, bins), unit=u.rad)
        else:
            bin_edges = bins

        counts, edges = np.histogram(azimuthal_angles, bins=bin_edges)
        self._bin_edges = edges
        self._bins = []
        for i in range(len(self._bin_edges) - 1):
            self._bins.append((self._bin_edges[i] + self._bin_edges[i+1]) / 2)
        errors = calculate_uncertainties(counts)

        asad = {'counts': counts, 'uncertainties': errors}

        return asad

    def asad_sinusoid(self, x, a, b, c):
        """
        Sinusoid to fit to ASAD.
        
        Parameters
        ----------
        x : float
            Azimuthal scattering angle (radians)
        a : float
            First parameter
        b : float
            Second parameter
        c : float
            Third parameter
            
        Returns
        -------
        asad_function : float
            Y-value of ASAD
        """

        asad_function = a - (b * np.cos(2 * (x - c)))
    
        return asad_function
    
    def fit_asad(self, counts, p0, bounds, sigma):
        """
        Fit the ASAD with a sinusoid.
        
        Parameters
        ----------
        counts : list
            Counts in each azimuthal scattering angle bin
        p0 : list or np.array
            Initial guess for parameter values
        bounds : 2-tuple of float, list, or np.array
            Lower & upper bounds on parameters
        sigma : float, list, or np.array
            Uncertainties in y data
            
        Returns
        -------
        popt : np.ndarray
            Fitted parameter values
        uncertainties : list
            Uncertainty on each parameter value
        """

        popt, pcov = curve_fit(self.asad_sinusoid, Angle(self._bins).rad, counts, p0=p0, bounds=bounds, sigma=sigma)
        uncertainties = [] 
        for i in range(len(pcov)):
            uncertainties.append(np.sqrt(pcov[i][i]))

        return popt, uncertainties
    
    def plot_asad(self, counts, error, title, coefficients=[]):
        """
        Plot the ASAD.
        
        Parameters
        ----------
        counts : list
            Counts in each azimuthal scattering angle bin
        error : np.ndarray
            Lower & upper uncertainties for each bin
        title : str
            Title of plot
        coefficients : list, optional
            Coefficients to plot fitted sinusoidal function
        """

        plt.scatter(Angle(self._bins).degree, counts)
        plt.errorbar(Angle(self._bins).degree, counts, yerr=error, linewidth=0, elinewidth=1)
        plt.title(title)
        plt.xlabel('Azimuthal Scattering Angle (degrees)')
        
        if len(coefficients) == 3:
            x = np.linspace(-np.pi, np.pi, 1000)
            y = []
            for item in x:
                y.append(self.asad_sinusoid(item, coefficients[0], coefficients[1], coefficients[2]))
            plt.plot(list(np.rad2deg(x)), y, color='green')

        plt.show()

    def correct_asad(self, data_asad, unpolarized_asad):
        """
        Correct the ASAD using the ASAD of an unpolarized source.
        
        Parameters
        ----------
        data_asad : dict
            Counts and uncertainties in each azimuthal scattering angle bin of data
        unpolarized_asad : dict
            Counts and uncertainties in each azimuthal scattering angle bin of unpolarized source
            
        Returns
        -------
        asad : dict
            Normalized counts and uncertainties in each azimuthal scattering angle bin
        """
    
        corrected = []
        for i in range(len(self._bins)):
            corrected.append(data_asad['counts'][i] / np.sum(data_asad['counts']) / unpolarized_asad['counts'][i] * np.sum(unpolarized_asad['counts']))
  
        errors_low = []
        errors_high = []
        for i in range(len(self._bins)):
            sigma_corrected_low = corrected[i] * np.sqrt(((data_asad['uncertainties'][0][i])/data_asad['counts'][i])**2 + ((unpolarized_asad['uncertainties'][0][i])/unpolarized_asad['counts'][i])**2)
            sigma_corrected_high = corrected[i] * np.sqrt(((data_asad['uncertainties'][1][i])/data_asad['counts'][i])**2 + ((unpolarized_asad['uncertainties'][1][i])/unpolarized_asad['counts'][i])**2)
            errors_low.append(sigma_corrected_low)
            errors_high.append(sigma_corrected_high)

        asad = {'counts': corrected, 'uncertainties': np.array([errors_low, errors_high])}

        return asad

    def calculate_mu(self, counts_corrected, p0=None, bounds=(-np.inf, np.inf), sigma=None):
        """
        Calculate the modulation (mu).
        
        Parameters
        ----------
        counts_corrected : list
            Counts in each azimuthal scattering angle bin
        p0 : list or np.array
            Initial guess for parameter values
        bounds : 2-tuple of float, list, or np.array
            Lower & upper bounds on parameters
        sigma : float, list, or np.array
            Uncertainties for each azimuthal scattering angle bin
            
        Returns
        -------
        modulation : dict
            Modulation and uncertainty of fitted sinusoid
        """

        if isinstance(sigma, np.ndarray) and len(sigma.shape) == 2:
            for i in range(len(sigma[0])):
                if sigma[0][i] != sigma[1][i]:
                    print('Warning: Uncertainty in at least one bin of ASAD is not Gaussian. Making error bars symmetric. Fit may not be accurate.')
                    break
            symmetric_sigma = []
            for i in range(len(sigma[0])):
                symmetric_sigma.append((sigma[0][i] + sigma[1][i]) / 2)
            sigma = symmetric_sigma

        parameter_values, uncertainties = self.fit_asad(counts_corrected, p0, bounds, sigma)
    
        mu = parameter_values[1] / parameter_values[0]
        mu_uncertainty = mu * np.sqrt((uncertainties[0]/parameter_values[0])**2 + (uncertainties[1]/parameter_values[1])**2)

        modulation = {'mu': mu, 'uncertainty': mu_uncertainty}

        print('Modulation:', round(mu, 3), '+/-', round(mu_uncertainty, 3))
    
        return modulation

    def fit(self, mu_100, counts_corrected, p0=None, bounds=(-np.inf, np.inf), sigma=None):
        """
        Fit the polarization fraction and angle.
        
        Parameters
        ----------
        mu_100 : dict
            Modulation and uncertainty of a 100% polarized source
        counts_corrected : list
            Counts in each azimuthal scattering angle bin
        p0 : list or np.array
            Initial guess for parameter values
        bounds : 2-tuple of float, list, or np.array
            Lower & upper bounds on parameters
        sigma : float, list, or np.array
            Uncertainties for each azimuthal scattering angle bin
            
        Returns
        -------
        polarization : dict
            Polarization fraction, polarization angle, and best fit parameter values for fitted sinusoid, and associated uncertainties
        """

        if isinstance(sigma, np.ndarray) and len(sigma.shape) == 2:
            for i in range(len(sigma[0])):
                if sigma[0][i] != sigma[1][i]:
                    print('Warning: Uncertainty in at least one bin of ASAD is not Gaussian. Making error bars symmetric. Fit may not be accurate.')
                    break
            symmetric_sigma = []
            for i in range(len(sigma[0])):
                symmetric_sigma.append((sigma[0][i] + sigma[1][i]) / 2)
            sigma = symmetric_sigma

        parameter_values, uncertainties = self.fit_asad(counts_corrected, p0, bounds, sigma)
    
        polarization_fraction = parameter_values[1] / (parameter_values[0] * mu_100['mu'])
        polarization_fraction_uncertainty = polarization_fraction * np.sqrt((uncertainties[0]/parameter_values[0])**2 + (uncertainties[1]/parameter_values[1])**2 + (mu_100['uncertainty']/mu_100['mu'])**2)

        polarization_angle = Angle(parameter_values[2], unit=u.rad)
        polarization_angle.wrap_at(180 * u.deg, inplace=True)
        if polarization_angle.degree < 0:
            polarization_angle += Angle(180, unit=u.deg)
        polarization_angle = PolarizationAngle(polarization_angle, self._source_vector, convention=self._convention).transform_to(IAUPolarizationConvention())
        polarization_angle_uncertainty = Angle(uncertainties[2], unit=u.rad)

        polarization = {'fraction': polarization_fraction, 'angle': polarization_angle, 'fraction uncertainty': polarization_fraction_uncertainty, 'angle uncertainty': polarization_angle_uncertainty, 'best fit parameter values': parameter_values, 'best fit parameter uncertainties': uncertainties}
    
        print('Best fit polarization fraction:', round(polarization_fraction, 3), '+/-', round(polarization_fraction_uncertainty, 3))
        print('Best fit polarization angle:', round(polarization_angle.angle.degree, 3), '+/-', round(polarization_angle_uncertainty.degree, 3))
        
        return polarization
