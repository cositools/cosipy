import numpy as np
from astropy.coordinates import Angle, SkyCoord
import astropy.units as u
from astropy.stats import poisson_conf_interval
from astropy.time import Time
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from cosipy.polarization.polarization_angle import PolarizationAngle
from cosipy.polarization.conventions import MEGAlibRelativeX, MEGAlibRelativeY, MEGAlibRelativeZ, IAUPolarizationConvention
from cosipy.response import FullDetectorResponse
from threeML import LinearPolarization
from scoords import SpacecraftFrame
from histpy import Histogram

import logging
logger = logging.getLogger(__name__)

class PolarizationASAD():
    """
    Azimuthal scattering angle distribution (ASAD) method to fit polarization.

    Parameters
    ----------
    source_vector : astropy.coordinates.sky_coordinate.SkyCoord
        Source direction
    source_spectrum : astromodels.functions.functions_1D
        Spectrum of source
    asad_bin_edges : astropy.coordinates.angles.core.Angle
        Bin edges for azimuthal scattering angle distribution
    data : dict or cosipy.data_io.BinnedData
        Binned or unbinned data, or list of binned/unbinned data if separated in time
    background : dict or cosipy.data_io.BinnedData
        Binned or unbinned background model
    sc_orientation : cosipy.spacecraftfile.SpacecraftFile.SpacecraftFile
        Spacecraft orientation
    response_file : str or pathlib.Path
        Path to detector response
    response_convention : str, optional
        Polarization reference convention used in response ('RelativeX', 'RelativeY', or 'RelativeZ'). Default is 'RelativeX'
    fit_convention : cosipy.polarization.conventions.PolarizationConvention, optional
        Polarization reference convention to use for fit. Default is IAU convention
    show_plots : bool, optional
        Option to show plots. Default is False
    """

    def __init__(self, source_vector, source_spectrum, asad_bin_edges, data, background, sc_orientation, response_file, response_convention='RelativeX', fit_convention=IAUPolarizationConvention(), show_plots=False):

        if isinstance(fit_convention.frame, SpacecraftFrame) and not isinstance(source_vector.frame, SpacecraftFrame):
            attitude = sc_orientation.get_attitude()[0]
            source_vector = source_vector.transform_to(SpacecraftFrame(attitude=attitude))
            logger.warning('The source direction is being converted to the spacecraft frame using the attitude at the first timestamp of the orientation.')
        elif not isinstance(fit_convention.frame, SpacecraftFrame):
            source_vector = source_vector.transform_to('icrs')

        if ((isinstance(fit_convention, MEGAlibRelativeX) and response_convention != 'RelativeX') or 
            (isinstance(fit_convention, MEGAlibRelativeY) and response_convention != 'RelativeY') or 
            (isinstance(fit_convention, MEGAlibRelativeZ) and response_convention != 'RelativeZ')):
            raise RuntimeError("If performing fit in spacecraft frame, fit convention must match convention of response.")

        self._convention = fit_convention
        self._response_convention = response_convention

        self._source_vector = source_vector
        self._spectrum = source_spectrum

        if not type(data) == list:
            self._data = [data]
        else:
            self._data = data

        if not type(background) == list:
            self._background = [background]
        else:
            self._background = background

        self._asad_bin_edges = asad_bin_edges

        self._reference_vector = self._convention.get_basis(source_vector)[0] #px

        self._source_vector_cartesian = [self._source_vector.cartesian.x.value,
                                         self._source_vector.cartesian.y.value, 
                                         self._source_vector.cartesian.z.value]

        self._reference_vector_cartesian = [self._reference_vector.cartesian.x.value, 
                                            self._reference_vector.cartesian.y.value, 
                                            self._reference_vector.cartesian.z.value]

        self._response_file = response_file
        self._response = FullDetectorResponse.open(response_file)

        self._energy_range = [min(self._response.axes['Em'].edges.value), max(self._response.axes['Em'].edges.value)]

        self._ori = sc_orientation

        self._asads, source_duration, background_duration = self.create_asads()

        self._mu_100 = self.calculate_mu100(self._asads['polarized'], self._asads['unpolarized'], show_plots)

        if show_plots == True:
            titles = {'source': 'Source ASAD', 'source & background': 'Source+background ASAD', 'background': 'Background ASAD', 'unpolarized': 'Unpolarized ASAD'}
            for key in titles.keys():
                if key == 'source & background' or key == 'background':
                    self.plot_asad(self._asads[key].contents.data, titles[key], self._asads[key].bin_error[:])
                elif key == 'source':
                    self.plot_asad(self._asads[key].contents.data, titles[key], np.sqrt(self._asads['source & background'].bin_error[:]**2 + (self._asads['background'].bin_error[:] * source_duration / background_duration)**2))
                else:
                    self.plot_asad(self._asads[key].contents.data, titles[key])

        asad_corrected, self._sigma = self.correct_asad(self._asads['source'], self._asads['unpolarized'], np.sqrt(self._asads['source & background'].bin_error[:]**2 + (self._asads['background'].bin_error[:] * source_duration / background_duration)**2))

        self._asads['source (corrected)'] = asad_corrected

        self._mdp = self.calculate_mdp()

    def calculate_mdp(self):
        """
        Calculate the minimum detectable polarization (MDP) of the source.
        
        Returns
        -------
        mdp : float
            MDP of source
        """

        source_counts = np.sum(self._asads['source'].contents.data)
        background_counts = np.sum(self._asads['background (scaled)'].contents.data)

        mdp = 4.29 / self._mu_100['mu'] * np.sqrt(source_counts + background_counts) / source_counts

        logger.info('Minimum detectable polarization (MDP) of source: ' + str(round(mdp, 3)))

        return mdp

    def calculate_azimuthal_scattering_angles(self, unbinned_data):
        """
        Calculate the azimuthal scattering angles for all events in a dataset.
        
        Parameters
        ----------
        unbinned_data : dict
            Unbinned data including polar and azimuthal angles (radians) of scattered photon in local coordinates

        Returns
        -------
        azimuthal_angles : list of astropy.coordinates.Angle
            Azimuthal scattering angles
        """

        azimuthal_angles = []

        if isinstance(self._convention.frame, SpacecraftFrame):
            for i in range(len(unbinned_data['Psi local'])):
                if unbinned_data['Energies'][i] >= self._energy_range[0] and unbinned_data['Energies'][i] <= self._energy_range[1]:
                    psichi = SkyCoord(lat=(np.pi/2) - unbinned_data['Psi local'][i], lon=unbinned_data['Chi local'][i], unit=u.rad, frame=self._convention.frame)
                    azimuthal_angle = PolarizationAngle.from_scattering_direction(psichi, self._source_vector, self._convention)
                    azimuthal_angles.append(azimuthal_angle.angle)
        else:
            for i in range(len(unbinned_data['Psi galactic'])):
                if unbinned_data['Energies'][i] >= self._energy_range[0] and unbinned_data['Energies'][i] <= self._energy_range[1]:
                    psichi = SkyCoord(l=unbinned_data['Chi galactic'][i], b=unbinned_data['Psi galactic'][i], frame='galactic', unit=u.deg).transform_to('icrs')
                    azimuthal_angle = PolarizationAngle.from_scattering_direction(psichi, self._source_vector, self._convention)
                    azimuthal_angles.append(azimuthal_angle.angle)

        return azimuthal_angles

    def create_asad_from_response(self, spectrum, polarization_level, polarization_angle, bins=20):
        """
        Convolve source spectrum with response and calculate azimuthal scattering angle bins.

        Parameters
        ----------
        spectrum : :py:class:`threeML.Model`
            Spectral model.
        polarization_level : float
            Polarization level (between 0 and 1).
        polarization_angle : :py:class:`cosipy.polarization.polarization_angle.PolarizationAngle`
            Polarization angle. If in the spacecraft frame, the angle must have the same convention as the response.
        bins : int or astropy.units.quantity.Quantity, optional
            Number of azimuthal scattering angle bins if int or array of edges of azimuthal scattering angle bins if Quantity

        Returns
        -------
        asad : histpy.Histogram
            Counts in each azimuthal scattering angle bin
        """

        if isinstance(self._convention.frame, SpacecraftFrame):
            
            target_in_sc_frame = self._ori.get_target_in_sc_frame(target_name='source', target_coord=self._source_vector.transform_to('galactic'))
            dwell_time_map = self._ori.get_dwell_map(response=self._response_file, src_path=target_in_sc_frame)
            psr = self._response.get_point_source_response(exposure_map=dwell_time_map, coord=self._source_vector.transform_to('galactic'))
            expectation = psr.get_expectation(spectrum, LinearPolarization(polarization_level * 100., polarization_angle.angle.deg))
            
            azimuthal_angle_bins = []

            for i in range(expectation.axes['PsiChi'].nbins):
                psichi = SkyCoord(lat=(np.pi/2) - expectation.axes['PsiChi'].pix2ang(i)[0], lon=expectation.axes['PsiChi'].pix2ang(i)[1], unit=u.rad, frame=self._convention.frame)
                azimuthal_angle = PolarizationAngle.from_scattering_direction(psichi, self._source_vector, self._convention)
                azimuthal_angle_bins.append(azimuthal_angle.angle)
        
        else:
            
            scatt_map = self._ori.get_scatt_map(self._source_vector, nside=self._response.nside*2, coordsys='galactic')
            psr = self._response.get_point_source_response(coord=self._source_vector, scatt_map=scatt_map)
            expectation = psr.get_expectation(spectrum, LinearPolarization(polarization_level * 100., polarization_angle.angle.deg))

            azimuthal_angle_bins = []

            for i in range(expectation.axes['PsiChi'].nbins):
                psichi = expectation.axes['PsiChi'].pix2skycoord(i).transform_to('icrs')
                azimuthal_angle = PolarizationAngle.from_scattering_direction(psichi, self._source_vector, self._convention)
                azimuthal_angle_bins.append(azimuthal_angle.angle)

        if isinstance(bins, int):
            bin_edges = Angle(np.linspace(-np.pi, np.pi, bins), unit=u.rad)
        else:
            bin_edges = bins

        asad = []

        for i in range(len(bin_edges)-1):
            counts = 0
            for j in range(expectation.project(['PsiChi']).nbins):
                if azimuthal_angle_bins[j] >= bin_edges[i] and azimuthal_angle_bins[j] < bin_edges[i+1]:
                    counts += expectation.project(['PsiChi'])[j]
            asad.append(counts)

        asad = Histogram(bin_edges, contents=asad)

        return asad

    def bin_asad(self, azimuthal_angles, bins=20):
        """
        Bin list of azimuthal scattering angles into ASAD.
        
        Parameters
        ----------
        azimuthal_angles : list
            Azimuthal scattering angles 
        bins : int or astropy.units.quantity.Quantity, optional
            Number of azimuthal scattering angle bins if int or array of edges of azimuthal scattering angle bins if Quantity

        Returns
        -------
        asad : histpy.Histogram
            Counts in each azimuthal scattering angle bin
        """

        if isinstance(bins, int):
            bin_edges = Angle(np.linspace(-np.pi, np.pi, bins), unit=u.rad)
        else:
            bin_edges = bins

        counts, edges = np.histogram(azimuthal_angles, bins=bin_edges)

        asad = Histogram(edges, contents=counts)
        self._bin_edges = asad.axis.edges
        self._bins = asad.axis.centers

        return asad

    def create_asad_from_binned_data(self, data, bins=20):
        """
        Create ASAD from binned data.
        
        Parameters
        ----------
        data : cosipy.data_io.BinnedData
            Data binned in Compton data space
        bins : int or astropy.units.quantity.Quantity, optional
            Number of azimuthal scattering angle bins if int or array of edges of azimuthal scattering angle bins if Quantity

        Returns
        -------
        asad : histpy.Histogram
            Counts in each azimuthal scattering angle bin
        """

        if data.binned_data.axes['PsiChi'].coordsys.name == 'spacecraftframe':

            azimuthal_angle_bins = []
            for i in range(data.binned_data.axes['PsiChi'].nbins):
                psichi = SkyCoord(lat=(np.pi/2) - data.binned_data.axes['PsiChi'].pix2ang(i)[0], lon=data.binned_data.axes['PsiChi'].pix2ang(i)[1], unit=u.rad, frame=self._convention.frame)
                azimuthal_angle = PolarizationAngle.from_scattering_direction(psichi, self._source_vector, self._convention)
                azimuthal_angle_bins.append(azimuthal_angle.angle)

        else:

            azimuthal_angle_bins = []
            for i in range(data.binned_data.axes['PsiChi'].nbins):
                psichi = data.binned_data.axes['PsiChi'].pix2skycoord(i).transform_to('icrs')
                azimuthal_angle = PolarizationAngle.from_scattering_direction(psichi, self._source_vector, self._convention)
                azimuthal_angle_bins.append(azimuthal_angle.angle)

        if isinstance(bins, int):
            bin_edges = Angle(np.linspace(-np.pi, np.pi, bins), unit=u.rad)
        else:
            bin_edges = bins

        asad = []

        for i in range(len(bin_edges)-1):
            counts = 0
            for j in range(data.binned_data.project(['PsiChi']).nbins):
                if azimuthal_angle_bins[j] >= bin_edges[i] and azimuthal_angle_bins[j] < bin_edges[i+1]:
                    counts += data.binned_data.project(['PsiChi'])[j]
            asad.append(counts)

        asad = Histogram(bin_edges, contents=asad)
        self._bin_edges = asad.axis.edges
        self._bins = asad.axis.centers

        return asad

    def create_unpolarized_asad(self, bins=None):
        """
        Create unpolarized ASAD from response.
        
        Parameters
        ----------
        bins : int or astropy.units.quantity.Quantity, optional
            Number of azimuthal scattering angle bins if int or array of edges of azimuthal scattering angle bins if Quantity

        Returns
        -------
        asad : histpy.Histogram
            Counts in each azimuthal scattering angle bin
        """

        if not bins == None:
            if isinstance(bins, int):
                bin_edges = Angle(np.linspace(-np.pi, np.pi, bins), unit=u.rad)
            else:
                bin_edges = bins
        else:
            bin_edges = self._bin_edges

        unpolarized_asad = self.create_asad_from_response(self._spectrum, 0, PolarizationAngle(Angle(0 * u.deg), self._source_vector, convention=self._convention), bins)

        return unpolarized_asad
    
    def create_polarized_asads(self, bins=None):
        """
        Create 100% polarized ASADs for each polarization angle bin of response.
        
        Parameters
        ----------
        bins : int or astropy.units.quantity.Quantity, optional
            Number of azimuthal scattering angle bins if int or array of edges of azimuthal scattering angle bins if Quantity

        Returns
        -------
        polarized_asads : dict of histpy.Histogram
            Counts in each azimuthal scattering angle bin for each polarization angle bin
        """

        if not bins == None:
            if isinstance(bins, int):
                bin_edges = Angle(np.linspace(-np.pi, np.pi, bins), unit=u.rad)
            else:
                bin_edges = bins
        else:
            bin_edges = self._bin_edges
        
        polarized_asads = {}
        for k in range(self._response.axes['Pol'].nbins):
            polarized_asads[k] = self.create_asad_from_response(self._spectrum, 1, PolarizationAngle(Angle(self._response.axes['Pol'].centers.to_value(u.deg)[k] * u.deg), self._source_vector, convention=self._convention), bins)

        return polarized_asads

    def create_asads(self):
        """
        Create azimuthal scattering angle distributions from data, background model, and response.
            
        Returns
        -------
        asads : dict
            Azimuthal scattering angle distributions (ASADs)
        source_duration : float
            Duration of source
        background_duration : float
            Duration of background
        """

        asads = {}

        for i in range(len(self._data)):

            if type(self._data[i]) == dict:

                azimuthal_angles = self.calculate_azimuthal_scattering_angles(self._data[i])
                if i == 0:
                    asads['source & background'] = self.bin_asad(azimuthal_angles, self._asad_bin_edges)
                    source_duration = np.max(self._data[i]['TimeTags']) - np.min(self._data[i]['TimeTags'])
                else:
                    asads['source & background'] += self.bin_asad(azimuthal_angles, self._asad_bin_edges)
                    source_duration += np.max(self._data[i]['TimeTags']) - np.min(self._data[i]['TimeTags'])

            else:

                if i == 0:
                    asads['source & background'] = self.create_asad_from_binned_data(self._data[i], self._asad_bin_edges)
                    source_duration = (np.max(self._data[i].binned_data.axes['Time'].edges) - np.min(self._data[i].binned_data.axes['Time'].edges)).value
                else:
                    asads['source & background'] += self.create_asad_from_binned_data(self._data[i], self._asad_bin_edges)
                    source_duration += (np.max(self._data[i].binned_data.axes['Time'].edges) - np.min(self._data[i].binned_data.axes['Time'].edges)).value

        for i in range(len(self._background)):

            if type(self._background[i]) == dict:

                azimuthal_angles = self.calculate_azimuthal_scattering_angles(self._background[i])
                if i == 0:
                    asads['background'] = self.bin_asad(azimuthal_angles, self._asad_bin_edges)
                    background_duration = np.max(self._background[i]['TimeTags']) - np.min(self._background[i]['TimeTags'])
                else:
                    asads['background'] += self.bin_asad(azimuthal_angles, self._asad_bin_edges)
                    background_duration += np.max(self._background[i]['TimeTags']) - np.min(self._background[i]['TimeTags'])

            else:

                if i == 0:
                    asads['background'] = self.create_asad_from_binned_data(self._background[i], self._asad_bin_edges)
                    background_duration = (np.max(self._background[i].binned_data.axes['Time'].edges) - np.min(self._background[i].binned_data.axes['Time'].edges)).value
                else:
                    asads['background'] += self.create_asad_from_binned_data(self._background[i], self._asad_bin_edges)
                    background_duration += (np.max(self._background[i].binned_data.axes['Time'].edges) - np.min(self._background[i].binned_data.axes['Time'].edges)).value

        scaled_background_asad = (asads['background'].contents.data * source_duration / background_duration).astype(int)
        source_asad = asads['source & background'].contents.data - scaled_background_asad

        asads['source'] = Histogram(asads['background'].axis.edges, contents=source_asad)
        asads['unpolarized'] = self.create_unpolarized_asad(self._bin_edges)
        asads['polarized'] = self.create_polarized_asads(self._bin_edges)
        asads['background (scaled)'] = Histogram(asads['background'].axis.edges, contents=scaled_background_asad)

        return asads, source_duration, background_duration

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
    
    def plot_asad(self, counts, title, error=None, coefficients=[]):
        """
        Plot the ASAD.
        
        Parameters
        ----------
        counts : list
            Counts in each azimuthal scattering angle bin
        title : str
            Title of plot
        error : float, list, or np.array
            Uncertainties for each bin
        coefficients : list, optional
            Coefficients to plot fitted sinusoidal function
        """

        plt.scatter(Angle(self._bins).degree, counts)
        if not error is None:
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

    def correct_asad(self, data_asad, unpolarized_asad, data_asad_uncertainties=None):
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
        asad : histpy.Histogram
            Normalized counts in each azimuthal scattering angle bin
        """
    
        corrected = []
        uncertainties = []
        for i in range(len(self._bins)):
            corrected.append(data_asad.contents.data[i] / np.sum(data_asad.contents.data) / unpolarized_asad.contents.data[i] * np.sum(unpolarized_asad.contents.data))
            if not data_asad_uncertainties is None:
                uncertainties.append(data_asad_uncertainties[i] / np.sum(data_asad.contents.data) / unpolarized_asad.contents.data[i] * np.sum(unpolarized_asad.contents.data))
        
        asad = Histogram(data_asad.axis.edges, contents=corrected)

        if not data_asad_uncertainties is None:
            return asad, uncertainties
        else:
            return asad

    def calculate_mu(self, counts_corrected, p0=None, bounds=None, sigma=None):
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
        parameter_values : np.ndarray
            Fitted parameter values
        """

        parameter_values, uncertainties = self.fit_asad(counts_corrected, p0, bounds, sigma)
    
        mu = parameter_values[1] / parameter_values[0]
        mu_uncertainty = mu * np.sqrt((uncertainties[0]/parameter_values[0])**2 + (uncertainties[1]/parameter_values[1])**2)

        modulation = {'mu': mu, 'uncertainty': mu_uncertainty}

        logger.info('Modulation:', round(mu, 3), '+/-', round(mu_uncertainty, 3))
    
        return modulation, parameter_values
    
    def constant(self, x, a):
        """
        Constant function to fit to mu_100 values.
        
        Parameters
        ----------
        x : float
            Mu_100
        a : float
            Parameter
            
        Returns
        -------
        a : float
            Constant value
        """

        return a
    
    def calculate_mu100(self, polarized_asads, unpolarized_asad, show_plots=False):
        """
        Calculate the modulation (mu) of an 100% polarized source.
        
        Parameters
        ----------
        polarized_asads : list
            Counts and Gaussian/Poisson errors in each azimuthal scattering angle bin for each polarization angle bin for 100% polarized source
        unpolarized_asad : list or np.array
            Counts and Gaussian/Poisson errors in each azimuthal scattering angle bin for unpolarized source
        show_plots : bool, optional
            Option to show plots. Default is False
            
        Returns
        -------
        mu_100 : dict
            Modulation of 100% polarized source and uncertainty of constant function fit to modulation in all polarization angle bins
        """

        mu_100_list = []
        mu_100_uncertainties = []
        for i in range(self._response.axes['Pol'].nbins):
            logger.info('Polarization angle bin: ' + str(self._response.axes['Pol'].edges.to_value(u.deg)[i]) + ' to ' + str(self._response.axes['Pol'].edges.to_value(u.deg)[i+1]) + ' deg')
            #asad_polarized = {'counts': polarized_asads['counts'][i], 'uncertainties': polarized_asads['uncertainties'][i]}
            asad_polarized_corrected = self.correct_asad(polarized_asads[i], unpolarized_asad)
            mu_100, coefficients = self.calculate_mu(asad_polarized_corrected.contents.data, bounds=([0, 0, 0], [np.inf,np.inf,np.pi]))
            fitted_angle = Angle(coefficients[2], unit=u.rad)
            fitted_angle.wrap_at(180 * u.deg, inplace=True)
            if fitted_angle.degree < 0:
                fitted_angle += Angle(180, unit=u.deg)
            logger.info('Fitted angle: ' + str(fitted_angle.degree) + ' deg')
            mu_100_list.append(mu_100['mu'])
            mu_100_uncertainties.append(mu_100['uncertainty'])
            if show_plots == True:
                self.plot_asad(asad_polarized_corrected.contents.data, 'Corrected 100% Polarized ASAD (' + str(int(self._response.axes['Pol'].centers[i].to_value(u.deg))) + ' deg)', coefficients=coefficients)

        popt, pcov = curve_fit(self.constant, self._response.axes['Pol'].centers.to_value(u.deg), mu_100_list, sigma=mu_100_uncertainties)
        mu_100 = {'mu': popt[0], 'uncertainty': pcov[0][0]}

        if show_plots == True:
            plt.scatter(self._response.axes['Pol'].centers.to_value(u.deg), mu_100_list)
            plt.errorbar(self._response.axes['Pol'].centers.to_value(u.deg), mu_100_list, yerr=mu_100_uncertainties, linewidth=0, elinewidth=1)
            plt.plot([0, 175], [mu_100['mu'], mu_100['mu']])
            plt.xlabel('Polarization Angle (degrees)')
            plt.ylabel('mu_100')
            plt.show()

        logger.info('mu_100:', round(mu_100['mu'], 2))

        return mu_100

    def fit(self, p0=None, bounds=([0, 0, 0], [np.inf,np.inf,np.pi]), show_plots=False):
        """
        Fit the polarization fraction and angle.
        
        Parameters
        ----------
        p0 : list or np.array, optional
            Initial guess for parameter values
        bounds : 2-tuple of float, list, or np.array, optional
            Lower & upper bounds on parameters. Default is ([0, 0, 0], [np.inf,np.inf,np.pi])
        show_plots : bool, optional
            Option to show plots. Default is False
            
        Returns
        -------
        polarization : dict
            Polarization fraction, polarization angle in the IAU convention, and best fit parameter values for fitted sinusoid, and associated uncertainties
        """

        parameter_values, uncertainties = self.fit_asad(self._asads['source (corrected)'].contents.data, p0, bounds, self._sigma)
    
        polarization_fraction = parameter_values[1] / (parameter_values[0] * self._mu_100['mu'])
        polarization_fraction_uncertainty = polarization_fraction * np.sqrt((uncertainties[0]/parameter_values[0])**2 + (uncertainties[1]/parameter_values[1])**2 + (self._mu_100['uncertainty']/self._mu_100['mu'])**2)

        polarization_angle = Angle(parameter_values[2], unit=u.rad)
        polarization_angle.wrap_at(180 * u.deg, inplace=True)
        if polarization_angle.degree < 0:
            polarization_angle += Angle(180, unit=u.deg)
        polarization_angle = PolarizationAngle(polarization_angle, self._source_vector, convention=self._convention).transform_to(IAUPolarizationConvention())
        polarization_angle_uncertainty = Angle(uncertainties[2], unit=u.rad)

        polarization = {'fraction': polarization_fraction, 'angle': polarization_angle, 'fraction uncertainty': polarization_fraction_uncertainty, 'angle uncertainty': polarization_angle_uncertainty, 'best fit parameter values': parameter_values, 'best fit parameter uncertainties': uncertainties}
    
        logger.info('Best fit polarization fraction:', round(polarization_fraction, 3), '+/-', round(polarization_fraction_uncertainty, 3))
        logger.info('Best fit polarization angle (IAU convention):', round(polarization_angle.angle.degree, 3), '+/-', round(polarization_angle_uncertainty.degree, 3))

        if self._mdp > polarization['fraction']:
            logger.info('Polarization fraction is below MDP!', 'MDP:', round(self._mdp, 3))

        if show_plots == True:
            self.plot_asad(self._asads['source (corrected)'].contents.data, 'Corrected Source ASAD', self._sigma, coefficients=polarization['best fit parameter values'])
        
        return polarization
