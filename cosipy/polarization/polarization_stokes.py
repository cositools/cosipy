import numpy as np
from astropy.coordinates import Angle
import astropy.units as u
# from astropy.stats import poisson_conf_interval
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from cosipy.polarization import PolarizationAngle
from cosipy.polarization.conventions import MEGAlibRelativeX, IAUPolarizationConvention
from cosipy.response import FullDetectorResponse
from scoords import SpacecraftFrame
import scipy.interpolate as interpolate


#we can define all these functions in a separate file to import

def R(x, A, B, C):
    """
    """
    return A + B*(np.cos(x + C)**2)

def constant(x, a):
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

def rotate_points_to_x_axis(x_, y_, angle_):
    """
    Rotate arrays of points (x_, y_) in the QN-UN plane by an angle
    """    
    # Create a matrix of rotation matrices for each point
    cos_vals = np.cos(2*angle_)
    sin_vals = np.sin(2*angle_)
    
    # Apply the rotation to each point
    rotated_x = x_ * cos_vals - y_ * sin_vals
    rotated_y = x_ * sin_vals + y_ * cos_vals
    
    return rotated_x, rotated_y

def polar_chart_backbone(ax):
    """ Preparing canvas for Stokes chart
    """        
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    c0 = plt.Circle((0,0), radius=0.25, facecolor='none', edgecolor='k', linewidth=1, linestyle='--', alpha=0.3)
    c1 = plt.Circle((0,0), radius=0.50, facecolor='none', edgecolor='k', linewidth=1, linestyle='--', alpha=0.3)
    c2 = plt.Circle((0,0), radius=0.75, facecolor='none', edgecolor='k', linewidth=1, linestyle='--', alpha=0.3)
    c3 = plt.Circle((0,0), radius=1.00, facecolor='none', edgecolor='k', linewidth=1, linestyle='-', alpha=0.5)
    plt.gca().add_artist(c0)
    plt.gca().add_artist(c1)
    plt.gca().add_artist(c2)
    plt.gca().add_artist(c3)
    plt.annotate('0.25', (0.25, 0), textcoords="offset points", xytext=(10,0), ha='center', fontsize=8, color='k', alpha=0.3)
    plt.annotate('0.50', (0.50, 0), textcoords="offset points", xytext=(10,0), ha='center', fontsize=8, color='k', alpha=0.3)
    plt.annotate('0.75', (0.75, 0), textcoords="offset points", xytext=(10,0), ha='center', fontsize=8, color='k', alpha=0.3)
    plt.annotate('1.00', (1.00, 0), textcoords="offset points", xytext=(10,0), ha='center', fontsize=8, color='k', alpha=0.3)
    plt.hlines(0, -1, 1, linewidth=1, color='k', linestyle='--', alpha=0.3)
    plt.vlines(0, -1, 1, linewidth=1, color='k', linestyle='--', alpha=0.3)
    plt.plot([1,-1], [1,-1], linewidth=1, color='k', linestyle='--', alpha=0.3)
    plt.plot([1,-1], [-1,1], linewidth=1, color='k', linestyle='--', alpha=0.3)

def calculate_azimuthal_scattering_angle(psi, chi, source_vector, reference_vector):
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
        
        source_vector_cartesian = [source_vector.cartesian.x.value,
                                   source_vector.cartesian.y.value, 
                                   source_vector.cartesian.z.value]
        reference_vector_cartesian = [reference_vector.cartesian.x.value, 
                                      reference_vector.cartesian.y.value, 
                                      reference_vector.cartesian.z.value]
        
        # Convert scattered photon vector from spherical to Cartesian coordinates
        scattered_photon_vector = [np.sin(psi) * np.cos(chi), np.sin(psi) * np.sin(chi), np.cos(psi)]

        # Project scattered photon vector onto plane perpendicular to source direction
        d = np.dot(scattered_photon_vector, source_vector_cartesian) / np.dot(source_vector_cartesian, source_vector_cartesian)
        projection = [scattered_photon_vector[0] - (d * source_vector_cartesian[0]), 
                      scattered_photon_vector[1] - (d * source_vector_cartesian[1]), 
                      scattered_photon_vector[2] - (d * source_vector_cartesian[2])]

        # Calculate angle between scattered photon vector & reference vector on plane perpendicular to source direction
        cross_product = np.cross(projection, reference_vector_cartesian)
        if np.dot(source_vector_cartesian, cross_product) < 0:
            sign = -1
        else:
            sign = 1
        normalization = np.sqrt(np.dot(projection, projection)) * np.sqrt(np.dot(reference_vector_cartesian, reference_vector_cartesian))
    
        azimuthal_angle = Angle(sign * np.arccos(np.dot(projection, reference_vector_cartesian) / normalization), unit=u.rad)
    
        return azimuthal_angle

def get_modulation(_x, _y, title='Modulation', show=False):
    """ Function to estimate the modulation factor.
        _x is the central value of the histogram bins
        _y is the value of the bins on the histograms
    """
    _x = _x[:-1] + (_x[1:] - _x[:-1])/2
    popt, pcov = curve_fit(R, _x, _y ) #sigma=np.sqrt(_y), absolute_sigma=True
    pcov[0][0], pcov[1][1], pcov[2][2] = np.sqrt(pcov[0][0]), np.sqrt(pcov[1][1]), np.sqrt(pcov[2][2])
    print('A = %.2f, B = %.2f, C = %.2f'%(popt[0], popt[1], popt[2]))

    Rmax, Rmin = np.amax(R(_x, *popt)), np.amin(R(_x, *popt))
    print('Rmax, Rmin:', Rmax, Rmin)
    mu = (Rmax-Rmin)/(Rmax+Rmin)
    print('Modulation mu = ', mu)
    
    mu_err = 2/(popt[1]+2*popt[0])**2 * np.sqrt(popt[1]**2 * pcov[0][0]**2 + popt[0]**2 * pcov[1][1]**2)
    
    if show:

        plt.figure()
        plt.title(title)
        plt.step(_x, _y, where='mid')
        perr = [popt[0]+np.sqrt(pcov[0][0]), popt[1]+np.sqrt(pcov[1][1]), popt[2]]
        merr = [popt[0]-np.sqrt(pcov[0][0]), popt[1]-np.sqrt(pcov[1][1]), popt[2]]
        plt.fill_between(_x, R(_x, *perr), R(_x, *merr), color='red', alpha=0.3)
        plt.plot(_x, R(_x, *popt), 'r-', label=r'$\mu=$%.3f'%(mu))
        plt.legend(fontsize=12)
        plt.xlabel('Azimuthal angle [rad]')
        plt.savefig('%s'%title)

    return mu, mu_err

class PolarizationStokes():
    """
    Stokes parameter method to fit polarization.

    Parameters
    ----------
    source_vector : astropy.coordinates.sky_coordinate.SkyCoord
        Source direction
    source_spectrum : astromodels.functions.functions_1D
        Spectrum of source
    response_file : str or pathlib.Path
        Path to detector response
    sc_orientation : cosipy.spacecraftfile.SpacecraftFile.SpacecraftFile
        Spacecraft orientation
    """
    

    def __init__(self, source_vector, source_spectrum, response_file, sc_orientation):

        ###################### This will need to be changed into IAUPolarizationConvention hardcoded!
        ######################
        print('This class loading takes around 30 seconds... \n')
        ######################

        self._convention = MEGAlibRelativeX(attitude=source_vector.attitude)
        reference_vector = self._convention.get_basis(source_vector)[0] #px

        if isinstance(source_vector.frame, SpacecraftFrame):
            self._source_vector = source_vector
        else:
            self._source_vector = source_vector.transform_to(SpacecraftFrame(attitude=source_vector.attitude))
        
        if isinstance(reference_vector.frame, SpacecraftFrame):
            self._reference_vector = reference_vector
        else:
            self._reference_vector = reference_vector.transform_to(SpacecraftFrame(attitude=source_vector.attitude))
        
        self._expectation, self._azimuthal_angle_bins = self.convolve_spectrum(source_spectrum, response_file, sc_orientation)

        self._energy_range = [min(self.response.axes['Em'].edges.value), max(self.response.axes['Em'].edges.value)]

        self._binedges = Angle(np.linspace(-np.pi, np.pi, 20), unit=u.rad)

        self._exposure = sc_orientation.get_time_delta().to_value(u.second).sum()
      
    def convolve_spectrum(self, spectrum, response_file, sc_orientation):
        """
        Convolve source spectrum with response and calculate azimuthal scattering angle bins.

        Parameters
        ----------
        response_file : str or pathlib.Path
            Path to detector response
        sc_orientation : cosipy.spacecraftfile.SpacecraftFile.SpacecraftFile
            Spacecraft orientation

        Returns
        -------
        expectation : cosipy.response.PointSourceResponse.PointSourceResponse
            Expected counts in each bin of Compton data space
        azimuthal_angle_bins : list
            Centers of azimuthal scattering angle bins calculated from PsiChi bins in response
        """

        self.response = FullDetectorResponse.open(response_file)

        target_in_sc_frame = sc_orientation.get_target_in_sc_frame(target_name='source', target_coord=self._source_vector.transform_to('galactic'))
        dwell_time_map = sc_orientation.get_dwell_map(response=response_file, src_path=target_in_sc_frame)

        psr = self.response.get_point_source_response(exposure_map=dwell_time_map, coord=self._source_vector.transform_to('galactic'))

        expectation = psr.get_expectation(spectrum)

        azimuthal_angle_bins = []

        for i in range(expectation.axes['PsiChi'].nbins):
            azimuthal_angle = calculate_azimuthal_scattering_angle(expectation.project(['PsiChi']).axes['PsiChi'].pix2ang(i)[0], 
                                                                   expectation.project(['PsiChi']).axes['PsiChi'].pix2ang(i)[1],
                                                                   self._source_vector, self._reference_vector)
            azimuthal_angle_bins.append(azimuthal_angle)

        return expectation, azimuthal_angle_bins

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
            if unbinned_data['Energies'][i] >= self._energy_range[0] and unbinned_data['Energies'][i] <= self._energy_range[1]:
                azimuthal_angle = calculate_azimuthal_scattering_angle(unbinned_data['Psi local'][i], 
                                                                       unbinned_data['Chi local'][i],
                                                                       self._source_vector, self._reference_vector)
                azimuthal_angles.append(azimuthal_angle)

        return azimuthal_angles

    def create_unpolarized_asad(self, bins=None):
        """
        Calculate the azimuthal scattering angles for all bins.
        
        Parameters
        ----------
        bins : int or np.array, optional
            Number of azimuthal scattering angle bins if int or edges of azimuthal scattering angle bins if np.array (radians)

        Returns
        -------
        asad : dict
            Counts and Gaussian/Poisson errors in each azimuthal scattering angle bin
        """
        print('Creating the unpolarized ASAD...')
        if not bins == None:
            if isinstance(bins, int):
                bin_edges = Angle(np.linspace(-np.pi, np.pi, bins), unit=u.rad)
                self._binedges = bin_edges
            else:
                bin_edges = bins
                self._binedges = bin_edges
        else:
            bin_edges = self._binedges

        unpolarized_asad = []
        for i in range(len(bin_edges)-1):
            counts = 0
            for j in range(self._expectation.project(['PsiChi']).nbins):
                if self._azimuthal_angle_bins[j] >= bin_edges[i] and self._azimuthal_angle_bins[j] < bin_edges[i+1]:
                    counts += self._expectation.project(['PsiChi'])[j]
            unpolarized_asad.append(counts)

        return bin_edges, np.array(unpolarized_asad)
    
    def create_polarized100_asad(self, bins=None):
        """
        Calculate the azimuthal scattering angles for a 100% polarized source.
        
        Parameters
        ----------
        bins : int or np.array, optional
            Number of azimuthal scattering angle bins if int or edges of azimuthal scattering angle bins if np.array (radians)

        Returns
        -------
        qs : list
            list of pseudo-q parameters for each photon (ordered as input array)
        us : list
            list of pseudo-u parameters for each photon (ordered as input array)
        """
        print('Creating the 100% polarized ASAD...')
        if not bins == None:
            if isinstance(bins, int):
                bin_edges = Angle(np.linspace(-np.pi, np.pi, bins), unit=u.rad)
                self._binedges = bin_edges
            else:
                bin_edges = bins
                self._binedges = bin_edges
        else:
            bin_edges = self._binedges
        
        _polarized100_asad_ = []
        for k in range(self._expectation.axes['Pol'].nbins):
            polarized100_asad_ = []
            for i in range(len(bin_edges)-1):
                counts = 0
                for j in range(self._expectation.project(['PsiChi']).nbins):
                    if self._azimuthal_angle_bins[j] >= bin_edges[i] and self._azimuthal_angle_bins[j] < bin_edges[i+1]:
                        counts += self._expectation.slice[{'Pol':slice(k,k+1)}].project(['PsiChi'])[j]
                polarized100_asad_.append(counts)
            _polarized100_asad_.append(polarized100_asad_)

        return bin_edges, np.array(_polarized100_asad_)
    
    def calculate_photon_mu():
        """ Funciont to comput the mu for each photon
            Should return an array of mu values
        """
        #############################
        #############################
        pass

    def calculate_average_mu(self, bins=20, show=False):
        """
        Calculate the PA-averaged modulation (mu) of an 100% polarized source. 
        This sohuld not depend on the specific events but only on our instrument responses at differend PA bins.
        In this sence we can pre-compute a cube of modulation factors to pull from.

        MN note: Mu is energy-dependent. In this sense it depends on teh source spectrum and the mu(E) 
        response should be folded with that. For the Stokes parameters we would like to have a mu for each photon 
        so a mu(E, PA)  
        
        Parameters
        ----------
        polarized_asads : list
            Counts and Gaussian/Poisson errors in each azimuthal scattering angle bin for 
            each polarization angle bin for 100% polarized source
        unpolarized_asad : list or np.array
            Counts and Gaussian/Poisson errors in each azimuthal scattering angle bin for 
            unpolarized source
            
        Returns
        -------
        mu : dict
            Modulation of 100% polarized source and uncertainty of constant function fit to modulation in all polarization angle bins
        """
        print('This task takes a couple of minutes to run... hold on...\n')

        be, polarized100_asad = self.create_polarized100_asad(bins=bins)
        be, unpolarized_asad = self.create_unpolarized_asad(bins=bins)

        mu_, mu_err_ = [], []
        for i, pol100asad_pa in enumerate(polarized100_asad):

            asad_corrected = pol100asad_pa / np.sum(pol100asad_pa) / unpolarized_asad * np.sum(unpolarized_asad)
            # print('be, asad_corrected:', be, asad_corrected)

            mu, mu_err = get_modulation(be.value, asad_corrected, title='Modulation PA bin %i'%i, show=True)
            mu_.append(mu)
            mu_err_.append(mu_err)

            # plt.figure()
            # plt.step(be[:-1], pol100asad_pa / np.sum(pol100asad_pa), where='post')
            # plt.step(be[:-1], unpolarized_asad / np.sum(unpolarized_asad), where='post')
            # plt.figure()
            # plt.step(be[:-1], asad_corrected, where='post', linewidth=3)
            # plt.show()

        mu_ = np.array(mu_)
        mu_err_ = np.array(mu_err_)

        popt, pcov = curve_fit(constant, self._expectation.axes['Pol'].centers, mu_)

        average_mu = popt[0]
        average_mu_err = np.sqrt(pcov[0][0])

        print('mu:', average_mu, '+/-', average_mu_err)

        if show:
            plt.figure()
            plt.errorbar(np.arange(len(mu_)), mu_, yerr=mu_err_)
            plt.hlines(average_mu, 0, len(mu_), color='red', linewidth=4,
                        label=r'$\mu$ = %.3f +/- %.3f'%(average_mu, average_mu_err))
            plt.hlines(average_mu+average_mu_err, 0, len(mu_), color='red', linestyle='--', linewidth=2)
            plt.hlines(average_mu-average_mu_err, 0, len(mu_), color='red', linestyle='--', linewidth=2)
            plt.xlabel('PA bin')
            plt.ylabel(r'$\mu$')
            plt.legend()
            plt.show()

        return average_mu, average_mu_err
    
    def compute_pseudo_stokes(self, azimuthal_angles, show=False):
        """
        Calculates photon-by-photon pseudo stokes parameters from the photon azimutal angle.
        
        Parameters
        ----------
        azimuthal_angles : list
            Azimuthal scattering angles (radians)
    
        Returns
        -------
        qs : list
            list of pseudo-q parameters for each photon (ordered as input array)
        us : list
            list of pseudo-u parameters for each photon (ordered as input array)
        """

        qs, us = [], []

        #this is stupid... need to fix!
        try:
            for a in azimuthal_angles.value:
                qs.append(np.cos(a * 2) * 2)
                us.append(np.sin(a * 2) * 2)
        except:

            for a in azimuthal_angles:
                qs.append(np.cos(a.value * 2) * 2)
                us.append(np.sin(a.value * 2) * 2)

        if show:
            plt.figure()
            plt.title('Source Stokes parameters')
            plt.hist(qs, bins=50, alpha=0.5, label='q$_s$')
            plt.hist(us, bins=50, alpha=0.5, label='u$_s$')
            plt.xlabel('Pseudo Stokes parameter')
            plt.legend()
            plt.show()
        
        return qs, us

    def create_unpolarized_pseudo_stokes(self, total_num_events, bins=20, show=False):
        """
        Calculate the azimuthal scattering angles for all bins. 
        
        Parameters
        ----------
        total_num_events: int
            total number of events that matches your polarized data
        bins : int or np.array, optional
            Number of azimuthal scattering angle bins if int or edges of azimuthal scattering angle bins if np.array (radians)

        Returns
        -------
        qs : list
            list of pseudo-q parameters for each photon (ordered as input array)
        us : list
            list of pseudo-u parameters for each photon (ordered as input array)
        """
        print('this task takes around 25 seconds...\n')

        be, unpolarized_asad = self.create_unpolarized_asad(bins=bins)
        be = be.value
        # I would like to radomly extract an azimutal angle for each photon based on the unpolarized response.
        # There might be an energy dependence here, so we should thing carfully

        # Create teh spline from the unpol azimutal angle distrib
        spline_unpol = interpolate.interp1d(be[:-1], unpolarized_asad)
        # Create fine bins and normalize to the area to get a probability density function (PDF)
        # also, avoiding edges that wouls break the spline
        fine_bins = np.linspace(be[0]-0.01*be[0], be[-2]-0.01*be[-2], 1000)
        fine_probabilities = spline_unpol(fine_bins)
        total_area = np.trapz(fine_probabilities, fine_bins)  # Numerical integration using trapezoidal rule
        fine_probabilities /= total_area
        
        # Compute the cumulative distribution function (CDF)
        cdf = np.cumsum(fine_probabilities)
        cdf = cdf / cdf[-1]  # Normalize the CDF to make it a proper probability distribution
        
        #Invert the CDF
        inv_cdf = interpolate.interp1d(cdf, fine_bins)
        
        #Generate random samples from a uniform distribution and map them to azimuthal angles
        random_values = np.random.uniform(low=np.min(cdf), high=np.max(cdf), size=total_num_events)
        print('random_values', random_values)
        unpol_azimuthal_angles = inv_cdf(random_values) * u.rad
        print('unpol_azimuthal_angles', unpol_azimuthal_angles)
        qs_unpol, us_unpol = self.compute_pseudo_stokes(unpol_azimuthal_angles)

        if show:
            plt.figure()
            plt.title('Unpolarized')
            plt.hist(qs_unpol, bins=50, alpha=0.5, label='q$_s$')
            plt.hist(us_unpol, bins=50, alpha=0.5, label='u$_s$')
            plt.xlabel('Pseudo Stokes parameter')
            plt.legend()
            plt.show()
        
        return qs_unpol, us_unpol
    
    def calculate_mdp(self, total_num_events, mu, bkg_rate=22.0):
        """ 
        Calculate the minimum detectable polarization of a given observation.
        Assumes a default background count rate (~22 ph/s), but also allows for a custom value.

        Uses the exposure computed from teh sc_orientation object: 
            sc_orientation.get_time_delta().to_value(u.second).sum()

        Parameters
        ----------
        total_num_events: int
            total number of events that matches your polarized data
        mu : float 
            PA-Averaged modulation factor
        bkg_rate : float, optional
            Background count rate (default is 22.0 ph/s)

        Returns
        -------
        MDP99 : float
            Minimum detectable polarization at 99% confidence level
        """

        print('Calculating the MDP...')
        print('Espoure:', self._exposure, 's')
        print('Total number of events:', total_num_events)
        print('Modulation factor:', mu)
        print('Background rate:', bkg_rate, 'ph/s')
        Ns = total_num_events - bkg_rate * self._exposure
        MDP99 = 4.29 / (mu * Ns) * np.sqrt(total_num_events)
        print('MDP_99%:', MDP99*100, '%')
        return MDP99
        

    def calculate_polarization(self, qs, us, qs_unpol, us_unpol, mu, show=False, ref_qu=(None, None), ref_pdpa=(None, None), ref_label=None, mdp=None):
        """
        Calculate the polarization degree (PD), polarization angle (PA),
        and their associated 1-sigma uncertainties given Q and U measurements 
        from both polarized and unpolarized data sets.

        This implements equations (21), (22), (36), and (37) from Kislat et al. (2015).

        Parameters
        ----------
        qs : array-like
            Array of Q measurements (from polarized source).
        us : array-like
            Array of U measurements (from polarized source).
        qs_unpol : array-like
            Array of Q measurements (from unpolarized source).
        us_unpol : array-like
            Array of U measurements (from unpolarized source).
        mu : float
            Modulation factor. Used to convert raw measurements into normalized Q/I and U/I.
        show : bool, optional
            If True, display a diagnostic plot in the Q-U plane with 
            uncertainty circles, by default False.
        ref_qu : tuple of (float or None, float or None), optional
            Reference (Q, U) point (e.g., from simulation) to be plotted for comparison,
            by default (None, None) (no reference shown).
        ref_pdpa : tuple of (float or None, float or None), optional
            Reference (PD, PA) point (e.g., from simulation) to be converted to Q/U 
            and plotted for comparison, by default (None, None) (no reference shown).

        Returns
        -------
        pol_PD : float
            Polarization degree, PD = sqrt(Q^2 + U^2).
        pol_1sigmaPD : float
            1-sigma statistical uncertainty on the polarization degree.
        pol_PA : astropy.coordinates.Angle
            Polarization angle (in radians internally), 
            computed as 90 - 0.5 * arctan2(U, Q) (converted into an Angle object).
        pol_1sigmaPA : float
            1-sigma statistical uncertainty on the polarization angle (in degrees).
        """

        pol_I = len(qs)
        pol_Q = np.sum(qs) / mu
        pol_U = np.sum(us) / mu
        unpol_I = len(qs_unpol)
        unpol_Q = np.sum(qs_unpol) / mu
        unpol_U = np.sum(us_unpol) / mu

        Q = pol_Q/pol_I - unpol_Q/unpol_I
        U = pol_U/pol_I - unpol_U/unpol_I

        polarization_fraction = np.sqrt(Q**2. + U**2.)
        pol_PD = polarization_fraction * 100
        pol_PA = 90 - 0.5 * np.degrees(np.arctan2(U, Q))

        ######################
        ###################### Need to understand why I need this rotation
        ######################
        Q, U = rotate_points_to_x_axis( Q, U, pol_PA)
        print('-------  Q/I, U/I', Q, U)

        pol_modulation = mu * polarization_fraction

        polarization_fraction_uncertainty = pol_sQ = np.sqrt((2. - pol_modulation**2.) / ((pol_I - 1.) * mu**2.))
        pol_1sigmaPD = polarization_fraction_uncertainty * 100
        pol_1sigmaPA = np.degrees(1 / (pol_modulation * np.sqrt(2. * (pol_I - 1.))))

        # print('PD: %.2f'%(pol_PD*100), '+/- %.2f'%(pol_1sigmaPD*100), '%') 
        # print('PA: %.2f'%pol_PA, '+/- %.2f'%pol_1sigmaPA, 'deg')

        if show:
            fig, ax = plt.subplots(figsize=(6.4, 6.4))
            polar_chart_backbone(ax)
            if ref_qu[0] != None:
                plt.plot(ref_qu[0], ref_qu[1], 'x', markersize=20, color='tab:green')
                plt.annotate(ref_label, (ref_qu[0], ref_qu[1]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=12)
            if ref_pdpa[0] != None:
                ref_q = ref_pdpa[0] * np.cos(2*ref_pdpa[1])
                ref_u = ref_pdpa[0] * np.sin(2*ref_pdpa[1])
                plt.plot(ref_q, ref_u, 'x', markersize=20, color='tab:green')
                plt.annotate(ref_label, (ref_q, ref_u), textcoords="offset points", xytext=(0,10), ha='center', color='tab:green', fontsize=12)
            if mdp != None:
                c_mdp = plt.Circle((0, 0), radius=mdp, facecolor='tab:red', alpha=0.3, linewidth=1, linestyle='--', label='MDP')
                plt.gca().add_artist(c_mdp)

            plt.plot(Q, U, 'o', markersize=5, color='red',label='Measured')
            pol_c = plt.Circle((Q, U), radius=polarization_fraction_uncertainty, facecolor='none', edgecolor='red', linewidth=1, label='Polarized source')
            pol_c2 = plt.Circle((Q, U), radius=2*polarization_fraction_uncertainty, facecolor='none', edgecolor='red', linewidth=1)
            pol_c3 = plt.Circle((Q, U), radius=3*polarization_fraction_uncertainty, facecolor='none', edgecolor='red', linewidth=1)
            plt.gca().add_artist(pol_c)
            plt.gca().add_artist(pol_c2)
            plt.gca().add_artist(pol_c3)
            plt.xlim(-1, 1)
            plt.ylim(-1, 1)
            plt.xlabel('Q/I')
            plt.ylabel('U/I')
            plt.tight_layout()

            plt.xlim(-1, 1)
            plt.ylim(-1, 1)
            plt.xlabel('Q/I')
            plt.ylabel('U/I')
            plt.tight_layout()

            plt.show()

        polarization_angle = Angle(pol_PA, unit=u.deg)
        polarization_angle = PolarizationAngle(polarization_angle, self._source_vector, convention=IAUPolarizationConvention())

        polarization_angle_uncertainty = Angle(pol_1sigmaPA, unit=u.deg)


        print('PD: %.2f'%(pol_PD), '+/- %.2f'%(pol_1sigmaPD), '%') 
        print('PA:', round(polarization_angle.angle.degree, 3), '+/-', round(polarization_angle_uncertainty.degree, 3))
        polarization = {'fraction': pol_PD, 'angle': polarization_angle, 'fraction uncertainty': polarization_fraction_uncertainty, 'angle uncertainty': polarization_angle_uncertainty, 'Q/I': Q, 'U/I': U, 'Stokes uncertainty': pol_sQ}

        return polarization


if __name__ == "__main__":

    print('Just some tests here...')

    pass