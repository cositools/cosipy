import numpy as np
from astropy.coordinates import Angle
import astropy.units as u
# from astropy.stats import poisson_conf_interval
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
# from cosipy.polarization import PolarizationAngle
from cosipy.polarization.conventions import MEGAlibRelativeX#, IAUPolarizationConvention
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
    cos_vals = np.cos(angle_)
    sin_vals = np.sin(angle_)
    
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
    
    def calculate_mu(self, bins=20, show=False):
        """
        Calculate the modulation (mu) of an 100% polarized source. 
        This sohuld not depend on the specific events but only on our instrument responses.
        In this sence we can pre-compute a cube of modulation factors to pull from.

        MN note: I don't think this should depend on a source spectrum: this can be
        
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
            plt.xlabel('bin')
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
    
    def calculate_mdp(self):
        """
        """
        #############################
        #############################
        pass

    def calculate_polarization(self, qs, us, qs_unpol, us_unpol, mu, show=False):
        """Calculate the polarization degree and angle, with the associated
        uncertainties, for a given q and u.

        This implements equations (21), (36), (22) and (37) in the paper Kislat et al 2015,
        respectively.

        """

        pol_I = len(qs)
        pol_Q = np.sum(qs) / mu
        pol_U = np.sum(us) / mu
        unpol_Q = np.sum(qs_unpol) / mu
        unpol_U = np.sum(us_unpol) / mu

        Q = pol_Q/pol_I - unpol_Q/pol_I
        U = pol_U/pol_I - unpol_U/pol_I

        pol_PD = np.sqrt(Q**2. + U**2.)# / pol_I
        pol_PA = 90 - 0.5 * np.degrees(np.arctan2(U, Q))

        pol_modulation = mu * pol_PD
        ###################### Need to understand why I need this rotation
        ######################
        Q, U = rotate_points_to_x_axis( Q, U, pol_PA)
        print('-------  Q/I, U/I', Q, U)

        pol_1sigmaPD = pol_sQ = np.sqrt((2. - pol_modulation**2.) / ((pol_I - 1.) * mu**2.))
        pol_1sigmaPA = np.degrees(1 / (pol_modulation * np.sqrt(2. * (pol_I - 1.))))

        print('PD:', pol_PD, '+/-', pol_1sigmaPD) 
        print('PA', pol_PA, '+/-', pol_1sigmaPA)

        if show:
            fig, ax = plt.subplots(figsize=(6.4, 6.4))
            polar_chart_backbone(ax)
            plt.plot(Q, U, 'o', markersize=5, color='red',label='Measured')
            pol_c = plt.Circle((Q, U), radius=pol_1sigmaPD, facecolor='none', edgecolor='red', linewidth=1, label='Polarized source')
            pol_c2 = plt.Circle((Q, U), radius=2*pol_1sigmaPD, facecolor='none', edgecolor='red', linewidth=1)
            pol_c3 = plt.Circle((Q, U), radius=3*pol_1sigmaPD, facecolor='none', edgecolor='red', linewidth=1)
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

        pol_PA = Angle(np.radians(pol_PA), unit=u.rad)
        return pol_PD, pol_1sigmaPD, pol_PA, pol_1sigmaPA


if __name__ == "__main__":

    print('loading files...')
    print('Simulated PD, PA: 70%, 83 degrees E of N')
    sim_pd, sim_pa = 0.7, np.radians(83)
    sim_u = sim_pd / np.sqrt((np.tan(2*sim_pa))**2 + 1)
    sim_q = sim_pd / np.sqrt((np.tan(2*sim_pa))**2 + 1) * np.tan(2*sim_pa)

    qs, us = np.load('qs.npy'), np.load('us.npy')
    qs_unpol, us_unpol = np.load('unpol_qs.npy'), np.load('unpol_us.npy')

    mu = 0.31

    pol_I = len(qs)
    pol_Q = np.sum(qs) / mu
    pol_U = np.sum(us) / mu
    unpol_Q = np.sum(qs_unpol) / mu
    unpol_U = np.sum(us_unpol) / mu

    Q = pol_Q/pol_I - unpol_Q/pol_I
    U = pol_U/pol_I - unpol_U/pol_I

    pol_PD = np.sqrt(Q**2. + U**2.)# / pol_I
    pol_PA = 90 - 0.5 * np.degrees(np.arctan2(U, Q))

    pol_modulation = mu * pol_PD
    ###################### Need to understand why I need this rotation
    ######################
    Q, U = rotate_points_to_x_axis( Q, U, pol_PA)
    print('-------  Q/I, U/I', Q, U)

    pol_1sigmaPD = pol_sQ = np.sqrt((2. - pol_modulation**2.) / ((pol_I - 1.) * mu**2.))
    pol_1sigmaPA = np.degrees(1 / (pol_modulation * np.sqrt(2. * (pol_I - 1.))))

    print('PD:', pol_PD, '+/-', pol_1sigmaPD) 
    print('PA', pol_PA, '+/-', pol_1sigmaPA)


    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    polar_chart_backbone(ax)
    plt.plot(sim_q, sim_u, 'x', markersize=20, color='tab:green',label='Simulated')
    plt.plot(Q, U, 'o', markersize=5, color='red',label='Measured')
    pol_c = plt.Circle((Q, U), radius=pol_1sigmaPD, facecolor='none', edgecolor='red', linewidth=1, label='Polarized source')
    pol_c2 = plt.Circle((Q, U), radius=2*pol_1sigmaPD, facecolor='none', edgecolor='red', linewidth=1)
    pol_c3 = plt.Circle((Q, U), radius=3*pol_1sigmaPD, facecolor='none', edgecolor='red', linewidth=1)
    plt.gca().add_artist(pol_c)
    plt.gca().add_artist(pol_c2)
    plt.gca().add_artist(pol_c3)

    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel('Q/I')
    plt.ylabel('U/I')
    plt.tight_layout()
    plt.show()