import numpy as np
from astropy.coordinates import Angle
import astropy.units as u
from astropy.stats import poisson_conf_interval
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
    
    perr = [popt[0]+np.sqrt(pcov[0][0]), popt[1]+np.sqrt(pcov[1][1]), popt[2]]
    merr = [popt[0]-np.sqrt(pcov[0][0]), popt[1]-np.sqrt(pcov[1][1]), popt[2]]
    
    if show:
        plt.figure()
        plt.title(title)
        plt.bar(_x, _y, align='center', width=0.07, alpha=0.5)
        plt.fill_between(_x, R(_x, *perr), R(_x, *merr), color='red', alpha=0.3)
        plt.plot(_x, R(_x, *popt), 'r-', label='$\mu=$%.2f'%mu)
        plt.legend(fontsize=12)
        plt.ylim(Rmin-500, Rmax+500)
        plt.xlabel('Azimuthal angle [rad]')
    return mu, popt, pcov

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

        # This will need to be changed into IAUPolarizationConvention hardcoded!
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
        Calculate the modulation (mu) of an 100% polarized source. This sohuld not depend on the specific events but only on our instrument responses.
        In this sence we can pre-compute a cube of modulation factors to pull from.

        MN note: I don't think this should depend on a source spectrum: this can be
        
        Parameters
        ----------
        polarized_asads : list
            Counts and Gaussian/Poisson errors in each azimuthal scattering angle bin for each polarization angle bin for 100% polarized source
        unpolarized_asad : list or np.array
            Counts and Gaussian/Poisson errors in each azimuthal scattering angle bin for unpolarized source
            
        Returns
        -------
        mu : dict
            Modulation of 100% polarized source and uncertainty of constant function fit to modulation in all polarization angle bins
        """

        be, polarized100_asad = self.create_polarized100_asad(bins=bins)
        print(polarized100_asad.shape)
        be, unpolarized_asad = self.create_unpolarized_asad(bins=bins)
        print(unpolarized_asad.shape) 

        mu_list = []
        for pol100asad_pa in polarized100_asad:
            asad_corrected = pol100asad_pa / np.sum(pol100asad_pa) / unpolarized_asad * np.sum(unpolarized_asad)
            print('be, asad_corrected:', be, asad_corrected)

            mu, popt, pcov = get_modulation(be, asad_corrected, title='Modulation', show=show)
            mu_list.append(mu)

        mu_final, popt, pcov = curve_fit(constant, self._expectation.axes['Pol'].centers, mu_list)

        print('mu:', mu_final)

        return mu_final
    
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

        for a in azimuthal_angles:
            qs.append(np.cos(a.radian * 2) * 2)
            us.append(np.sin(a.radian * 2) * 2)

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
        bins : int or np.array, optional
            Number of azimuthal scattering angle bins if int or edges of azimuthal scattering angle bins if np.array (radians)

        Returns
        -------
        qs : list
            list of pseudo-q parameters for each photon (ordered as input array)
        us : list
            list of pseudo-u parameters for each photon (ordered as input array)
        """

        be, unpolarized_asad = self.create_unpolarized_asad(bins=bins)

        # I would like to radomly extract an azimutal angle for each photon based on the unpolarized response.
        # There might be an energy dependence here, so we should thing carfully

        # Create teh spline from teh unpol azimutal angle distrib
        spline_unpol = interpolate.interp1d(be[:-1], unpolarized_asad)
        
        # Create fine bins and normalize to the area to get a probability density function (PDF)
        fine_bins = np.linspace(be[0]-0.1*be[0], be[-1]+0.1*be[-1], 1000)
        fine_probabilities = spline_unpol(fine_bins)
        total_area = np.trapz(fine_probabilities, fine_bins)  # Numerical integration using trapezoidal rule
        fine_probabilities /= total_area
        
        # Compute the cumulative distribution function (CDF)
        cdf = np.cumsum(fine_probabilities)
        cdf = cdf / cdf[-1]  # Normalize the CDF to make it a proper probability distribution
        
        #Invert the CDF
        inv_cdf = interpolate.interp1d(cdf, fine_bins)
        
        #Generate random samples from a uniform distribution and map them to azimuthal angles
        random_values = np.random.rand(total_num_events)
        unpol_azimuthal_angles = inv_cdf(random_values)

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
    
    def calculate_polarization(I, qs, us, unpol_qs, unpol_us , mu, W2=None):
        #
        #
        #
        # contunue here below
        # make sure that the output PA is a polarization angle object. E.g.:
        #    polarization_angle += Angle(180, unit=u.deg)
        #    polarization_angle = PolarizationAngle(polarization_angle, self._source_vector, convention=self._convention).transform_to(IAUPolarizationConvention())
        #
        #
        #
        """Calculate the polarization degree and angle, with the associated
        uncertainties, for a given q and u.

        This implements equations (21), (36), (22) and (37) in the paper,
        respectively.

        Note that the Stokes parameters passed as the input arguments are assumed
        to be normalized to the modulation factor (for Q and U) on an
        event-by-event basis and summed over the proper energy range.

        Great part of the logic is meant to avoid runtime zero-division errors.
        """
        if xStokesAnalysis._check_polarization_input(I, Q, U):
            abort('Invalid input to xStokesAnalysis.calculate_polarization()')
        # If W2 is not, i.e, we are not passing the sum of weights, we assume
        # that the analysis is un-weighted, and the acceptance correction is
        # not applied, in which case W2 = I and the scale for the errors is 1.
        if W2 is None:
            W2 = I
        # Initialize the output arrays.
        err_scale = np.full(I.shape, 1.)
        pd = np.full(I.shape, 0.)
        pd_err = np.full(I.shape, 0.)
        pa = np.full(I.shape, 0.)
        pa_err = np.full(I.shape, 0.)
        # Define the basic mask---we are only overriding the values for the array
        # elements that pass the underlying selection.
        # Note we need I > 1., and not simply I > 0., to avoid any possible
        # zero-division runtime error in the calculations, including the error
        # propagation.
        mask = I > 1.
        # First pass at the polarization degree, which is needed to compute the
        # modulation, which is in turn one of the ingredients of the error
        # propagation (remember that Q and U are the reconstructed quantities,
        # i.e., already divided by the modulation factor).
        pd[mask] = np.sqrt(Q[mask]**2. + U[mask]**2.) / I[mask]
        # Convert the polarization to modulation---this is needed later for the
        # error propagation.
        m = pd * mu
        # We want the bins to satify the relation (m^2 < 2), since (2 - m^2)
        # is one of the factors of the errors on the polarization.
        mask = np.logical_and(mask, m**2. < 2.)
        # We also want to make sure that the modulation factor is nonzero--see
        # formula for the polarization error.
        # It's not entirely clear to me why that would happen, but I assume that
        # if you have a bin with a couple of very-low energy events it is maybe
        # possible?
        mask = np.logical_and(mask, mu > 0.)
        # Create a masked version of the necessary arrays.
        _I = I[mask]
        _Q = Q[mask]
        _U = U[mask]
        _W2 = W2[mask]
        _mu = mu[mask]
        _m = m[mask]
        # Second pass on the polarization with the final mask.
        pd[mask] = np.sqrt(_Q**2. + _U**2.) / _I
        # See equations (A.4a) and (A.4b), and compare with equations (17a) and
        # (17b) for the origin of the factor sqrt(W2 / I). Also note that a
        # square root is missing in (A.4a) and (A.4b).
        err_scale[mask] = np.sqrt(_W2 / _I)
        # Calculate the errors on the polarization degree
        pd_err[mask] = err_scale[mask] * np.sqrt((2. - _m**2.) / ((_I - 1.) * _mu**2.))
        assert np.isfinite(pd).all()
        assert np.isfinite(pd_err).all()
        # And, finally, the polarization angle and fellow uncertainty.
        pa[mask] = 0.5 * np.arctan2(_U, _Q)
        pa_err[mask] = err_scale[mask] / (_m * np.sqrt(2. * (_I - 1.)))
        assert np.isfinite(pa).all()
        assert np.isfinite(pa_err).all()
        # Convert to degrees, if needed.
        if degrees:
            pa = np.degrees(pa)
            pa_err = np.degrees(pa_err)
        return pd, pd_err, pa, pa_err
