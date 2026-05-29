import numpy as np
from scipy.optimize import curve_fit

from astropy.coordinates import Angle, SkyCoord
import astropy.units as u

import matplotlib.pyplot as plt

from scoords import SpacecraftFrame
from threeML import LinearPolarization

from histpy import Histogram

from cosipy.polarization.polarization_angle import PolarizationAngle
from cosipy.polarization.conventions import (
    MEGAlibRelativeX,
    MEGAlibRelativeY,
    MEGAlibRelativeZ,
    IAUPolarizationConvention
)
from cosipy.response import FullDetectorResponse
from cosipy.response.functions import get_integrated_spectral_model

import logging
logger = logging.getLogger(__name__)

def R(x, A, B, C):
    """ Function to fit to the modulation of the azimuthal angle distribution.
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

def get_modulation(_x, _y, title='Modulation', show=False):
    """ Function to estimate the modulation factor.
        _x is the central value of the histogram bins
        _y is the value of the bins on the histograms

        Parameters
        ----------
        _x : array
            Central values of the histogram bins
        _y : array
            Values of the histogram bins
        title : str
            Title of the plot
        show : bool
            Whether to show the plot or not

        Returns
        -------
        mu : float
            Modulation factor
        mu_err : float
            Error on the modulation factor
    """

    popt, pcov = curve_fit(R, _x, _y )
    uncertainties = np.sqrt(np.diagonal(pcov))

    print('A = %.2f, B = %.2f, C = %.2f'%(popt[0], popt[1], popt[2]))

    Rmax, Rmin = np.amax(R(_x, *popt)), np.amin(R(_x, *popt))
    print('Rmax, Rmin:', Rmax, Rmin)
    mu = (Rmax-Rmin)/(Rmax+Rmin)
    print('Modulation mu = ', mu)

    mu_err = 2/(popt[1]+2*popt[0])**2 * np.sqrt(popt[1]**2 * uncertainties[0]**2 + popt[0]**2 * uncertainties[1]**2)

    if show:
        plt.figure()
        plt.title(title)
        plt.step(_x, _y, where='mid')
        perr = [popt[0]+uncertainties[0], popt[1]+uncertainties[1], popt[2]]
        merr = [popt[0]-uncertainties[0], popt[1]-uncertainties[1], popt[2]]
        plt.fill_between(_x, R(_x, *perr), R(_x, *merr), color='red', alpha=0.3)
        plt.plot(_x, R(_x, *popt), 'r-', label=r'$\mu=$%.3f'%(mu))
        plt.legend(fontsize=12)
        plt.xlabel('Azimuthal angle [rad]')
        plt.savefig(title)

    return mu, mu_err


def compute_scattering_angles(source_vector, ori, response, convention):

    if isinstance(convention.frame, SpacecraftFrame):
        source = source_vector.transform_to('galactic')
        dwell_time_map = ori.get_dwell_map(source, base=response)
        psr = response.get_point_source_response(exposure_map=dwell_time_map, coord=source)

        psichi_axis = psr.axes['PsiChi']
        colat, lon = psichi_axis.pix2ang(np.arange(psichi_axis.nbins))
        psichi = SkyCoord(lat = np.pi/2 - colat, lon = lon, unit = u.rad, frame = convention.frame)

    else:
        scatt_map = ori.get_scatt_map(nside=response.nside*2, target_coord=source_vector)
        psr = response.get_point_source_response(coord=source_vector, scatt_map=scatt_map)

        psichi_axis = psr.axes['PsiChi']
        psichi = psichi_axis.pix2skycoord(np.arange(psichi_axis.nbins)).transform_to('icrs')

    return psr, PolarizationAngle.from_scattering_direction(psichi, source_vector, convention).angle


def create_asads_from_response(spectral_flux, polarization_levels, polarization_angles,
                               source_vector, ori, response, convention, bin_edges):
    """
    Convolve source spectrum with response and calculate azimuthal scattering angle bins.

    Parameters
    ----------
    spectral_flux : Histogram
        Integrated spectral flux
    polarization_level : float
        Polarization level (between 0 and 1).
    polarization_angle : :py:class:`cosipy.polarization.polarization_angle.PolarizationAngle`
        Polarization angle. If in the spacecraft frame, the angle must have the same convention as the response.
    bins : int or astropy.units.quantity.Quantity, optional
        Number of azimuthal scattering angle bins if int or array of edges of azimuthal scattering angle bins if Quantity
    source_vector : astropy.coordinates.sky_coordinate.SkyCoord
        Source direction
    ori : cosipy.spacecraftfile.SpacecraftFile.SpacecraftFile
        Spacecraft orientation
    response : cosipy.response.FullDetectorResponse.FullDetectorResponse
        Response object
    convention : cosipy.polarization.PolarizationConvention
        Polarization convention

    Returns
    -------
    asad : histpy.Histogram
        Counts in each azimuthal scattering angle bin
    """

    psr, scattering_angles = compute_scattering_angles(source_vector, ori, response, convention)

    asads = []
    for pl, pa in zip(polarization_levels, polarization_angles):
        pol_angle = PolarizationAngle(pa, source_vector, convention=convention)
        expectation = psr.get_expectation(spectrum = None, flux = spectral_flux,
                                          polarization = LinearPolarization(pl * 100., pol_angle.angle.deg))

        asad, _ = np.histogram(scattering_angles, bins = bin_edges, weights = expectation.project(['PsiChi']).contents)
        asads.append(asad)

    return asads

class PolarizationStokes():
    """
    Stokes parameter method to fit polarization.

    Parameters
    ----------
    source_vector : astropy.coordinates.sky_coordinate.SkyCoord
        Source direction
    source_spectrum : astromodels.functions.functions_1D
        Spectrum of source

    data : list of dict
        Data to fit
    background : list of dict
        Background to fit
    response_convention : str
        Response convention
    response_file : str or pathlib.Path
        Path to detector response
    sc_orientation : cosipy.spacecraftfile.SpacecraftFile.SpacecraftFile
        Spacecraft orientation
    fit_convention : cosipy.polarization.PolarizationConvention
        Polarization convention for the fit
    show_plots : bool
        Whether to show plots or not
    """


    def __init__(self, source_vector, source_spectrum, data,
                 response_file, sc_orientation, background=None, response_convention='RelativeX',
                 fit_convention=IAUPolarizationConvention(), asad_bin_edges=None, show_plots=False):

        ###################### This will need to be changed into IAUPolarizationConvention hardcoded!

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

        self._ori = sc_orientation

        self._convention = fit_convention

        self._response = FullDetectorResponse.open(response_file, pa_convention=response_convention)

        self._source_vector = source_vector

        self._spectral_flux = get_integrated_spectral_model(source_spectrum, self._response.axes['Ei'])

        self._energy_range = [min(self._response.axes['Em'].edges.value), max(self._response.axes['Em'].edges.value)]

        print(f'Energy range considered (by responses design): {self._energy_range[0]} - {self._energy_range[1]} keV')

        # do a data cut before anything else! actually this should come as a separate routine: data selection and response
        # prep shold be done before analyzing the data
        if not isinstance(data, list):
            data = [data]

        self._data = []
        for dlist in data:
            iii = np.where((dlist['Energies'] >= self._energy_range[0]) & (dlist['Energies'] <= self._energy_range[1]))
            data_ecut = {key: dlist[key][iii] for key in dlist.keys()}
            self._data.append(data_ecut)

        self._data_azimuthal_angles = self.calculate_azimuthal_scattering_angles(self._data, show_plots=show_plots)
        self._data_duration = self.get_duration(self._data)

        if background is not None:
            print('Background provided. Make sure there is enough statistics.')

            if not isinstance(background, list):
                background = [background]

            self._background = []
            for bkg in background:
                iii = np.where((bkg['Energies'] >= self._energy_range[0]) & (bkg['Energies'] <= self._energy_range[1]))
                background_ecut = {key: bkg[key][iii] for key in bkg.keys()}
                self._background.append(background_ecut)

            self._background_azimuthal_angles = self.calculate_azimuthal_scattering_angles(self._background)
            self._background_duration = self.get_duration(self._background)
        else:
            print('No background provided. Will not subtract background from data.')
            self._background = None
            self._background_azimuthal_angles = None
            self._background_duration = 0

        self._mu100 = self.calculate_average_mu100(asad_bin_edges, show_plots=False)

        self._mdp99 = self.calculate_mdp(modulation_factor=self._mu100['mu'])

    @staticmethod
    def get_counts(data):
        """
        Calculate the total counts in data.

        Returns
        -------
        data_counts : int
            Total counts in data
        """
        counts = 0
        for dataset in data:
            if isinstance(dataset, dict):
                counts += len(dataset['TimeTags'])
            else:
                counts += dataset.binned_data.axes['Time'].nbins

        return counts

    @staticmethod
    def get_duration(data):
        """
        Calculate the total duration of data.

        Returns
        -------
        data_duration : float
            Total duration of data in seconds
        """
        duration = 0
        for dataset in data:
            if isinstance(dataset, dict):
                duration += np.ptp(dataset['TimeTags'])
            else:
                duration += np.ptp(dataset.binned_data.axes['Time'].edges).value

        return duration

    def calculate_azimuthal_scattering_angles(self, unbinned_data, show_plots=False):
        """
        Calculate the azimuthal scattering angles for all events in a dataset.

        Parameters
        ----------
        unbinned_data : list of dict
            Unbinned data including polar and azimuthal angles (radians) of scattered photon in local coordinates

        Returns
        -------
        azimuthal_angles : array of astropy.coordinates.Angle
            Azimuthal scattering angles
        """

        azimuthal_angles = []

        for dataset in unbinned_data:
            if isinstance(self._convention.frame, SpacecraftFrame):
                psichi = SkyCoord(lat = np.pi/2 - dataset['Psi local'],
                              lon = dataset['Chi local'],
                              unit = u.rad,
                              frame = self._convention.frame)
            else:
                psichi = SkyCoord(l = dataset['Chi galactic'],
                                  b = dataset['Psi galactic'],
                                  frame = 'galactic', unit = u.deg).transform_to('icrs')

            azimuthal_angle = PolarizationAngle.from_scattering_direction(psichi,
                                                                          self._source_vector,
                                                                          self._convention).angle
            azimuthal_angles.append(azimuthal_angle)

        azimuthal_angles = np.concat(azimuthal_angles)

        if show_plots:
            plt.figure()
            plt.title('Azimuthal scattering angles')
            plt.hist(azimuthal_angles, bins=50, alpha=0.5, label='Data fine binning')
            plt.hist(azimuthal_angles, bins=self._response.axes['Pol'].nbins, alpha=0.5,
                     histtype='step', linewidth=2, label='Response binning')
            plt.xlabel('Azimuthal angle (radians)')
            plt.ylabel('Counts')
            plt.legend()
            plt.show()

        return azimuthal_angles

    def calculate_average_mu100(self, asad_bin_edges, show_plots=False):
        """
        Calculate the modulation (mu) of an 100% polarized source.

        Parameters
        ----------
        asad_bin_edges : array-like, optional
            Bin edges for the ASAD. If None, default binning is used.
        show_plots : bool, optional
            Option to show plots. Default is False

        Returns
        -------
        mu_100 : dict
            Modulation of 100% polarized source and uncertainty of constant function fit to modulation in all polarization angle bins
        """

        if asad_bin_edges is None:
            nbins = self._response.axes["Pol"].nbins
            asad_bin_edges = Angle(np.linspace(-np.pi, np.pi, nbins), unit=u.rad)
        elif isinstance(asad_bin_edges, int):
            asad_bin_edges = Angle(np.linspace(-np.pi, np.pi, asad_bin_edges), unit=u.rad)

        pol_axis = self._response.axes['Pol']
        pas = np.concat(([Angle(0, unit=u.deg)], pol_axis.centers.angle.to(u.deg)))
        pls = np.concat(([0],                    np.repeat(1, len(pas))))

        asads = create_asads_from_response(self._spectral_flux, pls, pas,
                                           self._source_vector, self._ori,
                                           self._response, self._convention,
                                           bin_edges=asad_bin_edges)
        unpolarized_asad = asads[0]
        polarized_asads  = asads[1:]

        unpolarized_asad /= np.sum(unpolarized_asad)

        mu_100_list = []
        mu_100_uncertainties = []

        pol_axis = self._response.axes['Pol']
        edge_angles = pol_axis.edges.angle.to_value(u.deg)
        asad_bin_centers = 0.5*(asad_bin_edges[:-1] + asad_bin_edges[1:])

        for i in range(pol_axis.nbins):
            logger.info(f'Polarization angle bin: {edge_angles[i]} to {edge_angles[i+1]} deg')
            asad_corrected = polarized_asads[i] / np.sum(polarized_asads[i]) / unpolarized_asad
            mu, mu_err = get_modulation(asad_bin_centers.value, asad_corrected,
                                        title=f'Modulation PA bin {i}', show=show_plots)
            mu_100_list.append(mu)
            mu_100_uncertainties.append(mu_err)

        centers = pol_axis.centers.angle.to_value(u.deg)
        popt, pcov = curve_fit(constant, centers, mu_100_list,
                               sigma=mu_100_uncertainties, p0=np.mean(mu_100_list), absolute_sigma=True)
        mu_100 = {'mu': popt[0], 'uncertainty': pcov[0][0]}

        if show_plots == True:
            plt.figure()
            plt.scatter(centers, mu_100_list)
            plt.errorbar(centeres, mu_100_list,
                         yerr=mu_100_uncertainties, linewidth=0, elinewidth=1)
            plt.plot([0, 175], [mu_100['mu'], mu_100['mu']])
            plt.xlabel('Polarization Angle (degrees)')
            plt.ylabel('mu_100')
            plt.show()

        return mu_100

    def calculate_mdp(self, modulation_factor):
        """
        Calculate the minimum detectable polarization (MDP) of the source.

        Returns
        -------
        mdp : float
            MDP of source
        """
        source_counts = self.get_counts(self._data)
        source_data_rate = source_counts / self._data_duration

        if self._background is not None:
            background_counts = self.get_counts(self._background)
            background_data_rate = background_counts / self._background_duration

            mdp = 4.29 /  modulation_factor * np.sqrt(source_data_rate / self._data_duration +
                                                      background_data_rate / self._background_duration) / source_data_rate
        else:
            mdp = 4.29 /  modulation_factor / np.sqrt(source_counts)

        logger.info(f'Minimum detectable polarization (MDP) of source: {mdp:.3f}')

        return mdp

    ######################################################################

    @staticmethod
    def stokes_u(phi):
        """
        Calculate the U Stokes parameter from the azimuthal angle phi.

        Parameters
        ----------
        phi : float
        Azimuthal angle in radians

        Returns
        -------
        u : float
        U Stokes parameter
        """

        return np.sin(phi * 2) * 2

    @staticmethod
    def stokes_q(phi):
        """
        Calculate the Q Stokes parameter from the azimuthal angle phi.

        Parameters
        ----------
        phi : float
            Azimuthal angle in radians

        Returns
        -------
        q : float
            Q Stokes parameter
        """
        return np.cos(phi * 2) * 2

    def compute_data_pseudo_stokes(self, show_plots=False):
        """
        Calculates photon-by-photon pseudo stokes parameters from the photon azimutal angle.

        Parameters
        ----------
        show : bool, optional
            If True, display a diagnostic plot in the Q-U plane with
            uncertainty circles, by default False.

        Returns
        -------
        qs : array
            pseudo-q parameters for each photon (ordered as input array)
        us : array
            pseudo-u parameters for each photon (ordered as input array)
        """

        try:
            a_ = self._data_azimuthal_angles.value
        except:
            a_ = np.concatenate(self._data_azimuthal_angles).value

        ######################
        # ATTENTION: I need to add 90 degrees because the stokes convention assumes that EVPA //
        # source polarization, while for Compton scatttering it is perpendicular)
        qs = self.stokes_q(a_ - np.pi/2)
        us = self.stokes_u(a_ - np.pi/2)

        if show_plots:
            plt.figure()
            plt.title('Source Stokes parameters (%i events)'%len(qs))
            plt.hist(qs, bins=50, alpha=0.5, label='q$_s$')
            plt.hist(us, bins=50, alpha=0.5, label='u$_s$')
            plt.xlabel('Pseudo Stokes parameter')
            plt.legend()
            plt.show()

        return qs, us

    def compute_background_pseudo_stokes(self, show_plots=False):
        """
        Calculates photon-by-photon pseudo stokes parameters from the photon azimutal angle.

        Parameters
        ----------
        azimuthal_angles : list
            Azimuthal scattering angles (radians)

        Returns
        -------
        qs : array
            pseudo-q parameters for each photon (ordered as input array)
        us : array
            pseudo-u parameters for each photon (ordered as input array)
        """

        if self._background_azimuthal_angles is None:
            logger.warning('No background data provided, returning empty lists for pseudo Stokes parameters.')
            return np.array([]), np.array([])
        else:
            try:
                a_ = self._background_azimuthal_angles.value
            except:
                a_ = np.concatenate(self._background_azimuthal_angles).value

            qs = self.stokes_q(a_ - np.pi/2)
            us = self.stokes_u(a_ - np.pi/2)

            if show_plots:
                plt.figure()
                plt.title('Background Stokes parameters (%i events)'%len(qs))
                plt.hist(qs, bins=50, alpha=0.5, label='q$_b$')
                plt.hist(us, bins=50, alpha=0.5, label='u$_b$')
                plt.xlabel('Pseudo Stokes parameter')
                plt.legend()
                plt.show()

        return qs, us

    def calculate_polarization(self, qs, us, mu,
                               bkg_qs=None, bkg_us=None,
                               show_plots=False,
                               ref_qu=(None, None),
                               ref_pdpa=(None, None),
                               ref_label=None, mdp=None):
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
        mu : float
            Modulation factor. Used to convert raw measurements into normalized Q/I and U/I.
        bkg_qs : array-like, optional
            Array of Q measurements from unpolarized background or simulation data, by default None.
        bkg_us : array-like, optional
            Array of U measurements from unpolarized background or simulation data, by default None.
        show_plots : bool, optional
            If True, display a diagnostic plot in the Q-U plane with
            uncertainty circles, by default False.
        ref_qu : tuple of (float or None, float or None), optional
            Reference (Q, U) point (e.g., from simulation) to be plotted for comparison,
            by default (None, None) (no reference shown).
        ref_pdpa : tuple of (float or None, float or None), optional
            Reference (PD, PA) point (e.g., from simulation) to be converted to Q/U
            and plotted for comparison, by default (None, None) (no reference shown).
        ref_label : str, optional
            Label for the reference point in the plot, by default None (no label shown).
        mdp : float, optional
            Minimum detectable polarization (MDP) value to be used for uncertainty calculations,
            by default None (no MDP used).

        Returns
        -------
        polarization: dict

            fraction : float
                Polarization degree, PD = sqrt(Q^2 + U^2).
            fraction_uncertainty : float
                1-sigma statistical uncertainty on the polarization degree.
            angle : astropy.coordinates.Angle
                Polarization angle (in radians internally),
                computed as 90 - 0.5 * arctan2(U, Q) (converted into an Angle object).
            angle_uncertainty : float
                1-sigma statistical uncertainty on the polarization angle (in degrees).
        """
        BACKSCAL = self.get_backscal()

        if BACKSCAL is None:
            logger.warning('Background scaling factor is None, assuming the unpolarized signal'+
                           'has been simulated with the same statistics as THE data')
            BACKSCAL = 1

        pol_I = I = len(qs)
        pol_Q = np.sum(qs) / mu
        pol_U = np.sum(us) / mu
        print('I, Q, U, mu', pol_I, pol_Q, pol_U, mu)

        self.QN = pol_Q/pol_I
        self.UN = pol_U/pol_I
        print('Q, U (unsubtracted:)', self.QN, self.UN)

        if bkg_qs is None or bkg_us is None:
            print('No background data provided, assuming no background contribution.')
        else:
            print('Unpolarized bkg (or simulation) provided, subtracting its contribution.')
            bkg_qs = np.asarray(bkg_qs)
            bkg_us = np.asarray(bkg_us)
            if bkg_qs.ndim == 1:
                unpol_I = len(bkg_qs) * BACKSCAL
                unpol_Q = np.sum(bkg_qs) * BACKSCAL / mu
                unpol_U = np.sum(bkg_us) * BACKSCAL / mu

                I = pol_I - unpol_I
                print('check I(src+bkg) vs I(src):', pol_I, I)
            else:
                BACKSCAL = 1
                unpol_I = bkg_qs.shape[2] * BACKSCAL
                unpol_Q = np.sum(bkg_qs) / len(bkg_qs) * BACKSCAL / mu
                unpol_U = np.sum(bkg_qs) / len(bkg_us) * BACKSCAL / mu

            print('Q, U unpolarized:', unpol_Q/unpol_I, unpol_U/unpol_I)
            unpol_modulation = mu * np.sqrt(unpol_Q**2. + unpol_U**2.) / unpol_I
            unpol_sI = np.sqrt(unpol_I)
            unpol_sQ = np.sqrt((2 - unpol_modulation**2) * unpol_sI**2 / unpol_I**2 / mu**2)
            unpol_sU = np.sqrt((2 - unpol_modulation**2) * unpol_sI**2 / unpol_I**2 / mu**2)
            print('Q, U unpolarized uncertainty:', unpol_sQ*100, '%')

            self.QN = pol_Q/pol_I + unpol_Q/unpol_I * BACKSCAL
            self.UN = pol_U/pol_I + unpol_U/unpol_I * BACKSCAL

            print('Q, U, subtracted:', self.QN, self.UN)

        pol_sI = np.sqrt(I)
        pol_sQ = np.sqrt((2 - self.QN**2) * pol_sI**2 / I**2 / mu**2)
        pol_sU = np.sqrt((2 - self.UN**2) * pol_sI**2 / I**2 / mu**2)
        pol_covQNUN = - (self.QN * self.UN) / I**2
        print('Q/I, U/I, uncertainty:', pol_sQ, pol_sU, np.sqrt(pol_sQ))

        # Reconstructed polarization fraction uncertainty: See eq 36 in Kislat 2015
        polarization_fraction = np.sqrt(self.QN**2. + self.UN**2.)
        m = mu * polarization_fraction
        polarization_fraction_uncertainty = np.sqrt((2 - m**2)/((I - 1) * mu**2))
        pol_PD = polarization_fraction * 100
        pol_1sigmaPD = polarization_fraction_uncertainty * 100

        # Reconstructed polarization angle uncertainty: See eq 37 in Kislat 2015
        pol_PA = 0.5 * np.arctan2(self.UN, self.QN)
        # Convert to 0 to 180 deg (just the convention)
        if pol_PA < 0:
            pol_PA += np.pi

        pol_1sigmaPA = np.degrees(1 / (m * np.sqrt(2. * (I - 1.))))
        print('\n ############################## \n')
        print('     PD: %.2f'%(pol_PD), '+/- %.2f'%(pol_1sigmaPD), '%')
        print('     PA: %.2f'%(np.degrees(pol_PA)), '+/- %.2f'%pol_1sigmaPA, 'deg')
        print('\n ############################## \n')

        if show_plots:

            fig, ax = plt.subplots(figsize=(6.7, 6.4))

            self.polar_chart_backbone(ax)

            if ref_qu[0] != None:
                # print('Drawing Reference point:', ref_qu)
                plt.plot(ref_qu[0], ref_qu[1], 'x', markersize=20, color='tab:green')
                plt.annotate(ref_label, (ref_qu[0], ref_qu[1]), textcoords="offset points", xytext=(0,10),
                             ha='center', fontsize=12)
            if ref_pdpa[0] != None:
                # print('Drawing Reference point:', ref_pdpa)
                ref_q, ref_u = self.rotate_points_to_x_axis(ref_pdpa[0], np.radians(ref_pdpa[1]))
                plt.plot(ref_q, ref_u, 'x', markersize=20, color='tab:green')
                plt.annotate(ref_label, (ref_q, ref_u), textcoords="offset points", xytext=(0,10), ha='center',
                             color='tab:green', fontsize=12)

            if mdp != None:
                c_mdp = plt.Circle((0, 0), radius=mdp, facecolor='tab:red', alpha=0.3, linewidth=1, linestyle='--',
                                   label=r'MDP$_{99}$ = %.2f %%'%(self._mdp99*100))
                plt.gca().add_artist(c_mdp)


            if bkg_qs is None or bkg_us is None:
                label_data = ("PD = (%.1f ± %.1f)%%\n"
                              "PA = (%.1f ± %.1f) deg"
                             % (pol_PD, pol_1sigmaPD, np.degrees(pol_PA), pol_1sigmaPA) )
                pass
            else:
                label_data = ("Measured (Unpol subtracted)\n"
                          "PD = (%.1f ± %.1f)%%\n"
                          "PA = (%.1f ± %.1f) deg"
                          % (pol_PD, pol_1sigmaPD, np.degrees(pol_PA), pol_1sigmaPA) )
                plt.plot(unpol_Q/unpol_I, unpol_U/unpol_I,  'o', markersize=5, color='0.4', \
                        label=r'Unpol (PD$_{1\sigma}$ = %i %%)'%(unpol_sQ*100))
                unpol_c = plt.Circle((unpol_Q/unpol_I, unpol_U/unpol_I), radius=unpol_sQ, facecolor='none', edgecolor='0.4', linewidth=1)
                unpol_c2 = plt.Circle((unpol_Q/unpol_I, unpol_U/unpol_I), radius=2*unpol_sQ, facecolor='none', edgecolor='0.4', linewidth=1)
                unpol_c3 = plt.Circle((unpol_Q/unpol_I, unpol_U/unpol_I), radius=3*unpol_sQ, facecolor='none', edgecolor='0.4', linewidth=1)
                plt.gca().add_artist(unpol_c)
                plt.gca().add_artist(unpol_c2)
                plt.gca().add_artist(unpol_c3)

            plt.plot(self.QN, self.UN, 'o', markersize=5, color='red', label=label_data)
            pol_c = plt.Circle((self.QN, self.UN), radius=polarization_fraction_uncertainty, facecolor='none', edgecolor='red', linewidth=1)
            pol_c2 = plt.Circle((self.QN, self.UN), radius=2*polarization_fraction_uncertainty, facecolor='none', edgecolor='red', linewidth=1)
            pol_c3 = plt.Circle((self.QN, self.UN), radius=3*polarization_fraction_uncertainty, facecolor='none', edgecolor='red', linewidth=1)
            plt.gca().add_artist(pol_c)
            plt.gca().add_artist(pol_c2)
            plt.gca().add_artist(pol_c3)

            plt.xlim(-1, 1)
            plt.ylim(-1, 1)
            plt.xlabel('Q/I')
            plt.ylabel('U/I')
            plt.tight_layout()
            plt.legend(fontsize=12)

            plt.show()

        polarization_angle = Angle(np.degrees(pol_PA), unit=u.deg)
        polarization_angle = PolarizationAngle(polarization_angle, self._source_vector, convention=self._convention).transform_to(IAUPolarizationConvention())
        polarization_angle_uncertainty = Angle(pol_1sigmaPA, unit=u.deg)

        polarization = {'fraction': polarization_fraction,
                        'angle': polarization_angle,
                        'fraction_uncertainty': polarization_fraction_uncertainty,
                        'angle_uncertainty': polarization_angle_uncertainty,
                        'QN': self.QN,
                        'UN': self.UN,
                        'QN_ERR': pol_sQ,
                        'UN_ERR': pol_sU}

        return polarization

    def get_backscal(self):
        """
        Calculate the background scaling factor to match the source duration.

        Returns
        -------
        backscal : float
            Background scaling factor
        """
        if self._background_duration == 0:
            logger.warning('Background duration is zero, returning backscal = 0')
            backscal = None
        else:
            backscal = self._data_duration / self._background_duration

        return backscal

    @staticmethod
    def rotate_points_to_x_axis(newPD, newPA):
        """
        Rotate arrays of points (x_, y_) in the QN-UN plane by an angle

        Parameters
        ----------
        newPD : float
        Polarization degree
        newPA : float
        Polarization angle
        Returns
        -------
        rotated_Q : float
        Q Stokes parameter
        rotated_U : float
        U Stokes parameter

        """
        # Create a matrix of rotation matrices for each point
        rotated_Q = newPD * np.cos(2 * newPA)
        rotated_U = newPD * np.sin(2 * newPA)

        return rotated_Q, rotated_U

    @staticmethod
    def polar_chart_backbone(ax):
        """ Preparing canvas for Stokes chart
        Parameters
        ----------
        ax : matplotlib.axes._axes.Axes
        Axes to plot on
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
