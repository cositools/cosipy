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

class PolarizationStokes():
    """
    Stokes parameter method to fit polarization.

    Parameters
    ----------
    source : astropy.coordinates.sky_coordinate.SkyCoord
        Source direction
    source_spectrum : astromodels.functions.functions_1D
        Spectrum of source
    data : dict or list of same
        Unbinned data, or list of unbinned data if
        separated in time
    background : dict or list of same
        Unbinned background model, or list of backgrounds if
        separated in time
    sc_orientation : cosipy.spacecraftfile.SpacecraftHistory.SpacecraftHistory
        Spacecraft orientation
    response_file : str or pathlib.Path
        Path to detector response
    response_convention : str, optional
        Polarization reference convention used in response
        ('RelativeX', 'RelativeY', or 'RelativeZ'). Default is
        'RelativeX'
    fit_convention : cosipy.polarization.PolarizationConvention, optional
        Polarization reference convention to use for fit. Default is
        IAU convention
    show_plots : bool, optional
        Option to show plots. Default is False

    """

    def __init__(self, source, source_spectrum, asad_bin_edges,
                 data,
                 sc_orientation, response_file, response_convention='RelativeX', background = None,
                 fit_convention=IAUPolarizationConvention(), show_plots=False):

        if isinstance(fit_convention.frame, SpacecraftFrame):
            if not isinstance(source.frame, SpacecraftFrame):
                attitude = sc_orientation.get_attitude()[0]
                source = source.transform_to(SpacecraftFrame(attitude=attitude))
                logger.warning("The source direction is being converted to the spacecraft "
                               "frame using the attitude at the first timestamp of the orientation.")
        else:
            source = source.transform_to('icrs')

        self._response = FullDetectorResponse.open(response_file, pa_convention=response_convention)

        if isinstance(fit_convention.frame, SpacecraftFrame) and \
           fit_convention.registered_name != self._response.pa_convention.registered_name:
            raise RuntimeError("If performing fit in spacecraft frame, "
                               "fit convention must match convention of response.")

        self._convention = fit_convention

        self._source = source

        self._spectral_flux = get_integrated_spectral_model(source_spectrum, self._response.axes['Ei'])

        energy_edges = self._response.axes['Em'].edges.value
        self._energy_range = ( min(energy_edges), max(energy_edges) )

        self._ori = sc_orientation

        if not isinstance(data, list):
            data = [data]

        emin, emax = self._energy_range

        self._data = []
        for unbinned_data in data:
            energies = unbinned_data['Energies']
            emask = ((energies >= emin) & (energies <= emax))

            data_ecut = {key: unbinned_data[key][emask] for key in unbinned_data}
            self._data.append(data_ecut)

        self._data_azimuthal_angles, self._data_duration = \
            self.calculate_scattering_angles(self._data, show_plots=show_plots)

        if background is not None:
            logger.info('Background provided. Make sure there is enough statistics.')

            if not isinstance(background, list):
                background = [background]

            self._background = []
            for unbinned_data in background:
                energies = unbinned_data['Energies']
                emask = ((energies >= emin) & (energies <= emax))

                data_ecut = {key: unbinned_data[key][emask] for key in unbinned_data}
                self._background.append(data_ecut)

                self._background_azimuthal_angles, self._background_duration = \
                    self.calculate_scattering_angles(self._background)
        else:
            logger.info('No background provided. Will not subtract background from data.')
            self._background = None
            self._background_azimuthal_angles = None
            self._background_duration = 0

        asad_unpolarized, asads_polarized = self.create_simulated_asads(asad_bin_edges)

        self._mu100 = self.calculate_mu100(asads_polarized, asad_unpolarized,
                                           asad_bin_edges, show_plots=False)

        self._mdp99 = self.calculate_mdp(self._mu100['mu'])

    def calculate_scattering_angles(self, datasets, show_plots=False):
        """
        Calculate the azimuthal scattering angles for all events in a dataset.

        Parameters
        ----------
        unbinned_data : list of dict
            Unbinned data including polar and azimuthal angles
            (radians) of scattered photon in local coordinates

        Returns
        -------
        azimuthal_angles : array of astropy.coordinates.Angle
            Azimuthal scattering angles
        duration : float
            Total duration of all datasets
        """

        emin, emax = self._energy_range

        all_scattering_angles = []
        duration = 0

        for unbinned_data in datasets:

            scattering_dirs = self.scattering_dirs_from_unbinned_data(unbinned_data)

            # convert scattering dirs to azimuthal angles
            scattering_angles = PolarizationAngle.from_scattering_direction(scattering_dirs,
                                                                            self._source,
                                                                            self._convention).angle
            all_scattering_angles.append(scattering_angles)

            duration += np.ptp(unbinned_data['TimeTags'])

        scattering_angles = np.concat(all_scattering_angles)

        if show_plots:
            plt.figure()
            plt.title('Azimuthal scattering angles')
            plt.hist(scattering_angles, bins=50, alpha=0.5, label='Data fine binning')
            plt.hist(scattering_angles, bins=self._response.axes['Pol'].nbins, alpha=0.5,
                     histtype='step', linewidth=2, label='Response binning')
            plt.xlabel('Azimuthal angle (radians)')
            plt.ylabel('Counts')
            plt.legend()
            plt.show()

        return scattering_angles, duration

    def scattering_dirs_from_unbinned_data(self, unbinned_data):
        """
        Extract the scattering directions from an unbinned data set.

        Parameters
        ----------
        unbinned_data : dict
            Unbinned data including polar and azimuthal angles
           (radians) of scattered photon in local coordinates

        Returns
        -------
        scattering_dirs : SkyCoord array
           Array of scattering directions

        """

        # NB: ASAD method applies energy cut here, but not to TimeTags
        # elsewhere.  is this correct?

        if isinstance(self._convention.frame, SpacecraftFrame):
            # source is in spacecraft-local frame
            scattering_dirs = SkyCoord(lon = unbinned_data['Chi local'],
                                       lat = np.pi/2 - unbinned_data['Psi local'],
                                       unit = u.rad, frame = self._convention.frame)
        else:
            #source is in inertial frame
            scattering_dirs = SkyCoord(l = unbinned_data['Chi galactic'],
                                       b = unbinned_data['Psi galactic'],
                                       frame = 'galactic', unit = u.deg).transform_to('icrs')

        return scattering_dirs

    def scattering_dirs_from_response(self, spectral_flux,
                                      polarization_levels,
                                      polarization_angles):
        """
        Convolve source spectrum with response and extract weighted
        scattering directions from the result.  Weightings are
        computed assuming a certain polarization fraction and angle;
        the function computes them for a whole list of these at once,
        since they are all computed from the same response slice.

        Parameters
        ----------
        spectral_flux : np.ndarray
             Integrated spectral flux in each Ei bin of self._response
        polarization_levels : array-like of float
            Polarization levels (between 0 and 1).
        polarization_angles : array-like of float
            Polarization angles in degrees. If in the spacecraft
            frame, the angle must have the same convention as the
            response.

        Returns
        -------
        scattering_dirs : SkyCoord array
           Array of scattering directions
        weights : list of arrays of float
           Weights for each scattering direction

        """

        if isinstance(self._convention.frame, SpacecraftFrame):
            # source is in spacecraft-local frame
            source = self._source.transform_to('galactic')
            dwell_time_map = self._ori.get_dwell_map(source, base=self._response)
            psr = self._response.get_point_source_response(coord=source,
                                                           exposure_map=dwell_time_map)
            psichi_axis = psr.axes['PsiChi']
            pix = np.arange(psichi_axis.nbins)
            lon, lat = psichi_axis.pix2ang(pix, lonlat=True)
            scattering_dirs = SkyCoord(lon, lat,
                                       unit=u.deg, frame=self._convention.frame)
        else:
            # source is in inertial frame
            source = self._source
            scatt_map = self._ori.get_scatt_map(nside=self._response.nside*2,
                                                target_coord=source)
            psr = self._response.get_point_source_response(coord=source,
                                                           scatt_map=scatt_map)
            psichi_axis = psr.axes['PsiChi']
            pix = np.arange(psichi_axis.nbins)
            scattering_dirs = psichi_axis.pix2skycoord(pix).transform_to('icrs')

        weights = []
        for pl, pa in zip(polarization_levels, polarization_angles):
            expectation = psr.get_expectation(spectrum = None, flux = spectral_flux,
                                              polarization = LinearPolarization(pl * 100., pa))
            weights.append(expectation.project('PsiChi').contents)

        return scattering_dirs, weights

    def create_simulated_asads(self, bin_edges):
        """
        Create unpolarized ASAD and and 100% polarized ASADs for each
        polarization angle bin of response.

        Parameters
        ----------
        bin_edges : astropy.units.Quantity
            edges of azimuthal scattering angle bins

        Returns
        -------
        asad_unpolarized : np.ndarray
           for unpolarized ASAD, total weight in each
           azimuthal scattering angle bin
        asads_polarized : list of np.ndarray
           For each polarization angle bin, total weight in each
           azimuthal scattering angle bin

        """

        # unpolarized first, then all polarized
        pol_axis = self._response.axes['Pol']
        pol_fractions = np.hstack(([0.], np.ones(pol_axis.nbins)))
        pol_angles    = np.hstack(([0.], pol_axis.centers.angle.to_value(u.deg)))

        scattering_dirs, weights = self.scattering_dirs_from_response(self._spectral_flux,
                                                                      pol_fractions,
                                                                      pol_angles)

        asads = [ self.scattering_dirs_to_asad(scattering_dirs, bin_edges, weight)
                  for weight in weights ]

        return asads[0], asads[1:]

    def scattering_dirs_to_asad(self, directions, bin_edges, weights):
        """
        Convert a set of (possibly weighted) scattering directions to
        an ASAD. For each direction, determine its azimuthal angle
        relative to the source vector, and bin these angles according
        to the specified bin edges.

        Parameters
        ----------
        directions : SkyCoord
           scattering directions
        bin_edges : np.array of float
           azimuthal angle bin edges for ASAD
        weights : np.array of float
           weight for each direction

        Returns
        -------
        asad : np.array
            Total weight in each azimuthal scattering angle bin

        """

        azimuthal_angles = PolarizationAngle.from_scattering_direction(directions,
                                                                       self._source,
                                                                       self._convention)

        asad, _ = np.histogram(azimuthal_angles.angle, bins=bin_edges, weights=weights)

        return asad

    def calculate_mu100(self, asads_polarized, asad_unpolarized,
                        asad_bin_edges, show_plots=False):
        """Calculate the modulation (mu) of an 100% polarized source.

        Parameters
        ----------
        asads_polarized : list of array-like
            Counts and Gaussian/Poisson errors in each azimuthal
            scattering angle bin for each polarization angle bin for
            100% polarized source
        asad_unpolarized : array-like
            Counts and Gaussian/Poisson errors in each azimuthal
            scattering angle bin for unpolarized source
        asad_bin_edges : array
            Bin edges for the ASADs
        show_plots : bool, optional
            Option to show plots. Default is False

        Returns
        -------
        mu_100 : dict
            Modulation of 100% polarized source and uncertainty of
            constant function fit to modulation in all polarization
            angle bins

        """

        def correct_asad(asad_data, asad_unpolarized):
            sum_ratio = np.sum(asad_unpolarized) / np.sum(asad_data)
            return asad_data / asad_unpolarized * sum_ratio

        def constant(x, a):
            # constant approximation a to
            # mu_100 values x.
            return a

        pol_axis = self._response.axes['Pol']
        pol_angles = pol_axis.centers.angle.to_value(u.deg)

        bin_centers = 0.5*(asad_bin_edges[:-1] + asad_bin_edges[1:])

        mu_100s = []
        mu_100_uncertainties = []
        for i in range(pol_axis.nbins):
            logger.info(f'Polarization angle bin: {pol_axis.edges.angle[i]} to {pol_axis.edges.angle[i+1]} deg')

            asad_corrected = correct_asad(asads_polarized[i], asad_unpolarized)
            mu_100, mu_err = self.calculate_mu(bin_centers.value, asad_corrected,
                                               title=f'Modulation PA bin {i}', show=show_plots)
            print(f'Modulation @ {pol_angles[i]:.1f} deg [Stok method]: {mu_100:.3f} +/- {mu_err:.3f}')

            res, _ = self.calculate_mu2(bin_centers.value, asad_corrected)
            mu_100 = res["mu"]
            mu_err = res["uncertainty"]
            print(f'Modulation @ {pol_angles[i]:.1f} deg [ASAD method]: {mu_100:.3f} +/- {mu_err:.3f}')

            mu_100s.append(mu_100)
            mu_100_uncertainties.append(mu_err)

        popt, pcov = curve_fit(constant,
                               pol_angles, mu_100s,
                               sigma = mu_100_uncertainties,
                               p0 = np.mean(mu_100s),
                               absolute_sigma = False) # True for Stokes
        result = {'mu': popt[0], 'uncertainty': pcov[0][0]}

        if show_plots:
            plt.figure()
            plt.scatter(centers, mu_100_list)
            plt.errorbar(centers, mu_100_list,
                         yerr=mu_100_uncertainties, linewidth=0, elinewidth=1)
            plt.plot([0, 175], [result['mu'], result['mu']])
            plt.xlabel('Polarization Angle (degrees)')
            plt.ylabel('mu_100')
            plt.show()

        logger.info(f'mu_100: {result["mu"]:.2f}')

        return result

    @staticmethod
    def calculate_mu2(asad_bin_centers, asad_values):
        """
        Calculate the modulation (mu).

        Parameters
        ----------
        asad : Histogram
           ASAD

        Returns
        -------
        modulation : dict
            Modulation and uncertainty of fitted sinusoid
        parameter_values : np.ndarray
            Fitted parameter values

        """

        def asad_sinusoid(x, a, b, c):
            # Sinusoid to fit scattering angles x
            # (radians) with shift and scaling parameters
            return a - b * np.cos(2 * (x - c))

        params, pcov = curve_fit(asad_sinusoid,
                                 asad_bin_centers,
                                 asad_values,
                                 bounds=((0, 0, 0),
                                         (np.inf,np.inf,np.pi)))

        uncertainties = np.sqrt(np.diagonal(pcov))

        mu = params[1] / params[0]
        mu_uncertainty = mu * np.sqrt((uncertainties[0]/params[0])**2 +
                                      (uncertainties[1]/params[1])**2)

        logger.info(f'Modulation: {mu:.3f} +/- {mu_uncertainty:.3f}')

        modulation = {'mu': mu, 'uncertainty': mu_uncertainty}

        return modulation, params

    @staticmethod
    def calculate_mu(asad_bin_centers, asad_values,
                     title='Modulation', show=False):
        """
        Function to estimate the modulation factor.

        Parameters
        ----------
        asad_bin_centers : array
            Central values of the ASAD histogram bins
        asad_values : array
            Values of the ASAD histogram bins
        title : str
            Title of the plot
        show : bool
            Whether to show the plot or not

        Returns
        -------
        mu : float
            Modulation factor
        mu_uncertainty : float
            Uncertainty in the modulation factor
        """

        def asad_sinusoid(x, a, b, c):
            # Sinusoid to fit scattering angles x
            # (radians) with shift and scaling parameters
            return a + b * np.cos(x + c)**2

        params, pcov = curve_fit(asad_sinusoid,
                                 asad_bin_centers,
                                 asad_values,
                                 bounds=((0, 0, 0),
                                         (np.inf,np.inf,np.pi)))

        uncertainties = np.sqrt(np.diagonal(pcov))

        Rvals = asad_sinusoid(asad_bin_centers, *params)
        Rmax, Rmin = np.amax(Rvals), np.amin(Rvals)
        mu = (Rmax-Rmin)/(Rmax+Rmin)

        mu_uncertainty = 2/(params[1] + 2*params[0])**2 * \
            np.sqrt((params[1] * uncertainties[0])**2 +
                    (params[0] * uncertainties[1])**2)

        logger.info(f'Modulation: {mu:.3f} +/- {mu_uncertainty:.3f}')

        if show:
            plt.figure()
            plt.title(title)
            plt.step(asad_bin_centers, asad_values, where='mid')
            perr = (params[0]+uncertainties[0], params[1]+uncertainties[1], params[2])
            merr = (params[0]-uncertainties[0], params[1]-uncertainties[1], params[2])
            plt.fill_between(_x, asad_sinusoid(_x, *perr), asad_sinusoid(_x, *merr), color='red', alpha=0.3)
            plt.plot(_x, R(_x, *params), 'r-', label=fr'$\mu=${mu:.3f}')
            plt.legend(fontsize=12)
            plt.xlabel('Azimuthal angle [rad]')
            plt.show()

        return mu, mu_uncertainty

    def calculate_mdp(self, mu_100):
        """
        Calculate the minimum detectable polarization (MDP) of the source.

        Returns
        -------
        mdp : float
            MDP of source
        """

        source_counts = self.get_counts(self._data)

        if self._background is not None:
            background_counts_scaled = self.get_counts(self._background) * \
                self._data_duration / self._background_duration

            mdp = 4.29 / mu_100 * np.sqrt(source_counts + background_counts_scaled) / source_counts
        else:
            mdp = 4.29 / mu_100 / np.sqrt(source_counts)

        logger.info(f'Minimum detectable polarization (MDP) of source: {mdp:.3f}')

        return mdp

    @staticmethod
    def get_counts(data):
        """
        Calculate the total counts in unbinned data.

        Returns
        -------
        data_counts : int
            Total counts in data
        """
        counts = 0
        for dataset in data:
            counts += len(dataset['TimeTags'])

        return counts

    ######################################################################
    # STOKES-SPECIFIC PARTS
    ######################################################################

    def _compute_pseudo_stokes(self, azimuthal_angles, show_plots=False, label=None):
        """
        Calculates photon-by-photon pseudo stokes parameters from the
        photon azimutal angle.

        Parameters
        ----------
        azimuthal_angles : Angle array
            Azimuthal scattering angles (radians)
        show_plots : bool, optional
            Plot Stokes parameters (default False)
        label : string
            Label for type of Stokes parameters being plotted

        Returns
        -------
        qs : array
            pseudo-q parameters for each photon (ordered as input array)
        us : array
            pseudo-u parameters for each photon (ordered as input array)

        """

        def stokes_q(phi):
            return np.cos(phi * 2) * 2

        def stokes_u(phi):
            return np.sin(phi * 2) * 2

        angles = azimuthal_angles.value

        ######################
        # ATTENTION: I need to add 90 degrees because the stokes convention assumes that EVPA //
        # source polarization, while for Compton scatttering it is perpendicular)
        qs = stokes_q(angles - np.pi/2)
        us = stokes_u(angles - np.pi/2)

        if show_plots:
            plt.figure()
            plt.title(f'{label} Stokes parameters (%i events)'%len(qs))
            plt.hist(qs, bins=50, alpha=0.5, label='q$_s$')
            plt.hist(us, bins=50, alpha=0.5, label='u$_s$')
            plt.xlabel('Pseudo Stokes parameter')
            plt.legend()
            plt.show()

        return qs, us

    def _get_backscal(self):
        """
        Calculate the background scaling factor to match the source duration.

        Returns
        -------
        backscal : float
            Background scaling factor
        """

        return self._data_duration / self._background_duration


    def fit(self, show_plots=False,
            ref_qu=(None, None),
            ref_pdpa=(None, None),
            ref_label=None):
        """Calculate the polarization degree (PD), polarization angle (PA),
        and their associated 1-sigma uncertainties given Q and U
        measurements from both polarized and unpolarized data sets.

        This implements equations (21), (22), (36), and (37) from Kislat et al. (2015).

        Parameters
        ----------
        show_plots : bool, optional
            If True, display a diagnostic plot in the Q-U plane with
            uncertainty circles, by default False.
        ref_qu : tuple of (float or None, float or None), optional
            Reference (Q, U) point (e.g., from simulation) to be
            plotted for comparison, by default (None, None) (no
            reference shown).
        ref_pdpa : tuple of (float or None, float or None), optional
            Reference (PD, PA) point (e.g., from simulation) to be
            converted to Q/U and plotted for comparison, by default
            (None, None) (no reference shown).
        ref_label : str, optional
            Label for the reference point in the plot, by default None
            (no label shown).

        Returns
        -------
        polarization: dict
            fraction : float
                Polarization degree, PD = sqrt(Q^2 + U^2).
            fraction_uncertainty : float
                1-sigma statistical uncertainty on the polarization
                degree.
            angle : astropy.coordinates.Angle
                Polarization angle (in radians internally), computed
                as 90 - 0.5 * arctan2(U, Q) (converted into an Angle
                object).
            angle_uncertainty : float
                1-sigma statistical uncertainty on the polarization
                angle (in degrees).

        """

        has_background = self._background_azimuthal_angles is not None

        src_qs, src_us = self._compute_pseudo_stokes(self._data_azimuthal_angles, show_plots=False)

        mu = self._mu100["mu"]

        pol_I = len(src_qs)
        pol_Q = np.sum(src_qs) / mu
        pol_U = np.sum(src_us) / mu
        logger.info(f'I, Q, U, mu: {pol_I} {pol_Q} {pol_U} {mu}')

        if not has_background:
            logger.info('No background data provided, assuming no background contribution.')

            I = pol_I

            QN = pol_Q/pol_I
            UN = pol_U/pol_I

            logger.info(f'Q, U (unsubtracted): {QN} {UN}')

        else:
            logger.info('Unpolarized bkg (or simulation) provided, subtracting its contribution.')

            bkg_qs, bkg_us = self._compute_pseudo_stokes(self._background_azimuthal_angles, show_plots=False)

            BACKSCAL = self._get_backscal()

            unpol_I = len(bkg_qs) * BACKSCAL
            unpol_Q = np.sum(bkg_qs) * BACKSCAL / mu
            unpol_U = np.sum(bkg_us) * BACKSCAL / mu
            logger.info(f'Q, U unpolarized: {unpol_Q/unpol_I} {unpol_U/unpol_I}')

            I = pol_I - unpol_I
            logger.info(f'check I(src+bkg) vs I(src): {pol_I} {I}')

            QN = pol_Q/pol_I + unpol_Q/unpol_I * BACKSCAL
            UN = pol_U/pol_I + unpol_U/unpol_I * BACKSCAL

            logger.info(f'Q, U, subtracted: {QN} {UN}')

            # FIXED: analogous to Eqn 28a/b of Kislat 2015
            unpol_modulation = mu * np.sqrt(unpol_Q**2 + unpol_U**2) / unpol_I
            unpol_sI = np.sqrt(unpol_I - 1)
            unpol_sQ = np.sqrt(2/mu**2 - unpol_modulation**2) / unpol_sI
            logger.info(f'Q, U unpolarized uncertainty: {unpol_sQ*100} %')

        # Reconstructed polarization fraction + uncertainty: See eqs
        # 21, 36 in Kislat 2015
        polarization_fraction = np.sqrt(QN**2 + UN**2)
        m = mu * polarization_fraction
        polarization_fraction_uncertainty = np.sqrt((2 - m**2)/((I - 1) * mu**2))
        pol_PD = polarization_fraction * 100
        pol_1sigmaPD = polarization_fraction_uncertainty * 100

        # Reconstructed polarization angle + uncertainty: See eqs 22,
        # 37 in Kislat 2015
        pol_PA = 0.5 * np.arctan2(UN, QN)
        # Convert to 0 to 180 deg (just the convention)
        if pol_PA < 0:
            pol_PA += np.pi

        pol_1sigmaPA = np.degrees(1 / (m * np.sqrt(2 * (I - 1))))

        polarization_angle = Angle(np.degrees(pol_PA), unit=u.deg)
        polarization_angle = PolarizationAngle(polarization_angle, self._source,
                                               convention=self._convention).transform_to(IAUPolarizationConvention())
        polarization_angle_uncertainty = Angle(pol_1sigmaPA, unit=u.deg)

        # FIXED: corrected to ~ match Eqns 28a/b of Kislat 2015
        pol_sI = np.sqrt(I - 1)
        pol_sQ = np.sqrt(2/mu**2 - QN**2) / pol_sI
        pol_sU = np.sqrt(2/mu**2 - UN**2) / pol_sI
        logger.info(f'Q/I, U/I, uncertainty: {pol_sQ} {pol_sU} {np.sqrt(pol_sQ)}')

        polarization = {'fraction': polarization_fraction,
                        'angle': polarization_angle,
                        'fraction_uncertainty': polarization_fraction_uncertainty,
                        'angle_uncertainty': polarization_angle_uncertainty,
                        'QN': QN,
                        'UN': UN,
                        'QN_ERR': pol_sQ,
                        'UN_ERR': pol_sU}

        if show_plots:

            fig, ax = plt.subplots(figsize=(6.7, 6.4))

            self.polar_chart_backbone(ax)

            if ref_qu[0] != None:
                plt.plot(ref_qu[0], ref_qu[1], 'x', markersize=20, color='tab:green')
                plt.annotate(ref_label, (ref_qu[0], ref_qu[1]), textcoords="offset points", xytext=(0,10),
                             ha='center', fontsize=12)
            if ref_pdpa[0] != None:
                ref_q, ref_u = self.rotate_points_to_x_axis(ref_pdpa[0], np.radians(ref_pdpa[1]))
                plt.plot(ref_q, ref_u, 'x', markersize=20, color='tab:green')
                plt.annotate(ref_label, (ref_q, ref_u), textcoords="offset points", xytext=(0,10), ha='center',
                             color='tab:green', fontsize=12)

            c_mdp = plt.Circle((0, 0), radius=self._mdp99, facecolor='tab:red', alpha=0.3, linewidth=1, linestyle='--',
                               label=r'MDP$_{99}$ = %.2f %%'%(self._mdp99*100))
            plt.gca().add_artist(c_mdp)

            if not has_background:
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
                unpol_c  = plt.Circle((unpol_Q/unpol_I, unpol_U/unpol_I), radius=unpol_sQ,
                                      facecolor='none', edgecolor='0.4', linewidth=1)
                unpol_c2 = plt.Circle((unpol_Q/unpol_I, unpol_U/unpol_I), radius=2*unpol_sQ,
                                      facecolor='none', edgecolor='0.4', linewidth=1)
                unpol_c3 = plt.Circle((unpol_Q/unpol_I, unpol_U/unpol_I), radius=3*unpol_sQ,
                                      facecolor='none', edgecolor='0.4', linewidth=1)
                plt.gca().add_artist(unpol_c)
                plt.gca().add_artist(unpol_c2)
                plt.gca().add_artist(unpol_c3)

            plt.plot(QN, UN, 'o', markersize=5, color='red', label=label_data)
            pol_c = plt.Circle((QN, UN), radius=polarization_fraction_uncertainty,
                               facecolor='none', edgecolor='red', linewidth=1)
            pol_c2 = plt.Circle((QN, UN), radius=2*polarization_fraction_uncertainty,
                                facecolor='none', edgecolor='red', linewidth=1)
            pol_c3 = plt.Circle((QN, UN), radius=3*polarization_fraction_uncertainty,
                                facecolor='none', edgecolor='red', linewidth=1)
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

        return polarization

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
