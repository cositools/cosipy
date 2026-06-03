import numpy as np

from scipy.optimize import curve_fit

import matplotlib.pyplot as plt

import astropy.units as u
from astropy.coordinates import Angle, SkyCoord

from scoords import SpacecraftFrame
from threeML import LinearPolarization
from histpy import Axis, Histogram

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
    """Stokes parameter method to fit polarization.

        Parameters
    ----------
    source : astropy.coordinates.sky_coordinate.SkyCoord
        Source direction
    source_spectrum : astromodels.functions.functions_1D
        Spectrum of source
    asad_bin_edges : astropy.coordinates.angles.core.Angle
        Bin edges for azimuthal scattering angle distribution
    data : dict or list of same
        Unbinned data, or list of unbinned data if separated in time
    background : dict or list of same, optional
        Unbinned background model, or list of backgrounds if separated
        in time
    sc_orientation : cosipy.spacecraftfile.SpacecraftHistory.SpacecraftHistory
        Spacecraft orientation
    response_file : str or pathlib.Path
        Path to detector response
    response_convention : str, optional
        Polarization reference convention used in response
        ('RelativeX', 'RelativeY', or 'RelativeZ'). Default is
        'RelativeX'
    fit_convention : cosipy.polarization.conventions.PolarizationConvention, optional
        Polarization reference convention to use for fit. Default is
        IAU convention
    show_plots : bool, optional
        Option to show plots. Default is False

    """

    def __init__(self, source, source_spectrum, asad_bin_edges,
                 data, sc_orientation, response_file,
                 response_convention='RelativeX', background=None,
                 fit_convention=IAUPolarizationConvention(),
                 show_plots=False):

        def apply_energy_cut(data, erange):
            """
            For unbinned data sets, keep only events between specified min and
            max energy.  Do not attempt to apply cut to binned data
            sets.

            Parameters
            ----------
            data : list of binned and/or unbinned data sets
            erange : pair (emin, emax) of limits for cut

            Returns
            -------
            list of datasets, with cut applied
            """

            emin, emax = erange

            data_ecut = []
            for dataset in data:
                if isinstance(dataset, dict):
                    # unbinned data set -- apply cut
                    energies = dataset['Energies']
                    emask = ((energies >= emin) & (energies <= emax))
                    dataset_ecut = {key: dataset[key][emask] for key in dataset}
                    data_ecut.append(dataset_ecut)
                else:
                    # binned data set -- do not apply cut
                    data_ecut.append(dataset)

            return data_ecut


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
        energy_range = ( np.min(energy_edges), np.max(energy_edges) )

        self._ori = sc_orientation

        if not isinstance(data, list):
            data = [data]
        data = apply_energy_cut(data, energy_range)

        asad_sb, self._source_duration, self._data_scattering_angles = \
            self.asad_and_angles_from_data(data, asad_bin_edges, show_plots=show_plots)

        axis = Axis(asad_bin_edges)
        asads = { 'source_and_background' : Histogram(axis, contents=asad_sb, copy_contents=False) }

        if background is not None:
            if not isinstance(background, list):
                background = [background]
            background = apply_energy_cut(background, energy_range)

            asad_background, self._background_duration, self._background_scattering_angles = \
                self.asad_and_angles_from_data(background, asad_bin_edges, show_plots=show_plots)

            asad_background_scaled = asad_background * source_duration / background_duration
            asad_source = asad_sb - asad_background_scaled

            asads['background'] = Histogram(axis, contents=asad_background, copy_contents=False)
            asads['background_scaled'] = Histogram(axis, contents=asad_background_scaled, copy_contents=False),
            asads['source'] = Histogram(axis, contents=asad_source, copy_contents=False),

        else:
            logger.info('No background provided. Will not subtract background from data.')
            self._background_scattering_angles = None
            self._background_duration = 0
            asads['source'] = asads['source_and_background']

        asad_unpolarized, asads_polarized = self.create_simulated_asads(asad_bin_edges)
        asads['unpolarized'] = asad_unpolarized
        asads['polarized']   = asads_polarized

        self._mu100 = self.calculate_mu100(asads['polarized'],
                                           asads['unpolarized'],
                                           show_plots)

        self._mdp99 = self.calculate_mdp99(asads['source'],
                                           None if background is None else asads['background_scaled'],
                                           self._mu100['mu'])

        if show_plots:

            self.plot_asad(asads['source_and_background'],
                           'Source+background ASAD',
                           asads['source_and_background'].bin_error.contents)

            if background is not None:

                uncertainty = np.sqrt(asads['source_and_background'].bin_error.contents**2 +
                                      asads['background_scaled'].bin_error.contents**2)

                self.plot_asad(asads['background'],
                               'Background ASAD',
                               asads['background'].bin_error.contents)

                self.plot_asad(asads['source'],
                               'Source ASAD',
                               uncertainty)

            self.plot_asad(asads['unpolarized'],
                           'Unpolarized ASAD')


    def asad_and_angles_from_data(self, datasets, bin_edges, show_plots=False):
        """
        Calculate the azimuthal scattering angles for all events in a dataset.

        Parameters
        ----------
        datasets : list of dict
            Unbinned data including polar and azimuthal angles
            (radians) of scattered photon in local coordinates
        bin_edges : array
            bin edges for computing ASAD

        Returns
        -------
        asad : array
            Azimuthal angle scattering distribution over all
            data sets binned according to provided bin edges
        duration : float
            Total duration of all datasets
        azimuthal_angles : array of astropy.coordinates.Angle
            Azimuthal scattering angles
        """

        all_scattering_dirs = []
        duration = 0

        for unbinned_data in datasets:

            all_scattering_dirs.append(self.scattering_dirs_from_unbinned_data(unbinned_data))

            duration += np.ptp(unbinned_data['TimeTags'])

        scattering_dirs = np.concat(all_scattering_dirs)
        asad, scattering_angles = self.scattering_dirs_to_asad(scattering_dirs, bin_edges,
                                                               return_angles=True)

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

        return asad, duration, scattering_angles

    def create_simulated_asads(self, bin_edges):
        """
        Create simulated unpolarized ASAD and and 100% polarized ASADs for
        each polarization angle bin of response.

        Parameters
        ----------
        bin_edges : astropy.units.Quantity
            edges of azimuthal scattering angle bins

        Returns
        -------
        asad_unpolarized : Histogram
           for unpolarized ASAD, total weight in each
           azimuthal scattering angle bin
        asads_polarized : list of Histogram
           For each polarization angle bin, total weight in each
           azimuthal scattering angle bin

        """

        # unpolarized first, then all polarized
        pol_axis = self._response.axes['Pol']
        pol_fractions = np.hstack(([0.], np.ones(pol_axis.nbins)))
        pol_angles =    np.hstack(([0.], pol_axis.centers.angle.to_value(u.deg)))

        scattering_dirs, weights = self.scattering_dirs_from_response(self._spectral_flux,
                                                                      pol_fractions,
                                                                      pol_angles)

        asads = [ self.scattering_dirs_to_asad(scattering_dirs, bin_edges, weight)
                  for weight in weights ]

        axis = Axis(bin_edges)
        asad_unpolarized = Histogram(axis, contents=asads[0], copy_contents=False)
        asads_polarized = [ Histogram(axis, contents=asad, copy_contents=False) for asad in asads[1:] ]

        return asad_unpolarized, asads_polarized

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

        if isinstance(self._convention.frame, SpacecraftFrame):
            # source is in spacecraft-local frame
            scattering_dirs = SkyCoord(lon=unbinned_data['Chi local'],
                                       lat=np.pi/2 - unbinned_data['Psi local'],
                                       unit=u.rad, frame=self._convention.frame)
        else:
            # source is in inertial frame
            scattering_dirs = SkyCoord(l=unbinned_data['Chi galactic'],
                                       b=unbinned_data['Psi galactic'],
                                       unit=u.deg, frame='galactic').transform_to('icrs')

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

    def scattering_dirs_to_asad(self, directions, bin_edges, weights=None, return_angles=False):
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
        weights : np.array of float, optional
           weight for each direction
        return_angles : bool, optional
           return raw angles along with ASAD (default: False)
        Returns
        -------
        asad : np.array
            Total weight in each azimuthal scattering angle bin

        """

        azimuthal_angles = PolarizationAngle.from_scattering_direction(directions,
                                                                       self._source,
                                                                       self._convention).angle

        asad, _ = np.histogram(azimuthal_angles, bins=bin_edges, weights=weights)

        if return_angles:
            return asad, azimuthal_angles
        else:
            return asad

    @staticmethod
    def correct_asad(asad_data, asad_unpolarized,
                     asad_data_uncertainties=None):
        """
        Correct the ASAD using the ASAD of an unpolarized source.

        Parameters
        ----------
        asad_data : Histogram
            Counts in each azimuthal scattering angle bin of data
        asad_unpolarized : Histogram
            Counts in each azimuthal scattering angle bin of unpolarized
            source
        asad_data_uncertainties : np.array, optional
            Uncertainties for each angle bin in asad_data

        Returns
        -------
        asad : histpy.Histogram
            Normalized counts in each azimuthal scattering angle bin
        uncertainties : np.array (if asad_data_uncertainties is not None)
            Uncertainties for each angle bin in result
        """

        sum_ratio = np.sum(asad_unpolarized) / np.sum(asad_data)

        corrected = asad_data / asad_unpolarized * sum_ratio

        asad_corrected = Histogram(asad_data.axis, contents=corrected, copy_contents=False)

        if asad_data_uncertainties is not None:
            uncertainties = asad_data_uncertainties / asad_unpolarized.contents * sum_ratio
        else:
            uncertainties = None

        return asad_corrected, uncertainties

    def calculate_mu100(self, asads_polarized, asad_unpolarized, show_plots=False):
        """
        Calculate the modulation (mu) of an 100% polarized source.

        Parameters
        ----------
        asads_polarized : list of array-like
            Counts and Gaussian/Poisson errors in each azimuthal
            scattering angle bin for each polarization angle bin for
            100% polarized source
        asad_unpolarized : array-like
            Counts and Gaussian/Poisson errors in each azimuthal
            scattering angle bin for unpolarized source
        show_plots : bool, optional
            Option to show plots. Default is False

        Returns
        -------
        mu100 : dict
            Modulation of 100% polarized source and uncertainty of
            constant function fit to modulation in all polarization angle
            bins

        """

        def constant(x, a):
            # constant approximation a to x
            return a

        pol_axis = self._response.axes['Pol']
        pol_angles = pol_axis.centers.angle.to_value(u.deg)

        mu100_vals = []
        for i in range(pol_axis.nbins):
            logger.info(f'Polarization angle bin: {pol_axis.edges.angle[i]} to {pol_axis.edges.angle[i+1]} deg')

            asad_polarized_corrected, _ = self.correct_asad(asads_polarized[i], asad_unpolarized)
            mu100, coefficients = self.calculate_mu(asad_polarized_corrected)

            mu100_vals.append(mu100)

            # calculate_mu enforces angle between 0 and pi
            fitted_angle = Angle(coefficients[2], unit=u.rad)
            logger.info(f'Fitted angle: {fitted_angle.deg} deg')

            if show_plots:
                self.plot_asad(asad_polarized_corrected,
                               f'Corrected 100% Polarized ASAD ({int(pol_angles[i])} deg)',
                               coefficients=coefficients)

        mu100s               = [ m['mu']          for m in mu100_vals ]
        mu100_uncertainties  = [ m['uncertainty'] for m in mu100_vals ]
        popt, pcov = curve_fit(constant,
                               pol_angles, mu100s,
                               sigma = mu100_uncertainties)
        result = {'mu': popt[0], 'uncertainty': pcov[0][0]}

        if show_plots:
            plt.scatter(pol_angles, mu100s)
            plt.errorbar(pol_angles, mu100s,
                         yerr=mu100_uncertainties, linewidth=0, elinewidth=1)
            plt.plot((0, 175), (result['mu'], result['mu']))
            plt.xlabel('Polarization Angle (degrees)')
            plt.ylabel('mu100')
            plt.show()

        logger.info(f'mu100: {result["mu"]:.2f}')

        return result

    @classmethod
    def calculate_mu(cls, asad):

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

        Note that third parameter (the fitted angle) is guaranteed
        to lie between 0 and pi.

        """

        params, uncertainties = cls.fit_asad(asad)

        mu = params[1] / params[0]
        mu_uncertainty = mu * np.sqrt((uncertainties[0]/params[0])**2 +
                                      (uncertainties[1]/params[1])**2)

        logger.info(f'Modulation: {mu:.3f} +/- {mu_uncertainty:.3f}')

        modulation = {'mu': mu, 'uncertainty': mu_uncertainty}

        return modulation, params

    @classmethod
    def fit_asad(cls, asad, p0=None, bounds=None, sigma=None):
        """
        Fit the ASAD with a sinusoid.

        Parameters
        ----------
        asad : Histogram
            ASAD
        p0 : np.array or None
            Initial guess for parameter values
        bounds : 2-tuple of float or array-like or None
            Lower & upper bounds on parameters; default is
            (0, 0, 0) and (inf, inf, pi)
        sigma : float or array-like or None
            Uncertainties in y data

        Returns
        -------
        popt : np.ndarray
            Fitted parameter values
        uncertainties : np.ndarray
            Uncertainty on each parameter value
        """

        if bounds is None:
            bounds = ((0, 0, 0), (np.inf, np.inf, np.pi))

        popt, pcov = curve_fit(cls.asad_sinusoid,
                               asad.axis.centers,
                               asad.contents,
                               p0=p0,
                               bounds=bounds,
                               sigma=sigma)

        uncertainties = np.sqrt(np.diagonal(pcov))

        return popt, uncertainties

    @staticmethod
    def asad_sinusoid(x, a, b, c):
        # Sinusoid to fit scattering angles x
        # (radians) with shift and scaling parameters
        return a - b * np.cos(2 * (x - c))

    @staticmethod
    def calculate_mdp99(asad_source, asad_background_scaled, mu100):
        """
        Calculate the minimum detectable polarization (MDP) of the source.

        Parameters
        ----------
        asad_source : Histogram
            ASAD for source
        asad_background_scaled : Histogram or None
            ASAD for background (scaled) if background known
        mu100 : float
            Modulation of 100% polarized source

        Returns
        -------
        mdp99 : float
            MDP of source
        """

        source_counts = np.sum(asad_source)

        if asad_background_scaled is not None:
            background_counts = np.sum(asad_background_scaled)
        else:
            background_counts = 0

        mdp99 = 4.29 / mu100 * np.sqrt(source_counts + background_counts) / source_counts

        logger.info(f'Minimum detectable polarization (MDP) of source: {mdp99:.3f}')

        return mdp99

    @classmethod
    def plot_asad(cls, asad, title, error=None, coefficients=None):
        """
        Plot an ASAD

        Parameters
        ----------
        asad : Histogram
            ASAD
        title : str
            Title of plot
        error : float or array-like, optional
            Uncertainties for each bin
        coefficients : array-like, optional
            Coefficients to plot fitted sinusoidal function
        """

        angles = np.rad2deg(asad.axis.centers)
        plt.scatter(angles, asad.contents)
        if error is not None:
            plt.errorbar(angles, asad.contents,
                         yerr=error,
                         linewidth=0,
                         elinewidth=1)
        plt.title(title)
        plt.xlabel('Azimuthal Scattering Angle (degrees)')

        if coefficients is not None:
            x = np.linspace(-np.pi, np.pi, 1000)
            y = cls.asad_sinusoid(x, *coefficients)
            plt.plot(np.rad2deg(x), y, color='green')

        plt.show()

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

        return self._source_duration / self._background_duration


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

        has_background = self._background_scattering_angles is not None

        src_qs, src_us = self._compute_pseudo_stokes(self._data_scattering_angles, show_plots=False)

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

            bkg_qs, bkg_us = self._compute_pseudo_stokes(self._background_scattering_angles, show_plots=False)

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
                plt.plot(ref_qu[0], ref_qu[1], 'x',
                         markersize=20, color='tab:green')
                plt.annotate(ref_label, (ref_qu[0], ref_qu[1]),
                             textcoords="offset points", xytext=(0,10),
                             ha='center', fontsize=12)
            if ref_pdpa[0] != None:
                ref_q, ref_u = \
                    self.rotate_points_to_x_axis(ref_pdpa[0],
                                                 np.radians(ref_pdpa[1]))
                plt.plot(ref_q, ref_u, 'x', markersize=20, color='tab:green')
                plt.annotate(ref_label, (ref_q, ref_u),
                             textcoords="offset points", xytext=(0,10),
                             ha='center', color='tab:green', fontsize=12)

            c_mdp = plt.Circle((0, 0), radius=self._mdp99,
                               facecolor='tab:red', alpha=0.3, linewidth=1,
                               linestyle='--',
                               label=rf'MDP$_{{99}}$ = {self._mdp99*100:.2f} %')
            plt.gca().add_artist(c_mdp)

            pol_PD = polarization_fraction * 100
            pol_1sigmaPD = polarization_fraction_uncertainty * 100
            pol_PA = polarization_angle.angle.deg
            pol_1sigmaPA = polarization_angle_uncertainty.deg

            if not has_background:
                label_header = ""
            else:
                label_header = "Measured (Unpol subtracted)\n"

                plt.plot(unpol_Q/unpol_I, unpol_U/unpol_I,
                         'o', markersize=5, color='0.4',
                         label=rf'Unpol (PD$_{{1\sigma}}$ = {unpol_sQ*100:.0f} %)')

                for r in (1, 2, 3):
                    unpol_c  = plt.Circle((unpol_Q/unpol_I, unpol_U/unpol_I),
                                          radius=r*unpol_sQ,
                                          facecolor='none',
                                          edgecolor='0.4',
                                          linewidth=1)

                    plt.gca().add_artist(unpol_c)

            label_data = (label_header + \
                          f"PD = ({pol_PD:.1f} ± {pol_1sigmaPD:.1f})%\n"
                          f"PA = ({pol_PA:.1f} ± {pol_1sigmaPA:.1f}) deg")

            plt.plot(QN, UN, 'o', markersize=5, color='red', label=label_data)

            for r in (1, 2, 3):
                pol_c = plt.Circle((QN, UN),
                                   radius=r*polarization_fraction_uncertainty,
                                   facecolor='none',
                                   edgecolor='red',
                                   linewidth=1)
                plt.gca().add_artist(pol_c)

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

        for r in (0.25, 0.50, 0.75, 1.00):
            ls    = ('-' if r == 1.00 else '--')
            alpha = (0.5 if r == 1.00 else 0.3)

            c = plt.Circle((0,0), radius=r,
                           facecolor='none', edgecolor='k',
                           linewidth=1, linestyle=ls, alpha=alpha)
            plt.gca().add_artist(c)
            plt.annotate(f"{r:.2f}", (r, 0),
                         textcoords="offset points", xytext=(10,0),
                         ha='center', fontsize=8, color='k', alpha=0.3)

        plt.hlines(0, -1, 1, linewidth=1, color='k',
                   linestyle='--', alpha=0.3)
        plt.vlines(0, -1, 1, linewidth=1, color='k',
                   linestyle='--', alpha=0.3)

        plt.plot([1,-1], [1,-1], linewidth=1, color='k',
                 linestyle='--', alpha=0.3)
        plt.plot([1,-1], [-1,1], linewidth=1, color='k',
                 linestyle='--', alpha=0.3)

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
