import numpy as np

import matplotlib.pyplot as plt

import astropy.units as u
from astropy.coordinates import Angle

from histpy import Axis, Histogram

from cosipy.polarization.polarization_angle import PolarizationAngle
from cosipy.polarization.conventions import IAUPolarizationConvention
from cosipy.polarization_fitting import PolarizationFitting

import logging
logger = logging.getLogger(__name__)

class PolarizationStokes(PolarizationFitting):
    """
    Stokes parameter method to fit polarization.

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

        super().__init__(source, source_spectrum, sc_orientation,
                         response_file, response_convention,
                         fit_convention)

        energy_edges = self.response.axes['Em'].edges.value
        energy_range = ( np.min(energy_edges), np.max(energy_edges) )

        if not isinstance(data, list):
            data = [data]
        data = self.apply_energy_cut(data, energy_range)

        asad_sb, source_duration, self._data_scattering_angles = \
            self.asad_and_angles_from_data(data, asad_bin_edges, show_plots=show_plots)

        asads = { 'source_and_background' : asad_sb }

        if background is not None:
            if not isinstance(background, list):
                background = [background]
            background = self.apply_energy_cut(background, energy_range)

            asad_background, background_duration, self._background_scattering_angles = \
                self.asad_and_angles_from_data(background, asad_bin_edges, show_plots=show_plots)

            self._backscal = source_duration / background_duration
            asad_background_scaled = asad_background * self._backscal
            asad_source = asad_sb - asad_background_scaled

            asads['background'] = asad_background
            asads['background_scaled'] = asad_background_scaled
            asads['source'] = asad_source

        else:
            logger.info('No background provided. Will not subtract background from data.')
            self._background_scattering_angles = None
            self._background_duration = 0
            asads['background_scaled'] = None
            asads['source'] = asads['source_and_background']

        asads['unpolarized'], asads['polarized'] = \
            self.create_simulated_asads(asad_bin_edges)

        self._mu100 = self.calculate_mu100(asads['polarized'],
                                           asads['unpolarized'],
                                           show_plots)

        self._mdp99 = self.calculate_mdp99(asads['source'],
                                           asads['background_scaled'],
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
        asad : Histogram
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
            plt.hist(scattering_angles, bins=self.response.axes['Pol'].nbins, alpha=0.5,
                     histtype='step', linewidth=2, label='Response binning')
            plt.xlabel('Azimuthal angle (radians)')
            plt.ylabel('Counts')
            plt.legend()
            plt.show()

        asad_hist = Histogram(bin_edges, contents=asad, copy_contents=False)
        return asad_hist, duration, scattering_angles

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

        ############################################################
        # ATTENTION: I need to add 90 degrees because the stokes
        # convention assumes that EVPA // source polarization, while
        # for Compton scatttering it is perpendicular)
        qs = stokes_q(angles - np.pi/2)
        us = stokes_u(angles - np.pi/2)

        if show_plots:
            plt.figure()
            plt.title(f'{label} Stokes parameters ({len(qs)} events)')
            plt.hist(qs, bins=50, alpha=0.5, label='q$_s$')
            plt.hist(us, bins=50, alpha=0.5, label='u$_s$')
            plt.xlabel('Pseudo Stokes parameter')
            plt.legend()
            plt.show()

        return qs, us

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

        pol_I = len(src_qs)          # I in eqn 10a
        pol_Q = np.sum(src_qs) / mu  # 2/mu x Q in eqn 10b
        pol_U = np.sum(src_us) / mu  # 2/mu x q in eqn 10c
        logger.info(f'I, Q, U, mu: {pol_I} {pol_Q} {pol_U} {mu}')

        if not has_background:
            logger.info('No background data provided, assuming no background contribution.')

            I = pol_I
            QN = pol_Q/pol_I # 2/mu x mathcal{Q} in eqn 11b
            UN = pol_U/pol_I # 2/mu x mathcal{U} in eqn 11b

            logger.info(f'Q, U (unsubtracted): {QN} {UN}')

        else:
            logger.info('Unpolarized bkg (or simulation) provided, subtracting its contribution.')

            bkg_qs, bkg_us = self._compute_pseudo_stokes(self._background_scattering_angles, show_plots=False)

            unpol_I = len(bkg_qs) * self._backscal
            unpol_Q = np.sum(bkg_qs) * self._backscal / mu
            unpol_U = np.sum(bkg_us) * self._backscal / mu
            logger.info(f'Q, U unpolarized: {unpol_Q/unpol_I} {unpol_U/unpol_I}')

            I = pol_I - unpol_I
            logger.info(f'check I(src+bkg) vs I(src): {pol_I} {I}')

            QN = pol_Q/pol_I + unpol_Q/unpol_I * self._backscal
            UN = pol_U/pol_I + unpol_U/unpol_I * self._backscal

            logger.info(f'Q, U, subtracted: {QN} {UN}')

            # analogous to mu * mathcal{Q_r}
            unpol_modulation = mu * np.sqrt(unpol_Q**2 + unpol_U**2) / unpol_I

            # FIXED: analogous to eqn 28a/b of Kislat 2015
            unpol_sI = np.sqrt(unpol_I - 1)
            unpol_sQ = np.sqrt(2 - unpol_modulation**2) / unpol_sI / mu
            logger.info(f'Q, U unpolarized uncertainty: {unpol_sQ*100} %')

        # Reconstructed polarization fraction + uncertainty

        # eqn 21 of Kislat 2015
        polarization_fraction = np.sqrt(QN**2 + UN**2)

        # eqn 36 of Kislat 2015
        m = mu * polarization_fraction
        polarization_fraction_uncertainty = np.sqrt((2 - m**2)/((I - 1) * mu**2))

        # Reconstructed polarization angle + uncertainty

        # eqn 22 of Kislat 2015
        pol_PA = 0.5 * np.arctan2(UN, QN)
        # Convert to 0 to 180 deg (just the convention)
        if pol_PA < 0:
            pol_PA += np.pi

        # eqn 37 of Kislat 2015
        pol_1sigmaPA = np.degrees(1 / (m * np.sqrt(2 * (I - 1))))

        polarization_angle = Angle(np.degrees(pol_PA), unit=u.deg)
        polarization_angle = PolarizationAngle(polarization_angle, self.source,
                                               convention=self.convention).transform_to(IAUPolarizationConvention())
        polarization_angle_uncertainty = Angle(pol_1sigmaPA, unit=u.deg)

        # FIXED: eqn 28 a/b of Kislat 2015
        pol_sI = np.sqrt(I - 1)
        pol_sQ = np.sqrt(2 - (mu * QN)**2) / pol_sI / mu
        pol_sU = np.sqrt(2 - (mu * UN)**2) / pol_sI / mu
        logger.info(f'Q/I, U/I, uncertainty: {pol_sQ} {pol_sU} {np.sqrt(pol_sQ)}')

        polarization = {'fraction': polarization_fraction,
                        'angle': polarization_angle,
                        'fraction uncertainty': polarization_fraction_uncertainty,
                        'angle uncertainty': polarization_angle_uncertainty,
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

    def plot_pseudostokes(self):
        """
        Plot the pseudo-stokes parameters for the source and, if
        background was specified, for the background.
        """

        self._compute_pseudo_stokes(self._data_scattering_angles,
                                    show_plots=True, label="Source")

        if self._background_scattering_angles is not None:
            self._compute_pseudo_stokes(self._background_scattering_angles,
                                        show_plots=True, label="Background")


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
