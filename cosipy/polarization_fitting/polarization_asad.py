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

class PolarizationASAD():
    """
    Azimuthal scattering angle distribution (ASAD) method to fit
    polarization.

    Parameters
    ----------
    source : astropy.coordinates.sky_coordinate.SkyCoord
        Source direction
    source_spectrum : astromodels.functions.functions_1D
        Spectrum of source
    asad_bin_edges : astropy.coordinates.angles.core.Angle
        Bin edges for azimuthal scattering angle distribution
    data : dict or Histogram, or list of same
        Unbinned or binned data, or list of binned/unbinned data if
        separated in time
    background : dict or Histogram, or list of same
        Unbinned or binned background model, or list of backgrounds if
        separated in time
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
                 data, background,
                 sc_orientation, response_file, response_convention='RelativeX',
                 fit_convention=IAUPolarizationConvention(), show_plots=False):

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

        if not isinstance(background, list):
            background = [background]
        background = apply_energy_cut(background, energy_range)

        asads = self.create_data_asads(data, background, asad_bin_edges)

        asad_unpolarized, asads_polarized = self.create_simulated_asads(asad_bin_edges)
        asads['unpolarized'] = asad_unpolarized
        asads['polarized']   = asads_polarized

        self._asads = asads

        self._mu100 = self.calculate_mu100(asads['polarized'],
                                           asads['unpolarized'],
                                           show_plots)

        self._mdp99 = self.calculate_mdp99(asads['source'],
                                           asads['background_scaled'],
                                           self._mu100['mu'])

        if show_plots:

            uncertainty = np.sqrt(asads['source_and_background'].bin_error.contents**2 +
                                  asads['background_scaled'].bin_error.contents**2)

            self.plot_asad(asads['source'],
                           'Source ASAD',
                           uncertainty)

            self.plot_asad(asads['source_and_background'],
                           'Source+background ASAD',
                           asads['source_and_background'].bin_error.contents)

            self.plot_asad(asads['background'],
                           'Background ASAD',
                           asads['background'].bin_error.contents)

            self.plot_asad(asads['unpolarized'],
                           'Unpolarized ASAD')

    def create_data_asads(self, data, background, bin_edges):
        """
        Create azimuthal scattering angle distributions from data,
        and background model.

        Parameters
        ----------
        data : list
            list of source + background data sets
        background : list
            list of background models
        bin_edges : astropy.units.Quantity
            edges of azimuthal scattering angle bins

        Returns
        -------
        asads : dict
            Azimuthal scattering angle distributions (ASADs)

        """

        def compute_asad_from_datasets(datasets, bin_edges):
            """
            Accumulate an ASAD from a list of one or more data sets

            """

            asad = np.zeros(len(bin_edges) - 1)
            duration = 0.

            for s in datasets:
                if isinstance(s, dict): # unbinned
                    scattering_dirs = self.scattering_dirs_from_unbinned_data(s)
                    weights = None
                    times = s['TimeTags']
                else: # binned
                    scattering_dirs, weights = self.scattering_dirs_from_binned_data(s)
                    times = s.axes['Time'].edges.value

                asad += self.scattering_dirs_to_asad(scattering_dirs, bin_edges, weights)
                duration += np.ptp(times) # max - min

            return asad, duration

        asad_sb, source_duration = compute_asad_from_datasets(data, bin_edges)

        asad_background, background_duration = compute_asad_from_datasets(background, bin_edges)

        asad_background_scaled = asad_background * source_duration / background_duration
        asad_source = asad_sb - asad_background_scaled

        axis = Axis(bin_edges)
        asads = {
            'source_and_background' : Histogram(axis, contents=asad_sb, copy_contents=False),
            'background' : Histogram(axis, contents=asad_background, copy_contents=False),
            'background_scaled' : Histogram(axis, contents=asad_background_scaled, copy_contents=False),
            'source' : Histogram(axis, contents=asad_source, copy_contents=False),
        }

        return asads

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

    def scattering_dirs_from_binned_data(self, binned_data):
        """
        Extract scattering directions from a binned data set.

        Parameters
        ----------
        binned_data : Histogram
            Data binned in Compton data space
        bin_edges : astropy.units.Quantity
            edges of azimuthal scattering angle bins

        Returns
        -------
        scattering_dirs : SkyCoord array
           Array of scattering directions
        weights : array of float
           Weights for each scattering direction

        """

        psichi_axis = binned_data.axes['PsiChi']
        pix = np.arange(psichi_axis.nbins)

        if isinstance(psichi_axis.coordsys, SpacecraftFrame):
            # source is in spacecraft-local frame
            lon, lat = psichi_axis.pix2ang(pix, lonlat=True)
            scattering_dirs = SkyCoord(lon, lat,
                                       unit=u.deg, frame=self._convention.frame)
        else:
            # source is in inertial frame
            scattering_dirs = psichi_axis.pix2skycoord(pix).transform_to('icrs')

        weights = binned_data.project('PsiChi').to_dense(copy=False).contents

        # strip any unit
        weights = weights.value if isinstance(weights, u.Quantity) else weights

        return scattering_dirs, weights

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

    def fit(self, p0=None, bounds=None, show_plots=False):
        """
        Fit the polarization fraction and angle.

        Parameters
        ----------
        p0 : list or np.array, optional
            Initial guess for parameter values
        bounds : 2-tuple of float, list, or np.array, optional
            Lower & upper bounds on parameters. Default is
            (0, 0, 0) and (inf, inf, pi)
        show_plots : bool, optional
            Option to show plots. Default is False

        Returns
        -------
        polarization : dict
            Polarization fraction, polarization angle in the IAU
            convention, and best fit parameter values for fitted
            sinusoid, and associated uncertainties

        """

        uncertainty = np.sqrt(self._asads['source_and_background'].bin_error.contents**2 +
                              self._asads['background_scaled'].bin_error.contents**2)
        asad_source_corrected, sigma = self.correct_asad(self._asads['source'],
                                                         self._asads['unpolarized'],
                                                         uncertainty)

        params, uncertainties = self.fit_asad(asad_source_corrected,
                                              p0, bounds, sigma)

        # polarization fraction
        pf = params[1] / (params[0] * self._mu100['mu'])
        pf_uncertainty = pf * np.sqrt((uncertainties[0] / params[0])**2 +
                                      (uncertainties[1] / params[1])**2 +
                                      (self._mu100['uncertainty'] /
                                       self._mu100['mu'])**2)

        # polarization angle (must fix range if user supplied bounds)
        pa = Angle(params[2], unit=u.rad)
        pa.wrap_at(180 * u.deg, inplace=True)
        pa = np.where(pa < 0, pa + 180*u.deg, pa)

        pa_uncertainty = Angle(uncertainties[2], unit=u.rad)

        logger.info('Best fit polarization fraction: '
                    f'{pf:.3f} +/- {pf_uncertainty:.3f}')

        logger.info('Best fit polarization angle (IAU convention): '
                    f'{pa.deg:.3f} +/- {pa_uncertainty.deg:.3f}')

        if self._mdp99 > pf:
            logger.info('Polarization fraction is below MDP!',
                        f'MDP: {self._mdp99:.3f}')

        if show_plots:
            self.plot_asad(asad_source_corrected,
                           'Corrected Source ASAD',
                           sigma,
                           coefficients = params)

        # return angle as PolarizationAngle
        pa = PolarizationAngle(pa, self._source,
                               convention=self._convention)
        pa = pa.transform_to(IAUPolarizationConvention())

        return {
            'fraction': pf,
            'angle': pa,
            'fraction uncertainty': pf_uncertainty,
            'angle uncertainty': pa_uncertainty,
            'best fit parameter values': params,
            'best fit parameter uncertainties': uncertainties
        }
