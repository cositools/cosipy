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
    sc_orientation : cosipy.spacecraftfile.SpacecraftFile.SpacecraftFile
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

        if isinstance(fit_convention.frame, SpacecraftFrame):
            if not isinstance(source.frame, SpacecraftFrame):
                attitude = sc_orientation.get_attitude()[0]
                source = source.transform_to(SpacecraftFrame(attitude=attitude))
                logger.warning("The source direction is being converted to the spacecraft "
                               "frame using the attitude at the first timestamp of the orientation.")
        else:
            source = source.transform_to('icrs')

        if ((isinstance(fit_convention, MEGAlibRelativeX) and response_convention != 'RelativeX') or
            (isinstance(fit_convention, MEGAlibRelativeY) and response_convention != 'RelativeY') or
            (isinstance(fit_convention, MEGAlibRelativeZ) and response_convention != 'RelativeZ')):
            raise RuntimeError("If performing fit in spacecraft frame, "
                               "fit convention must match convention of response.")

        self._convention = fit_convention
        self._response_convention = response_convention

        self._source = source

        self._response = FullDetectorResponse.open(response_file, pa_convention=self._response_convention)

        self._spectral_flux = get_integrated_spectral_model(source_spectrum, self._response.axes['Ei'])

        energy_edges = self._response.axes['Em'].edges.value
        self._energy_range = ( min(energy_edges), max(energy_edges) )

        self._ori = sc_orientation

        if not isinstance(data, list):
            data = [data]

        if not isinstance(background, list):
            background = [background]

        asads = self.create_asads(data, background, asad_bin_edges)

        uncertainty = np.sqrt(asads['source_and_background'].bin_error.contents**2 +
                              asads['background_scaled'].bin_error.contents**2)
        asad_corrected, sigma = self.correct_asad(asads['source'],
                                                  asads['unpolarized'],
                                                  uncertainty)

        self._asad_source_corrected = asad_corrected
        self._sigma = sigma

        self._mu_100 = self.calculate_mu100(asads['polarized'],
                                            asads['unpolarized'],
                                            show_plots)

        self._mdp = self.calculate_mdp(asads['source'],
                                       asads['background_scaled'],
                                       self._mu_100['mu'])

        if show_plots:

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

    def create_asads(self, data, background, bin_edges):
        """
        Create azimuthal scattering angle distributions from data,
        background model, and response.

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

        asad_background_scaled = (asad_background * source_duration / background_duration)
        asad_source = asad_sb - asad_background_scaled

        asad_unpolarized, asads_polarized = self.create_simulated_asads(bin_edges)

        axis = Axis(bin_edges)
        asads = {
            'source' : Histogram(axis, contents=asad_source, copy_contents=False),
            'background' : Histogram(axis, contents=asad_background, copy_contents=False),
            'background_scaled' : Histogram(axis, contents=asad_background_scaled, copy_contents=False),
            'source_and_background' : Histogram(axis, contents=asad_sb, copy_contents=False),
            'unpolarized' : Histogram(axis, contents=asad_unpolarized, copy_contents=False),
            'polarized' : [ Histogram(axis, contents=asad, copy_contents=False) for asad in asads_polarized ]
        }

        return asads

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

        # select events by energy range
        energies = unbinned_data['Energies']
        emin, emax = self._energy_range
        emask = ((energies >= emin) & (energies <= emax))

        if isinstance(self._convention.frame, SpacecraftFrame):
            # source is in spacecraft-local frame
            scattering_dirs = SkyCoord(lon=unbinned_data['Chi local'][emask],
                                       lat=np.pi/2 - unbinned_data['Psi local'][emask],
                                       unit=u.rad, frame=self._convention.frame)
        else:
            # source is in inertial frame
            scattering_dirs = SkyCoord(l=unbinned_data['Chi galactic'][emask],
                                       b=unbinned_data['Psi galactic'][emask],
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

        # sparse contents has no unit
        assert binned_data.is_sparse
        weights = binned_data.project('PsiChi').todense().contents

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
            target_in_sc_frame = self._ori.get_target_in_sc_frame(source)
            dwell_time_map = self._ori.get_dwell_map(base=self._response,
                                                     src_path=target_in_sc_frame)
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
        pol_angles =    np.hstack(([0.], pol_axis.centers.to_value(u.deg)))

        scattering_dirs, weights = self.scattering_dirs_from_response(self._spectral_flux,
                                                                      pol_fractions,
                                                                      pol_angles)

        asads = [ self.scattering_dirs_to_asad(scattering_dirs, bin_edges, weight)
                  for weight in weights ]

        return asads[0], asads[1:]

    def scattering_dirs_to_asad(self, directions, bin_edges, weights=None):
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

    @staticmethod
    def plot_asad(asad, title, error=None, coefficients=None):
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
            y = PolarizationASAD.asad_sinusoid(x, *coefficients)
            plt.plot(np.rad2deg(x), y, color='green')

        plt.show()

    def correct_asad(self, asad_data, asad_unpolarized,
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
        mu_100 : dict
            Modulation of 100% polarized source and uncertainty of
            constant function fit to modulation in all polarization angle
            bins

        """

        def constant(x, a):
            # constant approximation a to
            # mu_100 values x.
            return a

        pol_axis = self._response.axes['Pol']
        pol_angles = pol_axis.centers.to_value(u.deg)

        mu_100_vals = []
        for i in range(pol_axis.nbins):
            logger.info(f'Polarization angle bin: {pol_axis.edges[i]} to {pol_axis.edges[i+1]} deg')

            asad_polarized_corrected, _ = self.correct_asad(asads_polarized[i], asad_unpolarized)
            mu_100, coefficients = self.calculate_mu(asad_polarized_corrected,
                                                     bounds=((0, 0, 0), (np.inf,np.inf,np.pi)))

            mu_100_vals.append(mu_100)

            fitted_angle = Angle(coefficients[2], unit=u.rad)
            fitted_angle.wrap_at(180 * u.deg, inplace=True)
            fitted_angle = np.where(fitted_angle < 0, fitted_angle + 180*u.deg, fitted_angle)
            logger.info(f'Fitted angle: {fitted_angle.deg} deg')

            if show_plots:
                self.plot_asad(asad_polarized_corrected,
                               f'Corrected 100% Polarized ASAD ({int(pol_angles[i])} deg)',
                               coefficients=coefficients)

        mu_100s               = [ m['mu']          for m in mu_100_vals ]
        mu_100_uncertainties  = [ m['uncertainty'] for m in mu_100_vals ]
        popt, pcov = curve_fit(constant,
                               pol_angles, mu_100s,
                               sigma = mu_100_uncertainties)
        result = {'mu': popt[0], 'uncertainty': pcov[0][0]}

        if show_plots:
            plt.scatter(pol_angles, mu_100s)
            plt.errorbar(pol_angles, mu_100s,
                         yerr=mu_100_uncertainties, linewidth=0, elinewidth=1)
            plt.plot((0, 175), (result['mu'], result['mu']))
            plt.xlabel('Polarization Angle (degrees)')
            plt.ylabel('mu_100')
            plt.show()

        logger.info(f'mu_100: {result["mu"]:.2f}')

        return result

    def calculate_mu(self, asad,
                     p0=None, bounds=None, sigma=None):
        """
        Calculate the modulation (mu).

        Parameters
        ----------
        asad : Histogram
           ASAD
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
        params, uncertainties = self.fit_asad(asad,
                                              p0, bounds, sigma)

        mu = params[1] / params[0]
        mu_uncertainty = mu * np.sqrt((uncertainties[0]/params[0])**2 +
                                      (uncertainties[1]/params[1])**2)

        logger.info(f'Modulation: {mu:.3f} +/- {mu_uncertainty:.3f}')

        modulation = {'mu': mu, 'uncertainty': mu_uncertainty}

        return modulation, params

    @staticmethod
    def calculate_mdp(asad_source, asad_background_scaled, mu_100):
        """
        Calculate the minimum detectable polarization (MDP) of the source.

        Parameters
        ----------
        asad_source : Histogram
            ASAD for source
        asad_background_scaled : Histogram
            ASAD for background (scaled)
        mu_100 : float
            Modulation of 100% polarized source

        Returns
        -------
        mdp : float
            MDP of source
        """

        source_counts = np.sum(asad_source)
        background_counts = np.sum(asad_background_scaled)

        mdp = 4.29 / mu_100 * np.sqrt(source_counts + background_counts) / source_counts

        logger.info(f'Minimum detectable polarization (MDP) of source: {mdp:.3f}')

        return mdp

    def fit(self, p0=None, bounds=None, show_plots=False):
        """
        Fit the polarization fraction and angle.

        Parameters
        ----------
        p0 : list or np.array, optional
            Initial guess for parameter values
        bounds : 2-tuple of float, list, or np.array, optional
            Lower & upper bounds on parameters. Default is ([0, 0, 0],
            [np.inf,np.inf,np.pi])
        show_plots : bool, optional
            Option to show plots. Default is False

        Returns
        -------
        polarization : dict
            Polarization fraction, polarization angle in the IAU
            convention, and best fit parameter values for fitted
            sinusoid, and associated uncertainties

        """

        if bounds is None:
            bounds = ((0, 0, 0), (np.inf,np.inf,np.pi))

        params, uncertainties = self.fit_asad(self._asad_source_corrected,
                                              p0, bounds, self._sigma)

        # polarization fraction
        pf = params[1] / (params[0] * self._mu_100['mu'])
        pf_uncertainty = pf * np.sqrt((uncertainties[0] / params[0])**2 +
                                      (uncertainties[1] / params[1])**2 +
                                      (self._mu_100['uncertainty'] /
                                       self._mu_100['mu'])**2)

        # polarization angle
        pa = Angle(params[2], unit=u.rad)
        pa.wrap_at(180 * u.deg, inplace=True)
        pa = np.where(pa < 0, pa + 180*u.deg, pa)

        pa_uncertainty = Angle(uncertainties[2], unit=u.rad)

        logger.info('Best fit polarization fraction: '
                    f'{pf:.3f} +/- {pf_uncertainty:.3f}')

        logger.info('Best fit polarization angle (IAU convention): '
                    f'{pa.deg:.3f} +/- {pa_uncertainty.deg:.3f}')

        if self._mdp > pf:
            logger.info('Polarization fraction is below MDP!',
                        f'MDP: {self._mdp:.3f}')

        if show_plots:
            self.plot_asad(self._asad_source_corrected,
                           'Corrected Source ASAD',
                           self._sigma,
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

    @staticmethod
    def fit_asad(asad, p0, bounds, sigma):
        """
        Fit the ASAD with a sinusoid.

        Parameters
        ----------
        asad : Histogram
            ASAD
        p0 : np.array or None
            Initial guess for parameter values
        bounds : 2-tuple of float or array-like
            Lower & upper bounds on parameters
        sigma : float or array-like
            Uncertainties in y data

        Returns
        -------
        popt : np.ndarray
            Fitted parameter values
        uncertainties : np.ndarray
            Uncertainty on each parameter value
        """

        popt, pcov = curve_fit(PolarizationASAD.asad_sinusoid,
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
