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

class PolarizationASAD(PolarizationFitting):
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

        super().__init__(source, source_spectrum, sc_orientation,
                         response_file, response_convention,
                         fit_convention)

        energy_edges = self.get_response().axes['Em'].edges.value
        energy_range = ( np.min(energy_edges), np.max(energy_edges) )

        if not isinstance(data, list):
            data = [data]
        data = self.apply_energy_cut(data, energy_range)

        if not isinstance(background, list):
            background = [background]
        background = self.apply_energy_cut(background, energy_range)

        asads = self.create_data_asads(data, background, asad_bin_edges)

        asads['unpolarized'], asads['polarized'] = \
            self.create_simulated_asads(asad_bin_edges)

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
        pa = PolarizationAngle(pa, self.get_source(),
                               convention=self.get_convention())
        pa = pa.transform_to(IAUPolarizationConvention())

        return {
            'fraction': pf,
            'angle': pa,
            'fraction uncertainty': pf_uncertainty,
            'angle uncertainty': pa_uncertainty,
            'best fit parameter values': params,
            'best fit parameter uncertainties': uncertainties
        }
