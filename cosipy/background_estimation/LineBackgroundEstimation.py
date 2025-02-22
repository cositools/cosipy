import logging
logger = logging.getLogger(__name__)

from histpy import Histogram, Axis, Axes
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import integrate
from iminuit import Minuit

class LineBackgroundEstimation:
    """
    A class for estimating and modeling background in line spectra.

    This class provides methods for setting up a background model,
    fitting it to data, and generating background model histograms.

    Attributes
    ----------
    event_histogram : Histogram
        The input event histogram.
    energy_axis : Axis
        The energy axis of the event histogram.
    energy_spectrum : Histogram
        The projected energy spectrum.
    bkg_spectrum_model : callable
        The background spectrum model function.
    bkg_spectrum_model_parameter : list
        The parameters of the background spectrum model.
    mask : ndarray
        Boolean mask for excluding regions from the fit.
    """

    def __init__(self, event_histogram):
        """
        Initialize the LineBackgroundEstimation object.

        Parameters
        ----------
        event_histogram : Histogram
            The input event histogram.
        """
        # event histogram
        self.event_histogram = event_histogram

        # projected histogram onto the energy axis
        self.energy_axis = self.event_histogram.axes['Em']
        self.energy_spectrum = self.event_histogram.project('Em')
        if self.energy_spectrum.is_sparse:
            self.energy_spectrum = self.energy_spectrum.to_dense()
        
        self.energy_spectrum.clear_underflow_and_overflow()

        # background fitting model
        self.bkg_spectrum_model = None
        self.bkg_spectrum_model_parameter = None

        # bins to be masked
        self.mask = np.zeros(self.energy_axis.nbins, dtype=bool)
        
    def set_bkg_energy_spectrum_model(self, bkg_spectrum_model, bkg_spectrum_model_parameter):
        """
        Set the background energy spectrum model and its initial parameters.

        Parameters
        ----------
        bkg_spectrum_model : callable
            The background spectrum model function.
        bkg_spectrum_model_parameter : list
            Initial parameters for the background spectrum model.
        """
        self.bkg_spectrum_model = bkg_spectrum_model
        self.bkg_spectrum_model_parameter = bkg_spectrum_model_parameter

    def set_mask(self, *mask_energy_ranges):
        """
        Set mask for excluding energy ranges from the fit.

        Parameters
        ----------
        *mask_energy_ranges : tuple
            Variable number of energy range tuples to be masked.
        """
        self.mask = np.zeros(self.energy_axis.nbins, dtype=bool)
        for mask_energy_range in mask_energy_ranges:
            this_mask = (mask_energy_range[0] <= self.energy_axis.bounds[:, 1]) & (self.energy_axis.bounds[:, 0] <= mask_energy_range[1])
            self.mask = self.mask | this_mask
    
    def _calc_expected_spectrum(self, *args):
        """
        Calculate the expected spectrum based on the current model and parameters.

        Parameters
        ----------
        *args : float
            Model parameters.

        Returns
        -------
        ndarray
            The calculated expected spectrum.
        """
        return np.array([integrate.quad(lambda x: self.bkg_spectrum_model(x, *args), *energy_range)[0] for energy_range in self.energy_axis.bounds.value])

    def _negative_log_likelihood(self, *args):
        """
        Calculate the negative log-likelihood for the current model and parameters.

        Parameters
        ----------
        *args : float
            Model parameters.

        Returns
        -------
        float
            The calculated negative log-likelihood.
        """
        expected_spectrum = self._calc_expected_spectrum(*args)
        return -np.sum(self.energy_spectrum.contents[~self.mask] * np.log(expected_spectrum)[~self.mask]) + np.sum(expected_spectrum[~self.mask])
    
    def plot_energy_spectrum(self):
        """
        Plot the energy spectrum and the fitted model if available.

        Returns
        -------
        tuple
            A tuple containing the matplotlib axis object and any additional objects returned by the plotting function.
        """
        ax, _ = self.energy_spectrum.draw(label='input data')

        # plot background model
        if self.bkg_spectrum_model is not None:
            expected_spectrum = self._calc_expected_spectrum(*self.bkg_spectrum_model_parameter)
            ax.plot(self.energy_axis.centers, expected_spectrum, label='model')

        # shade mask regions
        start, end = None, None
        for i, this_mask in enumerate(self.mask):
            if this_mask:
                if start is None:
                    start, end = self.energy_axis.bounds[i]
                else:
                    _, end = self.energy_axis.bounds[i]
            else:
                if start is not None:
                    ax.axvspan(start.value, end.value, color='lightgrey', alpha=0.5)
                    start, end = None, None
        
        if start is not None:
            ax.axvspan(start.value, end.value, color='lightgrey', alpha=0.5)

        # legend and grid
        ax.legend()
        ax.grid()
        
        return ax, _
        
    def fit_energy_spectrum(self):
        """
        Fit the background energy spectrum model to the data.

        Returns
        -------
        Minuit
            The Minuit object containing the fit results.
        """
        m = Minuit(self._negative_log_likelihood, *self.bkg_spectrum_model_parameter)
        m.errordef = Minuit.LIKELIHOOD
        
        m.migrad()
        m.hesse()

        # update the background model parameters
        self.bkg_spectrum_model_parameter = list(m.values)
        self.bkg_spectrum_model_parameter_errors = list(m.errors)
        
        return m

    def _get_weight_indices(self, energy_range):
        """
        Get the weight and indices for a given energy range.

        Parameters
        ----------
        energy_range : tuple
            The energy range to calculate the weight for.

        Returns
        -------
        tuple
            A tuple containing the calculated weight and the corresponding energy indices.
        """
        energy_indices = np.where((energy_range[0] <= self.energy_axis.lower_bounds) & (self.energy_axis.upper_bounds <= energy_range[1]))[0]

        if len(energy_indices) == 0:
            raise ValueError("The input energy range is too narrow to find a corresponding energy bin.")

        integrate_energy_range = [self.energy_axis.lower_bounds[energy_indices[0]].value, self.energy_axis.upper_bounds[energy_indices[-1]].value]

        if integrate_energy_range[0] != energy_range[0].value or integrate_energy_range[1] != energy_range[1].value:
            logger.info(f"The energy range {energy_range.value} is modified to {integrate_energy_range}")
        weight = integrate.quad(lambda x: self.bkg_spectrum_model(x, *self.bkg_spectrum_model_parameter), *integrate_energy_range)[0]
        return weight, energy_indices

    def generate_bkg_model_histogram(self, source_energy_range, bkg_estimation_energy_ranges):
        """
        Generate a background model histogram based on the fitted model.

        Parameters
        ----------
        bkg_estimation_energy_ranges : list of tuple
            List of energy ranges for background estimation.
        smoothing_fwhm : float, optional
            Full width at half maximum for smoothing, by default None.

        Returns
        -------
        Histogram
            The generated background model histogram.
        """
        # intergrated spectrum in the background estimation energy ranges
        weights = []
        energy_indices_list = []
        for bkg_estimation_energy_range in bkg_estimation_energy_ranges:
            weight, energy_indices = self._get_weight_indices(bkg_estimation_energy_range)
            weights.append(weight)
            energy_indices_list.append(energy_indices)
            
        # intergrated spectrum in the source region
        source_weight = integrate.quad(lambda x: self.bkg_spectrum_model(x, *self.bkg_spectrum_model_parameter), *source_energy_range.value)[0]

        # prepare a new histogram
        new_axes = []
        for axis in self.event_histogram.axes:
            if axis.label != "Em":
                new_axes.append(axis)
            else:
                new_axes.append(Axis(source_energy_range, label = "Em"))
        
        bkg_model_histogram = Histogram(new_axes)

        # fill contents
        for energy_indices in energy_indices_list:
            for energy_index in energy_indices:
                if new_axes[0].label != "Em":
                    bkg_model_histogram[:] += self.event_histogram[:,energy_index].todense()
                else:
                    bkg_model_histogram[:] += self.event_histogram[energy_index].todense()

        # normalization
        corr_factor = source_weight / np.sum(weights)
        bkg_model_histogram[:] *= corr_factor

        return bkg_model_histogram
