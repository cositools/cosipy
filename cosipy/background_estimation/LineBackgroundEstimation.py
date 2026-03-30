import logging
logger = logging.getLogger(__name__)

from histpy import Histogram, Axis, Axes
import healpy as hp
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import integrate
from iminuit import Minuit
from itertools import product

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

        # Avoid log(0) using NumPy's machine epsilon
        expected_spectrum = np.maximum(expected_spectrum, np.finfo(float).eps)

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
        
    def fit_energy_spectrum(self, param_limits=None, fixed_params=None, stepsize_params=None):
        """
        Fit the background energy spectrum model to the data.

        Parameters
        ----------
        param_limits : dict, optional
            Dictionary containing parameter limits in the format {param_index: (lower_limit, upper_limit)}.
            For example, {0: (0, 10), 2: (None, 5)} sets the first parameter between 0 and 10,
            and the third parameter to have no lower limit but an upper limit of 5.
        fixed_params : dict, optional
            Dictionary containing fixed parameter values in the format {param_index: value}.
            For example, {1: 2.5} fixes the second parameter to 2.5.
        stepsize_params : dict, optional
            Dictionary containing initial step sizes for parameters in the format {param_index: step_size}.
            For example, {0: 0.1} sets the step size for the first parameter.

        Returns
        -------
        Minuit
            The Minuit object containing the fit results.
        """
        # Initialize Minuit with parameters
        m = Minuit(self._negative_log_likelihood, *self.bkg_spectrum_model_parameter)
        m.errordef = Minuit.LIKELIHOOD

        # Set parameter limits if provided
        if param_limits:
            for param_idx, (lower, upper) in param_limits.items():
                if param_idx < 0 or param_idx >= len(self.bkg_spectrum_model_parameter):
                    logger.warning(f"Parameter index {param_idx} out of range, skipping")
                    continue
                    
                param_name = m.parameters[param_idx]
                
                # Set the limits
                if lower is not None and upper is not None:
                    m.limits[param_name] = (lower, upper)
                elif lower is not None:
                    m.limits[param_name] = (lower, None)
                elif upper is not None:
                    m.limits[param_name] = (None, upper)
        
        # Fix parameters if provided
        if fixed_params:
            for param_idx, value in fixed_params.items():
                if param_idx < 0 or param_idx >= len(self.bkg_spectrum_model_parameter):
                    logger.warning(f"Parameter index {param_idx} out of range, skipping")
                    continue
                    
                param_name = m.parameters[param_idx]
                m.values[param_name] = value
                m.fixed[param_name] = True
        
        # Set error parameters if provided
        if stepsize_params:
            for param_idx, step_size in stepsize_params.items():
                if param_idx < 0 or param_idx >= len(self.bkg_spectrum_model_parameter):
                    logger.warning(f"Parameter index {param_idx} out of range, skipping")
                    continue
                    
                param_name = m.parameters[param_idx]
                m.errors[param_name] = step_size

        # Run the optimization
        m.migrad()
        m.hesse()

        # Update the background model parameters
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

    def _apply_spatial_filter(self, bkg_model, new_axes, smoothing_fwhm=None, l_cut=None):
        """
        Apply spatial filter (smoothing or l_cut) to the last axis of bkg_model.
    
        Parameters
        ----------
        bkg_model : np.ndarray
            Background model array. The last axis is the HEALPix spatial axis.
        new_axes : Axes
            Axes object containing PsiChi axis with nside information.
        smoothing_fwhm : :py:class:`astropy.units.quantity.Quantity`, optional
            Full width at half maximum for Gaussian smoothing.
        l_cut : int, optional
            Maximum multipole moment to retain.
    
        Returns
        -------
        np.ndarray
            Filtered background model array.
        """
        if smoothing_fwhm is not None:
            for idx in product(*[range(s) for s in bkg_model.shape[:-1]]):
                bkg_model[idx] = hp.smoothing(bkg_model[idx], fwhm=smoothing_fwhm.to('rad').value)
    
        if l_cut is not None:
            logger.info(f"Applying low-pass filter with l_cut={l_cut}: retaining features on angular scales larger than ~{180.0 / l_cut:.1f} degrees.")

            lmax = new_axes['PsiChi'].nside * 4
            ell, _ = hp.Alm.getlm(lmax)

            for idx in product(*[range(s) for s in bkg_model.shape[:-1]]):
                alm = hp.map2alm(bkg_model[idx], lmax=lmax)
                alm[ell > l_cut] = 0.0
                bkg_model[idx] = hp.alm2map(alm, nside=new_axes['PsiChi'].nside)

            # clip negative values introduced by the low-pass filter
            bkg_model = np.clip(bkg_model, 0.0, None)
    
        return bkg_model
    
    def _rebin_phi(self, bkg_model, rebin_phi):
        """
        Rebin the second-to-last axis (Phi) by summing over rebin_phi bins.
    
        Parameters
        ----------
        bkg_model : np.ndarray
            Background model array.
        rebin_phi : int
            Number of bins to merge.
    
        Returns
        -------
        rebinned : np.ndarray
            Rebinned array.
        n_groups : int
            Number of groups after rebinning.
        phi_axis_idx : int
            Index of the Phi axis.
        """
        phi_axis_idx = len(bkg_model.shape) - 2
        n_phi = bkg_model.shape[phi_axis_idx]
        n_groups = n_phi // rebin_phi
    
        rebinned_shape = list(bkg_model.shape)
        rebinned_shape[phi_axis_idx] = n_groups
        rebinned = np.zeros(rebinned_shape)
    
        for g in range(n_groups):
            sl = [slice(None)] * len(bkg_model.shape)
            sl[phi_axis_idx] = slice(g * rebin_phi, (g + 1) * rebin_phi)
            rebinned_sl = [slice(None)] * len(rebinned.shape)
            rebinned_sl[phi_axis_idx] = g
            rebinned[tuple(rebinned_sl)] = np.sum(bkg_model[tuple(sl)], axis=phi_axis_idx)
    
        return rebinned, n_groups, phi_axis_idx
    
    def _unbin_phi(self, bkg_model, rebinned, n_groups, rebin_phi, phi_axis_idx):
        """
        Restore the rebinned Phi axis back to the original binning.
    
        The spatial pattern of each Phi bin is replaced by the filtered template,
        normalized so that the total count of each original Phi bin is preserved.
    
        Parameters
        ----------
        bkg_model : np.ndarray
            Original background model array (before rebinning).
        rebinned : np.ndarray
            Filtered rebinned array (template).
        n_groups : int
            Number of groups after rebinning.
        rebin_phi : int
            Number of bins per group.
        phi_axis_idx : int
            Index of the Phi axis.
    
        Returns
        -------
        np.ndarray
            Background model restored to original binning.
        """
        for g in range(n_groups):
            rebinned_sl = [slice(None)] * len(rebinned.shape)
            rebinned_sl[phi_axis_idx] = g
    
            # normalize template over spatial axis -> spatial pattern only
            # shape: same as one Phi slice of bkg_model
            template = rebinned[tuple(rebinned_sl)]
            template_sum = np.sum(template)
            template_normalized = np.where(template_sum > 0, template / template_sum, 0.0)
    
            # apply to each original Phi bin, preserving its total count
            for i in range(rebin_phi):
                sl_i = [slice(None)] * len(bkg_model.shape)
                sl_i[phi_axis_idx] = g * rebin_phi + i
    
                factor_i = np.sum(bkg_model[tuple(sl_i)])  # scalar: total count of this Phi bin
                bkg_model[tuple(sl_i)] = template_normalized * factor_i
    
        return bkg_model
    
    def generate_bkg_model_histogram(self, source_energy_range, bkg_estimation_energy_ranges,
                                      smoothing_fwhm=None, l_cut=None, rebin_phi=None):
        """
        Generate a background model histogram based on the fitted model.
    
        Parameters
        ----------
        source_energy_range : tuple
            Energy range for background model.
        bkg_estimation_energy_ranges : list of tuple
            List of energy ranges for background estimation.
        smoothing_fwhm : :py:class:`astropy.units.quantity.Quantity`, optional
            Full width at half maximum for Gaussian smoothing, by default None.
        l_cut : int, optional
            Maximum multipole moment to retain. Features on angular scales larger
            than approximately 180/l_cut degrees will be preserved. Default is None.
        rebin_phi : int, optional
            Number of Phi bins to merge before applying the spatial filter.
            After filtering, the original binning is restored: the spatial pattern
            of each Phi bin is replaced by the filtered template, normalized to
            preserve the original total count of each Phi bin. Default is None.
    
        Returns
        -------
        Histogram
            The generated background model histogram.
        """
        # validate options
        if smoothing_fwhm is not None and l_cut is not None:
            logger.error("smoothing_fwhm and l_cut cannot be specified at the same time.")
            raise ValueError("smoothing_fwhm and l_cut cannot be specified at the same time.")
    
        # intergrated spectrum in the background estimation energy ranges
        weights = []
        energy_indices_list = []
        for bkg_estimation_energy_range in bkg_estimation_energy_ranges:
            weight, energy_indices = self._get_weight_indices(bkg_estimation_energy_range)
            weights.append(weight)
            energy_indices_list.append(energy_indices)
    
        # intergrated spectrum in the source region
        source_weight = integrate.quad(
            lambda x: self.bkg_spectrum_model(x, *self.bkg_spectrum_model_parameter),
            *source_energy_range.value
        )[0]
    
        # prepare a new histogram
        new_axes = []
        for axis in self.event_histogram.axes:
            if axis.label != "Em":
                new_axes.append(axis)
            else:
                new_axes.append(Axis(source_energy_range, label="Em"))
        new_axes = Axes(new_axes, copy_axes=False)
        bkg_model = np.zeros(new_axes.shape)
    
        # fill contents
        for energy_indices in energy_indices_list:
            for energy_index in energy_indices:
                if new_axes[0].label != "Em":
                    bkg_model += self.event_histogram[:, energy_index].todense()
                else:
                    bkg_model += self.event_histogram[energy_index].todense()
    
        # normalization
        corr_factor = source_weight / np.sum(weights)
        bkg_model *= corr_factor
    
        # apply spatial filter (with or without phi rebinning)
        if smoothing_fwhm is not None or l_cut is not None:
            if rebin_phi is not None:
                # rebin Phi axis → filter → restore original binning
                rebinned, n_groups, phi_axis_idx = self._rebin_phi(bkg_model, rebin_phi)
                rebinned = self._apply_spatial_filter(rebinned, new_axes, smoothing_fwhm, l_cut)
                bkg_model = self._unbin_phi(bkg_model, rebinned, n_groups, rebin_phi, phi_axis_idx)
            else:
                # filter directly without rebinning
                bkg_model = self._apply_spatial_filter(bkg_model, new_axes, smoothing_fwhm, l_cut)
    
        return Histogram(new_axes, contents=bkg_model, copy_contents=False)
