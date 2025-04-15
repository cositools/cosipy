import os
import numpy as np
import astropy.units as u
import astropy.io.fits as fits
import logging
logger = logging.getLogger(__name__)

from histpy import Histogram

from .RichardsonLucySimple import RichardsonLucySimple

class RichardsonLucy(RichardsonLucySimple):
    """
    A class for the RichardsonLucy algorithm. 
    The algorithm here is based on Knoedlseder+99, Knoedlseder+05, Siegert+20.
    
    An example of parameter is as follows.

    iteration_max: 100
    minimum_flux:
        value: 0.0
        unit: "cm-2 s-1 sr-1"
    acceleration:
        activate: True
        alpha_max: 10.0
    response_weighting:
        activate: True
        index: 0.5
    smoothing:
        activate: True
        FWHM:
            value: 2.0
            unit: "deg"
    stopping_criteria:
        statistics: "log-likelihood"
        threshold: 1e-2
    background_normalization_optimization:
        activate: True
        range: {"albedo": [0.9, 1.1]}
    save_results:
        activate: True
        directory: "/results"
        only_final_result: True
    """

    def __init__(self, initial_model, dataset, mask, parameter):

        super().__init__(initial_model, dataset, mask, parameter)

        # acceleration
        self.do_acceleration = parameter.get('acceleration:activate', False)
        if self.do_acceleration == True:
            self.alpha_max = parameter.get('acceleration:alpha_max', 1.0)

        # smoothing
        self.do_smoothing = parameter.get('smoothing:activate', False)
        if self.do_smoothing:
            self.smoothing_fwhm = parameter.get('smoothing:FWHM:value') * u.Unit(parameter.get('smoothing:FWHM:unit'))
            logger.info(f"Gaussian filter with FWHM of {self.smoothing_fwhm} will be applied to delta images ...")

        # stopping criteria
        self.stopping_criteria_statistics = parameter.get('stopping_criteria:statistics', "log-likelihood")
        self.stopping_criteria_threshold  = parameter.get('stopping_criteria:threshold', 1e-2)

        if not self.stopping_criteria_statistics in ["log-likelihood"]:
            raise ValueError

    def initialization(self):
        """
        initialization before running the image deconvolution
        """

        super().initialization()

        # expected count histograms
        self.expectation_list = self.calc_expectation_list(model = self.initial_model, dict_bkg_norm = self.dict_bkg_norm)
        logger.info("The expected count histograms were calculated with the initial model map.")

    def pre_processing(self):
        """
        pre-processing for each iteration
        """
        pass

    def Estep(self):
        """
        E-step (but it will be skipped).
        Note that self.expectation_list is updated in self.post_processing().
        """
        pass

    def Mstep(self):
        """
        M-step in RL algorithm.
        """
        super().Mstep()

    def post_processing(self):
        """
        Here three processes will be performed.
        - response weighting filter: the delta map is renormalized as pixels with large exposure times will have more feedback.
        - gaussian smoothing filter: the delta map is blurred with a Gaussian function.
        - acceleration of RL algirithm: the normalization of delta map is increased as long as the updated image has no non-negative components.
        """

        self.processed_delta_model = self.delta_model.copy()

        if self.do_response_weighting:
            self.processed_delta_model *= self.response_weighting_filter

        if self.do_smoothing:
            self.processed_delta_model = self.processed_delta_model.smoothing(fwhm = self.smoothing_fwhm)
        
        if self.do_acceleration:
            self.alpha = self.calc_alpha(self.processed_delta_model, self.model)
        else:
            self.alpha = 1.0

        self.model += self.processed_delta_model * self.alpha
        self.model[:] = np.where(self.model.contents < self.minimum_flux, self.minimum_flux, self.model.contents)

        if self.mask is not None:
            self.model = self.model.mask_pixels(self.mask)
        
        # update expectation_list
        self.expectation_list = self.calc_expectation_list(self.model, dict_bkg_norm = self.dict_bkg_norm)
        logger.debug("The expected count histograms were updated with the new model map.")

        # update log_likelihood_list
        self.log_likelihood_list = self.calc_log_likelihood_list(self.expectation_list)
        logger.debug("The log-likelihood list was updated with the new expected count histograms.")

    def register_result(self):
        """
        The values below are stored at the end of each iteration.
        - iteration: iteration number
        - model: updated image
        - delta_model: delta map after M-step 
        - processed_delta_model: delta map after post-processing
        - alpha: acceleration parameter in RL algirithm
        - background_normalization: optimized background normalization
        - log-likelihood: log-likelihood
        """
        
        this_result = {"iteration": self.iteration_count, 
                       "model": self.model.copy(),
                       "delta_model": self.delta_model,
                       "processed_delta_model": self.processed_delta_model,
                       "background_normalization": self.dict_bkg_norm.copy(),
                       "alpha": self.alpha, 
                       "log-likelihood": self.log_likelihood_list}

        # show intermediate results
        logger.info(f'  alpha: {this_result["alpha"]}')
        logger.info(f'  background_normalization: {this_result["background_normalization"]}')
        logger.info(f'  log-likelihood: {this_result["log-likelihood"]}')
        
        # register this_result in self.results
        self.results.append(this_result)

    def check_stopping_criteria(self):
        """
        If iteration_count is smaller than iteration_max, the iterative process will continue.

        Returns
        -------
        bool
        """
        if self.iteration_count == 1:
            return False
        elif self.iteration_count == self.iteration_max:
            return True

        if self.stopping_criteria_statistics == "log-likelihood":

            log_likelihood = np.sum(self.results[-1]["log-likelihood"])
            log_likelihood_before = np.sum(self.results[-2]["log-likelihood"])

            logger.debug(f'Delta log-likelihood: {log_likelihood - log_likelihood_before}')

            if log_likelihood - log_likelihood_before < 0:

                logger.warning("The likelihood is not increased in this iteration. The image reconstruction may be unstable.")
                return False

            elif log_likelihood - log_likelihood_before < self.stopping_criteria_threshold:
                return True

        return False

    def finalization(self):
        """
        finalization after running the image deconvolution
        """
        if self.save_results == True:
            logger.info(f'Saving results in {self.save_results_directory}')

            counter_name = "iteration"

            # model
            histkey_filename = [("model", f"{self.save_results_directory}/model.hdf5"),
                                ("delta_model", f"{self.save_results_directory}/delta_model.hdf5"),
                                ("processed_delta_model", f"{self.save_results_directory}/processed_delta_model.hdf5")]

            for key, filename in histkey_filename:

                self.save_histogram(filename = filename,
                                    counter_name = counter_name,
                                    histogram_key = key,
                                    only_final_result = self.save_only_final_result)

            #fits
            fits_filename = f'{self.save_results_directory}/results.fits'

            self.save_results_as_fits(filename = fits_filename,
                                      counter_name = counter_name,
                                      values_key_name_format = [("alpha", "ALPHA", "D")],
                                      dicts_key_name_format = [("background_normalization", "BKG_NORM", "D")],
                                      lists_key_name_format = [("log-likelihood", "LOG-LIKELIHOOD", "D")])

    def calc_alpha(self, delta_model, model):
        """
        Calculate the acceleration parameter in RL algorithm.

        Returns
        -------
        float
            Acceleration parameter
        """
        diff = -1 * (model / delta_model).contents

        diff[(diff <= 0) | (delta_model.contents == 0)] = np.inf

        if self.mask is not None:
            diff[np.invert(self.mask.contents)] = np.inf

        alpha = min(np.min(diff), self.alpha_max)

        if alpha < 1.0:
            alpha = 1.0

        return alpha
