import os
import copy
import numpy as np
import astropy.units as u
import astropy.io.fits as fits
import logging
logger = logging.getLogger(__name__)

from histpy import Histogram

from .RichardsonLucySimple import RichardsonLucySimple

from .prior_tsv import PriorTSV

class MAP_RichardsonLucy(RichardsonLucySimple):
    """
    A class for the RichardsonLucy algorithm using prior distributions. 
    
    An example of parameter is as follows.

    iteration_max: 100
    minimum_flux:
        value: 0.0
        unit: "cm-2 s-1 sr-1"
    response_weighting:
        activate: True
        index: 0.5
    background_normalization_optimization:
        activate: True
        range: {"albedo": [0.01, 10.0]}
    save_results:
        activate: True
        directory: "./results"
        only_final_result: True
    stopping_criteria:
        statistics: "loglikelihood"
        threshold: 1e-2
    prior:
      TSV:
        coefficient: 1.e+6
      gamma:
        model:
          theta:
            value: .inf
            unit: "cm-2 s-1 sr-1"
          k:
            value: 0.9
        background:
          theta:
            value: .inf
          k:
            value: 1.0
    """

    prior_classes = {"TSV": PriorTSV}

    def __init__(self, initial_model, dataset, mask, parameter):

        super().__init__(initial_model, dataset, mask, parameter)
        
        # Prior distribution 
        self.prior_key_list = list(parameter.get('prior', {}).keys())
        self.priors = {}

        ## Gamma distribution
        if 'gamma' in self.prior_key_list:
            this_prior_parameter = parameter['prior']['gamma']
            self.load_gamma_prior(this_prior_parameter)
        else:
            self.load_gamma_prior(None)

        ## other priors
        for prior_name in self.prior_key_list:
            if prior_name == 'gamma':
                continue

            coefficient = parameter['prior'][prior_name]['coefficient']
            self.priors[prior_name] = self.prior_classes[prior_name](coefficient, initial_model)

        # stopping criteria
        self.stopping_criteria_statistics = parameter.get('stopping_criteria:statistics', "log-posterior")
        self.stopping_criteria_threshold  = parameter.get('stopping_criteria:threshold', 1e-2)

        if not self.stopping_criteria_statistics in ["loglikelihood", "log-posterior"]:
            raise ValueError

    def load_gamma_prior(self, parameter):

        if parameter is None:
            self.prior_gamma_model_theta, self.prior_gamma_model_k = np.inf * self.initial_model.unit, 1.0 #flat distribution
            self.prior_gamma_bkg_theta, self.prior_gamma_bkg_k = np.inf, 1.0 #flat distribution
        else:
            self.prior_gamma_model_theta = parameter['model']['theta']['value'] * u.Unit(parameter['model']['theta']['unit'])
            self.prior_gamma_model_k     = parameter['model']['k']['value']

            self.prior_gamma_bkg_theta = parameter['background']['theta']['value']
            self.prior_gamma_bkg_k     = parameter['background']['k']['value']

    def log_gamma_prior(self, model):

        eps = np.finfo(model.contents.dtype).eps
        
        # model
        pl_part_model = (self.prior_gamma_model_k - 1.0) * np.sum( np.log(model.contents + eps) ) if model.unit is None else \
                        (self.prior_gamma_model_k - 1.0) * np.sum( np.log(model.contents.value + eps) )

        log_part_model = - np.sum( model.contents / self.prior_gamma_model_theta )

        # background
        pl_part_bkg, log_part_bkg = 0, 0

        if self.do_bkg_norm_optimization:
            for key in self.dict_bkg_norm.keys():
                
                bkg_norm = self.dict_bkg_norm[key]

                pl_part_bkg += (self.prior_gamma_bkg_k - 1.0) * np.log(bkg_norm)

                log_part_bkg += -1.0 * np.sum( bkg_norm / self.prior_gamma_bkg_theta )

        return pl_part_model + log_part_model, pl_part_bkg + log_part_bkg

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

        ratio_list = [ data.event / expectation for data, expectation in zip(self.dataset, self.expectation_list) ]
        
        # model update (EM part)
        sum_T_product = self.calc_summed_T_product(ratio_list)
        model_EM = (self.model * sum_T_product + self.prior_gamma_model_k - 1.0) \
                    / (self.summed_exposure_map + 1.0 / self.prior_gamma_model_theta)
        model_EM[:] = np.where( model_EM.contents < self.minimum_flux, self.minimum_flux, model_EM.contents) 

        # model update (prior part)
        sum_grad_log_prior = np.zeros_like(self.summed_exposure_map)

        for key in self.priors.keys():
            sum_grad_log_prior += self.priors[key].grad_log_prior(model_EM)

        self.prior_filter = Histogram(self.model.axes, contents = np.exp( sum_grad_log_prior / self.summed_exposure_map.contents ) )

        self.model[:] = self.prior_filter.contents * model_EM.contents

        # applying response_weighting_filter
        if self.do_response_weighting:
            if self.iteration_count == 1:
                delta_model = self.model - self.initial_model
            else:
                delta_model = self.model - self.results[-1]['model']

            self.model[:] = (self.model.contents - delta_model.contents) + self.response_weighting_filter * delta_model.contents

        # masking
        if self.mask is not None:
            self.model = self.model.mask_pixels(self.mask)

        self.model[:] = np.where( self.model.contents < self.minimum_flux, self.minimum_flux, self.model.contents) 
        
        # background normalization optimization
        if self.do_bkg_norm_optimization:
            for key in self.dict_bkg_norm.keys():

                sum_bkg_T_product = self.calc_summed_bkg_model_product(key, ratio_list)
                sum_bkg_model = self.dict_summed_bkg_model[key]
                bkg_norm = (self.dict_bkg_norm[key] * sum_bkg_T_product + self.prior_gamma_bkg_k - 1.0) \
                            / (sum_bkg_model + 1.0 / self.prior_gamma_bkg_theta)

                bkg_range = self.dict_bkg_norm_range[key]
                if bkg_norm < bkg_range[0]:
                    bkg_norm = bkg_range[0]
                elif bkg_norm > bkg_range[1]:
                    bkg_norm = bkg_range[1]

                self.dict_bkg_norm[key] = bkg_norm

    def post_processing(self):
        """
        Here three processes will be performed.
        - response weighting filter: the delta map is renormalized as pixels with large exposure times will have more feedback.
        """

        #TODO: add acceleration SQUAREM

        # update expectation_list
        self.expectation_list = self.calc_expectation_list(self.model, dict_bkg_norm = self.dict_bkg_norm)
        logger.debug("The expected count histograms were updated with the new model map.")

        # update loglikelihood_list
        self.loglikelihood_list = self.calc_loglikelihood_list(self.expectation_list)
        logger.debug("The loglikelihood list was updated with the new expected count histograms.")

        # update log priors
        self.log_priors = {}

        self.log_priors['gamma_model'], self.log_priors['gamma_bkg'] = self.log_gamma_prior(self.model)

        for key in self.priors.keys():
            self.log_priors[key] = self.priors[key].log_prior(self.model)

        # log-posterior
        self.log_posterior = np.sum(self.loglikelihood_list) + np.sum([self.log_priors[key] for key in self.log_priors.keys()])

    def register_result(self):
        """
        The values below are stored at the end of each iteration.
        - iteration: iteration number
        - model: updated image
        - prior_filter: prior filter
        - background_normalization: optimized background normalization
        - loglikelihood: log-likelihood
        - log-prior: log-prior
        - log-posterior: log-posterior
        """
        
        this_result = {"iteration": self.iteration_count, 
                       "model": copy.deepcopy(self.model), 
                       "prior_filter": copy.deepcopy(self.prior_filter),
                       "background_normalization": copy.deepcopy(self.dict_bkg_norm),
                       "loglikelihood": copy.deepcopy(self.loglikelihood_list),
                       "log-prior": copy.deepcopy(self.log_priors),
                       "log-posterior": copy.deepcopy(self.log_posterior),
                       }

        # show intermediate results
        logger.info(f'  background_normalization: {this_result["background_normalization"]}')
        logger.info(f'  loglikelihood: {this_result["loglikelihood"]}')
        logger.info(f'  log-prior: {this_result["log_prior"]}')
        logger.info(f'  log-posterior: {this_result["log_posterior"]}')
        
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

        if self.stopping_criteria_statistics == "loglikelihood":

            loglikelihood = np.sum(self.results[-1]["loglikelihood"])
            loglikelihood_before = np.sum(self.results[-2]["loglikelihood"])
            
            if loglikelihood - loglikelihood_before < self.stopping_criteria_threshold:
                return True

        elif self.stopping_criteria_statistics == "log-posterior":
            
            log_posterior = self.results[-1]["log-posterior"]
            log_posterior_before = self.results[-2]["log-posterior"]

            if log_posterior - log_posterior_before < self.stopping_criteria_threshold:
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
                                ("prior_filter", f"{self.save_results_directory}/prior_filter.hdf5")]

            for key, filename in histkey_filename:

                self.save_histogram(filename = filename, 
                                    counter_name = counter_name,
                                    histogram_key = key,
                                    only_final_result = self.save_only_final_result)
            
            #fits
            fits_filename = f'{self.save_results_directory}/results.fits'

            self.save_results_as_fits(filename = fits_filename,
                                      counter_name = counter_name,
                                      values_key_name_format = [("log-posterior", "LOG-POSTERIOR", "D")],
                                      dicts_key_name_format  = [("background_normalization", "BKG_NORM", "D"), ("log-prior", "LOG-PRIOR", "D")],
                                      lists_key_name_format  = [("loglikelihood", "LOGLIKELIHOOD", "D")])
