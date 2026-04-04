import numpy as np
import astropy.units as u
import logging
logger = logging.getLogger(__name__)

from histpy import Histogram

from ..utils import _to_float
from .RichardsonLucy import RichardsonLucy

from .prior_tsv import PriorTSV
from .prior_entropy import PriorEntropy

from .response_weighting_filter import ResponseWeightingFilter 

from ..constants import DEFAULT_STOPPING_THRESHOLD, DEFAULT_RESPONSE_WEIGHTING_INDEX

class MAP_RichardsonLucy(RichardsonLucy):
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
        statistics: "log-likelihood"
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

    prior_classes = {"TSV": PriorTSV, "entropy": PriorEntropy}

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

            this_prior_parameter = parameter['prior'][prior_name]
            self.priors[prior_name] = self.prior_classes[prior_name](this_prior_parameter, initial_model)

        # response_weighting
        self.response_weighting_enabled = parameter.get('response_weighting:activate', False)
        if self.response_weighting_enabled:
            self.response_weighting_index = parameter.get('response_weighting:index', DEFAULT_RESPONSE_WEIGHTING_INDEX)

        # stopping criteria
        self.stopping_criteria_statistics = parameter.get('stopping_criteria:statistics', "log-posterior")
        self.stopping_criteria_threshold  = parameter.get('stopping_criteria:threshold', DEFAULT_STOPPING_THRESHOLD)

        if not self.stopping_criteria_statistics in ["log-likelihood", "log-posterior"]:
            raise ValueError

        # update parameter summary
        self._parameter_summary += [
            ("prior", parameter['prior']),
        ]

        self._parameter_summary += [
            ("response_weighting_enabled", self.response_weighting_enabled),
        ]
        if self.response_weighting_enabled:
            self._parameter_summary += [
                ("response_weighting_index", self.response_weighting_index),
            ]

        self._parameter_summary += [
            ("stopping_criteria_statistics", self.stopping_criteria_statistics),
            ("stopping_criteria_threshold", self.stopping_criteria_threshold),
        ]

    def load_gamma_prior(self, parameter):

        if parameter is None:
            self.prior_gamma_model_theta, self.prior_gamma_model_k = np.inf * self.initial_model.unit, 1.0 #flat distribution
            self.prior_gamma_bkg_theta, self.prior_gamma_bkg_k = np.inf, 1.0 #flat distribution
        else:
            self.prior_gamma_model_theta = parameter['model']['theta']['value'] * u.Unit(parameter['model']['theta']['unit'])
            self.prior_gamma_model_k     = parameter['model']['k']['value']

            self.prior_gamma_bkg_theta = parameter['background']['theta']['value']
            self.prior_gamma_bkg_k     = parameter['background']['k']['value']

    def log_gamma_prior(self, model, dict_bkg_norm):

        eps = np.finfo(model.contents.dtype).eps
        
        # model
        pl_part_model = (self.prior_gamma_model_k - 1.0) * np.sum( np.log(model.contents + eps) ) if model.unit is None else \
                        (self.prior_gamma_model_k - 1.0) * np.sum( np.log(model.contents.value + eps) )

        log_part_model = - np.sum( model.contents / self.prior_gamma_model_theta )

        # background
        pl_part_bkg, log_part_bkg = 0, 0

        if self.bkg_norm_optimization_enabled:
            for key in dict_bkg_norm.keys():
                
                bkg_norm = dict_bkg_norm[key]

                pl_part_bkg += (self.prior_gamma_bkg_k - 1.0) * np.log(bkg_norm)

                log_part_bkg += -1.0 * np.sum( bkg_norm / self.prior_gamma_bkg_theta )

        return _to_float(pl_part_model + log_part_model), _to_float(pl_part_bkg + log_part_bkg)

    def initialization(self):
        """
        Initialize before running image deconvolution.
        
        This method sets up response weighting filter based on the exposure map.
        """

        super().initialization()

        # response-weighting filter
        if self.response_weighting_enabled:
            self.response_weighting_filter = ResponseWeightingFilter(self.summed_exposure_map, self.response_weighting_index)

    def pre_processing(self):
        """
        pre-processing for each iteration
        """

        if self.iteration_count == 1:
            self.Estep()
            logger.info("The expected count histograms were calculated with the initial model map.")

    def processing_core(self):
        """
        Core processing for each iteration.
        """

        # Note that Estep() is performed in self.post_processing().
        self.Mstep()

    def Mstep(self):
        """
        M-step in RL algorithm.
        """

        ratio_list = [ data.event / expectation for data, expectation in zip(self.dataset, self.expectation_list) ]
        
        # model update (EM part)
        sum_T_product = self.dataset.calc_summed_T_product(ratio_list)
        model_EM = (self.model * sum_T_product + self.prior_gamma_model_k - 1.0) \
                    / (self.summed_exposure_map + 1.0 / self.prior_gamma_model_theta)
        model_EM[:] = np.where( model_EM.contents < self.minimum_flux, self.minimum_flux, model_EM.contents) 

        # model update (prior part)
        sum_grad_log_prior = np.zeros_like(self.summed_exposure_map)

        for key in self.priors.keys():
            sum_grad_log_prior += self.priors[key].grad_log_prior(model_EM)

        self.prior_filter = Histogram(self.model.axes, contents = np.exp( sum_grad_log_prior / (self.summed_exposure_map.contents + 1.0 / self.prior_gamma_model_theta)))

        self.delta_model = self.prior_filter * model_EM.contents - self.model

        # background normalization optimization
        if self.bkg_norm_optimization_enabled:
            for key in self.dict_bkg_norm.keys():

                sum_bkg_T_product = self.dataset.calc_summed_bkg_model_product(key, ratio_list)
                sum_bkg_model = self.dict_summed_bkg_model[key]
                bkg_norm = (self.dict_bkg_norm[key] * sum_bkg_T_product + self.prior_gamma_bkg_k - 1.0) \
                            / (sum_bkg_model + 1.0 / self.prior_gamma_bkg_theta)

                self.dict_delta_bkg_norm[key] = bkg_norm - self.dict_bkg_norm[key]

        # apply response_weighting_filter
        if self.response_weighting_enabled:
            self.delta_model = self.response_weighting_filter.apply(self.delta_model)

    def post_processing(self):
        """
        Here three processes will be performed.
        - response weighting filter: the delta map is renormalized as pixels with large exposure times will have more feedback.
        """

        # update model
        self.model += self.delta_model
        self._ensure_model_constraints()

        # update background normalization
        if self.bkg_norm_optimization_enabled:
            for key in self.dict_bkg_norm.keys():
                self.dict_bkg_norm[key] += self.dict_delta_bkg_norm[key]
            self._ensure_bkg_norm_range()

        # update expectation_list
        self.Estep()
        logger.debug("The expected count histograms were updated with the new model map.")

        # update log_likelihood_list
        self.log_likelihood_list = self.dataset.calc_log_likelihood_list(self.expectation_list)
        logger.debug("The log-likelihood list was updated with the new expected count histograms.")

        # update log priors
        self.log_priors = {}

        self.log_priors['gamma_model'], self.log_priors['gamma_bkg'] = self.log_gamma_prior(self.model, self.dict_bkg_norm)

        for key in self.priors.keys():
            self.log_priors[key] = _to_float(self.priors[key].log_prior(self.model))

        # log-posterior
        self.log_posterior = np.sum(self.log_likelihood_list) + np.sum([self.log_priors[key] for key in self.log_priors.keys()])

    def register_result(self):
        """
        The values below are stored at the end of each iteration.
        - iteration: iteration number
        - model: updated image
        - prior_filter: prior filter
        - background_normalization: optimized background normalization
        - log-likelihood: log-likelihood
        - log-prior: log-prior
        - log-posterior: log-posterior
        """
        
        this_result = {"iteration": self.iteration_count, 
                       "model": self.model.copy(), 
                       "prior_filter": self.prior_filter.copy(),
                       "background_normalization": self.dict_bkg_norm.copy(),
                       "log-likelihood": self.log_likelihood_list.copy(),
                       "log-prior": self.log_priors.copy(),
                       "log-posterior": self.log_posterior,
                       }

        # show intermediate results
        for key in ["background_normalization", "log-likelihood", "log-prior", "log-posterior"]:
            logger.info(f"{key}: {this_result[key]}")
        
        # register this_result in self.results
        self.results.append(this_result)

    def check_stopping_criteria(self):
        """
        If iteration_count is smaller than iteration_max, the iterative process will continue.

        Returns
        -------
        bool
        """

        if self.iteration_count >= self.iteration_max:
            return True

        if self.iteration_count == 1:
            return False  # need at least 2 results to compute delta

        if self.stopping_criteria_statistics == "log-likelihood":

            log_likelihood = np.sum(self.results[-1]["log-likelihood"])
            log_likelihood_before = np.sum(self.results[-2]["log-likelihood"])

            logger.debug(f'Delta log-likelihood: {log_likelihood - log_likelihood_before}')

            if log_likelihood - log_likelihood_before < 0:

                logger.warning("The likelihood was not increased in this iteration. The image reconstruction may be unstable.")
                return False 
            
            elif log_likelihood - log_likelihood_before < self.stopping_criteria_threshold:
                return True

        elif self.stopping_criteria_statistics == "log-posterior":
            
            log_posterior = self.results[-1]["log-posterior"]
            log_posterior_before = self.results[-2]["log-posterior"]

            logger.debug(f'Delta log-posterior: {log_posterior - log_posterior_before}')

            if log_posterior - log_posterior_before < 0:

                logger.warning("The posterior was not increased in this iteration. The image reconstruction may be unstable.")
                return False 
            
            elif log_posterior - log_posterior_before < self.stopping_criteria_threshold:
                return True

        return False

    def finalization(self):
        """
        finalization after running the image deconvolution
        """

        if not self.save_results:
            return

        logger.info(f"Saving results in {self.save_results_directory}")

        self._save_standard_results(
            counter_name           = "iteration",
            histogram_keys         = [("model", f"{self.save_results_directory}/model.hdf5", self.save_only_final_result),
                                      ("prior_filter", f"{self.save_results_directory}/prior_filter.hdf5", self.save_only_final_result)],
            fits_filename          = f"{self.save_results_directory}/results.fits",
            values_key_name_format = [("log-posterior", "LOG-POSTERIOR", "D")],
            dicts_key_name_format  = [("background_normalization", "BKG_NORM", "D"), ("log-prior", "LOG-PRIOR", "D")],
            lists_key_name_format  = [("log-likelihood", "LOG-LIKELIHOOD", "D")],
        )
