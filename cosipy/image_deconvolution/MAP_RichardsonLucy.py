import os
import copy
import numpy as np
import astropy.units as u
import astropy.io.fits as fits
import logging
logger = logging.getLogger(__name__)

from histpy import Histogram

from .deconvolution_algorithm_base import DeconvolutionAlgorithmBase

from .prior_tsv import PriorTSV

class MAP_RichardsonLucy(DeconvolutionAlgorithmBase):
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

        DeconvolutionAlgorithmBase.__init__(self, initial_model, dataset, mask, parameter)
        
        # response weighting filter
        self.do_response_weighting = parameter.get('response_weighting:activate', False)
        if self.do_response_weighting:
            self.response_weighting_index = parameter.get('response_weighting:index', 0.5)

        # background normalization optimization
        self.do_bkg_norm_optimization = parameter.get('background_normalization_optimization:activate', False)
        if self.do_bkg_norm_optimization:
            self.dict_bkg_norm_range = parameter.get('background_normalization_optimization:range', {key: [0.0, 100.0] for key in self.dict_bkg_norm.keys()})

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
        
        # saving results
        self.save_results = parameter.get('save_results:activate', False)
        self.save_results_directory = parameter.get('save_results:directory', './results')
        self.save_results_only_final = parameter.get('save_results:only_final_result', False)

        if self.save_results is True:
            if os.path.isdir(self.save_results_directory):
                logger.warning(f"A directory {self.save_results_directory} already exists. Files in {self.save_results_directory} may be overwritten. Make sure that is not a problem.")
            else:
                os.makedirs(self.save_results_directory)

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
        # clear counter 
        self.iteration_count = 0

        # clear results
        self.results.clear()

        # copy model
        self.model = copy.deepcopy(self.initial_model)

        # calculate exposure map
        self.summed_exposure_map = self.calc_summed_exposure_map()

        # mask setting
        if self.mask is None and np.any(self.summed_exposure_map.contents == 0):
            self.mask = Histogram(self.model.axes, contents = self.summed_exposure_map.contents > 0)
            self.model = self.model.mask_pixels(self.mask)
            logger.info("There are zero-exposure pixels. A mask to ignore them was set.")

        # response-weighting filter
        if self.do_response_weighting:
            self.response_weighting_filter = (self.summed_exposure_map.contents / np.max(self.summed_exposure_map.contents))**self.response_weighting_index
            logger.info("The response weighting filter was calculated.")

        # expected count histograms
        self.expectation_list = self.calc_expectation_list(model = self.initial_model, dict_bkg_norm = self.dict_bkg_norm)
        logger.info("The expected count histograms were calculated with the initial model map.")

        # calculate summed background models for M-step
        if self.do_bkg_norm_optimization:
            self.dict_summed_bkg_model = {}
            for key in self.dict_bkg_norm.keys():
                self.dict_summed_bkg_model[key] = self.calc_summed_bkg_model(key)

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

        self.prior_filter = Histogram(self.model.axes, contents = np.exp(-1.0 * sum_grad_log_prior / self.summed_exposure_map.contents) )

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
            self.log_priors[key] = -1.0 * self.priors[key].log_prior(self.model)
            # NOTE: the log_prior is defined as the negative of log of prior function

        # posterior
        self.posterior = np.sum(self.loglikelihood_list) + np.sum([self.log_priors[key] for key in self.log_priors.keys()])

    def register_result(self):
        """
        The values below are stored at the end of each iteration.
        - iteration: iteration number
        - model: updated image
        - prior_filter: prior filter
        - background_normalization: optimized background normalization
        - loglikelihood: log-likelihood
        - log_prior: log-prior
        - posterior: posterior
        """
        
        this_result = {"iteration": self.iteration_count, 
                       "model": copy.deepcopy(self.model), 
                       "prior_filter": copy.deepcopy(self.prior_filter),
                       "background_normalization": copy.deepcopy(self.dict_bkg_norm),
                       "loglikelihood": copy.deepcopy(self.loglikelihood_list),
                       "log_prior": copy.deepcopy(self.log_priors),
                       "posterior": copy.deepcopy(self.posterior),
                       }

        # show intermediate results
        logger.info(f'  background_normalization: {this_result["background_normalization"]}')
        logger.info(f'  loglikelihood: {this_result["loglikelihood"]}')
        logger.info(f'  log_prior: {this_result["log_prior"]}')
        logger.info(f'  posterior: {this_result["posterior"]}')
        
        # register this_result in self.results
        self.results.append(this_result)

    def check_stopping_criteria(self):
        """
        If iteration_count is smaller than iteration_max, the iterative process will continue.

        Returns
        -------
        bool
        """
        if self.iteration_count < self.iteration_max:
            return False
        return True

    def finalization(self):
        """
        finalization after running the image deconvolution
        """
        if self.save_results == True:
            logger.info('Saving results in {self.save_results_directory}')

            # model
            self.results[-1]["model"].write(f"{self.save_results_directory}/model.hdf5", name = 'result', overwrite = True)

            self.results[-1]["prior_filter"].write(f"{self.save_results_directory}/prior_filter.hdf5", name = 'result', overwrite = True)

            for this_result in self.results:

                iteration_count = this_result["iteration"]

                this_result["model"].write(f"{self.save_results_directory}/model.hdf5", name = f'iteration{iteration_count}', overwrite = True)

                this_result["prior_filter"].write(f"{self.save_results_directory}/prior_filter.hdf5", name = f'iteration{iteration_count}', overwrite = True)

            #fits
            primary_hdu = fits.PrimaryHDU()

            col_iteration = fits.Column(name='iteration', array=[float(result['iteration']) for result in self.results], format='K')
            cols_bkg_norm = [fits.Column(name=key, array=[float(result['background_normalization'][key]) for result in self.results], format='D') 
                             for key in self.dict_bkg_norm.keys()]
            cols_loglikelihood = [fits.Column(name=f"{self.dataset[i].name}", array=[float(result['loglikelihood'][i]) for result in self.results], format='D') 
                                  for i in range(len(self.dataset))]
            cols_log_prior = [fits.Column(name=key, array=[float(result['log_prior'][key]) for result in self.results], format='D') 
                              for key in self.results[0]['log_prior'].keys()]
            cols_posterior = fits.Column(name = 'posterior', array=[float(result['posterior']) for result in self.results], format='D')

            table_bkg_norm = fits.BinTableHDU.from_columns([col_iteration] + cols_bkg_norm)
            table_bkg_norm.name = "bkg_norm"

            table_loglikelihood = fits.BinTableHDU.from_columns([col_iteration] + cols_loglikelihood)
            table_loglikelihood.name = "loglikelihood"

            table_log_prior = fits.BinTableHDU.from_columns([col_iteration] + cols_log_prior)
            table_log_prior.name = "log_prior"

            table_posterior = fits.BinTableHDU.from_columns([col_iteration, cols_posterior])
            table_posterior.name = "posterior"

            hdul = fits.HDUList([primary_hdu, table_bkg_norm, table_loglikelihood, table_log_prior, table_posterior])
            hdul.writeto(f'{self.save_results_directory}/results.fits',  overwrite=True)
