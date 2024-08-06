import os
import copy
import numpy as np
import logging
logger = logging.getLogger(__name__)

from histpy import Histogram

from .deconvolution_algorithm_base import DeconvolutionAlgorithmBase

class RichardsonLucySimple(DeconvolutionAlgorithmBase):
    """
    A class for the original RichardsonLucy algorithm. 
    Basically, this class can be used for testing codes.
    
    An example of parameter is as follows.

    iteration_max: 100
    minimum_flux:
        value: 0.0
        unit: "cm-2 s-1 sr-1"
    background_normalization_optimization: True 
    """

    def __init__(self, initial_model, dataset, mask, parameter):

        DeconvolutionAlgorithmBase.__init__(self, initial_model, dataset, mask, parameter)

        self.do_bkg_norm_optimization = parameter.get('background_normalization_optimization', False)

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
        E-step in RL algoritm
        """

        self.expectation_list = self.calc_expectation_list(self.model, dict_bkg_norm = self.dict_bkg_norm)
        logger.debug("The expected count histograms were updated with the new model map.")

    def Mstep(self):
        """
        M-step in RL algorithm.
        """

        ratio_list = [ data.event / expectation for data, expectation in zip(self.dataset, self.expectation_list) ]
        
        # delta model
        sum_T_product = self.calc_summed_T_product(ratio_list)
        self.delta_model = self.model * (sum_T_product/self.summed_exposure_map - 1)
        
        # background normalization optimization
        if self.do_bkg_norm_optimization:
            for key in self.dict_bkg_norm.keys():

                sum_bkg_T_product = self.calc_summed_bkg_model_product(key, ratio_list)
                sum_bkg_model = self.dict_summed_bkg_model[key]

                self.dict_bkg_norm[key] = self.dict_bkg_norm[key] * (sum_bkg_T_product / sum_bkg_model)

    def post_processing(self):
        """
        Post-processing. 
        """
        self.model += self.delta_model
        self.model[:] = np.where(self.model.contents < self.minimum_flux, self.minimum_flux, self.model.contents)

        if self.mask is not None:
            self.model = self.model.mask_pixels(self.mask)

    def register_result(self):
        """
        Register results at the end of each iteration. 
        """
        
        this_result = {"iteration": self.iteration_count, 
                       "model": copy.deepcopy(self.model), 
                       "delta_model": copy.deepcopy(self.delta_model),
                       "background_normalization": copy.deepcopy(self.dict_bkg_norm)}

        # show intermediate results
        logger.info(f'  background_normalization: {this_result["background_normalization"]}')
        
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
        pass
