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
    background_normalization_optimization:
        activate: True
        range: {"albedo": [0.9, 1.1]}
    response_weighting:
        activate: True 
        index: 0.5
    save_results: 
        activate: True
        directory: "/results"
        only_final_result: True
    """

    def __init__(self, initial_model, dataset, mask, parameter):

        super().__init__(initial_model, dataset, mask, parameter)

        # background normalization optimization
        self.do_bkg_norm_optimization = parameter.get('background_normalization_optimization:activate', False)
        if self.do_bkg_norm_optimization:
            self.dict_bkg_norm_range = parameter.get('background_normalization_optimization:range', {key: [0.0, 100.0] for key in self.dict_bkg_norm.keys()})

        # response_weighting
        self.do_response_weighting = parameter.get('response_weighting:activate', False)
        if self.do_response_weighting:
            self.response_weighting_index = parameter.get('response_weighting:index', 0.5)

        # saving results
        self.save_results = parameter.get('save_results:activate', False)
        self.save_results_directory = parameter.get('save_results:directory', './results')
        self.save_only_final_result = parameter.get('save_results:only_final_result', False)

        if self.save_results is True:
            if os.path.isdir(self.save_results_directory):
                logger.warning(f"A directory {self.save_results_directory} already exists. Files in {self.save_results_directory} may be overwritten. Make sure that is not a problem.")
            else:
                os.makedirs(self.save_results_directory)

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
        
        # masking
        if self.mask is not None:
            self.delta_model = self.delta_model.mask_pixels(self.mask)
        
        # background normalization optimization
        if self.do_bkg_norm_optimization:
            for key in self.dict_bkg_norm.keys():

                sum_bkg_T_product = self.calc_summed_bkg_model_product(key, ratio_list)
                sum_bkg_model = self.dict_summed_bkg_model[key]
                bkg_norm = self.dict_bkg_norm[key] * (sum_bkg_T_product / sum_bkg_model)

                bkg_range = self.dict_bkg_norm_range[key]
                if bkg_norm < bkg_range[0]:
                    bkg_norm = bkg_range[0]
                elif bkg_norm > bkg_range[1]:
                    bkg_norm = bkg_range[1]

                self.dict_bkg_norm[key] = bkg_norm

    def post_processing(self):
        """
        Post-processing. 
        """
        
        # response_weighting
        if self.do_response_weighting:
            self.model[:] += self.delta_model.contents * self.response_weighting_filter
        else:
            self.model[:] += self.delta_model.contents

        self.model[:] = np.where(self.model.contents < self.minimum_flux, self.minimum_flux, self.model.contents)
        
        # masking again
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

        if self.save_results == True:
            logger.info('Saving results in {self.save_results_directory}')

            counter_name = "iteration"
               
            # model
            histkey_filename = [("model", f"{self.save_results_directory}/model.hdf5"), 
                                ("delta_model", f"{self.save_results_directory}/delta_model.hdf5")]

            for key, filename in histkey_filename:

                self.save_histogram(filename = filename, 
                                    counter_name = counter_name,
                                    histogram_key = key,
                                    only_final_result = self.save_only_final_result)
            
            #fits
            fits_filename = f'{self.save_results_directory}/results.fits'

            self.save_results_as_fits(filename = fits_filename,
                                      counter_name = counter_name,
                                      values_key_name_format = [],
                                      dicts_key_name_format = [("background_normalization", "BKG_NORM", "D")],
                                      lists_key_name_format = [])
