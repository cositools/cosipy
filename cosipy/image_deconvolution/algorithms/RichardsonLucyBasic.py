import os
import numpy as np
import astropy.units as u
import logging
logger = logging.getLogger(__name__)

from histpy import Histogram

from .deconvolution_algorithm_base import DeconvolutionAlgorithmBase
from ..constants import DEFAULT_MINIMUM_FLUX

class RichardsonLucyBasic(DeconvolutionAlgorithmBase):
    """
    Basic Richardson-Lucy algorithm.
    
    This class implements the core RL algorithm with minimal parameters.
    It is designed as an entry point to understand the fundamental EM structure of the Richardson-Lucy algorithm.
    
    Features:
    - E-step + M-step
    - Minimum flux constraint
    - Masking
    
    An example of parameter is as follows.

    iteration_max: 100
    minimum_flux:
        value: 0.0
        unit: "cm-2 s-1 sr-1"
    save_results: 
        activate: True
        directory: "./results"
        only_final_result: True
    """

    def __init__(self, initial_model, dataset, mask, parameter):

        super().__init__(initial_model, dataset, mask, parameter)

        # minimum flux
        self.minimum_flux = parameter.get('minimum_flux:value', DEFAULT_MINIMUM_FLUX)

        minimum_flux_unit = parameter.get('minimum_flux:unit', initial_model.unit)
        if minimum_flux_unit is not None:
            self.minimum_flux = self.minimum_flux*u.Unit(minimum_flux_unit)

        # saving results
        self.save_results = parameter.get('save_results:activate', False)
        self.save_results_directory = parameter.get('save_results:directory', './results')
        self.save_only_final_result = parameter.get('save_results:only_final_result', False)

        if self.save_results is True:
            if os.path.isdir(self.save_results_directory):
                logger.warning(f"A directory {self.save_results_directory} already exists. Files in {self.save_results_directory} may be overwritten. Make sure that is not a problem.")
            else:
                os.makedirs(self.save_results_directory)

        # update parameter summary
        self._parameter_summary += [
            ("minimum_flux", self.minimum_flux),
            ("save_results", self.save_results),
        ]
        if self.save_results:
            self._parameter_summary += [
                ("save_results_directory", self.save_results_directory),
                ("save_only_final_result", self.save_only_final_result),
            ]

    def initialization(self):
        """
        initialization before running the image deconvolution
        """
        
        # show parameters
        self._show_parameters()

        # clear counter 
        self.iteration_count = 0

        # clear results
        self.results.clear()

        # copy model
        self.model = self.initial_model.copy()

        # calculate exposure map
        self.summed_exposure_map = self.dataset.calc_summed_exposure_map()

        # mask setting
        if self.mask is None and np.any(self.summed_exposure_map.contents == 0):
            self.mask = Histogram(self.model.axes,
                                  contents = self.summed_exposure_map.contents > 0,
                                  copy_contents = False)
            self.model = self.model.mask_pixels(self.mask)
            logger.info("There are zero-exposure pixels. A mask to ignore them was set.")

    def pre_processing(self):
        """
        pre-processing for each iteration
        """

        pass

    def processing_core(self):
        """
        Core processing for each iteration.
        """

        self.Estep()
        self.Mstep()

    def Estep(self):
        """
        E-step. 
        In this step, self.expectation_list will be updated.
        """

        self.source_expectation_list = self.dataset.calc_source_expectation_list(self.model)
        self.bkg_expectation_list = self.dataset.calc_bkg_expectation_list(self.dict_bkg_norm)
        self.expectation_list = self.dataset.combine_expectation_list(self.source_expectation_list, self.bkg_expectation_list)

        logger.debug("The expected count histograms were updated.")

    def Mstep(self):
        """
        M-step. 
        In this step, self.delta_model will be updated.
        """

        ratio_list = [ data.event / expectation for data, expectation in zip(self.dataset, self.expectation_list) ]
        
        # delta model
        sum_T_product = self.dataset.calc_summed_T_product(ratio_list)
        self.delta_model = self.model * (sum_T_product/self.summed_exposure_map - 1)
        
        # masking
        if self.mask is not None:
            self.delta_model = self.delta_model.mask_pixels(self.mask)

        logger.debug("The delta model was updated.")

    def _ensure_model_constraints(self):
        """
        Ensure model satisfies physical constraints.
        
        This method enforces:
        1. Minimum flux constraint (non-negative flux)
        2. Masking (zero-exposure pixels)
        """

        # checking minimum flux
        self.model[:] = np.where(self.model.contents < self.minimum_flux, self.minimum_flux, self.model.contents)
        
        # masking again
        if self.mask is not None:
            self.model = self.model.mask_pixels(self.mask)

    def post_processing(self):
        """
        Post-processing. 
        """

        # updating model
        self.model[:] += self.delta_model.contents

        self._ensure_model_constraints()

    def register_result(self):
        """
        Register results at the end of each iteration. 
        """
        
        this_result = {"iteration": self.iteration_count, 
                       "model": self.model.copy()}

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
            histogram_keys         = [("model", f"{self.save_results_directory}/model.hdf5", self.save_only_final_result)],
            fits_filename          = f"{self.save_results_directory}/results.fits",
            values_key_name_format = [],
            dicts_key_name_format  = [],
            lists_key_name_format  = [],
        )
