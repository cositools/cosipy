import numpy as np
import logging
logger = logging.getLogger(__name__)

from histpy import Histogram

from ..utils import _to_float
from .RichardsonLucyBasic import RichardsonLucyBasic

from ..constants import DEFAULT_BKG_NORM_RANGE

class RichardsonLucy(RichardsonLucyBasic):
    """
    Standard Richardson-Lucy algorithm with background optimization.
    
    This class extends RichardsonLucyBasic with background normalization
    optimization, making it suitable for practical image reconstruction
    with real data.
    
    Features:
    - E-step + M-step (from RichardsonLucyBasic)
    - Background normalization optimization
    
    For advanced features (response weighting, acceleration, smoothing),
    use RichardsonLucyAdvanced.
    For MAP estimation with priors, use MAP_RichardsonLucy.
    
    An example of parameter is as follows.

    iteration_max: 100
    minimum_flux:
        value: 0.0
        unit: "cm-2 s-1 sr-1"
    background_normalization_optimization:
        activate: True
        range: {"albedo": [0.9, 1.1]}
    save_results: 
        activate: True
        directory: "./results"
        only_final_result: True
    """

    def __init__(self, initial_model, dataset, mask, parameter):

        super().__init__(initial_model, dataset, mask, parameter)

        # background normalization optimization
        self.bkg_norm_optimization_enabled = parameter.get('background_normalization_optimization:activate', False)
        if self.bkg_norm_optimization_enabled:
            self.dict_delta_bkg_norm = {}
            self.dict_bkg_norm_range = parameter.get('background_normalization_optimization:range', {key: DEFAULT_BKG_NORM_RANGE for key in self.dict_bkg_norm.keys()})

        # update parameter summary
        self._parameter_summary += [
            ("bkg_norm_optimization_enabled", self.bkg_norm_optimization_enabled),
        ]
        if self.bkg_norm_optimization_enabled:
            self._parameter_summary += [
                ("dict_bkg_norm_range", self.dict_bkg_norm_range),
            ]

    def initialization(self):
        """
        initialization before running the image deconvolution
        """

        super().initialization()

        # calculate summed background models for M-step
        if self.bkg_norm_optimization_enabled:
            self.dict_summed_bkg_model = {}
            for key in self.dict_bkg_norm.keys():
                self.dict_summed_bkg_model[key] = self.dataset.calc_summed_bkg_model(key)

    def Mstep(self):
        """
        M-step in RL algorithm.
        In this step, self.delta_model and self.delta_bkg_norm will be updated.
        """

        ratio_list = [ data.event / expectation for data, expectation in zip(self.dataset, self.expectation_list) ]
        
        # delta model
        sum_T_product = self.dataset.calc_summed_T_product(ratio_list)
        self.delta_model = self.model * (sum_T_product/self.summed_exposure_map - 1)
        
        # masking
        if self.mask is not None:
            self.delta_model = self.delta_model.mask_pixels(self.mask)

        logger.debug("The delta model was updated.")
        
        # background normalization optimization
        if self.bkg_norm_optimization_enabled:
            for key in self.dict_bkg_norm.keys():

                sum_bkg_T_product = self.dataset.calc_summed_bkg_model_product(key, ratio_list)
                sum_bkg_model = self.dict_summed_bkg_model[key]

                self.dict_delta_bkg_norm[key] = self.dict_bkg_norm[key] * (sum_bkg_T_product / sum_bkg_model - 1)

    def _ensure_bkg_norm_range(self):
        """
        Ensure background normalization is within allowed range.
        
        This method clips background normalization values to their
        allowed ranges. If a value is outside the range, it is
        set to the nearest boundary value.
        """

        for key in self.dict_bkg_norm.keys():
            bkg_norm = self.dict_bkg_norm[key]
            bkg_range = self.dict_bkg_norm_range[key]

            if bkg_norm < bkg_range[0]:
                bkg_norm = bkg_range[0]
            elif bkg_norm > bkg_range[1]:
                bkg_norm = bkg_range[1]

            self.dict_bkg_norm[key] = _to_float(bkg_norm)

    def post_processing(self):
        """
        Post-processing. 
        """

        # updating model
        self.model[:] += self.delta_model.contents
        self._ensure_model_constraints()

        # update background normalization
        if self.bkg_norm_optimization_enabled:
            for key in self.dict_bkg_norm.keys():
                self.dict_bkg_norm[key] += self.dict_delta_bkg_norm[key]
            self._ensure_bkg_norm_range()

    def register_result(self):
        """
        Register results at the end of each iteration. 
        """
        
        this_result = {"iteration": self.iteration_count, 
                       "model": self.model.copy(), 
                       "background_normalization": self.dict_bkg_norm.copy()}

        # show intermediate results
        for key in ["background_normalization"]:
            logger.info(f"{key}: {this_result[key]}")
        
        # register this_result in self.results
        self.results.append(this_result)

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
            dicts_key_name_format  = [("background_normalization", "BKG_NORM", "D")],
            lists_key_name_format  = [],
        )
