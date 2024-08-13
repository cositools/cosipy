import os
from pathlib import Path
import copy
import subprocess
import logging
logger = logging.getLogger(__name__)

# Import third party libraries
import numpy as np
import h5py
from histpy import Histogram

from .deconvolution_algorithm_base import DeconvolutionAlgorithmBase

class RichardsonLucyParallel(DeconvolutionAlgorithmBase):
    """
    NOTE: Comments copied from RichardsonLucy.py
    A class for a parallel implementation of the Richardson-
    Lucy algorithm. 
    
    An example of parameter is as follows.

    iteration_max: 100
    minimum_flux:
        value: 0.0
        unit: "cm-2 s-1 sr-1"
    background_normalization_optimization: True 
    """

    def __init__(self, initial_model, dataset, mask, parameter):
        """
        NOTE: Copied from RichardsonLucy.py
        """

        DeconvolutionAlgorithmBase.__init__(self, initial_model, dataset, mask, parameter)

        # TODO: these RL algorithm improvements are yet to be implemented/utilized in this file
        # self.do_acceleration = parameter.get('acceleration', False)

        # self.alpha_max = parameter.get('alpha_max', 1.0)

        # self.do_response_weighting = parameter.get('response_weighting', False)
        # if self.do_response_weighting:
        #     self.response_weighting_index = parameter.get('response_weighting_index', 0.5)

        # self.do_smoothing = parameter.get('smoothing', False)
        # if self.do_smoothing:
        #     self.smoothing_fwhm = parameter['smoothing_FWHM']['value'] * u.Unit(parameter['smoothing_FWHM']['unit'])
        #     logger.info(f"Gaussian filter with FWHM of {self.smoothing_fwhm} will be applied to delta images ...")

        self.do_bkg_norm_optimization = parameter.get('background_normalization_optimization', False)
        if self.do_bkg_norm_optimization:
            self.dict_bkg_norm_range = parameter.get('background_normalization_range', {key: [0.0, 100.0] for key in self.dict_bkg_norm.keys()})

        self.save_results = parameter.get('save_results', False)
        self.save_results_directory = parameter.get('save_results_directory', './results')

        if self.save_results is True:
            if os.path.isdir(self.save_results_directory):
                logger.warning(f"A directory {self.save_results_directory} already exists. Files in {self.save_results_directory} may be overwritten. Make sure that is not a problem.")
            else:
                os.makedirs(self.save_results_directory)

        # Specific to parallel implementation
        self.numproc = parameter.get('numproc', 1)
        self.iteration_max = parameter.get('iteration_max', 10)
        self.base_dir = os.getcwd()
        self.data_dir = parameter.get('data_dir', './data')     # NOTE: Data should ideally be present in disk scratch space.

        image_response = self.dataset[0]._image_response
        self.numrows = np.product(image_response.contents.shape[1:])
        self.numcols = np.product(image_response.contents.shape[:1])

    def iteration(self):
        """
        Performs all iterations of image deconvolution.

        NOTE: Overriding implementation in deconvolution_algorithm_base.py and invoking external script
        """

        # All arguments must be passed as type=str. Explicitly type cast boolean and number to string.
        subprocess.run(args=["mpiexec", "-n", str(self.numproc), "python", "mpitest.py", 
                             "--numrows", str(self.numrows),
                             "--numcols", str(self.numcols),
                             "--iteration_max", str(self.iteration_max), 
                             "--data_dir", self.data_dir,
                             "--save_results", str(self.save_results),
                             "--results_dir", self.save_results_directory],
                        text=True)

        # RLparallelscript already contains check_stopping_criteria and iteration_max break condition. 
        # NOTE: RichardsonLucy.py currently does not support a sophisticated break condition. 
        return True

    def initialization(self):
        """
        initialization before running the image deconvolution
        """
        # TODO: Write bkg and data to scratch space
        pass

    def finalization(self):
        """
        finalization after running the image deconvolution
        """
        pass

    def pre_processing(self):
        pass
    def Estep(self):
        pass
    def Mstep(self):
        pass
    def post_processing(self):
        pass
    def check_stopping_criteria(self):
        pass
    def register_result(self):
        pass