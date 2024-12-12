import os
from pathlib import Path
import copy
import logging

# logging setup
logger = logging.getLogger(__name__)

# Import third party libraries
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.io import fits
from mpi4py import MPI
from yayc import Configurator
from histpy import Histogram, HealpixAxis, Axes

from .deconvolution_algorithm_base import DeconvolutionAlgorithmBase
from .allskyimage import AllSkyImageModel

# Define MPI variable
MASTER = 0                      # Indicates master process

class RichardsonLucyWithParallel(DeconvolutionAlgorithmBase):
    """
    A class for the RichardsonLucy algorithm. 
    The algorithm here is based on Knoedlseder+99, Knoedlseder+05, Siegert+20.
    
    An example of parameter is as follows.

    iteration_max: 100
    minimum_flux:
        value: 0.0
        unit: "cm-2 s-1 sr-1"
    acceleration: True
    alpha_max: 10.0
    response_weighting: True 
    response_weighting_index: 0.5
    smoothing: True 
    smoothing_FWHM: 
      value: 2.0
      unit: "deg"
    background_normalization_optimization: True 
    background_normalization_range: {"albedo": [0.9, 1.1]}
    save_results: True
    save_results_directory: "./results"

    """

    def __init__(self, initial_model:Histogram, dataset:list, mask, parameter, comm=None):

        DeconvolutionAlgorithmBase.__init__(self, initial_model, dataset, mask, parameter)

        self.do_acceleration = parameter.get('acceleration', False)

        self.alpha_max = parameter.get('alpha_max', 1.0)

        self.do_response_weighting = parameter.get('response_weighting', False)
        if self.do_response_weighting:
            self.response_weighting_index = parameter.get('response_weighting_index', 0.5)

        self.do_smoothing = parameter.get('smoothing', False)
        if self.do_smoothing:
            self.smoothing_fwhm = parameter['smoothing_FWHM']['value'] * u.Unit(parameter['smoothing_FWHM']['unit'])
            logger.info(f"Gaussian filter with FWHM of {self.smoothing_fwhm} will be applied to delta images ...")

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

        self.parallel = False
        if comm is not None:
            self.comm = comm
            if self.comm.Get_size() > 1:
                self.parallel = True
                logger.info('Image Deconvolution set to run in parallel mode')
            # Specific to parallel implementation:
            # 1. Assume numproc is known by the process that invoked `run_deconvolution()`
            # 2. All processes have loaded event data, background, and initial model independently

    def initialization(self):
        """
        initialization before running the image deconvolution
        """

        if self.parallel:
            self.numtasks = self.comm.Get_size()
            self.taskid = self.comm.Get_rank()

        # clear results
        if (not self.parallel) or (self.parallel and (self.taskid == MASTER)):
            self.results.clear()

        # clear counter 
        self.iteration_count = 0

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
        E-step (but it will be skipped).
        """

        # expected count histograms
        expectation_list_slice = self.calc_expectation_list(model = self.model, dict_bkg_norm = self.dict_bkg_norm)
        logger.info("The expected count histograms were calculated with the initial model map.")

        if not self.parallel:
            self.expectation_list = expectation_list_slice     # If single process, then full = slice

        elif self.parallel:
            '''
            Synchronization Barrier 1
            '''
            
            self.expectation_list = []
            for data, epsilon_slice in zip(self.dataset, expectation_list_slice):
                # Gather the sizes of local arrays from all processes
                local_size = np.array([epsilon_slice.contents.size], dtype=np.int32)
                all_sizes = np.zeros(self.numtasks, dtype=np.int32)
                self.comm.Allgather(local_size, all_sizes)

                # Calculate displacements 
                displacements = np.insert(np.cumsum(all_sizes), 0, 0)[0:-1]

                # Create a buffer to receive the gathered data 
                total_size = int(np.sum(all_sizes))
                recvbuf = np.empty(total_size, dtype=np.float64)      # Receive buffer

                # Gather all arrays into recvbuf
                self.comm.Allgatherv(epsilon_slice.contents.flatten(), [recvbuf, all_sizes, displacements, MPI.DOUBLE])   # For multiple MPI processes, full = [slice1, ... sliceN]

                # Reshape the received buffer back into the original 3D array shape
                epsilon = np.concatenate([ recvbuf[displacements[i]:displacements[i] + all_sizes[i]].reshape((-1,) + epsilon_slice.contents.shape[1:]) for i in range(self.numtasks) ], axis=-1)

                # Create Histogram that will be appended to self.expectation_list
                axes = []
                for axis in data.event.axes:
                    if axis.label == 'PsiChi':
                        axes.append(HealpixAxis(edges = axis.edges, 
                                                label = axis.label, 
                                                scale = axis._scale,
                                                coordsys = axis._coordsys,
                                                nside = axis.nside))
                    else:
                        axes.append(axis)

                # Add to list that manages multiple datasets
                self.expectation_list.append(Histogram(Axes(axes), contents=epsilon, unit=data.event.unit))

        # At the end of this function, all processes should have a complete `self.expectation_list`
        # to proceed to the Mstep function

    def Mstep(self):
        """
        M-step in RL algorithm.
        """

        ratio_list = [ data.event / expectation for data, expectation in zip(self.dataset, self.expectation_list) ]
        
        # delta model
        C_slice = self.calc_summed_T_product(ratio_list)

        if not self.parallel:
            sum_T_product = C_slice

        elif self.parallel:
            '''
            Synchronization Barrier 2
            '''
            
            # Gather the sizes of local arrays from all processes
            local_size = np.array([C_slice.contents.size], dtype=np.int32)
            all_sizes = np.zeros(self.numtasks, dtype=np.int32)
            self.comm.Allgather(local_size, all_sizes)

            # Calculate displacements 
            displacements = np.insert(np.cumsum(all_sizes), 0, 0)[0:-1]

            # Create a buffer to receive the gathered data 
            total_size = int(np.sum(all_sizes)) 
            recvbuf = np.empty(total_size, dtype=np.float64)      # Receive buffer

            # Gather all arrays into recvbuf
            self.comm.Gatherv(C_slice.contents.value.flatten(), [recvbuf, all_sizes, displacements, MPI.DOUBLE])   # For multiple MPI processes, full = [slice1, ... sliceN]

            if self.taskid == MASTER:
                # Reshape the received buffer back into the original 2D array shape
                C = np.concatenate([ recvbuf[displacements[i]:displacements[i] + all_sizes[i]].reshape((-1,) + C_slice.contents.shape[1:]) for i in range(self.numtasks) ], axis=0)

                # Create Histogram object for sum_T_product
                axes = []
                for axis in self.model.axes:
                    if axis.label == 'lb':
                        axes.append(HealpixAxis(edges = axis.edges, 
                                                label = axis.label, 
                                                scale = axis._scale,
                                                coordsys = axis._coordsys,
                                                nside = axis.nside))
                    else:
                        axes.append(axis)

                # C_slice (only slice operated on by current node) --> sum_T_product (all )
                sum_T_product = Histogram(Axes(axes), contents=C, unit=C_slice.unit)

        if (not self.parallel) or ((self.parallel) and (self.taskid == MASTER)):
            self.delta_model = self.model * (sum_T_product/self.summed_exposure_map - 1)

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

        # At the end of this function, just the MASTER MPI process needs to have a full
        # copy of delta_model

    def post_processing(self):
        """
        Here three processes will be performed.
        - response weighting filter: the delta map is renormalized as pixels with large exposure times will have more feedback.
        - gaussian smoothing filter: the delta map is blurred with a Gaussian function.
        - acceleration of RL algirithm: the normalization of delta map is increased as long as the updated image has no non-negative components.
        """

        if (not self.parallel) or ((self.parallel) and (self.taskid == MASTER)):
            self.processed_delta_model = copy.deepcopy(self.delta_model)

            if self.do_response_weighting:
                self.processed_delta_model[:] *= self.response_weighting_filter

            if self.do_smoothing:
                self.processed_delta_model = self.processed_delta_model.smoothing(fwhm = self.smoothing_fwhm)
            
            if self.do_acceleration:
                self.alpha = self.calc_alpha(self.processed_delta_model, self.model)
            else:
                self.alpha = 1.0

            self.model = self.model + self.processed_delta_model * self.alpha
            self.model[:] = np.where(self.model.contents < self.minimum_flux, self.minimum_flux, self.model.contents)

            if self.mask is not None:
                self.model = self.model.mask_pixels(self.mask)

            # update loglikelihood_list
            self.loglikelihood_list = self.calc_loglikelihood_list(self.expectation_list)
            logger.debug("The loglikelihood list was updated with the new expected count histograms.")

        if self.parallel:
            '''
            Synchronization Barrier 3
            '''
            # Initialize new variable as MPI only sends values and not units
            if self.taskid == MASTER:
                buffer = self.model.contents.value
            else:
                buffer = np.empty(self.model.contents.shape, dtype=np.float64)

            self.comm.Bcast([buffer, MPI.DOUBLE], root=MASTER)  # This synchronization barrier is not required during the final iteration

            if self.taskid > MASTER:
                # Reconstruct ModelBase object for self.model
                new_model = self.model.__class__(nside = self.model.axes['lb'].nside,       # self.model.__class__ will return the Class of which `model` is an object
                                                energy_edges = self.model.axes['Ei'].edges, 
                                                scheme = self.model.axes['lb']._scheme, 
                                                coordsys = self.model.axes['lb'].coordsys,
                                                unit = self.model.unit)
                
                new_model[:] = buffer * self.model.unit
                self.model = new_model

        # At the end of this function, all MPI processes needs to have a full
        # copy of updated model. 

    def register_result(self):
        """
        The values below are stored at the end of each iteration.
        - iteration: iteration number
        - model: updated image
        - delta_model: delta map after M-step 
        - processed_delta_model: delta map after post-processing
        - alpha: acceleration parameter in RL algirithm
        - background_normalization: optimized background normalization
        - loglikelihood: log-likelihood
        """
        
        if (not self.parallel) or ((self.parallel) and (self.taskid == MASTER)):
            this_result = {"iteration": self.iteration_count, 
                        "model": copy.deepcopy(self.model), 
                        "delta_model": copy.deepcopy(self.delta_model),
                        "processed_delta_model": copy.deepcopy(self.processed_delta_model),
                        "background_normalization": copy.deepcopy(self.dict_bkg_norm),
                        "alpha": self.alpha, 
                        "loglikelihood": copy.deepcopy(self.loglikelihood_list)}

            # show intermediate results
            logger.info(f'  alpha: {this_result["alpha"]}')
            logger.info(f'  background_normalization: {this_result["background_normalization"]}')
            logger.info(f'  loglikelihood: {this_result["loglikelihood"]}')
        
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

        if (not self.parallel) or ((self.parallel) and (self.taskid == MASTER)):
            if self.save_results == True:
                logger.info('Saving results in {self.save_results_directory}')

                # model
                for this_result in self.results:
                    iteration_count = this_result["iteration"]

                    this_result["model"].write(f"{self.save_results_directory}/model_itr{iteration_count}_2.hdf5", overwrite = True)
                    this_result["delta_model"].write(f"{self.save_results_directory}/delta_model_itr{iteration_count}_2.hdf5", overwrite = True)
                    this_result["processed_delta_model"].write(f"{self.save_results_directory}/processed_delta_model_itr{iteration_count}_2.hdf5", overwrite = True)

                #fits
                primary_hdu = fits.PrimaryHDU()

                col_iteration = fits.Column(name='iteration', array=[float(result['iteration']) for result in self.results], format='K')
                col_alpha = fits.Column(name='alpha', array=[float(result['alpha']) for result in self.results], format='D')
                cols_bkg_norm = [fits.Column(name=key, array=[float(result['background_normalization'][key]) for result in self.results], format='D') 
                                for key in self.dict_bkg_norm.keys()]
                cols_loglikelihood = [fits.Column(name=f"{self.dataset[i].name}", array=[float(result['loglikelihood'][i]) for result in self.results], format='D') 
                                    for i in range(len(self.dataset))]

                table_alpha = fits.BinTableHDU.from_columns([col_iteration, col_alpha])
                table_alpha.name = "alpha"

                table_bkg_norm = fits.BinTableHDU.from_columns([col_iteration] + cols_bkg_norm)
                table_bkg_norm.name = "bkg_norm"

                table_loglikelihood = fits.BinTableHDU.from_columns([col_iteration] + cols_loglikelihood)
                table_loglikelihood.name = "loglikelihood"

                hdul = fits.HDUList([primary_hdu, table_alpha, table_bkg_norm, table_loglikelihood])
                hdul.writeto(f'{self.save_results_directory}/results.fits',  overwrite=True)

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
