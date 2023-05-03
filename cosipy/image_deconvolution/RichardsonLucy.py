import copy
import numpy as np
import astropy.units as u
from tqdm import tqdm

from .deconvolution_algorithm_base import DeconvolutionAlgorithmBase

class RichardsonLucy(DeconvolutionAlgorithmBase):
    use_sparse = False

    def __init__(self, initial_model_map, data, parameter):
        DeconvolutionAlgorithmBase.__init__(self, initial_model_map, data, parameter)

        spherical_axis = initial_model_map.axes['NuLambda']
        self.nside = spherical_axis.nside
        self.npix = spherical_axis.npix
        self.pixelarea = 4 * np.pi / self.npix * u.sr
        energy_axis = initial_model_map.axes['Ei']
        self.nbands = len(energy_axis) - 1

        self.loglikelihood = None

        self.alpha_max = parameter['alpha_max']

    def pre_processing(self):
        pass

    def Estep(self):
        self.expectation = self.calc_expectation(self.model_map, self.data, self.use_sparse)

    def Mstep(self):
        if self.use_sparse:
            diff = self.data.event / self.expectation - 1
            diff = diff.to_dense()
        else:
            diff = self.data.event_dense / self.expectation - 1

        diff = self.data.image_response_mul_time.expand_dims(diff, ["Em", "Phi", "PsiChi"])

        if self.use_sparse:
            delta_map_part1 = self.model_map / self.data.image_response_mul_time_projected
            delta_map_part2 = (self.data.image_response_mul_time * diff).project("NuLambda", "Ei")
            self.delta_map  = delta_map_part1 * delta_map_part2
        else:
            delta_map_part1 = self.model_map / self.data.image_response_mul_time_dense_projected
            delta_map_part2 = (self.data.image_response_mul_time_dense * diff).project("NuLambda", "Ei")
            self.delta_map  = delta_map_part1 * delta_map_part2

    def post_processing(self):
        self.alpha = self.calc_alpha(self.delta_map, self.model_map)
        self.processed_delta_map = self.delta_map * self.alpha
        self.model_map += self.processed_delta_map 

    def check_stopping_criteria(self, i_iteration):
        if i_iteration < self.iteration_max:
            return False
        return True

    def register_result(self, i_iteration):
        loglikelihood = self.calc_loglikelihood(self.data, self.model_map)

        this_result = {"iteration": i_iteration, 
                       "model_map": copy.deepcopy(self.model_map), 
                       "delta_map": copy.deepcopy(self.delta_map),
                       "processed_delta_map": copy.copy(self.processed_delta_map),
                       "alpha": self.alpha, 
                       "loglikelihood": loglikelihood}

        self.result = this_result

    def calc_alpha(self, delta, model_map):
        alpha = -1.0 / np.min( delta / model_map ) * (1 - 1e-4) #1e-4 is to prevent the flux under zero
        alpha = min(alpha, self.alpha_max)
        return alpha
