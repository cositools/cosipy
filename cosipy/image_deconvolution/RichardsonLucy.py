import copy
import numpy as np
import astropy.units as u
from tqdm.autonotebook import tqdm

from .deconvolution_algorithm_base import DeconvolutionAlgorithmBase

class RichardsonLucy(DeconvolutionAlgorithmBase):
    use_sparse = False

    def __init__(self, initial_model_map, data, parameter):
        DeconvolutionAlgorithmBase.__init__(self, initial_model_map, data, parameter)

        self.loglikelihood = None

        self.alpha_max = parameter['alpha_max']

    def pre_processing(self):
        pass

    def Estep(self):
        self.expectation = self.calc_expectation(self.model_map, self.data, self.use_sparse)

    def Mstep(self):
        if self.use_sparse:
            diff = self.data.event_sparse / self.expectation - 1
            diff = diff.to_dense()
        else:
            diff = self.data.event_dense / self.expectation - 1

        diff = self.data.image_response_dense.expand_dims(diff, ["Time", "Em", "Phi", "PsiChi"])

        if self.use_sparse:
            delta_map_part1 = self.model_map / self.data.image_response_sparse_projected
            delta_map_part2 = (self.data.image_response_sparse * diff).project("lb", "Ei")
            self.delta_map  = delta_map_part1 * delta_map_part2
        else:
            delta_map_part1 = self.model_map / self.data.image_response_dense_projected
            delta_map_part2 = (self.data.image_response_dense * diff).project("lb", "Ei")
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

    def save_result(self, i_iteration):
        self.result["model_map"].write(f"model_map_itr{i_iteration}.hdf5", overwrite = True)
        self.result["delta_map"].write(f"delta_map_itr{i_iteration}.hdf5", overwrite = True)
        self.result["processed_delta_map"].write(f"processed_delta_map_itr{i_iteration}.hdf5", overwrite = True)

        with open(f"result_itr{i_iteration}.dat", "w") as f:
            f.write(f'alpha: {self.result["alpha"]}\n')
            f.write(f'loglikelihood: {self.result["loglikelihood"]}\n')

    def calc_alpha(self, delta, model_map):
        almost_zero = 1e-4 #it is to prevent the flux under zero
        alpha = -1.0 / np.min( delta / model_map ) * (1 - almost_zero)
        alpha = min(alpha, self.alpha_max)
        if alpha < 1.0:
            alpha = 1.0
        return alpha
