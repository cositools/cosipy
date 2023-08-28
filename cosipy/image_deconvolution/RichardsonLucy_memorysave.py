import copy
import numpy as np
import astropy.units as u
from tqdm.autonotebook import tqdm
import gc

from histpy import Histogram

from .deconvolution_algorithm_base import DeconvolutionAlgorithmBase

class RichardsonLucy_memorysave(DeconvolutionAlgorithmBase):
    use_sparse = False

    def __init__(self, initial_model_map, data, parameter):
        DeconvolutionAlgorithmBase.__init__(self, initial_model_map, data, parameter)

        self.loglikelihood = None

        self.alpha_max = parameter.get('alpha_max', 1.0)

        self.do_response_weighting = parameter.get('response_weighting', False)

        self.do_smoothing = parameter.get('smoothing', False)

        self.do_bkg_norm_fitting = parameter.get('background_normalization_fitting', False)

        if self.do_bkg_norm_fitting:
            self.bkg_norm_range = parameter.get('background_normalization_range', [0.5, 1.5])

        print("... calculating the expected events with the initial model map ...")
        self.expectation = self.calc_expectation(self.initial_model_map, self.data, self.use_sparse)

        if self.do_response_weighting:
            print("... calculating the response weighting filter...")

            response_weighting_index = parameter.get('response_weighting_index', 0.5)

            self.response_weighting_filter = data.image_response_dense_projected.contents / np.max(data.image_response_dense_projected.contents) 

            self.response_weighting_filter = self.response_weighting_filter**response_weighting_index

        if self.do_smoothing:
            print("... calculating the gaussian filter...")

            self.smoothing_sigma = parameter['smoothing_FWHM'] / 2.354820 # degree

            self.smoothing_max_sigma = parameter.get('smoothing_max_sigma', default = 5.0)
            self.gaussian_filter = self.calc_gaussian_filter(self.smoothing_sigma, self.smoothing_max_sigma)

    def pre_processing(self):
        pass

    def Estep(self):
#        self.expectation = self.calc_expectation(self.model_map, self.data, self.use_sparse)
        print("... skip E-step ...")

    def Mstep(self):
        diff = self.data.event_dense / self.expectation - 1

        delta_map_part1 = self.model_map / self.data.image_response_dense_projected
        delta_map_part2 = Histogram(self.model_map.axes, unit = self.data.image_response_dense_projected.unit)

        if self.data.response_on_memory == True:
            diff_x_response_this_pix = np.tensordot(diff.contents, self.data.image_response_dense.contents, axes = ([1,2,3], [2,3,4])) # Ti, NuLambda, Ei

            delta_map_part2[:] = np.tensordot(self.data.coordsys_conv_matrix.contents, diff_x_response_this_pix, axes = ([1,2], [0,1])) * diff_x_response_this_pix.unit * self.data.coordsys_conv_matrix.unit #lb, Ei
            # note that coordsys_conv_matrix is the sparse, so the unit should be recovered.

        else:
            for ipix in tqdm(range(self.npix)):
                response_this_pix = np.sum(self.data.full_detector_response[ipix].to_dense(), axis = (4,5)) # 'Ei', 'Em', 'Phi', 'PsiChi'

                diff_x_response_this_pix = np.tensordot(diff.contents, response_this_pix, axes = ([1,2,3], [1,2,3])) # Ti, Ei

                delta_map_part2 += np.tensordot(self.data.coordsys_conv_matrix[:,:,ipix], diff_x_response_this_pix, axes = ([1],[0])) * diff_x_response_this_pix.unit * self.data.coordsys_conv_matrix.unit #lb, Ei

        self.delta_map = delta_map_part1 * delta_map_part2

        if self.do_bkg_norm_fitting:
            self.bkg_norm += self.bkg_norm * np.sum(diff * self.data.bkg_dense) / np.sum(self.data.bkg_dense)

            if self.bkg_norm < self.bkg_norm_range[0]:
                self.bkg_norm = self.bkg_norm_range[0]
            elif self.bkg_norm > self.bkg_norm_range[1]:
                self.bkg_norm = self.bkg_norm_range[1]

            print("bkg_norm : ", self.bkg_norm)

    def post_processing(self):

        if self.do_response_weighting:
            self.delta_map[:,:] *= self.response_weighting_filter

        if self.do_smoothing:
            self.delta_map[:,:] = np.tensordot(self.gaussian_filter.contents, self.delta_map.contents, axes = [[0], [0]])

        self.alpha = self.calc_alpha(self.delta_map, self.model_map)

        self.processed_delta_map = self.delta_map * self.alpha

        self.model_map += self.processed_delta_map 

        print("... calculating the expected events with the updated model map ...")
        self.expectation = self.calc_expectation(self.model_map, self.data, self.use_sparse)

    def check_stopping_criteria(self, i_iteration):
        if i_iteration < self.iteration_max:
            return False
        return True

    def register_result(self, i_iteration):
        loglikelihood = self.calc_loglikelihood(self.data, self.model_map, self.expectation)

        this_result = {"iteration": i_iteration, 
                       "model_map": copy.deepcopy(self.model_map), 
                       "delta_map": copy.deepcopy(self.delta_map),
                       "processed_delta_map": copy.copy(self.processed_delta_map),
                       "alpha": self.alpha, 
                       "background_normalization": self.bkg_norm,
                       "loglikelihood": loglikelihood}

        self.result = this_result

    def save_result(self, i_iteration):
        self.result["model_map"].write(f"model_map_itr{i_iteration}.hdf5", overwrite = True)
        self.result["delta_map"].write(f"delta_map_itr{i_iteration}.hdf5", overwrite = True)
        self.result["processed_delta_map"].write(f"processed_delta_map_itr{i_iteration}.hdf5", overwrite = True)

        with open(f"result_itr{i_iteration}.dat", "w") as f:
            f.write(f'alpha: {self.result["alpha"]}\n')
            f.write(f'loglikelihood: {self.result["loglikelihood"]}\n')
            f.write(f'background_normalization: {self.result["background_normalization"]}\n')

    def calc_alpha(self, delta, model_map):
        almost_zero = 1e-4 #it is to prevent the flux under zero
        alpha = -1.0 / np.min( delta / model_map ) * (1 - almost_zero)
        alpha = min(alpha, self.alpha_max)
        if alpha < 1.0:
            alpha = 1.0
        return alpha

    def calc_expectation(self, model_map, data, use_sparse = False): ### test with separating the dwell time map
        print("calc_expectation, memory-save version")
        almost_zero = 1e-6

        expectation = Histogram(data.event_dense.axes) 

        map_rotated = np.tensordot(data.coordsys_conv_matrix.contents, model_map.contents, axes = ([0], [0])) # Time, NuLambda, Ei
        map_rotated *= data.coordsys_conv_matrix.unit * model_map.unit # data.coordsys_conv_matrix.contents is sparse, so the unit should be restored.

        if data.response_on_memory == True:
            expectation[:] = np.tensordot( map_rotated, data.image_response_dense.contents, axes = ([1,2], [0,1])) * self.pixelarea
        else:
            for ipix in tqdm(range(self.npix)):
                response_this_pix = np.sum(data.full_detector_response[ipix].to_dense(), axis = (4,5)) # 'Ei', 'Em', 'Phi', 'PsiChi'
                expectation += np.tensordot(map_rotated[:,ipix,:], response_this_pix, axes = ([1], [0])) * self.pixelarea

        expectation += data.bkg_dense * self.bkg_norm
        expectation += almost_zero
        
        return expectation

