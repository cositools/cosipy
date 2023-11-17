import gc
import numpy as np
import astropy.units as u
from tqdm.autonotebook import tqdm
import healpy as hp

from histpy import Histogram, Axes, Axis

class DeconvolutionAlgorithmBase(object):

    def __init__(self, initial_model_map, data, parameter):
        self.data = data 

        self.parameter = parameter 

        self.initial_model_map = initial_model_map

        image_axis = initial_model_map.axes['lb']

        self.nside = image_axis.nside
        self.npix = image_axis.npix
        self.pixelarea = 4 * np.pi / self.npix * u.sr
        energy_axis = initial_model_map.axes['Ei']
        self.nbands = len(energy_axis) - 1

        self.model_map = None
        self.delta_map = None
        self.processed_delta_map = None

        self.expectation = None

        self.bkg_norm = 1.0

        self.iteration_max = parameter['iteration']

        self.save_result = parameter.get("save_results_each_iteration", False)

        self.result = None

    def pre_processing(self):
        pass

    def Estep(self):
        # calculate expected events
        pass

    def Mstep(self):
        # calculate delta map 
        pass

    def post_processing(self):
        # applying filters to the delta map
        pass

    def check_stopping_criteria(self, i_iteration):
        if i_iteration < 0:
            return False
        return True

    def register_result(self, i_iteration):
        this_result = {"iteration": self.i_iteration + 1}
        self.result = this_result

    def save_result(self, i_iteration):
        pass

    def iteration(self):
        self.model_map = self.initial_model_map

        stop_iteration = False
        for i_iteration in tqdm(range(1, self.iteration_max + 1)):
            gc.collect()

            if stop_iteration:
                break

            print("  Iteration {}/{} ".format(i_iteration, self.iteration_max))

            print("--> pre-processing")
            self.pre_processing()

            print("--> E-step")
            self.Estep()
            gc.collect()

            print("--> M-step")
            self.Mstep()
            gc.collect()
            
            print("--> post-processing")
            self.post_processing()

            print("--> checking stopping criteria")
            stop_iteration = self.check_stopping_criteria(i_iteration)
            print("--> --> {}".format("stop" if stop_iteration else "continue"))

            print("--> registering results")
            self.register_result(i_iteration)
            
            if self.save_result == True:
                print("--> saving results")
                self.save_result(i_iteration)

            gc.collect()

            yield self.result
    
    #replaced with a function in other COSIpy libaray in the future?
    def calc_expectation(self, model_map, data): ### test with separating the dwell time map
        almost_zero = 1e-6

        expectation = Histogram(data.event_dense.axes) 

        map_rotated = np.tensordot(data.coordsys_conv_matrix.contents, model_map.contents, axes = ([0], [0])) # ['lb', 'Time/Scat', 'NuLambda'] x ['lb', 'Ei'] -> [Time/Scat, NuLambda, Ei]
        map_rotated *= data.coordsys_conv_matrix.unit * model_map.unit # data.coordsys_conv_matrix.contents is sparse, so the unit should be restored.

        if data.response_on_memory == True:
            expectation[:] = np.tensordot( map_rotated, data.image_response_dense.contents, axes = ([1,2], [0,1])) * self.pixelarea
        else:
            for ipix in tqdm(range(self.npix)):
                response_this_pix = np.sum(data.full_detector_response[ipix].to_dense(), axis = (4,5)) # ['Ei', 'Em', 'Phi', 'PsiChi', 'SigmaTau', 'Dist'] -> ['Ei', 'Em', 'Phi', 'PsiChi']
                expectation += np.tensordot(map_rotated[:,ipix,:], response_this_pix, axes = ([1], [0])) * self.pixelarea

        expectation += data.bkg_dense * self.bkg_norm
        expectation += almost_zero
        
        return expectation

    def calc_loglikelihood(self, data, model_map, expectation = None): # expectation will be a mandatory parameter in the future.
        if expectation is None:
            expectation = self.calc_expectation(model_map, data)

        loglikelood = np.sum( data.event_dense * np.log(expectation) ) - np.sum(expectation)

        return loglikelood

    def calc_gaussian_filter(self, sigma, max_sigma):
        gaussian_filter = Histogram( Axes( [Axis(edges = np.arange(self.npix+1)), Axis(edges = np.arange(self.npix+1))] ), sparse = False)

        for ipix in tqdm(range(self.npix)):
            vec_i = hp.pix2vec(self.nside, ipix, nest = False)
    
            for jpix in range(self.npix):
                vec_j = hp.pix2vec(self.nside, jpix, nest = False)
                
                delta_cos = vec_i[0] * vec_j[0] + vec_i[1] * vec_j[1] + vec_i[2] * vec_j[2]

                if delta_cos > 1.0:
                    delta_cos = 1.0
                elif delta_cos < -1.0:
                    delta_cos = -1.0
                    
                delta_ang = np.arccos(delta_cos) * 180.0 / np.pi
                
                if delta_ang / sigma < max_sigma:
                    gaussian_filter[ipix,jpix] = np.exp( - 0.5 * delta_ang**2 / sigma**2)
    
            gaussian_filter[ipix,:] /= np.sum(gaussian_filter[ipix,:])
    
        return gaussian_filter
