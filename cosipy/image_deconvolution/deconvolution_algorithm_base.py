import numpy as np
import astropy.units as u
from tqdm.autonotebook import tqdm

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

        self.iteration_max = parameter['iteration']

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

    def iteration(self):
        self.model_map = self.initial_model_map

        stop_iteration = False
        for i_iteration in tqdm(range(1, self.iteration_max + 1)):
            if stop_iteration:
                break

            print("  Iteration {}/{} ".format(i_iteration, self.iteration_max))

            print("--> pre-processing")
            self.pre_processing()

            print("--> E-step")
            self.Estep()

            print("--> M-step")
            self.Mstep()
            
            print("--> post-processing")
            self.post_processing()

            print("--> checking stopping criteria")
            stop_iteration = self.check_stopping_criteria(i_iteration)
            print("--> --> {}".format("stop" if stop_iteration else "continue"))

            print("--> registering results")
            self.register_result(i_iteration)
            yield self.result
    
    #replaced with a function in other COSIpy libaray in the future?
    def calc_expectation(self, model_map, data, use_sparse = False):
        almost_zero = 1e-6

        model_map_expanded = data.image_response_dense.expand_dims(model_map, ["lb", "Ei"])
        if use_sparse:
            expectation = (data.image_response_sparse * model_map_expanded).project(["Time", "Em", "Phi", "PsiChi"]) * model_map.unit * self.pixelarea
            expectation += data.bkg_sparse
            expectation += almost_zero
        else:
            expectation = (data.image_response_dense * model_map_expanded).project(["Time", "Em", "Phi", "PsiChi"]) * model_map.unit * self.pixelarea
            expectation += data.bkg_dense 
            expectation += almost_zero
        
        return expectation

    def calc_loglikelihood(self, data, model_map, use_sparse = False):
        expectation = self.calc_expectation(model_map, data, use_sparse)
        if use_sparse:
            loglikelood = (np.sum( data.event_sparse * np.log(expectation) ) - np.sum(expectation)).value
        else:
            loglikelood = (np.sum( data.event_dense * np.log(expectation) ) - np.sum(expectation)).value
        return loglikelood
