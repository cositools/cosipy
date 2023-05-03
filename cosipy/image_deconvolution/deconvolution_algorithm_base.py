import numpy as np
from tqdm import tqdm

class DeconvolutionAlgorithmBase(object):

    def __init__(self, initial_model_map, data, parameter):
        self.data = data 

        self.parameter = parameter 

        self.initial_model_map = initial_model_map
        self.model_map = self.initial_model_map
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

        model_map_expanded = data.image_response_mul_time.expand_dims(model_map, ["NuLambda", "Ei"])
        if use_sparse:
            expectation = (data.image_response_mul_time * model_map_expanded).project(["Em", "Phi", "PsiChi"]) * model_map.unit * self.pixelarea
            expectation += data.bkg
            expectation += almost_zero
        else:
            expectation = (data.image_response_mul_time_dense * model_map_expanded).project(["Em", "Phi", "PsiChi"]) * model_map.unit * self.pixelarea
            expectation += data.bkg_dense 
            expectation += almost_zero
        
        return expectation

    def calc_loglikelihood(self, data, model_map, use_sparse = False):
        expectation = self.calc_expectation(model_map, data, use_sparse)
        loglikelood = (np.sum( data.event * np.log(expectation) ) - np.sum(expectation)).value
        return loglikelood
