class PriorBase:

    allowerd_model_class = []

    def __init__(self, coefficient, model):
        
        self.coefficient = coefficient

        self.model_class = type(model) 

    def is_calculable(self, model):

        if type(model) in self.allowerd_model_class:

            return True

        return False

    def log_prior(self, model):

        raise NotImplementedError
    
    def grad_log_prior(self, x): 

        raise NotImplementedError
