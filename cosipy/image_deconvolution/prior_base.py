class PriorBase:

    usable_model_classes = []

    def __init__(self, coefficient, model):

        if not self.is_calculable(model):
            raise TypeError
        
        self.coefficient = coefficient

        self.model_class = type(model) 

    def is_calculable(self, model):

        if type(model) in self.usable_model_classes:

            return True

        return False

    def log_prior(self, model):

        raise NotImplementedError
    
    def grad_log_prior(self, x): 

        raise NotImplementedError
