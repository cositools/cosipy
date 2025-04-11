from abc import ABC, abstractmethod

class PriorBase:
    """
    Abstract base class for prior distributions.

    This class provides a framework for implementing different types of prior
    probability distributions for image reconstructions.

    Parameters
    ----------
    coefficient : float
        Scaling coefficient for the prior probability.
    model : object of a subclass of :py:class:`cosipy.image_deconvolution.ModelBase`
        Model object to which the prior will be applied.

    Attributes
    ----------
    usable_model_classes : list
        List of model classes that can be used with this prior.
    coefficient : float
        Scaling coefficient for the prior probability.
    model_class : type
        Type of the model being used.
    """

    usable_model_classes = []

    def __init__(self, coefficient, model):

        if not self.is_calculable(model):
            raise TypeError
        
        self.coefficient = coefficient

        self.model_class = type(model) 

    def is_calculable(self, model):
        """
        Check if the prior can be calculated for the given model.

        Parameters
        ----------
        model : object of a subclass of :py:class:`cosipy.image_deconvolution.ModelBase`
            Model to check for compatibility.

        Returns
        -------
        bool
            True if the model is compatible, False otherwise.
        """
        if type(model) in self.usable_model_classes:

            return True

        return False

    @abstractmethod
    def log_prior(self, model):
        """
        Calculate the logarithm of the prior probability.

        Parameters
        ----------
        model : object of a subclass of :py:class:`cosipy.image_deconvolution.ModelBase`
            Model for which to calculate the log prior.

        Returns
        -------
        float
            The logarithm of the prior probability.
        """

        raise NotImplementedError
    
    @abstractmethod
    def grad_log_prior(self, model): 
        """
        Calculate the gradient of the log prior with respect to model parameters.

        Parameters
        ----------
        model : object of a subclass of :py:class:`cosipy.image_deconvolution.ModelBase`
            Model for which to calculate the log prior.

        Returns
        -------
        numpy.ndarray
            Gradient of the log prior.
            Its shape must be the same as the input model.
        """

        raise NotImplementedError
