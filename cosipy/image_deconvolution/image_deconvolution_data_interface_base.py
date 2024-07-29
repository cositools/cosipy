from abc import ABC, abstractmethod

class ImageDeconvolutionDataInterfaceBase(ABC):
    """
    A base class for managing data for image analysis, i.e., 
    event data, background models, response, coordsys conversion matrix etc.
    Subclasses must override these attributes and methods.

    Attributes:
    - self._event 
        A binned histogram of events. It is an instance of histpy.Histogram.
        Its axes must be the same of self._data_axes.
    - self._bkg_models
        A dictionary of binned histograms of background models.
        It is a dictionary of histpy.Histogram with keys of names of background models.
        Their axes must be the same of self._model_axes.
    - self._summed_bkg_models
        A dictionary of summed values of the background model histograms.
    - self._exposure_map 
        A binned histogram of the exposures at each pixel in the model space.
        It is an instance of histpy.Histogram.
        Its axes must be the same of self._model_axes.
    - self._model_axes 
        Axes for the data space. It is an instance of histpy.Axes.
    - self._data_axes 
        Axes for the model space. It is an instance of histpy.Axes.
    
    Methods:
    - keys_bkg_models()
        It returns a list of names of background models.
    - bkg_model(key)
        It returns a binned histogram of a background model with the given key.
    - summed_bkg_model(key)
        It returns the summed value of the background histogram with the given key.
    - calc_expectation(model)
        It returns a histogram of expected counts from the given model.
    - calc_T_product(dataspace_histogram)
        It returns the product of the input histogram with the transonse matrix of the response function.
    - calc_bkg_model_product(key, dataspace_histogram)
        It returns the product of an input histogram with a background model with the given key.
    - calc_likelihood(expectation)
        It returns the log-likelihood with the given histogram of expected counts.

    The basic idea of this class is to separate the data structure
    from the development of the image deconvolution algorithm.
    
    When the image deconvolution is performed, the deconvolution algorithm will look at only the above 
    attributes and methods, and will not care about the actual response matrix and how it actually calculates
    expected counts or the product using the transpose matrix of the response function.
    """

    def __init__(self, name = None):
        self._name = name

        # must assign data to them somewhere
        self._event = None # histpy.Histogram
        self._bkg_models = {} # a dictionary of histpy.Histogram
        self._summed_bkg_models = {} # a dictionary of float
        self._exposure_map = None # histpy.Histogram
        self._model_axes = None # histpy.Axes
        self._data_axes = None # histpy.Axes

    @property
    def name(self):
        return self._name

    @property
    def event(self):
        return self._event

    @property
    def exposure_map(self):
        return self._exposure_map

    @property
    def model_axes(self):
        return self._model_axes

    @property
    def data_axes(self):
        return self._data_axes

    def keys_bkg_models(self):
        return list(self._bkg_models.keys())

    def bkg_model(self, key):
        return self._bkg_models[key]

    def summed_bkg_model(self, key):
        return self._summed_bkg_models[key]

    @abstractmethod
    def calc_expectation(self, model, dict_bkg_norm = None, almost_zero = 1e-12):
        """
        Calculate expected counts from a given model map.

        Parameters
        ----------
        model : :py:class:`cosipy.image_deconvolution.ModelBase` or its subclass
            Model
        dict_bkg_norm : dict, default None
            background normalization for each background model, e.g, {'albedo': 0.95, 'activation': 1.05}
        almost_zero : float, default 1e-12
            In order to avoid zero components in extended count histogram, a tiny offset is introduced.
            It should be small enough not to effect statistics.

        Returns
        -------
        :py:class:`histpy.Histogram`
            Expected count histogram
        """
        raise NotImplementedError

    @abstractmethod
    def calc_T_product(self, dataspace_histogram):
        """
        Calculate the product of the input histogram with the transonse matrix of the response function.
        Let R_{ij}, H_{i} be the response matrix and dataspace_histogram, respectively.
        Note that i is the index for the data space, and j is for the model space.
        In this method, \sum_{j} H{i} R_{ij}, namely, R^{T} H is calculated.

        Parameters
        ----------
        dataspace_histogram: :py:class:`histpy.Histogram`
            Its axes must be the same as self.data_axes

        Returns
        -------
        :py:class:`histpy.Histogram`
            The product with self.model_axes
        """
        raise NotImplementedError
    
    @abstractmethod
    def calc_bkg_model_product(self, key, dataspace_histogram):
        """
        Calculate the product of the input histogram with the background model.
        Let B_{i}, H_{i} be the background model and dataspace_histogram, respectively.
        In this method, \sum_{i} B_{i} H_{i} is calculated.

        Parameters
        ----------
        key: str
            Background model name
        dataspace_histogram: :py:class:`histpy.Histogram`
            its axes must be the same as self.data_axes

        Returns
        -------
        float
        """
        raise NotImplementedError
    
    @abstractmethod
    def calc_loglikelihood(self, expectation):
        """
        Calculate log-likelihood from given expected counts or model/expectation.

        Parameters
        ----------
        expectation : :py:class:`histpy.Histogram`
            Expected count histogram.

        Returns
        -------
        float
            Log-likelood
        """
        raise NotImplementedError
