import numpy as np
import astropy.units as u
import astropy.io.fits as fits
import functools
from abc import ABC, abstractmethod
import logging
logger = logging.getLogger(__name__)

class DeconvolutionAlgorithmBase(ABC):
    """
    A base class for image deconvolution algorithms.
    Subclasses should override these methods:

    - initialization 
    - pre_processing
    - Estep
    - Mstep
    - post_processing
    - register_result
    - check_stopping_criteria
    - finalization
    
    When the method run_deconvolution is called in ImageDeconvolution class, 
    the iteration method in this class is called for each iteration.

    Attributes
    ----------
    initial_model: :py:class:`cosipy.image_deconvolution.ModelBase` or its subclass
    dataset: list of :py:class:`cosipy.image_deconvolution.ImageDeconvolutionDataInterfaceBase` or its subclass
    parameter : py:class:`yayc.Configurator`
    results: list of results
    dict_bkg_norm: the dictionary of background normalizations
    dict_dataset_indexlist_for_bkg_models: the indices of data corresponding to each background model in the dataset
    """

    def __init__(self, initial_model, dataset, mask, parameter):

        self.initial_model = initial_model
        self.dataset = dataset
        self.mask = mask 
        self.parameter = parameter 
        self.results = []

        # background normalization
        self.dict_bkg_norm = {}
        self.dict_dataset_indexlist_for_bkg_models = {}
        for data in self.dataset:
            for key in data.keys_bkg_models():
                if not key in self.dict_bkg_norm.keys():
                    self.dict_bkg_norm[key] = 1.0
                    self.dict_dataset_indexlist_for_bkg_models[key] = []
        
        for key in self.dict_dataset_indexlist_for_bkg_models.keys():
            for index, data in enumerate(self.dataset):
                if key in data.keys_bkg_models():
                    self.dict_dataset_indexlist_for_bkg_models[key].append(index)

        logger.debug(f'dict_bkg_norm: {self.dict_bkg_norm}')
        logger.debug(f'dict_dataset_indexlist_for_bkg_models: {self.dict_dataset_indexlist_for_bkg_models}')

        # minimum flux
        self.minimum_flux = parameter.get('minimum_flux:value', 0.0)

        minimum_flux_unit = parameter.get('minimum_flux:unit', initial_model.unit)
        if minimum_flux_unit is not None:
            self.minimum_flux = self.minimum_flux*u.Unit(minimum_flux_unit)

        # parameters of the iteration
        self.iteration_count = 0
        self.iteration_max = parameter.get('iteration_max', 1)

    @abstractmethod
    def initialization(self):
        """
        initialization before running the image deconvolution
        """
        raise NotImplementedError

    @abstractmethod
    def pre_processing(self):
        """
        pre-processing for each iteration
        """
        raise NotImplementedError

    @abstractmethod
    def Estep(self):
        """
        E-step. 
        In this step, only self.expectation_list should be updated.
        If it will be performed in other step, typically post_processing() or check_stopping_criteria(),
        this step can be skipped.
        """
        raise NotImplementedError

    @abstractmethod
    def Mstep(self):
        """
        M-step. 
        In this step, only self.delta_model should be updated.
        If you want to apply some operations to self.delta_model,
        these should be performed in post_processing().
        """
        raise NotImplementedError

    @abstractmethod
    def post_processing(self):
        """
        Post-processing for each iteration. 
        In this step, if needed, you can apply some filters to self.delta_model and set it as self.processed_delta_model.
        Then, the updated model should be calculated as self.model.
        For example, Gaussian smoothing can be applied to self.delta_model in this step.
        """
        raise NotImplementedError

    @abstractmethod
    def register_result(self):
        """
        Register results at the end of each iteration. 
        Users can define what kinds of values are stored in this method.
        """
        raise NotImplementedError

    @abstractmethod
    def check_stopping_criteria(self) -> bool:
        """
        Check if the iteration process should be continued or stopped.
        When it returns True, the iteration will stopped.
        """
        raise NotImplementedError

    @abstractmethod
    def finalization(self):
        """
        finalization after running the image deconvolution
        """
        raise NotImplementedError

### A subclass should not override the methods below. ###

    def iteration(self):
        """
        Perform one iteration of image deconvolution.
        This method should not be overrided in subclasses.
        """
        self.iteration_count += 1

        logger.info(f"## Iteration {self.iteration_count}/{self.iteration_max} ##")

        logger.info("<< Pre-processing >>")
        self.pre_processing()

        logger.info("<< E-step >>")
        self.Estep()

        logger.info("<< M-step >>")
        self.Mstep()
            
        logger.info("<< Post-processing >>")
        self.post_processing()

        logger.info("<< Registering Result >>")
        self.register_result()

        logger.info("<< Checking Stopping Criteria >>")
        stop_iteration = self.check_stopping_criteria()
        logger.info("--> {}".format("Stop" if stop_iteration else "Continue"))

        return stop_iteration

    def calc_expectation_list(self, model, dict_bkg_norm = None, almost_zero = 1e-12):
        """
        Calculate a list of expected count histograms corresponding to each data in the registered dataset.

        Parameters
        ----------
        model: :py:class:`cosipy.image_deconvolution.ModelBase` or its subclass
            Model
        dict_bkg_norm : dict, default None
            background normalization for each background model, e.g, {'albedo': 0.95, 'activation': 1.05}
        almost_zero : float, default 1e-12
            In order to avoid zero components in extended count histogram, a tiny offset is introduced.
            It should be small enough not to effect statistics.

        Returns
        -------
        list of :py:class:`histpy.Histogram`
            List of expected count histograms
        """
        
        return [data.calc_expectation(model, dict_bkg_norm = dict_bkg_norm, almost_zero = almost_zero) for data in self.dataset]

    def calc_log_likelihood_list(self, expectation_list):
        """
        Calculate a list of log-likelihood from each data in the registered dataset and the corresponding given expected count histogram.

        Parameters
        ----------
        expectation_list : list of :py:class:`histpy.Histogram`
            List of expected count histograms

        Returns
        -------
        list of float
            List of Log-likelihood
        """

        return [data.calc_log_likelihood(expectation) for data, expectation in zip(self.dataset, expectation_list)]

    def calc_summed_exposure_map(self):
        """
        Calculate a list of exposure maps from the registered dataset.

        Returns
        -------
        :py:class:`histpy.Histogram`
        """

        return self._histogram_sum([ data.exposure_map for data in self.dataset ])

    def calc_summed_bkg_model(self, key):
        """
        Calculate the sum of histograms for a given background model in the registered dataset.

        Parameters
        ----------
        key: str
            Background model name

        Returns
        -------
        float
        """
        
        indexlist = self.dict_dataset_indexlist_for_bkg_models[key]

        return sum([self.dataset[i].summed_bkg_model(key) for i in indexlist])

    def calc_summed_T_product(self, dataspace_histogram_list):
        """
        For each data in the registered dataset, the product of the corresponding input histogram with the transonse of the response function is computed.
        Then, this method returns the sum of all of the products.

        Parameters
        ----------
        dataspace_histogram_list: list of :py:class:`histpy.Histogram`

        Returns
        -------
        :py:class:`histpy.Histogram`
        """

        return self._histogram_sum([data.calc_T_product(hist)
                                    for data, hist in zip(self.dataset, dataspace_histogram_list)])

    def calc_summed_bkg_model_product(self, key, dataspace_histogram_list):
        """
        For each data in the registered dataset, the product of the corresponding input histogram with the specified background model is computed.
        Then, this method returns the sum of all of the products.

        Parameters
        ----------
        key: str
            Background model name
        dataspace_histogram_list: list of :py:class:`histpy.Histogram`

        Returns
        -------
        flaot
        """

        indexlist = self.dict_dataset_indexlist_for_bkg_models[key]

        return sum(
            self.dataset[i].calc_bkg_model_product(key = key, dataspace_histogram = dataspace_histogram_list[i])
            for i in indexlist
        )

    @staticmethod
    def _histogram_sum(hlist):
        """
        Sum a list of Histograms.  If only one input, just return it.
        """
        if len(hlist) == 1:
            return hlist[0]
        else:
            result = hlist[0].copy()
            for h in hlist[1:]:
                result += h
            return result

    def save_histogram(self, filename, counter_name, histogram_key, only_final_result = False):

        # save last result
        self.results[-1][histogram_key].write(filename, name = 'result', overwrite = True)

        # save all results
        if not only_final_result:

            for result in self.results:

                counter = result[counter_name]

                result[histogram_key].write(filename, name = f'{counter_name}{counter}', overwrite = True)

    def save_results_as_fits(self, filename, counter_name, values_key_name_format, dicts_key_name_format, lists_key_name_format):

        hdu_list = []

        # primary HDU
        primary_hdu = fits.PrimaryHDU()

        hdu_list.append(primary_hdu)

        # counter
        col_counter = fits.Column(name=counter_name, array=[int(result[counter_name]) for result in self.results], format = 'K') #64bit integer

        # values
        for key, name, fits_format in values_key_name_format:

            col_value = fits.Column(name=key, array=[result[key] for result in self.results], format=fits_format)

            hdu = fits.BinTableHDU.from_columns([col_counter, col_value])

            hdu.name = name

            hdu_list.append(hdu)

        # dictionary
        for key, name, fits_format in dicts_key_name_format:

            dict_keys = list(self.results[0][key].keys())

            chunk_size = 998 # when the number of columns >= 1000, the fits file may not be saved.
            for i_chunk, chunked_dict_keys in enumerate([dict_keys[i:i+chunk_size] for i in range(0, len(dict_keys), chunk_size)]):

                cols_dict = [fits.Column(name=dict_key, array=[result[key][dict_key] for result in self.results], format=fits_format) for dict_key in chunked_dict_keys]

                hdu = fits.BinTableHDU.from_columns([col_counter] + cols_dict)

                hdu.name = name

                if i_chunk != 0:
                    hdu.name = name + f"{i_chunk}"

                hdu_list.append(hdu)

        # list
        for key, name, fits_format in lists_key_name_format:

            cols_list = [fits.Column(name=f"{self.dataset[i].name}", array=[result[key][i] for result in self.results], format=fits_format) for i in range(len(self.dataset))]

            hdu = fits.BinTableHDU.from_columns([col_counter] + cols_list)

            hdu.name = name

            hdu_list.append(hdu)

        # write
        fits.HDUList(hdu_list).writeto(filename, overwrite=True)
