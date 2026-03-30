import numpy as np
import astropy.io.fits as fits
import functools
from abc import ABC, abstractmethod
import logging
logger = logging.getLogger(__name__)

from ..constants import CHUNK_SIZE_FITS, DEFAULT_ITERATION_MAX
from ..utils import _to_float

class DeconvolutionAlgorithmBase(ABC):
    """
    A base class for image deconvolution algorithms.
    Subclasses should override these methods:

    - initialization 
    - pre_processing
    - processing_core
    - post_processing
    - register_result
    - check_stopping_criteria
    - finalization
    
    When the method run_deconvolution is called in ImageDeconvolution class, 
    the iteration method in this class is called for each iteration.

    Attributes
    ----------
    initial_model: :py:class:`cosipy.image_deconvolution.ModelBase` or its subclass
    dataset: :py:class:`cosipy.image_deconvolution.DataInterfaceCollection`
    parameter : py:class:`yayc.Configurator`
    results: list of results
    dict_bkg_norm: the dictionary of background normalizations
    """

    def __init__(self, initial_model, dataset, mask, parameter):

        self.initial_model = initial_model
        self.dataset = dataset
        self.mask = mask 
        self.parameter = parameter 
        self.results = []

        # background normalization
        self.dict_bkg_norm = {key: 1.0 for key in dataset.keys_bkg_models()}

        logger.debug(f'dict_bkg_norm: {self.dict_bkg_norm}')

        # parameters of the iteration
        self.iteration_count = 0
        self.iteration_max = parameter.get('iteration_max', DEFAULT_ITERATION_MAX)

        # parameter summary (subclasses can append to this list)
        self._parameter_summary = [
            ("iteration_max", self.iteration_max),
        ]

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
    def processing_core(self):
        """
        Core processing for each iteration.
        
        This method should implement the main algorithm logic.
        For EM-based algorithms, this typically includes:
        - E-step: Expectation calculation
        - M-step: Maximization/update
        
        Subclasses must implement this method.
        """
        raise NotImplementedError

    @abstractmethod
    def Estep(self):
        """
        E-step: Expectation calculation
        """
        raise NotImplementedError

    @abstractmethod
    def Mstep(self):
        """
        M-step: Maximization/update
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

    def iteration(self):
        """
        Perform one iteration of image deconvolution.
        This method should not be overrided in subclasses.
        """
        self.iteration_count += 1

        logger.info(f"## Iteration {self.iteration_count}/{self.iteration_max} ##")

        logger.info("<< Pre-processing >>")
        self.pre_processing()

        logger.info("<< Processing Core>>")
        self.processing_core()
            
        logger.info("<< Post-processing >>")
        self.post_processing()

        logger.info("<< Registering Result >>")
        self.register_result()

        logger.info("<< Checking Stopping Criteria >>")
        stop_iteration = self.check_stopping_criteria()
        logger.info("-> {}".format("Stop" if stop_iteration else "Continue"))

        return stop_iteration

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

            chunk_size = CHUNK_SIZE_FITS # when the number of columns >= 1000, the fits file may not be saved.
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

    def _save_standard_results(self, counter_name, histogram_keys, fits_filename, 
                               values_key_name_format=None, dicts_key_name_format=None, lists_key_name_format=None):
        """
        Save standard results including histograms and FITS files.
        
        Parameters
        ----------
        counter_name : str
            Name of the counter (e.g., "iteration")
        histogram_keys : list of tuple
            List of (key, filename, only_final_result) for histograms to save.
        fits_filename : str
            Path to the FITS file.
        values_key_name_format : list of tuple, optional
            List of (key, name, fits_format) for single values to save in FITS.
        dicts_key_name_format : list of tuple, optional
            List of (key, name, fits_format) for dictionaries to save in FITS.
        lists_key_name_format : list of tuple, optional
            List of (key, name, fits_format) for lists to save in FITS.
        """
        # Save histograms
        for key, filename, only_final_result in histogram_keys:
            self.save_histogram(
                filename = filename,
                counter_name = counter_name,
                histogram_key = key,
                only_final_result = only_final_result
            )
        
        # Save FITS file (use default if not specified)
        self.save_results_as_fits(
            filename = fits_filename,
            counter_name = counter_name,
            values_key_name_format = values_key_name_format if values_key_name_format is not None else [],
            dicts_key_name_format = dicts_key_name_format if dicts_key_name_format is not None else [],
            lists_key_name_format = lists_key_name_format if lists_key_name_format is not None else []
        )

    def _show_parameters(self):
        """
        Log all parameters registered in _parameter_summary.
        Subclasses should append their parameters to self._parameter_summary
        in their __init__, then call this method at the end of __init__.

        Example
        -------
        # In a subclass __init__:
        self._parameter_summary += [
            ("bkg_norm_optimization_enabled", self.bkg_norm_optimization_enabled),
            ("minimum_flux", self.minimum_flux),
        ]
        self._show_parameters()
        """

        logger.info(f"[{self.__class__.__name__}]")
        for name, value in self._parameter_summary:
            logger.info(f"  {name}: {value}")
