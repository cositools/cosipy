"""
Provides DataInterfaceCollection: a class that encapsulates a list of
ImageDeconvolutionDataInterfaceBase objects and exposes the collective
operations (expectation, log-likelihood, T-product, background operations)
that were previously scattered as methods of DeconvolutionAlgorithmBase.
"""

from collections.abc import Sequence
import numpy as np
import logging
from typing import Union

logger = logging.getLogger(__name__)


from ..utils import _to_float


class DataInterfaceCollection(Sequence):
    """
    Wraps a list of :py:class:`ImageDeconvolutionDataInterfaceBase` objects
    and provides collective statistical/response operations.

    Parameters
    ----------
    dataset : list of ImageDeconvolutionDataInterfaceBase
        The registered dataset.

    Attributes
    ----------
    dataset : list
        The underlying list of data interface objects.
    """

    def __init__(self, dataset: list):
        self._dataset = dataset

        self._dict_dataset_indexlist_for_bkg_models = {}

        for idx, data in enumerate(self._dataset):
            for key in data.keys_bkg_models():
                if key in self._dict_dataset_indexlist_for_bkg_models:
                    self._dict_dataset_indexlist_for_bkg_models[key].append(idx)
                else:
                    self._dict_dataset_indexlist_for_bkg_models[key] = [idx]

    # Basic access
    @property
    def dataset(self) -> list:
        return self._dataset

    def keys_bkg_models(self) -> list:
        return list(self._dict_dataset_indexlist_for_bkg_models.keys())

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx):
        return self._dataset[idx]
    

    # Expectation
    def calc_expectation_list(self, model, dict_bkg_norm = Union[dict,None]) -> list:
        """
        Calculate a list of expected count histograms corresponding to each data in the registered dataset.

        Parameters
        ----------
        model: :py:class:`cosipy.image_deconvolution.ModelBase` or its subclass
            Model
        dict_bkg_norm : dict, default None
            background normalization for each background model, e.g, {'albedo': 0.95, 'activation': 1.05}

        Returns
        -------
        list of :py:class:`histpy.Histogram`
            List of expected count histograms
        """
        
        return [
            data.calc_expectation(model, dict_bkg_norm) 
            for data in self._dataset
        ]

    def calc_source_expectation_list(self, model) -> list:
        """
        Compute source-only expected counts for every dataset entry.

        Returns
        -------
        list of histpy.Histogram
        """
        return [
            data.calc_source_expectation(model)
            for data in self._dataset
        ]

    def calc_bkg_expectation_list(self, dict_bkg_norm) -> list:
        """
        Compute background-only expected counts for every dataset entry.

        Returns
        -------
        list of histpy.Histogram
        """
        return [
            data.calc_bkg_expectation(dict_bkg_norm)
            for data in self._dataset
        ]

    def combine_expectation_list(self, expectation_list_src, expectation_list_bkg) -> list:
        """
        Sum source and background expectation lists element-wise.

        Returns
        -------
        list of histpy.Histogram
        """
        return [
            src + bkg
            for src, bkg in zip(expectation_list_src, expectation_list_bkg)
        ]

    # Log-likelihood
    def calc_log_likelihood_list(self, expectation_list: list) -> list:
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

        return [_to_float(data.calc_log_likelihood(expectation)) for data, expectation in zip(self._dataset, expectation_list)]

    def calc_total_log_likelihood(self, expectation_list: list) -> float:
        """
        Convenience: sum of all per-dataset log-likelihoods.
        """

        return float(np.sum(self.calc_log_likelihood_list(expectation_list)))

    # Response-transpose products
    def calc_summed_T_product(self, dataspace_histogram_list: list):
        """
        For each data in the registered dataset, the product of the corresponding input histogram with the transpose of the response function is computed.
        Then, this method returns the sum of all of the products.

        Parameters
        ----------
        dataspace_histogram_list: list of :py:class:`histpy.Histogram`

        Returns
        -------
        :py:class:`histpy.Histogram`
        """

        return self._histogram_sum([data.calc_T_product(hist)
                                    for data, hist in zip(self._dataset, dataspace_histogram_list)])

    # Exposure map
    def calc_summed_exposure_map(self):
        """
        Calculate a list of exposure maps from the registered dataset.

        Returns
        -------
        :py:class:`histpy.Histogram`
        """

        return self._histogram_sum([data.exposure_map for data in self._dataset])

    def calc_summed_bkg_model(self, key: str) -> float:
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
        
        indexlist = self._dict_dataset_indexlist_for_bkg_models[key]

        return sum([self._dataset[i].summed_bkg_model(key) for i in indexlist])

    def calc_summed_bkg_model_product(self, key: str, dataspace_histogram_list: list) -> float:
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

        if len(dataspace_histogram_list) != len(self._dataset):
            logger.error(f"The length of the input histogram list ({len(dataspace_histogram_list)}) is not equal to that of the dataset ({len(self._dataset)}).")
            raise ValueError

        indexlist = self._dict_dataset_indexlist_for_bkg_models[key]

        return sum(
            self._dataset[i].calc_bkg_model_product(key = key, dataspace_histogram = dataspace_histogram_list[i])
            for i in indexlist
        )

    # Utility
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
