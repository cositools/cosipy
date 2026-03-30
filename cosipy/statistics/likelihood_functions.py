import itertools
import logging
import operator
from typing import Optional

from cosipy import UnBinnedData
from cosipy.interfaces.expectation_interface import ExpectationInterface, ExpectationDensityInterface
from cosipy.util.iterables import itertools_batched
from cosipy.util.iterables import asarray

logger = logging.getLogger(__name__)

from cosipy.interfaces import (BinnedLikelihoodInterface,
                               UnbinnedLikelihoodInterface,
                               BinnedDataInterface,
                               BinnedExpectationInterface,
                               BinnedBackgroundInterface, DataInterface, BackgroundInterface, EventDataInterface,
                               BackgroundDensityInterface,
                               )

import numpy as np

__all__ = ['UnbinnedLikelihood',
           'PoissonLikelihood']

class UnbinnedLikelihood(UnbinnedLikelihoodInterface):
    def __init__(self,
                 expectation: ExpectationDensityInterface,
                 batch_size: Optional[int] = None):
        """
        Parameters
        ----------
        expectation : ExpectationDensityInterface
            Object that provides the expected counts and the
            ``expectation_density()`` iterator used to compute the likelihood.

        batch_size : int or None, optional
            Number of density values to process at a time in ``get_log_like()``.
            If None, all values are processed in a single batch.

            This parameter only affects iteration when the expectation density
            is provided as an iterator that is not a numpy array. If it is already
            an array batching is not applied.
        """

        self._expectation = expectation
        self._nobservations = None

        self._batch_size = batch_size

    @property
    def nobservations(self) -> int:
        """
        Calling get_log_like first is faster, since we don't need to loop though the
        events
        """

        if self._nobservations is None:
            self._nobservations = sum(1 for _ in  self._expectation.expectation_density())

        return self._nobservations

    def get_log_like(self) -> float:

        # Total number of events
        ntot = self._expectation.expected_counts()

        # It's faster to compute all log values at once, but requires keeping them in memory
        # Doing it by chunk is a compromise. We might need to adjust the chunk_size
        # Based on the system
        nobservations = 0
        density_log_sum = 0
        
        expectation_density = self._expectation.expectation_density()

        if (self._batch_size is None) or isinstance(expectation_density, np.ndarray):
            chunks = [expectation_density]
        else:
            chunks = itertools_batched(expectation_density, self._batch_size)
            
        for chunk in chunks:
            density = asarray(chunk, dtype=np.float64, force_dtype=False)

            # We don't have to continue
            if density.min() <= 0.0:
                return -np.inf

            nobservations += density.size
            density_log_sum += np.sum(np.log(density))

        self._nobservations = nobservations

        # Log L = -Ntot + sum_i (dN/dOmega)_i
        # (dN/dOmega)_i is the expectation density, not a derivative
        # (dN/dOmega)_i = Ntot*P_i, where P_i is the event probability
        log_like = density_log_sum - ntot

        return log_like


class PoissonLikelihood(BinnedLikelihoodInterface):
    def __init__(self, data:BinnedDataInterface,
                 response:BinnedExpectationInterface,
                 bkg:BinnedBackgroundInterface = None):

        self._data = data
        self._bkg = bkg
        self._response = response

    @property
    def has_bkg(self):
        return self._bkg is not None

    @property
    def nobservations(self) -> int:
        return self._data.data.contents.size

    def get_log_like(self) -> float:

        # Compute expectation including background
        # If we don't have background, we won't modify the expectation, so
        # it's safe to use the internal cache.
        expectation = self._response.expectation(copy = self.has_bkg)

        if self.has_bkg:
            # We won't modify the bkg expectation, so it's safe to use the internal cache
            expectation += self._bkg.expectation(copy = False)

        # Get the arrays
        expectation = expectation.contents
        data = self._data.data.contents

        # Compute the log-likelihood:
        log_like = np.nansum(data * np.log(expectation) - expectation)

        return log_like

