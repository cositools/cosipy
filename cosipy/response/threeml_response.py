import copy
from typing import Dict

from docutils.io import InputError
from numba.typed.dictobject import DictModel

import numpy as np
import astropy.units as u
from astropy.units import Quantity

from cosipy.interfaces import BinnedThreeMLModelFoldingInterface, BinnedThreeMLSourceResponseInterface, \
    BinnedDataInterface, DataInterface, ThreeMLSourceResponseInterface

from astromodels import Model, PointSource, ExtendedSource

from histpy import Axes, Histogram

__all__ = ["BinnedThreeMLModelFolding"]


def _add_broadcast(accumulator: Histogram, contribution: Histogram) -> None:
    """
    Add `contribution` into `accumulator` in place, broadcasting
    `contribution` across any extra axes that `accumulator` has but
    `contribution` doesn't.

    This supports source/background responses whose expectation is
    independent of an extra segment axis present in the data (e.g. a
    'Depth' axis along which only the background varies, while the
    source response itself does not depend on it).
    """

    if accumulator.axes == contribution.axes:
        accumulator += contribution
        return

    extra_labels = [l for l in contribution.axes.labels if l not in accumulator.axes.labels]

    if extra_labels:
        raise ValueError(f"Source response has axes {extra_labels} not "
                         "present in the data/expectation axes.")

    missing_labels = [l for l in accumulator.axes.labels if l not in contribution.axes.labels]

    if not missing_labels:
        raise ValueError("Axes mismatch: same labels, different binning.")

    contents = contribution.contents

    if contribution.is_sparse:
        contents = contents.todense()

    insert_positions = sorted(accumulator.axes.label_to_index(l) for l in missing_labels)

    for pos in insert_positions:
        contents = np.expand_dims(contents, axis=pos)

    unit = contribution.unit if contribution.unit is not None else u.dimensionless_unscaled

    accumulator += Quantity(contents, unit, copy=False)

class ThreeMLModelFoldingCacheSourceResponsesMixin:
    """
    Avoid duplicating code that is the same for the binned and unbinned case

    Needs:
    self._model,
    """
    _model: Model
    _source_responses: Dict[str, ThreeMLSourceResponseInterface]
    _psr: ThreeMLSourceResponseInterface
    _esr: ThreeMLSourceResponseInterface
    _cached_model_dict: dict

    def _cache_source_responses(self):
        """
        Create a copy of the PSR and ESR for each source

        Returns True if there was any update

        Updates _cached_model_dict and _source_responses
        """

        if self._model is None:
            raise RuntimeError("Call set_model() first.")

        # See this issue for the caveats of comparing models
        # https://github.com/threeML/threeML/issues/645
        current_model_dict = self._model.to_dict()

        # TODO: currently Model.__eq__ seems broken. It returns. True even
        #  if the internal parameters changed. Caching the expected value
        #  is not implemented. Remove the "False and" when fixed
        if self._cached_model_dict is not None and self._cached_model_dict == current_model_dict:
            # Nothing to do
            return False

        # This accounts for the possibility of some sources being added or
        # removed from the model.
        new_source_responses = {}

        for name, source in self._model.sources.items():

            if name in self._source_responses:
                # Use cache
                new_source_responses[name] = self._source_responses[name]
                continue

            if isinstance(source, PointSource):

                if self._psr is None:
                    raise RuntimeError("The model includes a point source but no point source response was provided")

                psr_copy = self._psr.copy()
                psr_copy.set_source(source)
                new_source_responses[name] = psr_copy
            elif isinstance(source, ExtendedSource):

                if self._esr is None:
                    raise RuntimeError("The model includes an extended source but no extended source response was provided")

                esr_copy = self._esr.copy()
                esr_copy.set_source(source)
                new_source_responses[name] = esr_copy
            else:
                raise RuntimeError(f"The model contains the source {name} "
                                   f"of type {type(source)}. I don't know "
                                   "how to handle it!")

        self._source_responses = new_source_responses

        # See this issue for the caveats of comparing models
        # https://github.com/threeML/threeML/issues/645
        self._cached_model_dict = current_model_dict

        return True

class BinnedThreeMLModelFolding(BinnedThreeMLModelFoldingInterface, ThreeMLModelFoldingCacheSourceResponsesMixin):

    def __init__(self,
                 data: BinnedDataInterface,
                 point_source_response:BinnedThreeMLSourceResponseInterface = None,
                 extended_source_response: BinnedThreeMLSourceResponseInterface = None):
        """

        Parameters
        ----------
        point_source_response:
            Response for :class:`astromodels.sources.PointSource`s.
            It can be None is you don't plan to use it for point sources.
        extended_source_response
            Response for :class:`astromodels.sources.ExtendedSource`s
            It can be None is you don't plan to use it for extended sources.
        """

        # Interface inputs
        self._model = None

        # Implementation inputs
        self._psr = point_source_response
        self._esr = extended_source_response

        if point_source_response is None and extended_source_response is None:
            raise RuntimeError("Provide PSR and/or ESR")

        axes = None

        if point_source_response is not None:
            axes = point_source_response.axes

        if extended_source_response is not None:
            if axes is None:
                axes = extended_source_response.axes
            else:
                if axes != extended_source_response.axes:
                    raise RuntimeError("The PSR and ESR expectation should have the same axes")

        self._axes = axes

        # Cache
        # Prevent unnecessary calculations and new memory allocations
        # See this issue for the caveats of comparing models
        # https://github.com/threeML/threeML/issues/645
        self._cached_model_dict = None
        self._source_responses = {}
        self._expectation = Histogram(data.axes)

    @property
    def axes(self):
        return self._axes

    def set_model(self, model: Model):
        """
        You need to call set_data() first.

        Parameters
        ----------
        model

        Returns
        -------

        """

        self._model = model

    def expectation(self, copy:bool = True)->Histogram:
        """

        Parameters
        ----------
        data
        copy

        Returns
        -------

        """

        if self._model is None:
            raise RuntimeError("Call set_data() and set_model() first")

        # Create a copy of the PSR and ESR for each source
        model_changed = self._cache_source_responses()

        if not model_changed:
            if copy:
                return self._expectation.copy()
            else:
                return self._expectation
        else:
            self._expectation.clear()

        # Convolve all sources with the response
        for source_name,psr in self._source_responses.items():
            _add_broadcast(self._expectation, psr.expectation())

        if copy:
            return self._expectation.copy()
        else:
            return self._expectation
