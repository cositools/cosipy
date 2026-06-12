from abc import ABC, abstractmethod
from dataclasses import dataclass 
from typing import Union
import numpy as np
import logging
logger = logging.getLogger(__name__)

from histpy import Histogram

from ...utils import _to_float

@dataclass
class EMStepResult:
    """
    Holds the state and results of one EM step (or the initial "before" state).

    em_results passed to AcceleratorBase.compute() is a list of EMStepResult:
    - em_results[0] : "before" state
    - em_results[1] : result of 1st EM step
    - em_results[2] : result of 2nd EM step (e.g. SQUAREM)
    - ...
    """

    model                   : Histogram
    dict_bkg_norm           : dict
    expectation_list        : Union[list, None] = None
    source_expectation_list : Union[list, None] = None
    bkg_expectation_list    : Union[list, None] = None


@dataclass
class AcceleratorResult:
    """
    Return value of AcceleratorBase.compute().
    """

    model         : Histogram
    dict_bkg_norm : dict
    extras        : Union[dict, None] = None


class AcceleratorBase(ABC):

    n_em_steps_required : int = 1

    # Fields from AcceleratorResult.extras to be saved in results and logged.
    # Each entry is a (field_name, fits_format) tuple, e.g. [("accel_factor", "D")].
    # Algorithm classes (e.g. RichardsonLucyAdvanced) iterate over this list in
    # register_result and finalization, so accelerator-specific keys never need
    # to be hardcoded in the algorithm.
    logged_result_fields: list = []

    def __init__(self, parameter):
        self._parameter = parameter

    @abstractmethod
    def compute(
        self,
        em_results : list,
        dataset,
        mask,
    ) -> AcceleratorResult:
        raise NotImplementedError

    def _compute_accel_factor_max(self, delta_model, model, mask) -> float:
        """
        Compute the maximum allowed accel_factor such that
        model + accel_factor * delta_model >= 0 element-wise.

        Parameters
        ----------
        delta_model : Histogram
            delta of model.
        model : Histogram
            model before the EM step.
        mask : Histogram or None

        Returns
        -------
        float
            Maximum safe accel_factor
        """

        diff = -1 * (model / delta_model).contents
        diff[(diff <= 0) | (delta_model.contents == 0)] = np.inf

        if mask is not None:
            diff[np.invert(mask.contents)] = np.inf

        accel_factor = _to_float(np.min(diff))

        if accel_factor < 1.0:
            accel_factor = 1.0

        return accel_factor

    def _compute_accel_factor_bkg_max(self, delta_bkg_norm, bkg_norm) -> float:
        """
        Compute the maximum allowed accel_factor_bkg such that
        bkg_norm + accel_factor_bkg * delta_bkg_norm >= 0 for all keys.

        For each key, the constraint is active only when delta_bkg_norm < 0.
        The returned value is the minimum across all keys, so a single
        accel_factor_bkg is guaranteed to be safe for all background components.

        Parameters
        ----------
        delta_bkg_norm : dict
            Per-key delta of background normalization.
        bkg_norm : dict
            Per-key background normalization before the EM step.

        Returns
        -------
        float
            Maximum safe accel_factor_bkg
        """

        accel_factor_bkg_max = np.inf

        for key in delta_bkg_norm:
            d = delta_bkg_norm[key]
            if d < 0:
                accel_factor_bkg_max = min(accel_factor_bkg_max, -bkg_norm[key] / d)

        return max(1.0, _to_float(accel_factor_bkg_max))
