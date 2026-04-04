"""
Acceleration via the maximum safe step: finds the largest accel_factor in
[1, accel_factor_max] such that the updated model remains non-negative
element-wise, then accepts it only if it improves the log-likelihood.

YAML configuration example
---------------------------
acceleration:
    activate: true
    algorithm: MaxStep
    accel_factor_max: 10.0   # hard upper bound for accel_factor
    accel_bkg_norm: false    # if true, bkg_norm is scaled by the same accel_factor
"""

import numpy as np
import logging

from .accelerator_base import AcceleratorBase, AcceleratorResult

from ...utils import _to_float

logger = logging.getLogger(__name__)

DEFAULT_ACCEL_FACTOR_MAX = 10.0


class MaxStepAccelerator(AcceleratorBase):
    """
    Finds the largest accel_factor in [1, accel_factor_max] such that

        model_before + accel_factor * delta_model >= 0   (element-wise)

    If accel_factor does not improve LH, falls back to accel_factor=1.
    accel_factor is stored in extras={"accel_factor": accel_factor}.

    algorithm: MaxStep
    accel_factor_max: 10.0
    accel_bkg_norm: False
    """

    n_em_steps_required = 1
    logged_result_fields = [("accel_factor", "D")]

    def __init__(self, parameter):
        super().__init__(parameter)
        self.accel_factor_max = float(parameter.get("accel_factor_max", DEFAULT_ACCEL_FACTOR_MAX))
        self.accel_bkg_norm = bool(parameter.get("accel_bkg_norm", False))
        logger.info(
            f"[MaxStepAccelerator]"
            f"\n  accel_factor_max: {self.accel_factor_max}"
            f"\n  accel_bkg_norm: {self.accel_bkg_norm}"
            )

    def compute(
        self,
        em_results : list,
        dataset,
        mask,
    ) -> AcceleratorResult:

        before = em_results[0]
        after  = em_results[1]

        delta_model = after.model - before.model

        accel_factor = self._compute_accel_factor(delta_model, before.model, mask)

        if accel_factor > 1.0:
            new_model = before.model + delta_model * accel_factor
            if self.accel_bkg_norm:
                new_dict_bkg_norm = {
                    key: before.dict_bkg_norm[key] + (after.dict_bkg_norm[key] - before.dict_bkg_norm[key]) * accel_factor
                    for key in before.dict_bkg_norm
                }
            else:
                new_dict_bkg_norm = after.dict_bkg_norm

            # LH check: reuse source expectation, recompute bkg only
            source_expectation_list = [
                src_before + (src_after - src_before) * accel_factor
                for src_before, src_after in zip(before.source_expectation_list, after.source_expectation_list)
            ]
            bkg_expectation_list = dataset.calc_bkg_expectation_list(new_dict_bkg_norm)
            expectation_list     = dataset.combine_expectation_list(source_expectation_list, bkg_expectation_list)

            ll_accel = np.sum(dataset.calc_log_likelihood_list(expectation_list))
            ll_after = np.sum(dataset.calc_log_likelihood_list(after.expectation_list))

            if ll_accel < ll_after:
                logger.debug(
                    f"[MaxStepAccelerator] accel_factor={accel_factor:.3f} did not improve LH "
                    f"({ll_accel:.6f} < {ll_after:.6f}). Falling back to accel_factor=1."
                )
                accel_factor      = 1.0
                new_model         = after.model
                new_dict_bkg_norm = after.dict_bkg_norm

        else:
            new_model         = after.model
            new_dict_bkg_norm = after.dict_bkg_norm

        logger.debug(f"[MaxStepAccelerator] accel_factor={accel_factor}")

        return AcceleratorResult(
            model         = new_model,
            dict_bkg_norm = new_dict_bkg_norm,
            extras        = {"accel_factor": accel_factor},
        )

    def _compute_accel_factor(self, delta_model, model, mask) -> float:

        accel_factor = min(self._compute_accel_factor_max(delta_model, model, mask), \
                           self.accel_factor_max)

        return accel_factor
