"""
Acceleration via line search: finds the accel_factor in [1, accel_factor_max]
that maximises the log-likelihood.

For the model-only (1-D) case, Brent's method (scipy.optimize.minimize_scalar)
is used. When background normalization is optimised independently (2-D), either
a gradient-based method (L-BFGS-B) or a grid search can be selected via the
``search_method`` parameter.

YAML configuration example
---------------------------
acceleration:
    activate: true
    algorithm: LineSearch
    accel_factor_max: 10.0       # hard upper bound for model accel_factor
    accel_bkg_norm:
        activate: false          # whether to accelerate bkg_norm at all
        independent: false       # if true, bkg gets its own accel_factor
        max: 10.0                # hard upper bound for bkg accel_factor (independent mode only)
        search_method: gradient  # "gradient" or "grid" (independent mode only)
        grid_n: 10               # grid points per axis (grid mode only)
"""

import numpy as np
import logging
from scipy.optimize import minimize_scalar, minimize

from .accelerator_base import AcceleratorBase, AcceleratorResult

logger = logging.getLogger(__name__)

DEFAULT_ACCEL_FACTOR_MAX = 10.0
DEFAULT_GRID_N           = 10


class LineSearchAccelerator(AcceleratorBase):
    """
    RL acceleration by line search over accel_factor.

    Three bkg_norm modes are supported:

    accel_bkg_norm.activate=False (default)
        bkg_norm is NOT accelerated; only the model is scaled.
        Line search is 1-D (Brent's method).

    accel_bkg_norm.activate=True, independent=False
        bkg_norm is scaled by the same accel_factor as the model.
        Line search is still 1-D (Brent's method).

    accel_bkg_norm.activate=True, independent=True
        bkg_norm gets its own accel_factor optimised independently.
        Line search becomes 2-D. Two search methods are available:
        - "gradient": L-BFGS-B (fast, may miss global optimum)
        - "grid"    : exhaustive grid search over grid_n x grid_n points
                      (slower but more robust)

    In all cases the search is bounded by accel_factor_max (and
    accel_bkg_norm.max for the independent bkg case).
    """

    n_em_steps_required  = 1
    logged_result_fields = [("accel_factor", "D"), ("accel_factor_bkg", "D")]

    def __init__(self, parameter):
        super().__init__(parameter)

        self.accel_factor_max           = float(parameter.get("accel_factor_max", DEFAULT_ACCEL_FACTOR_MAX))
        self.accel_bkg_norm             = bool(parameter.get("accel_bkg_norm:activate", False))
        self.accel_bkg_norm_independent = bool(parameter.get("accel_bkg_norm:independent", False))
        self.accel_factor_bkg_max       = float(parameter.get("accel_bkg_norm:max", DEFAULT_ACCEL_FACTOR_MAX))
        self.bkg_search_method          = parameter.get("accel_bkg_norm:search_method", "gradient")
        self.bkg_grid_n                 = int(parameter.get("accel_bkg_norm:grid_n", DEFAULT_GRID_N))

        if self.bkg_search_method not in ("gradient", "grid"):
            raise ValueError(
                f'Unknown search_method "{self.bkg_search_method}". '
                f'Available: "gradient", "grid"'
            )

        logger.info(
            f"[LineSearchAccelerator]"
            f"\n  accel_factor_max: {self.accel_factor_max}"
            f"\n  accel_bkg_norm: {self.accel_bkg_norm}"
            f"\n  accel_bkg_norm_independent: {self.accel_bkg_norm_independent}"
            f"\n  accel_factor_bkg_max: {self.accel_factor_bkg_max}"
            + (f"\n  search_method: {self.bkg_search_method}"
               f"\n  grid_n: {self.bkg_grid_n}"
               if self.accel_bkg_norm and self.accel_bkg_norm_independent else "")
        )

    def compute(self, em_results, dataset, mask) -> AcceleratorResult:

        before = em_results[0]
        after  = em_results[1]

        delta_model    = after.model - before.model
        delta_bkg_norm = {key: after.dict_bkg_norm[key] - before.dict_bkg_norm[key]
                          for key in before.dict_bkg_norm}

        # Upper bound from non-negativity constraint on model
        alpha_max = min(self._compute_accel_factor_max(delta_model, before.model, mask), 
                        self.accel_factor_max)

        # Upper bound from non-negativity constraint on bkg_norm
        alpha_bkg_max = min(self._compute_accel_factor_bkg_max(delta_bkg_norm, before.dict_bkg_norm), 
                            self.accel_factor_bkg_max)

        if alpha_max <= 1.0:
            # No room to accelerate
            return AcceleratorResult(
                model         = after.model,
                dict_bkg_norm = after.dict_bkg_norm,
                extras        = {"accel_factor": 1.0, "accel_factor_bkg": 1.0},
            )

        # --- define log-likelihood as a function of accel factors ---

        def _ll_1d(alpha):
            """LH as a function of a single accel_factor (model, and optionally bkg)."""
            new_model = before.model + delta_model * alpha

            if self.accel_bkg_norm and not self.accel_bkg_norm_independent:
                new_dict_bkg_norm = {key: before.dict_bkg_norm[key] + delta_bkg_norm[key] * alpha
                                     for key in before.dict_bkg_norm}
            else:
                new_dict_bkg_norm = after.dict_bkg_norm

            src_exp_list = [
                src_before + (src_after - src_before) * alpha
                for src_before, src_after
                in zip(before.source_expectation_list, after.source_expectation_list)
            ]
            bkg_exp_list  = dataset.calc_bkg_expectation_list(new_dict_bkg_norm)
            exp_list      = dataset.combine_expectation_list(src_exp_list, bkg_exp_list)
            return float(np.sum(dataset.calc_log_likelihood_list(exp_list)))

        def _ll_2d(alpha_m, alpha_b):
            """LH as a function of independent model and bkg accel_factors."""
            new_model = before.model + delta_model * alpha_m
            new_dict_bkg_norm = {key: before.dict_bkg_norm[key] + delta_bkg_norm[key] * alpha_b
                                 for key in before.dict_bkg_norm}
            src_exp_list = [
                src_before + (src_after - src_before) * alpha_m
                for src_before, src_after
                in zip(before.source_expectation_list, after.source_expectation_list)
            ]
            bkg_exp_list = dataset.calc_bkg_expectation_list(new_dict_bkg_norm)
            exp_list     = dataset.combine_expectation_list(src_exp_list, bkg_exp_list)
            return float(np.sum(dataset.calc_log_likelihood_list(exp_list)))

        # --- run line search ---

        ll_after = _ll_1d(1.0)

        if self.accel_bkg_norm and self.accel_bkg_norm_independent:
            # 2-D search
            accel_factor, accel_factor_bkg = self._search_2d(_ll_2d, alpha_max, alpha_bkg_max)
            ll_opt = _ll_2d(accel_factor, accel_factor_bkg)
            new_dict_bkg_norm = {key: before.dict_bkg_norm[key] + delta_bkg_norm[key] * accel_factor_bkg
                                 for key in before.dict_bkg_norm}
        else:
            # 1-D optimisation (Brent's method)
            # In same-factor mode, bkg is also constrained, so use the tighter bound
            alpha_1d_max = min(alpha_max, alpha_bkg_max) if self.accel_bkg_norm else alpha_max

            opt = minimize_scalar(
                lambda a: -_ll_1d(a),
                bounds=(1.0, alpha_1d_max),
                method="bounded",
            )
            accel_factor     = float(opt.x)
            accel_factor_bkg = accel_factor if (self.accel_bkg_norm and not self.accel_bkg_norm_independent) else 1.0
            ll_opt           = -float(opt.fun)

            if self.accel_bkg_norm and not self.accel_bkg_norm_independent:
                new_dict_bkg_norm = {key: before.dict_bkg_norm[key] + delta_bkg_norm[key] * accel_factor
                                     for key in before.dict_bkg_norm}
            else:
                new_dict_bkg_norm = after.dict_bkg_norm

        # Fall back to accel_factor=1 if line search did not improve LH
        if ll_opt < ll_after:
            logger.debug(
                f"[LineSearchAccelerator] line search (alpha={accel_factor:.3f}, alpha_bkg={accel_factor_bkg:.3f}) "
                f"did not improve LH ({ll_opt:.6f} < {ll_after:.6f}). Falling back to accel_factor=1."
            )
            accel_factor      = 1.0
            accel_factor_bkg  = 1.0
            new_model         = after.model
            new_dict_bkg_norm = after.dict_bkg_norm
        else:
            new_model = before.model + delta_model * accel_factor

        logger.debug(f"[LineSearchAccelerator] accel_factor={accel_factor:.4f}, accel_factor_bkg={accel_factor_bkg:.4f}")

        return AcceleratorResult(
            model         = new_model,
            dict_bkg_norm = new_dict_bkg_norm,
            extras        = {"accel_factor": accel_factor, "accel_factor_bkg": accel_factor_bkg},
        )

    def _search_2d(self, ll_2d_func, alpha_max, alpha_bkg_max):
        """
        2-D search for the optimal (accel_factor, accel_factor_bkg).

        Parameters
        ----------
        ll_2d_func : callable
            Function of (alpha_m, alpha_b) returning log-likelihood.
        alpha_max : float
            Upper bound for model accel_factor.
        alpha_bkg_max : float
            Upper bound for bkg accel_factor.

        Returns
        -------
        (float, float)
            Optimal (accel_factor, accel_factor_bkg).
        """

        if self.bkg_search_method == "grid":
            return self._grid_search_2d(ll_2d_func, alpha_max, alpha_bkg_max)
        else:
            # gradient (L-BFGS-B)

            # 1. Optimize source to get a good initial point
            opt_s = minimize_scalar(
                lambda s: -ll_2d_func(s, 1.0),
                bounds=(1.0, alpha_max),
                method="bounded",
            )
            alpha_init = float(opt_s.x)
            
            # 2. Optimize bkg to get a good initial point
            opt_b = minimize_scalar(
                lambda b: -ll_2d_func(1.0, b),
                bounds=(1.0, alpha_bkg_max),
                method="bounded",
            )
            alpha_bkg_init = float(opt_b.x)
            
            # 3. Optimize both with both initial points
            opt = minimize(
                lambda x: -ll_2d_func(x[0], x[1]),
                x0=[alpha_init, alpha_bkg_init],
                bounds=[(1.0, alpha_max), (1.0, alpha_bkg_max)],
                method="L-BFGS-B",
            )

            accel_factor     = float(opt.x[0])
            accel_factor_bkg = float(opt.x[1])

            return float(opt.x[0]), float(opt.x[1])

    def _grid_search_2d(self, ll_2d_func, alpha_max, alpha_bkg_max):
        """
        Exhaustive grid search over [1, alpha_max] x [1, alpha_bkg_max].

        Parameters
        ----------
        ll_2d_func : callable
        alpha_max : float
        alpha_bkg_max : float

        Returns
        -------
        (float, float)
            Optimal (accel_factor, accel_factor_bkg).
        """

        alphas_model = np.linspace(1.0, alpha_max,     self.bkg_grid_n)
        alphas_bkg   = np.linspace(1.0, alpha_bkg_max, self.bkg_grid_n)

        best_ll      = -np.inf
        best_alpha_m = 1.0
        best_alpha_b = 1.0

        for am in alphas_model:
            for ab in alphas_bkg:
                ll = ll_2d_func(am, ab)
                if ll > best_ll:
                    best_ll      = ll
                    best_alpha_m = am
                    best_alpha_b = ab

        logger.debug(
            f"[LineSearchAccelerator] grid search best: "
            f"alpha={best_alpha_m:.4f}, alpha_bkg={best_alpha_b:.4f}, ll={best_ll:.6f}"
        )

        return best_alpha_m, best_alpha_b
