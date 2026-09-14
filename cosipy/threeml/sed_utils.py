"""
Utilities for the response-configured arbitrary-bin COSI true-energy SED.

The SED itself is created with ``BinnedSED.from_response``. This module keeps
response diagnostics and profile-likelihood math separate from the spectral
function definition.
"""

import warnings

import numpy as np
from scipy.optimize import brentq

from .binned_sed import BinnedSED


__all__ = [
    "find_unconstrained_sed_bins",
    "freeze_binned_sed_bins",
    "check_binned_sed_response",
    "profile_likelihood_upper_limit",
]


def _dense_contents(hist):
    """Return histogram contents as a dense floating-point ndarray."""
    contents = hist.contents
    if hasattr(contents, "todense"):
        contents = contents.todense()
    return np.asarray(contents, dtype=float)


def find_unconstrained_sed_bins(templates, bin_indices=None, atol=0.0):
    """
    Identify SED bins whose forward-folded source template is zero.

    Parameters
    ----------
    templates : array-like
        Array with shape ``(n_sed_bins, ...)``. The remaining dimensions are
        arbitrary detector-space axes.
    bin_indices : iterable of int, optional
        Labels to return for each template. Defaults to ``0..n_sed_bins-1``.
    atol : float, optional
        A bin is unconstrained when the sum of the absolute template values is
        less than or equal to ``atol``. Default is exactly zero.

    Returns
    -------
    numpy.ndarray
        Labels of unconstrained bins.
    """

    templates = np.asarray(templates, dtype=float)

    if templates.ndim < 2:
        raise ValueError(
            "templates must have shape (n_sed_bins, ...detector axes...)."
        )

    if atol < 0.0:
        raise ValueError("atol must be non-negative.")

    if not np.all(np.isfinite(templates)):
        bad = np.where(
            ~np.all(np.isfinite(templates.reshape(templates.shape[0], -1)), axis=1)
        )[0]
        raise ValueError(
            f"Non-finite values were found in SED templates {bad.tolist()}."
        )

    n_bins = templates.shape[0]

    if bin_indices is None:
        labels = np.arange(n_bins, dtype=int)
    else:
        labels = np.asarray(list(bin_indices), dtype=int)
        if labels.size != n_bins:
            raise ValueError("bin_indices must contain one label per template.")

    totals = np.sum(
        np.abs(templates.reshape(n_bins, -1)),
        axis=1,
    )

    return labels[totals <= atol]


def freeze_binned_sed_bins(spectrum, local_bin_indices):
    """
    Freeze selected BinnedSED normalizations to zero.

    Parameters
    ----------
    spectrum : BinnedSED
        Response-configured SED spectrum.
    local_bin_indices : iterable of int
        Local SED-bin indices in ``0..spectrum.n_bins-1``.

    Returns
    -------
    tuple of int
        Local bin indices that were frozen.
    """

    if not isinstance(spectrum, BinnedSED):
        raise TypeError("spectrum must be a BinnedSED instance.")

    bins = tuple(sorted(set(int(i) for i in local_bin_indices)))

    for i in bins:
        if i < 0 or i >= spectrum.n_bins:
            raise IndexError(
                f"BinnedSED local bin indices must lie in [0, {spectrum.n_bins - 1}]."
            )

        par = getattr(spectrum, f"K{i}")
        par.value = 0.0
        par.free = False

    return bins


def _positive_test_value(parameter, default=1e-6):
    """Choose a positive in-bounds normalization for response diagnostics."""
    value = float(parameter.value)

    if np.isfinite(value) and value > 0.0:
        return value

    candidate = float(default)
    lo = parameter.min_value
    hi = parameter.max_value

    if hi is not None and candidate > hi:
        candidate = 0.5 * float(hi)

    if lo is not None and candidate <= lo:
        lo = float(lo)
        if hi is not None:
            candidate = lo + 0.5 * (float(hi) - lo)
        else:
            candidate = max(np.nextafter(lo, np.inf), default)

    if not np.isfinite(candidate) or candidate <= 0.0:
        raise ValueError(
            f"Could not choose a positive test value for parameter {parameter.path}."
        )

    return candidate


def check_binned_sed_response(
    spectrum,
    source_response,
    atol=0.0,
    freeze=True,
    verbose=True,
):
    """
    Forward-fold each BinnedSED bin and identify zero-response bins.

    One K_i at a time is temporarily set to a positive value while all other
    SED normalizations are set to zero. The original parameter values/free
    states are restored before the function returns. If ``freeze`` is True,
    bins with zero forward-folded counts are then fixed to K=0.

    Parameters
    ----------
    spectrum : BinnedSED
        Spectrum attached to the source used by ``source_response``.
    source_response : BinnedThreeMLExtendedSourceResponse
        Source response whose ``set_source`` method has already been called.
    atol : float, optional
        Template absolute-sum threshold defining an unconstrained bin.
    freeze : bool, optional
        If True, set each zero-response K_i to zero and freeze it.
    verbose : bool, optional
        Print a compact diagnostic for each SED bin.

    Returns
    -------
    dict
        ``local_indices``, ``response_indices``, and ``template_totals``.
    """

    if not isinstance(spectrum, BinnedSED):
        raise TypeError("spectrum must be a BinnedSED instance.")

    if atol < 0.0:
        raise ValueError("atol must be non-negative.")

    if not hasattr(source_response, "expectation") or not hasattr(
        source_response, "_source"
    ):
        raise TypeError(
            "source_response must be a BinnedThreeMLExtendedSourceResponse-like object."
        )

    if source_response._source is None:
        raise RuntimeError("Call source_response.set_source(source) first.")

    attached_spectrum = source_response._source.spectrum.main.shape
    if attached_spectrum is not spectrum:
        raise ValueError(
            "The supplied spectrum is not the spectrum attached to source_response."
        )

    parameters = list(spectrum.normalizations)
    saved_values = [float(par.value) for par in parameters]
    saved_free = [bool(par.free) for par in parameters]

    totals = np.zeros(spectrum.n_bins, dtype=float)

    try:
        for i, par_i in enumerate(parameters):
            for par in parameters:
                par.value = 0.0

            par_i.value = _positive_test_value(par_i)

            expectation = source_response.expectation(copy=False)
            template = _dense_contents(expectation)
            totals[i] = float(np.sum(np.abs(template)))

    finally:
        for par, value, free in zip(parameters, saved_values, saved_free):
            par.value = value
            par.free = free

    zero_local = np.where(totals <= atol)[0]

    response_indices = np.asarray(
        getattr(
            spectrum,
            "_cosipy_ei_bin_indices",
            tuple(range(spectrum.n_bins)),
        ),
        dtype=int,
    )
    zero_response = response_indices[zero_local]

    if freeze and zero_local.size:
        freeze_binned_sed_bins(spectrum, zero_local)

        warnings.warn(
            "The following BinnedSED bins produce zero source counts in the "
            "selected detector analysis space and were frozen to K=0: "
            f"local={zero_local.tolist()}, Ei={zero_response.tolist()}."
        )

    if verbose:
        zero_set = set(zero_local.tolist())
        for i in range(spectrum.n_bins):
            state = "ZERO" if i in zero_set else "ok"
            print(
                f"SED bin {i:2d} (Ei {response_indices[i]:2d}): "
                f"template counts={totals[i]:.6g}  [{state}]"
            )

    return {
        "local_indices": zero_local,
        "response_indices": zero_response,
        "template_totals": totals,
    }


def profile_likelihood_upper_limit(
    profile_nll,
    nll_best,
    best_value,
    sigma=None,
    delta_ts=2.705543,
    max_value=None,
    max_bracket_steps=60,
    rtol=1e-5,
):
    """
    Compute a one-sided profile-likelihood upper limit.

    ``profile_nll(value)`` must evaluate the negative log likelihood with the
    parameter of interest fixed to ``value`` while re-optimizing all desired
    nuisance parameters. The returned limit solves

        2 * [NLL_profile(value) - NLL_best] = delta_ts

    with ``delta_ts=2.705543`` corresponding to the usual one-sided 95%
    profile-likelihood threshold for one parameter.
    """

    nll_best = float(nll_best)
    best_value = max(float(best_value), 0.0)
    delta_ts = float(delta_ts)

    if not np.isfinite(nll_best):
        raise ValueError("nll_best must be finite.")
    if not np.isfinite(best_value):
        raise ValueError("best_value must be finite.")
    if delta_ts <= 0.0 or not np.isfinite(delta_ts):
        raise ValueError("delta_ts must be finite and positive.")

    if sigma is None or not np.isfinite(sigma) or sigma <= 0.0:
        sigma = max(abs(best_value), 1e-12)
    else:
        sigma = float(sigma)

    if max_value is not None:
        max_value = float(max_value)
        if max_value <= best_value:
            raise ValueError("max_value must be greater than best_value.")

    def root_function(value):
        value = max(float(value), 0.0)
        profiled = float(profile_nll(value))
        return 2.0 * (profiled - nll_best) - delta_ts

    lower = best_value
    upper = max(
        lower + 3.0 * sigma,
        2.0 * lower,
        10.0 * sigma,
    )

    if max_value is not None:
        upper = min(upper, max_value)

    for _ in range(int(max_bracket_steps)):
        f_upper = root_function(upper)

        if np.isfinite(f_upper) and f_upper >= 0.0:
            return float(
                brentq(
                    root_function,
                    lower,
                    upper,
                    rtol=rtol,
                    maxiter=100,
                )
            )

        if max_value is not None and upper >= max_value:
            break

        upper *= 2.0
        if max_value is not None:
            upper = min(upper, max_value)

    warnings.warn(
        "Could not bracket the requested profile-likelihood upper limit. "
        "Returning NaN."
    )
    return np.nan
