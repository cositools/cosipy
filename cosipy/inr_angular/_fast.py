"""Optional numba-fused evaluator of the exact localization score.

A pure acceleration of `scoring.LocalizationScore`: the identical mathematics - the
same Vincenty separation formula that `SkyCoord.separation` evaluates (longitudes and
latitudes only, no unit vectors), the same truncated Gaussian+Cauchy kernel with the
same per-event normalizations, the same floored 1/(2 pi sin alpha) Jacobian, and the
same safeguarded Newton profile over eta - fused into one parallel loop over candidate
pixels, so no (P x N) temporaries are materialized.

Jupyter/macOS safety, deliberate choices:
- The threading layer is pinned to numba's built-in ``workqueue`` (set via environment
  variable BEFORE numba is first imported).  The default layer auto-selects OpenMP or
  TBB when present; a second OpenMP runtime in the same process (PyTorch ships its own
  libomp on macOS) aborts the interpreter - the classic silent Jupyter kernel death.
  ``workqueue`` uses plain native threads and coexists with anything.
- Per-thread scratch is allocated ONCE per call as an (n_threads, N) workspace and
  indexed with ``get_thread_id()``; no per-pixel heap allocation inside ``prange``.
- Callers evaluate large pixel sets in bounded blocks (see scoring.py), so a full-sky
  call never pins more than a few MB of intermediates.

The numpy/astropy path in `scoring.py` remains the reference implementation;
`LocalizationScore` cross-validates this path against it on first use and falls back
transparently if numba is missing.
"""

from __future__ import annotations

import os

import numpy as np

# must be set before numba is first imported anywhere in the process; harmless if the
# user already chose a layer explicitly
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")

try:
    from numba import get_thread_id, njit, prange
    import numba as _numba
    # covers the case where numba was already imported before this module (the env
    # variable above is then too late); effective until the first parallel launch
    if _numba.config.THREADING_LAYER == "default":
        _numba.config.THREADING_LAYER = "workqueue"
    NUMBA_AVAILABLE = True
except Exception:                                     # pragma: no cover
    NUMBA_AVAILABLE = False

    def njit(*a, **k):                                # no-op decorator fallback
        def wrap(f):
            return f
        return wrap

    prange = range

    def get_thread_id():
        return 0


_SQRT_2PI = np.sqrt(2.0 * np.pi)
_U = 1.0 / (4.0 * np.pi)


def ts_block_fused(l_c, b_c, l_ax, sin_b_ax, cos_b_ax, phi_rad, trunc_norm,
                   sigma, tail_w, gamma, sin_floor, max_iter, tol,
                   parallel_ok=True):
    """ell and eta_hat for P candidates against N events (fused numba kernel).

    Plain float64 arrays; angles in radians.  Each pixel is one independent work item
    computed with a serial reduction, so the thread count cannot change any value.
    The per-thread scratch is allocated here (outside the jitted functions, so the
    compiled kernels stay cacheable - no repeated JIT per session) and reused across
    the whole block.  `parallel_ok=False` runs the serial twin of the same kernel
    (identical math and results; used when the machine's parallel probe failed).
    Returns (ell, eta).
    """
    if parallel_ok:
        import numba
        scratch = np.empty((numba.get_num_threads(), l_ax.shape[0]))
        return _ts_block_jit(l_c, b_c, l_ax, sin_b_ax, cos_b_ax, phi_rad, trunc_norm,
                             sigma, tail_w, gamma, sin_floor, max_iter, tol, scratch)
    scratch = np.empty((1, l_ax.shape[0]))
    return _ts_block_jit_serial(l_c, b_c, l_ax, sin_b_ax, cos_b_ax, phi_rad,
                                trunc_norm, sigma, tail_w, gamma, sin_floor,
                                max_iter, tol, scratch)


@njit(cache=True, parallel=True)
def _ts_block_jit(l_c, b_c, l_ax, sin_b_ax, cos_b_ax, phi_rad, trunc_norm,
                  sigma, tail_w, gamma, sin_floor, max_iter, tol, scratch):
    P = l_c.shape[0]
    N = l_ax.shape[0]
    ell_out = np.empty(P)
    eta_out = np.empty(P)
    core_norm = (1.0 - tail_w) / (sigma * _SQRT_2PI)
    tail_norm = tail_w * gamma / np.pi
    for p in prange(P):
        d = scratch[get_thread_id()]
        sb1 = np.sin(b_c[p]); cb1 = np.cos(b_c[p])
        for i in range(N):
            dl = l_ax[i] - l_c[p]
            sdl = np.sin(dl); cdl = np.cos(dl)
            num1 = cos_b_ax[i] * sdl
            num2 = cb1 * sin_b_ax[i] - sb1 * cos_b_ax[i] * cdl
            den = sb1 * sin_b_ax[i] + cb1 * cos_b_ax[i] * cdl
            alpha = np.arctan2(np.sqrt(num1 * num1 + num2 * num2), den)
            r = alpha - phi_rad[i]
            if r < -phi_rad[i] or r > np.pi - phi_rad[i]:
                h = 0.0
            else:
                z = r / sigma
                h = (core_norm * np.exp(-0.5 * z * z)
                     + tail_norm / (r * r + gamma * gamma)) / trunc_norm[i]
            sa = np.sin(alpha)
            if sa < sin_floor:
                sa = sin_floor
            d[i] = h / (2.0 * np.pi * sa) - _U
        # safeguarded Newton on f(eta) = sum ln(u + eta d_i), concave, eta in [0, 1]
        f0 = 0.0
        f1 = 0.0
        for i in range(N):
            f0 += d[i] / _U
            f1 += d[i] / (_U + d[i])
        if f0 <= 0.0:
            eta = 0.0
        elif f1 >= 0.0:
            eta = 1.0
        else:
            eta = 0.5
            lo = 0.0
            hi = 1.0
            for _ in range(max_iter):
                f1d = 0.0
                f2d = 0.0
                for i in range(N):
                    q = d[i] / (_U + eta * d[i])
                    f1d += q
                    f2d -= q * q
                if f1d > 0.0:
                    if eta > lo:
                        lo = eta
                else:
                    if eta < hi:
                        hi = eta
                step = -f1d / min(f2d, -1e-300)
                new = eta + step
                if new <= lo or new >= hi:
                    new = 0.5 * (lo + hi)
                if abs(new - eta) < tol:
                    eta = new
                    break
                eta = new
        s = 0.0
        for i in range(N):
            s += np.log(_U + eta * d[i])
        ell_out[p] = s
        eta_out[p] = eta
    return ell_out, eta_out


@njit(cache=True)
def _ts_block_jit_serial(l_c, b_c, l_ax, sin_b_ax, cos_b_ax, phi_rad, trunc_norm,
                  sigma, tail_w, gamma, sin_floor, max_iter, tol, scratch):
    P = l_c.shape[0]
    N = l_ax.shape[0]
    ell_out = np.empty(P)
    eta_out = np.empty(P)
    core_norm = (1.0 - tail_w) / (sigma * _SQRT_2PI)
    tail_norm = tail_w * gamma / np.pi
    for p in range(P):
        d = scratch[0]
        sb1 = np.sin(b_c[p]); cb1 = np.cos(b_c[p])
        for i in range(N):
            dl = l_ax[i] - l_c[p]
            sdl = np.sin(dl); cdl = np.cos(dl)
            num1 = cos_b_ax[i] * sdl
            num2 = cb1 * sin_b_ax[i] - sb1 * cos_b_ax[i] * cdl
            den = sb1 * sin_b_ax[i] + cb1 * cos_b_ax[i] * cdl
            alpha = np.arctan2(np.sqrt(num1 * num1 + num2 * num2), den)
            r = alpha - phi_rad[i]
            if r < -phi_rad[i] or r > np.pi - phi_rad[i]:
                h = 0.0
            else:
                z = r / sigma
                h = (core_norm * np.exp(-0.5 * z * z)
                     + tail_norm / (r * r + gamma * gamma)) / trunc_norm[i]
            sa = np.sin(alpha)
            if sa < sin_floor:
                sa = sin_floor
            d[i] = h / (2.0 * np.pi * sa) - _U
        # safeguarded Newton on f(eta) = sum ln(u + eta d_i), concave, eta in [0, 1]
        f0 = 0.0
        f1 = 0.0
        for i in range(N):
            f0 += d[i] / _U
            f1 += d[i] / (_U + d[i])
        if f0 <= 0.0:
            eta = 0.0
        elif f1 >= 0.0:
            eta = 1.0
        else:
            eta = 0.5
            lo = 0.0
            hi = 1.0
            for _ in range(max_iter):
                f1d = 0.0
                f2d = 0.0
                for i in range(N):
                    q = d[i] / (_U + eta * d[i])
                    f1d += q
                    f2d -= q * q
                if f1d > 0.0:
                    if eta > lo:
                        lo = eta
                else:
                    if eta < hi:
                        hi = eta
                step = -f1d / min(f2d, -1e-300)
                new = eta + step
                if new <= lo or new >= hi:
                    new = 0.5 * (lo + hi)
                if abs(new - eta) < tol:
                    eta = new
                    break
                eta = new
        s = 0.0
        for i in range(N):
            s += np.log(_U + eta * d[i])
        ell_out[p] = s
        eta_out[p] = eta
    return ell_out, eta_out
