"""The exact Method-1 localization score S(l, b).

Per event, with alpha_i(s) the axis-candidate separation and r_i = alpha_i - phi_i:

    g_i(s)        = h(r_i | phi_i) / (2 pi sin alpha_i)        [signal density, sr^-1]
    p_i(s, eta)   = eta * g_i(s) + (1 - eta) / (4 pi)          [mixture with the floor]
    ell(s)        = max_{eta in [0,1]}  sum_i ln p_i(s, eta)   [profiled per direction]
    S(s)          = 2 [ ell(s) - ell_0 ],   ell_0 = -N ln(4 pi)

The inner problem is concave in eta (log of an affine function), solved by a
safeguarded, fully vectorized Newton iteration with analytic first/second derivatives.

ell is a conditional (given-phi) quasi-likelihood - the isotropic floor is a declared
robustness term, not a background model - so S supports localization only; nothing here
assumes a chi^2 distribution for it.

Parallel evaluation (and why it is thread-based)
------------------------------------------------
Candidate directions are scored independently, so full-sky evaluation parallelizes
across pixels.  With numba present (the default), the fused kernel in `_fast.py` runs a
`prange` over pixels: worker THREADS share the read-only event arrays (no copying), each
pixel is one work item, per-thread scratch is preallocated, and the result ordering is
the input ordering by construction.  `n_workers` bounds the thread count (None = all
cores), and large pixel sets are evaluated in bounded blocks so no call pins large
intermediates.  Thread count does not change any value: each pixel's reduction is
computed serially inside its own task, so serial and parallel results are bitwise
identical (asserted by `verify_parallel_consistency`).

Without numba the evaluation is SERIAL, in small chunks that bound the (chunk x N)
numpy temporaries.  Deliberately no multiprocessing: process pools inside a Jupyter
kernel (spawn on macOS/Windows) re-import the heavy stack per worker and multiply the
numpy working set by the core count - the reliable way to kill a local kernel - for a
computation that the threaded fast path already covers.

Engine safety (why a subprocess probe)
--------------------------------------
A native abort inside a parallel launch (threading-runtime conflicts, LLVM issues)
kills the interpreter before any Python-level fallback can run - the kernel just dies.
So the parallel kernel is never executed in this process until a THROWAWAY SUBPROCESS
has run the same kind of parallel launch and exited cleanly.  If that probe crashes or
times out, this process permanently uses the serial numba kernel (identical math and
results, single core) and says so.  Override with the environment variable
INR_ANGULAR_ENGINE = "parallel" | "serial" | "numpy" (set before scoring starts).
"""

from __future__ import annotations

import os

import numpy as np

from .config import ObjectiveConfig
from .events import AngularEvents
from .kernel import ARMKernel

U_FLOOR = 1.0 / (4.0 * np.pi)   # uniform density on the sphere [sr^-1]


def _available_cores() -> int:
    return os.cpu_count() or 1


_PARALLEL_PROBE_RESULT = None      # None = not probed yet in this process

_PROBE_SRC = r"""
import os
os.environ["NUMBA_THREADING_LAYER"] = "workqueue"
import numpy as np
from numba import njit, prange, get_thread_id
import numba

@njit(parallel=True)
def probe(x, scratch):
    out = np.empty(x.shape[0])
    for p in prange(x.shape[0]):
        d = scratch[get_thread_id()]
        acc = 0.0
        for i in range(d.shape[0]):
            d[i] = np.sin(x[p] + 0.001 * i)
            acc += d[i]
        out[p] = acc
    return out

scratch = np.empty((numba.get_num_threads(), 256))
r = probe(np.linspace(0.0, 1.0, 4096), scratch)
assert np.isfinite(r).all()
print("OK")
"""


def probe_parallel_safe(timeout_s: float = 240.0, verbose: bool = True) -> bool:
    """Run a representative numba-parallel launch in a throwaway subprocess.

    Returns True only if the subprocess compiled and executed the parallel kernel and
    exited cleanly.  A crash (segfault, OpenMP abort, LLVM error) or timeout only kills
    the subprocess, never the calling kernel.  Result is cached per process.
    """
    global _PARALLEL_PROBE_RESULT
    if _PARALLEL_PROBE_RESULT is not None:
        return _PARALLEL_PROBE_RESULT
    import subprocess
    import sys
    try:
        res = subprocess.run([sys.executable, "-c", _PROBE_SRC],
                             capture_output=True, text=True, timeout=timeout_s)
        ok = (res.returncode == 0 and "OK" in res.stdout)
        if not ok and verbose:
            tail = (res.stderr or "").strip().splitlines()[-3:]
            print(f"[engine probe] parallel numba launch failed in the test "
                  f"subprocess (returncode {res.returncode}) - using the SERIAL "
                  f"numba kernel instead (identical results, single core).")
            for line in tail:
                print(f"[engine probe]   {line}")
    except Exception as exc:
        ok = False
        if verbose:
            print(f"[engine probe] parallel probe did not finish ({exc}) - using "
                  f"the SERIAL numba kernel instead.")
    _PARALLEL_PROBE_RESULT = ok
    return ok


class LocalizationScore:
    """Exact, vectorized evaluator of the localization score S(l, b) for one burst."""

    def __init__(self, events: AngularEvents, kernel: ARMKernel,
                 cfg: ObjectiveConfig | None = None, use_fast: bool = True,
                 n_workers: int | None = None):
        self.events = events
        self.kernel = kernel
        self.cfg = cfg or ObjectiveConfig()
        self.n_workers = n_workers          # None = all available cores
        self.n_events = len(events)
        # per-event precomputations (candidate-independent, done once)
        self._l_ax = np.radians(events.l_axis_deg)
        self._b_ax = np.radians(events.b_axis_deg)
        self._phi_rad = events.phi_rad
        self._trunc = kernel.trunc_norm(self._phi_rad)
        self._sin_floor = np.sin(np.radians(self.cfg.sin_alpha_floor_deg))
        self.ell0 = self.n_events * np.log(U_FLOOR)
        self.n_exact_evaluations = 0
        # optional numba acceleration (identical math; validated on first use)
        from . import _fast
        self._fast = _fast if (use_fast and _fast.NUMBA_AVAILABLE) else None
        self._fast_validated = False
        self._parallel_ok = None            # decided at first fast call (or by env)
        override = os.environ.get("INR_ANGULAR_ENGINE", "").strip().lower()
        if override == "numpy":
            self._fast = None
        elif override == "serial":
            self._parallel_ok = False
        elif override == "parallel":
            self._parallel_ok = True
        self._sin_b_ax = np.sin(self._b_ax)
        self._cos_b_ax = np.cos(self._b_ax)

    @property
    def engine(self) -> str:
        if self._fast is None:
            return "numpy-serial"
        if self._parallel_ok is None:
            return "numba (parallel probe pending)"
        return "numba-parallel" if self._parallel_ok else "numba-serial"

    # ----------------------------------------------------------------- signal density
    def _g_block(self, l_deg: np.ndarray, b_deg: np.ndarray) -> np.ndarray:
        """g_i(s) for a block of P candidates -> (P, N) matrix [sr^-1].

        Separations use the SkyCoord.separation formula (see geometry.separation_deg).
        The 1/(2 pi sin alpha) Jacobian is floored at sin(alpha_floor) - a numerical
        guard for the integrable singularity at alpha -> {0, pi}.
        """
        l_c = np.radians(np.asarray(l_deg, dtype=np.float64))[:, None]
        b_c = np.radians(np.asarray(b_deg, dtype=np.float64))[:, None]
        from astropy.coordinates import angular_separation
        alpha = angular_separation(l_c, b_c, self._l_ax[None, :], self._b_ax[None, :])
        r = alpha - self._phi_rad[None, :]
        h = self.kernel.pdf(r, self._phi_rad[None, :], trunc_norm=self._trunc[None, :])
        sin_a = np.maximum(np.sin(alpha), self._sin_floor)
        return h / (2.0 * np.pi * sin_a)

    # ------------------------------------------------------------------ eta profiling
    @staticmethod
    def _profile_eta_rows(g: np.ndarray, max_iter: int = 60, tol: float = 1e-10):
        """Profile eta row-wise: maximize f(eta) = sum_i ln(u + eta (g_i - u)), eta in [0,1].

        f is strictly concave; Newton with a bisection safeguard converges for every row
        simultaneously.  Returns (ell, eta_hat), each (P,).
        """
        u = U_FLOOR
        d = g - u                                   # (P, N)
        P = g.shape[0]

        def fprime(eta):
            return np.sum(d / (u + eta[:, None] * d), axis=1)

        f0 = fprime(np.zeros(P))
        f1 = fprime(np.ones(P))
        eta = np.full(P, 0.5)
        lo = np.zeros(P)
        hi = np.ones(P)
        at_zero = f0 <= 0.0                         # boundary: pure floor
        at_one = f1 >= 0.0                          # boundary: pure signal
        interior = ~(at_zero | at_one)
        eta[at_zero] = 0.0
        eta[at_one] = 1.0
        if np.any(interior):
            for _ in range(max_iter):
                q = d / (u + eta[:, None] * d)
                f1d = np.sum(q, axis=1)
                f2d = -np.sum(q * q, axis=1)
                lo = np.where(interior & (f1d > 0), np.maximum(lo, eta), lo)
                hi = np.where(interior & (f1d < 0), np.minimum(hi, eta), hi)
                step = np.where(interior, -f1d / np.minimum(f2d, -1e-300), 0.0)
                new = eta + step
                bad = (new <= lo) | (new >= hi)
                new = np.where(bad, 0.5 * (lo + hi), new)
                moved = np.abs(new - eta)
                eta = np.where(interior, new, eta)
                if np.all(moved[interior] < tol):
                    break
        ell = np.sum(np.log(u + eta[:, None] * d), axis=1)
        return ell, eta

    # ------------------------------------------------------------------- public API
    def loglike(self, l_deg, b_deg):
        """ell(s) and eta_hat(s) for scalar or array candidates.  Returns ((P,), (P,))."""
        l_arr = np.atleast_1d(np.asarray(l_deg, dtype=np.float64))
        b_arr = np.atleast_1d(np.asarray(b_deg, dtype=np.float64))
        if l_arr.shape != b_arr.shape:
            raise ValueError("l and b must have identical shapes")
        P = len(l_arr)
        if self._fast is not None:
            if self._parallel_ok is None:
                # never launch the parallel kernel here before a throwaway subprocess
                # has survived the same kind of launch on this machine
                self._parallel_ok = probe_parallel_safe()
            if not self._fast_validated:
                self._validate_fast_path()
            if self._parallel_ok:
                self._apply_num_threads()
            l_rad = np.radians(l_arr)
            b_rad = np.radians(b_arr)
            ell = np.empty(P)
            eta = np.empty(P)
            # bounded blocks: keeps any single fused call (and its per-thread scratch
            # lifetime) small, without changing per-pixel results
            block = 65_536
            for a in range(0, P, block):
                b = min(a + block, P)
                ell[a:b], eta[a:b] = self._fast.ts_block_fused(
                    l_rad[a:b], b_rad[a:b],
                    self._l_ax, self._sin_b_ax, self._cos_b_ax, self._phi_rad,
                    self._trunc, self.kernel.sigma_rad, self.kernel.tail_weight,
                    self.kernel.gamma_rad, self._sin_floor,
                    self.cfg.newton_max_iter, self.cfg.newton_tol,
                    parallel_ok=self._parallel_ok)
            self.n_exact_evaluations += P
            return ell, eta
        # no-numba path: serial, small chunks (bounds the (chunk x N) temporaries)
        ell = np.empty(P)
        eta = np.empty(P)
        step = int(self.cfg.chunk_pixels)
        for a in range(0, P, step):
            b = min(a + step, P)
            g = self._g_block(l_arr[a:b], b_arr[a:b])
            ell[a:b], eta[a:b] = self._profile_eta_rows(
                g, self.cfg.newton_max_iter, self.cfg.newton_tol)
            del g
        self.n_exact_evaluations += P
        return ell, eta

    def score(self, l_deg, b_deg):
        """Localization score S(s) = 2 [ ell(s) - ell_0 ] and eta_hat(s)."""
        ell, eta = self.loglike(l_deg, b_deg)
        return 2.0 * (ell - self.ell0), eta

    def score_scalar(self, l_deg: float, b_deg: float) -> float:
        return float(self.score(np.array([l_deg]), np.array([b_deg]))[0][0])

    # kept as thin aliases so earlier notebooks and tools keep running
    def ts(self, l_deg, b_deg):
        return self.score(l_deg, b_deg)

    def ts_scalar(self, l_deg: float, b_deg: float) -> float:
        return self.score_scalar(l_deg, b_deg)

    # ------------------------------------------------------------------- parallelism
    def _apply_num_threads(self):
        """Bound the numba thread count to n_workers (None = all cores).

        Only touches the setting when it actually changes; the workqueue layer pinned
        in `_fast.py` makes this safe alongside torch/OpenMP in the same process.
        """
        import numba
        limit = numba.config.NUMBA_NUM_THREADS
        n = max(1, min(self.n_workers or limit, limit))
        if numba.get_num_threads() != n:
            numba.set_num_threads(n)

    # ------------------------------------------------- helpers used by other modules
    def event_log_density(self, l_deg: float, b_deg: float, eta: float) -> np.ndarray:
        """ln p_i(s, eta) for every event at one direction (N,)."""
        g = self._g_block(np.array([l_deg]), np.array([b_deg]))[0]
        return np.log(U_FLOOR + eta * (g - U_FLOOR))

    def profile_eta_subset(self, l_deg: float, b_deg: float, mask: np.ndarray) -> float:
        """eta profiled on a subset of events at one direction (used by the kernel CV)."""
        g = self._g_block(np.array([l_deg]), np.array([b_deg]))[0][mask]
        _, eta = self._profile_eta_rows(g[None, :])
        return float(eta[0])

    def signal_membership(self, l_deg: float, b_deg: float, eta: float) -> np.ndarray:
        """Posterior signal-membership weights w_i = eta g_i / p_i at one direction."""
        g = self._g_block(np.array([l_deg]), np.array([b_deg]))[0]
        p = U_FLOOR + eta * (g - U_FLOOR)
        return eta * g / p

    def solid_angle_density_check(self, phi_deg: float = 45.0, n_alpha: int = 40_000) -> float:
        """Numerical check that int g dOmega = 1 (Jacobian + truncation together)."""
        phi = np.radians(phi_deg)
        alpha = np.linspace(1e-9, np.pi - 1e-9, n_alpha)
        h = self.kernel.pdf(alpha - phi, np.full_like(alpha, phi))
        return float(np.trapezoid(h, alpha))

    def _validate_fast_path(self, n_check: int = 24, tol: float = 1e-6):
        """Cross-check the numba fast path against the numpy/astropy reference.

        Runs once per instance; disagreement beyond `tol` (score units) disables the
        fast path and the reference implementation serves all calls.
        """
        rng = np.random.default_rng(0)
        l = rng.uniform(0.0, 360.0, n_check)
        b = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, n_check)))
        g = self._g_block(l, b)
        ell_ref, _ = self._profile_eta_rows(g, self.cfg.newton_max_iter,
                                            self.cfg.newton_tol)
        ell_fast, _ = self._fast.ts_block_fused(
            np.radians(l), np.radians(b), self._l_ax, self._sin_b_ax,
            self._cos_b_ax, self._phi_rad, self._trunc, self.kernel.sigma_rad,
            self.kernel.tail_weight, self.kernel.gamma_rad, self._sin_floor,
            self.cfg.newton_max_iter, self.cfg.newton_tol,
            parallel_ok=bool(self._parallel_ok))
        worst = float(np.max(np.abs(2.0 * (ell_fast - ell_ref))))
        self._fast_validated = True
        if worst > tol:
            import warnings
            warnings.warn(f"numba fast path disagrees with the reference by {worst:g} "
                          "score units - falling back to the numpy/astropy path")
            self._fast = None
        self.fast_path_max_dev_ts = worst           # attribute name kept for continuity


def verify_parallel_consistency(events: AngularEvents, kernel: ARMKernel,
                                cfg: ObjectiveConfig | None = None,
                                n_check: int = 512, seed: int = 0) -> float:
    """Serial (n_workers=1) vs parallel scores on random directions; returns max |diff|.

    Each pixel is one independent work item computed with the identical serial
    reduction, so the difference must be exactly zero; asserted below.
    """
    rng = np.random.default_rng(seed)
    l = rng.uniform(0.0, 360.0, n_check)
    b = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, n_check)))
    serial = LocalizationScore(events, kernel, cfg, n_workers=1)
    parallel = LocalizationScore(events, kernel, cfg, n_workers=None)
    s1, _ = serial.score(l, b)
    s2, _ = parallel.score(l, b)
    worst = float(np.max(np.abs(s1 - s2)))
    assert worst < 1e-9, f"serial and parallel scores disagree by {worst}"
    return worst


# backward-compatible names (earlier notebooks import these)
Method1Objective = LocalizationScore
