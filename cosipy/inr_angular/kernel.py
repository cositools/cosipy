"""The ARM kernel h(r | phi): a truncated, normalized residual density per event.

Model (methodology Sec. II.2): a zero-centred Gaussian core plus a Cauchy (Lorentzian)
shoulder - ARM distributions of Compton telescopes are peaked but heavy-tailed - with
support truncated to the physical residual range

    r in [-phi, pi - phi]        (alpha = phi + r must lie in [0, pi])

and renormalized per event so that   int_{-phi}^{pi-phi} h(r | phi) dr = 1.

All kernel math is in RADIANS (densities are per radian, so that the solid-angle signal
density g = h / (2 pi sin alpha) is per steradian, commensurable with the uniform floor
u = 1/(4 pi)).

Truth-free scale selection: no sim-distillation product ships with this repository, so
the scale sigma is chosen by the methodology's route 3 - K-fold pseudo-likelihood
cross-validation ON THE ON-BURST EVENTS ONLY, evaluated at a provisional peak obtained
with a robust default width.  No GRB truth, no off-burst data, no response information
is used anywhere in this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import erf

from .config import KernelConfig

_SQRT2 = np.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


@dataclass(frozen=True)
class ARMKernel:
    """Frozen kernel h(r|phi) with per-event truncation normalizations."""

    sigma_rad: float
    tail_weight: float
    gamma_rad: float

    @classmethod
    def from_config(cls, cfg: KernelConfig) -> "ARMKernel":
        if not (0.0 <= cfg.tail_weight < 1.0):
            raise ValueError("tail_weight must be in [0, 1)")
        if cfg.sigma_deg <= 0:
            raise ValueError("sigma must be positive")
        sigma = np.radians(cfg.sigma_deg)
        return cls(sigma_rad=float(sigma), tail_weight=float(cfg.tail_weight),
                   gamma_rad=float(cfg.tail_gamma_over_sigma * sigma))

    # -- unnormalized components (per radian) -----------------------------------------
    def _core_pdf(self, r):
        z = r / self.sigma_rad
        return _INV_SQRT_2PI / self.sigma_rad * np.exp(-0.5 * z * z)

    def _core_cdf(self, r):
        return 0.5 * (1.0 + erf(r / (self.sigma_rad * _SQRT2)))

    def _tail_pdf(self, r):
        g = self.gamma_rad
        return g / (np.pi * (r * r + g * g))

    def _tail_cdf(self, r):
        return 0.5 + np.arctan(r / self.gamma_rad) / np.pi

    # -- public API -------------------------------------------------------------------
    def trunc_norm(self, phi_rad: np.ndarray) -> np.ndarray:
        """Z(phi) = mass of the untruncated mixture on [-phi, pi - phi] (per event)."""
        lo = -np.asarray(phi_rad)
        hi = np.pi + lo
        w = self.tail_weight
        return ((1.0 - w) * (self._core_cdf(hi) - self._core_cdf(lo))
                + w * (self._tail_cdf(hi) - self._tail_cdf(lo)))

    def pdf(self, r_rad: np.ndarray, phi_rad: np.ndarray,
            trunc_norm: np.ndarray | None = None) -> np.ndarray:
        """h(r | phi) [per radian], truncated + normalized; 0 outside the support."""
        r = np.asarray(r_rad)
        phi = np.asarray(phi_rad)
        Z = self.trunc_norm(phi) if trunc_norm is None else trunc_norm
        w = self.tail_weight
        raw = (1.0 - w) * self._core_pdf(r) + w * self._tail_pdf(r)
        inside = (r >= -phi) & (r <= np.pi - phi)
        return np.where(inside, raw / Z, 0.0)

    def sample(self, phi_rad: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Draw residuals r ~ h(. | phi) (one per entry of phi) by rejection sampling.

        Used by the conditional parametric bootstrap; scatter angles stay fixed, only the
        residual (hence the axis) is redrawn.
        """
        phi = np.asarray(phi_rad, dtype=np.float64)
        out = np.empty_like(phi)
        todo = np.arange(len(phi))
        w = self.tail_weight
        for _ in range(1000):
            if len(todo) == 0:
                break
            m = len(todo)
            pick_tail = rng.random(m) < w
            r = np.where(pick_tail,
                         self.gamma_rad * np.tan(np.pi * (rng.random(m) - 0.5)),
                         rng.normal(0.0, self.sigma_rad, m))
            ok = (r >= -phi[todo]) & (r <= np.pi - phi[todo])
            out[todo[ok]] = r[ok]
            todo = todo[~ok]
        if len(todo):
            # pathological phi extremely close to 0 or pi: fall back to the support edge
            out[todo] = np.clip(0.0, -phi[todo], np.pi - phi[todo])
        return out

    def integral_check(self, phi_deg: float, n: int = 200_001) -> float:
        """Numerical check that int h(r|phi) dr = 1 on the truncated support."""
        phi = np.radians(phi_deg)
        r = np.linspace(-phi, np.pi - phi, n)
        return float(np.trapezoid(self.pdf(r, np.full_like(r, phi)), r))


def select_scale_cv(events, base_cfg: KernelConfig, objective_factory,
                    sigma_grid_deg=None, n_folds: int = 5, seed: int = 0,
                    provisional_sigma_deg: float = 2.0, coarse_nside: int = 16,
                    n_iterate: int = 1, verbose: bool = True):
    """Truth-free kernel-scale selection (methodology Sec. II.2, route 3).

    Procedure:
      1. localize provisionally with a robust default width (coarse HEALPix scan +
         Nelder-Mead polish) - truth-free;
      2. K-fold CV: for each candidate sigma, profile eta on the training folds AT the
         provisional peak and score the held-out events' mean log density there;
      3. adopt the sigma maximizing the summed held-out score; optionally re-localize
         and repeat once.

    `objective_factory(kernel)` must return an object with
    `.loglike(l, b)` -> (ell, eta_hat) and `.event_log_density(l, b, eta)` (see
    objective.Method1Objective).  Returns (best_kernel_cfg, diagnostics_dict).
    """
    import healpy as hp
    from scipy.optimize import minimize

    from . import geometry

    if sigma_grid_deg is None:
        sigma_grid_deg = np.array([0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0])
    rng = np.random.default_rng(seed)
    fold_ids = rng.integers(0, n_folds, len(events))

    sigma_current = float(provisional_sigma_deg)
    history = []
    for it in range(n_iterate + 1):
        # provisional peak with the current width (coarse scan + polish)
        obj_prov = objective_factory(ARMKernel.from_config(base_cfg.with_sigma(sigma_current)))
        npix = hp.nside2npix(coarse_nside)
        lp, bp = hp.pix2ang(coarse_nside, np.arange(npix), nest=True, lonlat=True)
        ell, _ = obj_prov.loglike(lp, bp)
        k = int(np.argmax(ell))
        l0, b0 = float(lp[k]), float(bp[k])

        def neg(x):
            l, b = geometry.local_chart_to_lb(x[0], x[1], l0, b0)
            return -obj_prov.loglike(float(l), float(b))[0][0]

        res = minimize(neg, np.zeros(2), method="Nelder-Mead",
                       options={"xatol": 1e-3, "fatol": 1e-6, "maxiter": 400})
        l_prov, b_prov = geometry.local_chart_to_lb(res.x[0], res.x[1], l0, b0)
        l_prov, b_prov = float(l_prov), float(b_prov)

        # K-fold held-out scoring at the provisional peak
        scores = np.zeros(len(sigma_grid_deg))
        for j, s_deg in enumerate(sigma_grid_deg):
            kern = ARMKernel.from_config(base_cfg.with_sigma(s_deg))
            obj = objective_factory(kern)
            total = 0.0
            for f in range(n_folds):
                train = fold_ids != f
                test = ~train
                eta_tr = obj.profile_eta_subset(l_prov, b_prov, train)
                total += float(np.sum(obj.event_log_density(l_prov, b_prov, eta_tr)[test]))
            scores[j] = total
        best = int(np.argmax(scores))
        history.append({"iteration": it, "provisional_peak": (l_prov, b_prov),
                        "sigma_grid_deg": np.asarray(sigma_grid_deg, dtype=float),
                        "cv_scores": scores, "sigma_selected_deg": float(sigma_grid_deg[best])})
        if verbose:
            print(f"[kernel CV] iter {it}: provisional peak (l, b) = "
                  f"({l_prov:.3f}, {b_prov:.3f}), selected sigma = "
                  f"{sigma_grid_deg[best]:.2f} deg")
        if abs(sigma_grid_deg[best] - sigma_current) < 1e-12:
            sigma_current = float(sigma_grid_deg[best])
            break
        sigma_current = float(sigma_grid_deg[best])
    return base_cfg.with_sigma(sigma_current), {"history": history,
                                               "sigma_selected_deg": sigma_current}
