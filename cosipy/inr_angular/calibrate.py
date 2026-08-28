"""Threshold calibration for the 90% region: Delta-TS, sandwich diagnostic, bootstrap.

Methodology Sec. VI.2.  ell_1 is a quasi-likelihood, so Wilks' chi2_2(0.90) = 4.605 is
REPORTED as a reference but never adopted blindly.  Two truth-free devices:

* the sandwich (Godambe) diagnostic - eigenvalues of H^{-1} J at s_hat, where H is the
  observed curvature of the profile objective and J the empirical score variance; under
  a correctly specified model H = J and the eigenvalues are 1.  The corrected reference
  threshold is the 90% quantile of lambda_1 Z_1^2 + lambda_2 Z_2^2.

* the conditional parametric bootstrap - replicates simulated FROM THE FITTED MODEL at
  s_hat with the observed scatter angles held fixed; each replicate is re-localized with
  a reduced search and eta re-profiled, and Delta-TS at the known generating position is
  recorded.  The empirical 90th percentile c90_hat is the ADOPTED threshold.

No GRB truth and no off-burst data enter anywhere.
"""

from __future__ import annotations

import time

import numpy as np

from .config import CHI2_2_90, BootstrapConfig
from .events import AngularEvents
from .geometry import lb_to_local_chart, local_chart_to_lb, offset_by
from .scoring import LocalizationScore
from .refine import exact_polish


def delta_ts(objective: LocalizationScore, ts_max: float, l_deg, b_deg):
    """Delta TS_1(s) = TS_1(s_hat) - TS_1(s), evaluated exactly."""
    ts, _ = objective.score(np.atleast_1d(l_deg), np.atleast_1d(b_deg))
    return ts_max - ts


# ----------------------------------------------------------------------- sandwich ----
def _adaptive_step(objective: LocalizationScore, l0: float, b0: float,
                   ts_max: float, target_dts: float = 1.0,
                   h0_deg: float = 0.1) -> float:
    """Find a chart step at which Delta-TS ~ target (keeps finite differences sane)."""
    h = h0_deg
    for _ in range(6):
        probes_l, probes_b = [], []
        for du, dv in ((h, 0.0), (-h, 0.0), (0.0, h), (0.0, -h)):
            l, b = local_chart_to_lb(du, dv, l0, b0)
            probes_l.append(float(l)); probes_b.append(float(b))
        d = float(np.mean(delta_ts(objective, ts_max, probes_l, probes_b)))
        if 0.25 * target_dts < d < 4.0 * target_dts:
            break
        h *= np.sqrt(target_dts / max(d, 1e-6))
        h = float(np.clip(h, 1e-4, 5.0))
    return h


def sandwich_diagnostic(objective: LocalizationScore, l_hat: float, b_hat: float,
                        eta_hat: float, ts_max: float, seed: int = 0,
                        n_mc: int = 400_000, verbose: bool = True) -> dict:
    """Godambe H^{-1}J eigenvalues and the corrected reference threshold.

    H  : -Hessian of the profile log-quasi-likelihood in the local chart (u, v) at
         s_hat, by central finite differences of EXACT evaluations.
    J  : sum over events of outer products of per-event score contributions
         (finite differences of ln p_i at the plug-in eta_hat).
    The 90% quantile of sum_k lambda_k Z_k^2 is computed by seeded Monte Carlo.
    """
    h = _adaptive_step(objective, l_hat, b_hat, ts_max)

    def ell_at(du, dv):
        l, b = local_chart_to_lb(du, dv, l_hat, b_hat)
        return float(objective.loglike(np.array([float(l)]), np.array([float(b)]))[0][0])

    f00 = ell_at(0.0, 0.0)
    fpu, fmu = ell_at(h, 0), ell_at(-h, 0)
    fpv, fmv = ell_at(0, h), ell_at(0, -h)
    fpp, fpm = ell_at(h, h), ell_at(h, -h)
    fmp, fmm = ell_at(-h, h), ell_at(-h, -h)
    H = -np.array([
        [(fpu - 2 * f00 + fmu) / h ** 2,
         (fpp - fpm - fmp + fmm) / (4 * h ** 2)],
        [(fpp - fpm - fmp + fmm) / (4 * h ** 2),
         (fpv - 2 * f00 + fmv) / h ** 2],
    ])

    # per-event scores at plug-in eta_hat (central differences, same step)
    def event_ld(du, dv):
        l, b = local_chart_to_lb(du, dv, l_hat, b_hat)
        return objective.event_log_density(float(l), float(b), eta_hat)

    s_u = (event_ld(h, 0) - event_ld(-h, 0)) / (2 * h)
    s_v = (event_ld(0, h) - event_ld(0, -h)) / (2 * h)
    S = np.stack([s_u, s_v], axis=1)               # (N, 2)
    S = S - S.mean(axis=0, keepdims=True)          # remove the (near-zero) mean score
    J = S.T @ S

    Hinv = np.linalg.inv(H)
    lam = np.sort(np.real(np.linalg.eigvals(Hinv @ J)))[::-1]
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_mc, 2))
    mix = lam[0] * z[:, 0] ** 2 + lam[1] * z[:, 1] ** 2
    c90_sand = float(np.quantile(mix, 0.90))
    out = {"step_deg": h, "H": H, "J": J, "eigenvalues": lam,
           "c90_sandwich": c90_sand, "c90_wilks": CHI2_2_90}
    if verbose:
        print(f"[sandwich] step {h:.4f} deg | eigenvalues of H^-1 J = "
              f"({lam[0]:.3f}, {lam[1]:.3f}) | corrected c90 = {c90_sand:.3f} "
              f"(Wilks 4.605)")
    return out


# ---------------------------------------------------------------------- bootstrap ----
def simulate_replicate(events: AngularEvents, kernel, l_hat: float, b_hat: float,
                       eta_hat: float, rng: np.random.Generator) -> AngularEvents:
    """One conditional parametric replicate: phi_i kept, axes redrawn from the fitted model.

    With probability eta_hat the event is signal: its axis lies at separation
    alpha = phi_i + r (r ~ h(.|phi_i)) from s_hat with uniform azimuth
    (SkyCoord.directional_offset_by); otherwise it is a floor draw, uniform on the
    sphere (inverse-CDF sampling in (l, b) - no unit vectors).
    """
    n = len(events)
    phi_rad = events.phi_rad
    is_sig = rng.random(n) < eta_hat
    l_new = np.empty(n); b_new = np.empty(n)
    ns = int(np.sum(is_sig))
    if ns:
        r = kernel.sample(phi_rad[is_sig], rng)
        alpha_deg_ = np.degrees(phi_rad[is_sig] + r)
        pa = rng.uniform(0.0, 360.0, ns)
        l_new[is_sig], b_new[is_sig] = offset_by(
            np.full(ns, l_hat), np.full(ns, b_hat), pa, alpha_deg_)
    nb = n - ns
    if nb:
        l_new[~is_sig] = rng.uniform(0.0, 360.0, nb)
        b_new[~is_sig] = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, nb)))
    return AngularEvents(phi_deg=events.phi_deg.copy(), l_axis_deg=l_new,
                         b_axis_deg=b_new, times=events.times.copy())


def bootstrap_c90(events: AngularEvents, kernel, l_hat: float, b_hat: float,
                  eta_hat: float, cfg: BootstrapConfig | None = None,
                  objective_cfg=None, verbose: bool = True) -> dict:
    """Conditional parametric bootstrap of Delta-TS at the generating position.

    For each replicate: simulate (phi fixed), re-localize with a reduced search (a chart
    lattice of half-width cap_radius_deg around s_hat - legitimate because the
    replicates are generated there - plus a Nelder-Mead polish), re-profile eta, and
    record  Delta-TS^(m) = TS^(m)(s_hat^(m)) - TS^(m)(s_hat).
    Returns c90_hat with its order-statistic 95% CI.
    """
    cfg = cfg or BootstrapConfig()
    rng = np.random.default_rng(cfg.seed)
    # reduced-search lattice in the chart (mapped to (l, b) once; exact evals on sphere)
    g = np.arange(-cfg.cap_radius_deg, cfg.cap_radius_deg + 1e-9, cfg.grid_step_deg)
    uu, vv = np.meshgrid(g, g)
    lat_l, lat_b = local_chart_to_lb(uu.ravel(), vv.ravel(), l_hat, b_hat)
    lat_l = np.asarray(lat_l, dtype=np.float64); lat_b = np.asarray(lat_b, dtype=np.float64)

    deltas = np.empty(cfg.n_replicates)
    t0 = time.perf_counter()
    for m in range(cfg.n_replicates):
        rep = simulate_replicate(events, kernel, l_hat, b_hat, eta_hat, rng)
        obj_m = LocalizationScore(rep, kernel, objective_cfg)
        ts_grid, _ = obj_m.score(lat_l, lat_b)
        k = int(np.argmax(ts_grid))
        if cfg.polish:
            pol = exact_polish(obj_m, float(lat_l[k]), float(lat_b[k]),
                               span_deg=2.0, verbose=False)
            ts_m_max = pol["ts"]
        else:
            ts_m_max = float(ts_grid[k])
        ts_at_gen = obj_m.score_scalar(l_hat, b_hat)
        deltas[m] = ts_m_max - ts_at_gen
        if verbose and (m + 1) % max(1, cfg.n_replicates // 10) == 0:
            print(f"[bootstrap] {m + 1}/{cfg.n_replicates} replicates "
                  f"({time.perf_counter() - t0:.0f} s)")
    deltas = np.sort(deltas)
    c90 = float(np.quantile(deltas, 0.90))
    # order-statistic 95% CI on the 0.90 quantile (binomial)
    from scipy.stats import binom
    M = cfg.n_replicates
    lo_k = int(binom.ppf(0.025, M, 0.90))
    hi_k = min(int(binom.ppf(0.975, M, 0.90)), M - 1)
    out = {"deltas": deltas, "c90_hat": c90,
           "c90_ci95": (float(deltas[lo_k]), float(deltas[hi_k])),
           "c90_wilks": CHI2_2_90, "honesty_ratio": c90 / CHI2_2_90,
           "n_replicates": M, "elapsed_s": time.perf_counter() - t0}
    if verbose:
        print(f"[bootstrap] c90_hat = {c90:.3f}  (95% CI {out['c90_ci95'][0]:.3f}-"
              f"{out['c90_ci95'][1]:.3f}) | Wilks 4.605 | honesty ratio "
              f"{out['honesty_ratio']:.2f} | {out['elapsed_s']:.0f} s")
    return out


# score-terminology alias
delta_score = delta_ts
