"""Exact local refinement: the quoted best fit is the EXACT-objective maximizer.

Methodology Part V, last step: the INR proposes a continuous seed; a derivative-free
Nelder-Mead polish of the exact TS_1 (in the locally metric-faithful chart, every point
mapped back to (l, b) before evaluation) produces the quoted (l_hat, b_hat).  Surrogate
error therefore never enters any quoted number.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .geometry import local_chart_to_lb
from .scoring import LocalizationScore


def exact_polish(objective: LocalizationScore, l_seed: float, b_seed: float,
                 span_deg: float = 3.0, xatol_deg: float = 1e-4,
                 verbose: bool = True) -> dict:
    """Nelder-Mead maximization of the exact TS_1 seeded at (l_seed, b_seed).

    Returns {l, b, ts, eta, n_evaluations, seed_ts}.
    """
    n0 = objective.n_exact_evaluations
    seed_ts = objective.score_scalar(l_seed, b_seed)

    def neg_ts(x):
        u = float(np.clip(x[0], -span_deg, span_deg))
        v = float(np.clip(x[1], -span_deg, span_deg))
        l, b = local_chart_to_lb(u, v, l_seed, b_seed)
        return -objective.score_scalar(float(l), float(b))

    res = minimize(neg_ts, np.zeros(2), method="Nelder-Mead",
                   options={"xatol": xatol_deg, "fatol": 1e-8,
                            "maxiter": 600, "initial_simplex": np.array(
                                [[0.0, 0.0], [0.15, 0.0], [0.0, 0.15]])})
    l_hat, b_hat = local_chart_to_lb(res.x[0], res.x[1], l_seed, b_seed)
    l_hat, b_hat = float(l_hat), float(b_hat)
    ts_hat, eta_hat = objective.score(np.array([l_hat]), np.array([b_hat]))
    out = {"l": l_hat, "b": b_hat, "ts": float(ts_hat[0]), "eta": float(eta_hat[0]),
           "seed_ts": seed_ts,
           "n_evaluations": objective.n_exact_evaluations - n0}
    if verbose:
        print(f"[polish] seed TS = {seed_ts:.2f} -> exact max TS = {out['ts']:.2f} at "
              f"(l, b) = ({l_hat:.4f}, {b_hat:.4f})  "
              f"[{out['n_evaluations']} exact evaluations]")
    return out
