"""The INR surrogate f_theta(l, b) ~ S(l, b): model, training, gate, maximization.

The INR is a smooth interpolant of the EXACT score field, fitted to the
(l_k, b_k, S_k) pairs banked by the HEALPix search - it never sees the event arrays -
and is used to extract a continuous maximizer (and, in the calibrated pipeline, to
bracket the region contour).  Every quoted number downstream is computed from the
exact statistic.

Input encoding: the one sanctioned unit-vector use.  (l, b) fails as a network input
twice - the longitude wrap-around seam and the pole degeneracy - so positions are
embedded as s = (cos b cos l, cos b sin l, sin b), for which the Euclidean input
distance 2 sin(alpha/2) is a monotone, sky-uniform function of the true angular
separation.  The 3-vector is an internal network representation ONLY; every angular
quantity and every training target is computed in (l, b) with SkyCoord-style
separations.  On top of it, a dyadic positional encoding defeats the spectral bias of
a small tanh MLP.

torch is imported lazily, inside the functions that need it: the full-sky scoring
phase therefore runs in a torch-free process, so no torch runtime (its libomp in
particular) is even loaded while the numba kernels are working - one less native-
library interaction that can destabilize a local Jupyter kernel.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .config import INRConfig
from .geometry import local_chart_to_lb


def _embed_unit_vectors(l_deg: np.ndarray, b_deg: np.ndarray) -> np.ndarray:
    l = np.radians(np.asarray(l_deg, dtype=np.float64))
    b = np.radians(np.asarray(b_deg, dtype=np.float64))
    return np.stack([np.cos(b) * np.cos(l), np.cos(b) * np.sin(l), np.sin(b)], axis=-1)


def _build_net(cfg: INRConfig):
    """Positional encoding + small tanh MLP -> scalar z-scored score (torch module).

    Defined inside a factory so the module works without torch until training is
    actually requested.
    """
    import torch

    class PositionalEncoding(torch.nn.Module):
        """gamma(s) = [s, sin(2^k pi s), cos(2^k pi s)]_{k=0..L-1}, componentwise."""

        def __init__(self, n_frequencies):
            super().__init__()
            self.n_frequencies = int(n_frequencies)
            self.out_dim = 3 * (1 + 2 * self.n_frequencies)

        def forward(self, s):
            parts = [s]
            for k in range(self.n_frequencies):
                w = (2.0 ** k) * torch.pi
                parts.append(torch.sin(w * s))
                parts.append(torch.cos(w * s))
            return torch.cat(parts, dim=-1)

    class INRNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encode = PositionalEncoding(cfg.n_frequencies)
            layers = []
            d = self.encode.out_dim
            for h in cfg.hidden:
                layers += [torch.nn.Linear(d, h), torch.nn.Tanh()]
                d = h
            layers += [torch.nn.Linear(d, 1)]
            self.mlp = torch.nn.Sequential(*layers)

        def forward(self, s):
            return self.mlp(self.encode(s)).squeeze(-1)

    return INRNet()


@dataclass
class INRSurrogate:
    """A trained surrogate with its normalization, gate report, and query helpers."""

    net: object
    ts_mean: float
    ts_std: float
    cfg: INRConfig
    gate: dict = field(default_factory=dict)
    training_info: dict = field(default_factory=dict)

    def predict_ts(self, l_deg, b_deg) -> np.ndarray:
        """f_theta at (l, b) [score units]; inputs in degrees, any shape."""
        import torch

        l = np.atleast_1d(np.asarray(l_deg, dtype=np.float64))
        b = np.atleast_1d(np.asarray(b_deg, dtype=np.float64))
        s = torch.from_numpy(_embed_unit_vectors(l.ravel(), b.ravel())).float()
        with torch.no_grad():
            z = self.net(s).numpy()
        return (z * self.ts_std + self.ts_mean).reshape(l.shape)

    def continuous_argmax(self, l0_deg: float, b0_deg: float,
                          span_deg: float = 2.0, iters: int = 300):
        """Maximize f_theta continuously near (l0, b0).

        Optimization runs in the locally metric-faithful chart u = (l - l0) cos b0,
        v = b - b0 (bounded to +-span via tanh reparameterization); every iterate is
        mapped back to (l, b) before the embedding, so the sphere geometry is exact.
        Returns (l, b, f_theta) of the surrogate maximizer.
        """
        import torch

        torch.manual_seed(self.cfg.seed)
        raw = torch.zeros(2, requires_grad=True)
        cos_b0 = max(np.cos(np.radians(b0_deg)), 1e-6)
        opt = torch.optim.Adam([raw], lr=5e-2)
        for _ in range(iters):
            opt.zero_grad()
            uv = span_deg * torch.tanh(raw)
            l_rad = torch.deg2rad(torch.tensor(l0_deg)) + torch.deg2rad(uv[0]) / cos_b0
            b_rad = torch.deg2rad(torch.tensor(b0_deg)) + torch.deg2rad(uv[1])
            s = torch.stack([torch.cos(b_rad) * torch.cos(l_rad),
                             torch.cos(b_rad) * torch.sin(l_rad),
                             torch.sin(b_rad)])
            loss = -self.net(s[None, :].float())[0]
            loss.backward()
            opt.step()
        with torch.no_grad():
            uv = (span_deg * torch.tanh(raw)).numpy()
        l_best, b_best = local_chart_to_lb(float(uv[0]), float(uv[1]), l0_deg, b0_deg)
        l_best, b_best = float(np.asarray(l_best)), float(np.asarray(b_best))
        return l_best, b_best, float(np.asarray(self.predict_ts(l_best, b_best)).ravel()[0])


def prepare_training_data(skymap):
    """INR training samples from a scored SkyMap: (l_deg, b_deg, scores).

    The targets are exactly the banked exact scores - nothing interpolated or
    re-evaluated - so `train_inr(*prepare_training_data(skymap))` fits the surrogate to
    the saved map values.
    """
    return skymap.l_deg, skymap.b_deg, skymap.score


def train_inr(l_deg: np.ndarray, b_deg: np.ndarray, ts: np.ndarray,
              cfg: INRConfig | None = None, verbose: bool = True) -> INRSurrogate:
    """Fit f_theta to the banked exact evaluations {(l_k, b_k) -> S_k}.

    - Targets are z-scored score values (mean/std stored for inversion).
    - The loss is up-weighted toward the peak band (S >= S_max - gate_band_ts), so
      surrogate capacity concentrates where the maximizer and the region boundary live.
    - A held-out fraction is reserved for the acceptance gate: within the peak band the
      surrogate error is compared against gate_tol_ts (score units).  The gate result
      is REPORTED and stored; a failed gate does not corrupt any quoted number, because
      the exact statistic decides everything downstream.
    - Deterministic seeding throughout (mini-batched Adam warm-up, then L-BFGS on the
      peak-weighted core set).
    """
    import torch

    cfg = cfg or INRConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    l_deg = np.asarray(l_deg, dtype=np.float64)
    b_deg = np.asarray(b_deg, dtype=np.float64)
    ts = np.asarray(ts, dtype=np.float64)
    n = len(ts)
    if n < 100:
        raise ValueError("too few training points for the surrogate")

    ts_mean, ts_std = float(np.mean(ts)), float(np.std(ts) + 1e-12)
    z = (ts - ts_mean) / ts_std
    ts_max = float(np.max(ts))
    band = ts >= ts_max - cfg.gate_band_ts
    weights = 1.0 + cfg.peak_weight * np.clip(
        (ts - (ts_max - cfg.gate_band_ts)) / cfg.gate_band_ts, 0.0, 1.0)

    rng = np.random.default_rng(cfg.seed)
    idx = rng.permutation(n)
    n_val = max(1, int(cfg.val_fraction * n))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    s_all = torch.from_numpy(_embed_unit_vectors(l_deg, b_deg)).float()
    z_all = torch.from_numpy(z).float()
    w_all = torch.from_numpy(weights).float()
    s_tr, z_tr, w_tr = s_all[train_idx], z_all[train_idx], w_all[train_idx]

    net = _build_net(cfg)
    t0 = time.perf_counter()
    # Adam warm-up: mini-batched for large banks (deterministic shuffling), full-batch
    # for small ones.  L-BFGS polish then runs on the peak-weighted core set: all points
    # in the peak band plus a seeded random subsample of the rest (capped) - the region
    # where surrogate fidelity matters, at tractable full-batch cost.
    opt = torch.optim.Adam(net.parameters(), lr=cfg.adam_lr)
    n_tr = len(train_idx)
    batch = min(8192, n_tr)
    gen = torch.Generator().manual_seed(cfg.seed)
    for it in range(cfg.adam_iters):
        sel = torch.randint(0, n_tr, (batch,), generator=gen)
        opt.zero_grad()
        loss = torch.mean(w_tr[sel] * (net(s_tr[sel]) - z_tr[sel]) ** 2)
        loss.backward()
        opt.step()
    band_tr = ts[train_idx] >= ts_max - cfg.gate_band_ts
    core = np.where(band_tr)[0]
    rest = np.where(~band_tr)[0]
    if len(rest) > 20_000:
        rest = rng.choice(rest, 20_000, replace=False)
    core = np.concatenate([core, rest])
    s_co, z_co, w_co = s_tr[core], z_tr[core], w_tr[core]
    lbfgs = torch.optim.LBFGS(net.parameters(), max_iter=cfg.lbfgs_iters,
                              line_search_fn="strong_wolfe")

    def closure():
        lbfgs.zero_grad()
        loss = torch.mean(w_co * (net(s_co) - z_co) ** 2)
        loss.backward()
        return loss

    lbfgs.step(closure)
    elapsed = time.perf_counter() - t0

    # ------------------------------- acceptance gate ---------------------------------
    # batched full-bank prediction (memory safety for multi-million-point banks;
    # numerically identical to a single pass)
    with torch.no_grad():
        chunks = []
        for a in range(0, n, 262_144):
            chunks.append(net(s_all[a:a + 262_144]).numpy())
        pred_all = np.concatenate(chunks) * ts_std + ts_mean
    resid = pred_all - ts
    val_band = np.intersect1d(val_idx, np.where(band)[0])
    gate = {
        "val_rmse_ts": float(np.sqrt(np.mean(resid[val_idx] ** 2))),
        "val_max_abs_ts": float(np.max(np.abs(resid[val_idx]))) if len(val_idx) else np.nan,
        "band_definition_ts": cfg.gate_band_ts,
        "n_val_in_band": int(len(val_band)),
        "val_band_max_abs_ts": (float(np.max(np.abs(resid[val_band])))
                                if len(val_band) else np.nan),
        "tolerance_ts": cfg.gate_tol_ts,
    }
    gate["passed"] = (len(val_band) > 0
                      and gate["val_band_max_abs_ts"] < cfg.gate_tol_ts)
    if verbose:
        print(f"[INR] trained on {len(train_idx):,} pts ({elapsed:.1f} s); "
              f"held-out RMSE = {gate['val_rmse_ts']:.3f} | "
              f"peak-band held-out max|err| = {gate['val_band_max_abs_ts']:.3f} "
              f"(tol {cfg.gate_tol_ts}) -> gate "
              f"{'PASSED' if gate['passed'] else 'NOT passed (surrogate remains a seed-only device; the exact statistic decides all quoted numbers)'}")
    return INRSurrogate(net=net, ts_mean=ts_mean, ts_std=ts_std, cfg=cfg, gate=gate,
                        training_info={"n_train": int(len(train_idx)),
                                       "n_val": int(len(val_idx)),
                                       "elapsed_s": elapsed})


def train_inr_active(l_deg, b_deg, ts, objective, cfg: INRConfig | None = None,
                     rounds: int = 2, points_per_round: int = 2000,
                     verbose: bool = True):
    """train_inr + an active-learning loop (used by the calibrated pipeline).

    After each fit, if the acceptance gate fails, new EXACT evaluations are added where
    surrogate fidelity matters - seeded low-discrepancy points inside the peak band's
    footprint plus the surrogate's own predicted maximizer - and the surrogate is refit
    (0-2 rounds typically).  Returns (surrogate, l_aug, b_aug, ts_aug): the augmented
    exact-evaluation bank.
    """
    from .geometry import local_chart_to_lb, separation_deg

    cfg = cfg or INRConfig()
    l_aug = np.asarray(l_deg, dtype=np.float64).copy()
    b_aug = np.asarray(b_deg, dtype=np.float64).copy()
    ts_aug = np.asarray(ts, dtype=np.float64).copy()
    rng = np.random.default_rng(cfg.seed + 1)
    sur = train_inr(l_aug, b_aug, ts_aug, cfg, verbose=verbose)
    for rd in range(rounds):
        if sur.gate["passed"]:
            break
        ts_max = float(np.max(ts_aug))
        k = int(np.argmax(ts_aug))
        l0, b0 = float(l_aug[k]), float(b_aug[k])
        band = ts_aug >= ts_max - cfg.gate_band_ts
        radius = float(np.max(separation_deg(l0, b0, l_aug[band], b_aug[band]))) + 0.5
        # seeded uniform points in the chart disc (area-uniform in the local chart)
        rr = radius * np.sqrt(rng.random(points_per_round))
        th = rng.uniform(0.0, 2.0 * np.pi, points_per_round)
        l_new, b_new = local_chart_to_lb(rr * np.cos(th), rr * np.sin(th), l0, b0)
        # plus the surrogate's own predicted maximizer (checked exactly)
        li, bi, _ = sur.continuous_argmax(l0, b0)
        l_new = np.append(np.asarray(l_new), li)
        b_new = np.append(np.asarray(b_new), bi)
        ts_new, _ = objective.score(l_new, b_new)
        l_aug = np.concatenate([l_aug, l_new])
        b_aug = np.concatenate([b_aug, b_new])
        ts_aug = np.concatenate([ts_aug, ts_new])
        if verbose:
            print(f"[INR active] round {rd + 1}: added {len(l_new)} exact evaluations "
                  f"within {radius:.2f} deg of the peak; refitting")
        sur = train_inr(l_aug, b_aug, ts_aug, cfg, verbose=verbose)
    return sur, l_aug, b_aug, ts_aug
