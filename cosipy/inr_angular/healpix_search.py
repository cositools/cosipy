"""Regular (single-resolution) full-sky HEALPix evaluation of the localization score.

Choose NSIDE -> enumerate all N_pix = 12 NSIDE^2 equal-area pixels (NESTED, Galactic)
-> pixel centers (l_p, b_p) via hp.pix2ang(lonlat=True) -> evaluate the exact score
S(s) at every center, in parallel across pixels (see scoring.py) -> the argmax pixel is
the initial localization.  The scores array holds exactly one exact scalar per pixel,
in NESTED pixel order.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import healpy as hp
import numpy as np

from .scoring import LocalizationScore


@dataclass
class SkyMap:
    """A set of exactly scored sky directions (regular map or hierarchical bank)."""

    nside: np.ndarray        # (P,) NSIDE at which each entry was evaluated
    ipix: np.ndarray         # (P,) NESTED pixel index at that NSIDE
    l_deg: np.ndarray        # (P,) pixel-center Galactic longitude
    b_deg: np.ndarray        # (P,) pixel-center Galactic latitude
    score: np.ndarray        # (P,) exact localization score at the center
    eta: np.ndarray          # (P,) profiled eta_hat at the center
    meta: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.score)

    # earlier notebooks address the score column as .ts
    @property
    def ts(self) -> np.ndarray:
        return self.score

    @property
    def argmax(self) -> int:
        return int(np.argmax(self.score))

    @property
    def max_score(self) -> float:
        return float(np.max(self.score))

    @property
    def ts_max(self) -> float:
        return self.max_score

    @property
    def best_lb(self):
        k = self.argmax
        return float(self.l_deg[k]), float(self.b_deg[k])

    def to_full_healpix_array(self) -> np.ndarray:
        """For a single-resolution map: the (12 nside^2,) NESTED score array."""
        ns = int(self.nside[0])
        if not np.all(self.nside == ns):
            raise ValueError("map is multi-resolution; use rasterize() instead")
        arr = np.full(hp.nside2npix(ns), hp.UNSEEN)
        arr[self.ipix] = self.score
        return arr

    def to_uniq(self):
        """Multi-order (MOC) representation of a hierarchical bank.

        A banked pixel is frozen (kept at its own order) iff it was never subdivided;
        the frozen pixels tile the sphere exactly once (asserted).  Returns
        (uniq, score, eta) with uniq = 4 nside^2 + ipix (NESTED).
        """
        have = set(zip(self.nside.astype(int).tolist(), self.ipix.astype(int).tolist()))
        frozen = np.array([(int(2 * ns), int(4 * ip)) not in have
                           for ns, ip in zip(self.nside, self.ipix)])
        ns_f = self.nside[frozen].astype(np.int64)
        ip_f = self.ipix[frozen].astype(np.int64)
        uniq = 4 * ns_f * ns_f + ip_f
        area = float(np.sum(1.0 / (ns_f.astype(float) ** 2))) * (np.pi / 3.0)
        assert abs(area - 4.0 * np.pi) < 1e-9, "frozen pixels do not tile the sphere"
        return uniq, self.score[frozen], self.eta[frozen]

    def rasterize(self, nside_out: int) -> np.ndarray:
        """Paint every entry onto its NESTED descendants at nside_out (display)."""
        arr = np.full(hp.nside2npix(nside_out), hp.UNSEEN)
        for ns in np.sort(np.unique(self.nside)):
            sel = self.nside == ns
            f = (nside_out // int(ns)) ** 2
            base = self.ipix[sel].astype(np.int64) * f
            idx = (base[:, None] + np.arange(f)[None, :]).ravel()
            arr[idx] = np.repeat(self.score[sel], f)
        return arr


def regular_scan(scorer: LocalizationScore, nside: int = 128,
                 n_workers: int | None = None, verbose: bool = True) -> SkyMap:
    """Score every pixel center of a full-sky NESTED HEALPix grid, in parallel.

    `n_workers` overrides the scorer's worker setting for this scan (None = all cores).
    Result ordering is the NESTED pixel order regardless of parallelism.
    """
    if n_workers is not None:
        scorer.n_workers = n_workers
    npix = hp.nside2npix(nside)                       # N_pix = 12 nside^2
    ipix = np.arange(npix)
    l_deg, b_deg = hp.pix2ang(nside, ipix, nest=True, lonlat=True)
    t0 = time.perf_counter()
    scores, eta = scorer.score(l_deg, b_deg)
    elapsed = time.perf_counter() - t0
    engine = getattr(scorer, "engine", "unknown")
    workers = (1 if engine != "numba-parallel"
               else scorer.n_workers or (__import__("os").cpu_count() or 1))
    if verbose:
        k = int(np.argmax(scores))
        print(f"[full-sky scan] NSIDE={nside}: {npix:,} pixels "
              f"({np.degrees(hp.nside2resol(nside)):.3f} deg) scored in {elapsed:.1f} s "
              f"| engine = {engine}, {workers} worker(s); S_max = {scores[k]:.1f} at "
              f"(l, b) = ({l_deg[k]:.3f}, {b_deg[k]:.3f})")
    return SkyMap(nside=np.full(npix, nside), ipix=ipix, l_deg=l_deg, b_deg=b_deg,
                  score=scores, eta=eta,
                  meta={"kind": "regular", "nside": nside, "elapsed_s": elapsed,
                        "n_workers": workers, "n_evaluations": int(npix)})
