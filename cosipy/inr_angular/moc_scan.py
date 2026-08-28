"""Multi-resolution (MOC) HEALPix search - a pure acceleration of the regular scan.

Methodology Sec. IV.2.  The ladder evaluates the IDENTICAL exact statistic as
`healpix_scan.regular_scan`; a child pixel at NSIDE 512 gets exactly the value the flat
NSIDE-512 map would assign it.  Multi-resolution only changes WHERE computation is
spent, and the two searches must agree when sufficiently resolved (Sec. IV.2,
consistency statement) - the comparison notebook verifies this.

Ladder: start with a full sky at nside0 -> keep every pixel with
TS >= TS_max - Delta_keep (ALL islands, generous kappa-inflated margin) -> pad with all
HEALPix neighbors -> split each kept NESTED parent into its 4 children (NSIDE doubles)
-> evaluate the children exactly -> repeat to nside_max or documented convergence.
Every exact evaluation is banked; the banked set is the INR training set.
"""

from __future__ import annotations

import time

import healpy as hp
import numpy as np

from .config import CHI2_2_999, MOCConfig
from .geometry import separation_deg
from .healpix_search import SkyMap
from .scoring import LocalizationScore


def _keep_margin(cfg: MOCConfig, level: int, n_levels: int) -> float:
    """Delta_keep = kappa(level) * chi2_2(0.999), kappa shrinking coarse -> fine."""
    if n_levels <= 1:
        kappa = cfg.kappa_fine
    else:
        t = level / (n_levels - 1)
        kappa = cfg.kappa_coarse + t * (cfg.kappa_fine - cfg.kappa_coarse)
    return kappa * CHI2_2_999


def moc_scan(objective: LocalizationScore, cfg: MOCConfig | None = None,
             verbose: bool = True) -> SkyMap:
    """Run the multi-resolution ladder; return the banked evaluations as a SkyMap.

    The returned SkyMap contains EVERY exact evaluation made at every level (the INR
    training bank).  `meta` records per-level bookkeeping, the number of exact
    evaluations, and the final active set.
    """
    cfg = cfg or MOCConfig()
    n_levels = int(np.log2(cfg.nside_max // cfg.nside0)) + 1
    t0 = time.perf_counter()

    # --- rung 0: regular coarse full sky (the first rung IS the regular scan) --------
    nside = cfg.nside0
    npix = hp.nside2npix(nside)
    ipix = np.arange(npix)
    l_deg, b_deg = hp.pix2ang(nside, ipix, nest=True, lonlat=True)
    t_rung = time.perf_counter()
    ts, eta = objective.score(l_deg, b_deg)
    rung0_elapsed = time.perf_counter() - t_rung

    bank_ns = [np.full(npix, nside)]
    bank_ip = [ipix.copy()]
    bank_l = [l_deg]; bank_b = [b_deg]; bank_ts = [ts]; bank_eta = [eta]
    levels_info = [{"nside": nside, "n_selected_parents": 0,
                    "n_new_evaluations": int(npix), "Delta_keep": np.nan,
                    "argmax_move_deg": np.nan,
                    "pixel_scale_deg": float(np.degrees(hp.nside2resol(nside))),
                    "active_ts_range": float(np.max(ts) - np.min(ts)),
                    "elapsed_s": rung0_elapsed}]
    prev_best = (float(l_deg[np.argmax(ts)]), float(b_deg[np.argmax(ts)]))

    cur_ipix, cur_ts = ipix, ts
    level = 0
    while nside < cfg.nside_max:
        margin = _keep_margin(cfg, level, n_levels)
        ts_max = max(float(np.max(cur_ts)), max(float(np.max(t)) for t in bank_ts))
        keep = cur_ipix[cur_ts >= ts_max - margin]          # all qualifying islands
        if cfg.pad_neighbors and len(keep):
            neigh = hp.get_all_neighbours(nside, keep, nest=True)
            keep = np.union1d(keep, neigh[neigh >= 0].ravel())
        # split each kept NESTED parent into its 4 children: pix -> 4 pix + {0,1,2,3}
        children = (keep.astype(np.int64)[:, None] * 4 + np.arange(4)[None, :]).ravel()
        nside *= 2
        level += 1
        l_deg, b_deg = hp.pix2ang(nside, children, nest=True, lonlat=True)
        t_rung = time.perf_counter()
        ts, eta = objective.score(l_deg, b_deg)
        rung_elapsed = time.perf_counter() - t_rung

        bank_ns.append(np.full(len(children), nside)); bank_ip.append(children)
        bank_l.append(l_deg); bank_b.append(b_deg); bank_ts.append(ts); bank_eta.append(eta)

        k = int(np.argmax(ts))
        best = (float(l_deg[k]), float(b_deg[k]))
        move_deg = float(separation_deg(best[0], best[1], prev_best[0], prev_best[1]))
        pix_scale_deg = np.degrees(hp.nside2resol(nside))
        active_range = float(np.max(ts) - np.min(ts))
        levels_info.append({"nside": nside, "n_selected_parents": int(len(keep)),
                            "n_new_evaluations": int(len(children)),
                            "Delta_keep": margin, "argmax_move_deg": move_deg,
                            "pixel_scale_deg": pix_scale_deg,
                            "active_ts_range": active_range,
                            "elapsed_s": rung_elapsed})
        if verbose:
            print(f"[MOC] NSIDE {nside:>4}: kept {len(keep):>5} parents -> "
                  f"{len(children):>6} children evaluated | Delta_keep = {margin:5.1f} | "
                  f"argmax moved {move_deg:6.3f} deg (pixel {pix_scale_deg:.3f} deg)")
        prev_best = best
        cur_ipix, cur_ts = children, ts

        # documented early-stop: argmax stationary at the pixel scale AND the active set
        # straddles the keep band (its TS range exceeds the margin, so the eventual
        # region threshold is bracketed).  Both must hold (methodology Sec. IV.2).
        if (nside < cfg.nside_max
                and move_deg < cfg.argmax_move_frac * pix_scale_deg
                and active_range > margin):
            if verbose:
                print(f"[MOC] converged early at NSIDE {nside}")
            break

    elapsed = time.perf_counter() - t0
    out = SkyMap(nside=np.concatenate(bank_ns), ipix=np.concatenate(bank_ip),
                 l_deg=np.concatenate(bank_l), b_deg=np.concatenate(bank_b),
                 score=np.concatenate(bank_ts), eta=np.concatenate(bank_eta),
                 meta={"kind": "moc", "nside0": cfg.nside0,
                       "nside_final": int(nside), "levels": levels_info,
                       "elapsed_s": elapsed,
                       "n_evaluations": int(sum(len(t) for t in bank_ts))})
    if verbose:
        k = out.argmax
        print(f"[MOC] total exact evaluations: {out.meta['n_evaluations']:,} "
              f"in {elapsed:.1f} s; S_max = {out.score[k]:.1f} at "
              f"(l, b) = ({out.l_deg[k]:.3f}, {out.b_deg[k]:.3f})")
    return out
