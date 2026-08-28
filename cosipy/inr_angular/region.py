"""Exact 90% C.L. region: membership by the exact objective, HEALPix area, components.

Methodology Sec. VI.4: the INR may BRACKET the contour (pre-filter where the exact
statistic must be evaluated), but membership is decided ONLY by exact Delta-TS values
at NESTED pixel centers of a fine HEALPix grid (default NSIDE 1024, ~0.057 deg), so the
contour is never surrogate- or pixelation-limited.  All connected components satisfying
the threshold are kept (multimodality is flagged, never suppressed).  The solid angle
is member-pixel count x the exact equal-area pixel area.  The spherical-ellipse summary
(vendored FitSphericalEllipse - the elliptic cone x^T Q x = 0, the second sanctioned
Cartesian use) is fitted to the exact boundary of the dominant component, afterwards,
for catalog convenience only.
"""

from __future__ import annotations

import numpy as np
import healpy as hp

from .geometry import separation_deg
from .config import RegionConfig
from .scoring import LocalizationScore


def disc_pixels(nside: int, l0: float, b0: float, radius_deg: float,
                coarse_nside: int = 64) -> np.ndarray:
    """NESTED pixels of `nside` whose centers lie within `radius_deg` of (l0, b0).

    Hierarchical, separation-only implementation (no unit-vector disc query): coarse
    NESTED parents are selected by center separation with a half-diagonal margin, then
    their descendants are filtered by their own center separations.
    """
    coarse_nside = min(coarse_nside, nside)
    npc = hp.nside2npix(coarse_nside)
    lc, bc = hp.pix2ang(coarse_nside, np.arange(npc), nest=True, lonlat=True)
    margin = 1.5 * np.degrees(hp.nside2resol(coarse_nside))
    par = np.where(separation_deg(l0, b0, lc, bc) <= radius_deg + margin)[0]
    f = (nside // coarse_nside) ** 2
    child = (par.astype(np.int64)[:, None] * f + np.arange(f)[None, :]).ravel()
    lf, bf = hp.pix2ang(nside, child, nest=True, lonlat=True)
    return child[separation_deg(l0, b0, lf, bf) <= radius_deg]


def _connected_components(nside: int, members: np.ndarray) -> list[np.ndarray]:
    """Connected components of a NESTED member-pixel set via HEALPix neighbor BFS."""
    member_set = set(int(p) for p in members)
    seen = set()
    comps = []
    for p in members:
        p = int(p)
        if p in seen:
            continue
        stack = [p]
        seen.add(p)
        comp = []
        while stack:
            q = stack.pop()
            comp.append(q)
            for nb in hp.get_all_neighbours(nside, q, nest=True):
                nb = int(nb)
                if nb >= 0 and nb in member_set and nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        comps.append(np.array(sorted(comp)))
    comps.sort(key=len, reverse=True)
    return comps


def extract_region(objective: LocalizationScore, l_hat: float, b_hat: float,
                   ts_max: float, c90: float, cfg: RegionConfig | None = None,
                   surrogate=None, verbose: bool = True) -> dict:
    """Extract {s : Delta TS_1(s) <= c90} on a fine NESTED HEALPix grid.

    The search disc around s_hat expands (x2) until the region no longer touches the
    disc boundary, so the full contour - including any secondary island connected to
    the disc - is captured.  If `surrogate` is given, its prediction pre-filters which
    pixels receive exact evaluation (INR bracketing); pixels the surrogate excludes by
    more than `inr_prefilter_margin_ts` are dropped, everything else is decided exactly.
    """
    cfg = cfg or RegionConfig()
    nside = cfg.nside_region
    pix_area_deg2 = hp.nside2pixarea(nside, degrees=True)
    n_exact0 = objective.n_exact_evaluations

    radius = 1.0
    members = boundary = None
    for _ in range(6):
        pix = disc_pixels(nside, l_hat, b_hat, radius)
        lp, bp = hp.pix2ang(nside, pix, nest=True, lonlat=True)
        if surrogate is not None:
            pred_deficit = ts_max - surrogate.predict_ts(lp, bp)
            keep = pred_deficit <= c90 + cfg.inr_prefilter_margin_ts
            # always evaluate a core disc exactly, whatever the surrogate says
            keep |= separation_deg(l_hat, b_hat, lp, bp) <= 0.5
            pix, lp, bp = pix[keep], lp[keep], bp[keep]
        ts, _ = objective.score(lp, bp)
        dts = ts_max - ts
        inside = dts <= c90
        members = pix[inside]
        if len(members) == 0:
            # region smaller than a pixel: keep the pixel containing s_hat
            members = np.atleast_1d(hp.ang2pix(nside, l_hat, b_hat, nest=True,
                                               lonlat=True))
            break
        lm, bm = hp.pix2ang(nside, members, nest=True, lonlat=True)
        max_sep = float(np.max(separation_deg(l_hat, b_hat, lm, bm)))
        if max_sep < radius - 2.0 * np.degrees(hp.nside2resol(nside)):
            break
        radius *= 2.0
        if verbose:
            print(f"[region] contour touches the search disc - expanding to "
                  f"{radius:.0f} deg")

    comps = _connected_components(nside, members)
    dominant = comps[0]
    # boundary of the dominant component: member pixels with a non-member neighbor
    member_set = set(int(p) for p in dominant)
    boundary = np.array([p for p in dominant
                         if any(int(nb) >= 0 and int(nb) not in member_set
                                for nb in hp.get_all_neighbours(nside, int(p), nest=True))],
                        dtype=np.int64)
    if len(boundary) == 0:
        boundary = dominant
    lb_boundary = hp.pix2ang(nside, boundary, nest=True, lonlat=True)

    out = {
        "nside": nside, "c90": float(c90), "members": members,
        "members_uniq": (4 * nside * nside + members.astype(np.int64)),
        "components": comps, "n_components": len(comps),
        "multimodal": len(comps) > 1,
        "area_deg2": float(len(members) * pix_area_deg2),
        "area_dominant_deg2": float(len(dominant) * pix_area_deg2),
        "boundary_l_deg": np.asarray(lb_boundary[0]),
        "boundary_b_deg": np.asarray(lb_boundary[1]),
        "n_exact_evaluations": objective.n_exact_evaluations - n_exact0,
    }
    if verbose:
        print(f"[region] Delta-TS <= {c90:.2f} at NSIDE {nside}: "
              f"{len(members)} pixels, {out['area_deg2']:.4f} deg^2, "
              f"{len(comps)} component(s)"
              + (" [MULTIMODAL - all components kept]" if out['multimodal'] else ""))
    return out


def fit_ellipse_summary(region: dict, l_hat: float, b_hat: float,
                        verbose: bool = True) -> dict:
    """Optional spherical-ellipse summary of the dominant component's exact boundary.

    Uses the vendored FitSphericalEllipse (verbatim copy of the TS-map pipeline's
    fitter): elliptic cone x^T Q x = 0 through the sphere center, least-squares fitted
    to the boundary points; angular semi-axes arctan(a), arctan(b).  A summary only -
    it never redefines region membership or coverage.
    """
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    from ._fit_ellipse_vendored import FitSphericalEllipse

    edge = SkyCoord(l=region["boundary_l_deg"] * u.deg,
                    b=region["boundary_b_deg"] * u.deg, frame="galactic")
    r0 = max(np.sqrt(region["area_dominant_deg2"] / np.pi), 0.02)
    fitter = FitSphericalEllipse(
        edge_coords=edge,
        skycoord_init=SkyCoord(l=l_hat * u.deg, b=b_hat * u.deg, frame="galactic"),
        semi_major_init=r0, semi_minor_init=0.8 * r0, theta_init=45.0)
    fitter.fit_ellipse()
    center_l = float(np.rad2deg(fitter.lon_fit_rad) % 360.0)
    center_b = float(np.rad2deg(fitter.lat_fit_rad))
    a_deg = float(np.rad2deg(np.arctan(fitter.a_fit)))
    b_deg = float(np.rad2deg(np.arctan(fitter.b_fit)))
    theta = float(np.rad2deg(fitter.theta_fit_rad)) % 180.0
    if b_deg > a_deg:
        # enforce semi-major >= semi-minor (the fitter's a/b are unordered); the
        # position angle rotates by 90 deg under the swap
        a_deg, b_deg = b_deg, a_deg
        theta = (theta + 90.0) % 180.0
    curve = FitSphericalEllipse.generate_spherical_ellipse(
        np.radians(center_l), np.radians(center_b),
        np.tan(np.radians(a_deg)), np.tan(np.radians(b_deg)),
        np.radians(theta), npoints=721)
    ellipse_area = np.pi * a_deg * b_deg
    out = {"center_l_deg": center_l, "center_b_deg": center_b,
           "semi_major_deg": a_deg, "semi_minor_deg": b_deg,
           "position_angle_deg": theta,
           "curve_l_deg": np.asarray(curve.l.deg), "curve_b_deg": np.asarray(curve.b.deg),
           "ellipse_area_deg2": float(ellipse_area),
           "area_fidelity_ratio": float(ellipse_area / max(region["area_dominant_deg2"],
                                                           1e-12))}
    if verbose:
        print(f"[ellipse] center ({center_l:.4f}, {center_b:.4f}), a = {a_deg:.4f} deg, "
              f"b = {b_deg:.4f} deg, PA = {theta:.1f} deg | "
              f"area fidelity {out['area_fidelity_ratio']:.2f}")
    return out


def hull_boundary(region_skycoord):
    """Convex hull of region-pixel coordinates (the TS-map notebook's construction).

    Returns (edge_pixels, edge_skycoord): the hull vertices as an (M, 2) array of
    (l, b) in degrees and as a Galactic SkyCoord.
    """
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from scipy.spatial import ConvexHull

    coords = np.array([region_skycoord.l.deg, region_skycoord.b.deg]).T
    near_pole = float(np.max(np.abs(coords[:, 1]))) > 80.0
    wraps = float(np.ptp(coords[:, 0])) > 180.0
    if near_pole or wraps:
        # the planar (l, b) chart is degenerate here (longitude compresses toward the
        # poles / wraps at 0-360), so take the hull in a tangent-plane chart centered
        # on the region instead; the returned vertices are still the original (l, b)
        center = SkyCoord(region_skycoord.cartesian.mean(), frame="galactic")
        local = region_skycoord.transform_to(center.skyoffset_frame())
        chart = np.array([local.lon.wrap_at(180 * u.deg).deg, local.lat.deg]).T
        hull = ConvexHull(chart, incremental=True)
    else:
        hull = ConvexHull(coords, incremental=True)
    edge_pixels = coords[hull.vertices]
    edge_skycoord = SkyCoord(l=edge_pixels[:, 0] * u.deg, b=edge_pixels[:, 1] * u.deg,
                             frame="galactic")
    return edge_pixels, edge_skycoord


def fit_ellipse_from_hull(edge_skycoord, l_init: float, b_init: float,
                          semi_major_init: float = 0.01, semi_minor_init: float = 0.01,
                          theta_init: float = 45.0):
    """FitSphericalEllipse on hull vertices, initialized at (l_init, b_init).

    Same fitter and call style as the TS-map notebook; the initialization is the
    INR-refined location in the Method-1 workflow.  Returns (fit, result_dict) with
    center/axes/theta in degrees and the equivalent radius sqrt(a * b).
    """
    from astropy.coordinates import SkyCoord

    from ._fit_ellipse_vendored import FitSphericalEllipse

    center_init = SkyCoord(l_init, b_init, unit="deg", frame="galactic")
    fit = FitSphericalEllipse(edge_coords=edge_skycoord, skycoord_init=center_init,
                              semi_major_init=semi_major_init,
                              semi_minor_init=semi_minor_init, theta_init=theta_init)
    fit.fit_ellipse()
    center_coord, semi_major_axis, semi_minor_axis, theta = fit.ellipse_param
    result = {
        "center_coord": center_coord,
        "center_l_deg": float(center_coord.l.deg),
        "center_b_deg": float(center_coord.b.deg),
        "semi_major_deg": float(semi_major_axis),
        "semi_minor_deg": float(semi_minor_axis),
        "theta_deg": float(theta),
        "r_eq_deg": float(np.sqrt(semi_major_axis * semi_minor_axis)),
    }
    return fit, result
