"""The three main localization plots, styled after the DC4 TS-map notebook
(`DC4_GRB_Localization_with_True_BKG/GRB_bn080802386/TS_map_fitting_GRB_bn080802386.ipynb`):

1. `plot_fullsky_map`     - full-sky HEALPix localization-score map (mollview), best fit +
                            (post-hoc) true position marked;
2. `plot_local_zoom`      - cartview zoom around the localization: exact TS_1 field,
                            best fit, truth, and the exact 90% C.L. HEALPix boundary;
3. `plot_region_ellipse`  - the 90% C.L. region figure in the TS-map notebook's exact
                            style (viridis clipped to [TS_max - c90, TS_max], gray C.L.
                            edge points from the HEALPix region, orange spherical
                            ellipse + center, deepskyblue peak marker, red truth star).

All maps are HEALPix-native (mhealpy `HealpixMap`, single- or multi-order/MOC); the 90%
region drawn in plots 2-3 is the exact HEALPix Delta-TS region - the ellipse is only
the reporting summary fitted afterward to its boundary.

The mhealpy plotting incantation (`ax_kw={'coord': 'G'}, coord="C"`) is copied verbatim
from the TS-map notebook - it renders a Galactic map on Galactic cartview/mollview axes
with `ax.get_transform('world')` in Galactic (l, b), longitude increasing to the left.

A few numeric diagnostic helpers (kernel curve, surrogate parity, bootstrap histogram)
are kept at the end of the module for interactive use, but the notebooks intentionally
do not call them: the final presentation contains exactly the three plots above.
"""

from __future__ import annotations

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import astropy.units as u
from mhealpy import HealpixMap

from .config import CHI2_2_90
from .geometry import separation_deg


# ------------------------------------------------------------------ map construction
def dense_healpix_map(skymap) -> HealpixMap:
    """Single-resolution SkyMap -> mhealpy HealpixMap (NESTED, Galactic)."""
    return HealpixMap(data=skymap.to_full_healpix_array(), scheme="NESTED",
                      coordsys="G")


def moc_healpix_map(skymap) -> HealpixMap:
    """Hierarchical SkyMap bank -> genuine multi-order (MOC) mhealpy HealpixMap.

    Uses the frozen-pixel UNIQ tiling from `SkyMap.to_uniq()` (pixels never subdivided,
    at their own order).  `density=True` because TS_1 is an intensive field.
    """
    uniq, ts, _ = skymap.to_uniq()
    order = np.argsort(uniq)
    return HealpixMap(data=ts[order], uniq=uniq[order], density=True, coordsys="G")


def window_moc_map(objective, l0: float, b0: float, radius_deg: float = 7.0,
                   nside_base: int = 16, nside_window: int = 512) -> HealpixMap:
    """Exact-TS multi-order map: fine NESTED pixels inside a disc, coarse outside.

    Built for the zoom/region figures: every NSIDE-`nside_base` parent whose center
    lies within `radius_deg` (+ safety margin) of (l0, b0) is replaced by ALL of its
    NESTED descendants at `nside_window`; everything else stays at the base order.  The
    exact Method-1 TS_1 is evaluated at every pixel center (this is presentation-time
    evaluation; the localization itself is already frozen).  Selection uses center
    separations only - no unit-vector disc queries.
    """
    npix = hp.nside2npix(nside_base)
    ip_base = np.arange(npix)
    lb, bb = hp.pix2ang(nside_base, ip_base, nest=True, lonlat=True)
    margin = 1.5 * np.degrees(hp.nside2resol(nside_base))
    cover = separation_deg(l0, b0, lb, bb) <= radius_deg + margin
    coarse = ip_base[~cover].astype(np.int64)
    f = (nside_window // nside_base) ** 2
    fine = (ip_base[cover].astype(np.int64)[:, None] * f + np.arange(f)[None, :]).ravel()

    l_c, b_c = lb[~cover], bb[~cover]
    l_f, b_f = hp.pix2ang(nside_window, fine, nest=True, lonlat=True)
    ts_c, _ = objective.score(l_c, b_c)
    ts_f, _ = objective.score(l_f, b_f)

    uniq = np.concatenate([4 * nside_base ** 2 + coarse,
                           4 * nside_window ** 2 + fine])
    vals = np.concatenate([ts_c, ts_f])
    order = np.argsort(uniq)
    return HealpixMap(data=vals[order], uniq=uniq[order], density=True, coordsys="G")


# ----------------------------------------------------------------- shared axis style
def _style_cart_axes(ax, tick_deg: float = 5.0):
    """The TS-map notebook's cartview axis conventions (ticks bottom/left, degrees)."""
    ax.set_xlabel("Galactic Longitude (deg)", fontsize=10)
    ax.set_ylabel("Galactic Latitude (deg)", fontsize=10)
    for k in (0, 1):
        ax.coords[k].set_ticks_visible(True)
        ax.coords[k].set_ticklabel_visible(True)
    ax.coords[0].set_ticks_position('b'); ax.coords[0].set_ticklabel_position('b')
    ax.coords[1].set_ticks_position('l'); ax.coords[1].set_ticklabel_position('l')
    ax.coords[0].set_ticks(spacing=tick_deg * u.deg)
    ax.coords[1].set_ticks(spacing=tick_deg * u.deg)
    ax.coords[0].set_major_formatter('d')
    ax.coords[1].set_major_formatter('d')


def _lat_window(b_center: float, half: float):
    """Latitude range [b-half, b+half], shifted (not shrunk) to stay inside +-90.

    cartview rejects latra outside [-90, 90]; for a localization near a Galactic
    pole the window slides inward so its angular height is preserved.
    """
    lo, hi = b_center - half, b_center + half
    if lo < -90.0:
        hi = min(90.0, hi + (-90.0 - lo))
        lo = -90.0
    if hi > 90.0:
        lo = max(-90.0, lo - (hi - 90.0))
        hi = 90.0
    return [lo, hi]


def _lon_window(l_center: float, b_center: float, half: float):
    """Longitude range for a cartview window centered on (l_center, b_center).

    Away from the poles this is the usual [l-half, l+half].  Within 10 deg of a
    Galactic pole a fixed-longitude window is meaningless (meridians converge: points
    a couple of degrees apart on the sky can differ by ~180 deg in l), so the window
    opens to all longitudes and the plot shows the whole polar cap.
    """
    if abs(b_center) > 80.0:
        return [-179.99, 179.99]
    return [l_center - half, l_center + half]


# -------------------------------------------------------------------- the three plots
def plot_fullsky_map(m: HealpixMap, grb_name: str, best=None, truth=None, save=None):
    """Plot 1 - full-sky HEALPix TS_1 map (mollview, TS-map style)."""
    fig = plt.figure(figsize=(8, 5), dpi=200)
    ax = fig.add_subplot(projection="mollview")
    m.plot(ax, ax_kw={'coord': 'G'}, coord="C", cbar=True, cmap="viridis")
    if best is not None:
        ax.scatter(best[0], best[1], s=30, marker="P",
                   transform=ax.get_transform('world'), color="deepskyblue",
                   edgecolor="black", linewidth=0.4, label="Best fit")
    if truth is not None:
        ax.scatter(truth[0], truth[1], s=25, marker="*",
                   transform=ax.get_transform('world'), color="red",
                   label="True location")
    ax.set_title(f"{grb_name} - exact localization score map", size=10)
    if best is not None or truth is not None:
        ax.legend(loc="lower right", fontsize=7)
    if save:
        fig.savefig(save, dpi=200, bbox_inches="tight")
    return fig


def plot_local_zoom(m: HealpixMap, grb_name: str, l_hat: float, b_hat: float,
                    region: dict, truth=None, half_window_deg: float = 5.0,
                    stretch_ts: float = 100.0, save=None):
    """Plot 2 - cartview zoom around the localization.

    Shows the exact TS_1 field with a wide color stretch (structure visible), the exact
    best fit, the (post-hoc) true position, and the exact 90% C.L. HEALPix region
    boundary (gray points = boundary pixel centers of the Delta-TS region).
    """
    fig = plt.figure(figsize=(6, 6), dpi=300)
    lonra = _lon_window(l_hat, b_hat, half_window_deg)
    latra = _lat_window(b_hat, half_window_deg)
    ax = fig.add_subplot(projection="cartview", lonra=lonra, latra=latra)
    ax.set_title(f"{grb_name} - local score map and 90% C.L. boundary", size=10)
    max_ts = float(np.max(m))
    m.plot(ax, ax_kw={'coord': 'G'}, coord="C", cbar=False,
           vmin=max_ts - stretch_ts, vmax=max_ts)
    ax.scatter(region["boundary_l_deg"], region["boundary_b_deg"], s=1, marker=".",
               transform=ax.get_transform('world'), color="gray",
               label="90% C.L. boundary (exact HEALPix)")
    ax.scatter(l_hat, b_hat, s=8, marker="P", transform=ax.get_transform('world'),
               color="deepskyblue", edgecolor="black", linewidth=0.3,
               label="Best fit (exact)")
    if truth is not None:
        ax.scatter(truth[0], truth[1], s=4, marker="*",
                   transform=ax.get_transform('world'), color="red",
                   label="True location")
    _style_cart_axes(ax)
    plt.legend()
    if save:
        fig.savefig(save, dpi=300, bbox_inches="tight")
    return fig


def plot_region_ellipse(m: HealpixMap, grb_name: str, l_hat: float, b_hat: float,
                        c90: float, region: dict, ellipse: dict, truth=None,
                        half_window_deg: float = 5.0, save=None):
    """Plot 3 - the 90% C.L. region figure, in the TS-map notebook's exact style.

    Color scale clipped to [TS_max - c90, TS_max] (here c90 is the CALIBRATED
    threshold, where the baseline used Wilks' 4.605), so exactly the 90% HEALPix region
    lights up; gray C.L. edge points are the exact region-boundary pixel centers;
    the orange curve/center is the spherical-ellipse SUMMARY (FitSphericalEllipse -
    the same fitter the TS-map notebook uses); deepskyblue marker = the exact best fit.
    """
    fig = plt.figure(figsize=(6, 6), dpi=300)
    lonra = _lon_window(l_hat, b_hat, half_window_deg)
    latra = _lat_window(b_hat, half_window_deg)
    ax = fig.add_subplot(projection="cartview", lonra=lonra, latra=latra)
    ax.set_title("90% C.L. region", size=10)
    max_ts = float(np.max(m))
    m.plot(ax, ax_kw={'coord': 'G'}, coord="C", cbar=False,
           vmin=max_ts - c90, vmax=max_ts)
    if truth is not None:
        ax.scatter(truth[0], truth[1], s=4, marker="*",
                   transform=ax.get_transform('world'), color="red",
                   label="True location")
    step = max(1, len(region["boundary_l_deg"]) // 600)
    ax.scatter(region["boundary_l_deg"][::step], region["boundary_b_deg"][::step],
               s=1, marker="*", transform=ax.get_transform('world'), color="gray",
               label="C.L. edge points")
    ax.scatter(ellipse["curve_l_deg"], ellipse["curve_b_deg"], s=0.5, marker=".",
               transform=ax.get_transform('world'), color="orange",
               label="Localization ellipse")
    ax.scatter(ellipse["center_l_deg"], ellipse["center_b_deg"], s=4, marker=".",
               transform=ax.get_transform('world'), color="orange",
               label="Ellipse center")
    ax.scatter(l_hat, b_hat, s=8, marker="P", transform=ax.get_transform('world'),
               color="deepskyblue", edgecolor="black", linewidth=0.3,
               label="INR map peak")
    _style_cart_axes(ax)
    plt.legend()
    if save:
        fig.savefig(save, dpi=300, bbox_inches="tight")
    return fig


# --------------------------------------------------- numeric diagnostics (not plotted
# in the notebooks; kept for interactive use) -----------------------------------------
def plot_kernel(kernel, phi_deg_list=(20.0, 45.0, 70.0), r_max_deg=15.0, save=None):
    """h(r|phi) curves (diagnostic; not part of the notebook presentation)."""
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    r = np.radians(np.linspace(-r_max_deg, r_max_deg, 1001))
    for phi_deg in phi_deg_list:
        phi = np.radians(phi_deg)
        ax.plot(np.degrees(r), kernel.pdf(r, np.full_like(r, phi)),
                label=fr"$\phi = {phi_deg:.0f}^\circ$")
    ax.set_xlabel(r"$r = \alpha - \phi$ [deg]")
    ax.set_ylabel(r"$h(r\,|\,\phi)$ [rad$^{-1}$]")
    ax.legend(); ax.grid(alpha=0.3)
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig


def plot_inr_validation(surrogate, l_deg, b_deg, ts_exact, ts_max, band_ts=25.0,
                        save=None):
    """Surrogate parity/residual diagnostic (not part of the notebook presentation)."""
    pred = surrogate.predict_ts(l_deg, b_deg)
    resid = pred - ts_exact
    band = ts_exact >= ts_max - band_ts
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
    axes[0].scatter(ts_exact, pred, s=2, alpha=0.3, color="steelblue")
    lims = [min(ts_exact.min(), pred.min()), max(ts_exact.max(), pred.max())]
    axes[0].plot(lims, lims, color="crimson", lw=0.8)
    axes[0].set_xlabel(r"exact $TS_1$"); axes[0].set_ylabel(r"INR $f_\theta$")
    axes[1].hist(resid[band], bins=40, color="steelblue", alpha=0.8)
    axes[1].set_xlabel(r"$f_\theta - TS_1$ (peak band)")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig


def plot_bootstrap(boot, save=None):
    """Bootstrap Delta-TS histogram (not part of the notebook presentation)."""
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.hist(boot["deltas"], bins=30, color="steelblue", alpha=0.8, density=True)
    ax.axvline(boot["c90_hat"], color="crimson", lw=1.5,
               label=fr"$\hat c_{{90}} = {boot['c90_hat']:.2f}$")
    ax.axvline(CHI2_2_90, color="k", ls="--", lw=1.2, label="Wilks 4.605")
    ax.set_xlabel(r"bootstrap $\Delta TS^{(m)}$"); ax.set_ylabel("density")
    ax.legend(); ax.grid(alpha=0.3)
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig


def plot_score_map_with_ellipse(score_hpx_map, center_window, critical: float,
                                edge_pixels, ellipse_edge, center_coord,
                                peak=None, inr=None, truth=None, save=None,
                                extra_curves=None, half_window_deg: float = 5.0,
                                vmin=None):
    """The 90% C.L. figure in the TS-map notebook's exact layout, score terminology.

    `score_hpx_map` is the mhealpy map read back from {GRB}_Score_Map.fits; the color
    scale is clipped to [S_max - critical, S_max] so exactly the nominal 90% region
    (Delta S <= critical) lights up.  `center_window` sets the +-5 deg cartview window;
    truth is a post-hoc overlay only.  Within 10 deg of a Galactic pole the cartview
    chart degenerates (meridians converge), so the figure switches to a gnomonic
    (tangent-plane) view centered on the window, sized to contain the fitted ellipse.
    `extra_curves` is an optional list of (skycoord, color, label) overlays (e.g. a
    calibrated ellipse); they share the window-sizing with the main ellipse.
    """
    fig = plt.figure(figsize=(6, 6), dpi=300)
    polar = abs(center_window[1]) > 80.0
    if polar:
        from astropy import units as u
        from astropy.coordinates import SkyCoord
        c0 = SkyCoord(l=center_window[0] * u.deg, b=center_window[1] * u.deg,
                      frame="galactic")
        half = max(half_window_deg, 1.2 * float(np.max(c0.separation(ellipse_edge).deg)))
        for crv, _, _ in (extra_curves or []):
            half = max(half, 1.2 * float(np.max(c0.separation(crv).deg)))
        ax = fig.add_subplot(projection="gnomview",
                             rot=(center_window[0], center_window[1], 0),
                             reso=2.0 * half * 60.0 / 1800.0)
    else:
        lonra = _lon_window(center_window[0], center_window[1], half_window_deg)
        latra = _lat_window(center_window[1], half_window_deg)
        ax = fig.add_subplot(projection="cartview", lonra=lonra, latra=latra)
    ax.set_title("90% C.L. region", size=10)

    max_score = float(np.max(score_hpx_map[:]))
    lo = float(vmin) if vmin is not None else max_score - critical
    score_hpx_map.plot(ax, vmin=lo, vmax=max_score,
                       ax_kw={'coord': 'G'}, coord="C", cbar=False)

    if truth is not None:
        ax.scatter(truth[0], truth[1], s=4, marker="*",
                   transform=ax.get_transform('world'), color="red",
                   label="True location")
    ax.scatter(edge_pixels[:, 0], edge_pixels[:, 1], s=1, marker="*",
               transform=ax.get_transform('world'), color="gray",
               label="C.L. edge points")
    ax.scatter(ellipse_edge.l.deg, ellipse_edge.b.deg, s=0.5, marker=".",
               transform=ax.get_transform('world'), color="orange",
               label="Localization ellipse")
    ax.scatter(center_coord.l.deg, center_coord.b.deg, s=4, marker=".",
               transform=ax.get_transform('world'), color="orange",
               label="Ellipse center")
    for crv, color, label in (extra_curves or []):
        ax.scatter(crv.l.deg, crv.b.deg, s=0.5, marker=".",
                   transform=ax.get_transform('world'), color=color, label=label)
    if peak is not None:
        ax.scatter(peak[0], peak[1], s=8, marker="P",
                   transform=ax.get_transform('world'), color="deepskyblue",
                   edgecolor="black", linewidth=0.3, label="Score map peak")
    if inr is not None:
        ax.scatter(inr[0], inr[1], s=8, marker="X",
                   transform=ax.get_transform('world'), color="magenta",
                   edgecolor="black", linewidth=0.3, label="INR predicted location")
    _style_cart_axes(ax, tick_deg=15.0 if polar else 5.0)
    plt.legend(loc="upper right") if polar else plt.legend()
    if save:
        fig.savefig(save, dpi=300, bbox_inches="tight")
    return fig
