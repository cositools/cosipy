"""Compton-cone geometry: angular separations, alpha_i(s) and r_i(s) = alpha_i - phi_i.

Coordinate policy (methodology Sec. I.3): all angular computation is native Galactic
(l, b).  Every separation is the great-circle separation exactly as computed by
`astropy.coordinates.SkyCoord.separation`.  For vectorized bulk evaluation we call
`astropy.coordinates.angular_separation` - the very function `SkyCoord.separation`
uses internally (the Vincenty formula on longitudes/latitudes) - on radian arrays,
which avoids per-call SkyCoord object overhead while remaining bit-for-bit the same
computation.  `validate_against_skycoord()` asserts that equivalence numerically.

No unit-vector conversion is used for any angular distance.
"""

from __future__ import annotations

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, angular_separation


def separation_deg(l1_deg, b1_deg, l2_deg, b2_deg) -> np.ndarray:
    """Great-circle separation [deg] between (l1, b1) and (l2, b2), broadcastable.

    Thin wrapper over `astropy.coordinates.angular_separation` (the SkyCoord.separation
    kernel), evaluated directly on longitudes and latitudes - numerically stable at all
    separations, exactly periodic in longitude, regular at the poles.
    """
    sep = angular_separation(np.radians(l1_deg), np.radians(b1_deg),
                             np.radians(l2_deg), np.radians(b2_deg))
    return np.degrees(sep)


def separation_skycoord_deg(l1_deg, b1_deg, l2_deg, b2_deg) -> np.ndarray:
    """Same separation through explicit SkyCoord objects (reference implementation)."""
    c1 = SkyCoord(l=np.asarray(l1_deg) * u.deg, b=np.asarray(b1_deg) * u.deg,
                  frame="galactic")
    c2 = SkyCoord(l=np.asarray(l2_deg) * u.deg, b=np.asarray(b2_deg) * u.deg,
                  frame="galactic")
    return np.atleast_1d(c1.separation(c2).deg)


def validate_against_skycoord(n: int = 512, seed: int = 0, tol_deg: float = 1e-9) -> float:
    """Assert the bulk path agrees with SkyCoord.separation to machine precision.

    Returns the maximum absolute deviation [deg]; raises AssertionError beyond `tol_deg`.
    """
    rng = np.random.default_rng(seed)
    l1 = rng.uniform(0.0, 360.0, n); l2 = rng.uniform(0.0, 360.0, n)
    b1 = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, n)))
    b2 = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, n)))
    fast = separation_deg(l1, b1, l2, b2)
    ref = separation_skycoord_deg(l1, b1, l2, b2)
    worst = float(np.max(np.abs(fast - ref)))
    assert worst < tol_deg, f"separation implementations disagree by {worst} deg"
    return worst


def alpha_deg(l_deg, b_deg, events) -> np.ndarray:
    """alpha_i(s): separations [deg] between candidate(s) and every event axis.

    Parameters
    ----------
    l_deg, b_deg : scalars or (P,) arrays of candidate Galactic coordinates.
    events : AngularEvents

    Returns
    -------
    (N,) array for scalar input, (P, N) array for array input.
    """
    l_c = np.atleast_1d(np.asarray(l_deg, dtype=np.float64))
    b_c = np.atleast_1d(np.asarray(b_deg, dtype=np.float64))
    out = separation_deg(l_c[:, None], b_c[:, None],
                         events.l_axis_deg[None, :], events.b_axis_deg[None, :])
    return out[0] if np.isscalar(l_deg) or np.ndim(l_deg) == 0 else out


def residual_deg(l_deg, b_deg, events) -> np.ndarray:
    """Cone residual r_i(s) = alpha_i(s) - phi_i [deg].

    r_i = 0 exactly when the candidate lies ON event i's Compton cone (the small circle
    of angular radius phi_i about the axis); |r_i| is, to first order, the perpendicular
    angular distance from the candidate to that cone, because alpha_i is a geodesic
    distance function with unit-norm surface gradient (methodology Sec. I.2).
    """
    return alpha_deg(l_deg, b_deg, events) - (np.asarray(events.phi_deg)[None, :]
                                              if np.ndim(l_deg) else events.phi_deg)


def offset_by(l_deg, b_deg, position_angle_deg, separation_deg_):
    """Spherical offset from (l, b) by `separation` along `position_angle`.

    SkyCoord-native (directional_offset_by); used by the parametric bootstrap to draw
    replicate cone axes at radius alpha = phi + r about s_hat with uniform azimuth.
    Returns (l_new_deg, b_new_deg) arrays.
    """
    base = SkyCoord(l=np.asarray(l_deg) * u.deg, b=np.asarray(b_deg) * u.deg,
                    frame="galactic")
    off = base.directional_offset_by(np.asarray(position_angle_deg) * u.deg,
                                     np.asarray(separation_deg_) * u.deg)
    return np.atleast_1d(off.l.deg), np.atleast_1d(off.b.deg)


def local_chart_to_lb(u_deg, v_deg, l0_deg, b0_deg):
    """Locally metric-faithful chart around (l0, b0): u = (l - l0) cos b0, v = b - b0.

    Used only to hand a 2-D parameter pair to optimizers (Nelder-Mead, INR ascent);
    every chart point is mapped back to (l, b) BEFORE any angular evaluation, so no
    chart approximation enters any separation (methodology Sec. V).
    """
    cos_b0 = max(np.cos(np.radians(b0_deg)), 1e-6)
    l = np.mod(l0_deg + np.asarray(u_deg) / cos_b0, 360.0)
    b = np.clip(b0_deg + np.asarray(v_deg), -90.0, 90.0)
    return l, b


def lb_to_local_chart(l_deg, b_deg, l0_deg, b0_deg):
    """Inverse of `local_chart_to_lb` (longitude difference wrapped to [-180, 180))."""
    cos_b0 = max(np.cos(np.radians(b0_deg)), 1e-6)
    dl = (np.asarray(l_deg) - l0_deg + 180.0) % 360.0 - 180.0
    return dl * cos_b0, np.asarray(b_deg) - b0_deg
