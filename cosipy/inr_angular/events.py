"""Loading and on-burst selection of the angular event data for Method 1.

The Method-1 event input is ONLY the angular triplet per event

    (phi_i, l_axis_i, b_axis_i)

phi_i   : Compton scattering angle;
(l,b)_i : cone-axis direction in Galactic coordinates ("Chi galactic", "Psi galactic"
          columns of the cosipy unbinned FITS; lon_deg / lat_deg of the .npz cache).

Event times are used ONLY to apply the on-burst selection

    data_start_time = GRB_start_time - 0
    data_end_time   = GRB_end_time   + 0

Energies and interaction distances are deliberately NOT carried into the localization
data structure (methodology Sec. I.1): they may exist in the files but are excluded
features.  No off-burst events are used anywhere in Method 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from astropy.io import fits


@dataclass(frozen=True)
class AngularEvents:
    """The Method-1 event sample: angular triplets of the on-burst events only."""

    phi_deg: np.ndarray      # (N,) Compton scattering angle [deg]
    l_axis_deg: np.ndarray   # (N,) Galactic longitude of the cone axis [deg, 0..360)
    b_axis_deg: np.ndarray   # (N,) Galactic latitude  of the cone axis [deg, -90..90]
    times: np.ndarray        # (N,) event time tags [s] - selection bookkeeping only

    def __post_init__(self):
        n = len(self.phi_deg)
        for name in ("l_axis_deg", "b_axis_deg", "times"):
            arr = getattr(self, name)
            if arr.shape != (n,):
                raise ValueError(f"{name} has shape {arr.shape}, expected ({n},)")
        if not np.all(np.isfinite(self.phi_deg)):
            raise ValueError("non-finite scattering angles")
        if np.any((self.phi_deg <= 0.0) | (self.phi_deg >= 180.0)):
            raise ValueError("scattering angles must lie strictly inside (0, 180) deg")
        if np.any(np.abs(self.b_axis_deg) > 90.0):
            raise ValueError("axis latitudes outside [-90, 90] deg")

    def __len__(self) -> int:
        return len(self.phi_deg)

    @property
    def phi_rad(self) -> np.ndarray:
        return np.radians(self.phi_deg)


def _read_raw(path: Path) -> dict:
    """Read one unbinned event file (.fits/.fits.gz or cached .npz).

    Only the angular columns and times are returned; energies are intentionally not
    propagated (excluded feature).
    """
    path = Path(path)
    if path.suffix == ".npz":
        payload = np.load(path)
        keys = set(payload.files)
        if {"lon_deg", "lat_deg", "scatter_rad"} <= keys:
            return {
                "l_axis_deg": np.asarray(payload["lon_deg"], dtype=np.float64),
                "b_axis_deg": np.asarray(payload["lat_deg"], dtype=np.float64),
                "phi_rad": np.asarray(payload["scatter_rad"], dtype=np.float64),
                "times": np.asarray(payload["times"], dtype=np.float64),
            }
        if {"chi", "psi", "phi"} <= keys:
            # project background-cache schema: chi/psi/phi all in RADIANS
            return {
                "l_axis_deg": np.degrees(np.asarray(payload["chi"], dtype=np.float64)),
                "b_axis_deg": np.degrees(np.asarray(payload["psi"], dtype=np.float64)),
                "phi_rad": np.asarray(payload["phi"], dtype=np.float64),
                "times": np.asarray(payload["times"], dtype=np.float64),
            }
        raise ValueError(f"unrecognized npz event schema: {sorted(keys)}")
    with fits.open(path) as handle:
        data = handle[1].data
        return {
            "l_axis_deg": np.asarray(data["Chi galactic"], dtype=np.float64),
            "b_axis_deg": np.asarray(data["Psi galactic"], dtype=np.float64),
            "phi_rad": np.asarray(data["Phi"], dtype=np.float64),
            "times": np.asarray(data["TimeTags"], dtype=np.float64),
        }


def load_onburst_events(paths: Iterable[Path | str],
                        grb_start_time: float,
                        grb_end_time: float) -> AngularEvents:
    """Load event files and keep ONLY events inside the GRB interval.

    Implements the mandated selection with zero padding on either side:

        data_start_time = GRB_start_time - 0
        data_end_time   = GRB_end_time   + 0

    Parameters
    ----------
    paths : iterable of paths
        Unbinned event files (the GRB signal file and, if available, the cached
        background extraction covering the burst window - the burst interval of the
        background is part of the on-burst sample; off-burst events are dropped here
        and never seen again).
    grb_start_time, grb_end_time : float
        The GRB interval in the mission time scale.
    """
    data_start_time = float(grb_start_time) - 0
    data_end_time = float(grb_end_time) + 0
    parts = []
    for p in paths:
        if p is None:
            continue
        raw = _read_raw(Path(p))
        mask = (raw["times"] >= data_start_time) & (raw["times"] <= data_end_time)
        parts.append({k: v[mask] for k, v in raw.items()})
    if not parts:
        raise ValueError("no event files given")
    merged = {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
    if len(merged["times"]) == 0:
        raise ValueError("no events inside the GRB interval")
    return AngularEvents(
        phi_deg=np.degrees(merged["phi_rad"]),
        l_axis_deg=np.mod(merged["l_axis_deg"], 360.0),
        b_axis_deg=merged["b_axis_deg"],
        times=merged["times"],
    )


def find_data_file(file_name: str, search_dirs: Iterable[Path | str]) -> Optional[Path]:
    """Locate a data file across the usual project locations (first match wins)."""
    for d in search_dirs:
        cand = Path(d) / file_name
        if cand.exists():
            return cand
    return None


def find_background_cache(search_dirs: Iterable[Path | str],
                          stem: str,
                          data_start_time: float,
                          data_end_time: float) -> Optional[Path]:
    """Find a cached background extraction `<stem>_burst_<lo>_<hi>.npz` covering the window."""
    for d in search_dirs:
        for cand in sorted(Path(d).glob(f"{stem}_burst_*.npz")):
            try:
                lo, hi = (float(v) for v in cand.stem.split("_burst_")[1].split("_")[:2])
            except ValueError:
                continue
            if lo <= data_start_time and hi >= data_end_time:
                return cand
    return None
