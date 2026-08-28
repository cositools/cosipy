"""Result-saving utilities: JSON summary, ellipse text file, and map arrays."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()
                if not isinstance(v, np.ndarray) or v.size <= 64}
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def save_results(output_dir: Path | str, grb_name: str, summary: dict,
                 skymap=None, region=None, boot=None, tag: str = "") -> dict:
    """Persist the run: `<GRB>[_tag]_summary.json`, `_Ellipse_result.txt`, `_maps.npz`.

    The ellipse text file follows the project convention
    `Center_lon Center_lat a b theta TS_peak` used by the TS-map and INROld pipelines.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    paths = {}

    p_json = output_dir / f"{grb_name}{suffix}_summary.json"
    with open(p_json, "w") as fh:
        json.dump(_jsonable(summary), fh, indent=2)
    paths["summary_json"] = str(p_json)

    ell = summary.get("ellipse")
    if ell:
        # same file convention as the DC4 TS-map notebook's {GRB}_Ellipse_result.txt
        p_txt = output_dir / f"{grb_name}{suffix}_Ellipse_result.txt"
        best = summary.get("best_fit", {})
        results_array = np.array([
            best.get("l", np.nan), best.get("b", np.nan),
            ell["center_l_deg"], ell["center_b_deg"],
            ell["semi_major_deg"], ell["semi_minor_deg"],
            ell["position_angle_deg"], summary["ts_max"]])
        np.savetxt(p_txt, results_array,
                   header="Peak_l, Peak_b, Center_lon, Center_lat, a, b, theta, region_ts_peak",
                   fmt="%.8f")
        paths["ellipse_txt"] = str(p_txt)

    arrays = {}
    if skymap is not None:
        arrays.update({"map_nside": skymap.nside, "map_ipix": skymap.ipix,
                       "map_l_deg": skymap.l_deg, "map_b_deg": skymap.b_deg,
                       "map_score": skymap.score, "map_eta": skymap.eta})
    if region is not None:
        arrays.update({"region_nside": np.array([region["nside"]]),
                       "region_members": region["members"],
                       "region_members_uniq": region["members_uniq"],
                       "region_area_deg2": np.array([region["area_deg2"]]),
                       "region_boundary_l_deg": region["boundary_l_deg"],
                       "region_boundary_b_deg": region["boundary_b_deg"]})
    if boot is not None:
        arrays.update({"bootstrap_deltas": boot["deltas"]})
    if arrays:
        p_npz = output_dir / f"{grb_name}{suffix}_maps.npz"
        np.savez_compressed(p_npz, **arrays)
        paths["maps_npz"] = str(p_npz)
    return paths


def save_healpix_map_fits(output_dir: Path | str, file_name: str, m) -> str:
    """Write an mhealpy HealpixMap (single- or multi-order/MOC) to FITS, like the
    TS-map notebook's `{GRB}_TS_Map.fits`."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / file_name
    m.write_map(str(path), overwrite=True)
    return str(path)


def save_ellipse_result_txt(path, peak_lb, inr_lb, ellipse: dict,
                            region_score_peak: float) -> str:
    """{GRB}_Ellipse_result.txt: score-map peak pixel, INR location, ellipse summary.

    One value per line in header order, np.savetxt convention (as in the TS-map
    pipeline's result file).
    """
    results_array = np.array([
        peak_lb[0], peak_lb[1],
        inr_lb[0], inr_lb[1],
        ellipse["center_l_deg"], ellipse["center_b_deg"],
        ellipse["semi_major_deg"], ellipse["semi_minor_deg"],
        ellipse["r_eq_deg"], ellipse["theta_deg"],
        region_score_peak,
    ])
    np.savetxt(path, results_array,
               header="Peak_l Peak_b INR_l INR_b Center_lon Center_lat "
                      "Semi_major_axis Semi_minor_axis Equivalent_radius Theta "
                      "Region_score_peak",
               fmt="%.8f")
    return str(path)
