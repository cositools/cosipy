#!/usr/bin/env python
# coding: UTF-8

"""
Build the HDF5 input expected by IRFRelativeHistUnpolarized.from_h5() out of
a pair of MEGAlib .rsp.gz responses binned in relative coordinates (see
MEGAlib's feature/binned-imaging-reparametrization branch): one for the full
6D differential response (NuLambda, Ei, Epsilon, Phi, Theta, Zeta), and one
for the total effective area alone (NuLambda, Ei).

Pipeline
--------
1. Convert both .rsp.gz files to HDF5 with RspConverter.
2. Build the standalone total effective area histogram ("AEFF" group):
   raw counts * per-Ei EFF_AREA correction, projected onto (NuLambda, Ei).
3. Build the full differential response ("IRF" group):
   a. Smooth the raw per-bin counts by blending them towards the
      Ei-conditional marginal distributions of Epsilon, (Phi, Theta) and
      Zeta (see _smooth_counts() and the "Smoothing" section of
      https://github.com/user-attachments/files/31315263/20260428-BinnedRelResponse-cosipy-Israel.pdf)
      -- bins with too little
      local statistics borrow more from the smooth marginal, at the cost of
      washing out real correlations, controlled by `smoothing_k`.
   b. Convert to effective area (same per-Ei EFF_AREA correction as above).
   c. Zero out bins that are kinematically unphysical (Phi + Theta outside
      [0, pi], i.e. zero (Phi, Theta, Zeta) phase space) -- MEGAlib never
      populates these, but the smoothing step above can leave stray NaNs
      there (0/0), so they are masked directly instead of divided out.
   d. Tag the Zeta axis as a PolarizationAxis with the response's
      polarization convention (default "RelativeX"), as
      IRFRelativeHistUnpolarized requires.
4. Write both histograms to the same output HDF5 file, so
   IRFRelativeHistUnpolarized.from_h5() picks up the "AEFF" group
   automatically.

Note on units: bin *contents* are written as per-bin effective area (cm^2),
i.e. NOT divided by phase-space volume -- IRFRelativeHistUnpolarized does
that division itself at load time to get the differential effective area it
interpolates on (see its class docstring).
"""

import logging
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Union

import numpy as np
import h5py as h5

import astropy.units as u

from histpy import Histogram, Axes

from cosipy.response import RspConverter
from cosipy.response.relative_coordinates import RelativeCDSCoordinates
from cosipy.polarization import PolarizationAxis

logger = logging.getLogger(__name__)


def _default_h5_name(rsp_path: Union[str, Path]) -> str:
    """Same ".rsp"/".gz" -> ".h5" filename derivation as
    RspConverter.convert_to_h5() uses when h5_filename is not given, but
    returning just the filename (not the full path), so callers can place
    it under a directory of their choosing."""

    h5_path = Path(rsp_path)
    while h5_path.suffix in {".rsp", ".gz"}:
        h5_path = h5_path.with_suffix("")

    return h5_path.name + ".h5"


def convert_rsp_to_h5(irf_rsp_path: Union[str, Path],
                      aeff_rsp_path: Union[str, Path],
                      irf_h5_path: Union[str, Path] = None,
                      aeff_h5_path: Union[str, Path] = None,
                      intermediate_dir: Union[str, Path] = None,
                      pa_convention: str = "RelativeX",
                      overwrite: bool = False):
    """
    Convert the differential-response and aeff-only .rsp.gz files to HDF5
    (RspConverter's native "DRM"-group format, not yet the final
    IRFRelativeHistUnpolarized-compatible layout).

    Parameters
    ----------
    irf_h5_path, aeff_h5_path : str or Path, optional
        Explicit output paths. If not given, default to
        _default_h5_name(irf_rsp_path/aeff_rsp_path) under
        `intermediate_dir` (or, if `intermediate_dir` is also None, next
        to the input .rsp.gz file, i.e. RspConverter's own default).
    intermediate_dir : str or Path, optional
        Directory the default irf_h5_path/aeff_h5_path are placed under.
        Created if it doesn't exist. Ignored for a path given explicitly.
        See `intermediate_dir` in build_relative_hist_irf_from_rsp() for
        why you'd want this instead of the (source-directory) default --
        namely, to avoid colliding with an unrelated same-named file
        there, and to control where the (fairly large) intermediate
        files end up.

    Returns
    -------
    (Path, Path)
        Paths to the converted (irf, aeff) HDF5 files.
    """

    if intermediate_dir is not None:
        intermediate_dir = Path(intermediate_dir)
        intermediate_dir.mkdir(parents=True, exist_ok=True)

        if irf_h5_path is None:
            irf_h5_path = intermediate_dir / _default_h5_name(irf_rsp_path)
        if aeff_h5_path is None:
            aeff_h5_path = intermediate_dir / _default_h5_name(aeff_rsp_path)

    converter = RspConverter()

    irf_h5_path = converter.convert_to_h5(irf_rsp_path, irf_h5_path,
                                          pa_convention=pa_convention,
                                          overwrite=overwrite)

    aeff_h5_path = converter.convert_to_h5(aeff_rsp_path, aeff_h5_path,
                                           pa_convention=pa_convention,
                                           overwrite=overwrite)

    return Path(irf_h5_path), Path(aeff_h5_path)


def build_aeff_hist(aeff_drm_h5_path: Union[str, Path]) -> Histogram:
    """
    Build the standalone total-effective-area histogram (axes
    ['NuLambda', 'Ei'], in cm^2) out of an RspConverter-produced HDF5 file,
    for the "AEFF" group of the final output.
    """

    with h5.File(aeff_drm_h5_path, 'r') as f:
        drm = f['DRM']
        counts = np.asarray(drm['COUNTS'], dtype=float)
        axes = Axes.open(drm["AXES"])
        aeff_corr = np.asarray(drm['EFF_AREA'])  # per-Ei cm^2/count, already averaged over NuLambda npix

    h = Histogram(axes, contents=counts, dtype=float, copy_contents=True)

    h *= axes.expand_dims(aeff_corr, axes.label_to_index("Ei"))
    h.to(u.cm * u.cm, update=False, copy=False)

    return h.project('NuLambda', 'Ei')


def _phase_space(axes: Axes) -> np.ndarray:
    """Phase-space volume of every (Phi, Theta, Zeta) bin."""

    phi_edges_mesh, theta_edges_mesh, zeta_edges_mesh = np.meshgrid(
        axes['Phi'].edges.to_value(u.rad),
        axes['Theta'].edges.to_value(u.rad),
        axes['Zeta'].edges.to_value(u.rad),
        indexing='ij')

    return RelativeCDSCoordinates.get_relative_cds_phase_space(
        phi_edges_mesh[:-1, :-1, :-1], phi_edges_mesh[1:, :-1, :-1],
        theta_edges_mesh[:-1, :-1, :-1], theta_edges_mesh[:-1, 1:, :-1],
        zeta_edges_mesh[:-1, :-1, :-1], zeta_edges_mesh[:-1, :-1, 1:])


def _smooth_counts(h: Histogram, smoothing_k: float) -> np.ndarray:
    """
    Blend the raw per-bin counts of `h` (axes ['NuLambda', 'Ei', 'Epsilon',
    'Phi', 'Theta', 'Zeta']) towards the Ei-conditional marginal
    distributions of Epsilon, (Phi, Theta) and Zeta, to fill in bins that
    don't have enough statistics on their own.

    `smoothing_k` controls the trade-off (Poisson-with-Gamma-prior blend,
    mu=mean, stdev=sqrt(smoothing_k)*n): smoothing_k=0 disables smoothing
    entirely (returns the original counts); larger values weight the smooth
    marginal-distribution estimate more heavily, at the cost of washing out
    real correlations between axes.

    Returns
    -------
    numpy.ndarray
        Smoothed counts, same shape as h.contents, still in raw-count
        units and normalized to preserve each (NuLambda, Ei) bin's total
        count.
    """

    def mean_dist(vs_axes, dist_axes):
        all_axes = vs_axes + dist_axes
        dist = h.project(all_axes)
        norm = dist.axes.expand_dims(h.project(vs_axes), vs_axes) if vs_axes else np.sum(h)
        dist = dist / norm
        return h.axes.expand_dims(dist, all_axes)

    epsilon_dist = mean_dist(['Ei'], ['Epsilon'])
    cds_dist = mean_dist(['Ei'], ['Phi', 'Theta']) * mean_dist(['Ei'], ['Zeta'])

    trig_counts = h.axes.expand_dims(h.project(['NuLambda', 'Ei']), ['NuLambda', 'Ei'])

    mean_response = trig_counts * epsilon_dist * cds_dist

    counts = h.contents
    smoothed = (counts ** 3 + smoothing_k * mean_response ** 2) / (counts ** 2 + smoothing_k * mean_response)

    # Renormalize so each (NuLambda, Ei) bin's total count is unchanged.
    norm = h.axes.expand_dims(np.nansum(smoothed, axis=(2, 3, 4, 5)), ['NuLambda', 'Ei'])
    smoothed = smoothed * trig_counts / norm

    # NaNs remain wherever a bin has zero local statistics on all sides
    # (0/0) -- fix those (and any -- expected zero -- unphysical bins) to 0.
    smoothed = np.where(np.isnan(smoothed), 0, smoothed)

    return smoothed


def build_irf_hist(irf_drm_h5_path: Union[str, Path],
                   pa_convention: str = "RelativeX",
                   smoothing_k: float = 0) -> Histogram:
    """
    Build the full differential-response histogram (axes ['NuLambda', 'Ei',
    'Epsilon', 'Phi', 'Theta', 'Zeta'], contents in cm^2) out of an
    RspConverter-produced HDF5 file, for the "IRF" group of the final
    output. See the module docstring for the pipeline.
    """

    with h5.File(irf_drm_h5_path, 'r') as f:
        drm = f['DRM']
        counts = np.asarray(drm['COUNTS'], dtype=float)
        axes = Axes.open(drm["AXES"])
        aeff_corr = np.asarray(drm['EFF_AREA'])  # per-Ei cm^2/count

    h = Histogram(axes, contents=counts, dtype=float, copy_contents=True)

    phase_space_cds = _phase_space(axes)  # (nPhi, nTheta, nZeta)
    unphysical = axes.broadcast(phase_space_cds <= 0, ['Phi', 'Theta', 'Zeta'])

    h[:] = _smooth_counts(h, smoothing_k)

    # To effective area (cm^2).
    h *= axes.expand_dims(aeff_corr, axes.label_to_index("Ei"))
    h.to(u.cm * u.cm, update=False, copy=False)

    # Kinematically unphysical (Phi, Theta) bins (zero phase space) should
    # already be ~0 (MEGAlib can't populate them), but the smoothing step
    # can leave stray numerical noise there -- mask them explicitly.
    h[unphysical] = 0 * h.unit

    # Tag the Zeta axis with the response's polarization convention, as
    # required by IRFRelativeHistUnpolarized.
    h.axes['Zeta'] = PolarizationAxis(h.axes['Zeta'], convention=pa_convention)

    return h


@contextmanager
def _resolve_intermediate_dir(intermediate_dir: Optional[Union[str, Path]]):
    """
    Yield `intermediate_dir` as given, or, if None, a freshly created
    temporary directory that is removed again on exit (including on
    error) -- so a default run doesn't leave the (fairly large)
    intermediate per-rsp HDF5 files lying around, or risk colliding with
    an unrelated same-named file next to the input .rsp.gz.
    """

    if intermediate_dir is not None:
        yield Path(intermediate_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="relative_hist_irf_from_rsp_") as tmp_dir:
            yield Path(tmp_dir)


def build_relative_hist_irf_from_rsp(irf_rsp_path: Union[str, Path],
                                     aeff_rsp_path: Union[str, Path],
                                     output_path: Union[str, Path],
                                     intermediate_dir: Optional[Union[str, Path]] = None,
                                     pa_convention: str = "RelativeX",
                                     smoothing_k: float = 0,
                                     overwrite: bool = False) -> Path:
    """
    End-to-end: convert both .rsp.gz files to HDF5, build the weighted
    "IRF" and "AEFF" histograms, and write them to `output_path` -- ready
    to be loaded with IRFRelativeHistUnpolarized.from_h5(output_path).

    Parameters
    ----------
    intermediate_dir : str or Path, optional
        Directory the intermediate RspConverter-produced HDF5 files
        (irf/aeff, in RspConverter's native "DRM" layout -- see
        convert_rsp_to_h5()) are written to. Defaults to a temporary
        directory that is removed again once this function returns (or
        raises), so it neither collides with an unrelated same-named
        file next to the input .rsp.gz files (RspConverter's own
        default location) nor leaves those (fairly large) intermediate
        files behind. Pass an explicit directory to keep them instead,
        e.g. to reuse across repeated runs.
    """

    output_path = Path(output_path)

    with _resolve_intermediate_dir(intermediate_dir) as resolved_intermediate_dir:
        irf_drm_path, aeff_drm_path = convert_rsp_to_h5(
            irf_rsp_path, aeff_rsp_path, intermediate_dir=resolved_intermediate_dir,
            pa_convention=pa_convention, overwrite=overwrite)

        aeff_hist = build_aeff_hist(aeff_drm_path)
        irf_hist = build_irf_hist(irf_drm_path, pa_convention=pa_convention, smoothing_k=smoothing_k)

    aeff_hist.write(str(output_path), name="AEFF", overwrite=overwrite)
    irf_hist.write(str(output_path), name="IRF", overwrite=overwrite)

    return output_path


def _parse_args():

    import argparse

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--irf-rsp-path", type=Path, required=True,
                   help="Path to the full 6D differential-response .rsp(.gz) file.")
    p.add_argument("--aeff-rsp-path", type=Path, required=True,
                   help="Path to the aeff-only .rsp(.gz) file.")
    p.add_argument("--output-path", type=Path, required=True,
                   help="Path to write the final HDF5 file to (loadable with "
                        "IRFRelativeHistUnpolarized.from_h5()).")
    p.add_argument("--intermediate-dir", type=Path, default=None,
                   help="Directory for the intermediate RspConverter-produced HDF5 files. "
                        "Default: a temporary directory, removed again once the script exits.")
    p.add_argument("--pa-convention", default="RelativeX",
                   help="Polarization angle convention for the Zeta axis. Default: %(default)s")
    p.add_argument("--smoothing-k", type=float, default=0,
                   help="Mean-response smoothing weight -- see build_irf_hist()/_smooth_counts(). "
                        "0 disables smoothing. Default: %(default)s")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite output_path (and any intermediate converted HDF5 files) if it exists.")

    return p.parse_args()


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    args = _parse_args()

    output_path = build_relative_hist_irf_from_rsp(
        irf_rsp_path=args.irf_rsp_path,
        aeff_rsp_path=args.aeff_rsp_path,
        output_path=args.output_path,
        intermediate_dir=args.intermediate_dir,
        pa_convention=args.pa_convention,
        smoothing_k=args.smoothing_k,
        overwrite=args.overwrite)

    print(f"Wrote {output_path}")
