#!/usr/bin/env python
# coding: UTF-8

"""
Benchmark IRFRelativeHistUnpolarized's nthreads/batch_size parallelization
of effective_area_cm2()/differential_effective_area_cm2() (see
cosipy/response/relative_irf_hist.py).

This is a standalone timing script, not a pytest test: it prints a table
of wall time and speedup vs. nthreads=1, and has no pass/fail assertions.
Its purpose is to sanity-check (and, if needed, adjust) the batch_size
default against a realistically-sized histogram -- the number that
actually matters is the *speedup*, not the absolute times, which will vary
with the machine this runs on.

By default it builds a synthetic histogram of configurable (but
"realistic", in the sense of matching production axis sizes) shape --
set REAL_IRF_PATH below to a real HDF5 file to benchmark against a real
response's actual bin sizes/values instead, which is what you should do
before relying on this to justify a default.
"""

import time
from pathlib import Path

import numpy as np
import astropy.units as u
from histpy import Axis, Axes, HealpixAxis, Histogram
from scoords import SpacecraftFrame

from cosipy.polarization import PolarizationAxis, StereographicConvention
from cosipy.response.relative_irf_hist import IRFRelativeHistUnpolarized

# Set to a path to benchmark against a real response file instead of the
# synthetic one built by _build_synthetic_irf_hist() below.
REAL_IRF_PATH = None  # e.g. Path("/path/to/ResponseContinuum.area.relative.nonsparse.h5")

# Synthetic histogram shape, used only if REAL_IRF_PATH is None. These are
# in the ballpark of a real production response (see the module docstring
# of relative_hist_irf_from_nf_response.py in this same directory for the
# scale discussion), but kept modest so building it in-memory here is
# quick -- adjust to more closely match what you actually run on.
SYNTHETIC_NSIDE = 8
SYNTHETIC_N_EI = 10
SYNTHETIC_N_EPSILON = 20
SYNTHETIC_N_PHI = 30
SYNTHETIC_N_THETA = 30
SYNTHETIC_N_ZETA = 30

N_POINTS = [100, 1_000, 10_000, 100_000, 1_000_000]
NTHREADS = [1, 2, 4, 8]

# batch_size to use for every nthreads > 1 case below. This is the number
# being sanity-checked -- lower it and re-run if the speedup at your
# smallest relevant N is worse than expected, raise it if larger N are
# still slower than nthreads=1.
BATCH_SIZE = 20_000


class _FakePhotonList:
    def __init__(self, lon_rad, lat_rad, energy_keV):
        self.direction_lon_rad_sc = lon_rad
        self.direction_lat_rad_sc = lat_rad
        self.energy_keV = energy_keV


class _FakeEventData:
    def __init__(self, lon_rad, lat_rad, phi_rad, energy_keV):
        self.scattered_lon_rad_sc = lon_rad
        self.scattered_lat_rad_sc = lat_rad
        self.scattering_angle_rad = phi_rad
        self.energy_keV = energy_keV


def _build_synthetic_irf_hist():
    rng = np.random.default_rng(0)

    axes = Axes([
        HealpixAxis(nside=SYNTHETIC_NSIDE, scheme='ring', coordsys=SpacecraftFrame(), label='NuLambda'),
        Axis(np.geomspace(100, 10000, SYNTHETIC_N_EI + 1) * u.keV, label='Ei', scale='log'),
        Axis(np.linspace(-1, 1, SYNTHETIC_N_EPSILON + 1), label='Epsilon'),
        Axis(np.linspace(0, 180, SYNTHETIC_N_PHI + 1) * u.deg, label='Phi'),
        Axis(np.linspace(-180, 180, SYNTHETIC_N_THETA + 1) * u.deg, label='Theta'),
        PolarizationAxis(np.linspace(0, 360, SYNTHETIC_N_ZETA + 1) * u.deg,
                         convention=StereographicConvention(), label='Zeta'),
    ])

    contents = rng.random(axes.nbins)

    return Histogram(axes, contents=contents, unit=u.cm * u.cm)


def _make_photons_and_events(n, seed=0):
    rng = np.random.default_rng(seed)

    photon_lon = rng.uniform(-np.pi, np.pi, n)
    photon_lat = rng.uniform(-np.pi / 2, np.pi / 2, n)
    photon_energy = rng.uniform(100, 1000, n)

    psichi_lon = rng.uniform(-np.pi, np.pi, n)
    psichi_lat = rng.uniform(-np.pi / 2, np.pi / 2, n)
    phi_kin = rng.uniform(0, np.pi, n)
    measured_energy = rng.uniform(100, 1000, n)

    photons = _FakePhotonList(photon_lon, photon_lat, photon_energy)
    events = _FakeEventData(psichi_lon, psichi_lat, phi_kin, measured_energy)

    return photons, events


def _time_call(fn, *args, reps):
    # Call once, untimed, to shake out any one-time setup cost that isn't
    # representative of steady-state calls (e.g. lazy imports inside the
    # interpolation path).
    fn(*args)

    t0 = time.perf_counter()
    for _ in range(reps):
        fn(*args)
    t1 = time.perf_counter()

    return (t1 - t0) / reps


def _load_or_build_irf_hist():
    if REAL_IRF_PATH is not None:
        return Histogram.open(str(REAL_IRF_PATH), "IRF")
    else:
        return _build_synthetic_irf_hist()


def main():
    if REAL_IRF_PATH is not None:
        print(f"Benchmarking against real IRF at {REAL_IRF_PATH}")
    else:
        print("Benchmarking against a synthetic IRF "
             f"(nside={SYNTHETIC_NSIDE}, "
             f"Ei/Epsilon/Phi/Theta/Zeta bins="
             f"{SYNTHETIC_N_EI}/{SYNTHETIC_N_EPSILON}/{SYNTHETIC_N_PHI}/{SYNTHETIC_N_THETA}/{SYNTHETIC_N_ZETA})")

    print(f"{'method':<30} {'N':>10} {'nthreads':>9} {'time/call (ms)':>16} {'speedup':>9}")

    baseline = {}  # (method_name, n) -> nthreads=1 time, for the speedup column

    for nthreads in NTHREADS:
        # A fresh histogram per model rather than one shared/copied
        # instance: IRFRelativeHistUnpolarized.__init__ mutates its axes
        # in place (unit standardization) even under copy=True, since
        # Histogram.copy() doesn't deep-copy individual Axis objects --
        # reusing one across models would corrupt each other's axes.
        model = IRFRelativeHistUnpolarized(_load_or_build_irf_hist(), nthreads=nthreads,
                                           batch_size=BATCH_SIZE, copy=False)

        for method_name in ["effective_area_cm2", "differential_effective_area_cm2"]:
            for n in N_POINTS:
                photons, events = _make_photons_and_events(n)

                reps = max(1, min(50, 100_000 // max(n, 1)))

                if method_name == "effective_area_cm2":
                    fn = model._effective_area_cm2
                    args = (photons,)
                else:
                    fn = model._differential_effective_area_cm2
                    args = (photons, events)

                t = _time_call(fn, *args, reps=reps)

                key = (method_name, n)
                if nthreads == 1:
                    baseline[key] = t
                speedup = baseline[key] / t

                print(f"{method_name:<30} {n:>10} {nthreads:>9} {t * 1000:>16.3f} {speedup:>8.2f}x")


if __name__ == "__main__":
    main()
