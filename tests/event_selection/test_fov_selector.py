from typing import Iterable

import numpy as np
import pytest
import astropy.units as u

from cosipy.event_selection.fov_selection import FOVSelector
from cosipy.interfaces.data_interface import ComptonDataSpaceInSCFrameEventDataInterface
from cosipy.util.iterables import asarray

# Dummy events: scattering angle phi and spacecraft-frame latitude of the psichi direction.
# lat = 90 deg means the psichi direction is exactly at the spacecraft zenith.
phi_deg = np.array([10., 10., 10., 30., 30., 90.])
lat_deg = np.array([90., 70., 50., 70., 10., 0.])


class DummyFOVEventData(ComptonDataSpaceInSCFrameEventDataInterface):

    @property
    def scattering_angle_rad(self) -> Iterable[float]:
        return np.deg2rad(phi_deg)

    @property
    def scattered_lat_rad_sc(self) -> Iterable[float]:
        return np.deg2rad(lat_deg)

    @property
    def scattered_lon_rad_sc(self) -> Iterable[float]:
        return np.zeros_like(phi_deg)


events = DummyFOVEventData()


def test_fov_selector_requires_cap_angle():
    with pytest.raises(ValueError):
        FOVSelector(cutvalue=0.5)


def test_fov_selector_requires_cutvalue():
    with pytest.raises(ValueError):
        FOVSelector(cap_angle=30 * u.deg)


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_fov_selector_rejects_bad_cutvalue(bad):
    with pytest.raises(ValueError):
        FOVSelector(cap_angle=30 * u.deg, cutvalue=bad)


def test_fov_selector_fully_inside_cap_is_always_kept():
    # psi=0 (ring centered on zenith), circle fully inside a wide cap -> f_fov = 1
    selector = FOVSelector(cap_angle=90 * u.deg, cutvalue=1.0)

    class OnAxis(ComptonDataSpaceInSCFrameEventDataInterface):
        @property
        def scattering_angle_rad(self):
            return np.array([np.deg2rad(5.)])

        @property
        def scattered_lat_rad_sc(self):
            return np.array([np.deg2rad(90.)])

        @property
        def scattered_lon_rad_sc(self):
            return np.array([0.])

    mask = asarray(selector.select(OnAxis()), dtype=bool)
    assert np.all(mask)


def test_fov_selector_fully_outside_cap_is_always_rejected():
    # Ring centered far from the zenith, cap much smaller than the distance to the ring -> f_fov = 0
    selector = FOVSelector(cap_angle=5 * u.deg, cutvalue=1e-6)

    class FarOffAxis(ComptonDataSpaceInSCFrameEventDataInterface):
        @property
        def scattering_angle_rad(self):
            return np.array([np.deg2rad(5.)])

        @property
        def scattered_lat_rad_sc(self):
            return np.array([np.deg2rad(0.)])  # psi = 90 deg from zenith

        @property
        def scattered_lon_rad_sc(self):
            return np.array([0.])

    mask = asarray(selector.select(FarOffAxis()), dtype=bool)
    assert not np.any(mask)


def test_fov_selector_matches_hand_computed_fraction():
    # psi = 0 (on-axis at zenith), cap radius 20 deg, circle radius phi = 30 deg.
    # By symmetry, half the circle (the half within 20 deg of zenith on the near side)
    # lies on each side, but since the whole circle sits at colatitude 30 deg > 20 deg cap
    # radius, the cap and the circle only touch, except the circle passes through points
    # at colatitude exactly 30 deg for every azimuth -- i.e. it's entirely outside a 20 deg cap.
    selector = FOVSelector(cap_angle=20 * u.deg, cutvalue=1e-6)

    class OnAxisWide(ComptonDataSpaceInSCFrameEventDataInterface):
        @property
        def scattering_angle_rad(self):
            return np.array([np.deg2rad(30.)])

        @property
        def scattered_lat_rad_sc(self):
            return np.array([np.deg2rad(90.)])  # psi = 0

        @property
        def scattered_lon_rad_sc(self):
            return np.array([0.])

    ffov = FOVSelector.calculate_cap_fraction(
        np.cos(0.0), np.deg2rad(30.), np.cos(np.deg2rad(20.)))
    assert np.isclose(ffov, 0.0)

    mask = asarray(selector.select(OnAxisWide()), dtype=bool)
    assert not np.any(mask)


def test_fov_selector_partial_overlap_fraction_threshold():
    # psi = 30 deg, phi = 30 deg, cap = 20 deg: the ring straddles the cap boundary
    # (closest point at |psi-phi|=0 deg is inside, farthest at psi+phi=60 deg is outside),
    # so 0 < f_fov < 1.
    ffov = FOVSelector.calculate_cap_fraction(
        np.cos(np.deg2rad(30.)), np.deg2rad(30.), np.cos(np.deg2rad(20.)))
    assert 0.0 < ffov < 1.0

    strict_selector = FOVSelector(cap_angle=20 * u.deg, cutvalue=float(ffov) + 0.05)
    loose_selector = FOVSelector(cap_angle=20 * u.deg, cutvalue=float(ffov) - 0.05)

    class Straddling(ComptonDataSpaceInSCFrameEventDataInterface):
        @property
        def scattering_angle_rad(self):
            return np.array([np.deg2rad(30.)])

        @property
        def scattered_lat_rad_sc(self):
            return np.array([np.deg2rad(60.)])  # psi = 30 deg

        @property
        def scattered_lon_rad_sc(self):
            return np.array([0.])

    assert not np.any(asarray(strict_selector.select(Straddling()), dtype=bool))
    assert np.any(asarray(loose_selector.select(Straddling()), dtype=bool))


def test_fov_selector_batch_mask():
    # cap = 60 deg: keep rings with at least half their circumference inside the cap.
    selector = FOVSelector(cap_angle=60 * u.deg, cutvalue=0.5)

    mask = asarray(selector.select(events), dtype=bool)

    assert mask.shape == phi_deg.shape
    # on-axis, small circle well inside the cap -> kept
    assert mask[0]
    # far off-axis (psi=90, phi=90) circle through the zenith but mostly outside -> rejected
    assert not mask[-1]
