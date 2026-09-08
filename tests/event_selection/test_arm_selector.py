import numpy as np
import pytest
import astropy.units as u
from astropy.coordinates import SkyCoord
from scoords import SpacecraftFrame

from cosipy.event_selection import ARMSelector
from cosipy.data_io.EmCDSUnbinnedData import EmCDSEventDataInSCFrameFromArrays


def make_source_sc(lon_deg=0.0, lat_deg=0.0):
    return SkyCoord(
        lon=lon_deg * u.deg,
        lat=lat_deg * u.deg,
        frame=SpacecraftFrame(),
    )


def make_events(scattered_lon_deg, scattered_lat_deg, phi_deg):
    scattered_lon_deg = np.asarray(scattered_lon_deg, dtype=np.float64)
    scattered_lat_deg = np.asarray(scattered_lat_deg, dtype=np.float64)
    phi_deg = np.asarray(phi_deg, dtype=np.float64)

    return EmCDSEventDataInSCFrameFromArrays(
        energy_keV=np.full(scattered_lon_deg.size, 1000.0),
        scattered_lon_rad_sc=np.deg2rad(scattered_lon_deg),
        scattered_lat_rad_sc=np.deg2rad(scattered_lat_deg),
        scatt_angle_rad=np.deg2rad(phi_deg),
    )


def test_arm_selector_initialization():
    source = make_source_sc()

    selector = ARMSelector(
        target_coord=source,
        arm_cut=13.0 * u.deg,
    )

    assert selector.target_coord is source
    assert selector.arm_cut == 13.0 * u.deg
    assert selector.polarization_convention is not None


def test_target_coord_must_be_skycoord():
    with pytest.raises(
        TypeError,
        match="target_coord must be an astropy.coordinates.SkyCoord object",
    ):
        ARMSelector(
            target_coord=object(),
            arm_cut=13.0 * u.deg,
        )


def test_target_coord_must_be_spacecraft_frame():
    source = SkyCoord(
        l=0.0 * u.deg,
        b=0.0 * u.deg,
        frame="galactic",
    )

    with pytest.raises(
        ValueError,
        match="target_coord must be in SpacecraftFrame",
    ):
        ARMSelector(
            target_coord=source,
            arm_cut=13.0 * u.deg,
        )


def test_invalid_polarization_convention_type_raises_type_error():
    source = make_source_sc()

    with pytest.raises(
        TypeError,
        match="polarization_convention must be either",
    ):
        ARMSelector(
            target_coord=source,
            arm_cut=13.0 * u.deg,
            polarization_convention=object(),
        )


def test_compute_arm_values_from_event_interface():
    source = make_source_sc()

    selector = ARMSelector(
        target_coord=source,
        arm_cut=13.0 * u.deg,
    )

    # Source is at spacecraft lon=0 deg, lat=0 deg.
    # Events at lon=0, 10, 30 deg and lat=0 deg have angular distances
    # 0, 10, and 30 deg from the source.
    # ARM = angular_distance - Phi.
    events = make_events(
        scattered_lon_deg=[0.0, 10.0, 30.0],
        scattered_lat_deg=[0.0, 0.0, 0.0],
        phi_deg=[0.0, 5.0, 0.0],
    )

    arm = selector._compute_arm(events).to_value(u.deg)

    expected_arm = np.array([0.0, 5.0, 30.0])

    np.testing.assert_allclose(arm, expected_arm, atol=1e-10)


def test_select_returns_boolean_mask_from_event_interface():
    source = make_source_sc()

    selector = ARMSelector(
        target_coord=source,
        arm_cut=13.0 * u.deg,
    )

    # ARM values are 0, 10, and 20 deg.
    # With arm_cut = 13 deg, only the first two events should be selected.
    events = make_events(
        scattered_lon_deg=[0.0, 10.0, 20.0],
        scattered_lat_deg=[0.0, 0.0, 0.0],
        phi_deg=[0.0, 0.0, 0.0],
    )

    mask = selector._select(events)

    assert mask.dtype == bool
    np.testing.assert_array_equal(mask, np.array([True, True, False]))


def test_public_select_method_works_from_event_interface():
    source = make_source_sc()

    selector = ARMSelector(
        target_coord=source,
        arm_cut=13.0 * u.deg,
    )

    events = make_events(
        scattered_lon_deg=[0.0, 30.0],
        scattered_lat_deg=[0.0, 0.0],
        phi_deg=[0.0, 0.0],
    )

    mask = selector.select(events)

    np.testing.assert_array_equal(mask, np.array([True, False]))
