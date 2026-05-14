import numpy as np
import pytest
import astropy.units as u
from astropy.coordinates import SkyCoord

from cosipy.event_selection import ARMSelector


CHI_COL = "Chi galactic"
PSI_COL = "Psi galactic"
PHI_COL = "Phi"


def make_events(chi_rad, psi_rad, phi_rad):
    return {
        CHI_COL: np.asarray(chi_rad, dtype=np.float64),
        PSI_COL: np.asarray(psi_rad, dtype=np.float64),
        PHI_COL: np.asarray(phi_rad, dtype=np.float64),
    }


def test_arm_selector_initialization():
    source = SkyCoord(l=0.0 * u.deg, b=0.0 * u.deg, frame="galactic")
    spacecraft_history = object()

    selector = ARMSelector(
        spacecraft_history=spacecraft_history,
        target_coord=source,
        arm_cut=13.0 * u.deg,
        chi_col=CHI_COL,
        psi_col=PSI_COL,
        phi_col=PHI_COL,
    )

    assert selector.spacecraft_history is spacecraft_history
    assert selector.arm_cut == 13.0 * u.deg
    assert selector.chi_col == CHI_COL
    assert selector.psi_col == PSI_COL
    assert selector.phi_col == PHI_COL
    assert np.isclose(selector.target_coord.l.to_value(u.deg), 0.0)
    assert np.isclose(selector.target_coord.b.to_value(u.deg), 0.0)


def test_compute_arm_values():
    source = SkyCoord(l=0.0 * u.deg, b=0.0 * u.deg, frame="galactic")

    selector = ARMSelector(
        spacecraft_history=None,
        target_coord=source,
        arm_cut=13.0 * u.deg,
        chi_col=CHI_COL,
        psi_col=PSI_COL,
        phi_col=PHI_COL,
    )

    # Source is at l=0 deg, b=0 deg.
    # Therefore, events at chi=0,10,30 deg and psi=0 deg have
    # angular separations of 0,10,30 deg from the source.
    chi_rad = np.deg2rad(np.array([0.0, 10.0, 30.0]))
    psi_rad = np.deg2rad(np.array([0.0, 0.0, 0.0]))
    phi_rad = np.deg2rad(np.array([0.0, 5.0, 0.0]))

    events = make_events(chi_rad, psi_rad, phi_rad)

    arm = selector._compute_arm(events).to_value(u.deg)

    expected_arm = np.array([0.0, 5.0, 30.0])

    np.testing.assert_allclose(arm, expected_arm, atol=1e-10)


def test_select_returns_boolean_mask():
    source = SkyCoord(l=0.0 * u.deg, b=0.0 * u.deg, frame="galactic")

    selector = ARMSelector(
        spacecraft_history=None,
        target_coord=source,
        arm_cut=13.0 * u.deg,
        chi_col=CHI_COL,
        psi_col=PSI_COL,
        phi_col=PHI_COL,
    )

    # ARM values will be 0, 10, and 20 deg.
    # With arm_cut = 13 deg, only the first two should be selected.
    chi_rad = np.deg2rad(np.array([0.0, 10.0, 20.0]))
    psi_rad = np.deg2rad(np.array([0.0, 0.0, 0.0]))
    phi_rad = np.deg2rad(np.array([0.0, 0.0, 0.0]))

    events = make_events(chi_rad, psi_rad, phi_rad)

    mask = selector._select(events)

    assert mask.dtype == bool
    np.testing.assert_array_equal(mask, np.array([True, True, False]))


def test_chi_psi_shape_mismatch_raises_value_error():
    source = SkyCoord(l=0.0 * u.deg, b=0.0 * u.deg, frame="galactic")

    selector = ARMSelector(
        spacecraft_history=None,
        target_coord=source,
        arm_cut=13.0 * u.deg,
        chi_col=CHI_COL,
        psi_col=PSI_COL,
        phi_col=PHI_COL,
    )

    events = make_events(
        chi_rad=np.array([0.0, 1.0, 2.0]),
        psi_rad=np.array([0.0, 1.0]),
        phi_rad=np.array([0.0, 0.0, 0.0]),
    )

    with pytest.raises(ValueError, match="Chi and Psi columns"):
        selector._compute_arm(events)


def test_phi_shape_mismatch_raises_value_error():
    source = SkyCoord(l=0.0 * u.deg, b=0.0 * u.deg, frame="galactic")

    selector = ARMSelector(
        spacecraft_history=None,
        target_coord=source,
        arm_cut=13.0 * u.deg,
        chi_col=CHI_COL,
        psi_col=PSI_COL,
        phi_col=PHI_COL,
    )

    events = make_events(
        chi_rad=np.array([0.0, 1.0, 2.0]),
        psi_rad=np.array([0.0, 1.0, 2.0]),
        phi_rad=np.array([0.0, 0.0]),
    )

    with pytest.raises(ValueError, match="Chi/Psi and Phi columns"):
        selector._compute_arm(events)
