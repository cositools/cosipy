import numpy as np
import pytest

import astropy.units as u
from astropy.coordinates import Angle, SkyCoord
from astropy.time import Time
from scoords import SpacecraftFrame

from cosipy.data_io.UnBinnedData import UnBinnedData
from cosipy.data_io.EmCDSUnbinnedData import (
    TimeTagEmCDSEventInSCFrame,
    TimeTagEmCDSDistanceEventInSCFrame,
    TimeTagEmCDSEventDataInSCFrameFromArrays,
    TimeTagEmCDSDistanceEventDataInSCFrameFromArrays,
    TimeTagEmCDSEventDataInSCFrameFromDC3Fits,
    TimeTagEmCDSDistanceEventDataInSCFrameFromDC3Fits,
)
from cosipy.event_selection.distance_selection import DistanceSelector


def test_time_tag_emcds_distance_event():
    time = Time(60000.0, format='jd')

    event = TimeTagEmCDSDistanceEventInSCFrame(time.jd1, time.jd2,
                                                energy=511.0,
                                                scatt_angle=0.5,
                                                scatt_lon=1.0,
                                                scatt_lat=0.2,
                                                distance_cm=3.5,
                                                event_id=7)

    assert isinstance(event, TimeTagEmCDSEventInSCFrame)

    assert event.id == 7
    assert event.time == time
    assert event.energy_keV == 511.0
    assert event.energy == 511.0 * u.keV
    assert event.scattering_angle_rad == 0.5
    assert event.scattering_angle == Angle(0.5, u.rad)
    assert event.distance_cm == 3.5
    assert event.distance == 3.5 * u.cm


def test_time_tag_emcds_distance_event_data_from_arrays():
    n = 6

    jd1 = np.full(n, 2460000.0)
    jd2 = np.linspace(0, 0.5, n)
    energy_keV = np.linspace(100, 600, n)
    scatt_angle_rad = np.linspace(0.1, 1.0, n)
    scatt_lon_rad = np.linspace(0, 1, n)
    scatt_lat_rad = np.linspace(-0.5, 0.5, n)
    distance_cm = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    event_id = np.arange(100, 100 + n)

    data = TimeTagEmCDSDistanceEventDataInSCFrameFromArrays(
        jd1, jd2, energy_keV, scatt_lon_rad, scatt_lat_rad, scatt_angle_rad, distance_cm, event_id)

    assert data.nevents == n
    assert np.array_equal(data.distance_cm, distance_cm)
    assert np.allclose(data.distance.to_value(u.cm), distance_cm)

    events = list(data)
    assert len(events) == n

    for i, event in enumerate(events):
        assert isinstance(event, TimeTagEmCDSDistanceEventInSCFrame)
        assert event.id == event_id[i]
        assert event.distance_cm == distance_cm[i]
        assert event.energy_keV == energy_keV[i]

    # __getitem__
    event2 = data[2]
    assert event2.id == event_id[2]
    assert event2.distance_cm == distance_cm[2]


def test_time_tag_emcds_distance_event_data_from_astropy():
    n = 4

    time = Time(60000.0 + np.arange(n) * 0.01, format='jd')
    energy = np.array([100., 200., 300., 400.]) * u.keV
    scattering_angle = Angle([0.1, 0.2, 0.3, 0.4], u.rad)
    scattered_direction = SkyCoord([0., 0.1, 0.2, 0.3],
                                    [0., 0.05, 0.1, 0.15],
                                    unit=u.rad,
                                    frame=SpacecraftFrame())
    distance = np.array([1., 2., 3., 4.]) * u.cm

    data = TimeTagEmCDSDistanceEventDataInSCFrameFromArrays.from_astropy(
        time, energy, scattering_angle, scattered_direction, distance)

    assert data.nevents == n
    assert np.allclose(data.energy_keV, energy.to_value(u.keV))
    assert np.allclose(data.distance_cm, distance.to_value(u.cm))


def test_time_tag_emcds_distance_event_data_selection():
    n = 6

    jd1 = np.full(n, 2460000.0)
    jd2 = np.linspace(0, 0.5, n)
    energy_keV = np.linspace(100, 600, n)
    scatt_angle_rad = np.linspace(0.1, 1.0, n)
    scatt_lon_rad = np.linspace(0, 1, n)
    scatt_lat_rad = np.linspace(-0.5, 0.5, n)
    distance_cm = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    event_id = np.arange(100, 100 + n)

    selector = DistanceSelector(min_distance=1.5 * u.cm, max_distance=4.5 * u.cm)
    expected_mask = (distance_cm >= 1.5) & (distance_cm < 4.5)

    filtered = TimeTagEmCDSDistanceEventDataInSCFrameFromArrays(
        jd1, jd2, energy_keV, scatt_lon_rad, scatt_lat_rad, scatt_angle_rad, distance_cm, event_id,
        selection=selector)

    assert filtered.nevents == np.count_nonzero(expected_mask)
    assert np.array_equal(filtered.distance_cm, distance_cm[expected_mask])
    assert np.array_equal(np.asarray(filtered.jd2), jd2[expected_mask])
    assert np.array_equal(np.asarray(filtered.energy_keV), energy_keV[expected_mask])
    assert np.array_equal(np.asarray(filtered.ids), event_id[expected_mask])


class _FakeFitsFiles:
    """
    Small helper that stubs out UnBinnedData.get_dict_from_fits so the
    *FromDC3Fits classes can be tested without a real fits file on disk.
    """

    def __init__(self, monkeypatch, files_data):
        self._files_data = files_data
        monkeypatch.setattr(UnBinnedData, "get_dict_from_fits", self._get_dict_from_fits)

    def _get_dict_from_fits(self, _self_or_none, input_fits):
        return self._files_data[input_fits]


def test_from_dc3_fits_with_and_without_distance(tmp_path, monkeypatch):
    file1 = tmp_path / "f1.fits"
    file2 = tmp_path / "f2.fits"

    files_data = {
        str(file1): {
            'TimeTags': np.array([10., 30.]),
            'Energies': np.array([100., 300.]),
            'Phi': np.array([0.1, 0.3]),
            'Psi local': np.array([1.0, 1.2]),
            'Chi local': np.array([0.5, 0.6]),
            'Distance': np.array([1.0, 3.0]),
        },
        str(file2): {
            'TimeTags': np.array([20.]),
            'Energies': np.array([200.]),
            'Phi': np.array([0.2]),
            'Psi local': np.array([1.1]),
            'Chi local': np.array([0.55]),
            'Distance': np.array([2.0]),
        },
    }

    _FakeFitsFiles(monkeypatch, files_data)

    # Distance-aware loader: events should come back time-sorted (10,20,30)
    # and concatenated across both files, carrying the Distance column.
    data = TimeTagEmCDSDistanceEventDataInSCFrameFromDC3Fits([file1, file2])

    assert data.nevents == 3
    assert np.allclose(data.energy_keV, [100., 200., 300.])
    assert np.allclose(data.distance_cm, [1.0, 2.0, 3.0])

    # The plain (non-distance) loader must keep working even when the
    # fits files don't have a Distance column at all.
    files_data_no_distance = {
        path: {key: value for key, value in file_data.items() if key != 'Distance'}
        for path, file_data in files_data.items()
    }

    _FakeFitsFiles(monkeypatch, files_data_no_distance)

    data_no_distance = TimeTagEmCDSEventDataInSCFrameFromDC3Fits([file1, file2])

    assert data_no_distance.nevents == 3
    assert np.allclose(data_no_distance.energy_keV, [100., 200., 300.])
