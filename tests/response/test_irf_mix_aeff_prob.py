import numpy as np
import pytest
from unittest.mock import MagicMock

from cosipy.response.irf_mix_aeff_prob import IRFMixAeffProbUnpolarized
from cosipy.interfaces.instrument_response_interface import FarFieldSpectralInstrumentResponseFunctionInterface


class _FakePhotonListType:
    """Stand-in photon_list_type, distinct from any real interface, to
    prove IRFMixAeffProbUnpolarized works with arbitrary matching types."""


class _FakeEventDataType:
    """Stand-in event_data_type, see _FakePhotonListType."""


def _make_source(photon_list_type=_FakePhotonListType, event_data_type=_FakeEventDataType):
    source = MagicMock(spec=FarFieldSpectralInstrumentResponseFunctionInterface)
    source.photon_list_type = photon_list_type
    source.event_data_type = event_data_type
    return source


@pytest.fixture
def mock_aeff_source():
    return _make_source()


@pytest.fixture
def mock_diff_aeff_source():
    return _make_source()


class TestIRFMixAeffProbUnpolarized:

    def test_initialization(self, mock_aeff_source, mock_diff_aeff_source):
        irf = IRFMixAeffProbUnpolarized(mock_aeff_source, mock_diff_aeff_source)
        assert irf._aeff_source is mock_aeff_source
        assert irf._diff_aeff_source is mock_diff_aeff_source
        assert irf.photon_list_type is mock_aeff_source.photon_list_type
        assert irf.event_data_type is mock_aeff_source.event_data_type

    def test_initialization_mismatched_photon_list_type_raises(self, mock_aeff_source, mock_diff_aeff_source):
        mock_diff_aeff_source.photon_list_type = object()
        with pytest.raises(ValueError, match="photon_list_type"):
            IRFMixAeffProbUnpolarized(mock_aeff_source, mock_diff_aeff_source)

    def test_initialization_mismatched_event_data_type_raises(self, mock_aeff_source, mock_diff_aeff_source):
        mock_diff_aeff_source.event_data_type = object()
        with pytest.raises(ValueError, match="event_data_type"):
            IRFMixAeffProbUnpolarized(mock_aeff_source, mock_diff_aeff_source)

    def test_effective_area_delegates_to_aeff_source_only(self, mock_aeff_source, mock_diff_aeff_source):
        irf = IRFMixAeffProbUnpolarized(mock_aeff_source, mock_diff_aeff_source)

        mock_photons = object()
        mock_aeff_source._effective_area_cm2.return_value = np.array([1.0, 2.0, 3.0])

        result = irf._effective_area_cm2(mock_photons)

        mock_aeff_source._effective_area_cm2.assert_called_once_with(mock_photons)
        mock_diff_aeff_source._effective_area_cm2.assert_not_called()
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_differential_effective_area_delegates_to_diff_aeff_source_only(self, mock_aeff_source, mock_diff_aeff_source):
        irf = IRFMixAeffProbUnpolarized(mock_aeff_source, mock_diff_aeff_source)

        mock_photons = object()
        mock_events = object()
        mock_diff_aeff_source._differential_effective_area_cm2.return_value = np.array([0.1, 0.2, 0.3])

        result = irf._differential_effective_area_cm2(mock_photons, mock_events)

        mock_diff_aeff_source._differential_effective_area_cm2.assert_called_once_with(mock_photons, mock_events)
        mock_aeff_source._differential_effective_area_cm2.assert_not_called()
        np.testing.assert_array_equal(result, [0.1, 0.2, 0.3])

    def test_random_events_not_implemented(self, mock_aeff_source, mock_diff_aeff_source):
        irf = IRFMixAeffProbUnpolarized(mock_aeff_source, mock_diff_aeff_source)

        with pytest.raises(NotImplementedError):
            irf._random_events(object())
