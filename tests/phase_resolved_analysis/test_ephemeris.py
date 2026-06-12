import pytest
import numpy as np
import astropy.units as u
from astropy.time import Time
from cosipy.phase_resolved_analysis.ephemeris import PulsarTimingModel

@pytest.fixture
def dummy_model():
    # 10 Hz pulsar starting at MJD 59000
    t0 = Time(59000.0, format='mjd', scale='tdb')
    return PulsarTimingModel(f0=10.0 * u.Hz, t0=t0)

def test_interval_validation():
    """Test that the timing model catches mathematically invalid intervals."""
    # Valid intervals should pass silently
    valid = [(0.1, 0.4), (0.6, 0.9)]
    assert PulsarTimingModel.validate_intervals(valid) == valid
    
    # Invalid intervals should raise ValueErrors
    with pytest.raises(ValueError):
        PulsarTimingModel.validate_intervals([(0.8, 0.2)]) # Stop before start
        
    with pytest.raises(ValueError):
        PulsarTimingModel.validate_intervals([(-0.1, 0.5)]) # Out of bounds
        
    with pytest.raises(ValueError):
        PulsarTimingModel.validate_intervals([(0.5, 1.1)]) # Out of bounds

def test_get_duty_cycle(dummy_model):
    """Test that the total integrated livetime is calculated correctly."""
    t_start = Time(59000.0, format='mjd', scale='tdb')
    t_stop = Time(59000.0 + 1.0, format='mjd', scale='tdb') # 1 day duration
    
    # 30% duty cycle
    intervals = [(0.0, 0.1), (0.8, 1.0)] 
    
    duration = dummy_model.get_duty_cycle(t_start, t_stop, intervals)
    expected_seconds = (1.0 * u.day).to(u.s) * 0.30
    
    assert np.isclose(duration.value, expected_seconds.value)

def test_get_phase(dummy_model):
    """Test absolute phase calculation."""
    # 0.05 seconds after t0 for a 10Hz pulsar = 0.5 phase
    test_time = dummy_model.t0 + 0.05 * u.s
    phase = dummy_model.get_phase(test_time)
    
    assert np.isclose(phase, 0.5)