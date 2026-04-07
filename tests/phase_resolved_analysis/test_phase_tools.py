import pytest
import numpy as np
from astropy.io import fits
import os
import matplotlib
matplotlib.use('Agg')

# Import your classes
from cosipy.phase_resolved_analysis import PhaseAssigner, PhaseSelector, PlotPulseProfile

def test_phase_assigner_math():
    """Verify the core folding math: Phase = (T * F0) % 1.0"""
    # Create a dummy .par file
    with open("test_pulsar.par", "w") as f:
        f.write("F0 10.0\n") # 10 Hz frequency (Period = 0.1s)
    
    assigner = PhaseAssigner("test_pulsar.par")
    assert assigner.f0 == 10.0
    
    # Test times: 0.0s (phase 0), 0.05s (phase 0.5), 0.11s (phase 0.1)
    test_times = np.array([0.0, 0.05, 0.11])
    expected_phases = np.array([0.0, 0.5, 0.1])
    
    calculated_phases = (test_times * assigner.f0) % 1.0
    np.testing.assert_allclose(calculated_phases, expected_phases, atol=1e-7)
    
    os.remove("test_pulsar.par")

def test_phase_selector_wrap_around():
    """Check if the selector correctly handles intervals that cross the 1.0/0.0 boundary."""
    # Logic: Interval (0.8, 0.2) should pick up 0.9 and 0.1, but NOT 0.5
    intervals = [(0.8, 0.2)]
    selector = PhaseSelector(intervals)
    
    test_phases = np.array([0.1, 0.5, 0.9])
    mask = selector._get_vectorized_mask(test_phases)
    
    assert mask[0] == True  # 0.1 is <= 0.2
    assert mask[1] == False # 0.5 is not in range
    assert mask[2] == True  # 0.9 is >= 0.8

def test_phase_selector_multiple_intervals():
    """Check if multiple disjoint intervals work together."""
    intervals = [(0.1, 0.2), (0.5, 0.6)]
    selector = PhaseSelector(intervals)
    
    test_phases = np.array([0.15, 0.3, 0.55])
    mask = selector._get_vectorized_mask(test_phases)
    
    assert np.array_equal(mask, [True, False, True])

def test_plotter_data_handling():
    """Ensure the plotter identifies 'TimeTags' and 'TIME' columns correctly."""
    # Create mock data with 'TimeTags'
    data_tags = np.zeros(10, dtype=[('PULSE_PHASE', 'f8'), ('TimeTags', 'f8')])
    plotter_1 = PlotPulseProfile(data_tags)
    assert hasattr(plotter_1, 'times')
    
    # Create mock data with 'TIME'
    data_time = np.zeros(10, dtype=[('PULSE_PHASE', 'f8'), ('TIME', 'f8')])
    plotter_2 = PlotPulseProfile(data_time)
    assert hasattr(plotter_2, 'times')
