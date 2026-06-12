import pytest
import numpy as np
from astropy.io import fits
import os
import matplotlib
matplotlib.use('Agg') # Fixes the headless server crash
import matplotlib.pyplot as plt

from cosipy.phase_resolved_analysis import PhaseAssigner, PhaseSelector, PlotPulseProfile

import astropy.units as u
from astropy.time import Time
from cosipy.phase_resolved_analysis.ephemeris import PulsarTimingModel
from cosipy.phase_resolved_analysis.phase_assigner import PhaseAssigner

def test_phase_assigner_math():
    """Verify that the Assigner correctly stores the ephemeris protocol."""

    
    # 1. Initialize the new protocol object instead of a .par file
    t0 = Time(59000.0, format='mjd', scale='tdb')
    timing_model = PulsarTimingModel(f0=10.0 * u.Hz, t0=t0)
    
    # 2. Pass it to the assigner
    assigner = PhaseAssigner(timing_model)
    
    # 3. Verify it is wired up correctly!
    assert assigner.ephemeris.f0.value == 10.0


def test_plotter_data_handling():
    """Verify that the plotter correctly handles NumPy structured arrays."""
    # Create mock data as a structured array (this triggered the previous error)
    data = np.zeros(10, dtype=[('PULSE_PHASE', 'f8'), ('TimeTags', 'f8')])
    plotter = PlotPulseProfile(data)
    
    assert len(plotter.phases) == 10
    assert len(plotter.times) == 10
    
    # Run plot to ensure no crash
    plotter.plot()
    plt.close('all')
