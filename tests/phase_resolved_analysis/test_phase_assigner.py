import pytest
import numpy as np
from astropy.io import fits
from cosipy.phase_resolved_analysis.phase_assigner import PhaseAssigner

class MockTimingModel:
    """A dummy protocol object to safely test the Assigner without real math."""
    def get_phase(self, times):
        # Always return 0.75 for any given time array
        return np.full(len(times), 0.75)

@pytest.fixture
def dummy_fits_file(tmp_path):
    """Creates a temporary FITS file with a TIME column for testing."""
    file_path = tmp_path / "dummy_events.fits"
    
    # Create dummy data: 5 events at 1, 2, 3, 4, 5 seconds (MET)
    times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    col1 = fits.Column(name='TIME', format='D', array=times)
    
    # Create header with standard COSI/Fermi mission epoch
    header = fits.Header()
    header['MJDREFI'] = 51910.0
    header['MJDREFF'] = 7.428703703703703e-4
    
    hdu = fits.BinTableHDU.from_columns([col1], header=header)
    primary_hdu = fits.PrimaryHDU()
    
    fits.HDUList([primary_hdu, hdu]).writeto(file_path)
    return str(file_path)

def test_add_phase_column(dummy_fits_file, tmp_path):
    """Test that the assigner reads FITS, delegates to the protocol, and saves."""
    output_file = str(tmp_path / "assigned_events.fits")
    
    model = MockTimingModel()
    assigner = PhaseAssigner(model)
    
    # Run the assigner
    result_path = assigner.add_phase_column(dummy_fits_file, output_file)
    
    # Verify the output FITS file
    with fits.open(result_path) as hdul:
        data = hdul[1].data
        assert 'PULSE_PHASE' in data.dtype.names
        # Our mock model always returns 0.75
        np.testing.assert_allclose(data['PULSE_PHASE'], 0.75)