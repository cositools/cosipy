import logging
import numpy as np
import astropy.units as u
from astropy.io import fits

logger = logging.getLogger(__name__)

class PhaseSelector:
    """
    Selects events based on a list of pulsar phase intervals. 
    Optimized for FITS processing via NumPy vectorization.
    """        
    def __init__(self, intervals, ephemeris=None):
        self.ephemeris = ephemeris
        from .ephemeris import PulsarTimingModel
        self.intervals = PulsarTimingModel.validate_intervals(intervals)

    def _validate_intervals(self, intervals):
        """Ensures all phase values are strictly within [0, 1] and start < stop."""
        if not isinstance(intervals, list):
            intervals = [intervals]
            
        validated = []
        for start, stop in intervals:
            if not (0.0 <= start <= 1.0 and 0.0 <= stop <= 1.0):
                raise ValueError(f"Phase boundaries must be between 0 and 1. Got: ({start}, {stop})")
            
            # Enforce start < stop as requested
            if start >= stop:
                raise ValueError(f"Start phase must be strictly less than stop phase. Got: ({start}, {stop})")
                
            validated.append((float(start), float(stop)))
        return validated

    def _get_vectorized_mask(self, phases: np.ndarray) -> np.ndarray:
        """Applies all interval ranges to a NumPy array using bitwise OR."""
        combined_mask = np.zeros(phases.shape, dtype=bool)
        
        for pstart, pstop in self.intervals:
            # Removed wrap-around logic; using linear range only
            mask = (phases >= pstart) & (phases <= pstop)
            combined_mask |= mask
            
        return combined_mask

    def filter_events(self, input_fits, output_fits=None):
        """
        Filters a FITS file based on phase intervals.
        Returns the filtered structured array.
        """
        logger.info(f"Filtering FITS file: {input_fits}")
        
        with fits.open(input_fits) as hdul:
            data = hdul[1].data
            mask = self._get_vectorized_mask(data['PULSE_PHASE'])
            selected_data = data[mask] 
            
            if output_fits is not None:
                self._save_fits_fast(selected_data, output_fits, hdul)
            
            return selected_data

    def _save_fits_fast(self, structured_array, output_filename, original_hdul):
        """Saves a NumPy structured array directly using headers from the original file."""
        if len(structured_array) == 0:
            logger.warning("No events matched the phase criteria. File not saved.")
            return

        logger.info(f"Saving {len(structured_array)} events to {output_filename}...")
        try:
            new_hdu = fits.BinTableHDU(data=structured_array, header=original_hdul[1].header)
            primary_hdu = fits.PrimaryHDU(header=original_hdul[0].header)
            
            hdul_new = fits.HDUList([primary_hdu, new_hdu])
            hdul_new.writeto(output_filename, overwrite=True)
            logger.info("Successfully saved.")
        except Exception as e:
            logger.error(f"Failed to save FITS file: {e}")