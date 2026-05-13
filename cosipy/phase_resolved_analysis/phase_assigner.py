import logging
import warnings
import numpy as np
from astropy.io import fits
from astropy.time import Time

logger = logging.getLogger(__name__)

class PhaseAssigner:
    def __init__(self, ephemeris):
        """
        Parameters
        ----------
        ephemeris : PhaseEphemeris
            An instantiated object that adheres to the PhaseEphemeris Protocol.
        """
        self.ephemeris = ephemeris

    def add_phase_column(self, input_fits, output_fits=None):
        """Calculates pulsar phases and injects them as a new column in a FITS file.

        This method reads the time tags from the event data, converts them from 
        Mission Elapsed Time (MET) into absolute Astropy Time objects, and 
        delegates the phase calculation to the provided PhaseEphemeris protocol.

        Args:
            input_fits (str): Path to the source FITS file containing event data.
            output_fits (str, optional): Path where the modified FITS will be saved.
                If None, the input file will be overwritten in-place.

        Returns:
            str: The path to the saved FITS file.
        """
        warnings.warn(
            "CAVEAT: The time coordinate system in the event FITS file is currently assumed "
            "to be compatible with the pulsar timing model (e.g., Mission Elapsed Time). "
            "The exact time definition (UTC, TT, etc.) and reference epoch for COSI data "
            "and MEGAlib simulations are not yet fully standardized. Please manually ensure "
            "your ephemeris and data time systems align to avoid phase shifts.",
            UserWarning
        )
        with fits.open(input_fits) as hdul:
            data = hdul[1].data
            header = hdul[1].header
            
            # 1. Extract raw time tags (Mission Elapsed Time in seconds)
            if 'TIME' in data.dtype.names:
                times_raw = data['TIME']
            elif 'TimeTags' in data.dtype.names:
                times_raw = data['TimeTags']
            else:
                raise KeyError("Could not find a valid time column ('TIME' or 'TimeTags').")

            # 2. Convert relative FITS MET into absolute Astropy Time objects
            # Grab the Mission Epoch from the header (defaulting to COSI/Fermi standard)
            mjdrefi = header.get('MJDREFI', 51910.0) 
            mjdreff = header.get('MJDREFF', 7.428703703703703e-4)
            
            # Convert MET seconds into absolute MJD days
            times_mjd = (mjdrefi + mjdreff) + (times_raw / 86400.0)
            
            logger.info("Converting raw FITS times to Astropy Time objects...")
            absolute_times = Time(times_mjd, format='mjd', scale='tdb')

            # 3. Calculate phase using the PhaseEphemeris protocol!
            logger.info("Calculating absolute phases...")
            phase = self.ephemeris.get_phase(absolute_times)
            
            # 4. Column Management: Update existing or append new
            if 'PULSE_PHASE' in data.dtype.names:
                logger.info("Overwriting existing PULSE_PHASE column.")
                data['PULSE_PHASE'] = phase
                new_hdu = fits.BinTableHDU(data=data, header=header)
            else:
                logger.info("Creating new PULSE_PHASE column.")
                new_col = fits.Column(name='PULSE_PHASE', format='D', array=phase)
                new_hdu = fits.BinTableHDU.from_columns(data.columns + new_col, header=header)

            # File I/O
            out = output_fits or input_fits
            fits.HDUList([hdul[0], new_hdu]).writeto(out, overwrite=True)
            logger.info(f"PULSE_PHASE assigned to: {out}")
            return out