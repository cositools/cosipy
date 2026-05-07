import numpy as np
import astropy.units as u
from astropy.time import Time
from typing import Protocol, List, Tuple

class PhaseEphemeris(Protocol):
    """
    Protocol defining the interface for pulsar ephemeris models.
    
    Any ephemeris class used for phase-resolved analysis must implement 
    these methods to ensure compatibility with event selection and 
    exposure correction tools.
    """
    
    def get_phase(self, time: Time) -> np.ndarray:
        """
        Calculate the rotational phase for a given set of timestamps.

        Parameters
        ----------
        time : astropy.time.Time
            The timestamps for which to calculate the phase.

        Returns
        -------
        numpy.ndarray
            An array of phases corresponding to the input times, strictly 
            bounded between 0.0 and 1.0.
        """
        ...
        
    def get_duty_cycle(self, time_start: Time, time_stop: Time, 
                       intervals: List[Tuple[float, float]]) -> u.Quantity:
        """
        Calculate the total time spent within the specified phase intervals.

        Parameters
        ----------
        time_start : astropy.time.Time
            The start time of the observation window.
        time_stop : astropy.time.Time
            The stop time of the observation window.
        intervals : list of tuple of float
            A list of (start_phase, stop_phase) intervals.

        Returns
        -------
        astropy.units.Quantity
            The total integrated time (in seconds) spent within the 
            provided phase intervals during the observation window.
        """
        ...

class PulsarTimingModel:
    """
    A simple pulsar ephemeris model assuming a constant spin frequency (F0).
    
    This model calculates phases using a single frequency parameter and a 
    reference epoch. It does not account for frequency derivatives (spin-down) 
    or binary orbital mechanics.

    Attributes
    ----------
    f0 : astropy.units.Quantity
        The constant spin frequency of the pulsar in Hz.
    t0 : astropy.time.Time
        The reference epoch (time zero) for the phase calculation.
        
    TODO : consider barycentric correction
    
    """
    
    def __init__(self, f0: u.Quantity, t0: Time):
        """
        Initialize the constant-frequency ephemeris.

        Parameters
        ----------
        f0 : astropy.units.Quantity
            The pulsar spin frequency (must be convertible to Hz).
        t0 : astropy.time.Time
            The reference epoch. Phases will be calculated relative to this time.
        """
        self.f0 = f0.to(u.Hz)
        self.t0 = t0
        
    def get_phase(self, time: Time) -> np.ndarray:
        """
        Calculate the rotational phase using the constant frequency.

        Parameters
        ----------
        time : astropy.time.Time
            The timestamps to convert to phases.

        Returns
        -------
        numpy.ndarray
            The calculated phases, wrapped to the range [0.0, 1.0).
        """
        dt = (time - self.t0).to(u.s)
        return (self.f0 * dt).value % 1.0
    
    @staticmethod
    def validate_intervals(intervals):
        """Validates that phase intervals are mathematically sound."""
        valid_intervals = []
        for start, stop in intervals:
            if not (0.0 <= start < stop <= 1.0):
                raise ValueError(
                    f"Invalid phase interval: ({start}, {stop}). "
                    "Must satisfy 0.0 <= start < stop <= 1.0"
                )
            valid_intervals.append((start, stop))
        return valid_intervals        
        
    def get_duty_cycle(self, time_start: Time, time_stop: Time, 
                       intervals: List[Tuple[float, float]]) -> u.Quantity:
        """
        Calculate the duty cycle for a constant-frequency pulsar.
        
        For a constant frequency, the duty cycle is independent of the absolute 
        start and stop times and depends only on the total observation duration 
        and the fractional width of the phase intervals.

        Parameters
        ----------
        time_start : astropy.time.Time
            The start time of the observation window.
        time_stop : astropy.time.Time
            The stop time of the observation window.
        intervals : list of tuple of float
            A list of (start_phase, stop_phase) intervals.

        Returns
        -------
        astropy.units.Quantity
            The effective live time within the phase intervals.
        """
        valid_intervals = self.validate_intervals(intervals)
        duration = (time_stop - time_start).to(u.s)
        total_width = sum([stop - start for start, stop in intervals])
        return duration * total_width

    @classmethod
    def from_par_file(cls, filepath, met_ref_epoch: Time):
        """
        Instantiate an Ephemeris object by reading an F0 value from a .par file.

        This method scans a standard pulsar parameter file for the 'F0' key.

        Parameters
        ----------
        filepath : str or pathlib.Path
            The path to the .par file.
        met_ref_epoch : astropy.time.Time
            The reference epoch to use for the resulting Ephemeris instance.

        Returns
        -------
        Ephemeris
            An instance of the Ephemeris class initialized with the parsed F0.
            
        Raises
        ------
        ValueError
            If the 'F0' parameter cannot be found in the provided file.
        """
        f0_val = None
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.split()
                if not parts or parts[0].upper() != 'F0': continue
                
                # Normalize Fortran 'D/d' to standard Python 'E' before floating
                raw = parts[1].strip()
                raw = raw.replace('D', 'E').replace('d', 'E')
                f0_val = float(raw)
                break
        
        if f0_val is None:
            raise ValueError(f"F0 not found in {filepath}")
            
        return cls(f0=f0_val * u.Hz, t0=met_ref_epoch)