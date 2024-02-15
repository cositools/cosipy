# Imports:
import sys
import os
import yaml
import argparse
import cosipy.data_io
from cosipy.config import Configurator

class DataIO:

    """Handles main inputs and outputs."""

    def __init__(self, input_yaml, pw=None):
    
        """
        Parameters
        ----------
        input_yaml : yaml file
            Input yaml file containing all needed inputs for analysis.
    
        Notes
        -----
        The main inputs must currently be passed with the yaml file.
        The parameter configurator will be updated in the near future,
        to allow for much more flexibility. 
        """
        
        # Data I/O:
        inputs = Configurator().open(input_yaml)
        self.data_file = inputs['data_file'] # Full path to input data file.
        self.ori_file = inputs['ori_file'] # Full path to ori file. 
        self.unbinned_output = inputs['unbinned_output'] # fits or hdf5
        self.time_bins = inputs['time_bins'] # Time bin size in seconds. Takes int, float, or list of bin edges.
        self.energy_bins = inputs['energy_bins'] # Needs to match response. Takes list. 
        self.phi_pix_size = inputs['phi_pix_size'] # Binning of Compton scattering angle [deg]
        self.nside = inputs['nside'] # Healpix binning of psi chi local
        self.scheme = inputs['scheme'] # Healpix binning of psi chi local
        self.tmin = inputs['tmin'] # Min time in seconds. 
        self.tmax = inputs['tmax'] # Max time in seconds. 
