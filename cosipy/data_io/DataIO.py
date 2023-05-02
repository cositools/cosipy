# Imports:
import sys
import os
import yaml
import argparse
import cosipy.data_io
from cosipy.config import Configurator

class DataIO:
    
    def __init__(self, input_yaml, pw=None):
       
        #print(argv)
        """Main user inputs are specified in inputs.yaml file."""
        #parser = argparse.ArgumentParser()
        #parser.add_argument('-pw', '--pw', help = 'username')
        #args = parser.parse_args(argv)
        #self.pw = args.pw
        #print(self.pw)

        # Load housekeeping inputs:
        housekeeping_path_prefix = os.path.split(cosipy.data_io.__file__)[0]
        housekeeping_dir = os.path.join(housekeeping_path_prefix,"housekeeping_files")
        housekeeping_file = os.path.join(housekeeping_dir,"housekeeping_data_io.yaml")
        housekeeping_inputs = Configurator().open(housekeeping_file)

        # Data I/O:
        inputs = Configurator().open(input_yaml)
        self.data_file = inputs['data_file'] # Full path to input data file.
        self.unbinned_output = inputs['unbinned_output'] # fits or hdf5
        self.time_bins = inputs['time_bins'] # Time bin size. Takes int or list.
        self.energy_bins = inputs['energy_bins'] # Needs to match response. Takes list. 
        self.phi_pix_size = inputs['phi_pix_size'] # Binning of Compton scattering angle [deg]
        self.nside = inputs['nside'] # Healpix binning of psi chi local
        self.scheme = inputs['scheme'] # Healpix binning of psi chi local
    
    def parse_args(self, argv=None):
        parser = argparse.ArgumentParser()
        parser.add_argument('-pw', '--pw', help = 'username')
        args = vars(parser.parse_args(argv))
        print(args) 
