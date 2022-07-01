# Imports:
import ROOT as M
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import yaml
from astropy.table import Table
from astropy.io import fits
import ntpath
import h5py
from histpy import Histogram
from mhealpy import HealpixMap
import healpy as hp
import time
import pandas as pd
import cosipy.data_io
from cosipy.make_plots import MakePlots

from tqdm.autonotebook import tqdm
from IPython.display import HTML

from fit import COSI_model_fit

import warnings
warnings.filterwarnings('ignore')

# Load MEGAlib into ROOT
M.gSystem.Load("$(MEGAlib)/lib/libMEGAlib.so")

# Initialize MEGAlib
G = M.MGlobal()
G.Initialize()

class DataIO:
    
    def __init__(self, input_yaml):
        
        """Main user inputs are specified in inputs.yaml file."""

        # Load housekeeping inputs:
        housekeeping_path_prefix = os.path.split(cosipy.data_io.__file__)[0]
        housekeeping_dir = os.path.join(housekeeping_path_prefix,"housekeeping_files")
        housekeeping_file = os.path.join(housekeeping_dir,"housekeeping_data_io.yaml")
        with open(housekeeping_file,"r") as file:
             inputs = yaml.load(file,Loader=yaml.FullLoader)
        self.mimrec_config = os.path.join(housekeeping_dir,inputs['mimrec_config']) # Default mimrec configuration file

        # Load main user inputs from input_yaml file:
        with open(input_yaml,"r") as file:
             inputs = yaml.load(file,Loader=yaml.FullLoader)
        
        # Mass model (this needs to be moved to housekeeping_data_io.yaml):
        self.geo_file = inputs['geo_file'] # Full path to geometry file

        # Event selections:
        self.data_file = inputs['data_file'] # Full path to input data file.
        self.use_ps = inputs['use_ps'] # 'true' for point source selection or 'false' for all sky
        self.coordinates = inputs['coordinates'] # 1 for Galactic
        self.ps_glon = inputs['ps_glon'] # Galactic longitude of ps location [deg]
        self.ps_glat = inputs['ps_glat'] # Galactic latitude of ps location [deg]
        self.ps_rad_max = inputs['ps_rad_max'] # Maximum radial window (= ARM for Compton) [deg]
        self.time_mode = inputs['time_mode'] # 0 for all time, 1 for time selection
        self.tmin = inputs['tmin'] # Minimum time [s]
        self.tmax = inputs['tmax'] # Maximum time [s]
        self.emin = inputs['emin'] # Minimum energy [keV]
        self.emax = inputs['emax'] # Maximum energy [keV]

        # Data I/O:
        self.write_unbinned_data = inputs['write_unbinned_data'] # True or False
        self.unbinned_output = inputs['unbinned_output'] # 'fits' or 'hdf5'

        # Data binning:
        self.time_bins = inputs['time_bins'] # Number of time bins. Takes int or array.
        self.energy_bins = inputs['energy_bins'] # Needs to match response. Takes list. 
        self.phi_pix_size = inputs['phi_pix_size'] # Binning of Compton scattering angle [deg]
        self.nside = inputs['nside'] # Healpix binning of psi chi local
        self.scheme = inputs['scheme'] # Healpix binning of psi chi local

class UnBinnedData(DataIO):

    def select_data(self):
      
        """
        Reads in full data set and makes selections based on user inputs.
        """

        # Make print statement:
        print()
        print("running select_data...")
        print()

        # Define output file:
        output_events = "selected_data.inc1.id1.extracted.tra.gz" 

        # Extract events:
        os.system("mimrec -g %s -c %s -f %s -x -o %s -n \
                   -C EventSelections.Source.UsePointSource=%s \
                   -C EventSelections.Source.Coordinates=%s \
                   -C EventSelections.Source.Longitude=%s \
                   -C EventSelections.Source.Latitude=%s \
                   -C EventSelections.Source.ARM.Max=%s \
                   -C EventSelections.TimeMode=%s \
                   -C EventSelections.Time.Min=%s \
                   -C EventSelections.Time.Max=%s \
                   -C EventSelections.FirstEnergyWindow.Min=%s \
                   -C EventSelections.FirstEnergyWindow.Max=%s" \
                   %(self.geo_file, self.mimrec_config, self.data_file, output_events, \
                   self.use_ps, \
                   self.coordinates, \
                   self.ps_glon, \
                   self.ps_glat, \
                   self.ps_rad_max, \
                   self.time_mode, \
                   self.tmin, \
                   self.tmax, \
                   self.emin, \
                   self.emax))

        return

    def read_tra(self,tra_file="selected_data.inc1.id1.extracted.tra.gz"):
        
        """
        Reads in MEGAlib .tra (or .tra.gz) file.
        Returns COSI dataset as a dictionary of the form:
        cosi_dataset = {'Full filename':self.data_file,
                        'Energies':erg,
                        'TimeTags':tt,
                        'Xpointings':np.array([lonX,latX]).T,
                        'Ypointings':np.array([lonY,latY]).T,
                        'Zpointings':np.array([lonZ,latZ]).T,
                        'Phi':phi,
                        'Chi local':chi_loc,
                        'Psi local':psi_loc,
                        'Distance':dist,
                        'Chi galactic':chi_gal,
                        'Psi galactic':psi_gal}
        
        Input (optional):
        tra_file: Name of tra file to read. Default is output from select_data method.
        """

        # Make print statement:
        print()
        print("running read_tra...")
        print()

        # Check if file exists:
        Reader = M.MFileEventsTra()
        if Reader.Open(M.MString(tra_file)) == False:
            print("Unable to open file %s. Aborting!" %self.data_file)
            sys.exit()

        # Initialise empty lists:
            
        # Total photon energy
        erg = []
        # Time tag in UNIX time
        tt = []
        # Event Type (0: CE; 4:PE; ...)
        et = []
        # Latitude of X direction of spacecraft
        latX = []
        # Lontitude of X direction of spacecraft
        lonX = []
        # Latitude of Z direction of spacecraft
        latZ = []
        # Longitude of Z direction of spacecraft
        lonZ = []
        # Compton scattering angle
        phi = []
        # Measured data space angle chi (azimuth direction; 0..360 deg)
        chi_loc = []
        # Measured data space angle psi (polar direction; 0..180 deg)
        psi_loc = []
        # First lever arm distance in cm
        dist = []
        # Measured gal angle chi (lon direction)
        chi_gal = []
        # Measured gal angle psi (lat direction)
        psi_gal = [] 

        # Browse through tra file, select events, and sort into corresponding list:
        # Note: The Reader class from MEGAlib knows where an event starts and ends and
        # returns the Event object which includes all information of an event.
        # Note: Here only select Compton events (will add Photo events later as optional).
        # Note: All calculations and definitions taken from:
        # /MEGAlib/src/response/src/MResponseImagingBinnedMode.cxx.

        while True:
            
            Event = Reader.GetNextEvent()
            if not Event:
                break
                
            # Total Energy:
            erg.append(Event.Ei())
            # Time tag in UNIX seconds:
            tt.append(Event.GetTime().GetAsSeconds())
            # Event type (0 = Compton, 4 = Photo):
            et.append(Event.GetEventType())
            # x axis of space craft pointing at GAL latitude:
            latX.append(Event.GetGalacticPointingXAxisLatitude())
            # x axis of space craft pointing at GAL longitude:
            lonX.append(Event.GetGalacticPointingXAxisLongitude())
            # z axis of space craft pointing at GAL latitude:
            latZ.append(Event.GetGalacticPointingZAxisLatitude())
            # z axis of space craft pointing at GAL longitude:
            lonZ.append(Event.GetGalacticPointingZAxisLongitude())    
            # Compton scattering angle:
            phi.append(Event.Phi()) 
            # Data space angle chi (azimuth):
            chi_loc.append((-Event.Dg()).Phi())
            # Data space angle psi (polar):
            psi_loc.append((-Event.Dg()).Theta())
            # Interaction length between first and second scatter in cm:
            dist.append(Event.FirstLeverArm())
            # Gal longitude angle corresponding to chi:
            chi_gal.append((Event.GetGalacticPointingRotationMatrix()*Event.Dg()).Phi())
            # Gal longitude angle corresponding to chi:
            psi_gal.append((Event.GetGalacticPointingRotationMatrix()*Event.Dg()).Theta())
                
        # Initialize arrays:
        erg = np.array(erg)
        tt = np.array(tt)
        et = np.array(et)
            
        latX = np.array(latX)
        lonX = np.array(lonX)
        # Change longitudes to from 0..360 deg to -180..180 deg
        lonX[lonX > np.pi] -= 2*np.pi
        
        latZ = np.array(latZ)
        lonZ = np.array(lonZ)
        # Change longitudes to from 0..360 deg to -180..180 deg
        lonZ[lonZ > np.pi] -= 2*np.pi
        
        phi = np.array(phi)
        
        chi_loc = np.array(chi_loc)
        # Change azimuth angle to 0..360 deg
        chi_loc[chi_loc < 0] += 2*np.pi
        
        psi_loc = np.array(psi_loc)
        
        dist = np.array(dist)
        
        chi_gal = np.array(chi_gal)
        psi_gal = np.array(psi_gal)
        
        # Construct Y direction from X and Z direction
        lonlatY = construct_scy(np.rad2deg(lonX),np.rad2deg(latX),
                                np.rad2deg(lonZ),np.rad2deg(latZ))
        lonY = np.deg2rad(lonlatY[0])
        latY = np.deg2rad(lonlatY[1])
        
        # Avoid negative zeros
        chi_loc[np.where(chi_loc == 0.0)] = np.abs(chi_loc[np.where(chi_loc == 0.0)])
        
        # Make observation dictionary
        cosi_dataset = {'Energies':erg,
                        'TimeTags':tt,
                        'Xpointings':np.array([lonX,latX]).T,
                        'Ypointings':np.array([lonY,latY]).T,
                        'Zpointings':np.array([lonZ,latZ]).T,
                        'Phi':phi,
                        'Chi local':chi_loc,
                        'Psi local':psi_loc,
                        'Distance':dist,
                        'Chi galactic':chi_gal,
                        'Psi galactic':psi_gal} 
        self.cosi_dataset = cosi_dataset

        # Write unbinned data to file (either fits or hdf5):
        if self.write_unbinned_data == True:
        
            # Data units:
            units=['keV','s','rad:[glon,glat]','rad:[glon,glat]',
                    'rad:[glon,glat]','rad','rad','rad','cm','rad','rad']
            
            # For fits output: 
            if self.unbinned_output == 'fits':
                table = Table(list(self.cosi_dataset.values()),\
                        names=list(self.cosi_dataset.keys()), \
                        units=units, \
                        meta={'data file':ntpath.basename(self.data_file)})
                table.write("unbinned_data.fits", overwrite=True)
                os.system('gzip -f unbinned_data.fits')

            # For hdf5 output:
            if self.unbinned_output == 'hdf5':
                with h5py.File('unbinned_data.hdf5', 'w') as hf:
                    for each in list(self.cosi_dataset.keys()):
                        dset = hf.create_dataset(each, data=self.cosi_dataset[each], compression='gzip')        

        return 

class BinnedData(UnBinnedData):
  
    def get_dict_from_fits(self,input_fits):

        """Constructs dictionary from input fits file"""

        # Initialize dictionary:
        this_dict = {}
        
        # Fill dictionary from input fits file:
        hdu = fits.open(input_fits,memmap=True)
        cols = hdu[1].columns
        data = hdu[1].data
        for i in range(0,len(cols)):
            this_key = cols[i].name
            this_data = data[this_key]
            this_dict[this_key] = this_data

        return this_dict

    def get_dict_from_hdf5(self,input_hdf5):
        
        """Constructs dictionary from input hdf5 file"""

        # Initialize dictionary:
        this_dict = {}
        
        # Fill dictionary from input h5fy file:
        hf = h5py.File(input_hdf5,"r")
        keys = list(hf.keys())
        for each in keys:
            this_dict[each] = hf[each][:]

        return this_dict
    
    def get_binned_data(self, unbinned_data=None, make_binning_plots=False):

        """ 
        Bin the data using histpy and mhealpy.
        
        Inputs:
        - unbinned_data(optional): read in unbinned data from file. 
          Input file is either .fits or .hdf5 and must correspond to 
          the unbinned_output parameter in inputs.yaml.     
        - make_binning_plots: Option to plot the binning.
        """
        
        # Make print statement:
        print()
        print("running get_binned_data...")
        print()
      
        # Option to read in unbinned data file:
        if unbinned_data:
            if self.unbinned_output == 'fits':
                self.cosi_dataset = self.get_dict_from_fits(unbinned_data)
            if self.unbinned_output == 'hdf5':
                self.cosi_dataset = self.get_dict_from_hdf5(unbinned_data)

        # Get time bins:
        min_time = np.amin(self.cosi_dataset['TimeTags'])
        max_time = np.amax(self.cosi_dataset['TimeTags'])

        if type(self.time_bins).__name__ == 'int':
            time_bin_edges = np.linspace(min_time,max_time,self.time_bins+1)
        
        if type(self.time_bins).__name__ == 'list':
            # Check that bins correspond to min and max time:
            if (self.time_bins[0] > min_time) | (self.time_bins[-1] < max_time):
                print()
                print("ERROR: Time bins do not cover the full selected data range!")
                print()
                sys.exit()
            time_bin_edges = np.array(self.time_bins)

        # Get energy bins:
        energy_bin_edges = np.array(self.energy_bins)

        # Get phi bins:
        number_phi_bins = int(180./self.phi_pix_size)
        phi_bin_edges = np.linspace(0,np.pi,number_phi_bins+1)

        # Get healpix binning for psi and chi local:
        npix = hp.nside2npix(self.nside)
        print()
        print("Approximate resolution at NSIDE {} is {:.2} deg".format(self.nside, hp.nside2resol(self.nside, arcmin=True) / 60))
        print()

        # Define the grid and initialize
        self.m = HealpixMap(nside = self.nside, scheme = self.scheme, dtype = int)

        # Bin psi and chi data:
        # Note: psi and chi correspond to the default colatitude and longitude definitions in healpy (double check).
        PsiChi_pixs = self.m.ang2pix(self.cosi_dataset['Psi local'],self.cosi_dataset['Chi local'])
        PsiChi_bin_edges = np.arange(0,npix+2,1)
     
        # Fill map for ploting:
        unique, unique_counts = np.unique(PsiChi_pixs, return_counts=True)
        self.m[unique] = unique_counts

        # Initialize histogram:
        self.binned_data = Histogram([time_bin_edges, energy_bin_edges, phi_bin_edges, PsiChi_bin_edges], labels = ['Time','Energy','Phi','PsiChi'], sparse=True)

        # Fill histogram:
        self.binned_data.fill(self.cosi_dataset['TimeTags'], self.cosi_dataset['Energies'], self.cosi_dataset['Phi'], PsiChi_pixs)
 
        # Save binned data to hdf5 file:
        self.binned_data.write('binned_data.hdf5',overwrite=True)

        # Get binning information:
        self.get_binning_info(self.binned_data)

        # Plot the binned data:
        if make_binning_plots == True:
            self.plot_binned_data(self.binned_data, self.m)           

        return

    def get_binning_info(self, binned_data):

        """
        Get binning information from Histpy histogram.
        Input:
        binned_data: Histogram object (hdf5).
        """
 
        # Get time binning information:
        self.time_hist = binned_data.project('Time').contents.todense()
        self.num_time_bins = binned_data.axes['Time'].nbins
        self.time_bin_centers = binned_data.axes['Time'].centers
        self.time_bin_edges = binned_data.axes['Time'].edges
        self.time_bin_widths = binned_data.axes['Time'].widths
        self.total_time = self.time_bin_edges[-1] - self.time_bin_edges[0]

        # Get energy binning information:
        self.energy_hist = binned_data.project('Energy').contents.todense()
        self.num_energy_bins = binned_data.axes['Energy'].nbins
        self.energy_bin_centers = binned_data.axes['Energy'].centers
        self.energy_bin_edges = binned_data.axes['Energy'].edges
        self.energy_bin_widths = binned_data.axes['Energy'].widths
 
        return

    def plot_binned_data(self, binned_data, healpix_map):

        """
        Plot binnned data for all axes.
        Input:
        binned_data: Histogram object (hdf5).
        healpix_map: Healpix map object.
        """

        # Define plot dictionaries:
        time_energy_plot = {"projection":["Time","Energy"],"xlabel":"Time [s]",\
                "ylabel":"Energy [keV]","savefig":"2d_time_energy.png"}
        time_plot = {"projection":"Time","xlabel":"Time [s]",\
                "ylabel":"Counts","savefig":"time_binning.pdf"}
        energy_plot = {"projection":"Energy","xlabel":"Energy [keV]",\
                "ylabel":"Counts","savefig":"energy_binning.pdf"}
        phi_plot = {"projection":"Phi","xlabel":"Phi [rad]",\
                "ylabel":"Counts","savefig":"phi_binning.pdf"}
        psichi_plot = {"projection":"PsiChi",\
                "xlabel":"PsiChi [HEALPix ring pixel number]",\
                "ylabel":"Counts","savefig":"psichi_binning.pdf"}
        
        # Make plots:
        plot_list = [time_energy_plot,time_plot,energy_plot,phi_plot,psichi_plot]
        for each in plot_list:
            binned_data.project(each["projection"]).plot()
            plt.xlabel(each["xlabel"],fontsize=12)
            plt.ylabel(each["ylabel"], fontsize=12)
            plt.savefig(each["savefig"])
            plt.show()
            plt.close() 
 
        # Plot PsiChi mhealpy default:
        plot,ax = healpix_map.plot()
        ax.get_figure().set_figwidth(4) 
        ax.get_figure().set_figheight(3)
        plt.title("PsiChi Binning (counts)")
        plt.savefig("psichi_default.png",bbox_inches='tight')
        plt.show()
        plt.close()

        # Plot PsiChi mhealpy rotated:
        plot,ax = healpix_map.plot(ax = 'orthview', ax_kw = {'rot':[0,90,0]})
        ax.get_figure().set_figwidth(3)  
        ax.get_figure().set_figheight(4)
        plt.title("PsiChi Binning (counts)")
        plt.savefig("psichi_rotated.png",bbox_inches='tight')
        plt.show()
        plt.close()
    
        return
 
    def plot_raw_spectrum(self, binned_data=None, time_rate=False, write_spec=False):

        """
        Plot raw spectrum of binned data.
        
        Inputs (all optional):
        binned_data: Binnned data file (hdf5).
        time_rate: If True plots ct/keV/s. The defualt is ct/keV. 
        write_spec: If True will write spectrum to .dat file.
        """

        # Make print statement:
        print()
        print("running plot_raw_spectrum...")
        print()

        # Option to read in binned data from hdf5 file:
        if binned_data:
            self.binned_data = Histogram.open(binned_data)
            self.get_binning_info(self.binned_data) 

        # Option to normalize by total time:
        if time_rate==False:
            raw_rate = self.energy_hist/self.energy_bin_widths
            ylabel = "$\mathrm{ct \ keV^{-1}}$"
        if time_rate==True:
            raw_rate = self.energy_hist/self.energy_bin_widths/self.total_time
            ylabel = "$\mathrm{ct \ keV^{-1} \ s^{-1}}$"

        # Plot:
        MakePlots().make_basic_plot(self.energy_bin_centers, raw_rate, \
                x_error=self.energy_bin_widths/2.0,\
                legend="raw spectrum",\
                xlabel="Energy [keV]",\
                ylabel=ylabel,\
                savefig="raw_spectrum.pdf")

        # Option to write data:
        if write_spec == True:
            d = {"Energy[keV]":self.energy_bin_centers,"Rate[ct/keV]":raw_rate}
            df = pd.DataFrame(data=d)
            df.to_csv("raw_spectrum.dat",float_format='%10.5f',index=False,sep="\t",columns=["Energy[keV]","Rate[ct/keV]"])
        
        return

    def plot_raw_lightcurve(self, binned_data=None, write_lc=False):

        """
        Plot raw lightcurve of binned data.
        
        Inputs (all optional):
        binned_data: Binnned data file (hdf5).
        write_lc: If True will write lightcurve to .dat file.
        """

        # Make print statement:
        print()
        print("running plot_raw_lightcurve...")
        print()

        # Option to read in binned data from hdf5 file:
        if binned_data:
            self.binned_data = Histogram.open(binned_data)
            self.get_binning_info(self.binned_data) 
        
        # Only plot non-zero bins:
        nonzero_index = self.time_hist > 0
        raw_lc = self.time_hist[nonzero_index]/self.time_bin_widths[nonzero_index]

        # Plot:
        MakePlots().make_basic_plot(self.time_bin_centers[nonzero_index], raw_lc, \
                legend="raw lightcurve",\
                xlabel="Time [s]",\
                ylabel="$\mathrm{ct \ s^{-1}}$",\
                savefig="raw_lighcurve.pdf",\
                marker="",\
                ls="-")
            
        # Option to write data:
        if write_lc == True:
            d = {"Time[UTC]":self.time_bin_centers,"Rate[ct/s]":self.time_hist/self.time_bin_widths}
            df = pd.DataFrame(data=d)
            df.to_csv("raw_LC.dat",float_format='%10.5f',index=False,sep="\t",columns=["Time[UTC]","Rate[ct/s]"])

        return


# Utility methods below:

def construct_scy(scx_l, scx_b, scz_l, scz_b):
    
    """
    Construct y-coordinate of spacecraft/balloon given x and z directions
    Note that here, z is the optical axis
    param: scx_l   longitude of x direction
    param: scx_b   latitude of x direction
    param: scz_l   longitude of z direction
    param: scz_b   latitude of z direction
    """
        
    x = polar2cart(scx_l, scx_b)
    z = polar2cart(scz_l, scz_b)
    
    return cart2polar(np.cross(z,x,axis=0))

def polar2cart(ra,dec):
    
    """
    Coordinate transformation of ra/dec (lon/lat) [phi/theta] polar/spherical coordinates
    into cartesian coordinates
    param: ra   angle in deg
    param: dec  angle in deg
    """
        
    x = np.cos(np.deg2rad(ra)) * np.cos(np.deg2rad(dec))
    y = np.sin(np.deg2rad(ra)) * np.cos(np.deg2rad(dec))
    z = np.sin(np.deg2rad(dec))
    
    return np.array([x,y,z])

def cart2polar(vector):
    
    """
    Coordinate transformation of cartesian x/y/z values into spherical (deg)
    param: vector   vector of x/y/z values
    """
        
    ra = np.arctan2(vector[1],vector[0]) 
    dec = np.arcsin(vector[2])
    
    return np.rad2deg(ra), np.rad2deg(dec)
