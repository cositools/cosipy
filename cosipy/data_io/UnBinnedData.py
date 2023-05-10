# Imports:
import numpy as np
from astropy.table import Table
from astropy.io import fits
import h5py
import time
from cosipy.data_io import DataIO
import gzip
import astropy.coordinates as astro_co
import astropy.units as u
from astropy.coordinates import SkyCoord
from scoords import Attitude
from scoords import SpacecraftFrame
import logging
import sys
import math
from tqdm import tqdm
import subprocess
logger = logging.getLogger(__name__)

class UnBinnedData(DataIO):
 
    def read_tra(self, output_name="unbinned_data"):
        
        """
        Reads in MEGAlib .tra (or .tra.gz) file.
        Returns COSI dataset as a dictionary of the form:
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
        
        Arrays contain unbinned data.
        
        output_name: prefix of output file. 

        Note: The current code is only able to handle data with Compton events.
              It will need to be modified to handle single site and pair. 
            
        """
    
        # Initialise empty lists:
            
        # Total photon energy
        erg = []
        # Time tag in UNIX time
        tt = []
        # Event Type (CE or PE)
        et = []
        # Galactic latitude of X direction of spacecraft
        latX = []
        # Galactic lontitude of X direction of spacecraft
        lonX = []
        # Galactic latitude of Z direction of spacecraft
        latZ = []
        # Galactic longitude of Z direction of spacecraft
        lonZ = []
        # Compton scattering angle
        phi = []
        # Measured data space angle chi (azimuth direction; 0..360 deg)
        chi_loc = []
        # Measured data space angle psi (polar direction; 0..180 deg)
        psi_loc = []
        # Measured gal angle chi (lon direction)
        chi_gal = []
        # Measured gal angle psi (lat direction)
        psi_gal = [] 
        # Components of dg (position vector from 1st interaion to 2nd)
        dg_x = []
        dg_y = []
        dg_z = []

        # Define electron rest energy, which is used in calculation
        # of Compton scattering angle.
        c_E0 = 510.9989500015 # keV


        print("Preparing to read file...")

        # Open .tra.gz file:
        if self.data_file.endswith(".gz"):
            f = gzip.open(self.data_file,"rt")
            
            # Need to get number of lines for progress bar.
            # First try fast method for unix-based systems:
            try:
                proc=subprocess.Popen('gunzip -c %s | wc -l' %self.data_file, \
                        shell=True, stdout=subprocess.PIPE)
                num_lines = float(proc.communicate()[0])

            # If fast method fails, use long method, which should work in all cases.
            except:
                print("Initial attempt failed.")
                print("Using long method...")
                g = gzip.open(self.data_file,"rt")
                num_lines = sum(1 for line in g)
                g.close()

        # Open .tra file:
        elif self.data_file.endswith(".tra"):
            f = open(self.data_file,"r")

            try:
                proc=subprocess.Popen('wc -l < %s' %self.data_file, \
                        shell=True, stdout=subprocess.PIPE)
                num_lines = float(proc.communicate()[0])
                
            except:
                print("Initial attempt failed.")
                print("Using long method...")
                g = open(self.data_file,"rt")
                num_lines = sum(1 for line in g)
                g.close()

        else: 
            print()
            print("ERROR: Input data file must have '.tra' or '.gz' extenstion.")
            print()
            sys.exit()
        
        # Read tra file line by line:
        print("Reading file...")
        pbar = tqdm(total=num_lines) # start progress bar
        for line in f:
         
            this_line = line.strip().split()
            pbar.update(1) # update progress bar

            # Total photon energy and Compton angle: 
            if this_line[0] == "CE":

                # Compute the total photon energy:
                m_Eg = float(this_line[1]) # Energy of scattered gamma ray in keV
                m_Ee = float(this_line[3]) # Energy of recoil electron in keV
                this_erg = m_Eg + m_Ee
                erg.append(this_erg) 
             
                # Compute the Compton scatter angle due to the standard equation,
                # i.e. neglect the movement of the electron,
                # which would lead to a Doppler-broadening.
                this_value = 1.0 - c_E0 * (1.0/m_Eg - 1.0/(m_Ee + m_Eg))
                this_phi = np.arccos(this_value) # radians
                phi.append(this_phi)
            
            # Time tag in Unix time (seconds):
            if this_line[0] == "TI":
                tt.append(float(this_line[1]))

            # Event type: 
            if this_line[0] == "ET":
                et.append(this_line[1])

            # X axis of detector orientation in Galactic coordinates:
            if this_line[0] == "GX":
                this_lonX = np.deg2rad(float(this_line[1])) # radians
                this_latX = np.deg2rad(float(this_line[2])) # radians
                lonX.append(this_lonX)
                latX.append(this_latX)
            
            # Z axis of detector orientation in Galactic coordinates:
            if this_line[0] == "GZ":
                this_lonZ = np.deg2rad(float(this_line[1])) # radians
                this_latZ = np.deg2rad(float(this_line[2])) # radians
                lonZ.append(this_lonZ)
                latZ.append(this_latZ)
 
            # Interaction position information: 
            if (this_line[0] == "CH"):
                
                # First interaction:
                if this_line[1] == "0":
                    v1 = np.array((float(this_line[2]),\
                            float(this_line[3]),float(this_line[4])))
                
                # Second interaction:
                if this_line[1] == "1":
                    v2 = np.array((float(this_line[2]),
                        float(this_line[3]), float(this_line[4])))
                
                    # Compute position vector between first two interactions:
                    dg = v2 - v1
                    dg_x.append(dg[0])
                    dg_y.append(dg[1])
                    dg_z.append(dg[2])
                
        # Close progress bar:
        pbar.close()
        print("Making COSI data set...")

        # Initialize arrays:
        erg = np.array(erg)
        phi = np.array(phi)
        tt = np.array(tt)
        et = np.array(et)

        # Convert dg vector from 3D cartesian coordinates 
        # to spherical polar coordinates, and then extract distance 
        # b/n first two interactions (in cm), psi (rad), and chi (rad).
        # Note: the resulting angles are latitude/longitude (or elevation/azimuthal), 
        # i.e. the origin is along the equator rather than at the north pole.
        conv = astro_co.cartesian_to_spherical(np.array(dg_x), np.array(dg_y), np.array(dg_z))
        dist = conv[0].value 
        psi_loc = conv[1].value 
        chi_loc = conv[2].value

        # Attitude vectors:
        latX = np.array(latX)
        lonX = np.array(lonX)
        
        latZ = np.array(latZ)
        lonZ = np.array(lonZ)

        # Calculate chi_gal and psi_gal from chi_loc and psi_loc::
        xcoords = SkyCoord(lonX*u.rad, latX*u.rad, frame = 'galactic')
        zcoords = SkyCoord(lonZ*u.rad, latZ*u.rad, frame = 'galactic')
        attitude = Attitude.from_axes(x=xcoords, z=zcoords, frame = 'galactic')
        c = SkyCoord(lon = chi_loc*u.rad, lat = psi_loc*u.rad, frame = SpacecraftFrame(attitude = attitude))
        c.transform_to('galactic')
        chi_gal = np.array(c.lon.rad)
        psi_gal = np.array(c.lat.rad)
        #self.chi_gal_new = chi_gal

        # Change longitudes from 0..360 deg to -180..180 deg
        lonX[lonX > np.pi] -= 2*np.pi
        
        # Change longitudes from 0..360 deg to -180..180 deg
        lonZ[lonZ > np.pi] -= 2*np.pi 

        # Construct Y direction from X and Z direction
        lonlatY = self.construct_scy(np.rad2deg(lonX),np.rad2deg(latX),
                                np.rad2deg(lonZ),np.rad2deg(latZ))
        lonY = np.deg2rad(lonlatY[0])
        latY = np.deg2rad(lonlatY[1])
 
        # Rotate psi_loc to colatitude, 
        # measured from the negative z direction. 
        # Note: the detector is placed at z<0 in the local frame.
        # Note: the rotation is done to match the historical 
        # definition of COMPTEL.
        psi_loc += (np.pi/2.0)  

        # Rotate chi_loc to be defined relative to negative x-axis.
        # Note: this is done to match the historical definition of COMPTEL.
        index1 = (chi_loc < np.pi) 
        index2 = (chi_loc >= np.pi)
        chi_loc[index1] = chi_loc[index1] + np.pi
        chi_loc[index2] = chi_loc[index2] - np.pi
          
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
        print("Saving file...")
        self.write_unbinned_output(output_name=output_name) 
        
        return 
 
    def construct_scy(self, scx_l, scx_b, scz_l, scz_b):
    
        """
        Construct y-coordinate of spacecraft/balloon given x and z directions
        Note that here, z is the optical axis
        param: scx_l   longitude of x direction
        param: scx_b   latitude of x direction
        param: scz_l   longitude of z direction
        param: scz_b   latitude of z direction
        """
        
        x = self.polar2cart(scx_l, scx_b)
        z = self.polar2cart(scz_l, scz_b)
    
        return self.cart2polar(np.cross(z,x,axis=0))

    def polar2cart(self, ra, dec):
    
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

    def cart2polar(self, vector):
    
        """
        Coordinate transformation of cartesian x/y/z values into spherical (deg)
        param: vector   vector of x/y/z values
        """
        
        ra = np.arctan2(vector[1],vector[0]) 
        dec = np.arcsin(vector[2])
    
        return np.rad2deg(ra), np.rad2deg(dec)

    def write_unbinned_output(self, output_name="unbinned_data"):

        """
        Writes unbinned data file to either fits or hdf5.
        
        output_name: Option to specify name of output file. 
        """

        # Data units:
        units=['keV','s','rad:[glon,glat]','rad:[glon,glat]',
                'rad:[glon,glat]','rad','rad','rad','cm','rad','rad']
            
        # For fits output: 
        if self.unbinned_output == 'fits':
            table = Table(list(self.cosi_dataset.values()),\
                    names=list(self.cosi_dataset.keys()), \
                    units=units, \
                    meta={'data file':ntpath.basename(self.data_file)})
            table.write("%s.fits" %output_name, overwrite=True)
            os.system('gzip -f %s.fits' %output_name)

        # For hdf5 output:
        if self.unbinned_output == 'hdf5':
            with h5py.File('%s.hdf5' %output_name, 'w') as hf:
                for each in list(self.cosi_dataset.keys()):
                    dset = hf.create_dataset(each, data=self.cosi_dataset[each], compression='gzip')        
    
        return

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

        """
        Constructs dictionary from input hdf5 file
        
        input_hdf5: Name of input hdf5 file. 
        """

        # Initialize dictionary:
        this_dict = {}

        # Fill dictionary from input h5fy file:
        hf = h5py.File(input_hdf5,"r")
        keys = list(hf.keys())
        for each in keys:
            this_dict[each] = hf[each][:]

        return this_dict

    def select_data(self, unbinned_data=None, output_name="selected_unbinned_data"):

        """
        Applies cuts to unbinnned data dictionary. 
        Only cuts in time are allowed for now. 
        
        unbinned_data: Unbinned dictionary file. 
        output_name: Prefix of output file. 
        """
        
        print("Making data selections...")

        # Option to read in unbinned data file:
        if unbinned_data:
            if self.unbinned_output == 'fits':
                self.cosi_dataset = self.get_dict_from_fits(unbinned_data)
            if self.unbinned_output == 'hdf5':
                self.cosi_dataset = self.get_dict_from_hdf5(unbinned_data)

        # Get time cut index:
        time_array = self.cosi_dataset["TimeTags"]
        time_cut_index = (time_array >= self.tmin) & (time_array <= self.tmax)
    
        # Apply cuts to dictionary:
        for key in self.cosi_dataset:

            self.cosi_dataset[key] = self.cosi_dataset[key][time_cut_index]

        # Write unbinned data to file (either fits or hdf5):
        self.write_unbinned_output(output_name=output_name)

        return

    def combine_unbinned_data(self, input_files, output_name="combined_unbinned_data"):

        """
        Combines input unbinned data files.
        
        Inputs:
        input_files: List of file names to combine.
        output_name: prefix of output file. 
        """

        self.cosi_dataset = {}
        counter = 0
        for each in input_files:

            print()
            print("adding %s..." %each)
            print()
    
            # Read dict from hdf5 or fits:
            if self.unbinned_output == 'hdf5':
                this_dict = self.get_dict_from_hdf5(each)
            if self.unbinned_output == 'fits':
                this_dict = get_dict_from_fits(each)

            # Combine dictionaries:
            if counter == 0:
                for key in this_dict:
                    self.cosi_dataset[key] = this_dict[key]
            
            if counter > 0:
                for key in this_dict:
                    self.cosi_dataset[key] = np.concatenate((self.cosi_dataset[key],this_dict[key]))
                    
            counter =+ 1
        
        # Write unbinned data to file (either fits or hdf5):
        self.write_unbinned_output(output_name=output_name)

        return

