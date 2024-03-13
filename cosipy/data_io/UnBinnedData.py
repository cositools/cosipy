# Imports:
import numpy as np
from astropy.table import Table
from astropy.io import fits
from scipy import interpolate
import h5py
import time
from cosipy.data_io import DataIO
from cosipy.spacecraftfile import SpacecraftFile
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
import gc
import os
import time
logger = logging.getLogger(__name__)

class UnBinnedData(DataIO):
    """Handles unbinned data."""

    def read_tra(self, output_name=None, run_test=False, use_ori=False,
            event_min=None, event_max=None):
        
        """Reads MEGAlib .tra (or .tra.gz) file and creates cosi datset.
        
        Parameters
        ----------
        output_name : str, optional
            Prefix of output file (default is None, in which case no 
            output is written). 
        run_test : bool, optional 
            This is for unit testing only! Keep False unless 
            comparing to MEGAlib calculations. 
        use_ori : bool, optional
            Option to get pointing information from the orientation 
            file, based on event time-stamps (default is False, in 
            which case the pointing information comes from the event 
            file itself). Note: this is an option for now, but will 
            later be the default. 
        event_min : int, optional
            Minimum event number to process (inclusive). All events 
            below this will be skipped.        
        event_max : int, optional
            Maximum event number to process (non-inclusive). All 
            events at and above this will be skipped. 
            
            Note: event_min and event_max correspond to the total 
            number of events in the file, which is not necessarily the 
            same as the event ID number. The purpose of this is to 
            allow the data to be read in chunks, in order to overcome 
            memory limitations, depending on the user's system.

        Returns
        -------
        cosi_dataset, dict
            The returned dictionary contains the COSI dataset, which
            has the form:
            cosi_dataset = {'Energies':erg,\
                        'TimeTags':tt,\
                        'Xpointings':np.array([lonX,latX]).T,\
                        'Ypointings':np.array([lonY,latY]).T,\
                        'Zpointings':np.array([lonZ,latZ]).T,\
                        'Phi':phi,\
                        'Chi local':chi_loc,\
                        'Psi local':psi_loc,\
                        'Distance':dist,\
                        'Chi galactic':chi_gal,\
                        'Psi galactic':psi_gal}\
            Arrays contain unbinned photon data. 
        
        Notes
        -----
        The current code is only able to handle data with Compton 
        events. It will need to be modified to handle single-site 
        and pair events. 

        This method sets the instance attribute, cosi_dataset, 
        but it does not explicitly return this.  
        """
   
        start_time = time.time()

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

        # This is for unit testing purposes only.
        # Use same value as MEGAlib for direct comparison: 
        if run_test == True:
            c_E0 = 510.999

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
        N_events = 0 # number of events
        pbar = tqdm(total=num_lines) # start progress bar
        for line in f:
         
            this_line = line.strip().split()
            pbar.update(1) # update progress bar

            # Make sure line isn't empty:
            if len(this_line) == 0:
                continue

            # Count the number of events:
            if this_line[0] == "ID":
                N_events += 1
                
            # Option to only parse a subset of events:
            if event_min != None:
                if N_events < event_min:
                    continue
            if event_max != None:
                if N_events >= event_max:
                    pbar.close()
                    print("Stopping here: only reading a subset of events")
                    break

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
                    dg = v1 - v2
                    dg_x.append(dg[0])
                    dg_y.append(dg[1])
                    dg_z.append(dg[2])

        # Close progress bar:
        pbar.close()
        print("Making COSI data set...")
        print("total events to procecss: " + str(len(erg)))

        # Clear unused memory:
        gc.collect()

        # Initialize arrays:
        print("Initializing arrays...")
        erg = np.array(erg)
        phi = np.array(phi)
        tt = np.array(tt)
        et = np.array(et)
        lonX = np.array(lonX)
        latX = np.array(latX)
        lonZ = np.array(lonZ)
        latZ = np.array(latZ)
        dg_x = np.array(dg_x)
        dg_y = np.array(dg_y)
        dg_z = np.array(dg_z)
 
        # Check if the input data has pointing information, 
        # if not, get it from the spacecraft file:
        if (use_ori == False) & (len(lonZ)==0):
            print("WARNING: No pointing information in input data.")
            print("Getting pointing information from spacecraft file.")
            use_ori = True

        # Option to get X and Z pointing information from orientation file:
        if use_ori == True:
            self.instrument_pointing()
            lonX = self.xl_interp(tt)
            latX = self.xb_interp(tt)
            lonZ = self.zl_interp(tt)
            latZ = self.zb_interp(tt)

        # Convert dg vector from 3D cartesian coordinates 
        # to spherical polar coordinates, and then extract distance 
        # b/n first two interactions (in cm), psi (rad), and chi (rad).
        # Note: the resulting angles are latitude/longitude (or elevation/azimuthal).
        conv = astro_co.cartesian_to_spherical(dg_x, dg_y, dg_z)
        dist = conv[0].value 
        psi_loc = conv[1].value 
        chi_loc = conv[2].value

        # Calculate chi_gal and psi_gal from x,y,z coordinates of events:
        xcoords = SkyCoord(lonX*u.rad, latX*u.rad, frame = 'galactic')
        zcoords = SkyCoord(lonZ*u.rad, latZ*u.rad, frame = 'galactic')
        attitude = Attitude.from_axes(x=xcoords, z=zcoords, frame = 'galactic')
        c = SkyCoord(dg_x, dg_y, dg_z, \
            representation_type='cartesian', frame = SpacecraftFrame(attitude = attitude))   
        c_rotated = c.transform_to('galactic')
        chi_gal = np.array(c_rotated.l.deg)
        psi_gal = np.array(c_rotated.b.deg)

        # Change longitudes from 0..360 deg to -180..180 deg
        lonX[lonX > np.pi] -= 2*np.pi
        lonZ[lonZ > np.pi] -= 2*np.pi

        # Construct Y direction from X and Z direction
        lonlatY = self.construct_scy(np.rad2deg(lonX),np.rad2deg(latX),
                            np.rad2deg(lonZ),np.rad2deg(latZ))
        lonY = np.deg2rad(lonlatY[0])
        latY = np.deg2rad(lonlatY[1])
    
        # Rotate psi_loc to colatitude, measured from positive z direction.
        # This is requred for mhealpy input.
        psi_loc = (np.pi/2.0) - psi_loc 
        
        # Define test values for psi and chi local;
        # this is only for comparing to MEGAlib:
        self.psi_loc_test = psi_loc
        self.chi_loc_test = chi_loc

        # Do the same for psi and chi galactic.
        # First need to convert to radians:
        chi_gal_rad = np.array(c_rotated.l.rad)
        psi_gal_rad = np.array(c_rotated.b.rad)
        
        # Rotate psi_gal_rad to colatitude, measured from positive 
        # z direction:
        psi_gal_rad = (np.pi/2.0) - psi_gal_rad
        self.psi_gal_test = psi_gal_rad
        
        # Rotate chi_gal_test by pi, defined with respect to 
        # the negative x-axis:
        self.chi_gal_test = chi_gal_rad - np.pi
        
        # Make observation dictionary
        print("Making dictionary...")
        cosi_dataset = {'Energies':erg,
                        'TimeTags':tt,
                        'Xpointings (glon,glat)':np.array([lonX,latX]).T,
                        'Ypointings (glon,glat)':np.array([lonY,latY]).T,
                        'Zpointings (glon,glat)':np.array([lonZ,latZ]).T,
                        'Phi':phi,
                        'Chi local':chi_loc,
                        'Psi local':psi_loc,
                        'Distance':dist,
                        'Chi galactic':chi_gal,
                        'Psi galactic':psi_gal} 
        self.cosi_dataset = cosi_dataset

        # Option to write unbinned data to file (either fits or hdf5):
        if output_name != None:
            print("Saving file...")
            self.write_unbinned_output(output_name) 
        
        # Get processing time:
        end_time = time.time()
        processing_time = end_time - start_time
        print("total processing time [s]: " + str(processing_time))
        
        return 

    def instrument_pointing(self):

        """Get pointing information from ori file.
        
        Initializes interpolated functions for lonx, latx, lonz, latz 
        in radians.
        
        Returns
        -------
        xl_interp : scipy:interpolate:interp1d
        xb_interp : scipy:interpolate:interp1d
        zl_interp : scipy:interpolate:interp1d
        zb_interp : scipy:interpolate:interp1d

        Note
        ----
            This method sets the instance attributes, 
            but it does not explicitly return them.
        """

        # Get ori info:
        ori = SpacecraftFile.parse_from_file(self.ori_file)
        time_tags = ori._load_time
        x_pointings = ori.x_pointings
        z_pointings = ori.z_pointings

        # Interpolate:
        self.xl_interp = interpolate.interp1d(time_tags, x_pointings.l.rad, kind='linear')
        self.xb_interp = interpolate.interp1d(time_tags, x_pointings.b.rad, kind='linear')
        self.zl_interp = interpolate.interp1d(time_tags, z_pointings.l.rad, kind='linear')
        self.zb_interp = interpolate.interp1d(time_tags, z_pointings.b.rad, kind='linear')
        
        return 

    def construct_scy(self, scx_l, scx_b, scz_l, scz_b):
    
        """Construct y-coordinate of instrument pointing given x and z directions.
        
        Parameters
        ----------
        scx_l : float
            Longitude of x direction in degrees.
        scx_b : float
            Latitude of x direction in degrees.
        scz_l : float
            Longitude of z direction in degrees.
        scz_b : float
            Latitude of z direction in degrees.

        Returns
        -------
        ra : float
            Right ascension (in degrees) for y-coordinate of instrument pointing.
        dec : float
            Declination (in degrees) for y-coordinate of instrument pointing.

        Note
        ----
            Here, z is the optical axis.
        """
        
        x = self.polar2cart(scx_l, scx_b)
        z = self.polar2cart(scz_l, scz_b)
    
        return self.cart2polar(np.cross(z,x,axis=0))

    def polar2cart(self, ra, dec):
    
        """Coordinate transformation of ra/dec (lon/lat) [phi/theta] 
        polar/spherical coordinates into cartesian coordinates.

        Parameters
        ----------
        ra : float
            Right ascension in degrees. 
        dec: float
            Declination in degrees. 
        
        Returns
        -------
        array
            x, y, and z cartesian coordinates in radians.
        """
        
        x = np.cos(np.deg2rad(ra)) * np.cos(np.deg2rad(dec))
        y = np.sin(np.deg2rad(ra)) * np.cos(np.deg2rad(dec))
        z = np.sin(np.deg2rad(dec))
    
        return np.array([x,y,z])

    def cart2polar(self, vector):
    
        """Coordinate transformation of cartesian x/y/z values into 
        spherical (deg).

        Parameters
        ----------
        vector : vec
            Vector of x/y/z values.

        Returns
        -------
        ra : float
            Right ascension in degrees.
        dec : float
            Declination in degrees. 
        """
        
        ra = np.arctan2(vector[1],vector[0]) 
        dec = np.arcsin(vector[2])
    
        return np.rad2deg(ra), np.rad2deg(dec)

    def write_unbinned_output(self, output_name):

        """Writes unbinned data file to either fits or hdf5.
        
        Parameters
        ----------
        output_name : str 
            Name of output file. Only include prefix (not file type). 
        """

        # Data units:
        units=['keV','s','rad','rad',
                'rad','rad','rad','rad','cm','deg','deg']
            
        # For fits output: 
        if self.unbinned_output == 'fits':
            table = Table(list(self.cosi_dataset.values()),\
                    names=list(self.cosi_dataset.keys()), \
                    units=units, \
                    meta={'data file':os.path.basename(self.data_file), \
                    'version':1.0})
            table.write("%s.fits" %output_name, overwrite=True)
            os.system('gzip -f %s.fits' %output_name)

        # For hdf5 output:
        if self.unbinned_output == 'hdf5':
            with h5py.File('%s.hdf5' %output_name, 'w') as hf:
                for each in list(self.cosi_dataset.keys()):
                    dset = hf.create_dataset(each, data=self.cosi_dataset[each], compression='gzip')        
    
        return

    def get_dict_from_fits(self, input_fits):

        """Constructs dictionary from input fits file.
        
        Parameters
        ----------
        input_fits : str
            Name of input fits file.

        Returns
        -------
        dict
            Dictionary constructed from input fits file.
        """

        # Initialize dictionary:
        this_dict = {}
        
        # Fill dictionary from input fits file:
        hdu = fits.open(input_fits,memmap=True)
        cols = hdu[1].columns
        data = hdu[1].data
        for i in range(0,len(cols)):
            
            this_key = cols[i].name
            this_dict[this_key] = data[this_key]
            
            # Clear unused memory:
            gc.collect()

        return this_dict

    def get_dict_from_hdf5(self, input_hdf5):

        """Constructs dictionary from input hdf5 file
        
        Parameters
        ----------
        input_hdf5 : str
            Name of input hdf5 file. 

        Returns
        -------
        dict
            Dictionary constructed from input hdf5 file.
        """

        # Initialize dictionary:
        this_dict = {}

        # Fill dictionary from input h5fy file:
        hf = h5py.File(input_hdf5,"r")
        keys = list(hf.keys())
        for each in keys:
            this_dict[each] = hf[each][:]

        return this_dict

    def select_data(self, output_name=None, unbinned_data=None):

        """Applies cuts to unbinnned data dictionary. 
        
        Parameters
        ----------
        unbinned_data : str, optional
            Name of unbinned dictionary file.
        output_name : str, optional
            Prefix of output file (default is None, in which case no 
            file is saved).
        
        Note
        ----
        Only cuts in time are allowed for now. 
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
        time_cut_index = (time_array >= self.tmin) & (time_array < self.tmax)
    
        # Apply cuts to dictionary:
        for key in self.cosi_dataset:

            self.cosi_dataset[key] = self.cosi_dataset[key][time_cut_index]

        # Write unbinned data to file (either fits or hdf5):
        if output_name != None:
            print("Saving file...")
            self.write_unbinned_output(output_name)

        return

    def combine_unbinned_data(self, input_files, output_name=None):

        """Combines input unbinned data files.
        
        Parameters
        ----------
        input_files : list
            List of file names to combine.
        output_name : str, optional
            Prefix of output file.
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
                this_dict = self.get_dict_from_fits(each)

            # Combine dictionaries:
            if counter == 0:
                for key in this_dict:
                    self.cosi_dataset[key] = this_dict[key]
            
            if counter > 0:
                for key in this_dict:
                    self.cosi_dataset[key] = np.concatenate((self.cosi_dataset[key],this_dict[key]))
                    
            counter =+ 1
            
            # Clear unused memory:
            gc.collect()

        # Write unbinned data to file (either fits or hdf5):
        if output_name != None:
            self.write_unbinned_output(output_name)

        return
