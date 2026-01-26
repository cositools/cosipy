import sys
import gc
import time
import subprocess
import gzip

from tqdm import tqdm

import numpy as np
from scipy import interpolate

import astropy.units as u
from astropy.coordinates import SkyCoord, cartesian_to_spherical
from astropy.table import Table
from astropy.io import fits

import h5py

from scoords import Attitude, SpacecraftFrame

import cosipy
from cosipy.data_io import DataIO
from cosipy.spacecraftfile import SpacecraftFile

import logging
logger = logging.getLogger(__name__)


class UnBinnedData(DataIO):
    """Handles unbinned data."""

    def read_tra(self, input_name=None, output_name=None,
                 run_test=False, use_ori=False,
                 sc_orientation = None,
                 event_min=None, event_max=None):

        """Reads MEGAlib .tra or .tra.gz file and creates COSI datset.

        Parameters
        ----------
	input_name : str, optional
            Path of input file (default is None, in which case the
	    input file name is taken from the yaml file).
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
            file itself).
        sc_orientation : Path/string or Attitude, optional
            If not None, either a single orientation (Attitude of length 1)
            or the name of a file with spacecraft orientation data. This will
            be used to compute X and Z axes for each data point, overriding
            any values in either the .tra file or any orientation data
            that is part of the UnbinnedData object.
        event_min : int, optional
            Minimum valid event number to process (inclusive). All
            valid events before this one will be skipped.
        event_max : int, optional
            Maximum valid event number to process (non-inclusive). All
            events at and above this one will be skipped.

        Note: event_min and event_max index the *valid* events in the
        file, and they are 1-based (so event_min=2 skips the first
        valid event but returns the second). Invalid events,
        specifically those with only a single Compton scatter, do not
        count toward the event_min and event_max thresholds. These
        values do *not* reference the event ID number. Their purpose
        is to allow the data to be read in chunks, in order to
        overcome memory limitations of the user's system.

        Result
        -------
        This method sets the instance attribute cosi_dataset, a
        dictionary with the following members:

        cosi_dataset = {
            'Energies'   : 1D np.ndarray [float] (keV)
            'TimeTags'   : 1D np.ndarray [float] (UNIX sec)
            'Xpointings' : 2D np.ndarray [float] N x 2 (lon/lat radians)
            'Ypointings' : 2D np.ndarray [float] N x 2 (lon/lat radians)
            'Zpointings' : 2D np.ndarray [float] N x 2 (lon/lat radians)
            'Phi'        : 1D np.ndarray [float] (radians)
            'Chi local'  : 1D np.ndarray [float] (radians)
            'Psi local'  : 1D np.ndarray [float] (radians)
            'Distance'   : 1D np.ndarray [float] (cm)
            'Chi galactic' : 1D np.ndarray [float] (degrees)
            'Psi galactic' : 1D np.ndarray [float] (degrees)
            'CO seq'     : 1D np.ndarray [int]
        }

        Arrays contain unbinned photon data and are bare arrays,
        not Quantities.

        Notes
        -----
        The current code is only able to handle data with Compton
        events. It will need to be modified to handle single-site
        and pair events.

        """

        start_time = time.time()

        if input_name is not None:
            self.data_file = input_name

        if event_min is not None:
            event_min -= 1  # input is 1-based; make it 0-based
        else:
            event_min = 0

        if event_max is not None:
            event_max -= 1 # input is 1-based; make it 0-based

        logger.info("Preparing to read file...")

        if self.data_file.endswith(".tra.gz"):
            # Need to get number of lines for progress bar.  First try
            # fast method for unix-based systems

            try:
                proc=subprocess.Popen(f'gunzip -c {self.data_file} | wc -l',
                                      shell=True, stdout=subprocess.PIPE)
                num_lines = int(proc.communicate()[0])

            # If fast method fails, use long method, which should work
            # in all cases
            except:
                logger.info("Initial attempt failed.")
                logger.info("Using long method...")
                g = gzip.open(self.data_file,"rt")
                num_lines = sum(1 for line in g)
                g.close()

            f = gzip.open(self.data_file,"rt")

        elif self.data_file.endswith(".tra"):
            # Need to get number of lines for progress bar.  First try
            # fast method for unix-based systems
            try:
                proc=subprocess.Popen(f'wc -l < {self.data_file}',
                                      shell=True, stdout=subprocess.PIPE)
                num_lines = int(proc.communicate()[0])

            # If fast method fails, use long method, which should work
            # in all cases

            except:
                logger.info("Initial attempt failed.")
                logger.info("Using long method...")
                g = open(self.data_file,"rt")
                num_lines = sum(1 for line in g)
                g.close()

            # Open uncompressed .tra file
            f = open(self.data_file,"r")

        else:
            logger.error("ERROR: Input data file must have '.tra' or '.tra.gz' extenstion.")
            sys.exit()

        # Read tra file line by line
        logger.info("Reading file...")

        pbar = tqdm(total = num_lines)


        # lists to hold parsed values
        m_eg, m_ee, tt, CO_seq = [], [], [], []

        # Components of dg (position vector from 1st interaction to 2nd)
        dg_x, dg_y, dg_z = [], [], []

        # Galactic lat/lon of X,Z directions of spacecraft
        lonX, latX, lonZ, latZ = [], [], [], []

        N_valid_events = 0 # number of *valid* events read
        prev_event_valid = False
        this_event = None
        lines_read = 0

        for line in f:
            lines_read += 1

            this_line = line.lstrip()
            match this_line[:2]: # will be "" for empty line
                case "SE": # New event marker

                    # Previous event was read but is not valid.
                    # This is very common because we are invalidating
                    # every single-hit event, so don't spend time
                    # logging such events.
                    '''
                    if not prev_event_valid and this_event is not None:
                        logger.info("bad_event: no second hit info")
                        logger.info(f"bad event ID: {this_event['id']}")
                    '''

                    if event_max is not None and N_valid_events >= event_max:
                        logger.info(f"Stopping after reading {event_max} valid events")
                        break

                    # update progress bar for prior event
                    pbar.update(lines_read)
                    lines_read = 0

                    # start a new event
                    this_event = {}

                case "ET": # Event type
                    fields = this_line.split(maxsplit=2)
                    event_type = fields[1]
                    if event_type != "CO":
                        raise ValueError(f"Error: Expected CO event, but got '{event_type}'")

                # used only for printing invalid events
                #case "ID": # Event identifier
                #    fields = this_line.split(maxsplit=2)
                #    this_event["id"] = fields[1]

                case "CE": # energies of scattered gamma ray and recoil electron
                    fields = this_line.split(maxsplit=4)
                    this_event["m_eg"] = float(fields[1])
                    this_event["m_ee"] = float(fields[3])

                case "TI": # Time tag in Unix time (seconds)
                    fields = this_line.split(maxsplit=2)
                    this_event["tt"] = float(fields[1])

                case "GX": # X axis of detector orientation in Galactic coords
                    fields = this_line.split(maxsplit=3)
                    this_event["lonX"] = float(fields[1]) # degrees
                    this_event["latX"] = float(fields[2]) # degrees

                case "GZ": # Z axis of detector orientation in Galactic coords
                    fields = this_line.split(maxsplit=3)
                    this_event["lonZ"] = float(fields[1]) # degrees
                    this_event["latZ"] = float(fields[2]) # degrees

                case "SQ": # Number of Compton scattering interactions
                    fields = this_line.split(maxsplit=2)
                    this_event["CO_seq"] = int(fields[1])

                case "CH": # Position info for one interaction
                    fields = this_line.split(maxsplit=5)
                    interaction_id = int(fields[1])
                    if interaction_id > 1:
                        continue

                    v = (float(fields[2]),
                         float(fields[3]),
                         float(fields[4]))

                    # we need the locations of the first two
                    # interactions to create a valid event
                    if interaction_id == 0:
                        # First interaction - save
                        this_event["v1"] = v
                    elif interaction_id == 1:
                        # Second interaction - complete event
                        prev_event_valid = True
                        N_valid_events += 1

                        if N_valid_events > event_min:
                            # save this event
                            m_eg.append(this_event["m_eg"])
                            m_ee.append(this_event["m_ee"])
                            tt.append(this_event["tt"])
                            CO_seq.append(this_event["CO_seq"])

                            # compute position vector between
                            # first two interactions
                            v1 = this_event["v1"]
                            dg_x.append(v1[0] - v[0])
                            dg_y.append(v1[1] - v[1])
                            dg_z.append(v1[2] - v[2])

                            # detector orientation may or may not
                            # be provided in .tra file
                            if "latX" in this_event:
                                latX.append(this_event["latX"])
                                lonX.append(this_event["lonX"])
                                latZ.append(this_event["latZ"])
                                lonZ.append(this_event["lonZ"])

        pbar.update(num_lines) # update progress bar for end of read
        pbar.close()

        f.close()

        logger.info("Making COSI data set...")
        logger.info(f"total events to process: {len(tt)}")

        # Initialize arrays
        logger.info("Initializing arrays...")

        m_Eg = np.array(m_eg)
        m_Ee = np.array(m_ee)

        # total energy of photon
        erg = m_Eg + m_Ee

        # Electron rest energy, used to compute Compton scattering
        # angle.  When testing, use same value as MEGAlib for direct
        # comparison.
        c_E0 = 510.999 if run_test else 510.9989500015 # keV

        # Compton scatter angle, computed via standard Compton law. We
        # neglect the movement of the electron, which would lead to
        # Doppler-broadening.
        phi = np.arccos(1. - c_E0 * (1./m_Eg - 1./erg))

        tt = np.array(tt)
        CO_seq = np.array(CO_seq)
        dg_x = np.array(dg_x)
        dg_y = np.array(dg_y)
        dg_z = np.array(dg_z)

        if sc_orientation is not None:
            if isinstance(sc_orientation, Attitude):
                # input is an Attitude to be applied to all time points
                
                if sc_orientation.shape != (): # not a scalar Attitude
                    if sc_orientation.shape == (1,): 
                        sc_orientation = sc_orientation[0] # extract scalar version
                    else:
                        raise ValueError("Attitude supplied as sc_orientation must contain only a single pointing")

                x, _, z = sc_orientation.as_axes()
                lonX = np.full(len(tt), x.l.rad)
                latX = np.full(len(tt), x.b.rad)
                lonZ = np.full(len(tt), z.l.rad)
                latZ = np.full(len(tt), z.b.rad)
            else:
                # input is name of orientation file to use
                self.instrument_pointing(sc_orientation)
                lonX = self.xl_interp(tt)
                latX = self.xb_interp(tt)
                lonZ = self.zl_interp(tt)
                latZ = self.zb_interp(tt)
        elif use_ori:
            # Use the X and Z pointing information in the orientation
            # file; ignore any info read from the .tra file
            self.instrument_pointing(self.ori_file)
            lonX = self.xl_interp(tt)
            latX = self.xb_interp(tt)
            lonZ = self.zl_interp(tt)
            latZ = self.zb_interp(tt)
        elif len(lonZ) == 0:
            # No pointing info provided in the .tra file
            raise ValueError("No pointing information in input data and no orientation file.")
        else:
            lonX = np.deg2rad(lonX)
            latX = np.deg2rad(latX)
            lonZ = np.deg2rad(lonZ)
            latZ = np.deg2rad(latZ)

        # Change longitudes from 0..360 deg to -180..180 deg
        lonX[lonX > np.pi] -= 2*np.pi
        lonZ[lonZ > np.pi] -= 2*np.pi

        # Construct Y direction from X and Z directions
        lonY, latY = self.construct_scy(lonX, latX,
                                        lonZ, latZ)

        # Convert dg vector from 3D cartesian coordinates to spherical
        # polar coordinates, and then extract distance b/n first two
        # interactions (in cm), psi (rad), and chi (rad). The angles
        # psi/chi are latitude/longitude (or elevation/azimuthal).
        conv = cartesian_to_spherical(dg_x, dg_y, dg_z)
        dist    = conv[0].value
        psi_loc = conv[1].value
        chi_loc = conv[2].value

        # calculate galactic-frame psi and chi
        xcoords = SkyCoord(lonX, latX, unit=u.rad, frame='galactic')
        zcoords = SkyCoord(lonZ, latZ, unit=u.rad, frame='galactic')
        attitude = Attitude.from_axes(x=xcoords, z=zcoords, frame='galactic')

        c = SkyCoord(lat=psi_loc, lon=chi_loc, unit=u.rad,
                     frame=SpacecraftFrame(attitude=attitude))
        c_rotated = c.transform_to('galactic')
        psi_gal = c_rotated.b.deg
        chi_gal = c_rotated.l.deg

        # psi_loc must be expressed as co-latitude for mhealpy
        # compatibility
        psi_loc = np.pi/2. - psi_loc

        if run_test:
            # Define test values for psi and chi in local and
            # galactic; this is only for comparing to MEGAlib.

            self.psi_loc_test = psi_loc
            self.chi_loc_test = chi_loc

            chi_gal_rad = np.deg2rad(chi_gal)
            psi_gal_rad = np.deg2rad(psi_gal)

            # Rotate psi_gal_rad to colatitude, measured from positive
            # z direction
            self.psi_gal_test = np.pi/2. - psi_gal_rad

            # Rotate chi_gal_test by pi, defined with respect to
            # the negative x-axis
            self.chi_gal_test = chi_gal_rad - np.pi

        # Make observation dictionary
        logger.info("Making dictionary...")

        cosi_dataset = {
            'Energies' : erg,
            'TimeTags' : tt,
            'Xpointings (glon,glat)' : np.column_stack((lonX,latX)),
            'Ypointings (glon,glat)' : np.column_stack((lonY,latY)),
            'Zpointings (glon,glat)' : np.column_stack((lonZ,latZ)),
            'Phi' : phi,
            'Chi local' : chi_loc,
            'Psi local' : psi_loc,
            'Distance' : dist,
            'Chi galactic' : chi_gal,
            'Psi galactic' : psi_gal,
            'Compton Seq' : CO_seq
        }
        
        # For simulation the timetags are shuffled, so we need to sort
        # it by ascending order .
        logger.info("Sorting the dictionary by ascending TimeTags")

        # Build sorting index
        idx = np.argsort(cosi_dataset["TimeTags"])

        # Reorder all columns using the same index
        for key in cosi_dataset:
            cosi_dataset[key] = cosi_dataset[key][idx]        
        
        self.cosi_dataset = cosi_dataset

        if output_name is not None:
            # write unbinned data to file
            logger.info("Saving file...")
            self.write_unbinned_output(output_name)

        end_time = time.time()
        processing_time = end_time - start_time
        logger.info(f"total processing time [s]: {processing_time}")


    def instrument_pointing(self, ori_file):

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

        # Get orientation info
        ori = SpacecraftFile.open(self.ori_file)
        time_tags = ori.get_time().to_value(format="unix")
        x_pointings = ori.x_pointings
        z_pointings = ori.z_pointings

        # Interpolate:
        self.xl_interp = interpolate.interp1d(time_tags, x_pointings.l.rad, kind='linear')
        self.xb_interp = interpolate.interp1d(time_tags, x_pointings.b.rad, kind='linear')
        self.zl_interp = interpolate.interp1d(time_tags, z_pointings.l.rad, kind='linear')
        self.zb_interp = interpolate.interp1d(time_tags, z_pointings.b.rad, kind='linear')

    def construct_scy(self, scx_l, scx_b, scz_l, scz_b):

        """Construct y-coordinate of instrument pointing given x and z directions.

        Parameters
        ----------
        scx_l : float
            Longitude of x direction in radians.
        scx_b : float
            Latitude of x direction in radians.
        scz_l : float
            Longitude of z direction in radians.
        scz_b : float
            Latitude of z direction in radians.

        Returns
        -------
        ra : float
            Right ascension (in radians) for y-coordinate of instrument pointing.
        dec : float
            Declination (in radians) for y-coordinate of instrument pointing.

        Note
        ----
        Here, z is the optical axis.
        """

        x = self.polar2cart(scx_l, scx_b)
        z = self.polar2cart(scz_l, scz_b)

        y = np.cross(z, x, axis=0)
        return self.cart2polar(y)

    def polar2cart(self, ra, dec):

        """Coordinate transformation of ra/dec (lon/lat) [phi/theta]
        polar/spherical coordinates into cartesian coordinates.

        Parameters
        ----------
        ra : float
            Right ascension in radians.
        dec: float
            Declination in radians.

        Returns
        -------
        array
            x, y, and z cartesian coordinates in radians.
        """

        cosdec = np.cos(dec)
        x = np.cos(ra) * cosdec
        y = np.sin(ra) * cosdec
        z = np.sin(dec)

        return np.array((x,y,z))

    def cart2polar(self, vector):

        """Coordinate transformation of cartesian x/y/z values into
        spherical (rad).

        Parameters
        ----------
        vector : vec
            Vector of x/y/z values.

        Returns
        -------
        ra : float
            Right ascension in radians.
        dec : float
            Declination in radians.

        """

        x, y, z = vector
        ra  = np.arctan2(y, x)
        dec = np.arcsin(z)

        return ra, dec

    def write_unbinned_output(self, output_name):

        """Writes unbinned data file to either fits or hdf5.

        Parameters
        ----------
        output_name : str
            Name of output file. Only include prefix (not file type).
        """

        # Units for new DC4 structure of the data
        units = (u.keV, u.s,   u.rad, u.rad,
                 u.rad, u.rad, u.rad, u.rad,
                 u.cm,  u.deg, u.deg,
                 u.dimensionless_unscaled)

        # Old UnBinned data structure did not have the last field
        # (CO_seq); this special case should be removed for DC4.
        units = units[:len(self.cosi_dataset.keys())]

        if self.unbinned_output == 'fits':
            # For fits output
            table = Table(self.cosi_dataset,
                          units=units,
                          meta={'version':cosipy.__version__})
            table.write(f"{output_name}.fits.gz", overwrite=True)
        elif self.unbinned_output == 'hdf5':
            # For hdf5 output
            with h5py.File(f'{output_name}.hdf5', 'w') as hf:
                for each in self.cosi_dataset.keys():
                    hf.create_dataset(each, data=self.cosi_dataset[each], compression='gzip')

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

        with fits.open(input_fits,memmap=True) as hdu:
            cols = hdu[1].columns.names
            data = hdu[1].data

            this_dict = {}
            for key in cols:
                this_dict[key] = data[key]

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

        with h5py.File(input_hdf5,"r") as hf:
            this_dict = {}
            for each in hf.keys():
                this_dict[each] = np.array(hf[each])

        return this_dict

    def get_dict(self, input_file):

        """Constructs dictionary from input file.

        Parameters
        ----------
        input_file : str
            Name of input file.

        Returns
        -------
        dict
            Dictionary constructed from input file.
        """

        if self.unbinned_output == 'fits':
            this_dict = self.get_dict_from_fits(input_file)
        elif self.unbinned_output == 'hdf5':
            this_dict = self.get_dict_from_hdf5(input_file)

        return this_dict

    def select_data_time(self, output_name=None, unbinned_data=None):

        """Applies time cuts to unbinned data dictionary.

        Parameters
        ----------
        unbinned_data : str, optional
            Name of unbinned dictionary file.
        output_name : str, optional
            Prefix of output file (default is None, in which case no
            file is saved).

        """

        logger.info("Making data selections...")

        # Option to read in unbinned data file
        if unbinned_data:
            self.cosi_dataset = self.get_dict(unbinned_data)

        # identify all entries with times in
        # half-open range [self.tmin, self.tmax)
        time_array = self.cosi_dataset["TimeTags"]
        time_cut_index = (time_array >= self.tmin) & (time_array < self.tmax)

        # Apply cuts to dictionary
        for key in self.cosi_dataset:
            self.cosi_dataset[key] = self.cosi_dataset[key][time_cut_index]

        # Write unbinned data to file (either fits or hdf5)
        if output_name is not None:
            logger.info("Saving file...")
            self.write_unbinned_output(output_name)


    def select_data_energy(self, emin, emax,
                           output_name=None,
                           unbinned_data=None):

        """Applies energy cuts to unbinnned data dictionary.

        Parameters
        ----------
        emin : float or int
            Minimum energy in keV.
        emax : float or int
            Maximum energy in keV.
        unbinned_data : str, optional
            Name of unbinned dictionary file.
        output_name : str, optional
            Prefix of output file (default is None, in which case no
            file is saved).
        """

        logger.info("Making data selections on photon energy...")

        # Option to read in unbinned data file
        if unbinned_data:
            self.cosi_dataset = self.get_dict(unbinned_data)

        # identify all entries with energies in
        # half-open range [emin, emax)
        energy_array = self.cosi_dataset["Energies"]
        energy_cut_index = (energy_array >= emin) & (energy_array < emax)

        # Apply cuts to dictionary
        for key in self.cosi_dataset:
            self.cosi_dataset[key] = self.cosi_dataset[key][energy_cut_index]

        # Write unbinned data to file (either fits or hdf5)
        if output_name is not None:
            logger.info("Saving file...")
            self.write_unbinned_output(output_name)


    def select_data_COseq(self, seqmin, seqmax,
                          output_name=None, unbinned_data=None):

        """Applies CO sequence cuts [seqmin,seqmax) to unbinnned data dictionary.

        Parameters
        ----------
        seqmin :  int
            Minimum number of interaction.
        seqmax :  int
            Maximum number of interaction.
        unbinned_data : str, optional
            Name of unbinned dictionary file.
        output_name : str, optional
            Prefix of output file (default is None, in which case no
            file is saved).
        """

        logger.info("Making data selections on Compton Sequence...")

        # Option to read in unbinned data file
        if unbinned_data:
            self.cosi_dataset = self.get_dict(unbinned_data)

        # identify all entries with # of Compton scatters
        # in half-open range [seqmin, seqmax)
        COseq_array = self.cosi_dataset["Compton Seq"]
        COseq_cut_index = (COseq_array >= seqmin) & (COseq_array < seqmax)

        # Apply cuts to dictionary:
        for key in self.cosi_dataset:
            self.cosi_dataset[key] = self.cosi_dataset[key][COseq_cut_index]

        # Write unbinned data to file (either fits or hdf5):
        if output_name is not None:
            logger.info("Saving file...")
            self.write_unbinned_output(output_name)


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

            logger.info(f"adding {each}...")

            # Read dict from hdf5 or fits
            this_dict = self.get_dict(each)

            # Combine dictionaries
            for key in this_dict:
                if counter == 0:
                    self.cosi_dataset[key] = this_dict[key]
                else: # counter > 0
                    self.cosi_dataset[key] = np.concatenate((self.cosi_dataset[key], this_dict[key]))

            counter =+ 1

            del this_dict
            gc.collect()

        # Write unbinned data to file (either fits or hdf5):
        if output_name is not None:
            self.write_unbinned_output(output_name)

        return

    def find_bad_intervals(self, times, values, bad_value=0.0):

        """Finds intervals where livetime is 0.0.

        Parameters
        ----------
        times : array
            Array of times from ori file.
        values : array
            Array of livetimes from ori file.
        bad_value : float or int
            The value that defines a bad time interval. It must match
            exactly the value in the ori file, including the
            type (float or int). Default is 0.0.

        Returns
        -------
        bad_intervals : list of tuples
            List of bad time intervals.
        """

        bad_intervals = []
        start = None

        for time, value in zip(times[:-1], values[:-1]):
            if value == bad_value:
                if start is None:
                    start = time
            else:
                if start is not None:
                    bad_intervals.append((start.value, time.value))
                    start = None

        if start is not None:
            bad_intervals.append((start.value, times[-1].value))

        return bad_intervals

    def filter_good_data(self, times, bad_intervals):

        """Removes entries that fall within bad intervals.

        Parameters
        ----------
        times : array
            Array of photon event times.
        bad_intervals : list
            List of bad intervals defined by livetime = 0.0.

        Returns
        -------
        filtered_index : list
            List of indices of good events.
        """

        filtered_index = []

        # Get indices for good times.  The inequality below
        # corresponds to left end point in the orientation file.
        for i in range(len(times)-1):
            if not any(start < times[i] < end for start, end in bad_intervals):
                filtered_index.append(i)

        return filtered_index

    def cut_SAA_events(self, unbinned_data=None, output_name=None):

        """Cuts events corresponding to SAA passage based on input ori file.

        Parameters
        ----------
        unbinned_data : str, optional
            Name of unbinned dictionary file.
        output_name : str, optional
            Prefix of output file (default is None, in which case no
            file is saved).
        """

        # Option to read in unbinned data file
        if unbinned_data:
            self.cosi_dataset = self.get_dict(unbinned_data)

        # Get orientation info
        ori = SpacecraftFile.open(self.ori_file)

        # Get bad time intervals
        bti = self.find_bad_intervals(ori._time, ori.livetime)

        # Get indices for good photons
        time_keep_index = self.filter_good_data(self.cosi_dataset['TimeTags'], bti)

        # Apply cuts to dictionary
        for key in self.cosi_dataset:
            self.cosi_dataset[key] = self.cosi_dataset[key][time_keep_index]

        # Write unbinned data to file (either fits or hdf5)
        if output_name is not None:
            logger.info("Saving file...")
            self.write_unbinned_output(output_name)
