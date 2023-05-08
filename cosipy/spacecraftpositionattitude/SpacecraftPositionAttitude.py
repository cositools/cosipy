import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.io import fits
from astropy.coordinates import SkyCoord, cartesian_to_spherical
from mhealpy import HealpixMap
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import cm, colors
from astropy.time import Time

from cosipy.coordinates import Attitude
from cosipy.response import FullDetectorResponse
from cosipy.coordinates import SpacecraftFrame

class SourceSpacecraft(object):
    
    def __init__(self, src_name, src_coord, out_name = None, instrument = "COSI", frame = "galactic"):
        
        """
        Defines the source information you want to work with.
        
        Parameters
        ----------
        src_name : str; the name of the target source
        src_coord: SkyCoord object; the coordiate of the targe source
        out_name : str; the name of the outputs (saved numpy arrays, fits files, ...). 
                   If not defined, it will be synced with the src_name
        instrument : str; default COSI
        frame : str; default galactic
        """
        
        self.src_name = src_name
        self.src_coord = src_coord
        self.instrument = instrument
        self.frame = frame 
        if out_name is None:
            self.out_name = self.src_name
            
        # check inputs
        if type(self.src_coord) != SkyCoord:
            raise ValueError("The targe source coordiate must be a SkyCoord object")
    
    def __str_or_array(self, var):
        
        """
        Takes the numpy array object or the directtory to the save nunmy array (.npy).
        
        Parameters
        ----------
        var: str or numpy array object
        
        Returns
        -------
        Numpy array object
        """
        
        if isinstance(var, str):
            loaded = np.load(var)
        elif isinstance(var, Time):
            loaded = var
        return loaded
    
    def sc_frame(self, x_pointings = None, y_pointings = None, z_pointings = None):
        
        """
        Converts the x, y and z pointings of the spacescraft axes to the path of the source 
        in the spacecraft frame
        
        Specify the pointings of at least two axes.
        
        Parameters
        ----------
        x_pointings : SkyCoord object; the pointings of the x axis of the spacecraft.
        y_pointings : SkyCoord object; the pointings of the y axis of the spacecraft.
        z_pointings : SkyCoord object; the pointings of the z axis of the spacecraft.
        frame : :py:class:`astropy.coordinates.BaseCoordinateFrame`
                Inertial reference frame
                
        Returns
        -------
        SkyCoord object
        """
        
        self.x_pointings = x_pointings
        self.y_pointings = y_pointings
        self.z_pointings = z_pointings
        
        list_ = [self.x_pointings, self.y_pointings, self.z_pointings]
        coord_list_of_path = [x for x in list_ if x!=None]  # check how many pointings the user input
        
        # Check if the user input pointings from at least two axes
        if len(coord_list_of_path) <= 1:
            raise ValueError("You must input pointings of at least two axes")
        
        # Check if the inputs are SkyCoord objects
        for i in coord_list_of_path:
            if type(i) != SkyCoord:
                raise ValueError("The coordiates must be a SkyCoord object")
            
        print("Now converting to the Spacecraft frame...")
        
        self.attitude = Attitude.from_axes(x=self.x_pointings, 
                                           y=self.y_pointings, 
                                           z=self.z_pointings, 
                                           frame = self.frame)
        
        self.src_path_cartesian = SkyCoord(np.dot(self.attitude.rot.inv().as_matrix(), self.src_coord.cartesian.xyz.value),
                          representation_type = 'cartesian',
                          frame = SpacecraftFrame())
        
        # The conversion above is in Cartesian frame, so we have to convert them to the spherical one.
        self.src_path_spherical = cartesian_to_spherical(self.src_path_cartesian.x, 
                                                         self.src_path_cartesian.y, 
                                                         self.src_path_cartesian.z)
        print(f"Conversion completed!")
        
        # new generate the numpy array of l and b to save to a npy file
        l = np.array(self.src_path_spherical[2].deg)  # note that 0 is Quanty, 1 is latitude and 2 is longitude and they are in rad not deg
        b = np.array(self.src_path_spherical[1].deg)
        self.src_path_lb = np.stack((l,b), axis=-1)
        np.save(self.out_name+"_source_path_in_SC_frame", self.src_path_lb)
        
        # convert to SkyCoord objects to get the output object of this method
        self.src_path_skycoord = SkyCoord(self.src_path_lb[:,0], self.src_path_lb[:,1], unit = "deg", frame = SpacecraftFrame())
        
        return self.src_path_skycoord
    
    
    def get_dwell_map(self, response, dts, src_path = None):
        
        """
        Generates the dwell time map for the source
        
        Parameters
        ----------
        response : .h5 file; the response for the observation,
        dts : numpy array or str; the elapsed time for each pointing. It must has the same size as the pointings
        src_path : SkyCoord object; the movement of source in the detector frame.
        
        Returns
        -------
        dwell time map
        """
        
        self.response_file = response
        
        self.dts = self.__str_or_array(dts)
        
        if src_path != None:
            path = src_path
            if type(path) != SkyCoord:
                raise TypeError("The coordiates of the source movement in the Spacecraft frame must be a SkyCoord object")
        else:
            path = self.src_path_skycoord
            
        if path.shape[0] != self.dts.shape[0]:
            raise ValueError("The dimension of the dts and source coordinates are not the same. Please check your inputs.")
        
        with FullDetectorResponse.open(self.response_file) as response:
            self.dwell_map = HealpixMap(base = response,
                                   unit = u.s,
                                   coordsys = SpacecraftFrame())
            
            if len(self.dts) != 1:
                print("Using true dts")
                for coord in path:
                    pixels, weights = self.dwell_map.get_interp_weights(coord)
                    
                    for p, w, dt in zip(pixels, weights, self.dts):
                        self.dwell_map[p] += w*(dt.unix*u.s)
        
        self.dwell_map.write_map(self.out_name + "_DwellMap.fits", overwrite=True)
        
        return self.dwell_map
        
    def get_psr_rsp(self, response = None, dwell_map = None, dts = None):
        
        """
        Generates the point source response based on the response file and dwell time map.
        dts is used to find the exposure time for this observation.
        
        Parameters
        ----------
        response : .h5 file; the response for the observation
        dwell_map : fits file; the time dwell map for the source, you can load saved dwell time map 
                    using this parameterif you've saved it before
        dts : numpy array or the file path; the elapsed time for each pointing. It must has the same 
              size as the pointings. If you have saved this array, you can load it using this parameter
        
        Returns
        -------
        Ei_edges : numpy array; the edges of the incident energy
        Ei_lo : numpy array; the lower edges of the incident energy
        Ei_hi : numpy array; the upper edges of the incident energy
        Em_edges : numpy array; the edges of the measured energy
        Em_lo : numpy array; the lower edges of the measured energy
        Em_hi : numpy array; the upper edges of the measured energy
        areas " numpy array; the effective area of each energy bin
        matrix : numpy.array; the energy dispersion matrix
        """
        
        if response != None:
            self.response_file = response

        if dwell_map is not None:
            pass # will read the saved dwell_map
        
        if dts != None:
            self.dts = self.__str_or_array(dts)
        
        with FullDetectorResponse.open(self.response_file) as response:
            
            # get point source response
            self.psr = response.get_point_source_response(self.dwell_map)
            
            self.Ei_edges = np.array(response.axes['Ei'].edges)
            self.Ei_lo = np.float32(self.Ei_edges[:-1])  # use float32 to match the requirement of the data type
            self.Ei_hi = np.float32(self.Ei_edges[1:])
            
            self.Em_edges = np.array(response.axes['Em'].edges)
            self.Em_lo = np.float32(self.Em_edges[:-1])  
            self.Em_hi = np.float32(self.Em_edges[1:])
            
         # get the effective area and matrix
        print("Getting the effective area ...")
        self.areas = np.float32(np.array(self.psr.project('Ei').to_dense().contents))/self.dts.unix.sum()
        spectral_response = np.float32(np.array(self.psr.project(['Ei','Em']).to_dense().contents))
        self.matrix = np.float32(np.zeros((self.Ei_lo.size,self.Em_lo.size))) # initate the matrix
        
        print("Getting the energy redistribution matrix ...")
        for i in np.arange(self.Ei_lo.size):
            new_raw = spectral_response[i,:]/spectral_response[i,:].sum()
            self.matrix[i,:] = new_raw
        self.matrix = self.matrix.T
        
        return self.Ei_edges, self.Ei_lo, self.Ei_hi, self.Em_edges, self.Em_lo, self.Em_hi, self.areas, self.matrix
        
        
    def get_arf(self):
        
        """
        Converts the point source response to an arf file that can be read by XSPEC
        
        Parameters
        ----------
        
        Returns
        -------
        None
        """
        
        # blow write the arf file
        copyright_string="  FITS (Flexible Image Transport System) format is defined in 'Astronomy and Astrophysics', volume 376, page 359; bibcode: 2001A&A...376..359H "

        ## Create PrimaryHDU
        primaryhdu = fits.PrimaryHDU() # create an empty primary HDU
        primaryhdu.header["BITPIX"] = -32 # since it's an empty HDU, I can just change the data type by resetting the BIPTIX value
        primaryhdu.header["COMMENT"] = copyright_string # add comments
        primaryhdu.header # print headers and their values

        col1_energ_lo = fits.Column(name="ENERG_LO", format="E",unit = "keV", array=self.Em_lo)
        col2_energ_hi = fits.Column(name="ENERG_HI", format="E",unit = "keV", array=self.Em_hi)
        col3_specresp = fits.Column(name="SPECRESP", format="E",unit = "cm**2", array=self.areas)
        cols = fits.ColDefs([col1_energ_lo, col2_energ_hi, col3_specresp]) # create a ColDefs (column-definitions) object for all columns
        specresp_bintablehdu = fits.BinTableHDU.from_columns(cols) # create a binary table HDU object

        specresp_bintablehdu.header.comments["TTYPE1"] =  "label for field   1"
        specresp_bintablehdu.header.comments["TFORM1"] =  "data format of field: 4-byte REAL"
        specresp_bintablehdu.header.comments["TUNIT1"] =  "physical unit of field"
        specresp_bintablehdu.header.comments["TTYPE2"] =  "label for field   2"
        specresp_bintablehdu.header.comments["TFORM2"] =  "data format of field: 4-byte REAL"
        specresp_bintablehdu.header.comments["TUNIT2"] =  "physical unit of field"
        specresp_bintablehdu.header.comments["TTYPE3"] =  "label for field   3"
        specresp_bintablehdu.header.comments["TFORM3"] =  "data format of field: 4-byte REAL"
        specresp_bintablehdu.header.comments["TUNIT3"] =  "physical unit of field"

        specresp_bintablehdu.header["EXTNAME"] = ("SPECRESP","name of this binary table extension")
        specresp_bintablehdu.header["TELESCOP"] = ("COSI","mission/satellite name")
        specresp_bintablehdu.header["INSTRUME"] = ("COSI","instrument/detector name")
        specresp_bintablehdu.header["FILTER"] = ("NONE","filter in use")
        specresp_bintablehdu.header["HDUCLAS1"] = ("RESPONSE","dataset relates to spectral response")
        specresp_bintablehdu.header["HDUCLAS2"] = ("SPECRESP","extension contains an ARF")
        specresp_bintablehdu.header["HDUVERS"] = ("1.1.0","version of format")

        new_arfhdus = fits.HDUList([primaryhdu, specresp_bintablehdu])
        new_arfhdus.writeto(f'{self.out_name}.arf', overwrite=True)
        
        return
    
    def get_rmf(self):
        
        """
        Converts the point source response to an rmf file that can be read by XSPEC
        
        Parameters
        ----------
        
        Returns
        -------
        None
        """
        
        # blow write the arf file
        copyright_string="  FITS (Flexible Image Transport System) format is defined in 'Astronomy and Astrophysics', volume 376, page 359; bibcode: 2001A&A...376..359H "

        ## Create PrimaryHDU
        primaryhdu = fits.PrimaryHDU() # create an empty primary HDU
        primaryhdu.header["BITPIX"] = -32 # since it's an empty HDU, I can just change the data type by resetting the BIPTIX value
        primaryhdu.header["COMMENT"] = copyright_string # add comments
        primaryhdu.header # print headers and their values

        ## Create binary table HDU for MATRIX
        ### prepare colums
        energ_lo = []
        energ_hi = []
        n_grp = []
        f_chan = []
        n_chan = []
        matrix = []
        for i in np.arange(len(self.Ei_lo)):
            energ_lo_temp = np.float32(self.Em_lo[i])
            energ_hi_temp = np.float32(self.Ei_hi[i])

            if self.matrix[:,i].sum() != 0:
                nz_matrix_idx = np.nonzero(self.matrix[:,i])[0] # non-zero index for the matrix
                subsets = np.split(nz_matrix_idx, np.where(np.diff(nz_matrix_idx) != 1)[0]+1)
                n_grp_temp = np.int16(len(subsets))
                f_chan_temp = []
                n_chan_temp = []
                matrix_temp = []
                for m in np.arange(n_grp_temp):
                    f_chan_temp += [subsets[m][0]]
                    n_chan_temp += [len(subsets[m])]
                for m in nz_matrix_idx:
                    matrix_temp += [self.matrix[:,i][m]]
                f_chan_temp = np.int16(np.array(f_chan_temp))
                n_chan_temp = np.int16(np.array(n_chan_temp))
                matrix_temp = np.float32(np.array(matrix_temp))
            else:
                n_grp_temp = np.int16(0)
                f_chan_temp = np.int16(np.array([0]))
                n_chan_temp = np.int16(np.array([0]))
                matrix_temp = np.float32(np.array([0]))

            energ_lo.append(energ_lo_temp)
            energ_hi.append(energ_hi_temp)
            n_grp.append(n_grp_temp)
            f_chan.append(f_chan_temp)
            n_chan.append(n_chan_temp)
            matrix.append(matrix_temp)

        col1_energ_lo = fits.Column(name="ENERG_LO", format="E",unit = "keV", array=energ_lo)
        col2_energ_hi = fits.Column(name="ENERG_HI", format="E",unit = "keV", array=energ_hi)
        col3_n_grp = fits.Column(name="N_GRP", format="I", array=n_grp)
        col4_f_chan = fits.Column(name="F_CHAN", format="PI(54)", array=f_chan)
        col5_n_chan = fits.Column(name="N_CHAN", format="PI(54)", array=n_chan)
        col6_n_chan = fits.Column(name="MATRIX", format="PE(161)", array=matrix)
        cols = fits.ColDefs([col1_energ_lo, col2_energ_hi, col3_n_grp, col4_f_chan, col5_n_chan, col6_n_chan]) # create a ColDefs (column-definitions) object for all columns
        matrix_bintablehdu = fits.BinTableHDU.from_columns(cols) # create a binary table HDU object

        matrix_bintablehdu.header.comments["TTYPE1"] = "label for field   1 "
        matrix_bintablehdu.header.comments["TFORM1"] = "data format of field: 4-byte REAL"
        matrix_bintablehdu.header.comments["TUNIT1"] = "physical unit of field"
        matrix_bintablehdu.header.comments["TTYPE2"] = "label for field   2"
        matrix_bintablehdu.header.comments["TFORM2"] = "data format of field: 4-byte REAL"
        matrix_bintablehdu.header.comments["TUNIT2"] = "physical unit of field"
        matrix_bintablehdu.header.comments["TTYPE3"] = "label for field   3 "
        matrix_bintablehdu.header.comments["TFORM3"] = "data format of field: 2-byte INTEGER"
        matrix_bintablehdu.header.comments["TTYPE4"] = "label for field   4"
        matrix_bintablehdu.header.comments["TFORM4"] = "data format of field: variable length array"
        matrix_bintablehdu.header.comments["TTYPE5"] = "label for field   5"
        matrix_bintablehdu.header.comments["TFORM5"] = "data format of field: variable length array"
        matrix_bintablehdu.header.comments["TTYPE6"] = "label for field   6"
        matrix_bintablehdu.header.comments["TFORM6"] = "data format of field: variable length array"

        matrix_bintablehdu.header["EXTNAME"] = ("MATRIX","name of this binary table extension")
        matrix_bintablehdu.header["TELESCOP"] = ("COSI","mission/satellite name")
        matrix_bintablehdu.header["INSTRUME"] = ("COSI","instrument/detector name")
        matrix_bintablehdu.header["FILTER"] = ("NONE","filter in use")
        matrix_bintablehdu.header["CHANTYPE"] = ("PI","total number of detector channels")
        matrix_bintablehdu.header["DETCHANS"] = (len(self.Em_lo),"total number of detector channels")
        matrix_bintablehdu.header["HDUCLASS"] = ("OGIP","format conforms to OGIP standard")
        matrix_bintablehdu.header["HDUCLAS1"] = ("RESPONSE","dataset relates to spectral response")
        matrix_bintablehdu.header["HDUCLAS2"] = ("RSP_MATRIX","dataset is a spectral response matrix")
        matrix_bintablehdu.header["HDUVERS"] = ("1.3.0","version of format")
        matrix_bintablehdu.header["TLMIN4"] = (0,"minimum value legally allowed in column 4")
        
        ## Create binary table HDU for EBOUNDS
        channels = np.int16(np.arange(len(self.Em_lo)))
        e_min = np.float32(self.Em_lo)
        e_max = np.float32(self.Em_hi)

        col1_channels = fits.Column(name="CHANNEL", format="I", array=channels)
        col2_e_min = fits.Column(name="E_MIN", format="E",unit="keV", array=e_min)
        col3_e_max = fits.Column(name="E_MAX", format="E",unit="keV", array=e_max)
        cols = fits.ColDefs([col1_channels, col2_e_min, col3_e_max])
        ebounds_bintablehdu = fits.BinTableHDU.from_columns(cols)

        ebounds_bintablehdu.header.comments["TTYPE1"] = "label for field   1"
        ebounds_bintablehdu.header.comments["TFORM1"] = "data format of field: 2-byte INTEGER"
        ebounds_bintablehdu.header.comments["TTYPE2"] = "label for field   2"
        ebounds_bintablehdu.header.comments["TFORM2"] = "data format of field: 4-byte REAL"
        ebounds_bintablehdu.header.comments["TUNIT2"] = "physical unit of field"
        ebounds_bintablehdu.header.comments["TTYPE3"] = "label for field   3"
        ebounds_bintablehdu.header.comments["TFORM3"] = "data format of field: 4-byte REAL"
        ebounds_bintablehdu.header.comments["TUNIT3"] = "physical unit of field"

        ebounds_bintablehdu.header["EXTNAME"] = ("EBOUNDS","name of this binary table extension")
        ebounds_bintablehdu.header["TELESCOP"] = ("COSI","mission/satellite")
        ebounds_bintablehdu.header["INSTRUME"] = ("COSI","nstrument/detector name")
        ebounds_bintablehdu.header["FILTER"] = ("NONE","filter in use")
        ebounds_bintablehdu.header["CHANTYPE"] = ("PI","channel type (PHA or PI)")
        ebounds_bintablehdu.header["DETCHANS"] = (len(self.Em_lo),"total number of detector channels")
        ebounds_bintablehdu.header["HDUCLASS"] = ("OGIP","format conforms to OGIP standard")
        ebounds_bintablehdu.header["HDUCLAS1"] = ("RESPONSE","dataset relates to spectral response")
        ebounds_bintablehdu.header["HDUCLAS2"] = ("EBOUNDS","dataset is a spectral response matrix")
        ebounds_bintablehdu.header["HDUVERS"] = ("1.2.0","version of format")
        
        new_rmfhdus = fits.HDUList([primaryhdu, matrix_bintablehdu,ebounds_bintablehdu])
        new_rmfhdus.writeto(f'{self.out_name}.rmf', overwrite=True)
        
        return
    
    def get_pha(self, src_counts, errors, rmf_file = None, arf_file = None, bkg_file = "None", exposure_time = None, dts = None, telescope="COSI", instrument="COSI"):
        
        """
        Generate the pha file that can be read by XSPEC. This file stores the counts info of the source
        
        Parameters
        ----------
        src_counts : np.array; the counts in each energy band. If you have src_counts with unit counts/kev/s, you must
                     convert it to counts by multiplying it with exposure time and the energy band width
        errors : np.array; the error for counts. It has the same unit requirement as src_counts
        rmf_file : str; the rmf file name
        arf_file : str; the arf file name
        bkg_file : str; the background file name. If the src_counts is source counts only, you don't need to edit this parameter
        exposure_time : number; the exposure time for this source observation
        dts : numpy array or str; it's used to calculate the exposure time. It has the same effect as exposure_time. If both 
              exposure_time and dts are given, dts will write over the exposure_time
        telescope : str; the name of the telecope. Default is COSI.
        instrument : str; the name of the instrument. Default is COSI.
        
        Returns
        -------
        None
        """
        
        self.src_counts = src_counts
        self.errors = errors
        self.bkg_file = bkg_file
        
        if rmf_file != None:
            self.rmf_file = rmf_file
        else:
            self.rmf_file = f'{self.out_name}.rmf'
            
        if arf_file != None:
            self.arf_file = arf_file
        else:
            self.arf_file = f'{self.out_name}.arf'

        if exposure_time != None:
            self.exposure_time = exposure_time
        if dts != None:
            self.dts = self.__str_or_array(dts)
            self.exposure_time = self.dts.sum()
        self.telescope = telescope
        self.instrument = instrument
        self.channel_number = len(self.src_counts)
        
        # define other hardcoded inputs
        copyright_string="  FITS (Flexible Image Transport System) format is defined in 'Astronomy and Astrophysics', volume 376, page 359; bibcode: 2001A&A...376..359H "
        channels = np.arange(self.channel_number)
        
        # Create PrimaryHDU
        primaryhdu = fits.PrimaryHDU() # create an empty primary HDU
        primaryhdu.header["BITPIX"] = -32 # since it's an empty HDU, I can just change the data type by resetting the BIPTIX value
        primaryhdu.header["COMMENT"] = copyright_string # add comments
        primaryhdu.header["TELESCOP"] = telescope # add telescope keyword valie
        primaryhdu.header["INSTRUME"] = instrument # add instrument keyword valie
        primaryhdu.header # print headers and their values
        
        # Create binary table HDU
        a1 = np.array(channels,dtype="int32") # I guess I need to convert the dtype to match the format J
        a2 = np.array(self.src_counts,dtype="int64")  # int32 is not enough for counts
        a3 = np.array(self.errors,dtype="int64") # int32 is not enough for errors
        col1 = fits.Column(name="CHANNEL", format="J", array=a1)
        col2 = fits.Column(name="COUNTS", format="K", array=a2,unit="count")
        col3 = fits.Column(name="STAT_ERR", format="K", array=a3,unit="count")
        cols = fits.ColDefs([col1, col2, col3]) # create a ColDefs (column-definitions) object for all columns
        bintablehdu = fits.BinTableHDU.from_columns(cols) # create a binary table HDU object

        #add other BinTableHDU hear keywords,their values, and comments
        bintablehdu.header.comments["TTYPE1"] = "label for field 1"
        bintablehdu.header.comments["TFORM1"] = "data format of field: 32-bit integer"
        bintablehdu.header.comments["TTYPE2"] = "label for field 2"
        bintablehdu.header.comments["TFORM2"] = "data format of field: 32-bit integer"
        bintablehdu.header.comments["TUNIT2"] = "physical unit of field 2"


        bintablehdu.header["EXTNAME"] = ("SPECTRUM","name of this binary table extension")
        bintablehdu.header["TELESCOP"] = (self.telescope,"telescope/mission name")
        bintablehdu.header["INSTRUME"] = (self.instrument,"instrument/detector name")
        bintablehdu.header["FILTER"] = ("NONE","filter type if any")
        bintablehdu.header["EXPOSURE"] = (self.exposure_time,"integration time in seconds")
        bintablehdu.header["BACKFILE"] = (self.bkg_file,"background filename")
        bintablehdu.header["BACKSCAL"] = (1,"background scaling factor")
        bintablehdu.header["CORRFILE"] = ("NONE","associated correction filename")
        bintablehdu.header["CORRSCAL"] = (1,"correction file scaling factor")
        bintablehdu.header["CORRSCAL"] = (1,"correction file scaling factor")
        bintablehdu.header["RESPFILE"] = (self.rmf_file,"associated rmf filename")
        bintablehdu.header["ANCRFILE"] = (self.arf_file,"associated arf filename")
        bintablehdu.header["AREASCAL"] = (1,"area scaling factor")
        bintablehdu.header["STAT_ERR"] = (0,"statistical error specified if any")
        bintablehdu.header["SYS_ERR"] = (0,"systematic error specified if any")
        bintablehdu.header["GROUPING"] = (0,"grouping of the data has been defined if any")
        bintablehdu.header["QUALITY"] = (0,"data quality information specified")
        bintablehdu.header["HDUCLASS"] = ("OGIP","format conforms to OGIP standard")
        bintablehdu.header["HDUCLAS1"] = ("SPECTRUM","PHA dataset")
        bintablehdu.header["HDUVERS"] = ("1.2.1","version of format")
        bintablehdu.header["POISSERR"] = (False,"Poissonian errors to be assumed, T as True")
        bintablehdu.header["CHANTYPE"] = ("PI","channel type (PHA or PI)")
        bintablehdu.header["DETCHANS"] = (self.channel_number,"total number of detector channels")
        
        new_phahdus = fits.HDUList([primaryhdu, bintablehdu])
        new_phahdus.writeto(f'{self.out_name}.pha', overwrite=True)
        
        return
    

    def plot_arf(self, file_name = None, save_name = None, dpi = 300):
        
        """
        Read the arf fits file, plot and save it.
        
        Parameters
        ----------
        file_name: str; the directory if the arf fits file. 
        save_name: str; the name of the saved image of effective area
        dpi: int; the dpi of the saved image
        
        Returns
        -------
        None
        """
        
        if file_name != None:
            self.file_name = file_name
        else:
            self.file_name = f'{self.out_name}.arf'
            
        if save_name != None:
            self.save_name = save_name
        else:
            self.save_name = self.out_name
            
        self.dpi = dpi
            
        self.arf = fits.open(self.file_name) # read file
        
        # SPECRESP HDU
        self.specresp_hdu = self.arf["SPECRESP"]
        
        self.areas = np.array(self.specresp_hdu.data["SPECRESP"])
        self.Em_lo = np.array(self.specresp_hdu.data["ENERG_LO"])
        self.Em_hi = np.array(self.specresp_hdu.data["ENERG_HI"])
        
        E_center = (self.Em_lo+self.Em_hi)/2
        E_edges = np.append(self.Em_lo,self.Em_hi[-1])
        
        fig, ax = plt.subplots()
        ax.hist(E_center,E_edges,weights=self.areas,histtype='step')
        
        ax.set_title("Effective area")
        ax.set_xlabel("Energy[$keV$]")
        ax.set_ylabel(r"Effective area [$cm^2$]")
        ax.set_xscale("log")
        fig.savefig(f"Effective area for {self.save_name}.png", bbox_inches = "tight", pad_inches=0.1, dpi=self.dpi)
        #fig.show()
        
        return
        
    
    def plot_rmf(self, file_name = None, save_name = None, dpi = 300):
        
        """
        Read the rmf fits file, plot and save it.
        
        Parameters
        ----------
        file_name: str; the directory if the rmf fits file. 
        save_name: str; the name of the saved image of effective area
        dpi: int; the dpi of the saved image
        
        Returns
        -------
        None
        """
        
        if file_name != None:
            self.file_name = file_name
        else:
            self.file_name = f'{self.out_name}.rmf'
            
        if save_name != None:
            self.save_name = save_name
        else:
            self.save_name = self.out_name
            
        self.dpi = dpi
        
        # Read rmf file
        self.rmf = fits.open(self.file_name) # read file
        
        # Read the ENOUNDS information
        ebounds_ext = self.rmf["EBOUNDS"]
        channel_low = ebounds_ext.data["E_MIN"] # energy bin lower edges for channels (channels are just incident energy bins)
        channel_high = ebounds_ext.data["E_MAX"] # energy bin higher edges for channels (channels are just incident energy bins)
        
        # Read the MATRIX extension
        matrix_ext = self.rmf['MATRIX']
        #print(repr(matrix_hdu.header[:60]))
        energy_low = matrix_ext.data["ENERG_LO"] # energy bin lower edges for measured energies
        energy_high = matrix_ext.data["ENERG_HI"] # energy bin higher edges for measured energies
        data = matrix_ext.data
        
        # Create a 2-d numpy array and store probability data into the redistribution matrix
        rmf_matrix = np.zeros((len(energy_low),len(channel_low))) # create an empty matrix
        for i in np.arange(data.shape[0]): # i is the measured energy index, examine the matrix_ext.data rows by rows
            if data[i][5].sum() == 0: # if the sum of probabilities is zero, then skip since there is no data at all
                pass
            else:
                #measured_energy_index = np.argwhere(energy_low == data[157][0])[0][0]
                f_chan = data[i][3] # get the starting channel of each subsets
                n_chann = data[i][4] # get the number of channels in each subsets
                matrix = data[i][5] # get the probabilities of this row (incident energy)
                indices = []
                for k in f_chan:
                    channels = 0
                    channels = np.arange(k,k + n_chann[np.argwhere(f_chan == k)]).tolist() # generate the cha
                    indices += channels # fappend the channels togeter
                indices = np.array(indices)
                for m in indices:
                    rmf_matrix[i][m] = matrix[np.argwhere(indices == m)[0][0]] # write the probabilities into the empty matrix
                    
                    
        # plot the redistribution matrix
        xcenter = np.divide(energy_low+energy_high,2)
        x_center_coords = np.repeat(xcenter, 10)
        y_center_coords = np.tile(xcenter, 10)
        energy_all_edges = np.append(energy_low,energy_high[-1])
        #bin_edges = np.array([incident_energy_bins,incident_energy_bins]) # doesn't work
        bin_edges = np.vstack((energy_all_edges, energy_all_edges))
        #print(bin_edges)

        self.probability = []
        for i in np.arange(10):
            for j in np.arange(10):
                self.probability.append(rmf_matrix[i][j])
        #print(type(probability))

        plt.hist2d(x=x_center_coords,y=y_center_coords,weights=self.probability,bins=bin_edges, norm=LogNorm())
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("Incident energy [$keV$]")
        plt.ylabel("Measured energy [$keV$]")
        plt.title("Redistribution matrix")
        #plt.xlim([70,10000])
        #plt.ylim([70,10000])
        plt.colorbar(norm=LogNorm())
        plt.savefig(f"Redistribution matrix for {self.save_name}.png", bbox_inches = "tight", pad_inches=0.1, dpi=300)
        #plt.show()
        
        return
