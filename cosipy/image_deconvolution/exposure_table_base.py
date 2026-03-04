import logging
logger = logging.getLogger(__name__)

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import healpy as hp
from astropy.io import fits
import astropy.units as u
from histpy import Histogram, HealpixAxis, Axis, Axes

class ExposureTableBase(pd.DataFrame, ABC):
    """
    A class to analyze exposure time per each spacecraft attitude
    
    Default table columns are:
    - binning_index_name (e.g., scatt_binning_index, time_binning_index): int
    - zpointing: np.array of [l, b] in degrees. Array of z-pointings assigned to each scatt bin.
    - xpointing: np.array of [l, b] in degrees. Array of x-pointings assigned to each scatt bin.
    - zpointing_averaged: [l, b] in degrees. Averaged z-pointing in each scatt bin.
    - xpointing_averaged: [l, b] in degrees. Averaged x-pointing in each scatt bin.
    - earth_zenith: np.array of [l, b] in degrees. Array of earth zenith assigned to each scatt bin.
    - altitude: float in km. The satellite altitude assigned to each scatt bin.
    - livetime: np.array of float in second. live times for pointings assigned to each scatt bin.
    - total_livetime: float in second. total livetime for each scatt bin.
    - num_pointings: number of pointings assigned to each scatt bin.

    Attributes
    ----------
    df : :py:class:`pd.DataFrame`
        pandas dataframe with the above columns
    nside : int
        Healpix NSIDE parameter.
    scheme : str, default 'ring'
        Healpix scheme. Either 'ring', 'nested'. 
    """
    binning_index_name = None
    binning_method = None # used for histogram's axis label, e.g., 'Time' or 'ScAtt'
    additional_column_scaler = None # list of (column_name, format, unit)
    additional_column_array  = None # list of (column_name, format, unit)
    required_init_params = []  # New: list of required __init__ parameters for subclass

    def __init__(self, df, **kwargs):

        super().__init__(pd.DataFrame(df))

        # Store additional parameters defined in required_init_params
        for param in self.required_init_params:
            if param in kwargs:
                setattr(self, param, kwargs[param])

    def __eq__(self, other):

        # default columns
        for name in [self.binning_index_name, 'total_livetime', 'num_pointings']:
            if not np.all(self[name] == other[name]):
                return False
        
        for name in ['livetime', 'zpointing', 'xpointing', 'zpointing_averaged', 'xpointing_averaged', 'earth_zenith', 'altitude']:
            for self_, other_ in zip(self[name], other[name]):
                if not np.all(self_ == other_):
                    return False

        # additional columns
        if self.additional_column_scaler is not None:
            for (name, format, unit) in self.additional_column_scaler:
                if not np.all(self[name] == other[name]):
                    return False

        if self.additional_column_array is not None:
            for (name, format, unit) in self.additional_column_array:
                for self_, other_ in zip(self[name], other[name]):
                    if not np.all(self_ == other_):
                        return False

        for param in self.required_init_params:
            if not getattr(self, param) == getattr(other, param):
                return False

        return True

    @classmethod
    @abstractmethod
    def from_orientation(cls, orientation, **kwargs):
        """
        Produce exposure table from orientation.

        Returns
        -------
        :py:class:`cosipy.image_deconvolution.ExposureTableBase`
        """
        raise NotImplementedError

    @classmethod
    def from_fits(cls, filename):
        """
        Read exposure table from a fits file.

        Parameters
        ----------
        filename : str
            Path to file
        
        Returns
        -------
        :py:class:`cosipy.image_deconvolution.ExposureTableBase`
        """

        infile = fits.open(filename)
        hdu = infile[1]
    
        if hdu.name.upper() != "EXPOSURETABLE":
            logger.error("cannot find EXPOSURETABLE")
            infile.close()
            return
    
        # Convert to native byte order to avoid endianness issues
        indices_binning = np.asarray(hdu.data[cls.binning_index_name], dtype=np.int64)

        zpointings = [ [ [l, b] for (l, b) in zip(z_l, z_b) ] 
                      for (z_l, z_b) in zip(hdu.data['zpointing_l'], hdu.data['zpointing_b']) ]
        zpointings = [ np.asarray(_, dtype=np.float64) for _ in zpointings]

        xpointings = [ [ [l, b] for (l, b) in zip(x_l, x_b) ] 
                      for (x_l, x_b) in zip(hdu.data['xpointing_l'], hdu.data['xpointing_b']) ]
        xpointings = [ np.asarray(_, dtype=np.float64) for _ in xpointings]

        earth_zeniths = [ [ [l, b] for (l, b) in zip(x_l, x_b) ] 
                         for (x_l, x_b) in zip(hdu.data['earth_zenith_l'], hdu.data['earth_zenith_b']) ]
        earth_zeniths = [ np.asarray(_, dtype=np.float64) for _ in earth_zeniths]

        zpointings_averaged = [ np.asarray([z_ave_l, z_ave_b], dtype=np.float64) 
                               for (z_ave_l, z_ave_b) in zip(hdu.data['zpointing_averaged_l'], 
                                                              hdu.data['zpointing_averaged_b']) ]

        xpointings_averaged = [ np.asarray([x_ave_l, x_ave_b], dtype=np.float64) 
                               for (x_ave_l, x_ave_b) in zip(hdu.data['xpointing_averaged_l'], 
                                                              hdu.data['xpointing_averaged_b']) ]

        # Handle variable-length arrays (altitude and livetime)
        altitudes = [ np.asarray(alt, dtype=np.float64) for alt in hdu.data['altitude'] ]

        livetimes = [ np.asarray(lt, dtype=np.float64) for lt in hdu.data['livetime'] ]

        total_livetimes = np.asarray(hdu.data['total_livetime'], dtype=np.float64)

        num_pointings = np.asarray(hdu.data['num_pointings'], dtype=np.int64)
        
        # data dictionary
        data = {cls.binning_index_name: indices_binning,
                'zpointing': zpointings,
                'xpointing': xpointings,
                'zpointing_averaged': zpointings_averaged,
                'xpointing_averaged': xpointings_averaged,
                'earth_zenith': earth_zeniths,
                'altitude': altitudes, 
                'livetime': livetimes,
                'total_livetime': total_livetimes,
                'num_pointings': num_pointings}

        # adding additional columns
        if cls.additional_column_scaler is not None:
            for (name, format_str, unit) in cls.additional_column_scaler:
                # Determine dtype based on FITS format
                # See: https://fits.gsfc.nasa.gov/standard40/fits_standard40aa-le.pdf
                if format_str == 'B':  # Unsigned byte
                    this_data = np.asarray(hdu.data[name], dtype=np.uint8)
                elif format_str == 'I':  # 16-bit integer
                    this_data = np.asarray(hdu.data[name], dtype=np.int16)
                elif format_str == 'J':  # 32-bit integer
                    this_data = np.asarray(hdu.data[name], dtype=np.int32)
                elif format_str == 'K':  # 64-bit integer
                    this_data = np.asarray(hdu.data[name], dtype=np.int64)
                elif format_str == 'E':  # Single precision float
                    this_data = np.asarray(hdu.data[name], dtype=np.float32)
                elif format_str == 'D':  # Double precision float
                    this_data = np.asarray(hdu.data[name], dtype=np.float64)
                elif format_str == 'L':  # Logical (boolean)
                    this_data = np.asarray(hdu.data[name], dtype=bool)
                elif format_str.endswith('A'):  # String (e.g., '10A', '20A')
                    # Convert bytes to string if necessary
                    this_data = np.asarray(hdu.data[name], dtype=str)
                else:
                    # Default: keep original dtype but convert to native byte order
                    this_data = np.asarray(hdu.data[name])
                data[name] = this_data

        if cls.additional_column_array is not None:
            for (name, format_str, unit) in cls.additional_column_array:
                # Variable-length arrays - process each element
                if format_str.startswith('P'):  # Variable-length array format
                    # Determine base dtype from format (after 'P')
                    base_format = format_str[1] if len(format_str) > 1 else 'D'
                    
                    if base_format == 'B':  # Unsigned byte
                        this_data = [ np.asarray(elem, dtype=np.uint8) for elem in hdu.data[name] ]
                    elif base_format == 'I':  # 16-bit integer
                        this_data = [ np.asarray(elem, dtype=np.int16) for elem in hdu.data[name] ]
                    elif base_format == 'J':  # 32-bit integer
                        this_data = [ np.asarray(elem, dtype=np.int32) for elem in hdu.data[name] ]
                    elif base_format == 'K':  # 64-bit integer
                        this_data = [ np.asarray(elem, dtype=np.int64) for elem in hdu.data[name] ]
                    elif base_format == 'E':  # Single precision float
                        this_data = [ np.asarray(elem, dtype=np.float32) for elem in hdu.data[name] ]
                    elif base_format == 'D':  # Double precision float
                        this_data = [ np.asarray(elem, dtype=np.float64) for elem in hdu.data[name] ]
                    elif base_format == 'L':  # Logical (boolean)
                        this_data = [ np.asarray(elem, dtype=bool) for elem in hdu.data[name] ]
                    else:
                        this_data = [ np.asarray(elem) for elem in hdu.data[name] ]
                else:
                    # Fixed-length arrays - default to float64
                    this_data = np.asarray(hdu.data[name], dtype=np.float64)
                data[name] = this_data
        
        # finalize
        df = pd.DataFrame(data=data)

        # Read required_init_params from header
        init_params = {}
        for param in cls.required_init_params:
            if param.upper() in hdu.header:
                init_params[param] = hdu.header[param.upper()]

        infile.close()

        new = cls(df, **init_params)

        return new

    def save_as_fits(self, filename, overwrite = False):
        """
        Save exposure table as a fits file.

        Parameters
        ----------
        filename : str
            Path to file
        overwrite : bool, default False
        """

        # primary HDU
        primary_hdu = fits.PrimaryHDU()
    
        #exposure table
        names = [self.binning_index_name, 'total_livetime', 'num_pointings']
        formats = ['K', 'D', 'K']
        units = ['', 's', '']
        
        columns = [ fits.Column(name=names[i], array=self[names[i]].to_numpy(), format = formats[i], unit = units[i]) 
                     for i in range(len(names))]
        
        column_altitude = fits.Column(name='altitude', format='PD()', unit = 'km',
                                        array=np.array(self['altitude'].array, dtype=np.object_))
        columns.append(column_altitude)    
    
        column_livetime = fits.Column(name='livetime', format='PD()', unit = 's',
                                        array=np.array(self['livetime'].array, dtype=np.object_))
        columns.append(column_livetime)    
        
        column_zpointing_l = fits.Column(name='zpointing_l', format='PD()', unit = 'degree',
                                        array=np.array([[pointing[0] for pointing in pointings] for pointings in self['zpointing']], dtype=np.object_))
        columns.append(column_zpointing_l)    
    
        column_zpointing_b = fits.Column(name='zpointing_b', format='PD()', unit = 'degree',
                                        array=np.array([[pointing[1] for pointing in pointings] for pointings in self['zpointing']], dtype=np.object_))
        columns.append(column_zpointing_b)   
    
        column_xpointing_l = fits.Column(name='xpointing_l', format='PD()', unit = 'degree',
                                        array=np.array([[pointing[0] for pointing in pointings] for pointings in self['xpointing']], dtype=np.object_))
        columns.append(column_xpointing_l)    
    
        column_xpointing_b = fits.Column(name='xpointing_b', format='PD()', unit = 'degree',
                                        array=np.array([[pointing[1] for pointing in pointings] for pointings in self['xpointing']], dtype=np.object_))
        columns.append(column_xpointing_b)  

        column_zpointing_averaged_l = fits.Column(name='zpointing_averaged_l', format='D', unit = 'degree',
                                                  array=np.array([_[0] for _ in self['zpointing_averaged']]))
        columns.append(column_zpointing_averaged_l)    

        column_zpointing_averaged_b = fits.Column(name='zpointing_averaged_b', format='D', unit = 'degree',
                                                  array=np.array([_[1] for _ in self['zpointing_averaged']]))
        columns.append(column_zpointing_averaged_b)    

        column_xpointing_averaged_l = fits.Column(name='xpointing_averaged_l', format='D', unit = 'degree',
                                                  array=np.array([_[0] for _ in self['xpointing_averaged']]))
        columns.append(column_xpointing_averaged_l)    

        column_xpointing_averaged_b = fits.Column(name='xpointing_averaged_b', format='D', unit = 'degree',
                                                  array=np.array([_[1] for _ in self['xpointing_averaged']]))
        columns.append(column_xpointing_averaged_b)    
    
        column_earth_zenith_l = fits.Column(name='earth_zenith_l', format='PD()', unit = 'degree',
                                        array=np.array([[pointing[0] for pointing in pointings] for pointings in self['earth_zenith']], dtype=np.object_))
        columns.append(column_earth_zenith_l)    
    
        column_earth_zenith_b = fits.Column(name='earth_zenith_b', format='PD()', unit = 'degree',
                                        array=np.array([[pointing[1] for pointing in pointings] for pointings in self['earth_zenith']], dtype=np.object_))
        columns.append(column_earth_zenith_b)  

        # additional columns
        if self.additional_column_scaler is not None:
            for (name, format, unit) in self.additional_column_scaler:
                new_column = fits.Column(name=name, format=format, unit=unit, 
                                         array=self[name].to_numpy())
                columns.append(new_column)

        if self.additional_column_array is not None:
            for (name, format, unit) in self.additional_column_array:
                new_column = fits.Column(name=name, format=format, unit=unit, 
                                         array=np.array(self[name].array, dtype=np.object_))
                columns.append(new_column)
        
        # finalize
        table_hdu = fits.BinTableHDU.from_columns(columns) 
        table_hdu.name = 'exposuretable'

        # Add metadata to header
        table_hdu.header['BINMETH'] = self.binning_method
        
        # Add required_init_params to header
        for param in self.required_init_params:
            if hasattr(self, param):
                value = getattr(self, param)
                table_hdu.header[param.upper()] = value
        
        #save file    
        hdul = fits.HDUList([primary_hdu, table_hdu])    
        hdul.writeto(filename, overwrite = overwrite)

    @classmethod
    def _get_averaged_pointing(cls, pointing, livetime):
        """
        Calculate an averaged pointing from given lists of pointings and exposure time on each pointing

        Parameters
        ----------
        pointing : list of np.array
            List of pointings in degrees, e.g., [ np.array([l, b]), np.array([l, b]), ...]
        livetime : list of float
            List of livetime in seconds for each pointing, e.g, [ 1.0, 1.0, ...] 

        Returns
        -------
        :py:class:`np.array`
            Averaged pointing in degrees, as np.array([l, b])
        """
        if np.all(livetime == 0) == True:
            averaged_vector = np.sum(hp.ang2vec(pointing.T[0], pointing.T[1], lonlat = True).T, axis = (1))
            logger.warning("Livetime is all zero")
        else:
            averaged_vector = np.sum(hp.ang2vec(pointing.T[0], pointing.T[1], lonlat = True).T * livetime, axis = (1))

        averaged_vector /= np.linalg.norm(averaged_vector)

        averaged_l = hp.vec2ang(averaged_vector, lonlat = True)[0][0]
        averaged_b = hp.vec2ang(averaged_vector, lonlat = True)[1][0]
    
        averaged_pointing = np.array([averaged_l, averaged_b])

        return averaged_pointing 
    
    @abstractmethod
    def get_binned_data(self, unbinned_event, psichi_binning = 'local', sparse = False):
        """
        Create binned data from unbinned events using spacecraft attitude binning.
    
        Events are grouped by spacecraft attitude (z- and x-pointing), energy, 
        Compton scattering angle, and scatter direction.
    
        Parameters
        ----------
        unbinned_event : :py:class:`cosipy.data_io.UnbinnedData`
            Unbinned event data.
        psichi_binning : str, default 'local'
            Coordinate system for PsiChi axis: 'local' or 'galactic'.
        sparse : bool, default False
            If True, use sparse array representation.
    
        Returns
        -------
        :py:class:`histpy.Histogram`
            Binned data with axes [binning_index_name, "Em", "Phi", "PsiChi"]. 
        """
        raise NotImplementedError
