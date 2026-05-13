import logging

from astropy.time import Time

logger = logging.getLogger(__name__)

from tqdm.autonotebook import tqdm
import numpy as np
import healpy as hp
import pandas as pd
from astropy.io import fits
import astropy.units as u
from astropy.coordinates import SkyCoord
from histpy import Histogram, HealpixAxis, Axis, Axes
from scoords import SpacecraftFrame

from cosipy.spacecraftfile import SpacecraftAxisMap

from .exposure_table_base import ExposureTableBase

class TimeBinnedExposureTable(ExposureTableBase):
    """
    A class to analyze exposure time per each time range 
    
    Table columns are:
    - time_binning_index: int 
    - tstart: float
    - tstop: float
    - zpointing: np.array of [l, b] in degrees. Array of z-pointings assigned to each scatt bin.
    - xpointing: np.array of [l, b] in degrees. Array of x-pointings assigned to each scatt bin.
    - zpointing_averaged: [l, b] in degrees. Averaged z-pointing in each scatt bin.
    - xpointing_averaged: [l, b] in degrees. Averaged x-pointing in each scatt bin.
    - earth_zenith: np.array of [l, b] in degrees. Array of earth zenith assigned to each scatt bin.
    - altitude: float in km. The satellite altitude assigned to each scatt bin.
    - livetime: np.array of float in second. Exposure times for pointings assigned to each scatt bin.
    - total_livetime: float in second. total livetime for each scatt bin.
    - num_pointings: number of pointings assigned to each scatt bin.

    Attributes
    ----------
    df : :py:class:`pd.DataFrame`
        pandas dataframe with the above columns
    time_format: str
    time_scale: str
    """

    binning_index_name = 'time_binning_index' 
    binning_method = 'Time'
    additional_column_scaler = [('tstart', 'D', ''),
                                ('tstop', 'D', '')]
    additional_column_array  = None
    required_init_params = ['format', 'scale'] 

    def __init__(self, df, format, scale):
        """
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with exposure table data
        time_format: str
        time_scale: str
        """

        super().__init__(df, format=format, scale=scale)

    @classmethod
    def from_orientation(cls, orientation, tstart_list, tstop_list, **kwargs):
        """
        Produce livetime table from orientation.

        Parameters
        ----------
        orientation : :py:class:`cosipy.spacecraftfile.SpacecraftFile` 
            Orientation
        tstart_list : astropy.time.Time (array)
            Start times of GTI intervals
        tstop_list : astropy.time.Time (array)
            Stop times of GTI intervals

        Returns
        -------
        :py:class:`cosipy.spacecraftfile.SpacecraftAttitudeExposureTable`
        """

        if tstart_list.isscalar == True:
            tstart_list = Time([tstart_list])

        if tstop_list.isscalar == True:
            tstop_list = Time([tstop_list])

        if not tstart_list.format == tstop_list.format:
            tstop_list.format = tstart_list.format

        if not tstart_list.scale == tstop_list.scale :
            tstop_list = getattr(start_list, tstart_list.scale)

        time_binning_indices = []
        livetimes = []
        xpointings = [] # [l_x, b_x]
        zpointings = [] # [l_z, b_z]
        earth_zenith = [] # [earth_zenith_l, earth_zenith_b]
        altitude_list = []

        for time_binning_index, (tstart, tstop) in enumerate(zip(tstart_list, tstop_list)):

            this_orientation = orientation.select_interval(tstart, tstop)

            time_binning_indices.append(time_binning_index)

            attitude = this_orientation.attitude[:-1]
        
            pointing_list = attitude.transform_to("galactic").as_axes()

            n_pointing = len(pointing_list[0])

            x_pointings, _, z_pointings = this_orientation.attitude.as_axes()

            l_x = x_pointings.l.value[:-1]
            b_x = x_pointings.b.value[:-1]

            l_z = z_pointings.l.value[:-1]
            b_z = z_pointings.b.value[:-1]

            earth_zenith_coord = this_orientation.earth_zenith.transform_to('galactic')

            earth_zenith_l = earth_zenith_coord.l.value[:-1]
            earth_zenith_b = earth_zenith_coord.b.value[:-1]

            livetime = this_orientation.livetime.to_value(u.s)
            altitude = this_orientation.location.spherical.distance[:-1].to_value(u.km)
    
            # appending the value
            livetimes.append(livetime)
            xpointings.append([[l_x_, b_x_] for (l_x_, b_x_) in zip(l_x, b_x)])
            zpointings.append([[l_z_, b_z_] for (l_z_, b_z_) in zip(l_z, b_z)])
            earth_zenith.append([[earth_zenith_l_, earth_zenith_b_] for (earth_zenith_l_, earth_zenith_b_) in zip(earth_zenith_l, earth_zenith_b)])
            altitude_list.append(altitude)

        # to numpy
        zpointings = [ np.array(_) for _ in zpointings]
        xpointings = [ np.array(_) for _ in xpointings]
        earth_zenith = [ np.array(_) for _ in earth_zenith]

        zpointings_averaged = [ cls._get_averaged_pointing(z, dt) for (z, dt) in zip(zpointings, livetimes) ]
        xpointings_averaged = [ cls._get_averaged_pointing(x, dt) for (x, dt) in zip(xpointings, livetimes) ]

        total_livetimes = [ np.sum(np.array(_)) for _ in livetimes]
        num_pointings = [ len(_) for _ in livetimes]

        df = pd.DataFrame(data = {'time_binning_index': time_binning_indices,
                                  'tstart': tstart_list.value,
                                  'tstop': tstop_list.value,
                                  'zpointing': zpointings,
                                  'xpointing': xpointings,
                                  'zpointing_averaged': zpointings_averaged,
                                  'xpointing_averaged': xpointings_averaged,
                                  'earth_zenith': earth_zenith,
                                  'altitude': altitude_list, 
                                  'livetime': livetimes, 
                                  'total_livetime': total_livetimes,
                                  'num_pointings': num_pointings})

        return cls(df, format = tstart_list.format, scale = tstart.scale)

    def get_binned_data(self, unbinned_event, psichi_binning = 'local', sparse = False):
        """
        Create binned data from unbinned events using time binning.

        Events are grouped by time bins, energy, 
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
            Binned data with axes ["Time", "Em", "Phi", "PsiChi"].
        """

        # Get TimeTags from unbinned event
        time_tags = unbinned_event.cosi_dataset['TimeTags']

        # Warning about assumptions
        logger.warning("This method assumes that TimeTags are sorted in ascending order.")
        logger.warning(f"This method assumes that TimeTags have the same time format ('{self.format}') "
                       f"and scale ('{self.scale}') as tstart/tstop in the exposure table.")

        # Get energy bins:
        energy_bin_edges = np.array(unbinned_event.energy_bins)
        
        # Get phi bins:
        number_phi_bins = int(180./unbinned_event.phi_pix_size)
        phi_bin_edges = np.linspace(0,180,number_phi_bins+1)
        
        # Define psichi axis and data for binning:
        if psichi_binning == 'galactic':
            psichi_axis = HealpixAxis(nside = unbinned_event.nside, scheme = unbinned_event.scheme, coordsys = 'galactic', label='PsiChi')
            coords = SkyCoord(l=unbinned_event.cosi_dataset['Chi galactic']*u.deg, b=unbinned_event.cosi_dataset['Psi galactic']*u.deg, frame = 'galactic')
        if psichi_binning == 'local':
            psichi_axis = HealpixAxis(nside = unbinned_event.nside, scheme = unbinned_event.scheme, coordsys = SpacecraftFrame(), label='PsiChi')
            coords = SkyCoord(lon=unbinned_event.cosi_dataset['Chi local']*u.rad, lat=((np.pi/2.0) - unbinned_event.cosi_dataset['Psi local'])*u.rad, frame = SpacecraftFrame())
        
        # Define time axis and data for binning
        n_time_bins = len(self)
        time_axis = Axis(np.arange(n_time_bins + 1), label='Time')
        
        # Assign each event to a time bin using searchsorted
        time_bin_data = np.full(len(time_tags), -1, dtype=int)  # Initialize with -1 (no bin)
        
        for i, row in self.iterrows():
            tstart = row['tstart']
            tstop = row['tstop']
            
            # Find events in this time range
            idx_start = np.searchsorted(time_tags, tstart, side='left')
            idx_stop = np.searchsorted(time_tags, tstop, side='right')
            
            # Assign time bin index (add 0.5 for bin center)
            time_bin_data[idx_start:idx_stop] = i
        
        # Add 0.5 to place events at bin centers (for histpy)
        time_bin_data = time_bin_data.astype(float) + 0.5
        
        # Initialize histogram:
        binned_data = Histogram([time_axis,
                                 Axis(energy_bin_edges*u.keV, label='Em'),
                                 Axis(phi_bin_edges*u.deg, label='Phi'),
                                 psichi_axis],
                                 sparse=sparse)
    
        # Fill histogram (only events with valid time bins, i.e., >= 0)
        valid_mask = time_bin_data >= 0
        binned_data.fill(time_bin_data[valid_mask], 
                        unbinned_event.cosi_dataset['Energies'][valid_mask]*u.keV, 
                        np.rad2deg(unbinned_event.cosi_dataset['Phi'][valid_mask])*u.deg, 
                        coords[valid_mask])    
        
        return binned_data
