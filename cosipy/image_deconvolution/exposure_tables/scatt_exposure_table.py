import logging

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

class SpacecraftAttitudeExposureTable(ExposureTableBase):
    """
    A class to analyze exposure time per each spacecraft attitude
    
    Table columns are:
    - scatt_binning_index: int 
    - healpix_index_zpointing: int 
    - healpix_index_xpointing: int 
    - nside: int
    - scheme: str
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
    nside : int
        Healpix NSIDE parameter.
    scheme : str, default 'ring'
        Healpix scheme. Either 'ring', 'nested'. 
    """
    binning_index_name = 'scatt_binning_index' 
    binning_method = 'ScAtt'
    additional_column_scaler = [('healpix_index_z_pointing', 'K', ''),
                                ('healpix_index_x_pointing', 'K', '')]
    additional_column_array  = None
    required_init_params = ['nside', 'scheme']  # New: required parameters
    
    def __init__(self, df, nside, scheme='ring'):
        """
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with exposure table data
        nside : int
            Healpix NSIDE parameter
        scheme : str, default 'ring'
            Healpix scheme ('ring' or 'nested')
        """

        super().__init__(df, nside=nside, scheme=scheme)

    @classmethod
    def from_orientation(cls, orientation, nside, scheme = 'ring', start = None, stop = None, min_livetime = 1e-3, min_num_pointings = 1, **kwargs):
        """
        Produce livetime table from orientation.

        Parameters
        ----------
        orientation : :py:class:`cosipy.spacecraftfile.SpacecraftFile` 
            Orientation
        nside : int
            Healpix NSIDE parameter.
        scheme : str, default 'ring'
            Healpix scheme. Either 'ring', 'nested'.
        start : :py:class:`astropy.time.Time` or None, default None
            Start time to analyze the orientation
        stop : :py:class:`astropy.time.Time` or None, default None
            Stop time to analyze the orientation
        min_livetime : float or None, default None
            Minimum livetime time required for each scatt bin
        min_num_pointings : int or None, default None
            Minimum number of pointings required for each scatt bin

        Returns
        -------
        :py:class:`cosipy.spacecraftfile.SpacecraftAttitudeExposureTable`
        """

        df = cls.analyze_orientation(orientation, nside, scheme, start, stop, min_livetime, min_num_pointings)
        
        # nside and scheme are no longer stored in df
        return cls(df, nside=nside, scheme=scheme)

    # GTI should be a mandary parameter
    @classmethod
    def analyze_orientation(cls, orientation, nside, scheme = 'ring', start = None, stop = None, min_livetime = None, min_num_pointings = None):
        """
        Produce pd.DataFrame from orientation.

        Parameters
        ----------
        orientation : :py:class:`cosipy.spacecraftfile.SpacecraftFile` 
            Orientation
        nside : int
            Healpix NSIDE parameter.
        scheme : str, default 'ring'
            Healpix scheme. Either 'ring', 'nested'.
        start : :py:class:`astropy.time.Time` or None, default None
            Start time to analyze the orientation
        stop : :py:class:`astropy.time.Time` or None, default None
            Stop time to analyze the orientation
        min_livetime : float or None, default None
            Minimum livetime time required for each scatt bin
        min_num_pointings : int or None, default None
            Minimum number of pointings required for each scatt bin

        Returns
        -------
        :py:class:`pd.DataFrame`
        """

        logger.info(f'angular resolution: {hp.nside2resol(nside) * 180 / np.pi} deg.')    
    
        indices_healpix = [] # (idx_z, idx_x) # only for calculation
        healpix_indices_zpointing = []
        healpix_indices_xpointing = []
        livetimes = []
        xpointings = [] # [l_x, b_x]
        zpointings = [] # [l_z, b_z]
        earth_zenith = [] # [earth_zenith_l, earth_zenith_b]
        altitude_list = []
                
        if start is not None and stop is not None:
            orientation = orientation.select_interval(start, stop)
        elif start is not None:
            logger.error("please specify the stop time")
            return
        elif stop is not None:
            logger.error("please specify the start time")
            return
        
        ori_time = orientation.obstime
            
        logger.info(f'duration: {(ori_time[-1] - ori_time[0]).to("day")}')
        
        attitude = orientation.attitude[:-1]
        
        pointing_list = attitude.transform_to("galactic").as_axes()

        n_pointing = len(pointing_list[0])

        x_pointings, _, z_pointings = orientation.attitude.as_axes()

        l_x = x_pointings.l.value[:-1]
        b_x = x_pointings.b.value[:-1]

        l_z = z_pointings.l.value[:-1]
        b_z = z_pointings.b.value[:-1]

        earth_zenith_coord = orientation.earth_zenith.transform_to('galactic')

        earth_zenith_l = earth_zenith_coord.l.value[:-1]
        earth_zenith_b = earth_zenith_coord.b.value[:-1]

        if scheme == 'ring':
            nest = False
        elif scheme == 'nested':
            nest = True
        else:
            logger.warning('Warning: the scheme should be "ring" or "nested". It was set to "ring".')
            nest = False

        idx_x = hp.ang2pix(nside, l_x, b_x, nest=nest, lonlat=True)
        idx_z = hp.ang2pix(nside, l_z, b_z, nest=nest, lonlat=True)
        
        livetime = orientation.livetime.to_value(u.s)
        altitude = orientation.location.spherical.distance[:-1].to_value(u.km)
        
        for i in tqdm(range(n_pointing)):
            
            if (idx_z[i], idx_x[i]) in indices_healpix:
                idx = indices_healpix.index((idx_z[i], idx_x[i]))

                livetimes[idx].append(livetime[i])
                xpointings[idx].append([l_x[i], b_x[i]])
                zpointings[idx].append([l_z[i], b_z[i]])            
                earth_zenith[idx].append([earth_zenith_l[i], earth_zenith_b[i]])
                altitude_list[idx].append(altitude[i])
            else:
                indices_healpix.append((idx_z[i], idx_x[i]))

                healpix_indices_zpointing.append(idx_z[i])
                healpix_indices_xpointing.append(idx_x[i])

                livetimes.append([livetime[i]])
                xpointings.append([[l_x[i], b_x[i]]])
                zpointings.append([[l_z[i], b_z[i]]])
                earth_zenith.append([[earth_zenith_l[i], earth_zenith_b[i]]])
                altitude_list.append([altitude[i]])
        
        indices_scatt_binning = [i for i in range(len(indices_healpix))] 
        
        # to numpy
        zpointings = [ np.array(_) for _ in zpointings]
        xpointings = [ np.array(_) for _ in xpointings]
        earth_zenith = [ np.array(_) for _ in earth_zenith]

        zpointings_averaged = [ cls._get_averaged_pointing(z, dt) for (z, dt) in zip(zpointings, livetimes) ]
        xpointings_averaged = [ cls._get_averaged_pointing(x, dt) for (x, dt) in zip(xpointings, livetimes) ]

        total_livetimes = [ np.sum(np.array(_)) for _ in livetimes]
        num_pointings = [ len(_) for _ in livetimes]
        
        df = pd.DataFrame(data = {'scatt_binning_index': indices_scatt_binning,
                                  'healpix_index_z_pointing': healpix_indices_zpointing,
                                  'healpix_index_x_pointing': healpix_indices_xpointing,
                                  'zpointing': zpointings,
                                  'xpointing': xpointings,
                                  'zpointing_averaged': zpointings_averaged,
                                  'xpointing_averaged': xpointings_averaged,
                                  'earth_zenith': earth_zenith,
                                  'altitude': altitude_list, 
                                  'livetime': livetimes, 
                                  'total_livetime': total_livetimes,
                                  'num_pointings': num_pointings})
        
        if min_livetime is not None:
            df = df[df['total_livetime'] >= min_livetime]

        if min_num_pointings is not None:
            df = df[df['num_pointings'] >= min_num_pointings]
        
        if min_livetime is not None or min_num_pointings is not None:
            df['scatt_binning_index'] = [i for i in range(len(df))] 
            df = df.reset_index(drop=True)
        
        return df 

    def calc_pointing_trajectory_map(self):
        """
        Calculate a 2-dimensional map showing exposure time for each spacecraft attitude.

        Returns
        -------
        :py:class:`cosipy.spacecraft.SpacecraftAxisMap`

        Notes
        -----
        The default axes in SpacecraftAttitudeMap is x- and y-pointings,
        but here the spacecraft attitude is described with z- and x-pointings. 
        """
    
        map_pointing_zx = SpacecraftAxisMap(nside = self.nside,
                                            scheme = self.scheme,
                                            coordsys = 'galactic',
                                            labels = ('z', 'x'))

        # HEALPix pixel indices for axes 1 and 2 (stored as a Series of 2-tuples,
        # converted to a tuple of indices per axis)
        #pix0, pix1 = tuple(zip(*self['healpix_index']))

        total_livetime = u.Quantity(self['total_livetime'].values, unit = u.s, copy=False)

        map_pointing_zx.fill(self['healpix_index_z_pointing'], self['healpix_index_x_pointing'], weight = total_livetime)
        
        return map_pointing_zx

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
            Binned data with axes ["ScAtt", "Em", "Phi", "PsiChi"].
        """

        exposure_dict = {(row['healpix_index_z_pointing'], row['healpix_index_x_pointing']): row['scatt_binning_index'] for _, row in self.iterrows()}
            
        # from BinnedData.py
 
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

        # Define scatt axis and data for binning
        n_scatt_bins = len(self)
        scatt_axis = Axis(np.arange(n_scatt_bins + 1), label='ScAtt')
        
        is_nest = True if self.scheme == 'nested' else False
        
        nside_scatt = self.nside
        
        zindex = hp.ang2pix(nside_scatt, unbinned_event.cosi_dataset['Zpointings (glon,glat)'].T[0] * 180 / np.pi, 
                            unbinned_event.cosi_dataset['Zpointings (glon,glat)'].T[1] * 180 / np.pi, nest=is_nest, lonlat=True)
        xindex = hp.ang2pix(nside_scatt, unbinned_event.cosi_dataset['Xpointings (glon,glat)'].T[0] * 180 / np.pi, 
                            unbinned_event.cosi_dataset['Xpointings (glon,glat)'].T[1] * 180 / np.pi, nest=is_nest, lonlat=True)    
        scatt_data = np.array( [ exposure_dict[(z, x)] + 0.5 if (z,x) in exposure_dict.keys() else -1 for z, x in zip(zindex, xindex)] ) # should this "0.5" be needed?
        
        # Initialize histogram:
        binned_data = Histogram([scatt_axis,
                                 Axis(energy_bin_edges*u.keV, label='Em'),
                                 Axis(phi_bin_edges*u.deg, label='Phi'),
                                 psichi_axis],
                                 sparse=sparse)

        # Fill histogram:
        binned_data.fill(scatt_data, unbinned_event.cosi_dataset['Energies']*u.keV, np.rad2deg(unbinned_event.cosi_dataset['Phi'])*u.deg, coords)    
        
        return binned_data
