from histpy import Histogram, HealpixAxis, Axis, Axes

import numpy as np
import healpy as hp
import pandas as pd
from astropy.io import fits
import astropy.units as u
from tqdm.autonotebook import tqdm

class ScatExposureTable(pd.DataFrame):
    """
    scat_binning_index : 
    healpix_index : list of tuple, (healpix_index_zpointing, healpix_index_xpointing)
    zpointing : np.array, [l, b] in degrees
    xpointing : np.array, [l, b] in degrees
    delta_time : 
    exposure : 
    num_pointings :
    bkg_group :
    nside : nside for the model map
    scheme : healpix scheme (ring or nested)
    """

    def __init__(self, df, nside, scheme = 'ring'):

        super().__init__(pd.DataFrame(df))

        self.nside = nside

        if scheme == 'ring' or scheme == 'nested':
            self.scheme = scheme
        else:
            print('Warning: the scheme should be "ring" or "nested". It was set to "ring".')
            self.scheme = 'ring'

    def __eq__(self, other):
        for name in ['scat_binning_index', 'healpix_index', 'exposure', 'num_pointings', 'bkg_group']:
            if not np.all(self[name] == other[name]):
                return False
        
        for name in ['delta_time', 'zpointing', 'xpointing']:
            for self_, other_ in zip(self[name], other[name]):
                if not np.all(self_ == other_):
                    return False

        return (self.nside == other.nside) and (self.scheme == other.scheme)

    @classmethod
    def from_pickle(cls, filename, nside, scheme = 'ring'):

        df = pd.read_pickle(filename)

        new = cls(df, nside, scheme)

        return new

    @classmethod
    def from_orientation(cls, orientation, nside, scheme = 'ring', start = None, stop = None, min_exposure = None, min_num_pointings = None):
        
        df = cls.analyze_orientation(orientation, nside, scheme, start, stop, min_exposure, min_num_pointings)

        new = cls(df, nside, scheme)

        return new

    # GTI should be a mandary parameter
    @classmethod
    def analyze_orientation(cls, orientation, nside, scheme = 'ring', start = None, stop = None, min_exposure = None, min_num_pointings = None):

        print("angular resolution: ", hp.nside2resol(nside) * 180 / np.pi, "deg.")    
    
        indices_healpix = [] # (idx_z, idx_x)
        delta_times = []
        xpointings = [] # [l_x, b_x]
        zpointings = [] # [l_z, b_z]
                
        if start is not None and stop is not None:
            orientation = orientation.source_interval(start, stop)
        elif start is not None:
            print("please specify the stop time")
        elif stop is not None:
            print("please specify the start time")
        
        ori_time = orientation.get_time()
            
        print("duration: ", (ori_time[-1] - ori_time[0]).to("day"))
        
        attitude = orientation.get_attitude()
        
        pointing_list = attitude.transform_to("galactic").as_axes()

        n_pointing = len(pointing_list[0])
        
        x_1, x_2 = pointing_list[0][:-1], pointing_list[0][1:]
        l_x, b_x =  0.5 * (x_1.l.degree + x_2.l.degree), 0.5 * (x_1.b.degree + x_2.b.degree)

        z_1, z_2 = pointing_list[2][:-1], pointing_list[2][1:]
        l_z, b_z =  0.5 * (z_1.l.degree + z_2.l.degree), 0.5 * (z_1.b.degree + z_2.b.degree)

        if scheme == 'ring':
            nest = False
        elif scheme == 'nested':
            nest = True
        else:
            print('Warning: the scheme should be "ring" or "nested". It was set to "ring".')
            nest = False

        idx_x = hp.ang2pix(nside, l_x, b_x, nest=nest, lonlat=True)
        idx_z = hp.ang2pix(nside, l_z, b_z, nest=nest, lonlat=True)
        
        delta_time = (ori_time[1:] - ori_time[:-1]).to('s').value
        
        for i in tqdm(range(n_pointing - 1)):
            
            if (idx_z[i], idx_x[i]) in indices_healpix:
                idx = indices_healpix.index((idx_z[i], idx_x[i]))
                delta_times[idx].append(delta_time[i])
                xpointings[idx].append([l_x[i], b_x[i]])
                zpointings[idx].append([l_z[i], b_z[i]])            
                
            else:
                indices_healpix.append((idx_z[i], idx_x[i]))
                delta_times.append([delta_time[i]])
                xpointings.append([[l_x[i], b_x[i]]])
                zpointings.append([[l_z[i], b_z[i]]])
        
        indices_scat_binning = [i for i in range(len(indices_healpix))] 
        
        df = pd.DataFrame(data = {'scat_binning_index': indices_scat_binning, 'healpix_index': indices_healpix, 
                                  'zpointing': [ np.array(_) for _ in zpointings], 
                                  'xpointing': [ np.array(_) for _ in xpointings], 
                                  'delta_time': delta_times, 'exposure': [ np.sum(np.array(_)) for _ in delta_times],
                                  'num_pointings': [ len(_) for _ in delta_times],
                                  'bkg_group': [ 0 for i in delta_times]})
        
        if min_exposure is not None:
            df = df[df['exposure'] >= min_exposure]

        if min_num_pointings is not None:
            df = df[df['num_pointings'] >= min_num_pointings]
        
        if min_exposure is not None or min_num_pointings is not None:
            df['scat_binning_index'] = [i for i in range(len(df))] 
        
        return df 

    @classmethod
    def from_fits(cls, filename):
        infile = fits.open(filename)
        hdu = infile[1]
    
        if hdu.name != "EXPOSURETABLE":
            print("cannot find EXPOSURETABLE")
            return 0
    
        indices_scat_binning = hdu.data['scat_binning_index']
        indices_healpix = [ (z, x) for (z, x) in zip(hdu.data['healpix_index_z_pointing'], hdu.data['healpix_index_x_pointing']) ]
        zpointings = [ [ [l, b] for (l, b) in zip(z_l, z_b) ] for (z_l, z_b) in zip(hdu.data['zpointing_l'], hdu.data['zpointing_b']) ]
        xpointings = [ [ [l, b] for (l, b) in zip(x_l, x_b) ] for (x_l, x_b) in zip(hdu.data['xpointing_l'], hdu.data['xpointing_b']) ]
        delta_times = np.array(hdu.data['delta_time'])
        exposures = hdu.data['exposure']
        num_pointings = hdu.data['num_pointings']
        bkg_groups = hdu.data['bkg_group']
        
        df = pd.DataFrame(data = {'scat_binning_index': indices_scat_binning, 'healpix_index': indices_healpix, 
                                  'zpointing': [ np.array(_) for _ in zpointings], 
                                  'xpointing': [ np.array(_) for _ in xpointings], 
                                  'delta_time': delta_times, 'exposure': [ np.sum(np.array(_)) for _ in delta_times],
                                  'num_pointings': [ len(_) for _ in delta_times],
                                  'bkg_group': [ 0 for i in delta_times]})

        nside = hdu.header['NSIDE']
        scheme = hdu.header['SCHEME']
        
        new = cls(df, nside, scheme)

        return new

    def save_as_fits(self, filename, overwrite = True):
        # primary HDU
        primary_hdu = fits.PrimaryHDU()
    
        #exposure table
        names = ['scat_binning_index', 'exposure', 'num_pointings', 'bkg_group']
        formats = ['K', 'D', 'K', 'K']
        units = ['', 's', '', '']
        
        columns = [ fits.Column(name=names[i], array=self[names[i]].to_numpy(), format = formats[i], unit = units[i]) 
                     for i in range(len(names))]
        
        column_healpix_index_z_pointing = fits.Column(name='healpix_index_z_pointing', 
                                                      array=np.array([idx[0] for idx in self['healpix_index']]), format = 'K')
        column_healpix_index_x_pointing = fits.Column(name='healpix_index_x_pointing', 
                                                      array=np.array([idx[1] for idx in self['healpix_index']]), format = 'K')
        
        columns.append(column_healpix_index_z_pointing)
        columns.append(column_healpix_index_x_pointing)
    
        column_delta_time = fits.Column(name='delta_time', format='PD()', unit = 's',
                                        array=np.array(self['delta_time'].array, dtype=np.object_))
        columns.append(column_delta_time)    
        
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
        
        table_hdu = fits.BinTableHDU.from_columns(columns) 
        table_hdu.name = 'exposuretable'

        table_hdu.header['nside'] = self.nside
        table_hdu.header['scheme'] = self.scheme
        
        #save file    
        hdul = fits.HDUList([primary_hdu, table_hdu])    
        hdul.writeto(filename, overwrite = overwrite)

    def calc_pointing_trajectory_map(self):
        axes_z = HealpixAxis(nside = self.nside, scheme = self.scheme, label = "zpointing")
        axes_x = HealpixAxis(nside = self.nside, scheme = self.scheme, label = "xpointing")
        axes_zx = Axes([axes_z, axes_x])
    
        map_pointing_zx = Histogram(axes_zx, unit = u.s, sparse = False)
    
        for hp_index, exposure in zip(self['healpix_index'], self['exposure']):
            map_pointing_zx[hp_index[0], hp_index[1]] = exposure * u.s
        
        return map_pointing_zx
