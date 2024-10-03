#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 15:39:42 2024

@author: shengyong
"""

import numpy as np
#import healpy as hp
from mhealpy import HealpixMap
from mhealpy.pixelfunc.moc import *
from mhealpy.pixelfunc.single import *
#import matplotlib.pyplot as plt
from .fast_ts_fit import FastTSMap

from copy import deepcopy
import astropy.units as u
from astropy.coordinates import SkyCoord
from copy import deepcopy
import multiprocessing
from itertools import product
import time
import logging
logger = logging.getLogger(__name__)


class MOCTSMap(FastTSMap):
    
    def __init__(self, data, bkg_model, response_path, orientation = None, cds_frame = "local"):
        
        """
        """
        
        super().__init__(data, bkg_model, response_path, orientation = orientation, cds_frame = cds_frame)
        
        
    @staticmethod
    def upscale_moc_map(m, uniq_mother, new_order):

        """
        Upscale the MOC map on certain mother pixels. All the child pixels will be filled by the value of the mother pixel.

        Parameters
        ----------
        m : mhealpy.containers.healpix_map.HealpixMap
            The input map to be upscaled.
        uniq_mother_pix : int or array
            The uniq number the mother pixels to be upscaled.
        new_order : int
            The order of the child pixels upscaled from the mother pixels.

        Returns
        -------
        """

        if not m.is_moc:
            raise TypeError("The input map must be a MOC map.")

        # copy the uniq numbers and the data from the original map
        new_uniq = deepcopy(m.uniq)
        new_data = deepcopy(m.data)

        new_nside = 2**new_order

        uniq_child_all = []

        for mother_uniq in uniq_mother:

            # get the index of the mother pixel
            idx = np.where(new_uniq == mother_uniq)[0][0]  # note that idx is the index of the uniq number, not the uniq number

            # get the start and stop of the child pixel number in the NESTED scheme (also the index in the NESTED scheme case)
            start_nest = uniq2range(new_nside, mother_uniq)[0]
            stop_nest = uniq2range(new_nside, mother_uniq)[1]

            # convert the child pixel number from NESTED scheme to UNIQ scheme
            uniq_child = nest2uniq(new_nside, np.arange(start_nest, stop_nest))
            uniq_child_all += list(uniq_child)

            # update the moc map
            new_uniq = np.concatenate((new_uniq[:idx], 
                                       uniq_child, 
                                       new_uniq[idx+1:]))

            new_data = np.concatenate((new_data[:idx], 
                                       np.repeat(new_data[idx], stop_nest-start_nest), 
                                       new_data[idx + 1:]))

        m_new = HealpixMap(data = new_data, uniq = new_uniq)

        return m_new, np.array(uniq_child_all)
    
    @staticmethod
    def uniq2skycoord(uniq):

        """
        Convert the uniq number to the corresponding central skycoord
        
        Parameters
        ----------
        uniq : int, list or numpy.ndarray
            The uniq number(s) of the pixel(s)
            
        Returns
        -------
        astropy.Coordinates.SkyCoord
            The galactic skycoord of the input uniq pixels.
        """

        nside, pix_num_nested = uniq2nest(uniq)

        lon, lat = pix2ang(nside = nside, ipix = pix_num_nested, nest = True, lonlat = True)

        return SkyCoord(l = lon, b = lat, unit = (u.deg, u.deg), frame = "galactic")
    
    @staticmethod
    def uniq2pixidx(m, uniq):
        
        """
        Convert the uniq to the pixel index in the map
        
        Parameters
        ----------
        m : mhealpy.containers.healpix_map.HealpixMap
            The map that contains the moc pixels
        uniq : int, list or numpy.ndarray:
            The uniq number(s) of the pixel(s)
        
        Returns
        -------
        list
            The list of the pixel index of the corresponding uniq pixels in the map
        """
        
        return [np.where(m.uniq == i)[0][0] for i in uniq]

    def fill_up_moc_map(pixidx, m, results):

        """
        Fill up the moc map based on the pixidx

        Parameters
        ----------
        pixidx : int or list
            The pixel index, not the uniq number of the pixels
        m : mhealpy.containers.healpix_map.HealpixMap
            The MOC map to be filled
        results : numpy.ndarray
            The ts fit results.

        Returns
        -------
        mhealpy.containers.healpix_map.HealpixMap
            The filled map
        """

        if isinstance(pixidx, int):
            pixidx = [pixidx]

        for pixidx_ in pixidx:
            pixidx_ = int(pixidx_)

            idx = np.where(results[:,0].astype(int) == pixidx_)[0]  # idx is the row idx of the result array where the first column equals to pixidx_
            if idx.shape[0] != 1:
                raise ValueError(f"Pixel with pixel index {pixidx_} has {idx.shape[0]} fits! ")
            else:
                m[pixidx_] = results[idx,1]

        return m

    
    def moc_ts_fit(self, max_moc_order, top_number, energy_channel, spectrum, start_method = "fork", cpu_cores = None, ts_nside = None):
        
        """
        Fit the MOC map
        """
        
        # initialize the order 
        order = 0
        
        # initialize the 0th order moc map, which is equlivent to a 0th order single resolution map
        uniq = nest2uniq(1, np.arange(12))
        moc_map_ts = HealpixMap(data = np.repeat(0, 12), uniq = uniq)
        
        # make the 0th order fit over all pixels
        hypothesis_coords = MOCTSMap.uniq2skycoord(moc_map_ts.uniq)
        hypothesis_coords_list = [i for i in hypothesis_coords]  # have to split the SkyCoord object into SkyCoord object
        pixidx = MOCTSMap.uniq2pixidx(moc_map_ts, moc_map_ts.uniq)
        
        print(f"fitting order = {order}")
        print(f"fitting {len(hypothesis_coords_list)} hypothesis coordinates")
        ts_results = self.parallel_ts_fit(hypothesis_coords = hypothesis_coords_list, energy_channel = energy_channel, spectrum = spectrum, pixel_idx = pixidx)

        # fill up the 0th order moc map
        moc_map_ts = MOCTSMap.fill_up_moc_map(pixidx, moc_map_ts, ts_results)
        
        
        # # if the user requires higher order fit
        # threshold = moc_map_ts[:].max() - MOCTSMap.get_chi_critical_value(split_containment)  # the threshold value to decide the mother pixels to be split
        order += 1
        print("--------------------------------------------------------------------------------")
        
        # start the while loop
        while order <= max_moc_order:
            
            # decide the mother pixels to divide
            # threshold = moc_map_ts[:].max() - MOCTSMap.get_chi_critical_value(split_containment)
            top_number_arg_array = np.argpartition(moc_map_ts, -top_number)[-top_number:]
            threshold = min(moc_map_ts[top_number_arg_array])
            
            mother_idx = np.where(moc_map_ts[:] >= threshold)[0]
            mother_uniq = moc_map_ts.uniq[mother_idx]
            
            # upscale the resolution of the mother pixels by 1 order, now the moc map is updated
            moc_map_ts, child_uniq = MOCTSMap.upscale_moc_map(moc_map_ts, uniq_mother = mother_uniq, new_order = order)
            
            # get the sky coordinates of the child pixels
            hypothesis_coords = MOCTSMap.uniq2skycoord(child_uniq)
            hypothesis_coords_list = [i for i in hypothesis_coords]  # have to split the SkyCoord object into SkyCoord object
            child_idx = MOCTSMap.uniq2pixidx(moc_map_ts, child_uniq)  # child_idx is used to make sure that the ts values are filled into the correct pixels
            print(f"fitting order {order} with {len(hypothesis_coords_list)} hypothesis coordinates")
            ts_results = self.parallel_ts_fit(hypothesis_coords = hypothesis_coords_list, energy_channel = energy_channel, spectrum = spectrum, ts_nside = 2**order, pixel_idx = child_idx)
            
            # fill up the child pixels
            moc_map_ts = MOCTSMap.fill_up_moc_map(child_idx, moc_map_ts, ts_results)
                
            order +=1
            print("--------------------------------------------------------------------------------")
            
        
        return moc_map_ts
