#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 15:39:42 2024

@author: shengyong
"""

import numpy as np
from mhealpy import HealpixMap
from mhealpy.pixelfunc.moc import *
from mhealpy.pixelfunc.single import *
from .fast_ts_fit import FastTSMap
import matplotlib.pyplot as plt
from copy import deepcopy
import astropy.units as u
from astropy.coordinates import SkyCoord
from pathlib import Path
import logging
logger = logging.getLogger(__name__)


class MOCTSMap(FastTSMap):
    
    def __init__(self, data, bkg_model, response_path, orientation = None, cds_frame = "local"):
        
        """
        Initialize the instance of a MOC TS map fit.
        
        Parameters
        ----------
        data : histpy.Histogram
            Observed data, which includes counts from both signal and background.
        bkg_model : histpy.Histogram
            Background model, which includes the background counts to model the background in the observed data.
        response_path : str or pathlib.Path
            The path to the response file.
        orientation : cosipy.SpacecraftFile, optional
            The orientation of the spacecraft when data are collected (the default is `None`, which implies the orientation file is not needed).
        cds_frame : str, optional
            "local" or "galactic", it's the Compton data space (CDS) frame of the data, bkg_model and the response. In other words, they should have the same cds frame (the default is "local", which implied that a local frame that attached to the spacecraft).
        
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
        tuple
            The upscaled the map and the unique numbers of the child pixels.
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
        Convert the uniq number to the corresponding central skycoord.
        
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
        Convert the uniq to the pixel index in the map.
        
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
        Fill up the moc map based on the pixidx.

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

    
    def moc_ts_fit(self, max_moc_order, top_number, energy_channel, spectrum, start_method = "fork", cpu_cores = None):
        
        """
        Fit the MOC map.
        
        Parameters
        ----------
        max_moc_order : int
            The order of the MOC map to stop the fitting.
        top_number : int
            The pixels with the top likelihood to will be upscaled. For example, pixels with top eight likelihoods will be considered as mother pixels to be split into the child pixels.
        energy_channel : list
            The energy channel to be used for the MOC map fitting.
        spectrum : 
            The spectrum model of the source to fit the model.
        start_method : str, optional
            The starting method of the parallel computation (the default is "fork", which implies using the fork method to start parallel computation).
        cpu_cores : int, optional
            The number of cpu cores you wish to use for the parallel computation (the default is `None`, which implies using all the available number of cores -1 to perform the parallel computation).
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
        results = self.parallel_ts_fit(hypothesis_coords = hypothesis_coords_list, energy_channel = energy_channel, spectrum = spectrum, pixel_idx = pixidx)
        self.ts_array = results[:,1]

        # fill up the 0th order moc map
        moc_map_ts = MOCTSMap.fill_up_moc_map(pixidx, moc_map_ts, results)
        self.moc_map_ts = moc_map_ts

        # store all ts maps
        self.all_maps = []
        self.all_maps += [moc_map_ts]
        
        
        # # if the user requires higher order fit
        # threshold = moc_map_ts[:].max() - MOCTSMap.get_chi_critical_value(split_containment)  # the threshold value to decide the mother pixels to be split
        order += 1
        print("--------------------------------------------------------------------------------")
        
        # start the while loop
        while order <= max_moc_order:
            
            # decide the mother pixels to divide
            # threshold = moc_map_ts[:].max() - MOCTSMap.get_chi_critical_value(split_containment)
            top_number_arg_array = np.argpartition(moc_map_ts, -top_number)[-top_number:]
            print(f"The top {top_number} ts values are: {top_number_arg_array} in the last iteration, splitting these pixels...")
            threshold = min(moc_map_ts[top_number_arg_array])
            
            mother_idx = np.where(moc_map_ts[:] >= threshold)[0]
            mother_uniq = moc_map_ts.uniq[mother_idx]
            
            # upscale the resolution of the mother pixels by 1 order, now the moc map is updated
            moc_map_ts, child_uniq = MOCTSMap.upscale_moc_map(moc_map_ts, uniq_mother = mother_uniq, new_order = order)
            
            # get the sky coordinates of the child pixels
            hypothesis_coords = MOCTSMap.uniq2skycoord(child_uniq)
            hypothesis_coords_list = [i for i in hypothesis_coords]  # have to split the SkyCoord object into SkyCoord object list
            child_idx = MOCTSMap.uniq2pixidx(moc_map_ts, child_uniq)  # child_idx is used to make sure that the ts values are filled into the correct pixels
            print(f"fitting order {order} with {len(hypothesis_coords_list)} hypothesis coordinates")
            results = self.parallel_ts_fit(hypothesis_coords = hypothesis_coords_list, energy_channel = energy_channel, spectrum = spectrum, ts_nside = 2**order, pixel_idx = child_idx)
            self.ts_array = results[:,1]
            
            # fill up the child pixels
            moc_map_ts = MOCTSMap.fill_up_moc_map(child_idx, moc_map_ts, results)
            self.moc_map_ts = moc_map_ts
            self.all_maps += [moc_map_ts]
    
                
            order +=1
            print("--------------------------------------------------------------------------------")
            
        
        return moc_map_ts
    
    def plot_ts(self, moc_map = None, skycoord = None, containment = None, save_plot = False, save_dir = "", save_name = "ts_map.png", dpi = 300):

        """
        Plot the containment region of the TS map.

        Parameters
        ----------
        ts_array : numpy.ndarray
            The array of ts values from parallel ts fit.
        skyoord : astropy.coordinates.SkyCoord, optional
            The true location of the source (the default is `None`, which implies that there are no coordiantes to be printed on the TS map).
        containment : float, optional
            The containment level of the source (the default is `None`, which will plot raw TS values).
        save_plot : bool, optional
            Set `True` to save the plot (the default is `False`, which means it won't save the plot.
        save_dir : str or pathlib.Path, optional
            The directory to save the plot.
        save_name : str, optional
            The file name of the plot to be save.
        dpi : int, optional
            The dpi for plotting and saving.
        """

            
        if moc_map is None:
            moc_map = self.moc_map_ts
 
        # decide the critical value
        if containment is not None:
            critical = MOCTSMap.get_chi_critical_value(containment = 0.9)
            max_ts = np.max(moc_map[:])
        
        # get plotting canvas
        fig = plt.figure(dpi = dpi)
        
        axMoll = fig.add_subplot(1,1,1, projection = 'mollview')
        
        # Plot in one of the axes 
        if containment is None:
            plotMoll, projMoll = moc_map.plot(ax = axMoll)
        else:
            plotMoll, projMoll = moc_map.plot(ax = axMoll, vmin = max_ts-critical, vmax = max_ts)
            
        moc_map.plot_grid(ax = plt.gca(), color = 'grey', linewidth = 0.1);
        
        
        # plot the sky cooordinates if given
        if skycoord is not None:
            
            axMoll.text(skycoord.l.deg, skycoord.b.deg, "x", size = 4, 
                        horizontalalignment='center',
                        verticalalignment='center',
                        transform = axMoll.get_transform('world'), color = "red")

        if save_plot == True:

            fig.savefig(Path(save_dir)/save_name, dpi = dpi)
    
