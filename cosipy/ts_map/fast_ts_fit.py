from histpy import Histogram, Axis, Axes
import h5py as h5
import sys
from cosipy import SpacecraftFile
from cosipy.response import PointSourceResponse
import healpy as hp
from mhealpy import HealpixMap
import numpy as np
import os
import multiprocessing
from itertools import product
from .fast_norm_fit import FastNormFit as fnf
from pathlib import Path
from cosipy.response import FullDetectorResponse
import time
import scipy.stats

class FastTSMap():
    
    def __init__(self, data, bkg_model, response_path, orientation = None, cds_frame = "local", scheme = "RING"):
        
        """
        Initialize the instance
        
        Parameters
        ----------
        data: histpy.Histogram; observed data, whichincludes counts from both signal and background
        bkg_model: histpy.Histogram; background model, which includes the background counts to model the background in observed data
        orientation: cosipy.SpacecraftFile; the orientation of the spacecraft when data are collected
        response_path: pathlib.Path; the path to the response file
        cds_frame: str; "local" or "galactic", it's the frame of the data, bkg_model and the response 
        
        Returns
        -------
        """
        
        self._data = data.project(["Em", "PsiChi", "Phi"])
        self._bkg_model = bkg_model.project(["Em", "PsiChi", "Phi"])
        self._orientation = orientation
        self._response_path = Path(response_path)
        self._cds_frame = cds_frame
        self._scheme = scheme
        
    @staticmethod
    def slice_energy_channel(hist, channel_start, channel_stop):
        """
        Slice one or more bins along first axis
        
        Parameters
        ----------
        hist: histpy.Histogram; the hist to be sliced
        channel_start: int; the start of the slice (inclusive)
        channel_stop: int; the stop of the slice (exclusive)
        
        Returns
        -------
        sliced_hist: histpy.Histogram: the sliced hist
        """
        
        sliced_hist = hist.slice[channel_start:channel_stop,:]
        
        return sliced_hist
    
    @staticmethod
    def get_hypothesis_coords(nside, scheme = "RING", coordsys = "galactic"):
        
        """
        Get a list of hypothesis coordinates
        
        Parameters
        ----------
        nside: int; the nside of the map
        
        Returns
        -------
        hypothesis_coords: list; the list of the hypothesis coordinates at the center of each pixel
        """
        
        data_array = np.zeros(hp.nside2npix(nside))
        ts_temp = HealpixMap(data = data_array, scheme = scheme, coordsys = coordsys)
        
        hypothesis_coords = []
        for i in np.arange(data_array.shape[0]):
            hypothesis_coords += [ts_temp.pix2skycoord(i)]
            
        return hypothesis_coords
    
    
    @staticmethod
    def get_cds_array(hist, energy_channel):
        
        """
        Get the flattened cds array from input Histogram.
        
        Parameters
        -----------
        hist: histpy.Histogram; input Histogram
        energy_channel: list; [lower_channel, upper_chanel]
        
        Returns
        -------
        cds_array
        
        """
        if not isinstance(hist, Histogram):
            raise TypeError("Please input hist must be a histpy.Histogram object.")
        
        hist_axes_labels = hist.axes.labels
        cds_labels = ["PsiChi", "Phi"]
        if not all([label in hist_axes_labels for label in cds_labels]):
            raise ValueError("The data doesn't contain the full Compton Data Space!")
            
        hist = hist.project(["Em", "PsiChi", "Phi"]) # make sure the first axis is the measured energy
        hist_cds_sliced = FastTSMap.slice_energy_channel(hist, energy_channel[0], energy_channel[1])   
        hist_cds = hist_cds_sliced.project(["PsiChi", "Phi"])
        cds_array = np.array(hist_cds.to_dense()[:]).flatten()  # here [:] is equivalent to [:, :]
        
        return cds_array
    
    @staticmethod
    def get_expectation_in_galactic(hypothesis_coord, response_path, spectrum):
        
        """
        Get the expectation in galactic. Please be aware that you must use a galactic response!
        # to do: to make the weight parameter not hardcoded
        
        Parameters
        ----------
        hypothesis_coord:
        spectrum:
        
        Returns
        -------
        expectation
        """
        
        # Open the response
        # Notes from Israel: Inside it contains a single histogram with all the regular axes for a Compton Data Space (CDS) analysis, in galactic coordinates. Since there is no class yet to handle it, this is how to read in the HDF5 manually.
        
        with h5.File(response_path) as f:

            axes_group = f['hist/axes']
            axes = []
            for axis in axes_group.values():
                # Get class. Backwards compatible with version
                # with only Axis
                axis_cls = Axis
                if '__class__' in axis.attrs:
                    class_module, class_name = axis.attrs['__class__']
                    axis_cls = getattr(sys.modules[class_module], class_name)
                axes += [axis_cls._open(axis)]
        axes = Axes(axes)
        
        # get the pixel number of the hypothesis coordinate
        map_temp = HealpixMap(base = axes[0])
        hypothesis_coord_pix_number = map_temp.ang2pix(hypothesis_coord)
        
        # get the expectation for the hypothesis coordinate (a point source)
        with h5.File(response_path) as f:
            pix = hypothesis_coord_pix_number
            psr = PointSourceResponse(axes[1:], f['hist/contents'][pix+1], unit = f['hist'].attrs['unit'])
                
        return psr
    
    
    @staticmethod
    def get_ei_cds_array(hypothesis_coord, energy_channel, response_path, spectrum, cds_frame, orientation = None):
                         
        """
        Get the expected counts in CDS in local or galactic frame.
        
        Parameters
        ----------
        hypothesis_coord
        energy_channel
        orientation
        response_path
        spectrum
        
        Returns
        -------
        cds_array
        """
                         
        # check inputs, will complete later
                         
        # the local and galactic frame works very differently, so we need to compuate the point source response (psr) accordingly 
        if cds_frame == "local":
            
            if orientation == None:
                raise TypeError("The when the data are binned in local frame, orientation must be provided to compute the expected counts.")
            
            # convert the hypothesis coord to the local frame (Spacecraft frame)
            hypothesis_in_sc_frame = orientation.get_target_in_sc_frame(target_name = "Hypothesis", 
                                                                        target_coord = hypothesis_coord, 
                                                                        quiet = True)
            # get the dwell time map: the map of the time spent on each pixel in the local frame
            dwell_time_map = orientation.get_dwell_map(response = response_path)
            
            # convolve the response with the dwell_time_map to get the point source response
            with FullDetectorResponse.open(response_path) as response:
                psr = response.get_point_source_response(dwell_time_map)

        elif cds_frame == "galactic":
            
            psr = FastTSMap.get_expectation_in_galactic(hypothesis_coord = hypothesis_coord, response_path = response_path, spectrum = spectrum)
            
        else:
            raise ValueError("The point source response must be calculated in the local and galactic frame. Others are not supported (yet)!")
            
        # convolve the point source reponse with the spectrum to get the expected counts
        expectation = psr.get_expectation(spectrum)

        # slice energy channals and project it to CDS
        ei_cds_array = FastTSMap.get_cds_array(expectation, energy_channel)
        
        return ei_cds_array
    
    @staticmethod
    def fast_ts_fit(hypothesis_coord, 
                    energy_channel, data_cds_array, bkg_model_cds_array, 
                    orientation, response_path, spectrum, cds_frame,
                    ts_nside, ts_scheme):
        
        start_fast_ts_fit = time.time()
        
        # get the pix number of the ts map
        data_array = np.zeros(hp.nside2npix(ts_nside))
        ts_temp = HealpixMap(data = data_array, scheme = ts_scheme, coordsys = "galactic")
        pix = ts_temp.ang2pix(hypothesis_coord)
        
        # get the expected counts in the flattened cds array
        start_ei_cds_array = time.time()
        ei_cds_array = FastTSMap.get_ei_cds_array(hypothesis_coord = hypothesis_coord, cds_frame = cds_frame,
                                                  energy_channel = energy_channel, orientation = orientation, 
                                                  response_path = response_path, spectrum = spectrum)
        end_ei_cds_array = time.time()
        time_ei_cds_array = end_ei_cds_array - start_ei_cds_array
        
        # start the fit
        start_fit = time.time()
        fit = fnf(max_iter=1000)
        result = fit.solve(data_cds_array, bkg_model_cds_array, ei_cds_array)
        end_fit = time.time()
        time_fit = end_fit - start_fit

        end_fast_ts_fit = time.time()
        time_fast_ts_fit = end_fast_ts_fit - start_fast_ts_fit
        
        return [pix, result[0], result[1], result[2], result[3], result[4], time_ei_cds_array, time_fit, time_fast_ts_fit]

        
    def parallel_ts_fit(self, hypothesis_coords, energy_channel, spectrum, ts_scheme = "RING", start_method = "fork", cpu_cores = None):
        
        """
        Perform parallel computation on all the hypothesis coordinates.
        
        Parameters
        ----------
        hypothesis_coords: list; a list of the hypothesis coordinates
        energy_channel: list; the energy channel you want to use: [lower_channel, upper_channel]
        spectrum: astromodels; the model to be placed at the hypothesis coordinates
        ts_scheme: str; "RING" or "NESTED"
        
        Returns
        -------
        ts_values
        """
        
        # decide the ts_nside from the list of hypothesis coordinates
        ts_nside = hp.npix2nside(len(hypothesis_coords))
        
        # get the flattened data_cds_array
        data_cds_array = FastTSMap.get_cds_array(self._data, energy_channel).flatten()
        bkg_model_cds_array = FastTSMap.get_cds_array(self._bkg_model, energy_channel).flatten()
        
        if (data_cds_array[bkg_model_cds_array ==0]!=0).sum() != 0:
            #raise ValueError("You have data!=0 but bkg=0, check your inputs!")
            # let's try to set the data bin to zero when the corresponding bkg bin isn't zero.
            # Need further investigate, why bkg = 0 but data!=0 happens? ==> it's more like an issue related to simulated data instead of code
            # This first happened in GRB fitting, but got fixed somehow <== I now understand it's caused by using different PsiChi binning in the same fit
            # But it also happened to Crab while the PsiChi binning are both galactic for Crab and the Albedo, why???? ?_?
            data_cds_array[bkg_model_cds_array == 0] =0
            
        
        # set up the number of cores to use for the parallel computation
        total_cores = multiprocessing.cpu_count()
        if cpu_cores == None or cpu_cores >= total_cores:
            # if you don't specify the number of cpu cores to use or the specified number of cpu cores is the same as the total number of cores you have
            # it will use the [total_cores - 1] number of cores to run the parallel computation.
            cores = total_cores - 1
        else:
            cores = cpu_cores
            
        start = time.time()
        multiprocessing.set_start_method(start_method, force = True)
        pool = multiprocessing.Pool(processes = cores)
        results = pool.starmap(FastTSMap.fast_ts_fit, product(hypothesis_coords, [energy_channel], [data_cds_array], [bkg_model_cds_array], 
                                                             [self._orientation], [self._response_path], [spectrum], [self._cds_frame], 
                                                             [ts_nside], [ts_scheme]))
            
        pool.close()
        pool.join()
        
        end = time.time()
        
        elapsed_seconds = end - start
        elapsed_minutes = elapsed_seconds/60
        print(f"The time used for the parallel TS map computation is {elapsed_minutes} minutes")
        
        results = np.array(results)
        self.result_array = results
        
        return results

    @staticmethod
    def _plot_ts(result_array, skycoord = None, containment = 0.9):

        """
        Plot the containment region of the TS map.

        Parameters
        ----------
        result_array: the result array from parallel ts fit
        skycoord: the true location of the source
        containment: None or float; the containment level of the source. If None, it will plot raw TS values

        Returns
        -------
        None
        """


        if skycoord != None:
            lon = skycoord.l.deg
            lat = skycoord.b.deg
        
        # sort the array by the pixel number
        result_array = result_array[result_array[:, 0].argsort()]

        # get the ts value colum
        m_ts = result_array[:,1]

        # plot the ts map with containment region
        if containment != None:
            critical = FastTSMap.get_chi_critical_value(containment = containment)
            percentage = containment*100
            max_ts = np.max(m_ts[:])
            min_ts = np.min(m_ts[:])        
            hp.mollview(m_ts[:], max = max_ts, min = max_ts-critical, title = f"Containment {percentage}%") 
        elif containment == None:
            hp.mollview(m_ts[:]) 
            
        
        if skycoord != None:
            hp.projtext(lon, lat, "x", lonlat=True, coord = "G", label = f"True location at l={lon}, b={lat}", color = "fuchsia");
        #hp.projtext(40, -17, "True Location", lonlat=True, coord = "G", label = "True location at l=51, b=-17", color = "fuchsia")
        hp.projtext(0, 0, "o", lonlat=True, coord = "G", color = "red");
        hp.projtext(350, 0, "(l=0, b=0)", lonlat=True, coord = "G", color = "red");

        return

    def plot_ts(self, skycoord = None, result_array = None, containment = None):

        if result_array == None:
            result_array = self.result_array

        FastTSMap._plot_ts(result_array = result_array, skycoord = skycoord, containment = containment)

        return

    @staticmethod
    def get_chi_critical_value(containment = 0.90):

        return scipy.stats.chi2.ppf(containment, df=2)
    
        
    