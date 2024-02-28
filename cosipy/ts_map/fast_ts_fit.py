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
import os
import psutil
import gc

class FastTSMap():
    
    def __init__(self, data, bkg_model, response_path, orientation = None, cds_frame = "local", scheme = "RING"):
        
        """
        Initialize the instance if a TS map fit.
        
        Parameters
        ----------
        data : histpy.Histogram
            Observed data, which includes counts from both signal and background.
        bkg_model : histpy.Histogram
            Background model, which includes the background counts to model the background in the observed data.
        response_path : str or pathlib.Path
            The path to the response file.
        orientation : cosipy.SpacecraftFile or NoneType, optional
            The orientation of the spacecraft when data are collected (the default is None, which implies the orientation file is not needed).
        cds_frame : str, optional
            "local" or "galactic", it's the Compton data space (CDS) frame of the data, bkg_model and the response. In other words, they should have the same cds frame (the default is "local", which implied that a local frame that attached to the spacecraft).
        scheme : str, optional
            The scheme of the CDS of data (the default is "RING", which implies a "RING" scheme of the data).
        
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
        Slice one or more bins along first axis of the `hist`.
        
        Parameters
        ----------
        hist : histpy.Histogram
            The histogram object to be sliced.
        channel_start : int
            The start of the slice (inclusive)
        channel_stop : int
            The stop of the slice (exclusive)
        
        Returns
        -------
        sliced_hist : histpy.Histogram
            The sliced histogram.
        """
        
        sliced_hist = hist.slice[channel_start:channel_stop,:]
        
        return sliced_hist
    
    @staticmethod
    def get_hypothesis_coords(nside, scheme = "RING", coordsys = "galactic"):
        
        """
        Get a list of hypothesis coordinates.
        
        Parameters
        ----------
        nside : int
            The nside of the map.
        scheme : str, optional
            The scheme of the map where the hypothesis coordinates are generated (the default is "RING", which implies the "RING" scheme is used to get the hypothesis coordinates).
        coordsys : str, optional
            The coordinate system used in the map where the hypothesis coordinates are generated (the default is "galactic", which implies the galactic coordinates system is used).
        
        Returns
        -------
        hypothesis_coords : list
            The list of the hypothesis coordinates at the center of each pixel
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
        hist : histpy.Histogram
            The input Histogram.
        energy_channel : list
            The format is `[lower_channel, upper_chanel]`. The lower_channel is inclusive while the upper_channel is exclusive.

        Returns
        -------
        cds_array : numpy.ndarray
            The flattended Compton data space (CDS) array.
        
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
        del hist
        del hist_cds_sliced
        del hist_cds
        gc.collect()
        
        return cds_array
    
    @staticmethod
    def get_psr_in_galactic(hypothesis_coord, response_path, spectrum):
        
        """
        Get the point source response (psr) in galactic. Please be aware that you must use a galactic response!
        To do: to make the weight parameter not hardcoded
        
        Parameters
        ----------
        hypothesis_coord : astropy.coordinates.SkyCoord
            The hypothesis coordinate.
        response_path : str or path.lib.Path
            The path to the response.
        spectrum : astromodel spectrum
            The spectrum of the source to be placed at the hypothesis coordinate.
        
        Returns
        -------
        psr : histpy.Histogram
            The point source response of the spectrum at the hypothesis coordinate.
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
        hypothesis_coord : astropy.coordinates.SkyCoord
            The hypothesis coordinate. 
        energy_channel : list
            The format is `[lower_channel, upper_chanel]`. The lower_channel is inclusive while the upper_channel is exclusive.
        response_path : str or pathlib.Path
            The path to the response file.
        spectrum : astromodel spectrum
            The spectrum of the source.
        cds_frame : str, optional
            "local" or "galactic", it's the Compton data space (CDS) frame of the data, bkg_model and the response. In other words, they should have the same cds frame.
        orientation : cosipy.SpacecraftFile or NoneType, optional
            The orientation of the spacecraft when data are collected (the default is None, which implies the orientation file is not needed).
        
        Returns
        -------
        cds_array : numpy.ndarray
            The flattended Compton data space (CDS) array.
        """
                         
        # check inputs, will complete later
        
        # the local and galactic frame works very differently, so we need to compuate the point source response (psr) accordingly
        #time_cds_start = time.time()
        if cds_frame == "local":
            
            if orientation == None:
                raise TypeError("The when the data are binned in local frame, orientation must be provided to compute the expected counts.")

            #time_coord_convert_start = time.time()
            # convert the hypothesis coord to the local frame (Spacecraft frame)
            hypothesis_in_sc_frame = orientation.get_target_in_sc_frame(target_name = "Hypothesis", 
                                                                        target_coord = hypothesis_coord, 
                                                                        quiet = True)
            #time_coord_convert_end = time.time()
            #time_coord_convert_used = time_coord_convert_end - time_coord_convert_start
            #print(f"The time used for coordinate conversion is {time_coord_convert_used}s.")

            #time_dwell_start = time.time()
            # get the dwell time map: the map of the time spent on each pixel in the local frame
            dwell_time_map = orientation.get_dwell_map(response = response_path)
            #time_dwell_end = time.time()
            #time_dwell_used = time_dwell_end - time_dwell_start
            #print(f"The time used for dwell time map is {time_dwell_used}s.")
            
            #time_psr_start = time.time()
            # convolve the response with the dwell_time_map to get the point source response
            with FullDetectorResponse.open(response_path) as response:
                psr = response.get_point_source_response(dwell_time_map)
            #time_psr_end = time.time()
            #time_psr_used = time_psr_end - time_psr_start
            #print(f"The time used for psr is {time_psr_used}s.")

        elif cds_frame == "galactic":
            
            psr = FastTSMap.get_psr_in_galactic(hypothesis_coord = hypothesis_coord, response_path = response_path, spectrum = spectrum)
            
        else:
            raise ValueError("The point source response must be calculated in the local and galactic frame. Others are not supported (yet)!")
            
        # convolve the point source reponse with the spectrum to get the expected counts
        expectation = psr.get_expectation(spectrum)
        del psr
        gc.collect()

        # slice energy channals and project it to CDS
        ei_cds_array = FastTSMap.get_cds_array(expectation, energy_channel)
        del expectation
        gc.collect()

        #time_cds_end = time.time()
        #time_cds_used = time_cds_end - time_cds_start
        #print(f"The time used for cds is {time_cds_used}s.")
        
        return ei_cds_array
    
    @staticmethod
    def fast_ts_fit(hypothesis_coord, 
                    energy_channel, data_cds_array, bkg_model_cds_array, 
                    orientation, response_path, spectrum, cds_frame,
                    ts_nside, ts_scheme):

        """
        Perform a TS fit on a single location at `hypothesis_coord`.

        Parameters
        ----------
        hypothesis_coord : astropy.coordinates.SkyCoord
            The hypothesis coordinate. 
        energy_channel : list
            The format is `[lower_channel, upper_chanel]`. The lower_channel is inclusive while the upper_channel is exclusive.
        data_cds_array : numpy.ndarray
            The flattened Compton data space (CDS) array of the data.
        bkg_model_cds_array : numpy.ndarray
            The flattened Compton data space (CDS) array of the background model.
        orientation : cosipy.SpacecraftFile
            The orientation of the spacecraft when data are collected. 
        response_path : str or pathlib.Path
            The path to the response file.
        spectrum : astromodel spectrum
            The spectrum of the source.
        cds_frame : str
            "local" or "galactic", it's the Compton data space (CDS) frame of the data, bkg_model and the response. In other words, they should have the same cds frame .
        ts_nside : int
            The nside of the ts map.
        ts_scheme : str
            The scheme of the Ts map.

        Rsturns
        -------
        list
            The list of the resulting TS fit: [pix number, ts value, norm, norm_err, failed, iterations, time_ei_cds_array, time_fit, time_fast_ts_fit]
        """
        
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
        hypothesis_coords : list
            A list of the hypothesis coordinates
        energy_channel : list
            the energy channel you want to use: [lower_channel, upper_channel]. lower_channel is inclusive while upper_channel is exclusive.
        spectrum : astromodel spectrum
            The spectrum of the source.
        ts_scheme : str, optional
            The scheme of the TS map (the default is "RING", which implies a "RING" scheme of the TS map).
        start_method : str, optional
            The starting method of the parallel computation (the default is "fork", which implies using the fork method to start parallel computation).
        cpu_cores : int or NoneType, optional
            The number of cpu cores you wish to use for the parallel computation (the default is None, which implies using all the available number of cores -1 to perform the parallel computation).
        
        Returns
        -------
        results : numpy.ndarray
            The result of the ts fit over all the hypothesis coordinates.
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
            print(f"You have total {total_cores} CPU cores, using {cores} CPU cores for parallel computation.")
        else:
            cores = cpu_cores
            print(f"You have total {total_cores} CPU cores, using {cores} CPU cores for parallel computation.")

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
    def _plot_ts(result_array, skycoord = None, containment = None):

        """
        Plot the containment region of the TS map.

        Parameters
        ----------
        result_array : numpy.ndarray
            The result array from parallel ts fit.
        skyoord : astropy.coordinates.SkyCoord or NoneType, optional
            The true location of the source (the default is None, which implies that there are no coordiantes to be printed on the TS map).
        containment: NoneType or float, optional
            The containment level of the source (the default is None, which will plot raw TS values).

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

    def plot_ts(self, skycoord = None, containment = None):

        """
        Plot the containment region of the TS map.

        Parameters
        ----------
        skyoord : astropy.coordinates.SkyCoord or NoneType, optional
            The true location of the source (the default is None, which implies that there are no coordiantes to be printed on the TS map).
        containment: NoneType or float, optional
            The containment level of the source (the default is 0.9, which implies plot the 90% containment region).

        Returns
        -------
        None
        """


        result_array = self.result_array

        FastTSMap._plot_ts(result_array = result_array, skycoord = skycoord, containment = containment)

        return

    @staticmethod
    def get_chi_critical_value(containment = 0.90):
        
        """
        Get the critical value of the chi^2 distribution based ob the confidence level.

        Parameters
        ----------
        containment : float, optional
            The confidence level of the chi^2 distribution (the default is 0.9, which implies that the 90% containment region).

        Returns
        -------
        float
            The critical value corresponds to the confidence level.
        """

        return scipy.stats.chi2.ppf(containment, df=2)

    @staticmethod
    def show_memory_info(hint):
        pid = os.getpid()
        p = psutil.Process(pid)
    
        info = p.memory_full_info()
        memory = info.uss / 1024. / 1024
        print('{} memory used: {} MB'.format(hint, memory))
        
    