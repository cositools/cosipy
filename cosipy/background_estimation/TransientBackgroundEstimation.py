from histpy import Histogram
import numpy as np
import matplotlib.pyplot as plt
import itertools
import copy

class TransientBackgroundEstimation:
    
    """
    Estimate transient background using a Histogram that
    includes a 'Time' axis.
    
    The estimated background preserves all axes instead of
    projecting to `Time` axis.
    
    YS: now it can support one background windows before
    and after the burst. More than one window on each siede 
    needs further tests.
    
    YS: now it assumes only one burst per data.
    
    YS: The burst window can be recoginized by Bayesian blocks
    automatically in the future. I have the code but I need to 
    do more tests before adding it.
    
    """
    
    def __init__(self, data):
        
        """
        Initialize the instance if a transient background.
        
        Parameters
        ----------
        data : histpy.histogram.Histogram
            The histogram containing both signal and background components.
            Must include an axis labeled 'Time'.
        """
        
        # check if input is histogram object
        if not isinstance(data, Histogram):
            raise TypeError(
                "`data` must be a histpy.histogram.Histogram, \
                got {type(data).__name__}."
            )
            
        self._data = data
                    
        self._axes_labels = list(self._data.axes.labels)
        
        # check if the data has Time axis
        if not "Time" in self._axes_labels:
            raise ValueError(
                "A 'Time' axis must be present for backrgound estimation \
                of transients."
            )
        
        # check if the Time axis is at index 0
        time_axis_index = self._axes_labels.index("Time")
        if time_axis_index != 0:
            raise ValueError(
                f"The `Time` axis must have index of 0, but you have \
                index of {time_axis_index}."
            )
                
        # All the time tags
        self._timetags = self._data.axes["Time"].edges.value
        
        # max and min time tages
        self._timetag_min = self._timetags.min()
        self._timetag_max = self._timetags.max()
        
        #full data duration
        self._full_duration = self._timetag_max - self._timetag_min
        
        # initialize the background window
        # the background window is defined by the start and end time tags
        # such as [[1,2], [3,4], [5,6]]
        self._bkg_windows = []
        
        # initialize the burst window
        # I use plural windows to avoid API changes after supporting multiple windows
        self._burst_windows = []
        
        
    def slice_by_timetags(self, start, end):
        
        """
        It slice the data by timetags.
        
        Parameters
        ----------
        timetag_start : float
            The start of the time slicing
        timetag_end : float
            The end of the time slicing
            
        Returns
        -------
        sliced_data : histpy.histogram.Histogram
            The sliced data by time tags.
        """
        
        if start >= end:
            raise ValueError(
                "`start` must be smaller than `end`."
            )
            
        if start < self._timetag_min or end > self._timetag_max:
            raise ValueError(
                f"`start` and `end` must be within \
                [{self._timetag_min}, {self._timetag_max}]."
                )
            
        # find the indices of the starting and ending time tags
        start_idx = np.searchsorted(self._timetags, start, side='right')
        end_idx = np.searchsorted(self._timetags, end, side='left')
        
        # slice data by time tag indices
        sliced_data = self._data.slice[start_idx:end_idx,:]
        
        return sliced_data
    
    def add_bkg_windows(self, *timetags):
        
        """
        Define bkg windows using their timetags
        
        Parameters
        ----------
        timetages : list
            The star and end time tags of the background windows
        """
        
        for i in timetags:
            
            self._bkg_windows += [i]
        
        return
    
    @property
    def bkg_windows(self):
        return self._bkg_windows
    
    @property
    def bkg_durations(self):
        return [j-i for (i,j) in self._bkg_windows]
    
    def add_burst_windows(self, *timetags):
        
        if len(timetags) > 1:
            
            raise ValueError(
                "Only one burst windows is supported now"
            )
            
        for i in timetags:
            
            if i[0] >= i[1]:
                
                raise ValueError(
                    "`start` must be smaller than `end`."
                )
            
            self._burst_windows += [i]
        
        return
    
    @property
    def burst_durations(self):
        return [j-i for (i,j) in self._burst_windows]

    @property
    def burst_windows(self): 
        # I use plural windows to avoid API changes after supporting multiple windows
        return self._burst_windows
    

    def plot_lightcurve(self, burst_windows = False, bkg_windows = False,
                        plot_limits = None):
        
        colors = itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

        fig, ax = plt.subplots(figsize=(16,8), sharex=False, nrows=1) ##sharex=True

        plt.tick_params(axis="both", which="both", labelleft=True, labelright=False, labelbottom=True, labeltop=False, labelsize = 15)

        ax.step(self._timetags[:-1], self._data.project("Time")[:].todense(), where='post', label='Light curve')
        
        #ax.ticklabel_format(axis='x', style='plain', useOffset=True, useMathText=True)
        
        if plot_limits is not None:
            
            ax.set_xlim(plot_limits)
            
        if burst_windows:
            for idx, (t0, t1) in enumerate(self._burst_windows):
                ax.axvspan(xmin=t0, xmax=t1, alpha=0.3, color=next(colors), label=f"Burst window")
            
        if bkg_windows:
            for idx, (t0, t1) in enumerate(self._bkg_windows):  
                ax.axvspan(xmin=t0, xmax=t1, alpha=0.3, color=next(colors), label=f"Background window {idx}")

        ax.set_ylabel("Counts [ph]")
        ax.set_xlabel("Time [s]")
        
        plt.legend(fontsize = 15)
        
        return
    
    
    def make_background_model(self, scaling = "duration", save_path = None):
        
        """
        
        Parameters
        ----------
        scaling : str
            The scaling method: duration or fitting.
            `duration` scales the background counts by the duration ratio  
            between the burst and background windows. 
            `fitting` will fit the background windows before and after the
            burst to get the fitted background counts during the burst. It
            will be supported later.
            
        """
        
        if not self._bkg_windows:
            raise ValueError(
                "Please define background windows first."
            )
            
        if not self._burst_windows:
            raise ValueError(
                "Please define burst window first."
            )
            
        bkgs = []
        
        for (t0, t1) in self._bkg_windows:
            sliced_bkg = self.slice_by_timetags(t0, t1)
            bkgs += [sliced_bkg]
            
        # add all backgrounds
        total_bkg = copy.deepcopy(bkgs[0]).project_out("Time")
        for i in bkgs[1:]:
            total_bkg = total_bkg + i.project_out("Time")
            
        unit_bkg = total_bkg/(np.sum(self.bkg_durations))
            
        if scaling == "duration":
            bkg_model = self.burst_durations[0]*unit_bkg
        
        elif scaling == "fitting":
            raise NotImplementedError(
                "The scaling by fitting the background before and after the burst \
                is not implemented yet!"
            )
        else:
            raise ValueError(
                "Please use `duration` or `fitting` for scaling method"
            )
            
        if save_path is not None:
            bkg_model.write(save_path)
        
        return bkg_model
