from threeML.utils.time_series.binned_spectrum_series import BinnedSpectrumSeries
from threeML.utils.spectrum.binned_spectrum import BinnedSpectrumWithDispersion
from threeML.utils.data_builders.time_series_builder import TimeSeriesBuilder
from threeML.utils.spectrum.binned_spectrum import BinnedSpectrum
from threeML.utils.spectrum.binned_spectrum_set import BinnedSpectrumSet
from threeML.utils.time_interval import TimeIntervalSet
from threeML.utils.OGIP.response import OGIPResponse
from cosipy.spacecraftfile import SpacecraftFile
from cosipy import BinnedData
import numpy as np
from astropy.time import Time
from astropy.coordinates import SkyCoord



class COSIGRBData:
    def __init__(self, time_energy_counts, time_bins, energy_bins, deadtime=None):
        """
        Initialize COSI GRB data.
        
        :param time_energy_counts: 2D array of counts (time, energy)
        :param time_bins: Array of time bin edges
        :param energy_bins: Array of energy bin edges
        :param deadtime: Array of deadtime for each time bin (optional)
        """
        self.counts = time_energy_counts
        self.time_bins = time_bins
        self.energy_bins = energy_bins
        self.deadtime = deadtime if deadtime is not None else np.zeros(len(time_bins) - 1)
        
        self.times = (time_bins[:-1] + time_bins[1:]) / 2
        self.energies = (energy_bins[:-1] + energy_bins[1:]) / 2

        
    @property
    def n_channels(self):
        return len(self.energy_bins) - 1
    
    @property
    def time_start(self):
        return self.time_bins[0]
    
    @property
    def time_stop(self):
        return self.time_bins[-1]
    
    @property
    def livetime(self):
        bin_widths = np.diff(self.time_bins)
        return bin_widths - self.deadtime

    @property
    def response_file(self):
        """
        :returns: Path to the response file
        """
        return self._response_file

class TimeSeriesBuilderCOSI(TimeSeriesBuilder):
    def __init__(
        self,
        name,
        cosi_data,
        response_file,
        arf_file = None,
        l=None,
        b=None,
        ori_file=None,
        poly_order=-1,
        verbose=True,
        restore_poly_fit=None,
    ):
        """
        Initialize TimeSeriesBuilderCOSI.
        
        :param name: Name of the time series
        :param cosi_data: COSIGRBData object (.hdf5 containing signal + background)
        :param response_file: path to response file (either a .hdf5 file or .rsp file)
        :param l: Galactic longitude (optional if response_file is OGIP compatible)
        :param b: Galactic latitude (optional if response_file is OGIP compatible)
        :param ori_file: Path to orientation file (optional if response is OGIP compatible)
        :param poly_order: Polynomial order for background fitting
        :param verbose: Verbosity flag
        :param restore_poly_fit: File to restore background fit from
        """
        
        
        if (response_file.endswith(".rmf")):
            if (arf_file == None):
                response = OGIPResponse(rsp_file = response_file, arf_file= response_file[:-3] + "arf")
            else:
                response = OGIPResponse(rsp_file = response_file, arf_file= arf_file)
        else:
            if l is None or b is None:
                raise ValueError("Galactic coordinates (l, b) are required when response_file is not OGIP compatible")
            if ori_file is None:
                raise ValueError("Orientation file is required when response_file is not OGIP compatible")
            response = self.create_ogip_response(name, response_file, cosi_data.time_bins, l, b, ori_file)


        reference_time = cosi_data.time_bins[0]
        intervals = [TimeIntervalSet.INTERVAL_TYPE(start, stop) 
                     for start, stop in zip(cosi_data.time_bins[:-1], cosi_data.time_bins[1:])]
        time_intervals = TimeIntervalSet(intervals)

         # Create a list of BinnedSpectrum objects
        binned_spectrum_list = []
        for i in range(len(cosi_data.times)):
            binned_spectrum = BinnedSpectrum(
                counts=cosi_data.counts[i],
                exposure=cosi_data.livetime[i],
                ebounds=cosi_data.energy_bins,
                quality=None,  # Add quality array if available
                mission='COSI',
                instrument=name,
                is_poisson=True  # Assuming Poisson statistics
            )
            binned_spectrum_list.append(binned_spectrum)

        # Create BinnedSpectrumSet
        binned_spectrum_set = BinnedSpectrumSet(
            binned_spectrum_list=binned_spectrum_list,
            time_intervals=time_intervals,
            reference_time=reference_time
        )

        
        
        # Create a BinnedSpectrumSeries object
        binned_spectrum_series = BinnedSpectrumSeries(
            binned_spectrum_set,
            first_channel=0,
            mission='COSI',
            instrument=name,
            verbose=verbose
        )
        
        super(TimeSeriesBuilderCOSI, self).__init__(
            name,
            binned_spectrum_series,
            response=response,
            poly_order=poly_order,
            unbinned=False,
            verbose=verbose,
            restore_poly_fit=restore_poly_fit,
            container_type=BinnedSpectrumWithDispersion
        )
        
        self._cosi_data = cosi_data
        self._response_file = response_file

    @staticmethod
    def create_ogip_response(name, response_file, time_bins, l, b, ori_file):
        ori = SpacecraftFile.parse_from_file(ori_file)
        tmin = Time(time_bins[0],format = 'unix')
        tmax = Time(time_bins[-1],format = 'unix')

        sc_orientation = ori.source_interval(tmin, tmax)
        coord = SkyCoord( l = l, b=b, frame='galactic', unit='deg')
        sc_orientation.get_target_in_sc_frame(target_name = name, target_coord=coord)

        dwell_map = sc_orientation.get_dwell_map(response=response_file, save=False)
        psr_rsp = sc_orientation.get_psr_rsp(response=response_file)

        rmf = sc_orientation.get_rmf(out_name=name)
        arf = sc_orientation.get_arf(out_name=name)

        return OGIPResponse(rsp_file = name + '.rmf', arf_file= name + '.arf')

        
        
            
    @classmethod
    def from_cosi_grb_data(cls,
                           name,
                           yaml_file,
                           cosi_dataset,
                           response_file,
                           arf_file = None,
                           l=None,
                           b=None,
                           ori_file=None,
                           deadtime=None,
                           poly_order=-1,
                           verbose=True,
                           restore_poly_fit=None):
        """
        Create a TimeSeriesBuilderCOSI object from COSI GRB data.
        
        :param name: Name of the time series
        :param yaml_file: Path to yaml file
        :param cosi_dataset: COSIGRBData object (.hdf5 containing signal + background)
        :param response_file: Path to response file (either a .hdf5 file or .rsp file)
        :param arf_file: Path to arf file (only required when response is OGIP compatible and has a different name than the .rsp file)
        :param l: Galactic longitude of the source (optional if response_file is OGIP compatible)
        :param b: Galactic latitude of the source (optional if response_file is OGIP compatible)
        :param ori: Path to orientation file (optional if response is OGIP compatible)
        :param poly_order: Polynomial order for background fitting (optional)
        :param verbose: Verbosity flag
        :param restore_poly_fit: File to restore background fit from
        """
        
        analysis = BinnedData(yaml_file)
        analysis.load_binned_data_from_hdf5(binned_data=cosi_dataset)
        time_energy_counts = analysis.binned_data.project(['Time', 'Em']).contents.todense()
        time_bins = (analysis.binned_data.axes['Time'].edges).value
        energy_bins = (analysis.binned_data.axes['Em'].edges).value
        
        cosi_data = COSIGRBData(time_energy_counts, time_bins, energy_bins, deadtime)
        
        return cls(
            name,
            cosi_data,
            response_file,
            arf_file,
            l,
            b,
            ori_file,
            poly_order=poly_order,
            verbose=verbose,
            restore_poly_fit=restore_poly_fit
        )

   