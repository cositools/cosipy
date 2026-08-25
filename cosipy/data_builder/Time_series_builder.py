from threeML.utils.time_series.binned_spectrum_series import BinnedSpectrumSeries
from threeML.utils.spectrum.binned_spectrum import BinnedSpectrumWithDispersion
from threeML.utils.spectrum.binned_spectrum import BinnedSpectrum
from threeML.utils.spectrum.binned_spectrum_set import BinnedSpectrumSet
from threeML.utils.time_interval import TimeIntervalSet
from threeML.utils.OGIP.response import OGIPResponse
from cosipy.spacecraftfile import SpacecraftFile
from cosipy import BinnedData
import numpy as np
from astropy.time import Time
from astropy.coordinates import SkyCoord


def create_ogip_response(name, response_file, time_bins, l, b, ori_file):
    ori = SpacecraftFile.parse_from_file(ori_file)
    tmin = Time(time_bins[0],format = 'unix')
    tmax = Time(time_bins[-1],format = 'unix')

    sc_orientation = ori.source_interval(tmin, tmax)
    coord = SkyCoord( l = l, b=b, frame='galactic', unit='deg')
    sc_orientation.get_target_in_sc_frame(target_name = name, target_coord=coord)

    sc_orientation.get_dwell_map(response=response_file, save=False)
    sc_orientation.get_psr_rsp(response=response_file)

    sc_orientation.get_rmf(out_name=name)
    sc_orientation.get_arf(out_name=name)

    return OGIPResponse(rsp_file = name + '.rmf', arf_file= name + '.arf')

    
def from_cosi_data(cls,
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
                    restore_poly_fit=None,
                    **kwargs):
    """
    Builds a TimeSeries object from COSI data. 
    To be used as: cosi_data = from_cosi_data(TimeSeriesBuilder, ...)
    
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
    
    # 1. Load and process the data using cosipy BinnedData
    analysis = BinnedData(yaml_file)
    analysis.load_binned_data_from_hdf5(binned_data=cosi_dataset)
    time_energy_counts = analysis.binned_data.project(['Time', 'Em']).contents.todense()
    time_bins = (analysis.binned_data.axes['Time'].edges).value
    energy_bins = (analysis.binned_data.axes['Em'].edges).value

    # Calculate exposures (livetime)
    bin_widths = np.diff(time_bins)
    if deadtime is None:
        deadtime = np.zeros(len(bin_widths))
    exposures = bin_widths - deadtime

    # 2. Setup Response
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
        response = create_ogip_response(name, response_file, time_bins, l, b, ori_file)

    # 3. Build 3ML internal structures
    intervals = [TimeIntervalSet.INTERVAL_TYPE(start, stop) 
                 for start, stop in zip(time_bins[:-1], time_bins[1:])]
    time_intervals = TimeIntervalSet(intervals)

    binned_spectrum_list = []
    for i in range(len(time_energy_counts)):
        binned_spectrum = BinnedSpectrum(
            counts=time_energy_counts[i],
            exposure=exposures[i],
            ebounds=energy_bins,
            mission='COSI',
            instrument=name,
            is_poisson=True
        )
        binned_spectrum_list.append(binned_spectrum)

    binned_spectrum_set = BinnedSpectrumSet(
        binned_spectrum_list=binned_spectrum_list,
        time_intervals=time_intervals,
        reference_time=time_bins[0]
    )

    binned_spectrum_series = BinnedSpectrumSeries(
        binned_spectrum_set,
        first_channel=0,
        mission='COSI',
        instrument=name,
        verbose=verbose
    )
    
    # cosi_data = COSIGRBData(time_energy_counts, time_bins, energy_bins, deadtime)
    
    return cls(
        name,
        binned_spectrum_series,
        response = response,
        # arf_file,
        # l,
        # b,
        # ori_file,
        poly_order=poly_order,
        unbinned=False,
        verbose=verbose,
        restore_poly_fit=restore_poly_fit,
        container_type=BinnedSpectrumWithDispersion,
        **kwargs
    )

