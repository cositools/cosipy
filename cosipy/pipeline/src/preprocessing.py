from cosipy.pipeline.src.io import load_binned_data
from astropy.time import Time
from astropy.io.misc import yaml
from cosipy.response import FullDetectorResponse
from cosipy import BinnedData

import numpy as np


def write_yaml(udata_path, ori_path, resp_path, dt, tmin, tmax, bin_yaml_path):
    """
    Write a .yaml file that contains the information to bin a dataset according to a response file.

    Parameters
    ----------
    udata_path : str
        Path to the unbinned data file to use. Input file is either
        .fits or .hdf5.
    ori_path : str
        Path to the orientation file.
    resp_path : str
        Path to the response file.
    dt :  float
        Lengths of the time bins (s).
    tmin:  float
        Minimum of time axis.
    tmax:   float
        Maximum of the time axis.
    bin_yaml_path: str
        Path to .yaml file to write.

    """

    with FullDetectorResponse.open(resp_path) as response:
        bin_dict = {
            "data_file": udata_path,
            "ori_file": ori_path,
            "unbinned_output": udata_path[-4:],
            "time_bins": dt,
            "energy_bins": list(response.axes["Em"].edges.value),
            "phi_pix_size": int(180 / (response.axes["Phi"].nbins)),
            "nside": response.nside,
            "scheme": response.scheme,
            "tmin": tmin,
            "tmax": tmax
        }
        with open(bin_yaml_path, 'w') as outfile:
            yaml.dump(bin_dict, outfile, default_flow_style=False, sort_keys=False)
    return ()


def get_binned_data(yaml_path, udata_path, bdata_name, psichi_coo):
    """
    Creates a binned dataset from a .yaml file and an unbinned data file.

    Parameters
    ----------
    yaml_path : str
        Path to the .yaml file that contains the binning information.
    udata_path : str
        Path to the unbinned data file to use. Input file is either .fits or .hdf5.
    bdata_name: str
        Name of the binned dataset
    """
    data = BinnedData(yaml_path)
    data.get_binned_data(unbinned_data=udata_path, output_name=bdata_name, psichi_binning=psichi_coo,
                         make_binning_plots=False)
    return ()

