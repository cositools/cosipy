from cosipy import BinnedData
from cosipy.spacecraftfile import SpacecraftHistory
from astropy.time import Time
import numpy as np


def load_binned_data(yaml_path, data_path):
    """
    Return a binned histogram from an hdf5 file.

    Parameters
    ----------
    yaml_path : str
        Path to a yaml file containing the binning information.
    data_path: str
        Path to a binned hdf5 file.
    Returns
    -------
    binned_data : histpy:Histogram
        Data is binned in four axes: time, measured energy,
        Compton scattering angle (phi), and scattering direction (PsiChi).

    """

    data = BinnedData(yaml_path)
    data.load_binned_data_from_hdf5(data_path)
    return data.binned_data


def load_ori(ori_path):
    """
    Return a SpacecraftFile Object from an orientation file.

    Parameters
    ----------
    ori_path : str
        Path to an orientation file.

    Returns
    -------
    ori: cosipy.spacecraftfile.SpacecraftFile.SpacecraftFile
        The SpacecraftFile Object.
    """
    ori = SpacecraftHistory.open(ori_path)
    return ori


def tslice_binned_data(data, tmin: Time, tmax: Time):
    """
    Slice a Histogram in a time interval.

    Parameters
    ----------
    data : histpy:Histogram
        The Hystogram to be sliced
    tmin : astropy.Time
        Minimum of the time interval
    tmax: astropy.Time
        Maximum  of the time interval

    Returns
    -------
    tsliced_data : histpy:Histogram
        Histogram sliced in the time interval tmin, tmax.
    """
    idx_tmin = np.where(data.axes['Time'].edges.value >= tmin.unix)[0][0]
    idx_tmax_all = np.where(data.axes['Time'].edges.value <= tmax.unix)
    y = len(idx_tmax_all[0]) - 1
    idx_tmax = np.where(data.axes['Time'].edges.value <= tmax.unix)[0][y]
    tsliced_data = data.slice[{'Time': slice(idx_tmin, idx_tmax)}]
    return tsliced_data
