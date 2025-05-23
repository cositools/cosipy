from cosipy import BinnedData
from cosipy.spacecraftfile import SpacecraftFile

from astropy.time import Time


def load_binned_data(yaml_path,data_path):
    """
    Loads a BinnedData instance from an hdf5
    """
    data = BinnedData(yaml_path)
    data.load_binned_data_from_hdf5(data_path)
    return data.binned_data

def load_ori(ori_path):
    """
    Loads an orientation file
    """
    ori = SpacecraftFile.parse_from_file(ori_path)
    return ori

def tslice_binned_data(data, tmin:Time, tmax:Time):
    """Slice a binned dataset in time"""
    idx_tmin = np.where(data.axes['Time'].edges.value >= tmin.unix)[0][0]
    idx_tmax_all = np.where(data.axes['Time'].edges.value <= tmax.unix)
    y = len(idx_tmax_all[0]) - 1
    idx_tmax = np.where(data.axes['Time'].edges.value <= tmax.unix)[0][y]
    tsliced_data = data.slice[{'Time': slice(idx_tmin, idx_tmax)}]
    return tsliced_data




def tslice_ori(ori, tmin:Time, tmax:Time):
    """
    Slices time for the orientation file
    """
    tsliced_ori = ori.source_interval(tmin, tmax)
    return tsliced_ori
