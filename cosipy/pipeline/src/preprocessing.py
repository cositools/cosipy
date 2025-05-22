from cosipy.pipeline.src.io import load_binned_data
from astropy.time import Time


import numpy as np

def tslice_binned_data(data,tmin,tmax):
    """Slice a binned dataset in time"""
    idx_tmin = np.where(data.axes['Time'].edges.value >= tmin.value)[0][0]
    idx_tmax_all = np.where(data.axes['Time'].edges.value <= tmax.value)
    y = len(idx_tmax_all[0]) - 1
    idx_tmax = np.where(data.axes['Time'].edges.value <= tmax.value)[0][y]
    tsliced_data = data.slice[{'Time': slice(idx_tmin, idx_tmax)}]
    return tsliced_data



def tslice_ori(ori,tmin,tmax):
    """
    Slices time for the orientation file
    """
    #ori_min = Time(tmin,format = 'unix')
    #ori_max = Time(tmax,format = 'unix')
    tsliced_ori = ori.source_interval(tmin, tmax)
    return tsliced_ori