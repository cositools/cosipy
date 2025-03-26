from cosipy.pipeline.src.io import load_binned_data
from astropy.time import Time


import numpy as np

def tslice_binned_data(data,tmin,tmax):
    """Slice a binned dataset in time"""
    idx_tmin = np.where(data.axes['Time'].edges.value >= tmin)[0][0]
    idx_tmax_all = np.where(data.axes['Time'].edges.value <= tmax)
    y = len(idx_tmax_all[0]) - 1
    idx_tmax = np.where(data.axes['Time'].edges.value <= tmax)[0][y]
    tsliced_data = data.slice[{'Time': slice(idx_tmin, idx_tmax)}]
    return tsliced_data

def make_eqsize_tslices(tstart,tstop,nbins):
    """
    Takes a time interval and a number of desired bins and computes the edges
    of equal time slices
    Returns:
        tmins, tmaxs : edges
    """
    dt=(tstop-tstart)/nbins
    tmins = np.array([], dtype=float)# Initialize as empty numpy arrays
    tmaxs=np.array([], dtype=float)# Initialize as empty numpy arrays
    #
    for i in range(nbins):
        tmin=tstart+i*dt
        tmax=tmin+dt
        tmins = np.append(tmins, tmin)
        tmaxs=np.append(tmaxs, tmax)
    return tmins,tmaxs

def make_minsn_tslices(tstart, tstop, yaml_path, data_path, min_sn, max_slices):
    """
    Makes time slices requiring minimum total S/N

    """
    step = (tstop - tstart) / max_slices
    tmins = np.array([], dtype=float)
    tmaxs = np.array([], dtype=float)
    #
    data=load_binned_data(yaml_path,data_path)
    #
    tmax = tstart
    for i in range(max_slices):
        tmin = tmax
        tmax_i = tstart + (i + 1) * step
        #
        data_sliced=tslice_binned_data(data,tmin,tmax_i)
        signal = np.sum(data_sliced.todense().contents)
        noise = np.sqrt(signal)
        #
        sn = signal / noise
        if (sn >= min_sn and tmax_i < tstop):
            tmins = np.append(tmins, tmin)
            tmax = tmax_i
            tmaxs = np.append(tmaxs, tmax)
        elif (tmax_i == tstop):
            tmaxs[-1] = tmax_i
    return tmins, tmaxs

def tslice_ori(ori,tmin,tmax):
    """
    Slices time for the orientation file
    """
    ori_min = Time(tmin,format = 'unix')
    ori_max = Time(tmax,format = 'unix')
    tsliced_ori = ori.source_interval(ori_min, ori_max)
    return tsliced_ori