from cosipy.pipeline.src.io import load_binned_data
from astropy.time import Time

import numpy as np
import matplotlib.pyplot as plt

def tslice_binned_data(data, tmin, tmax):
    """Slice a binned dataset in time"""
    idx_tmin = np.where(data.axes['Time'].edges.value >= tmin)[0][0]
    idx_tmax_all = np.where(data.axes['Time'].edges.value <= tmax)
    y = len(idx_tmax_all[0]) - 1
    idx_tmax = np.where(data.axes['Time'].edges.value <= tmax)[0][y]
    tsliced_data = data.slice[{'Time': slice(idx_tmin, idx_tmax)}]
    return tsliced_data


def make_eqsize_tslices(tstart, tstop, nbins):
    """
    Takes a time interval and a number of desired bins and computes the edges
    of equal time slices
    Returns:
        tmins, tmaxs : edges
    """
    dt = (tstop - tstart) / nbins
    tmins = np.array([], dtype=float)  # Initialize as empty numpy arrays
    tmaxs = np.array([], dtype=float)  # Initialize as empty numpy arrays
    #
    for i in range(nbins):
        tmin = tstart + i * dt
        tmax = tmin + dt
        tmins = np.append(tmins, tmin)
        tmaxs = np.append(tmaxs, tmax)
    return tmins, tmaxs


def make_minsn_tslices(tstart, tstop, yaml_path, data_path, min_sn, max_slices):
    """
    Makes time slices requiring minimum total S/N

    """
    step = (tstop - tstart) / max_slices
    tmins = np.array([], dtype=float)
    tmaxs = np.array([], dtype=float)
    #
    data = load_binned_data(yaml_path, data_path)
    #
    tmax = tstart
    for i in range(max_slices):
        tmin = tmax
        tmax_i = tstart + (i + 1) * step
        #
        data_sliced = tslice_binned_data(data, tmin, tmax_i)
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


def tslice_ori(ori, tmin, tmax):
    """
    Slices time for the orientation file
    """
    ori_min = Time(tmin, format='unix')
    ori_max = Time(tmax, format='unix')
    tsliced_ori = ori.source_interval(ori_min, ori_max)
    return tsliced_ori


##### PLOTTING
def get_ene(data):
    """
    computes the central energy and the half-width of each bin
    (for plotting purposes)
    """
    #
    binned_energy_edges = data.axes['Em'].edges.value
    #
    ene = np.array([])
    e_ene = np.array([])
    #
    for j in range(len(binned_energy_edges) - 1):
        delta_ene = (binned_energy_edges[j + 1] - binned_energy_edges[j])
        ene = np.append(ene, (binned_energy_edges[j] + 0.5 * delta_ene))
        e_ene = np.append(e_ene, 0.5 * delta_ene)
        #
    return (ene, e_ene)


def get_counts_ene(data):
    """
    Projects the counts of the 5D histogram onto the E axis
    Computes Poissonian Error for energy bins
    (for plotting purposes)
    """
    cts = data.project('Em').todense().contents
    e_cts = np.sqrt(cts)
    return (cts, e_cts)


def get_fit_residuals(cts, e_cts, cts_exp):
    """
    computes the residuals of a fit
    Args:
        cts: np.array. counts
        e_cts:np.array. count errors
        cts_exp:np.array. expected counts
    (for plotting purposes)
    """
    resid = (cts - cts_exp) / e_cts
    e_resid = np.abs((e_cts / cts) * resid)
    return (resid, e_resid)


def plot_fit(sou, cts_exp, figname):
    # Save a plot of the current fit.
    #
    ene, e_ene = get_ene(sou)
    cts, e_cts = get_counts_ene(sou)
    #
    resid, e_resid = get_fit_residuals(cts, e_cts, cts_exp)
    #
    cm = 1 / 2.54
    #
    fig, ax = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [0.7, 0.3]})
    fig.tight_layout()
    #
    ax[0].scatter(ene, cts, color='purple')
    ax[0].errorbar(ene, cts, xerr=e_ene, yerr=e_cts, color='purple', fmt='o', capsize=0)
    #
    ax[0].step(ene, cts_exp, where='mid', color='red', label="Best fit convolved with response plus background")
    #
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_ylabel("Counts")
    ax[0].legend()
    #
    #
    #
    ax[1].errorbar(ene, resid, xerr=e_ene, yerr=e_resid, color='purple', fmt='o', capsize=0)
    ax[1].errorbar(ene, resid, xerr=e_ene, yerr=e_resid, color='purple', fmt='o', capsize=0)
    ax[1].axhline(y=0, color='black')
    ax[1].set_xscale("log")
    ax[1].set_yscale("linear")
    ax[1].set_ylabel("Obs-Model/Err")
    ax[1].set_xlabel("Energy (keV)")
    #
    # plt.show()
    plt.savefig(figname)
    return ()