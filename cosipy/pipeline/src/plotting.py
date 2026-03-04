from cosipy.pipeline.src.io import load_binned_data
from astropy.time import Time

import numpy as np
import matplotlib.pyplot as plt



##### PLOTTING
def get_ene(data):
    """
    Computes the central energy and the half-width of each energy bin of a dataset
    (for plotting purposes)


    Parameters
    ----------
    data : histpy:Histogram
        The dataset to be plotted.

    Returns
    -------
    ene : numpy:array
        The central energy value of each bin.
    e_ene: numpy:array
        The half-width of each energy bin.
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
    Computes the counts and poissonian errors for each energy bin of dataset
    (for plotting purposes).

    Parameters
    ----------
    data : histpy:Histogram
        The dataset to be plotted.

    Returns
    -------
    cts : numpy:array
        The counts in each energy bin.
    e_cts: numpy:array
        The poissonian error for the counts in each energy bin.
    """
    cts = data.project('Em').todense().contents
    e_cts = np.sqrt(cts)
    return (cts, e_cts)


def get_fit_residuals(cts, e_cts, cts_exp):
    """
     Computes the residuals, in terms of Chi-squared, of a fit.
     (for plotting purposes).

     Parameters
     ----------
     cts : numpy:array
        The counts in each energy bin.
     e_cts : numpy:array
        The poissonian errors for the counts in each energy bins.
     cts_exp : numpy:array
        The counts predicted in each energy bin by the fitted model.

     Returns
     -------
     resid : numpy:array
        The fit residuals in each energy bin: (data-model)/error
     e_resid: numpy:array
        The errors of the residuals.
     """
    #
    resid = (cts - cts_exp) / e_cts
    e_resid = np.abs((e_cts / cts) * resid)
    return (resid, e_resid)


def plot_fit(sou, cts_exp, figname):
    """
     Save a plot of the fit and fit residuals.
     (for plotting purposes).

     Parameters
     ----------
     sou : histpy:Histogram
        The dataset to be plotted.
     cts_exp : numpy:array
        The counts predicted in each energy bin by the fitted model.
     figname : str
        The name of the figure to be saved.
    """
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
    ax[0].errorbar(ene, cts, xerr=e_ene, yerr=e_cts, color='purple', fmt='.', capsize=0)
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
    ax[1].errorbar(ene, resid, xerr=e_ene, yerr=e_resid, color='purple', fmt='.', capsize=0)
    ax[1].errorbar(ene, resid, xerr=e_ene, yerr=e_resid, color='purple', fmt='.', capsize=0)
    ax[1].axhline(y=0, color='black')
    ax[1].set_xscale("log")
    ax[1].set_yscale("linear")
    ax[1].set_ylabel("Obs-Model/Err")
    ax[1].set_xlabel("Energy (keV)")
    #
    plt.savefig(figname)
    # plt.show()
    plt.close(fig)
    return ()
