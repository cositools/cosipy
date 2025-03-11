from cosipy import COSILike, BinnedData
from cosipy.spacecraftfile import SpacecraftFile
from cosipy.response.FullDetectorResponse import FullDetectorResponse
from cosipy.util import fetch_wasabi_file

from scoords import SpacecraftFrame

from astropy.time import Time, TimeDelta
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.stats import poisson_conf_interval

import numpy as np
import matplotlib.pyplot as plt

from threeML import *
from threeML import Band, PointSource, Model, JointLikelihood, DataList
from cosipy import Band_Eflux
from astromodels import Parameter
from astropy import units as u
from pathlib import Path

import os
import subprocess


def get_binned_data(yaml_path, udata_path, bdata_name, psichi_coo):
    """
    Creates a binned dataset from a yaml file and an unbinned data file.
    Args:
        psichi_coo: either "galactic" or "local"

    """
    data=BinnedData(yaml_path)
    data.get_binned_data(unbinned_data=udata_path, output_name=bdata_name, psichi_binning=psichi_coo,make_binning_plots=False)
    return data


def load_binned_data(yaml_path,data_path):
    data = BinnedData(yaml_path)
    data.load_binned_data_from_hdf5(data_path)
    return(data.binned_data)

def load_ori(ori_path):
    ori = SpacecraftFile.parse_from_file(ori_path)
    return(ori)

def tslice_binned_data(data,tmin,tmax):
    """Slice a binned dataset in time"""
    idx_tmin = np.where(data.axes['Time'].edges.value >= tmin)[0][0]
    idx_tmax_all = np.where(data.axes['Time'].edges.value <= tmax)
    y = len(idx_tmax_all[0]) - 1
    idx_tmax = np.where(data.axes['Time'].edges.value <= tmax)[0][y]
    tsliced_data = data.slice[{'Time': slice(idx_tmin, idx_tmax)}]
    return (tsliced_data)

def make_eqsize_tslices(tstart,tstop,nbins):
    dt=(tstop-tstart)/nbins
    tmins = np.array([], dtype=float)# Initialize as empty numpy arrays
    tmaxs=np.array([], dtype=float)# Initialize as empty numpy arrays
    #
    for i in range(nbins):
        tmin=tstart+i*dt
        tmax=tmin+dt
        tmins = np.append(tmins, tmin)
        tmaxs=np.append(tmaxs, tmax)
    return(tmins,tmaxs)

def make_minsn_tslices(tstart, tstop, yaml_path, data_path, min_sn, max_slices):
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
    return (tmins, tmaxs)

def tslice_ori(ori,tmin,tmax):
    ori_min = Time(tmin,format = 'unix')
    ori_max = Time(tmax,format = 'unix')
    tsliced_ori = ori.source_interval(ori_min, ori_max)
    return(tsliced_ori)

def get_ene(data):
    #
    binned_energy_edges = data.axes['Em'].edges.value
    #
    ene = np.array([])
    e_ene = np.array([])
    #
    for j in range(len(binned_energy_edges) - 1):
        delta_ene=(binned_energy_edges[j + 1]-binned_energy_edges[j])
        ene= np.append(ene, (binned_energy_edges[j]+0.5*delta_ene))
        e_ene = np.append(e_ene, 0.5*delta_ene)
        #
    return(ene,e_ene)

def get_counts_ene (data):
    cts=data.project('Em').todense().contents
    e_cts=np.sqrt(cts)
    return(cts,e_cts)

def build_spectrum(model, pars, par_minvalues, par_maxvalues):
    spectrum = model()  # Instantiate the model
    par_list = list(spectrum.parameters.keys())  # Correctly get the keys

    if len(pars) != len(par_list) :
        raise ValueError("Number of parameters provided does not match the model's expected parameters.") #Important check

    for i in range(len(pars)):
        parameter_name = par_list[i]  # Get the parameter name (key)
        print (parameter_name)
        #
        spectrum.parameters[parameter_name].unit= pars[i].unit #Adjust units
        #
        ##Adjust parameters range
        #
        if par_maxvalues is not None:
            spectrum.parameters[parameter_name].max_value= par_maxvalues[i]
        if par_minvalues is not None:
            spectrum.parameters[parameter_name].min_value= par_minvalues[i]
        #Inizializza il parametro
        spectrum.parameters[parameter_name].value=pars[i]
    return(spectrum)


def get_fit_residuals(cts, e_cts,cts_exp):
    resid=(cts-cts_exp)/e_cts
    e_resid=np.abs((e_cts/cts)*resid)
    return(resid, e_resid)

def plot_fit(sou,cts_exp,figname):
    # Save a plot of the current fit.
    #
    ene,e_ene=get_ene(sou)
    cts,e_cts=get_counts_ene(sou)
    #
    resid,e_resid=get_fit_residuals(cts, e_cts,cts_exp)
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
    #plt.show()
    plt.savefig(figname)
    return()



def get_fit_results(sou,bk,resp_path,ori,l,b,souname,bkname,spectrum):
    bkg_par = Parameter(bkname,  # background parameter
                        0.1,  # initial value of parameter
                        min_value=0,  # minimum value of parameter
                        max_value=5,  # maximum value of parameter
                        delta=1e-3,  # initial step used by fitting engine
                        desc="Background parameter for cosi")

    cosi = COSILike("cosi",  # COSI 3ML plugin
                    dr=resp_path,  # detector response
                    data=sou.project('Em', 'Phi', 'PsiChi'),
                    bkg=bk.project('Em', 'Phi', 'PsiChi'),
                    sc_orientation=ori,  # spacecraft orientation
                    nuisance_param=bkg_par)  # background parameter

    source = PointSource(souname,  # Name of source (arbitrary, but needs to be unique)
                         l=l,  # Longitude (deg)
                         b=b,  # Latitude (deg)
                         spectral_shape=spectrum)  # Spectral model

    model = Model(source)
    cosi.set_model(model)
    plugins = DataList(cosi)
    like = JointLikelihood(model, plugins, verbose=False)
    like.fit()
    results = like.results
    expectation = cosi._expected_counts[souname]
    tot_exp_counts=expectation.project('Em').todense().contents + (bkg_par.value * bk.project('Em').todense().contents)
    #
    #
    return(results,tot_exp_counts)

def get_fit_par (results,souname,bkname):
   #
    par_sou = {par.name: results.get_variates(par.path).median
               for par in results.optimized_model[souname].parameters.values()
               if par.free}
    #
    epar_sou = {par.name: results.get_variates(par.path).std
                for par in results.optimized_model[souname].parameters.values()
                if par.free}

    par_bk=results.get_variates(bkname).median
    epar_bk = results.get_variates(bkname).std
    return(par_sou,epar_sou,par_bk,epar_bk)

def get_fit_fluxes(results):
    threeML_config.point_source.integrate_flux_method = "trapz"
    result_fl = results.get_flux(
        ene_min=100. * u.keV,
        ene_max=10000. * u.keV,
        confidence_level=0.95,
        sum_sources=True,
        flux_unit="1/(cm2 s)"
    )
    #
    fl = result_fl["flux"].values[0].value
    e_low_fl = np.abs(result_fl["low bound"].values[0].value - fl)
    e_hi_fl = result_fl["hi bound"].values[0].value - fl
    return(fl,e_low_fl,e_hi_fl)


model_ids={'Band': Band}
