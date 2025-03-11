from cosipy import COSILike

import numpy as np

from threeML import *
from threeML import Band, PointSource, Model, JointLikelihood, DataList
from astromodels import Parameter
from astropy import units as u


MODEL_IDS={'Band': Band,
           }


def build_spectrum(model, pars, par_minvalues, par_maxvalues):
    """
    Builds a threeML spectrum given the model class and params
    Args:
        pars: list of astropy quantities
        par_minvalues: list of minimum values, must be the same length of pars (can be None)
        par_minvalues: list of maximum values, must be the same length of pars (can be None)
    Return:
        threeML spectrum with assigned par values
    """

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





def get_fit_results(sou,bk,resp_path,ori,l,b,souname,bkname,spectrum):
    """
    Fits the spectrum to the data
    """
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
    return results,tot_exp_counts

def get_fit_par(results,souname,bkname):
    """
    Extracts dictionaries from the fit results
    par_sou,epar_sou,par_bk,epar_bk
    """
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
    """
    reads the flux from the fit results
    """
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



