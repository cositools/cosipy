from cosipy import COSILike

import numpy as np

from threeML import *
from threeML import Band, PointSource, Model, JointLikelihood, DataList
from astromodels import Parameter
from astropy import units as u


def get_fit_results(sou, bk, resp_path, ori, bkname, model):
    """
    Fits a model to spectral data using threeML.

    Parameters
    ----------
    sou : histpy:Histogram
        The binned histogram of the source data
    bk : histpy:Histogram
        The binned histogram of the background data
    resp_path: str
        Path to the response file.
    ori: cosipy.spacecraftfile.SpacecraftFile.SpacecraftFile
        A SpacecraftFile Object, parsed from an orientation file.
    bkname: str
        Name of the results of the fit to the background (internal to threeML).
    model: astromodels.core.model.Model
        threeML model
    Returns
    -------
    results : threeML.analysis_results.MLEResults
        ThreeML fit result object.
    tot_exp_ counts: astropy.units.quantity.Quantity
        Array containing the total counts (source+background)
        predicted by the model in each energy bin.
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
                    nuisance_param=bkg_par,
                    earth_occ=True)  # background parameter

    cosi.set_model(model)
    plugins = DataList(cosi)
    like = JointLikelihood(model, plugins, verbose=False)
    like.fit()
    results = like.results

    # FIXME: Do no rely on protected variables. _expected_counts is not guaranteed to
    #   survive refactoring. Revisit this after the interfaces refactoring
    expectation = None
    for source_expectation in cosi._expected_counts.values():
        if expectation is None:
            expectation = source_expectation.copy()
        else:
            expectation += source_expectation

    tot_exp_counts = expectation.project('Em').todense().contents + (
                bkg_par.value * bk.project('Em').todense().contents)
    #
    #
    return results, tot_exp_counts


def get_fit_par(results):
    """
    Extracts a dictionary whose keys are the free parameters of the model,
    and values are a tuple with the median and standard deviation.

    Parameters
    ----------
    results : threeML.analysis_results.MLEResults
        ThreeML fit result object.

    Returns
    -------
    dict: dict
        Dictionary whose keys are the free parameters of the model,
        and values are a tuple with the median and standard deviation.
    """
    return {par_name: (results.get_variates(par.path).median, results.get_variates(par.path).std)
            for par_name, par in results.optimized_model.free_parameters.items()}


def get_fit_fluxes(results):
    """
    Compute the 0.1-10 MeV flux from a best-fit model in threeML.
    Parameters
    ----------
    results : threeML.analysis_results.MLEResults
        ThreeML fit result object.

    Returns
    -------
    fl: float
        0.1-10 MeV flux [photons/s/cm^2]
    e_low_fl: float
        Negative error for the flux.
     e_hi_fl: float
        Positive error for the flux.
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
    return (fl, e_low_fl, e_hi_fl)
