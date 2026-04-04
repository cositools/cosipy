from cosipy.response.FullDetectorResponse import FullDetectorResponse
from cosipy.statistics import PoissonLikelihood
from cosipy.background_estimation import FreeNormBinnedBackground
from cosipy.interfaces import ThreeMLPluginInterface
from cosipy.response import BinnedThreeMLModelFolding, BinnedInstrumentResponse, BinnedThreeMLPointSourceResponse
from cosipy.data_io import EmCDSBinnedData

import numpy as np

from threeML import *
from threeML import Band, PointSource, Model, JointLikelihood, DataList
from astromodels import Parameter
from astropy import units as u


def get_fit_results(sou, bk, resp_path, ori_sou, ori_bk, model):
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
    ori_sou: cosipy.spacecraftfile.SpacecraftFile.SpacecraftFile
        A SpacecraftFile Object sliced as the source
    ori_bk: cosipy.spacecraftfile.SpacecraftFile.SpacecraftFile
        A SpacecraftFile Object sliced as the background
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

    dr = FullDetectorResponse.open(resp_path)

    data = EmCDSBinnedData(sou.project('Em', 'Phi', 'PsiChi'))

    bkg = FreeNormBinnedBackground(bk.project('Em', 'Phi', 'PsiChi'),
                               sc_history=ori_bk,
                               copy = False)

    instrument_response = BinnedInstrumentResponse(dr, data)

    psr = BinnedThreeMLPointSourceResponse(data = data,
                                       instrument_response = instrument_response,
                                       sc_history=ori_sou,
                                       energy_axis = dr.axes['Ei'],
                                       polarization_axis = dr.axes['Pol'] if 'Pol' in dr.axes.labels else None,
                                       nside = 2*data.axes['PsiChi'].nside)

    response = BinnedThreeMLModelFolding(data = data, point_source_response = psr)

    like_fun = PoissonLikelihood(data, response, bkg)

    cosi = ThreeMLPluginInterface('cosi',
                              like_fun,
                              response,
                              bkg)
    
    cosi.bkg_parameter['bkg_norm'] = Parameter('bkg_norm',  # background parameter
                                      1.0,  # initial value of parameter
                                      min_value=0,  # minimum value of parameter
                                      max_value= 20,  # maximum value of parameter
                                      delta=0.05,  # initial step used by fitting engine
                                      unit = u.Hz
                                      )

    cosi.set_model(model)
    plugins = DataList(cosi)
    like = JointLikelihood(model, plugins, verbose=False)
    like.fit()
    results = like.results

    expectation = response.expectation()
    expectation_bkg = bkg.expectation()
    tot_exp_counts = expectation.project('Em').todense().contents + (
                expectation_bkg.project('Em').todense().contents)
    
    
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
