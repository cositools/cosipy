
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

from mhealpy import HealpixMap

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
    tot_exp_counts = expectation.project('Em').to_dense(copy=False).contents + (
                expectation_bkg.project('Em').to_dense(copy=False).contents)


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


def get_ts_results(ts_results, ts_uniq=None, multiresolution=False, nside=None):
    """
    Extract peak TS values, coordinates, and pixel spacing from TS map results.

    Parameters
    ----------
    ts_results : array-like
        Array of Test Statistic values.
    ts_uniq : array-like, optional
        HEALPix UNIQ indices for multiresolution maps (required if multiresolution=True).
    multiresolution : bool, optional
        If True, process as a MOC map. Default is False.
    nside : int, optional
        HEALPix resolution parameter (required if multiresolution=False).

    Returns
    -------
    max_ts : float
        Maximum TS value found in the map.
    max_coo : astropy.coordinates.SkyCoord
        SkyCoord object at the location of the maximum TS.
    max_l : float
        Galactic longitude [deg] at maximum TS.
    max_b : float
        Galactic latitude [deg] at maximum TS.
    pixel_mean_spacing : float
        Angular size of the pixel [deg]. For MOC, returns the minimum spacing.
    """
    if not multiresolution:
        if nside is None:
            raise ValueError("nside must be provided when multiresolution is False")

        max_ts = np.max(ts_results)
        highest_idx = np.argmax(ts_results)
        m = HealpixMap(nside=nside, scheme="nested", coordsys="galactic")
        max_coo = m.pix2skycoord(highest_idx)
        pixel_area = m.pixarea()
        pixel_mean_spacing = np.degrees(np.sqrt(pixel_area.value))

    else:
        if ts_uniq is None:
            raise ValueError("ts_uniq must be provided when multiresolution is True")

        max_ts = np.max(ts_results)
        highest_idx = np.argmax(ts_results)
        m = HealpixMap(data=ts_results, uniq=ts_uniq, coordsys="galactic")
        max_coo = m.pix2skycoord(highest_idx)
        pixel_area = np.min(m.pixarea())
        pixel_mean_spacing = np.degrees(np.sqrt(pixel_area.value))

    max_l = float(max_coo.l.value)
    max_b = float(max_coo.b.value)

    return (max_ts, max_coo,max_l, max_b, pixel_mean_spacing)