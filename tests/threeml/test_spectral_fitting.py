import sys

from cosipy import test_data, BinnedData
from cosipy.background_estimation import FreeNormBinnedBackground
from cosipy.data_io import EmCDSBinnedData
from cosipy.interfaces import ThreeMLPluginInterface
from cosipy.response import BinnedThreeMLModelFolding, FullDetectorResponse, BinnedInstrumentResponse, \
    BinnedThreeMLPointSourceResponse
from cosipy.spacecraftfile import SpacecraftHistory
import astropy.units as u
import numpy as np
from threeML import Powerlaw, PointSource, Model, JointLikelihood, DataList
from astromodels import Parameter
from astropy.coordinates import SkyCoord

from cosipy.statistics import PoissonLikelihood

data_path = test_data.path

sc_orientation = SpacecraftHistory.open(data_path / "20280301_2s.fits")
dr_path = str(data_path / "test_full_detector_response.h5") # path to detector response

l = 50
b = -45

index = -2
piv = 1. * u.MeV
K = 1 / u.cm / u.cm / u.s / u.keV

spectrum = Powerlaw()

spectrum.index.value = index
spectrum.K.value = K.value
spectrum.piv.value = piv.value

spectrum.K.unit = K.unit
spectrum.piv.unit = piv.unit

# source in galactic frame
source = PointSource("source",                     # Name of source (arbitrary, but needs to be unique)
                     l = l,                        # Longitude (deg)
                     b = b,                        # Latitude (deg)
                     spectral_shape = spectrum)    # Spectral model

def test_point_source_spectral_fit(background=None):

    # Create fake data and background using the same
    # response as the fit (for a circular test)
    fdr = FullDetectorResponse.open(dr_path)
    psr = fdr.get_point_source_response(coord=source.position.sky_coord,
                                        scatt_map=sc_orientation.get_scatt_map(fdr.nside * 2, # Use same nside as hardcoded in COSILke
                                                                               source.position.sky_coord))

    data_dist = psr.get_expectation(source.spectrum.main.shape)
    bkg_dist = data_dist.copy()
    bkg_dist[:] = np.mean(bkg_dist)  # Flat background
    data_dist += bkg_dist

    bkg_rate = np.sum(bkg_dist) / sc_orientation.cumulative_livetime()

    # Move initial guess slightly away from true values
    spectrum.index.value = index * 1.1
    spectrum.K.value = K.value * 1.1

    data = EmCDSBinnedData(data_dist)
    bkg = FreeNormBinnedBackground(bkg_dist,
                                   sc_history=sc_orientation,
                                   copy=False)

    dr = FullDetectorResponse.open(dr_path)
    instrument_response = BinnedInstrumentResponse(dr, data)

    psr = BinnedThreeMLPointSourceResponse(data=data,
                                           instrument_response=instrument_response,
                                           sc_history=sc_orientation,
                                           energy_axis=dr.axes['Ei'],
                                           polarization_axis=dr.axes['Pol'] if 'Pol' in dr.axes.labels else None,
                                           nside=2 * data.axes['PsiChi'].nside)

    response = BinnedThreeMLModelFolding(data=data, point_source_response=psr)

    like_fun = PoissonLikelihood(data, response, bkg)

    cosi = ThreeMLPluginInterface('cosi',
                                  like_fun,
                                  response,
                                  bkg)


    cosi.bkg_parameter['bkg_norm'] = Parameter("bkg_norm",  # background parameter
                                               1.1 * bkg_rate.to_value(u.Hz),  # initial value of parameter
                                               unit=u.Hz,
                                               min_value=0,  # minimum value of parameter
                                               max_value=1e6,  # maximum value of parameter
                                               delta=1e5,  # initial step used by fitting engine
                                               )

    plugins = DataList(cosi)

    model = Model(source)

    like = JointLikelihood(model, plugins, verbose = False)

    like.fit(compute_covariance = False) # avoid sampling-related threeML crashes

    sp = source.spectrum.main.shape

    assert np.allclose([sp.K.value, sp.index.value, cosi.bkg_parameter['bkg_norm'].value],
                       [K.value, index, bkg_rate.value])

    TS_ref = 6377269.127606418

    assert np.allclose([cosi.get_log_like()],
                       [TS_ref])

    # verify that the result is the same regardless of how we specify the source position

    # Move initial guess slightly away from true values
    spectrum.index.value = index * 1.1
    spectrum.K.value = K.value * 1.1

    # same source, but specified in ICRS
    c = SkyCoord(l=l, b=b, unit=u.deg, frame="galactic")
    c_icrs = c.transform_to("icrs")
    source_icrs = PointSource("source_icrs",
                              ra=c_icrs.ra.deg,
                              dec=c_icrs.dec.deg,
                              spectral_shape=spectrum)  # Spectral model

    model = Model(source_icrs)

    cosi = ThreeMLPluginInterface('cosi',
                                  like_fun,
                                  response,
                                  bkg)

    cosi.bkg_parameter['bkg_norm'] = Parameter("bkg_norm",  # background parameter
                                               1.1 * bkg_rate.to_value(u.Hz),  # initial value of parameter
                                               unit=u.Hz,
                                               min_value=0,  # minimum value of parameter
                                               max_value=1e6,  # maximum value of parameter
                                               delta=1e5,  # initial step used by fitting engine
                                               )

    plugins = DataList(cosi)

    like = JointLikelihood(model, plugins, verbose=False)

    # avoid output- and sampling-related threeML crashes
    like.fit(quiet=True, compute_covariance=False)

    sp_icrs = source_icrs.spectrum.main.shape

    # make sure result does not change (much -- bkg_par changes more than the rest)
    assert np.allclose([sp.K.value, sp.index.value, cosi.bkg_parameter['bkg_norm'].value],
                       [K.value, index, bkg_rate.value])

    assert np.allclose([cosi.get_log_like()],
                       [TS_ref])
