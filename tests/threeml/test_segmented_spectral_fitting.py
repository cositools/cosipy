import numpy as np
import astropy.units as u
from astropy.time import Time
from histpy import Histogram, Axis, Axes

from threeML import Powerlaw, PointSource, Model, JointLikelihood, DataList
from astromodels import Parameter

from cosipy import test_data
from cosipy.background_estimation import FreeNormBinnedBackground
from cosipy.data_io import SegmentedEmCDSBinnedData
from cosipy.image_deconvolution import TimeBinnedExposureTable
from cosipy.interfaces import ThreeMLPluginInterface
from cosipy.response import (
    BinnedInstrumentResponse,
    BinnedThreeMLModelFolding,
    BinnedThreeMLSegmentedPointSourceResponse,
    FullDetectorResponse,
    PointSourceResponse,
)
from cosipy.spacecraftfile import SpacecraftHistory
from cosipy.statistics import PoissonLikelihood

data_path = test_data.path

dr_path = str(data_path / "test_full_detector_response.h5")

l, b = 50, -45

index = -2
piv = 1. * u.MeV
K = 1 / u.cm / u.cm / u.s / u.keV


def test_time_segmented_point_source_spectral_fit():
    """
    End-to-end test of a time-segmented point-source spectral fit:
    data and response are both binned/built with two Time segments
    (using the same TimeBinnedExposureTable for both), and the fit
    should recover the injected spectral and background parameters,
    exactly like the non-segmented fit in test_spectral_fitting.py.
    """

    sc_orientation = SpacecraftHistory.open(data_path / "20280301_first_10sec.fits")

    spectrum = Powerlaw()
    spectrum.index.value = index
    spectrum.K.value = K.value
    spectrum.piv.value = piv.value
    spectrum.K.unit = K.unit
    spectrum.piv.unit = piv.unit

    source = PointSource("source", l=l, b=b, spectral_shape=spectrum)

    fdr = FullDetectorResponse.open(dr_path)

    # Two time segments covering the whole 10 s orientation file.
    t0 = sc_orientation.obstime[0]
    t_mid = sc_orientation.obstime[len(sc_orientation.obstime) // 2]
    t_end = sc_orientation.obstime[-1]

    tstart_list = Time([t0, t_mid])
    tstop_list = Time([t_mid, t_end])

    exposure_table = TimeBinnedExposureTable.from_orientation(sc_orientation,
                                                               tstart_list=tstart_list,
                                                               tstop_list=tstop_list)

    assert len(exposure_table) == 2

    psichi_axis_gal = fdr.axes['PsiChi'].copy()
    psichi_axis_gal.coordsys = 'galactic'

    time_axis = Axis(np.arange(len(exposure_table) + 1), label='Time')
    data_axes = Axes([time_axis, fdr.axes['Em'], fdr.axes['Phi'], psichi_axis_gal], copy_axes=False)

    skeleton = SegmentedEmCDSBinnedData(Histogram(data_axes))
    instrument_response = BinnedInstrumentResponse(fdr, skeleton.to_emcds())

    # Create fake data and background using the same (segmented)
    # response as the fit, for a circular test -- same philosophy as
    # test_point_source_spectral_fit() in test_spectral_fitting.py.
    psr_truth = PointSourceResponse.from_exposure_table(source.position.sky_coord,
                                                        skeleton,
                                                        instrument_response,
                                                        exposure_table,
                                                        energy_axis=fdr.axes['Ei'])

    data_dist = psr_truth.get_expectation(source.spectrum.main.shape)
    bkg_dist = data_dist.copy()
    bkg_dist[:] = np.mean(bkg_dist.contents)  # Flat background, same in each Time bin
    data_dist += bkg_dist

    bkg_rate = np.sum(bkg_dist.contents) / sc_orientation.cumulative_livetime()

    # Move initial guess slightly away from true values
    spectrum.index.value = index * 1.1
    spectrum.K.value = K.value * 1.1

    data = SegmentedEmCDSBinnedData(data_dist)
    bkg = FreeNormBinnedBackground(bkg_dist, sc_history=sc_orientation, copy=False)

    psr = BinnedThreeMLSegmentedPointSourceResponse(data=data,
                                                    instrument_response=instrument_response,
                                                    exposure_table=exposure_table,
                                                    energy_axis=fdr.axes['Ei'],
                                                    polarization_axis=fdr.axes['Pol'] if 'Pol' in fdr.axes.labels else None)

    response = BinnedThreeMLModelFolding(data=data, point_source_response=psr)

    like_fun = PoissonLikelihood(data, response, bkg)

    cosi = ThreeMLPluginInterface('cosi', like_fun, response, bkg)

    cosi.bkg_parameter['bkg_norm'] = Parameter("bkg_norm",
                                               1.1 * bkg_rate.to_value(u.Hz),
                                               unit=u.Hz,
                                               min_value=0,
                                               max_value=1e6,
                                               delta=1e5,
                                               )

    plugins = DataList(cosi)

    model = Model(source)

    like = JointLikelihood(model, plugins, verbose=False)

    like.fit(compute_covariance=False)

    sp = source.spectrum.main.shape

    assert np.allclose([sp.K.value, sp.index.value, cosi.bkg_parameter['bkg_norm'].value],
                       [K.value, index, bkg_rate.value],
                       rtol=1e-3)
