import numpy as np
import astropy.units as u
from astropy.time import Time
from histpy import Histogram, Axis, Axes

from threeML import Powerlaw, Model, JointLikelihood, DataList
from astromodels import ExtendedSource, Gaussian_on_sphere, Parameter

from cosipy import test_data
from cosipy.background_estimation import FreeNormBinnedBackground
from cosipy.data_io import SegmentedEmCDSBinnedData
from cosipy.image_deconvolution import CoordsysConversionMatrix, TimeBinnedExposureTable
from cosipy.interfaces import ThreeMLPluginInterface
from cosipy.response import (
    BinnedThreeMLModelFolding,
    BinnedThreeMLSegmentedExtendedSourceResponse,
    FullDetectorResponse,
)
from cosipy.spacecraftfile import SpacecraftHistory
from cosipy.statistics import PoissonLikelihood

data_path = test_data.path
dr_path = str(data_path / "test_full_detector_response.h5")

K_true = 1e-3
index_true = -2


def test_time_segmented_extended_source_spectral_fit():
    """
    End-to-end test of a time-segmented *extended*-source spectral fit:
    the expectation is built from a FullDetectorResponse + a
    CoordsysConversionMatrix with two Time segments (instead of the
    precomputed, time-integrated Galactic ExtendedSourceResponse used
    for non-segmented fits). The fit should recover the injected
    spectral and background parameters, like the point-source
    equivalent in test_segmented_spectral_fitting.py.
    """

    fdr = FullDetectorResponse.open(dr_path)
    ori = SpacecraftHistory.open(data_path / "20280301_first_10sec.fits")

    spectrum = Powerlaw()
    spectrum.K.value = K_true
    spectrum.K.unit = 1 / u.cm / u.cm / u.s / u.keV
    spectrum.piv.value = 1000
    spectrum.piv.unit = u.keV
    spectrum.index.value = index_true

    morphology = Gaussian_on_sphere(lon0=50, lat0=-45, sigma=10)
    morphology.lon0.free = False
    morphology.lat0.free = False
    morphology.sigma.free = False

    source = ExtendedSource('src', spatial_shape=morphology, spectral_shape=spectrum)

    # Two time segments covering the whole 10 s orientation file.
    t0 = ori.obstime[0]
    t_mid = ori.obstime[len(ori.obstime) // 2]
    t_end = ori.obstime[-1]

    exposure_table = TimeBinnedExposureTable.from_orientation(ori,
                                                               tstart_list=Time([t0, t_mid]),
                                                               tstop_list=Time([t_mid, t_end]))

    assert len(exposure_table) == 2

    ccm = CoordsysConversionMatrix.from_exposure_table(exposure_table, fdr,
                                                       nside_model=fdr.nside,
                                                       use_averaged_pointing=False)

    time_axis = Axis(np.arange(len(exposure_table) + 1), label='Time')
    data_axes = Axes([time_axis, fdr.axes['Em'], fdr.axes['Phi'], fdr.axes['PsiChi']], copy_axes=False)
    skeleton = SegmentedEmCDSBinnedData(Histogram(data_axes))

    # Create fake data and background using the same (segmented)
    # response as the fit, for a circular test -- same philosophy as
    # the point-source test.
    esr_truth = BinnedThreeMLSegmentedExtendedSourceResponse(skeleton, fdr, ccm)
    esr_truth.set_source(source)
    data_dist = esr_truth.expectation()

    bkg_dist = data_dist.copy()
    bkg_dist[:] = np.mean(bkg_dist.contents)  # Flat background, same in each Time bin
    data_dist += bkg_dist

    bkg_rate = np.sum(bkg_dist.contents) / ori.cumulative_livetime()

    # Move initial guess slightly away from true values
    spectrum.index.value = index_true * 1.1
    spectrum.K.value = K_true * 1.1

    data = SegmentedEmCDSBinnedData(data_dist)
    bkg = FreeNormBinnedBackground(bkg_dist, sc_history=ori, copy=False)

    esr = BinnedThreeMLSegmentedExtendedSourceResponse(data, fdr, ccm)

    response = BinnedThreeMLModelFolding(data=data, extended_source_response=esr)

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

    assert np.allclose([spectrum.K.value, spectrum.index.value, cosi.bkg_parameter['bkg_norm'].value],
                       [K_true, index_true, bkg_rate.value],
                       rtol=1e-3)
