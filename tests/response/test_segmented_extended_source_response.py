import numpy as np
import astropy.units as u
from astropy.time import Time

from histpy import Histogram, Axis, Axes

import pytest

from astromodels import ExtendedSource, Gaussian_on_sphere
from threeML import Powerlaw

from cosipy import test_data
from cosipy.data_io import SegmentedEmCDSBinnedData
from cosipy.image_deconvolution import CoordsysConversionMatrix, TimeBinnedExposureTable
from cosipy.response import BinnedThreeMLSegmentedExtendedSourceResponse, FullDetectorResponse
from cosipy.spacecraftfile import SpacecraftHistory

dr_path = str(test_data.path / "test_full_detector_response.h5")


def _make_source():

    spectrum = Powerlaw()
    spectrum.K.value = 1e-3
    spectrum.K.unit = 1 / u.cm / u.cm / u.s / u.keV
    spectrum.piv.value = 1000
    spectrum.piv.unit = u.keV
    spectrum.index.value = -2

    morphology = Gaussian_on_sphere(lon0=50, lat0=-45, sigma=10)

    return ExtendedSource('src', spatial_shape=morphology, spectral_shape=spectrum)


def _make_segmented_data(fdr, n_segments):

    time_axis = Axis(np.arange(n_segments + 1), label='Time')
    data_axes = Axes([time_axis, fdr.axes['Em'], fdr.axes['Phi'], fdr.axes['PsiChi']], copy_axes=False)

    return SegmentedEmCDSBinnedData(Histogram(data_axes))


def test_segmented_extended_source_response_matches_full_exposure():
    """
    Closure test, analogous to the point-source one: summing the
    segmented extended-source expectation (one Time bin per raw
    orientation sample) over the Time axis must exactly reproduce the
    expectation obtained from a single (non-segmented) exposure bin
    covering the full observation, as long as both CCMs are built with
    use_averaged_pointing=False (i.e. no attitude-averaging
    approximation on either side).
    """

    fdr = FullDetectorResponse.open(dr_path)
    ori = SpacecraftHistory.open(test_data.path / "20280301_first_10sec.fits")
    source = _make_source()

    exposure_table_full = TimeBinnedExposureTable.from_orientation(
        ori, tstart_list=Time([ori.obstime[0]]), tstop_list=Time([ori.obstime[-1]]))

    exposure_table_seg = TimeBinnedExposureTable.from_orientation(
        ori, tstart_list=Time(ori.obstime[:-1]), tstop_list=Time(ori.obstime[1:]))

    ccm_full = CoordsysConversionMatrix.from_exposure_table(
        exposure_table_full, fdr, nside_model=fdr.nside, use_averaged_pointing=False)

    ccm_seg = CoordsysConversionMatrix.from_exposure_table(
        exposure_table_seg, fdr, nside_model=fdr.nside, use_averaged_pointing=False)

    data_full = _make_segmented_data(fdr, len(exposure_table_full))
    data_seg = _make_segmented_data(fdr, len(exposure_table_seg))

    esr_full = BinnedThreeMLSegmentedExtendedSourceResponse(data_full, fdr, ccm_full)
    esr_full.set_source(source)
    exp_full = esr_full.expectation()

    esr_seg = BinnedThreeMLSegmentedExtendedSourceResponse(data_seg, fdr, ccm_seg)
    esr_seg.set_source(source)
    exp_seg = esr_seg.expectation()

    assert exp_seg.axes['Time'].nbins == len(exposure_table_seg)

    total_full = exp_full.project('Em', 'Phi', 'PsiChi')
    total_seg = exp_seg.project('Em', 'Phi', 'PsiChi')

    assert np.sum(total_full.contents) > 0
    assert np.allclose(total_seg.contents, total_full.contents, rtol=1e-8)


def test_segmented_extended_source_response_generic_path_matches_2d_cached_path():
    """
    For a 2D (energy-independent) morphology, the cached
    spatial-response path (used in expectation()) and the generic,
    uncached path (_compute_generic_expectation(), normally only used
    for energy-dependent/3D morphologies) must agree.
    """

    fdr = FullDetectorResponse.open(dr_path)
    ori = SpacecraftHistory.open(test_data.path / "20280301_first_10sec.fits")
    source = _make_source()

    exposure_table_full = TimeBinnedExposureTable.from_orientation(
        ori, tstart_list=Time([ori.obstime[0]]), tstop_list=Time([ori.obstime[-1]]))

    ccm_full = CoordsysConversionMatrix.from_exposure_table(
        exposure_table_full, fdr, nside_model=fdr.nside, use_averaged_pointing=False)

    data_full = _make_segmented_data(fdr, len(exposure_table_full))

    esr = BinnedThreeMLSegmentedExtendedSourceResponse(data_full, fdr, ccm_full)
    esr.set_source(source)

    cached = esr.expectation()  # uses the 2D-factorized cached path
    generic = esr._compute_generic_expectation(source)

    assert np.allclose(cached.project('Em', 'Phi', 'PsiChi').contents,
                       generic.project('Em', 'Phi', 'PsiChi').contents,
                       rtol=1e-8)


def test_segmented_extended_source_response_rejects_galactic_psichi():

    fdr = FullDetectorResponse.open(dr_path)
    ori = SpacecraftHistory.open(test_data.path / "20280301_first_10sec.fits")

    exposure_table = TimeBinnedExposureTable.from_orientation(
        ori, tstart_list=Time([ori.obstime[0]]), tstop_list=Time([ori.obstime[-1]]))

    ccm = CoordsysConversionMatrix.from_exposure_table(
        exposure_table, fdr, nside_model=fdr.nside, use_averaged_pointing=False)

    psichi_axis_gal = fdr.axes['PsiChi'].copy()
    psichi_axis_gal.coordsys = 'galactic'

    time_axis = Axis(np.arange(len(exposure_table) + 1), label='Time')
    data_axes = Axes([time_axis, fdr.axes['Em'], fdr.axes['Phi'], psichi_axis_gal], copy_axes=False)

    data = SegmentedEmCDSBinnedData(Histogram(data_axes))

    with pytest.raises(ValueError):
        BinnedThreeMLSegmentedExtendedSourceResponse(data, fdr, ccm)


def test_segmented_extended_source_response_rejects_mismatched_segment_axis():

    fdr = FullDetectorResponse.open(dr_path)
    ori = SpacecraftHistory.open(test_data.path / "20280301_first_10sec.fits")

    exposure_table = TimeBinnedExposureTable.from_orientation(
        ori, tstart_list=Time(ori.obstime[:-1]), tstop_list=Time(ori.obstime[1:]))

    ccm = CoordsysConversionMatrix.from_exposure_table(
        exposure_table, fdr, nside_model=fdr.nside, use_averaged_pointing=False)

    # Only one segment bin in data, but the ccm has len(exposure_table) bins.
    data = _make_segmented_data(fdr, 1)

    with pytest.raises(ValueError):
        BinnedThreeMLSegmentedExtendedSourceResponse(data, fdr, ccm)
