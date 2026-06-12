import pytest
from histpy import Histogram
import astropy.units as u
import numpy as np

from cosipy import LineBackgroundEstimation, test_data

def test_line_background_estimation():

    # prepare data
    data_path = test_data.path / "test_event_histogram_galacticCDS.hdf5"

    data = Histogram.open(data_path)
    data = data.project(['Em', 'Phi', 'PsiChi'])

    # prepare model
    def bkg_model(x, a, b):
        pivot = 1000.0
        return a * (x/pivot)**b

    # instantiate the line background class
    instance = LineBackgroundEstimation(data)

    # set background spectrum model
    instance.set_bkg_energy_spectrum_model(bkg_model, [1.0, -3.0])

    # set mask
    instance.set_mask((0.0, 1000.0) * u.keV, (3000.0, 5000.0) * u.keV)

    # run fitting w/ par limit
    m = instance.fit_energy_spectrum(param_limits = {1: (None, 100)})
    m = instance.fit_energy_spectrum(param_limits = {1: (-100, None)})
    m = instance.fit_energy_spectrum(param_limits = {1: (-100, 100)})

    # run fitting w/ par fixed
    m = instance.fit_energy_spectrum(fixed_params = {1: 0})

    # run fitting w/ stepsize; set limit to prevent overflow
    m = instance.fit_energy_spectrum(stepsize_params = {1: 0.1},
                                     param_limits={1: (None, 100)})

    # run fitting from scratch
    instance.set_bkg_energy_spectrum_model(bkg_model, [1.0, -3.0])
    m = instance.fit_energy_spectrum()

    # run plotting
    ax, _ = instance.plot_energy_spectrum()

    # set range for source region
    source_range = (2000.0, 2500.0)  * u.keV

    # generate background model

    ## Case 1: a single extracting region
    background_region = (1000.0, 1584.89) * u.keV

    bkg_model_histogram = instance.generate_bkg_model_histogram(source_range, [background_region])

    ### check sum
    assert np.isclose(np.sum(bkg_model_histogram), 0.6340438314826473, atol = 1e-5)

    ## Case 2: a single extracting region broader than the actual bin width
    background_region = (999.0, 1585.0) * u.keV

    bkg_model_histogram = instance.generate_bkg_model_histogram(source_range, [background_region])

    ## Case 3: a extracting region is too narrow. Check error
    background_region = (1121.0, 1121.001) * u.keV

    with pytest.raises(ValueError) as e_info:
        bkg_model_histogram = instance.generate_bkg_model_histogram(source_range, [background_region])

    ## Case 4: two extracting regions
    background_region_1 = (1000.0, 1585.0) * u.keV #background counts estimation before the line
    background_region_2 = (3500.0, 6310.0) * u.keV #background counts estimation before the line

    bkg_model_histogram = instance.generate_bkg_model_histogram(source_range, [background_region_1, background_region_2])

    ## Case 5: smoothing
    bkg_model_histogram_smoothing = instance.generate_bkg_model_histogram(
        source_range, [background_region_1, background_region_2],
        smoothing_fwhm=20 * u.deg)

    ### total counts should be preserved after smoothing
    assert np.isclose(np.sum(bkg_model_histogram_smoothing), np.sum(bkg_model_histogram), rtol=1e-3)

    ## Case 6: l_cut
    bkg_model_histogram_l_cut = instance.generate_bkg_model_histogram(
        source_range, [background_region_1, background_region_2],
        l_cut=3)

    ### shape should be unchanged after l_cut filtering
    assert bkg_model_histogram_l_cut.shape == bkg_model_histogram.shape

    ## Case 7: smoothing_fwhm and l_cut cannot be specified at the same time
    with pytest.raises(ValueError):
        instance.generate_bkg_model_histogram(
            source_range, [background_region_1, background_region_2],
            smoothing_fwhm=20 * u.deg, l_cut=3)

    ## Case 8: rebin_phi + smoothing
    bkg_model_histogram_rebin_smoothing = instance.generate_bkg_model_histogram(
        source_range, [background_region_1, background_region_2],
        smoothing_fwhm=20 * u.deg, rebin_phi=5)

    ### shape should be unchanged after rebin+unbin
    assert bkg_model_histogram_rebin_smoothing.shape == bkg_model_histogram.shape

    ### total counts per Phi bin should be preserved (rebin_phi only changes spatial pattern)
    phi_original = np.sum(bkg_model_histogram[:],       axis=-1)  # sum over PsiChi
    phi_rebin    = np.sum(bkg_model_histogram_rebin_smoothing[:], axis=-1)
    assert np.allclose(phi_original, phi_rebin, rtol=1e-5)

    ## Case 9: rebin_phi + l_cut
    bkg_model_histogram_rebin_l_cut = instance.generate_bkg_model_histogram(
        source_range, [background_region_1, background_region_2],
        l_cut=3, rebin_phi=5)

    ### shape should be unchanged
    assert bkg_model_histogram_rebin_l_cut.shape == bkg_model_histogram.shape

    ### total counts per Phi bin should be preserved
    phi_rebin_l_cut = np.sum(bkg_model_histogram_rebin_l_cut[:], axis=-1)
    assert np.allclose(phi_original, phi_rebin_l_cut, rtol=1e-5)
