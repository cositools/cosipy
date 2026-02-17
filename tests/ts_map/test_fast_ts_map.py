from pathlib import Path
import os

import numpy as np

import astropy.units as u
from astropy.coordinates import SkyCoord

from threeML import Powerlaw

from histpy import Histogram

from cosipy import test_data
from cosipy import FastTSMap, MOCTSMap, SpacecraftFile

def test_ts_fit():

    src_bkg_path = test_data.path / "ts_map_src_bkg.h5"
    bkg_path = test_data.path / "ts_map_bkg.h5"
    response_path = test_data.path / "test_full_detector_response.h5"

    orientation_path = test_data.path / "20280301_2s.fits"
    ori = SpacecraftFile.open(orientation_path)

    src_bkg = Histogram.open(src_bkg_path).project(['Em', 'PsiChi', 'Phi'])
    bkg = Histogram.open(bkg_path).project(['Em', 'PsiChi', 'Phi'])

    ts = FastTSMap(data = src_bkg, bkg_model = bkg, orientation = ori,
                   response_path = response_path, cds_frame = "local")

    index = -2.2
    K = 10 / u.cm / u.cm / u.s / u.keV
    piv = 100 * u.keV
    spectrum = Powerlaw()
    spectrum.index.value = index
    spectrum.K.value = K.value
    spectrum.piv.value = piv.value
    spectrum.K.unit = K.unit
    spectrum.piv.unit = piv.unit

    ts_results = ts.fit(nside = 1,
                        spectrum = spectrum,
                        max_cache_size = 10)

    assert np.allclose(ts_results,
                       [142.65320313, 146.40087766, 143.79688155,
                        147.26724713, 142.12808137, 141.04487277,
                        142.91736454, 143.37732116, 143.02080182,
                        142.36211847, 145.47734097, 143.22293343])


    ts_results = ts.fit(nside = 1, energy_channel = [2,3],
                        spectrum = spectrum, cpu_cores = 1)

    assert np.allclose(ts_results,
                       [40.18628386, 39.59382592, 37.4339627,
                        39.88459849, 40.20132198, 39.86762314,
                        37.2327797,  37.4506428,  40.54884861,
                        39.69773074, 38.83421249, 39.99131767])

    ts.plot_ts(ts_results,
               skycoord = SkyCoord(l=0, b=0, unit=u.deg, frame="galactic"))

    ts.plot_ts(ts_results, containment = 0.9, save_plot = True,
               save_dir = "", save_name = "ts_map.png")

    assert Path("ts_map.png").exists()

    os.remove("ts_map.png")

def test_ts_fit_galactic():

    src_bkg_path = test_data.path / "ts_map_src_bkg.h5"
    bkg_path = test_data.path / "ts_map_bkg.h5"
    response_path = test_data.path / "test_precomputed_response.h5"

    src_bkg = Histogram.open(src_bkg_path).project(['Em', 'PsiChi', 'Phi'])
    bkg = Histogram.open(bkg_path).project(['Em', 'PsiChi', 'Phi'])

    ts = FastTSMap(data = src_bkg, bkg_model = bkg, orientation = None,
                   response_path = response_path, cds_frame = "galactic")

    index = -2.2
    K = 10 / u.cm / u.cm / u.s / u.keV
    piv = 100 * u.keV
    spectrum = Powerlaw()
    spectrum.index.value = index
    spectrum.K.value = K.value
    spectrum.piv.value = piv.value
    spectrum.K.unit = K.unit
    spectrum.piv.unit = piv.unit

    ts_results = ts.fit(nside = 1, energy_channel = [2,3],
                        spectrum = spectrum)

    assert np.allclose(ts_results,
                       [39.75648143, 39.61688953, 39.33241148,
                        39.40114511, 39.23694982, 39.23751789,
                        39.17731599, 38.75229555, 37.41499832,
                        37.06397938, 37.03113461, 37.18839154])

def test_moc_ts_fit():

    src_bkg_path = test_data.path / "ts_map_src_bkg.h5"
    bkg_path = test_data.path / "ts_map_bkg.h5"
    response_path = test_data.path / "test_full_detector_response.h5"

    orientation_path = test_data.path / "20280301_2s.fits"
    ori = SpacecraftFile.open(orientation_path)

    src_bkg = Histogram.open(src_bkg_path).project(['Em', 'PsiChi', 'Phi'])
    bkg = Histogram.open(bkg_path).project(['Em', 'PsiChi', 'Phi'])

    ts = MOCTSMap(data = src_bkg, bkg_model = bkg, orientation = ori,
                  response_path = response_path, cds_frame = "local")

    index = -2.2
    K = 10 / u.cm / u.cm / u.s / u.keV
    piv = 100 * u.keV
    spectrum = Powerlaw()
    spectrum.index.value = index
    spectrum.K.value = K.value
    spectrum.piv.value = piv.value
    spectrum.K.unit = K.unit
    spectrum.piv.unit = piv.unit

    # test default top-k strategy
    ts_results = ts.fit(max_nside = 2, energy_channel = [2,3],
                        spectrum = spectrum, cpu_cores = 1)

    ts_values, pixels = ts_results
    assert all(pixels == [
        6,  10, 11, 14, 16,
        20, 28, 32, 36, 48,
        52, 60, 17, 21, 29,
        33, 37, 49, 53, 61,
        18, 22, 30, 34, 38,
        50, 54, 62, 19, 23,
        31, 35, 39, 51, 55,
        63
    ])

    assert np.allclose(ts_values, [
        37.4339627, 37.2327797, 37.45064281, 38.83421248, 40.31750178,
        39.40582835, 39.78630473, 40.39347595, 40.10805456, 40.14551663,
        39.92706711, 40.07420191, 40.07720833, 38.76345776, 40.19657905,
        40.41047826, 39.9701239, 40.28032052, 39.8313809, 39.90376762,
        40.20425492, 39.93311807, 39.07431592, 40.3217545, 40.19353339,
        40.41652632, 39.61022259, 40.09740223, 39.81314166, 39.49561082,
        39.65452286, 39.940159, 39.61014067, 40.65108076, 39.92533792,
        40.2989865
    ])

    ts.plot_ts(*ts_results,
               skycoord = SkyCoord(l=0, b=0, unit=u.deg, frame="galactic"))

    ts.plot_ts(*ts_results, containment = 0.9, save_plot = True,
               save_dir = "", save_name = "ts_map.png")

    assert Path("ts_map.png").exists()

    os.remove("ts_map.png")

    # test containment strategy
    ts_results = ts.fit(max_nside = 2, energy_channel = [2,3],
                        spectrum = spectrum,
                        strategy=MOCTSMap.ContainmentStrategy(0.9))

    ts_values, pixels = ts_results
    assert all(pixels == [
        16, 20, 24, 28, 32,
        36, 40, 44, 48, 52,
        56, 60, 17, 21, 25,
        29, 33, 37, 41, 45,
        49, 53, 57, 61, 18,
        22, 26, 30, 34, 38,
        42, 46, 50, 54, 58,
        62, 19, 23, 27, 31,
        35, 39, 43, 47, 51,
        55, 59, 63
    ])

    assert np.allclose(ts_values, [
        40.31750178, 39.40582835, 37.39229509, 39.78630473, 40.39347595,
        40.10805456, 38.79974818, 38.86985166, 40.14551663, 39.92706711,
        39.19653532, 40.07420191, 40.07720833, 38.76345776, 37.46243839,
        40.19657905, 40.41047826, 39.9701239, 37.23577506, 39.28060584,
        40.28032052, 39.8313809, 39.25707764, 39.90376762, 40.20425492,
        39.93311807, 37.35958207, 39.07431592, 40.3217545, 40.19353339,
        38.98305396, 37.2428405,  40.41652632, 39.61022259, 39.22641582,
        40.09740223, 39.81314166, 39.49561082, 37.43747447, 39.65452286,
        39.940159, 39.61014067, 37.24223294, 37.35636565, 40.65108076,
        39.92533792, 37.24385822, 40.2989865
    ])

    # test padding strategy over a different containment threshold
    # (yields same result as previous test)
    ts_results = ts.fit(max_nside = 2, energy_channel = [2,3],
                        spectrum = spectrum,
                        strategy=MOCTSMap.PaddingStrategy(
                            MOCTSMap.ContainmentStrategy(0.5)))

    ts_values, pixels = ts_results
    assert all(pixels == [
        16, 20, 24, 28, 32,
        36, 40, 44, 48, 52,
        56, 60, 17, 21, 25,
        29, 33, 37, 41, 45,
        49, 53, 57, 61, 18,
        22, 26, 30, 34, 38,
        42, 46, 50, 54, 58,
        62, 19, 23, 27, 31,
        35, 39, 43, 47, 51,
        55, 59, 63
    ])

    assert np.allclose(ts_values, [
        40.31750178, 39.40582835, 37.39229509, 39.78630473, 40.39347595,
        40.10805456, 38.79974818, 38.86985166, 40.14551663, 39.92706711,
        39.19653532, 40.07420191, 40.07720833, 38.76345776, 37.46243839,
        40.19657905, 40.41047826, 39.9701239, 37.23577506, 39.28060584,
        40.28032052, 39.8313809, 39.25707764, 39.90376762, 40.20425492,
        39.93311807, 37.35958207, 39.07431592, 40.3217545, 40.19353339,
        38.98305396, 37.2428405,  40.41652632, 39.61022259, 39.22641582,
        40.09740223, 39.81314166, 39.49561082, 37.43747447, 39.65452286,
        39.940159, 39.61014067, 37.24223294, 37.35636565, 40.65108076,
        39.92533792, 37.24385822, 40.2989865
    ])
