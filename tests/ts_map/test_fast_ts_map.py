from cosipy import test_data
from pytest import approx
from threeML import Powerlaw
from cosipy import FastTSMap, SpacecraftFile
from histpy import Histogram
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from pathlib import Path
import os

def test_parallel_ts_fit():

    src_bkg_path = test_data.path / "ts_map_src_bkg.h5"
    bkg_path = test_data.path / "ts_map_bkg.h5"
    response_path = test_data.path / "test_full_detector_response.h5"

    orientation_path = test_data.path / "20280301_2s.ori"
    ori = SpacecraftFile.parse_from_file(orientation_path)

    src_bkg = Histogram.open(src_bkg_path).project(['Em', 'PsiChi', 'Phi'])
    bkg = Histogram.open(bkg_path).project(['Em', 'PsiChi', 'Phi'])

    ts = FastTSMap(data = src_bkg, bkg_model = bkg, orientation = ori,
                   response_path = response_path, cds_frame = "local", scheme = "RING")

    hypothesis_coords = FastTSMap.get_hypothesis_coords(nside = 1)

    index = -2.2
    K = 10 / u.cm / u.cm / u.s / u.keV
    piv = 100 * u.keV
    spectrum = Powerlaw()
    spectrum.index.value = index
    spectrum.K.value = K.value
    spectrum.piv.value = piv.value
    spectrum.K.unit = K.unit
    spectrum.piv.unit = piv.unit

    ts_results = ts.parallel_ts_fit(hypothesis_coords = hypothesis_coords, energy_channel = [2,3], spectrum = spectrum, ts_scheme = "RING", cpu_cores = 2)

    assert np.allclose(ts_results[:,1],
                       [40.18628386, 39.59382592, 37.4339627,
                        39.88459849, 40.20132198, 39.86762314,
                        37.2327797,  37.4506428,  40.54884861,
                        39.69773074, 38.83421249, 39.99131767])

    ts.plot_ts(save_plot = True, save_dir = "", save_name = "ts_map.png", containment = 0.9)

    assert Path("ts_map.png").exists()

    os.remove("ts_map.png")


def test_get_psr_in_galactic():

    response_path = test_data.path / "test_precomputed_response.h5"

    index = -2.2
    K = 10 / u.cm / u.cm / u.s / u.keV
    piv = 100 * u.keV
    spectrum = Powerlaw()
    spectrum.index.value = index
    spectrum.K.value = K.value
    spectrum.piv.value = piv.value
    spectrum.K.unit = K.unit
    spectrum.piv.unit = piv.unit

    source_coord = SkyCoord(l = 50, b = -45, frame = "galactic", unit = "deg")

    psr = FastTSMap.get_psr_in_galactic(hypothesis_coord = source_coord,
                                        response_path = response_path,
                                        spectrum = spectrum)

    assert np.allclose(psr.project("Em")[:].to_value(u.cm*u.cm*u.s),
                       np.array([15.55747177,  65.20970322, 117.94904394, 149.31606904, 147.99053903,
                                 141.45116585, 139.85987318,  90.13002543,  27.58632961, 1.59177036]))

    assert np.allclose(psr.axes["Ei"].edges.value, 10**np.linspace(2, 4, 10 + 1))

    assert np.allclose(psr.axes["Em"].edges.value, psr.axes["Ei"].edges.value)

    assert np.allclose(psr.axes["Phi"].edges.value, np.linspace(0, 180, 180//6 + 1))

    assert psr.axes["PsiChi"].nside == 1


def test_get_ei_cds_array_galactic():

    response_path = test_data.path / "test_precomputed_response.h5"

    index = -2.2
    K = 10 / u.cm / u.cm / u.s / u.keV
    piv = 100 * u.keV
    spectrum = Powerlaw()
    spectrum.index.value = index
    spectrum.K.value = K.value
    spectrum.piv.value = piv.value
    spectrum.K.unit = K.unit
    spectrum.piv.unit = piv.unit

    source_coord = SkyCoord(l = 50, b = -45, frame = "galactic", unit = "deg")

    ei_cds = FastTSMap.get_ei_cds_array(hypothesis_coord = source_coord,
                                        energy_channel = [2,3],
                                        response_path = response_path,
                                        spectrum = spectrum,
                                        cds_frame = "galactic")

    ei_cds_read = np.load(test_data.path / "ei_cds_galactic.npy")

    assert np.allclose(ei_cds, ei_cds_read)


def test_get_ei_cds_array_detector():

    response_path = test_data.path / "test_full_detector_response.h5"
    orientation_path = test_data.path / "20280301_2s.ori"
    ori = SpacecraftFile.parse_from_file(orientation_path)

    index = -2.2
    K = 10 / u.cm / u.cm / u.s / u.keV
    piv = 100 * u.keV
    spectrum = Powerlaw()
    spectrum.index.value = index
    spectrum.K.value = K.value
    spectrum.piv.value = piv.value
    spectrum.K.unit = K.unit
    spectrum.piv.unit = piv.unit

    source_coord = SkyCoord(l = 50, b = -45, frame = "galactic", unit = "deg")

    ei_cds = FastTSMap.get_ei_cds_array(hypothesis_coord = source_coord,
                                        energy_channel = [2,3],
                                        response_path = response_path,
                                        spectrum = spectrum,
                                        cds_frame = "local",
                                        orientation = ori)

    ei_cds_read = np.load(test_data.path / "ei_cds_detector.npy")

    assert np.allclose(ei_cds, ei_cds_read)


def test_fast_ts_fit():

    response_path = test_data.path / "test_full_detector_response.h5"
    src_bkg_path = test_data.path / "ts_map_src_bkg.h5"
    bkg_path = test_data.path / "ts_map_bkg.h5"

    orientation_path = test_data.path / "20280301_2s.ori"
    ori = SpacecraftFile.parse_from_file(orientation_path)

    src_bkg = Histogram.open(src_bkg_path).project(['Em', 'PsiChi', 'Phi'])
    bkg = Histogram.open(bkg_path).project(['Em', 'PsiChi', 'Phi'])

    hypothesis_coord = FastTSMap.get_hypothesis_coords(nside = 1)[0]

    index = -2.2
    K = 10 / u.cm / u.cm / u.s / u.keV
    piv = 100 * u.keV
    spectrum = Powerlaw()
    spectrum.index.value = index
    spectrum.K.value = K.value
    spectrum.piv.value = piv.value
    spectrum.K.unit = K.unit
    spectrum.piv.unit = piv.unit

    ts_results = FastTSMap.fast_ts_fit(hypothesis_coord = hypothesis_coord,
                                 energy_channel = [1,4],
                                 data_cds_array = FastTSMap.get_cds_array(src_bkg, [1,4]).flatten() ,
                                 bkg_model_cds_array = FastTSMap.get_cds_array(bkg, [1,4]).flatten(),
                                 orientation = ori,
                                 response_path = response_path,
                                 spectrum = spectrum,
                                 cds_frame = "local",
                                 ts_nside = 1,
                                 ts_scheme = "RING")

    assert np.allclose(ts_results[1], 76.7966740541791)

    assert np.allclose(ts_results[2], 0.00032320804126248866)

    assert np.allclose(ts_results[3], 0.00011434249212330651)

    assert ts_results[4] is False

    assert np.allclose(ts_results[5], 14)
