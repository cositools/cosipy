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
                       [51.30709447, 51.16302889, 51.11429069, 
                        51.19306142, 51.27823575, 51.30579709, 
                        51.09094512, 51.10914182, 51.30271261, 
                        51.27412572, 51.15872202, 51.29249638])
    
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

    assert np.allclose(psr.project("Em")[:].value, 
                       np.array([ 0.65390537,  4.13318094,  8.93003058,  5.44094247, 11.22897029, 
                                 11.33399241, 11.16676396,  7.02890392,  2.4011114 ,  0.2157772 ]))

    assert np.allclose(psr.axes[0].edges[:].value, 
                       np.array([ 150.,  220.,  325.,  480.,  520.,  765., 1120., 1650., 2350., 3450., 5000.]))

    assert np.allclose(psr.axes[1].edges[:].value, 
                       np.array([ 150.,  220.,  325.,  480.,  520.,  765., 1120., 1650., 2350., 3450., 5000.]))

    assert np.allclose(psr.axes[2].edges[:].value, 
                       np.array([  0.,   6.,  12.,  18.,  24.,  30.,  36.,  42.,  48.,  54.,  60.,
                                 66.,  72.,  78.,  84.,  90.,  96., 102., 108., 114., 120., 126.,
                                 132., 138., 144., 150., 156., 162., 168., 174., 180.]))

    assert np.allclose(psr.axes[3].edges, 
                       np.array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12]))
    
    
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

    assert np.allclose(ts_results[1], 100.43007288938203)

    assert np.allclose(ts_results[2], 0.023332803099804223)

    assert np.allclose(ts_results[3], 0.008228874289752388)

    assert ts_results[4] is False

    assert np.allclose(ts_results[5], 15)