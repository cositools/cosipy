import numpy as np
from mhealpy import HealpixMap
from mhealpy.pixelfunc.moc import *
from mhealpy.pixelfunc.single import *
from cosipy import MOCTSMap, test_data, SpacecraftFile
import matplotlib.pyplot as plt
from copy import deepcopy
import astropy.units as u
from astropy.coordinates import SkyCoord
from pathlib import Path
from histpy import Histogram
from threeML import Powerlaw
import os

def test_upscale_moc_map():
    
    test_moc_map_path = test_data.path / "test_MOC_Map.fits"

    moc_map_ts = HealpixMap.read_map(test_moc_map_path)

    mother_uniq = np.array([ 5,  6,  9, 10, 13, 14])

    new_order = 1

    m_new, uniq_child_all = MOCTSMap.upscale_moc_map(moc_map_ts, uniq_mother = mother_uniq, new_order = new_order)

    assert np.allclose(uniq_child_all, 
                       np.array([20, 21, 22, 23, 24, 25, 26, 27, 36, 37, 38, 39, 40, 41, 42, 43, 52, 53, 54, 55, 56, 57, 58, 59]))

    assert np.allclose(m_new[:], 
                       np.array([205615, 368891, 368891, 368891, 368891, 356267, 356267, 356267, 
                                 356267, 199132, 172127, 268767, 268767, 268767, 268767, 888147, 
                                 888147, 888147, 888147, 252898, 215294, 346345, 346345, 346345, 
                                 346345, 378671, 378671, 378671, 378671, 234107]))
    
    
    
def test_uniq2skycoord():
    
    coord = MOCTSMap.uniq2skycoord(20)
    
    assert type(coord) is SkyCoord
    
    assert np.allclose(coord.l.deg, 135.0)
    
    assert np.allclose(coord.b.deg, 19.47122063449069)
    
    
def test_uniq2pixidx():
    
    test_moc_map_path = test_data.path / "test_MOC_Map.fits"

    moc_map_ts = HealpixMap.read_map(test_moc_map_path)

    uniq = np.array([9, 10])

    idx = MOCTSMap.uniq2pixidx(moc_map_ts, uniq)

    assert np.allclose(idx, 
                       np.array([5,6]))
    
    
def test_fill_up_moc_map():

    test_moc_map_path = test_data.path / "test_MOC_Map.fits"

    moc_map_ts = HealpixMap.read_map(test_moc_map_path)

    ts_fit_results = np.array([np.arange(12), np.repeat(1., moc_map_ts.uniq.size)]).T

    new_map = MOCTSMap.fill_up_moc_map(np.arange(12), moc_map_ts, ts_fit_results)

    assert np.allclose(new_map[:], 
                       np.repeat(1., moc_map_ts.uniq.size))
    
    
    
def test_moc_ts_fit():

    src_bkg_path = test_data.path / "ts_map_src_bkg.h5"
    bkg_path = test_data.path / "ts_map_bkg.h5"
    response_path = test_data.path / "test_full_detector_response.h5"

    orientation_path = test_data.path / "20280301_2s.ori"
    ori = SpacecraftFile.parse_from_file(orientation_path)

    src_bkg = Histogram.open(src_bkg_path).project(['Em', 'PsiChi', 'Phi'])
    bkg = Histogram.open(bkg_path).project(['Em', 'PsiChi', 'Phi'])

    # define a powerlaw spectrum
    index = -3
    K = 10**-3 / u.cm / u.cm / u.s / u.keV
    piv = 100 * u.keV
    spectrum = Powerlaw()
    spectrum.index.value = index
    spectrum.K.value = K.value
    spectrum.piv.value = piv.value 
    spectrum.K.unit = K.unit
    spectrum.piv.unit = piv.unit
    
    moc_fit = MOCTSMap(data = src_bkg, 
                       bkg_model = bkg, 
                       response_path = response_path, 
                       orientation = ori, # we don't need orientation since we are using the precomputed galactic reaponse
                       cds_frame = "local")
    
    moc_map = moc_fit.moc_ts_fit(max_moc_order = 1, # this is the maximum order of the final map
                                 top_number = 3, # In each iterations, only the pixels with top 8 likelihood values will be split in the next iteration
                                 energy_channel = [2,3],  # The energy channel used to perform the fit.
                                 spectrum = spectrum)
    
    assert np.allclose(moc_map[:], 
                       np.array([51.28650334, 51.30366672, 51.30492629, 51.11675696, 51.15264209, 
                                51.09963846, 51.18497425, 51.25122002, 51.30205529, 51.28966325, 
                                51.29978837, 51.29973519, 51.07010273, 51.09357204, 51.29127877, 
                                51.27676366, 51.147458  , 51.30377498, 51.30232323, 51.18908082,
                                51.26570991]))
    
    coord = SkyCoord(0, 0, unit = "deg", frame = "galactic")
    
    moc_fit.plot_ts(moc_map = moc_map, skycoord = coord, containment = 0.9, save_plot = True)
    
    plot_path = Path("ts_map.png")

    assert plot_path.exists()
    
    if plot_path.exists():
        os.remove(plot_path)