from cosipy import MOCTSMap
from cosipy import SpacecraftFile
from threeML import Powerlaw
from pathlib import Path
import os
from cosipy import test_data
import numpy as np
import astropy.units as u
from histpy import Histogram
import pytest
from astropy.coordinates import SkyCoord
from mhealpy import HealpixMap
from mhealpy.pixelfunc.moc import *
from mhealpy.pixelfunc.single import *
from cosipy import FastTSMap

response_path = test_data.path / "test_full_detector_response.h5"
src_bkg_path = test_data.path / "ts_map_src_bkg.h5"
bkg_path = test_data.path / "ts_map_bkg.h5"

orientation_path = test_data.path / "20280301_2s.ori"
ori = SpacecraftFile.parse_from_file(orientation_path)

src_bkg = Histogram.open(src_bkg_path).project(['Em', 'PsiChi', 'Phi'])
bkg = Histogram.open(bkg_path).project(['Em', 'PsiChi', 'Phi'])

index = -2.2
K = 10 / u.cm / u.cm / u.s / u.keV
piv = 100 * u.keV
spectrum = Powerlaw()
spectrum.index.value = index
spectrum.K.value = K.value
spectrum.piv.value = piv.value
spectrum.K.unit = K.unit
spectrum.piv.unit = piv.unit


def test_moc_ts_fit():
    
    moc_fit = MOCTSMap(data = src_bkg, 
                       bkg_model = bkg, 
                       response_path = response_path, 
                       orientation = ori,
                       cds_frame = "local")
    
    moc_map = moc_fit.moc_ts_fit(max_moc_order = 2, # this is the maximum order of the final map
                                 top_number = 8, # In each iterations, only the pixels with top 8 likelihood values will be split in the next iteration
                                 energy_channel = [2,3],  # The energy channel used to perform the fit.
                                 spectrum = spectrum)
    
    assert np.allclose(moc_map[:], 
                       np.array([40, 40, 40, 40, 39, 39, 40, 40, 40, 40, 40, 40, 39, 39, 38, 39, 39,
                                 37, 39, 40, 40, 40, 40, 39, 39, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                                 40, 40, 40, 39, 40, 39, 40, 40, 39, 40, 40, 40, 40, 39, 37, 37, 40,
                                 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 39, 39,
                                 39, 39, 38, 40, 40, 40, 39, 39, 40, 40, 39, 39, 40, 40, 39, 40]))
    
    # test the plotting function
    coord = SkyCoord(l=184.5551, b = -05.7877, unit = (u.deg, u.deg), frame = "galactic")
    moc_fit.plot_ts(moc_map = None, skycoord = coord, containment = 0.9, save_plot = True, save_dir = "", save_name = "ts_map.png", dpi = 300)
    generated_plot = Path("ts_map.png")
    assert generated_plot.exists
    generated_plot.unlink(missing_ok=True) # remove the generated file
    
    # test some other functions
    moc_fit.plot_ts(moc_map = None, skycoord = None, containment = None, save_plot = False, save_dir = "", save_name = "ts_map.png", dpi = 300)
    
    
def test_upscale_moc_map():
    
    # just to cover the raised errors
    # the main body is tested in test_moc_ts_fit
    
    moc_map = HealpixMap(nside = 4, scheme = "RING", dtype = int)
    
    with pytest.raises(TypeError):
        MOCTSMap.upscale_moc_map(moc_map,1,1)
        

def test_fill_up_moc_map():
    
    # just to cover the remaining lines
    
    # initialize the order 
    order = 0

    # initialize the 0th order moc map, which is equlivent to a 0th order single resolution map
    uniq = nest2uniq(1, np.arange(12))
    moc_map_ts = HealpixMap(data = np.repeat(0, 12), uniq = uniq)

    # make the 0th order fit over all pixels
    hypothesis_coords = MOCTSMap.uniq2skycoord(moc_map_ts.uniq)
    hypothesis_coords_list = [i for i in hypothesis_coords]  # have to split the SkyCoord object into SkyCoord object
    pixidx = MOCTSMap.uniq2pixidx(moc_map_ts, moc_map_ts.uniq)


    # here let's create a FastTSMap object for fitting the ts map in the following cells
    ts = FastTSMap(data = src_bkg, bkg_model = bkg, orientation = ori, 
                   response_path = response_path, cds_frame = "local", scheme = "RING")


    results = ts.parallel_ts_fit(hypothesis_coords = hypothesis_coords_list, energy_channel = [2,3], spectrum = spectrum, ts_scheme = "RING", cpu_cores = 56)
    
    moc_map_ts = MOCTSMap.fill_up_moc_map(1, moc_map_ts, results)