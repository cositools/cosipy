from cosipy import test_data
from pytest import approx
from threeML import Powerlaw
from cosipy import FastTSMap, SpacecraftFile
from histpy import Histogram
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u


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