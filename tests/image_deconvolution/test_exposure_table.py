from histpy import Histogram

from cosipy import test_data
from cosipy.image_deconvolution import SpacecraftAttitudeExposureTable
from cosipy.spacecraftfile import SpacecraftFile

def test_exposure_table(tmp_path):

    nside = 1

    ori = SpacecraftFile.parse_from_file(test_data.path / "20280301_first_10sec.ori")

    assert SpacecraftAttitudeExposureTable.analyze_orientation(ori, nside=nside, start=None, stop=ori.get_time()[-1], min_exposure=0, min_num_pointings=1) == None

    assert SpacecraftAttitudeExposureTable.analyze_orientation(ori, nside=nside, start=ori.get_time()[0], stop=None, min_exposure=0, min_num_pointings=1) == None

    exposure_table = SpacecraftAttitudeExposureTable.from_orientation(ori, nside=nside, 
                                                                      start=ori.get_time()[0], stop=ori.get_time()[-1], 
                                                                      min_exposure=0, min_num_pointings=1)

    exposure_table_nest = SpacecraftAttitudeExposureTable.from_orientation(ori, nside=nside, scheme = 'nested',
                                                                           start=ori.get_time()[0], stop=ori.get_time()[-1], 
                                                                           min_exposure=0, min_num_pointings=1)

    exposure_table_badscheme = SpacecraftAttitudeExposureTable.from_orientation(ori, nside=nside, scheme = None,
                                                                                start=ori.get_time()[0], stop=ori.get_time()[-1], 
                                                                                min_exposure=0, min_num_pointings=1)

    exposure_table.save_as_fits(tmp_path / "exposure_table_test_nside1_ring.fits")
    
    assert exposure_table == SpacecraftAttitudeExposureTable.from_fits(tmp_path / "exposure_table_test_nside1_ring.fits")

    map_pointing_zx = exposure_table.calc_pointing_trajectory_map()

    assert map_pointing_zx == Histogram.open(test_data.path / "image_deconvolution/map_pointing_zx_test_nside1_ring.hdf5")
