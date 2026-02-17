import numpy as np
import astropy.units as u
from astropy.time import Time
from histpy import Histogram

from cosipy import test_data
from cosipy.image_deconvolution import SpacecraftAttitudeExposureTable, TimeBinnedExposureTable
from cosipy.spacecraftfile import SpacecraftFile
from cosipy import response
from cosipy import BinnedData

def test_scatt_exposure_table(tmp_path):

    nside = 1

    ori = SpacecraftFile.open(test_data.path / "20280301_first_10sec.fits")

    assert SpacecraftAttitudeExposureTable.analyze_orientation(ori, nside=nside, start=None, stop=ori.get_time()[-1], min_livetime=0, min_num_pointings=1) == None

    assert SpacecraftAttitudeExposureTable.analyze_orientation(ori, nside=nside, start=ori.get_time()[0], stop=None, min_livetime=0, min_num_pointings=1) == None


    exposure_table = SpacecraftAttitudeExposureTable.from_orientation(ori, nside=nside,
                                                                      start=ori.get_time()[0], stop=ori.get_time()[-1],
                                                                      min_livetime=0, min_num_pointings=1)

    exposure_table_nest = SpacecraftAttitudeExposureTable.from_orientation(ori, nside=nside, scheme = 'nested',
                                                                           start=ori.get_time()[0], stop=ori.get_time()[-1],
                                                                           min_livetime=0, min_num_pointings=1)

    exposure_table_badscheme = SpacecraftAttitudeExposureTable.from_orientation(ori, nside=nside, scheme = None,
                                                                                start=ori.get_time()[0], stop=ori.get_time()[-1],
                                                                                min_livetime=0, min_num_pointings=1)

    exposure_table.save_as_fits(tmp_path / "exposure_table_test_nside1_ring.fits")

    assert exposure_table == SpacecraftAttitudeExposureTable.from_fits(tmp_path / "exposure_table_test_nside1_ring.fits")

    map_pointing_zx = exposure_table.calc_pointing_trajectory_map()

    assert np.all(map_pointing_zx.contents == Histogram.open(test_data.path / "image_deconvolution/map_pointing_zx_test_nside1_ring.hdf5").contents)

    # test_generating_histogram
    full_detector_response = response.FullDetectorResponse.open(test_data.path / "test_full_detector_response.h5")

    analysis = BinnedData(test_data.path / "inputs_crab.yaml")

    analysis.cosi_dataset = analysis.get_dict_from_hdf5(test_data.path / "unbinned_data_MEGAlib_calc.hdf5")

    # modify the following parameters for unit test
    analysis.energy_bins = full_detector_response.axes['Em'].edges.to(u.keV).value
    analysis.nside = full_detector_response.axes['PsiChi'].nside
    analysis.phi_pix_size = full_detector_response.axes['Phi'].widths[0].to(u.deg).value
    analysis.time_bins = 10 #s

    # NOTE: test_data.path / "unbinned_data_MEGAlib_calc.hdf5" is written in a old format!!!
    _ = analysis.cosi_dataset.pop('Xpointings')
    analysis.cosi_dataset['Xpointings (glon,glat)'] = _

    _ = analysis.cosi_dataset.pop('Ypointings')
    analysis.cosi_dataset['Ypointings (glon,glat)'] = _

    _ = analysis.cosi_dataset.pop('Zpointings')
    analysis.cosi_dataset['Zpointings (glon,glat)'] = _

    binned_signal = exposure_table.get_binned_data(analysis, psichi_binning = 'local', sparse = False)

    binned_signal_ref = Histogram.open(test_data.path / "image_deconvolution" / 'test_event_histogram_localCDS_scatt.h5')

    assert np.all(binned_signal.contents == binned_signal_ref.contents)

def test_time_binned_exposure_table(tmp_path):
    ori = SpacecraftFile.open(test_data.path / "20280301_first_10sec.fits")

    tstart_list = Time([1835478000.0], scale='utc', format='unix')
    tstop_list  = Time([1835478005.0], scale='utc', format='unix')

    exposure_table = TimeBinnedExposureTable.from_orientation(ori, tstart_list = tstart_list, tstop_list = tstop_list)

    exposure_table.save_as_fits(tmp_path / "exposure_table_test_time_binning.fits")

    assert exposure_table == TimeBinnedExposureTable.from_fits(tmp_path / "exposure_table_test_time_binning.fits")

    full_detector_response = response.FullDetectorResponse.open(test_data.path / "test_full_detector_response.h5")

    analysis = BinnedData(test_data.path / "inputs_crab.yaml")

    analysis.cosi_dataset = analysis.get_dict_from_hdf5(test_data.path / "unbinned_data_MEGAlib_calc.hdf5")

    # modify the following parameters for unit test
    analysis.energy_bins = full_detector_response.axes['Em'].edges.to(u.keV).value
    analysis.nside = full_detector_response.axes['PsiChi'].nside
    analysis.phi_pix_size = full_detector_response.axes['Phi'].widths[0].to(u.deg).value
    analysis.time_bins = 10 #s

    # NOTE: test_data.path / "unbinned_data_MEGAlib_calc.hdf5" is written in a old format!!!
    _ = analysis.cosi_dataset.pop('Xpointings')
    analysis.cosi_dataset['Xpointings (glon,glat)'] = _

    _ = analysis.cosi_dataset.pop('Ypointings')
    analysis.cosi_dataset['Ypointings (glon,glat)'] = _

    _ = analysis.cosi_dataset.pop('Zpointings')
    analysis.cosi_dataset['Zpointings (glon,glat)'] = _

    binned_signal = exposure_table.get_binned_data(analysis, psichi_binning = 'local', sparse = False)

    binned_signal_ref = Histogram.open(test_data.path / "image_deconvolution" / 'test_event_histogram_localCDS_time.h5')

    assert np.all(binned_signal.contents == binned_signal_ref.contents)
