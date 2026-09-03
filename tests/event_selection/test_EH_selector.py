from cosipy.event_selection.earth_horizon_selection import EHSelector
from cosipy.data_io.EmCDSUnbinnedData import TimeTagEmCDSEventDataInSCAndGalFrameFromDC3Fits
from cosipy import test_data
from cosipy.spacecraftfile import SpacecraftHistory


def test_EHevent_selector():
    data_file = test_data.path / "unbinned_data_MEGAlib_calc.hdf5"
    ori = SpacecraftHistory.open(test_data.path / "20280301_first_10sec.fits")
    EH_selector = EHSelector(ori,cutvalue=0.4,plotfsky=True)
    data = TimeTagEmCDSEventDataInSCAndGalFrameFromDC3Fits(data_file, selection=EH_selector)
