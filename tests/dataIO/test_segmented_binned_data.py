# Imports:
import os
import numpy as np
import pytest
import astropy.units as u
from astropy.coordinates import SkyCoord

from histpy import Histogram, Axis, Axes, HealpixAxis

from cosipy import BinnedData, test_data
from cosipy.data_io import EmCDSBinnedData, SegmentedEmCDSBinnedData

# Need to change the backend, otherwise testing plots can take long
# time (particularly with ssh):
import matplotlib
matplotlib.use('Svg')


def _make_histogram(extra_label=None, extra_nbins=3):

    em_axis = Axis([100, 200, 300] * u.keV, label='Em')
    phi_axis = Axis([0, 90, 180] * u.deg, label='Phi')
    psichi_axis = HealpixAxis(nside=1, scheme='ring', coordsys='galactic', label='PsiChi')

    axes = []

    if extra_label is not None:
        axes.append(Axis(np.arange(extra_nbins + 1), label=extra_label))

    axes += [em_axis, phi_axis, psichi_axis]

    return Histogram(Axes(axes, copy_axes=False))


def test_segmented_em_cds_binned_data_accepts_one_extra_axis():

    hist = _make_histogram(extra_label='Time')

    data = SegmentedEmCDSBinnedData(hist)

    assert data.segment_label == 'Time'
    assert set(data.axes.labels) == {'Time', 'Em', 'Phi', 'PsiChi'}
    assert data.data is hist


def test_segmented_em_cds_binned_data_explicit_label():

    hist = _make_histogram(extra_label='ScAtt')

    data = SegmentedEmCDSBinnedData(hist, segment_label='ScAtt')

    assert data.segment_label == 'ScAtt'


def test_segmented_em_cds_binned_data_rejects_wrong_explicit_label():

    hist = _make_histogram(extra_label='ScAtt')

    with pytest.raises(ValueError):
        SegmentedEmCDSBinnedData(hist, segment_label='Depth')


def test_segmented_em_cds_binned_data_rejects_no_extra_axis():

    hist = _make_histogram(extra_label=None)

    with pytest.raises(ValueError):
        SegmentedEmCDSBinnedData(hist)


def test_segmented_em_cds_binned_data_rejects_two_extra_axes():

    em_axis = Axis([100, 200, 300] * u.keV, label='Em')
    phi_axis = Axis([0, 90, 180] * u.deg, label='Phi')
    psichi_axis = HealpixAxis(nside=1, scheme='ring', coordsys='galactic', label='PsiChi')
    time_axis = Axis(np.arange(4), label='Time')
    depth_axis = Axis(np.arange(3), label='Depth')

    hist = Histogram(Axes([time_axis, depth_axis, em_axis, phi_axis, psichi_axis], copy_axes=False))

    with pytest.raises(ValueError):
        SegmentedEmCDSBinnedData(hist)


def test_segmented_em_cds_binned_data_to_emcds():

    hist = _make_histogram(extra_label='Depth', extra_nbins=2)
    hist[:] = np.arange(hist.contents.size).reshape(hist.contents.shape)

    data = SegmentedEmCDSBinnedData(hist)

    emcds = data.to_emcds()

    assert isinstance(emcds, EmCDSBinnedData)
    assert set(emcds.axes.labels) == {'Em', 'Phi', 'PsiChi'}
    assert np.allclose(emcds.data.contents, hist.project('Em', 'Phi', 'PsiChi').contents)


def test_get_segmented_em_cds_matches_get_em_cds(tmp_path):

    # Load file with dataIO class.
    # Note: this is the first 10 seconds of the crab sim from mini-DC2.
    yaml = os.path.join(test_data.path, "inputs_crab.yaml")
    analysis = BinnedData(yaml)
    analysis.data_file = os.path.join(test_data.path, analysis.data_file)
    analysis.ori_file = os.path.join(test_data.path, analysis.ori_file)

    analysis.read_tra()
    analysis.tmax = 1835478000.1
    analysis.select_data_time(output_name=tmp_path / "temp_unbinned")

    analysis.tmax = 1835478010.0
    analysis.nside = 1
    analysis.phi_pix_size = 90
    analysis.get_binned_data(unbinned_data=tmp_path / "temp_unbinned.hdf5",
                             output_name=tmp_path / "temp_binned_data")

    segmented = analysis.get_segmented_em_cds('Time')
    plain = analysis.get_em_cds()

    assert isinstance(segmented, SegmentedEmCDSBinnedData)
    assert segmented.segment_label == 'Time'
    assert set(segmented.axes.labels) == {'Time', 'Em', 'Phi', 'PsiChi'}

    # Summing (projecting out) the segment axis must reproduce the
    # plain, non-segmented Em/Phi/PsiChi histogram.
    reprojected = segmented.data.project('Em', 'Phi', 'PsiChi').to_dense(copy=False)
    plain_dense = plain.data.to_dense(copy=False)

    assert np.allclose(reprojected.contents, plain_dense.contents)
