from pathlib import Path

import numpy as np

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits

from cosipy import test_data, SpacecraftHistory
from cosipy.response import FullDetectorResponse
from cosipy.response import RspArfRmfConverter

energy_edges = 10**np.linspace(2, 4, 10 + 1) # ten bins from 100 to 10000 KeV

def test_get_psr_rsp():
    response_path = test_data.path / "test_full_detector_response.h5"
    response = FullDetectorResponse.open(response_path)
    ori_path = test_data.path / "20280301_first_10sec.fits"
    ori = SpacecraftHistory.open(ori_path)
    target_coord = SkyCoord(l=184.5551, b=-05.7877,
                            unit=u.deg, frame="galactic")
    converter = RspArfRmfConverter(response, ori, target_coord)

    Ei_edges, Ei_lo, Ei_hi, \
        Em_edges, Em_lo, Em_hi, \
        areas, matrix = converter.get_psr_rsp()

    assert np.allclose(Ei_edges, energy_edges)

    assert np.allclose(Ei_lo,energy_edges[:-1])

    assert np.allclose(Ei_hi,energy_edges[1:])

    assert np.allclose(Em_edges,energy_edges)

    assert np.allclose(Em_lo,energy_edges[:-1])

    assert np.allclose(Em_hi,energy_edges[1:])

    assert np.allclose(areas,
                       [9.07843857, 35.97189941, 56.56903076, 58.62650146, 53.77538452,
                       46.66890564, 37.5471283, 25.56105347, 18.39017029, 10.23398438])

    assert np.allclose(matrix,
                       [[9.82146084e-01, 6.52569011e-02, 3.30404416e-02, 1.34480894e-02,
                         8.81888345e-03, 7.15653040e-03, 6.46192394e-03, 6.94540003e-03,
                         7.08964514e-03, 9.14793275e-03],
                        [1.78539176e-02, 9.27872598e-01, 1.37546435e-01, 8.62949491e-02,
                         5.51867969e-02, 4.31010798e-02, 3.65878679e-02, 3.69836800e-02,
                         3.58317234e-02, 4.46425714e-02],
                        [0.00000000e+00, 6.87047699e-03, 8.26300919e-01, 1.80046827e-01,
                         9.57962275e-02, 7.33733699e-02, 6.65754601e-02, 7.09649101e-02,
                         6.98765442e-02, 8.52129683e-02],
                        [0.00000000e+00, 0.00000000e+00, 3.11220298e-03, 7.18503475e-01,
                         1.78951785e-01, 7.96607733e-02, 6.17865399e-02, 6.78083599e-02,
                         7.75652826e-02, 1.12138554e-01],
                        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.70663046e-03,
                         6.60251915e-01, 1.66121393e-01, 6.80495277e-02, 5.26736267e-02,
                         4.41736877e-02, 4.98283207e-02],
                        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                         9.94389760e-04, 6.30014181e-01, 1.64825916e-01, 6.65939748e-02,
                         4.36101966e-02, 4.12763469e-02],
                        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                         0.00000000e+00, 5.72687772e-04, 5.95490038e-01, 2.90101558e-01,
                         1.56857163e-01, 9.14273262e-02],
                        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                         0.00000000e+00, 0.00000000e+00, 2.22623014e-04, 4.07899320e-01,
                         4.00614947e-01, 2.29005918e-01],
                        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 2.92088534e-05,
                         1.64380059e-01, 3.01594704e-01],
                        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                         0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                         7.36859079e-07, 3.57253887e-02]])


def test_get_arf(tmp_path):

    response_path = test_data.path / "test_full_detector_response.h5"
    response = FullDetectorResponse.open(response_path)
    ori_path = test_data.path / "20280301_first_10sec.fits"
    ori = SpacecraftHistory.open(ori_path)
    target_coord = SkyCoord(l=184.5551, b=-05.7877,
                            unit=u.deg, frame="galactic")
    converter = RspArfRmfConverter(response, ori, target_coord)

    _ = converter.get_psr_rsp()

    converter.write_arf(out_name = tmp_path / "test.arf", overwrite=True)

    fits_file = fits.open(tmp_path / "test.arf")

    assert np.allclose(fits_file[1].data.field("ENERG_LO"),energy_edges[:-1])

    assert np.allclose(fits_file[1].data.field("ENERG_HI"), energy_edges[1:])

    assert np.allclose(fits_file[1].data.field("SPECRESP"),
                       [ 9.07843857, 35.97189941, 56.56903076, 58.62650146, 53.77538452,
                         46.66890564, 37.5471283, 25.56105347, 18.39017029, 10.23398438])


def test_get_rmf(tmp_path):
    response_path = test_data.path / "test_full_detector_response.h5"
    response = FullDetectorResponse.open(response_path)
    ori_path = test_data.path / "20280301_first_10sec.fits"
    ori = SpacecraftHistory.open(ori_path)
    target_coord = SkyCoord(l=184.5551, b=-05.7877,
                            unit=u.deg, frame="galactic")
    converter = RspArfRmfConverter(response, ori, target_coord)

    _ = converter.get_psr_rsp()

    converter.write_rmf(out_name= tmp_path / "test.rmf", overwrite=True)

    fits_file = fits.open(tmp_path / "test.rmf")

    assert np.allclose(fits_file[1].data.field("ENERG_LO"),energy_edges[:-1])

    assert np.allclose(fits_file[1].data.field("ENERG_HI"),energy_edges[1:])

    assert np.allclose(fits_file[1].data.field("N_GRP"),np.ones(10))

    matrix_flattened = []
    for i in fits_file[1].data.field("MATRIX"):
        matrix_flattened += i.tolist()

    assert np.allclose(matrix_flattened,
                       [0.9821460843086243, 0.01785391755402088, 0.06525690108537674, 0.9278725981712341, 0.006870476994663477,
                        0.03304044157266617, 0.13754643499851227, 0.8263009190559387, 0.003112202975898981, 0.013448089361190796,
                        0.08629494905471802, 0.18004682660102844, 0.718503475189209, 0.0017066304571926594, 0.008818883448839188,
                        0.05518679693341255, 0.09579622745513916, 0.17895178496837616, 0.6602519154548645,  0.0009943897603079677,
                        0.007156530395150185, 0.043101079761981964, 0.07337336987257004, 0.07966077327728271, 0.16612139344215393,
                        0.630014181137085, 0.0005726877716369927, 0.0064619239419698715, 0.03658786788582802, 0.06657546013593674,
                        0.06178653985261917, 0.06804952770471573, 0.1648259162902832, 0.595490038394928, 0.00022262301354203373,
                        0.006945400033146143, 0.0369836799800396, 0.07096491008996964, 0.0678083598613739, 0.05267362669110298,
                        0.06659397482872009, 0.290101557970047, 0.40789932012557983, 2.920885344792623e-05, 0.0070896451361477375,
                        0.03583172336220741, 0.0698765441775322, 0.0775652825832367, 0.04417368769645691, 0.04361019656062126,
                        0.15685716271400452, 0.4006149470806122, 0.1643800586462021, 7.368590786427376e-07, 0.00914793275296688,
                        0.04464257135987282, 0.08521296828985214, 0.11213855445384979, 0.04982832074165344, 0.041276346892118454,
                        0.09142732620239258, 0.22900591790676117, 0.30159470438957214, 0.035725388675928116])



def test_get_pha(tmp_path):
    response_path = test_data.path / "test_full_detector_response.h5"
    response = FullDetectorResponse.open(response_path)
    ori_path = test_data.path / "20280301_first_10sec.fits"
    ori = SpacecraftHistory.open(ori_path)
    target_coord = SkyCoord(l=184.5551, b=-05.7877,
                            unit=u.deg, frame="galactic")
    converter = RspArfRmfConverter(response, ori, target_coord)

    _ = converter.get_psr_rsp()
    converter.write_arf(out_name=tmp_path / "test.arf", overwrite=True)
    converter.write_rmf(out_name=tmp_path / "test.rmf", overwrite=True)

    counts = np.array([0.01094232, 0.04728866, 0.06744612, 0.01393708, 0.05420688,
                       0.03141498, 0.01818584, 0.00717219, 0.00189568, 0.00010503]) * 1000

    errors = np.sqrt(counts)

    converter.write_pha(tmp_path / "test.pha", src_counts=counts, errors=errors, exposure_time=10, rmf_file = "test.rmf", arf_file = 'test.arf', bkg_file = 'text.pha', overwrite=True)

    fits_file = fits.open(tmp_path / "test.pha")

    assert np.allclose(fits_file[1].data.field("CHANNEL"),
                       [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    assert np.allclose(fits_file[1].data.field("COUNTS"),
                       [10, 47, 67, 13, 54, 31, 18, 7, 1, 0])

    assert np.allclose(fits_file[1].data.field("STAT_ERR"),
                       [3, 6, 8, 3, 7, 5, 4, 2, 1, 0])


def test_plot_arf(tmp_path):
    response_path = test_data.path / "test_full_detector_response.h5"
    response = FullDetectorResponse.open(response_path)
    ori_path = test_data.path / "20280301_first_10sec.fits"
    ori = SpacecraftHistory.open(ori_path)
    target_coord = SkyCoord(l=184.5551, b=-05.7877,
                            unit=u.deg, frame="galactic")
    converter = RspArfRmfConverter(response, ori, target_coord)

    converter.plot_arf()

def test_plot_rmf(tmp_path):
    response_path = test_data.path / "test_full_detector_response.h5"
    response = FullDetectorResponse.open(response_path)
    ori_path = test_data.path / "20280301_first_10sec.fits"
    ori = SpacecraftHistory.open(ori_path)
    target_coord = SkyCoord(l=184.5551, b=-05.7877,
                            unit=u.deg, frame="galactic")
    converter = RspArfRmfConverter(response, ori, target_coord)

    converter.plot_rmf()
