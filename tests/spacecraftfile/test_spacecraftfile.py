from cosipy import test_data
from pytest import approx
from cosipy import SpacecraftFile
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
import os
from pathlib import Path
from astropy.time import Time

energy_edges = 10**np.linspace(2, 4, 10 + 1) # ten bins from 100 to 10000 KeV

def test_get_time():

    ori_path = test_data.path / "20280301_first_10sec.ori"

    ori = SpacecraftFile.parse_from_file(ori_path)

    start = 1835478000.0
    assert np.allclose(ori.get_time().value,
                       np.linspace(start, start + 10, 11))


def test_get_time_delta():

    ori_path = test_data.path / "20280301_first_10sec.ori"
    ori = SpacecraftFile.parse_from_file(ori_path)
    time_delta = ori.get_time_delta()
    time_delta.format = "sec"

    assert np.allclose(time_delta.value, np.ones(10))


def test_get_attitude():

    ori_path = test_data.path / "20280301_first_10sec.ori"
    ori = SpacecraftFile.parse_from_file(ori_path)

    attitude = ori.get_attitude()

    matrix = np.array([[[0.215904, -0.667290, -0.712818],
                        [0.193436, 0.744798, -0.638638],
                        [0.957062, 0.000000, 0.289883]],

                       [[0.216493, -0.667602, -0.712347],
                        [0.194127, 0.744518, -0.638754],
                        [0.956789, 0.000000, 0.290783]],

                       [[0.217081, -0.667914, -0.711875],
                        [0.194819, 0.744238, -0.638870],
                        [0.956515, -0.000000, 0.291683]],

                       [[0.217669, -0.668227, -0.711402],
                        [0.195511, 0.743958, -0.638985],
                        [0.956240, 0.000000, 0.292582]],

                       [[0.218255, -0.668539, -0.710929],
                        [0.196204, 0.743677, -0.639100],
                        [0.955965, 0.000000, 0.293481]],

                       [[0.218841, -0.668852, -0.710455],
                        [0.196897, 0.743396, -0.639214],
                        [0.955688, -0.000000, 0.294380]],

                       [[0.219426, -0.669165, -0.709980],
                        [0.197590, 0.743114, -0.639327],
                        [0.955411, 0.000000, 0.295279]],

                       [[0.220010, -0.669477, -0.709504],
                        [0.198284, 0.742833, -0.639440],
                        [0.955133, -0.000000, 0.296177]],

                       [[0.220594, -0.669790, -0.709027],
                        [0.198978, 0.742551, -0.639552],
                        [0.954854, 0.000000, 0.297075]],

                       [[0.221176, -0.670103, -0.708550],
                        [0.199673, 0.742268, -0.639663],
                        [0.954574, -0.000000, 0.297973]],

                       [[0.221758, -0.670416, -0.708072],
                        [0.200368, 0.741986, -0.639773],
                        [0.954294, -0.000000, 0.298871]]])

    assert np.allclose(attitude.as_matrix(), matrix)


def test_get_target_in_sc_frame():

    ori_path = test_data.path / "20280301_first_10sec.ori"
    ori = SpacecraftFile.parse_from_file(ori_path)

    target_name = "Crab"
    target_coord = SkyCoord(l=184.5551, b = -05.7877, unit = (u.deg, u.deg), frame = "galactic")

    path_in_sc = ori.get_target_in_sc_frame(target_name, target_coord)

    assert np.allclose(path_in_sc.lon.deg,
                       np.array([118.393522, 118.425255, 118.456868, 118.488362, 118.519735,
                                 118.550989, 118.582124, 118.613139, 118.644035, 118.674813, 118.705471]))

    assert np.allclose(path_in_sc.lat.deg,
                       np.array([46.733430, 46.687559, 46.641664, 46.595745, 46.549801, 46.503833,
                                 46.457841, 46.411825, 46.365785, 46.319722, 46.273634]))


def test_get_dwell_map():

    response_path =test_data.path / "test_full_detector_response.h5"
    ori_path = test_data.path / "20280301_first_10sec.ori"
    ori = SpacecraftFile.parse_from_file(ori_path)

    target_name = "Crab"
    target_coord = SkyCoord(l=184.5551, b = -05.7877, unit = (u.deg, u.deg), frame = "galactic")
    path_in_sc = ori.get_target_in_sc_frame(target_name, target_coord)

    dwell_map = ori.get_dwell_map(response = response_path)

    assert np.allclose(dwell_map[:].value,
                       np.array([1.895057, 7.615584, 0.244679, 0.244679, 0.000000, 0.000000,
                                0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]))


def test_get_psr_rsp():

    response_path = test_data.path / "test_full_detector_response.h5"
    ori_path = test_data.path / "20280301_first_10sec.ori"
    ori = SpacecraftFile.parse_from_file(ori_path)

    target_name = "Crab"
    target_coord = SkyCoord(l=184.5551, b = -05.7877, unit = (u.deg, u.deg), frame = "galactic")
    path_in_sc = ori.get_target_in_sc_frame(target_name, target_coord)

    dwell_map = ori.get_dwell_map(response = response_path)

    Ei_edges, Ei_lo, Ei_hi, Em_edges, Em_lo, Em_hi, areas, matrix = ori.get_psr_rsp()

    assert np.allclose(Ei_edges, energy_edges)

    assert np.allclose(Ei_lo, energy_edges[:-1])

    assert np.allclose(Ei_hi, energy_edges[1:])

    assert np.allclose(Em_edges, energy_edges)

    assert np.allclose(Em_lo, energy_edges[:-1])

    assert np.allclose(Em_hi, energy_edges[1:])

    assert np.allclose(areas,
                       np.array([ 9.07843857, 35.97189941, 56.56903076, 58.62650146, 53.77538452,
                                  46.66890564, 37.5471283, 25.56105347, 18.39017029, 10.23398438]))

    assert np.allclose(matrix,
                       np.array([[9.82146084e-01, 6.52569011e-02, 3.30404416e-02, 1.34480894e-02,
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
                                  7.36859079e-07, 3.57253887e-02]]))


def test_get_arf():

    response_path = test_data.path / "test_full_detector_response.h5"
    ori_path = test_data.path / "20280301_first_10sec.ori"
    ori = SpacecraftFile.parse_from_file(ori_path)

    target_name = "Crab"
    target_coord = SkyCoord(l=184.5551, b = -05.7877, unit = (u.deg, u.deg), frame = "galactic")

    path_in_sc = ori.get_target_in_sc_frame(target_name, target_coord)

    dwell_map = ori.get_dwell_map(response = response_path)

    _ = ori.get_psr_rsp()

    ori.get_arf(out_name = "test")

    fits_file = fits.open("test.arf")

    assert np.allclose(fits_file[1].data.field("ENERG_LO"), energy_edges[:-1])

    assert np.allclose(fits_file[1].data.field("ENERG_HI"), energy_edges[1:])

    assert np.allclose(fits_file[1].data.field("SPECRESP"),
                       np.array([ 9.07843857, 35.97189941, 56.56903076, 58.62650146, 53.77538452,
                                  46.66890564, 37.5471283, 25.56105347, 18.39017029, 10.23398438]))

    os.remove("test.arf")

def test_get_rmf():

    response_path = test_data.path / "test_full_detector_response.h5"
    ori_path = test_data.path / "20280301_first_10sec.ori"
    ori = SpacecraftFile.parse_from_file(ori_path)

    target_name = "Crab"
    target_coord = SkyCoord(l=184.5551, b = -05.7877, unit = (u.deg, u.deg), frame = "galactic")

    path_in_sc = ori.get_target_in_sc_frame(target_name, target_coord)

    dwell_map = ori.get_dwell_map(response = response_path)

    _ = ori.get_psr_rsp()

    ori.get_rmf(out_name = "test")

    fits_file = fits.open("test.rmf")

    assert np.allclose(fits_file[1].data.field("ENERG_LO"), energy_edges[:-1])

    assert np.allclose(fits_file[1].data.field("ENERG_HI"), energy_edges[1:])

    assert np.allclose(fits_file[1].data.field("N_GRP"), np.ones(10))

    matrix_flattened = []
    for i in fits_file[1].data.field("MATRIX"):
        matrix_flattened += i.tolist()

    assert np.allclose(matrix_flattened,
                       np.array([0.9821460843086243,   0.01785391755402088,   0.06525690108537674,   0.9278725981712341,    0.006870476994663477,
                                 0.03304044157266617,  0.13754643499851227,   0.8263009190559387,    0.003112202975898981,  0.013448089361190796,
                                 0.08629494905471802,  0.18004682660102844,   0.718503475189209,     0.0017066304571926594, 0.008818883448839188,
                                 0.05518679693341255,  0.09579622745513916,   0.17895178496837616,   0.6602519154548645,    0.0009943897603079677,
                                 0.007156530395150185, 0.043101079761981964,  0.07337336987257004,   0.07966077327728271,   0.16612139344215393,
                                 0.630014181137085,    0.0005726877716369927, 0.0064619239419698715, 0.03658786788582802,   0.06657546013593674,
                                 0.06178653985261917,  0.06804952770471573,   0.1648259162902832,    0.595490038394928,     0.00022262301354203373,
                                 0.006945400033146143, 0.0369836799800396,    0.07096491008996964,   0.0678083598613739,    0.05267362669110298,
                                 0.06659397482872009,  0.290101557970047,     0.40789932012557983,   2.920885344792623e-05, 0.0070896451361477375,
                                 0.03583172336220741,  0.0698765441775322,    0.0775652825832367,    0.04417368769645691,   0.04361019656062126,
                                 0.15685716271400452,  0.4006149470806122,    0.1643800586462021,    7.368590786427376e-07, 0.00914793275296688,
                                 0.04464257135987282,  0.08521296828985214,   0.11213855445384979,   0.04982832074165344,   0.041276346892118454,
                                 0.09142732620239258,  0.22900591790676117,   0.30159470438957214,   0.035725388675928116]))

    os.remove("test.rmf")


def test_get_pha():

    response_path = test_data.path / "test_full_detector_response.h5"
    ori_path = test_data.path / "20280301_first_10sec.ori"
    ori = SpacecraftFile.parse_from_file(ori_path)

    target_name = "Crab"
    target_coord = SkyCoord(l=184.5551, b = -05.7877, unit = (u.deg, u.deg), frame = "galactic")

    path_in_sc = ori.get_target_in_sc_frame(target_name, target_coord)
    dwell_map = ori.get_dwell_map(response = response_path)
    _ = ori.get_psr_rsp()
    ori.get_arf(out_name = "test")
    ori.get_rmf(out_name = "test")

    counts = np.array([0.01094232, 0.04728866, 0.06744612, 0.01393708, 0.05420688,
                       0.03141498, 0.01818584, 0.00717219, 0.00189568, 0.00010503])*1000

    errors = np.sqrt(counts)

    ori.get_pha(src_counts=counts, errors=errors, exposure_time=10)

    os.remove("test.arf")
    os.remove("test.rmf")

    fits_file = fits.open("test.pha")
    os.remove("test.pha")

    assert np.allclose(fits_file[1].data.field("CHANNEL"),
                           np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))

    assert np.allclose(fits_file[1].data.field("COUNTS"),
                           np.array([10, 47, 67, 13, 54, 31, 18,  7,  1,  0]))

    assert np.allclose(fits_file[1].data.field("STAT_ERR"),
                           np.array([3, 6, 8, 3, 7, 5, 4, 2, 1, 0]))

def test_plot_arf():

    response_path = test_data.path / "test_full_detector_response.h5"
    ori_path = test_data.path / "20280301_first_10sec.ori"
    ori = SpacecraftFile.parse_from_file(ori_path)

    target_name = "Crab"
    target_coord = SkyCoord(l=184.5551, b = -05.7877, unit = (u.deg, u.deg), frame = "galactic")

    path_in_sc = ori.get_target_in_sc_frame(target_name, target_coord)
    dwell_map = ori.get_dwell_map(response = response_path)
    _ = ori.get_psr_rsp()
    ori.get_arf(out_name = "test")

    ori.plot_arf()

    assert Path("Effective_area_for_test.png").exists()

    os.remove("test.arf")
    os.remove("Effective_area_for_test.png")

def test_plot_rmf():

    response_path = test_data.path / "test_full_detector_response.h5"
    ori_path = test_data.path / "20280301_first_10sec.ori"
    ori = SpacecraftFile.parse_from_file(ori_path)

    target_name = "Crab"
    target_coord = SkyCoord(l=184.5551, b = -05.7877, unit = (u.deg, u.deg), frame = "galactic")

    path_in_sc = ori.get_target_in_sc_frame(target_name, target_coord)
    dwell_map = ori.get_dwell_map(response = response_path)
    _ = ori.get_psr_rsp()
    ori.get_rmf(out_name = "test")

    ori.plot_rmf()

    assert Path("Redistribution_matrix_for_test.png").exists()

    os.remove("test.rmf")
    os.remove("Redistribution_matrix_for_test.png")

def test_source_interval():

    response_path = test_data.path / "test_full_detector_response.h5"
    ori_path = test_data.path / "20280301_first_10sec.ori"
    ori = SpacecraftFile.parse_from_file(ori_path)

    new_ori = ori.source_interval(Time(ori._load_time[0]+0.1, format = "unix"),
                                  Time(ori._load_time[0]+2.1, format = "unix"))

    assert np.allclose(new_ori._load_time,
                       np.array([1.835478e+09, 1.835478e+09, 1.835478e+09, 1.835478e+09]))

    assert np.allclose(new_ori._x_direction.flatten(),
                       np.array([41.86062093, 73.14368765, 41.88225011, 73.09517927,
                                 41.90629597, 73.0412838 , 41.9087019 , 73.03589454]))

    assert np.allclose(new_ori._z_direction.flatten(),
                       np.array([221.86062093,  16.85631235, 221.88225011,  16.90482073,
                                221.90629597,  16.9587162 , 221.9087019 ,  16.96410546]))
