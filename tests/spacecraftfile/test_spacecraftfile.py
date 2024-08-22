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

def test_get_time():

    ori_path = test_data.path / "20280301_first_10sec.ori"
    
    ori = SpacecraftFile.parse_from_file(ori_path)
    
    assert np.allclose(ori.get_time().value,
                       [1835478000.0, 1835478001.0, 1835478002.0,
                        1835478003.0, 1835478004.0, 1835478005.0,
                        1835478006.0, 1835478007.0, 1835478008.0,
                        1835478009.0, 1835478010.0])


def test_get_time_delta():

    ori_path = test_data.path / "20280301_first_10sec.ori"
    ori = SpacecraftFile.parse_from_file(ori_path)
    time_delta = ori.get_time_delta()
    time_delta.format = "sec"

    assert np.allclose(time_delta.value, np.array([1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 
                                                   1.000000, 1.000000, 1.000000, 1.000000, 1.000000]))



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
    
    assert np.allclose(Ei_edges, 
                       np.array([150., 220., 325., 480., 520., 765., 1120., 1650., 2350., 3450., 5000.]))
    
    assert np.allclose(Ei_lo, 
                       np.array([150., 220., 325., 480., 520., 765., 1120., 1650., 2350., 3450.]))
    
    assert np.allclose(Ei_hi, 
                       np.array([220.,  325.,  480.,  520.,  765., 1120., 1650., 2350., 3450., 5000.]))
    
    assert np.allclose(Em_edges, 
                       np.array([150.,  220.,  325.,  480.,  520.,  765., 1120., 1650., 2350., 3450., 5000.]))
    
    assert np.allclose(Em_lo, 
                       np.array([150.,  220.,  325.,  480.,  520.,  765., 1120., 1650., 2350., 3450.]))
    
    assert np.allclose(Em_hi, 
                       np.array([220.,  325.,  480.,  520.,  765., 1120., 1650., 2350., 3450., 5000.]))
    
    assert np.allclose(areas, 
                       np.array([0.06089862, 0.4563752 , 1.1601573 , 1.6237522 , 2.0216975 , 
                                 2.2039971 , 2.0773466 , 1.7005537 , 1.1626455 , 0.80194914]))
    
    assert np.allclose(matrix, 
                       np.array([[9.80996430e-01, 4.68325317e-02, 1.82471890e-02, 9.86817386e-03,
                                  5.82037494e-03, 3.47572053e-03, 2.80415593e-03, 3.13903880e-03,
                                  4.89909900e-03, 6.68705115e-03],
                                  [1.90035217e-02, 9.44634676e-01, 1.28470331e-01, 9.38407257e-02,
                                  4.32382338e-02, 2.23877952e-02, 1.63043533e-02, 1.73287615e-02,
                                  2.80312393e-02, 3.78256924e-02],
                                   [0.00000000e+00, 8.53277557e-03, 8.48568857e-01, 2.18858123e-01,
                                  1.85861006e-01, 7.39495233e-02, 4.45922092e-02, 4.06639054e-02,
                                  6.96888119e-02, 9.27841067e-02],
                                   [0.00000000e+00, 0.00000000e+00, 4.71363496e-03, 6.62667990e-01,
                                  6.19757064e-02, 2.71992888e-02, 1.51670892e-02, 1.46367634e-02,
                                  3.69769707e-02, 7.03022778e-02],
                                   [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.47649962e-02,
                                  7.00923026e-01, 2.60504693e-01, 9.65307504e-02, 7.03864172e-02,
                                  1.15635686e-01, 1.53913230e-01],
                                   [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                  2.18164618e-03, 6.11085474e-01, 2.28024259e-01, 9.29291621e-02,
                                  1.14003479e-01, 1.54005408e-01],
                                   [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                  0.00000000e+00, 1.39757351e-03, 5.95472097e-01, 2.54652113e-01,
                                  1.32362068e-01, 1.71157718e-01],
                                   [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 1.10507896e-03, 5.05610526e-01,
                                  2.00507417e-01, 1.41500503e-01],
                                   [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 6.53312833e-04,
                                  2.97714621e-01, 1.26633704e-01],
                                   [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                  0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                  1.80651987e-04, 4.51902114e-02]]))
    
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
    
    assert np.allclose(fits_file[1].data.field("ENERG_LO"), 
                       np.array([150.,  220.,  325.,  480.,  520.,  765., 1120., 1650., 2350., 3450.]))
    
    assert np.allclose(fits_file[1].data.field("ENERG_HI"), 
                       np.array([220.,  325.,  480.,  520.,  765., 1120., 1650., 2350., 3450., 5000.]))   
    
    assert np.allclose(fits_file[1].data.field("SPECRESP"), 
                       np.array([0.06089862, 0.4563752 , 1.1601573 , 1.6237522 , 2.0216975 , 
                                 2.2039971 , 2.0773466 , 1.7005537 , 1.1626455 , 0.80194914]))   
    
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
    
    assert np.allclose(fits_file[1].data.field("ENERG_LO"), 
                       np.array([150.,  220.,  325.,  480.,  520.,  765., 1120., 1650., 2350., 3450.]))
    
    assert np.allclose(fits_file[1].data.field("ENERG_HI"), 
                       np.array([220.,  325.,  480.,  520.,  765., 1120., 1650., 2350., 3450., 5000.]))   
    
    assert np.allclose(fits_file[1].data.field("N_GRP"), 
                       np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))  
    
    matrix_flattened = []
    for i in fits_file[1].data.field("MATRIX"):
        matrix_flattened += i.tolist()
    
    assert np.allclose(matrix_flattened,
                        [0.9809964299201965,
                         0.019003521651029587,
                         0.046832531690597534,
                         0.9446346759796143,
                         0.008532775565981865,
                         0.01824718900024891,
                         0.12847033143043518,
                         0.848568856716156,
                         0.0047136349603533745,
                         0.009868173860013485,
                         0.09384072571992874,
                         0.21885812282562256,
                         0.662667989730835,
                         0.014764996245503426,
                         0.005820374935865402,
                         0.043238233774900436,
                         0.1858610063791275,
                         0.06197570636868477,
                         0.7009230256080627,
                         0.00218164618127048,
                         0.003475720528513193,
                         0.02238779515028,
                         0.07394952327013016,
                         0.027199288830161095,
                         0.26050469279289246,
                         0.6110854744911194,
                         0.0013975735055282712,
                         0.0028041559271514416,
                         0.01630435325205326,
                         0.04459220916032791,
                         0.01516708917915821,
                         0.09653075039386749,
                         0.22802425920963287,
                         0.5954720973968506,
                         0.001105078961700201,
                         0.0031390388030558825,
                         0.017328761518001556,
                         0.04066390544176102,
                         0.014636763371527195,
                         0.07038641721010208,
                         0.0929291620850563,
                         0.25465211272239685,
                         0.5056105256080627,
                         0.000653312832582742,
                         0.004899099003523588,
                         0.0280312392860651,
                         0.0696888118982315,
                         0.03697697073221207,
                         0.11563568562269211,
                         0.11400347948074341,
                         0.13236206769943237,
                         0.20050741732120514,
                         0.29771462082862854,
                         0.0001806519867386669,
                         0.006687051150947809,
                         0.03782569244503975,
                         0.0927841067314148,
                         0.07030227780342102,
                         0.1539132297039032,
                         0.15400540828704834,
                         0.17115771770477295,
                         0.14150050282478333,
                         0.12663370370864868,
                         0.04519021138548851])
    
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
    
