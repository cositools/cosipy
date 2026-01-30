import numpy as np

from cosipy import test_data

from cosipy.response import FullDetectorResponse, RspConverter

rspgz_response_path = test_data.path / "test_full_detector_response.rsp.gz"

rspgz_nonorm_response_path = test_data.path / "test_full_detector_response_no_norm.rsp.gz"

h5_response_path = test_data.path / "test_full_detector_response.h5"


def test_convert_rsp_to_h5(tmp_path):

    tmp_h5_filename = tmp_path / "fdr.h5"

    c = RspConverter(bufsize = 100000)

    c.convert_to_h5(rspgz_response_path,
                    h5_filename = tmp_h5_filename,
                    overwrite = True)


    fdr = FullDetectorResponse.open(h5_response_path)
    fdr2 = FullDetectorResponse.open(tmp_h5_filename)

    h1 = fdr.to_dr()
    h2 = fdr2.to_dr()

    fdr.close()
    fdr2.close()

    assert h1 == h2


    c.convert_to_h5(rspgz_response_path,
                    h5_filename = tmp_h5_filename,
                    overwrite = True,
                    compress=False)

    fdr = FullDetectorResponse.open(h5_response_path)
    fdr2 = FullDetectorResponse.open(tmp_h5_filename)

    h1 = fdr.to_dr()
    h2 = fdr2.to_dr()

    fdr.close()
    fdr2.close()

    assert h1 == h2

def test_default_norms(tmp_path):

    tmp_h5_filename = tmp_path / "fdr.h5"

    c = RspConverter(bufsize = 100000,
                     norm = "Linear",
                     norm_params = [50, 10000])

    c.convert_to_h5(rspgz_nonorm_response_path,
                    h5_filename = tmp_h5_filename,
                    overwrite = True)

    fdr = FullDetectorResponse.open(tmp_h5_filename)
    norm_info = fdr.headers["SP"]
    assert norm_info == "Linear 50 10000"
    fdr.close()

    c = RspConverter(bufsize = 100000,
                     norm = "Mono")

    c.convert_to_h5(rspgz_nonorm_response_path,
                    h5_filename = tmp_h5_filename,
                    overwrite = True)

    fdr = FullDetectorResponse.open(tmp_h5_filename)
    norm_info = fdr.headers["SP"]
    assert norm_info == "Mono"
    fdr.close()

    c = RspConverter(bufsize = 100000,
                     norm = "powerlaw",
                     norm_params = [50, 10000, 0.9])

    c.convert_to_h5(rspgz_nonorm_response_path,
                    h5_filename = tmp_h5_filename,
                    overwrite = True)

    fdr = FullDetectorResponse.open(tmp_h5_filename)
    norm_info = fdr.headers["SP"]
    assert norm_info == "powerlaw 50 10000 0.9"
    fdr.close()

    c = RspConverter(bufsize = 100000,
                     norm = "powerlaw",
                     norm_params = [50, 10000, 1])

    c.convert_to_h5(rspgz_nonorm_response_path,
                    h5_filename = tmp_h5_filename,
                    overwrite = True)

    fdr = FullDetectorResponse.open(tmp_h5_filename)
    norm_info = fdr.headers["SP"]
    assert norm_info == "powerlaw 50 10000 1.0"
    fdr.close()

    c = RspConverter(bufsize = 100000,
                     norm = "powerlaw",
                     norm_params = [500, 1000, 1])

    c.convert_to_h5(rspgz_nonorm_response_path,
                    h5_filename = tmp_h5_filename,
                    overwrite = True)

    fdr = FullDetectorResponse.open(tmp_h5_filename)
    norm_info = fdr.headers["SP"]
    assert norm_info == "powerlaw 500 1000 1.0"

    # bins with no defined spectral normalization should
    # have their eff_area set to zero, not -inf or NaN
    assert all(np.isfinite(fdr.eff_area_correction))
    fdr.close()

    # cannot test Gaussian norm because it can only be used on
    # a response with a single Ei bin

def test_convert_h5_to_rsp(tmp_path):

    tmp_rsp_filename = tmp_path / "fdr.rsp.gz"
    tmp_h5_filename = tmp_path / "fdr.h5"

    fdr = FullDetectorResponse.open(h5_response_path)

    c = RspConverter(bufsize = 100000)

    c.convert_to_rsp(fdr, tmp_rsp_filename, overwrite=True)

    tmp_h5_filename = c.convert_to_h5(tmp_rsp_filename, overwrite=True)

    fdr2 = FullDetectorResponse.open(tmp_h5_filename)

    h1 = fdr.to_dr()
    h2 = fdr2.to_dr()

    fdr.close()
    fdr2.close()

    assert h1 == h2
