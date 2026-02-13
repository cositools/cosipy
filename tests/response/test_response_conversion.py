import numpy as np

from cosipy import test_data

from cosipy.response import FullDetectorResponse, RspConverter

from pytest import raises

rspgz_response_path = test_data.path / "test_full_detector_response.rsp.gz"

rspgz_nonorm_response_path = test_data.path / "test_full_detector_response_no_norm.rsp.gz"

rspgz_mono_nonorm_response_path = test_data.path / "test_full_detector_response_mono_no_norm.rsp.gz"

h5_response_path = test_data.path / "test_full_detector_response.h5"


def test_convert_rsp_to_h5(tmp_path):

    import gzip

    tmp_h5_filename = tmp_path / "fdr.h5"

    c = RspConverter(bufsize = 100000)

    # test opening compressed .rsp and writing compressed .h5
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

    # test opening compressed .rsp and writing uncompressed .h5
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

    # test opening uncompressed .rsp
    with gzip.open(rspgz_response_path) as gzfile:
        content = gzfile.read()
    with open(tmp_path / "response.rsp", "wb") as rspfile:
        rspfile.write(content)

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

def test_default_norms(tmp_path):

    tmp_h5_filename = tmp_path / "fdr.h5"

    c = RspConverter(bufsize = 100000,
                     norm = "Linear",
                     norm_params = [50, 10000])

    c.convert_to_h5(rspgz_nonorm_response_path,
                    h5_filename = tmp_h5_filename,
                    overwrite = True)

    with FullDetectorResponse.open(tmp_h5_filename) as fdr:
        norm_info = fdr.headers["SP"]
        assert norm_info == "Linear 50 10000"

    c = RspConverter(bufsize = 100000,
                     norm = "Mono",
                     norm_params = [511])

    c.convert_to_h5(rspgz_nonorm_response_path,
                    h5_filename = tmp_h5_filename,
                    overwrite = True)

    with FullDetectorResponse.open(tmp_h5_filename) as fdr:
        norm_info = fdr.headers["SP"]
        assert norm_info == "Mono 511"

        # bins with no defined spectral normalization should
        # have their eff_area set to zero, not -inf or NaN
        assert all(np.isfinite(fdr.eff_area_correction))

    # mono norm with no energy should work for single-bin response
    c = RspConverter(bufsize = 100000,
                     norm = "Mono",
                     norm_params = [])

    c.convert_to_h5(rspgz_mono_nonorm_response_path,
                    h5_filename = tmp_h5_filename,
                    overwrite = True)

    with FullDetectorResponse.open(tmp_h5_filename) as fdr:
        norm_info = fdr.headers["SP"]
        assert norm_info == "Mono"

        # bins with no defined spectral normalization should
        # have their eff_area set to zero, not -inf or NaN
        assert all(np.isfinite(fdr.eff_area_correction))

    # if no monoenergetic energy is given, Mono is
    # undefined with multiple Ei bins
    with raises(ValueError):
        c = RspConverter(bufsize = 100000,
                         norm = "Mono")

        c.convert_to_h5(rspgz_nonorm_response_path,
                        h5_filename = tmp_h5_filename,
                        overwrite = True)

    c = RspConverter(bufsize = 100000,
                     norm = "powerlaw",
                     norm_params = [50, 10000, 0.9])

    c.convert_to_h5(rspgz_nonorm_response_path,
                    h5_filename = tmp_h5_filename,
                    overwrite = True)

    with FullDetectorResponse.open(tmp_h5_filename) as fdr:
        norm_info = fdr.headers["SP"]
        assert norm_info == "powerlaw 50 10000 0.9"

    c = RspConverter(bufsize = 100000,
                     norm = "powerlaw",
                     norm_params = [50, 10000, 1])

    c.convert_to_h5(rspgz_nonorm_response_path,
                    h5_filename = tmp_h5_filename,
                    overwrite = True)

    with FullDetectorResponse.open(tmp_h5_filename) as fdr:
        norm_info = fdr.headers["SP"]
        assert norm_info == "powerlaw 50 10000 1.0"

    # powerlaw norm that does not span all of Ei
    c = RspConverter(bufsize = 100000,
                     norm = "powerlaw",
                     norm_params = [500, 1000, 1])

    c.convert_to_h5(rspgz_nonorm_response_path,
                    h5_filename = tmp_h5_filename,
                    overwrite = True)

    with FullDetectorResponse.open(tmp_h5_filename) as fdr:
        norm_info = fdr.headers["SP"]
        assert norm_info == "powerlaw 500 1000 1.0"

        # bins with no defined spectral normalization should
        # have their eff_area set to zero, not -inf or NaN
        assert all(np.isfinite(fdr.eff_area_correction))

    # Gaussian norm
    c = RspConverter(bufsize = 100000,
                     norm = "Gaussian",
                     norm_params = [100, 100, 3])

    c.convert_to_h5(rspgz_nonorm_response_path,
                    h5_filename = tmp_h5_filename,
                    overwrite = True)

    with FullDetectorResponse.open(tmp_h5_filename) as fdr:
        norm_info = fdr.headers["SP"]
        assert norm_info == "Gaussian 100.0 100.0 3.0"

        # bins with no defined spectral normalization should
        # have their eff_area set to zero, not -inf or NaN
        assert all(np.isfinite(fdr.eff_area_correction))


def test_convert_h5_to_rsp(tmp_path):

    tmp_rsp_filename = tmp_path / "fdr.rsp"
    tmp_rspgz_filename = tmp_path / "fdr.rsp.gz"
    tmp_h5_filename = tmp_path / "fdr.h5"

    c = RspConverter(bufsize = 100000)

    # test writing compressed .rsp.gz
    fdr = FullDetectorResponse.open(h5_response_path)

    c.convert_to_rsp(fdr, tmp_rspgz_filename, overwrite=True)

    tmp_h5_filename = c.convert_to_h5(tmp_rspgz_filename, overwrite=True)

    fdr2 = FullDetectorResponse.open(tmp_h5_filename)


    h1 = fdr.to_dr()
    h2 = fdr2.to_dr()

    fdr.close()
    fdr2.close()

    assert h1 == h2

    # test writing uncompressed .rsp
    fdr = FullDetectorResponse.open(h5_response_path)

    c.convert_to_rsp(fdr, tmp_rsp_filename, overwrite=True)

    tmp_h5_filename = c.convert_to_h5(tmp_rsp_filename, overwrite=True)

    fdr2 = FullDetectorResponse.open(tmp_h5_filename)

    h1 = fdr.to_dr()
    h2 = fdr2.to_dr()

    fdr.close()
    fdr2.close()

    assert h1 == h2
