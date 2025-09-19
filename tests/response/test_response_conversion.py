from cosipy import test_data

from cosipy.response import FullDetectorResponse, RspConverter

rspgz_response_path = test_data.path / "test_full_detector_response.rsp.gz"

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
