from cosipy import DetectorResponse

from cosipy.response import cosi_rsp_dump 

def test_init():

    DetectorResponse()

def test_dump():

    cosi_rsp_dump(['/path/to/file'])
