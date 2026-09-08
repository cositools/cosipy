from pathlib import Path

from cosipy.response import RspConverter

wdir = Path("/Users/imartin5/cosi/scratch/response_relative_coordinates/v4")

converter = RspConverter()

converter.convert_to_h5(wdir/'ResponseContinuum.rel.binnedimaging.imagingresponse.nonsparse.rsp.gz',
                        wdir/'ResponseContinuum.rel.binnedimaging.imagingresponse.nonsparse.h5',
                        pa_convention = "RelativeX")

converter.convert_to_h5(wdir/'ResponseContinuum.rel.binnedimaging.imagingresponse.nonsparse.aeff.rsp.gz',
                        wdir/'ResponseContinuum.rel.binnedimaging.imagingresponse.nonsparse.aeff.h5',
                        pa_convention = "RelativeX")
