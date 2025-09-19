#!/usr/bin/env python
# coding: UTF-8

import sys
import logging
logger = logging.getLogger('cosipy')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

from cosipy.spacecraftfile import SpacecraftFile
from cosipy.response import FullDetectorResponse, ExtendedSourceResponse

# load full detector response 
full_detector_response_path = "SMEXv12.Continuum.HEALPixO3_10bins_log_flat.binnedimaging.imagingresponse.nonsparse_nside8.area.good_chunks_unzip.h5"
full_detector_response = FullDetectorResponse.open(full_detector_response_path)

# basename should be the same as one used before
basename = "psr/psr_"

# merge the point source responses
extended_source_response = full_detector_response.merge_psr_to_extended_source_response(basename, coordsys = 'galactic', nside_image = None)

# save the extended source response
extended_source_response.write("extended_source_response_continuum_merged.h5", overwrite = True)
