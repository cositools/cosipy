#!/usr/bin/env python
# coding: UTF-8

import sys
import logging
logger = logging.getLogger('cosipy')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

from cosipy.spacecraftfile import SpacecraftFile
from cosipy.response import FullDetectorResponse, ExtendedSourceResponse

# file path
full_detector_response_path = "SMEXv12.Continuum.HEALPixO3_10bins_log_flat.binnedimaging.imagingresponse.nonsparse_nside8.area.good_chunks_unzip.h5"
orientation_path = "20280301_3_month_with_orbital_info.ori"

# load response and orientation
full_detector_response = FullDetectorResponse.open(full_detector_response_path)
orientation = SpacecraftFile.parse_from_file(orientation_path)

# generate your extended source response
extended_source_response = full_detector_response.get_extended_source_response(orientation,
                                                                               coordsys='galactic',
                                                                               nside_scatt_map=None,
                                                                               Earth_occ=True)

# save the extended source response
extended_source_response.write("extended_source_response_continuum.h5", overwrite = True)
