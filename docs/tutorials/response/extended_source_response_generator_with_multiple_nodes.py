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

# set the healpix pixel index list
ipix_image_list = [int(_) for _ in sys.argv[1:]]

print(ipix_image_list)

# generate a point source response at each pixel
basename = "psr/psr_"

for ipix_image in ipix_image_list:

    psr = full_detector_response.get_point_source_response_per_image_pixel(ipix_image, orientation, 
                                                                           coordsys='galactic',
                                                                           nside_image=None,
                                                                           nside_scatt_map=None,
                                                                           Earth_occ=True)
    
    psr.write(f"{basename}{ipix_image:08}.h5",overwrite = True)

# see also merge_response_generated_with_mutiple_nodes.py to know how we can merge the above point source responses as a single extended source response.
