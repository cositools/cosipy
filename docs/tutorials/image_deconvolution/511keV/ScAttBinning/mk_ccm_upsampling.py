from cosipy.response import FullDetectorResponse
from cosipy.image_deconvolution import SpacecraftAttitudeExposureTable, CoordsysConversionMatrix

filepath = "/Users/yoneda/Work/Exp/COSI/cosipy-2/data_challenge/DC2/prework/data/"

full_detector_response_filename = filepath + "response/SMEXv12.511keV.HEALPixO4.binnedimaging.imagingresponse.nonsparse_nside16.area.h5"
full_detector_response = FullDetectorResponse.open(full_detector_response_filename)

exposure_table = SpacecraftAttitudeExposureTable.from_fits("exposure_table_nside32.fits")

coordsys_conv_matrix = CoordsysConversionMatrix.spacecraft_attitude_binning_ccm(full_detector_response, exposure_table, nside_model = 32, use_averaged_pointing = True)
coordsys_conv_matrix.write("ccm_nside32.hdf5", overwrite = True)
