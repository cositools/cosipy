#/usr/bin/env bash

# Local mirror
cosi_pipeline_public="${HOME}/cosi/data/wasabi/cosi-pipelines-public"

#ln -s "${cosi_pipeline_public}/COSI-SMEX/DC2/Data/Orientation/20280301_3_month.ori" DataIO
#(cd DataIO && jupyter nbconvert --to html --execute --ExecutePreprocessor.kernel_name=python3 DataIO_example.ipynb)

#ln -s "${cosi_pipeline_public}/COSI-SMEX/DC2/Data/Orientation/20280301_3_month_with_orbital_info.ori" response
#ln -s "${cosi_pipeline_public}/COSI-SMEX/DC2/Responses/SMEXv12.Continuum.HEALPixO3_10bins_log_flat.binnedimaging.imagingresponse.nonsparse_nside8.area.good_chunks_unzip.h5.zip" response
#ln -s "${cosi_pipeline_public}/COSI-SMEX/DC2/Responses/SMEXv12.Continuum.HEALPixO3_10bins_log_flat.binnedimaging.imagingresponse.nonsparse_nside8.area.good_chunks_unzip.h5" response
#(cd response && jupyter nbconvert --to html --execute --ExecutePreprocessor.kernel_name=python3 SpacecraftFile.ipynb)

ln -s "${cosi_pipeline_public}/COSI-SMEX/DC2/Data/Orientation/20280301_3_month_with_orbital_info.ori" ts_map
ln -s "${cosi_pipeline_public}/COSI-SMEX/DC2/Responses/SMEXv12.Continuum.HEALPixO3_10bins_log_flat.binnedimaging.imagingresponse.nonsparse_nside8.area.good_chunks_unzip.h5.zip" ts_map
ln -s "${cosi_pipeline_public}/COSI-SMEX/DC2/Responses/SMEXv12.Continuum.HEALPixO3_10bins_log_flat.binnedimaging.imagingresponse.nonsparse_nside8.area.good_chunks_unzip.h5" ts_map
(cd ts_map && jupyter nbconvert --to html --execute --ExecutePreprocessor.kernel_name=python3 Parallel_TS_map_computation.ipynb)