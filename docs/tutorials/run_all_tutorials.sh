#/usr/bin/env bash

# Local mirror
cosi_pipeline_public="${HOME}/cosi/data/wasabi/cosi-pipelines-public"

#ln -s "${cosi_pipeline_public}/COSI-SMEX/DC2/Data/Orientation/20280301_3_month.ori" DataIO
#(cd DataIO && jupyter nbconvert --to html --execute --ExecutePreprocessor.kernel_name=python3 DataIO_example.ipynb)

#ln -s "${cosi_pipeline_public}/COSI-SMEX/DC2/Data/Orientation/20280301_3_month_with_orbital_info.ori" response
#ln -s "${cosi_pipeline_public}/COSI-SMEX/DC2/Responses/SMEXv12.Continuum.HEALPixO3_10bins_log_flat.binnedimaging.imagingresponse.nonsparse_nside8.area.good_chunks_unzip.h5.zip" response
#ln -s "${cosi_pipeline_public}/COSI-SMEX/DC2/Responses/SMEXv12.Continuum.HEALPixO3_10bins_log_flat.binnedimaging.imagingresponse.nonsparse_nside8.area.good_chunks_unzip.h5" response
#(cd response && jupyter nbconvert --to html --execute --ExecutePreprocessor.kernel_name=python3 SpacecraftFile.ipynb)

ln -fs "${cosi_pipeline_public}/COSI-SMEX/cosipy_tutorials/grb_spectral_fit_local_frame/grb_binned_data.hdf5" ts_map
ln -fs "${cosi_pipeline_public}/COSI-SMEX/cosipy_tutorials/ts_maps/bkg_binned_data_local.hdf5" ts_map
ln -fs "${cosi_pipeline_public}/COSI-SMEX/DC2/Data/Orientation/20280301_3_month_with_orbital_info.ori" ts_map
ln -fs "${cosi_pipeline_public}/COSI-SMEX/DC2/Responses/SMEXv12.Continuum.HEALPixO3_10bins_log_flat.binnedimaging.imagingresponse.nonsparse_nside8.area.good_chunks_unzip.h5.zip" ts_map
ln -fs "${cosi_pipeline_public}/COSI-SMEX/DC2/Responses/SMEXv12.Continuum.HEALPixO3_10bins_log_flat.binnedimaging.imagingresponse.nonsparse_nside8.area.good_chunks_unzip.h5" ts_map

ln -fs "${cosi_pipeline_public}/COSI-SMEX/DC2/Data/Sources/Crab_DC2_3months_unbinned_data.fits.gz" ts_map
ln -fs "${cosi_pipeline_public}/COSI-SMEX/DC2/Data/Backgrounds/albedo_photons_3months_unbinned_data.fits.gz" ts_map


(cd ts_map && jupyter nbconvert --to html --execute --ExecutePreprocessor.kernel_name=python3 Parallel_TS_map_computation.ipynb)