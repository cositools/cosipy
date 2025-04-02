#/usr/bin/env bash

kernel_name=python3

# Local mirror
cosi_pipeline_public="${HOME}/cosi/data/wasabi/cosi-pipelines-public"

#ln -fs "${cosi_pipeline_public}/COSI-SMEX/DC2/Data/Orientation/20280301_3_month.ori" DataIO
#(cd DataIO && jupyter nbconvert --to html --execute --ExecutePreprocessor.kernel_name=${kernel_name} DataIO_example.ipynb)

#ln -fs "${cosi_pipeline_public}/COSI-SMEX/DC2/Data/Orientation/20280301_3_month_with_orbital_info.ori" response
#ln -fs "${cosi_pipeline_public}/COSI-SMEX/DC2/Responses/SMEXv12.Continuum.HEALPixO3_10bins_log_flat.binnedimaging.imagingresponse.nonsparse_nside8.area.good_chunks_unzip.h5.zip" response
#ln -fs "${cosi_pipeline_public}/COSI-SMEX/DC2/Responses/SMEXv12.Continuum.HEALPixO3_10bins_log_flat.binnedimaging.imagingresponse.nonsparse_nside8.area.good_chunks_unzip.h5" response
#(cd response && jupyter nbconvert --to html --execute --ExecutePreprocessor.kernel_name=${kernel_name} SpacecraftFile.ipynb)

#ln -fs "${cosi_pipeline_public}/COSI-SMEX/DC2/Data/Orientation/20280301_3_month_with_orbital_info.ori" response
#ln -fs "${cosi_pipeline_public}/COSI-SMEX/DC2/Responses/SMEXv12.Continuum.HEALPixO3_10bins_log_flat.binnedimaging.imagingresponse.nonsparse_nside8.area.good_chunks_unzip.h5.zip" response
#ln -fs "${cosi_pipeline_public}/COSI-SMEX/DC2/Responses/SMEXv12.Continuum.HEALPixO3_10bins_log_flat.binnedimaging.imagingresponse.nonsparse_nside8.area.good_chunks_unzip.h5" response
#(cd response && jupyter nbconvert --to html --execute --ExecutePreprocessor.kernel_name=${kernel_name} DetectorResponse.ipynb)

#ln -fs "${cosi_pipeline_public}/COSI-SMEX/cosipy_tutorials/grb_spectral_fit_local_frame/grb_binned_data.hdf5" ts_map
#ln -fs "${cosi_pipeline_public}/COSI-SMEX/cosipy_tutorials/ts_maps/bkg_binned_data_local.hdf5" ts_map
#ln -fs "${cosi_pipeline_public}/COSI-SMEX/DC2/Data/Orientation/20280301_3_month_with_orbital_info.ori" ts_map
#ln -fs "${cosi_pipeline_public}/COSI-SMEX/DC2/Responses/SMEXv12.Continuum.HEALPixO3_10bins_log_flat.binnedimaging.imagingresponse.nonsparse_nside8.area.good_chunks_unzip.h5.zip" ts_map
#ln -fs "${cosi_pipeline_public}/COSI-SMEX/DC2/Responses/SMEXv12.Continuum.HEALPixO3_10bins_log_flat.binnedimaging.imagingresponse.nonsparse_nside8.area.good_chunks_unzip.h5" ts_map
#ln -fs "${cosi_pipeline_public}/COSI-SMEX/DC2/Data/Sources/Crab_DC2_3months_unbinned_data.fits.gz" ts_map
#ln -fs "${cosi_pipeline_public}/COSI-SMEX/DC2/Data/Backgrounds/albedo_photons_3months_unbinned_data.fits.gz" ts_map
#ln -fs "${cosi_pipeline_public}/COSI-SMEX/cosipy_tutorials/ts_maps/Crab_galactic_CDS_binned.hdf5" ts_map
#ln -fs "${cosi_pipeline_public}/COSI-SMEX/cosipy_tutorials/ts_maps/Albedo_galactic_CDS_binned.hdf5" ts_map
#ln -fs "${cosi_pipeline_public}/COSI-SMEX/DC2/Responses/PointSourceReponse/psr_gal_continuum_DC2.h5.zip" ts_map
#ln -fs "${cosi_pipeline_public}/COSI-SMEX/DC2/Responses/PointSourceReponse/psr_gal_DC2.h5" ts_map
#(cd ts_map && jupyter nbconvert --to html --execute --ExecutePreprocessor.kernel_name=${kernel_name} Parallel_TS_map_computation.ipynb)


#wdir=spectral_fits/continuum_fit/grb
#ln -fs "${cosi_pipeline_public}/COSI-SMEX/DC2/Data/Orientation/20280301_3_month_with_orbital_info.ori" $wdir
#ln -fs "${cosi_pipeline_public}/COSI-SMEX/cosipy_tutorials/grb_spectral_fit_local_frame/grb_bkg_binned_data.hdf5" $wdir
#ln -fs "${cosi_pipeline_public}/COSI-SMEX/cosipy_tutorials/grb_spectral_fit_local_frame/grb_binned_data.hdf5" $wdir
#ln -fs "${cosi_pipeline_public}/COSI-SMEX/cosipy_tutorials/grb_spectral_fit_local_frame/bkg_binned_data_1s_local.hdf5" $wdir
#ln -fs "${cosi_pipeline_public}/COSI-SMEX/DC2/Responses/SMEXv12.Continuum.HEALPixO3_10bins_log_flat.binnedimaging.imagingresponse.nonsparse_nside8.area.good_chunks_unzip.h5.zip" $wdir
#ln -fs "${cosi_pipeline_public}/COSI-SMEX/DC2/Responses/SMEXv12.Continuum.HEALPixO3_10bins_log_flat.binnedimaging.imagingresponse.nonsparse_nside8.area.good_chunks_unzip.h5" $wdir
#(cd $wdir && jupyter nbconvert --to html --execute --ExecutePreprocessor.kernel_name=${kernel_name} SpectralFit_GRB.ipynb)

wdir=spectral_fits/continuum_fit/crab
ln -fs "${cosi_pipeline_public}/COSI-SMEX/DC2/Data/Orientation/20280301_3_month_with_orbital_info.ori" $wdir
ln -fs "${cosi_pipeline_public}/COSI-SMEX/cosipy_tutorials/crab_spectral_fit_galactic_frame/crab_bkg_binned_data.hdf5" $wdir
ln -fs "${cosi_pipeline_public}/COSI-SMEX/cosipy_tutorials/crab_spectral_fit_galactic_frame/crab_binned_data.hdf5" $wdir
ln -fs "${cosi_pipeline_public}/COSI-SMEX/cosipy_tutorials/crab_spectral_fit_galactic_frame/bkg_binned_data.hdf5" $wdir
ln -fs "${cosi_pipeline_public}/COSI-SMEX/DC2/Responses/SMEXv12.Continuum.HEALPixO3_10bins_log_flat.binnedimaging.imagingresponse.nonsparse_nside8.area.good_chunks_unzip.h5.zip" $wdir
ln -fs "${cosi_pipeline_public}/COSI-SMEX/DC2/Responses/SMEXv12.Continuum.HEALPixO3_10bins_log_flat.binnedimaging.imagingresponse.nonsparse_nside8.area.good_chunks_unzip.h5" $wdir
(cd $wdir && jupyter nbconvert --to html --execute --ExecutePreprocessor.kernel_name=${kernel_name} SpectralFit_Crab.ipynb)
