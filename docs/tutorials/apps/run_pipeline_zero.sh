export MPLBACKEND=Agg
##
#python get_pipeline_zero_data.py
##
export known_tstart=1836496300.0
export known_tstop=1836496388.0
##
##Bin full background
##
cosi-bindata --config pipeline_zero.yaml --config_group "bindata_bk" --overwrite --suffix "galbkfull"
##
##Select in time and bin the data
##
cosi-bindata --config pipeline_zero.yaml --config_group "bindata_soubk" --tmin $known_tstart --tmax $known_tstop \
--suffix "galbk_grbdc3" --overwrite

#Estimate the position of the transient using the TS map
#
cosi-tsdetect --config pipeline_zero.yaml --overwrite > cosi-tsdetect.txt
##
##Fit the spectrum at the estimated position.
## Refine the position by having l and b free in the fit within 3*pixel_size.
##
export measured_l=$(awk -F'[=,]' '/Galactic coordinate/ {print $2}' cosi-tsdetect.txt)
echo $measured_l
export measured_b=$(awk -F'[=,]' '/Galactic coordinate/ {print $4}' cosi-tsdetect.txt)
echo $measured_b
export error_coo=$(awk -F': ' '/Linear Size/ {print $2}' cosi-tsdetect.txt)
echo $error_coo
#
export l_max=$(echo "$measured_l + 3 * $error_coo" | bc -l)
export l_min=$(echo "$measured_l - 3 * $error_coo" | bc -l)
#
export b_max=$(echo "$measured_b + 3 * $error_coo" | bc -l)
export b_min=$(echo "$measured_b - 3 * $error_coo" | bc -l)
#
cosi-threemlfit --config pipeline_zero.yaml --config_group "threemlfit_pw" \
--override "model:template (point source):position:l:value=$measured_l" \
"model:template (point source):position:b:value=$measured_b" \
"model:template (point source):position:l:free=true" \
"model:template (point source):position:l:min_value=$l_min" \
"model:template (point source):position:l:max_value=$l_max" \
"model:template (point source):position:b:free=true" \
"model:template (point source):position:b:min_value=$b_min" \
"model:template (point source):position:b:max_value=$b_max" \
--overwrite --suffix "pw"
##
##
##
#cosi-threemlfit --config pipeline_zero.yaml --config_group "threemlfit_band" \
#--override "model:template_grb (point source):position:l:value=$measured_l" \
#"model:template_grb (point source):position:b:value=$measured_b" \
#"model:template_grb (point source):position:l:free=true" \
#"model:template_grb (point source):position:l:min_value=$l_min" \
#"model:template_grb (point source):position:l:max_value=$l_max" \
#"model:template_grb (point source):position:b:free=true" \
#"model:template_grb (point source):position:b:min_value=$b_min" \
#"model:template_grb (point source):position:b:max_value=$b_max" \
#--overwrite --suffix "band"
#
