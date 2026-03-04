#
cosi-bindata --config prep_bk_fit_grbdc3.yaml --overwrite --suffix "galbk"
cosi-bindata --config prep_fit_grbdc3.yaml --overwrite --suffix "galbk_grbdc3"
#
cosi-threemlfit --config prep_fit_grbdc3.yaml --overwrite   --suffix "grbdc3_alltime"
#
cosi-threemlfit --config prep_fit_grbdc3.yaml --overwrite --tstart 1836496300.0 --tstop 1836496310.0   --suffix "grbdc3_first_ten_seconds"
cosi-threemlfit --config prep_fit_grbdc3.yaml --overwrite --override cuts:kwargs:tstart=1836496368.1 cuts:kwargs:tstop=1836496388.1 --suffix "grbdc3_last_twenty_seconds"
