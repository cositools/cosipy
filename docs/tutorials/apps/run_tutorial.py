from cosipy.pipeline.task.task import cosi_bindata,cosi_threemlfit
#
args=['--config','prep_bk_fit_grbdc3.yaml', '--overwrite', '--suffix','galbk']
cosi_bindata (argv=args)
#
args=['--config','prep_fit_grbdc3.yaml', '--overwrite', '--suffix','galbk_grbdc3']
cosi_bindata(argv=args)
#
args=['--config','prep_fit_grbdc3.yaml', '--overwrite', '--suffix','grbdc3_alltime']
cosi_threemlfit(argv=args)
#
args=['--config','prep_fit_grbdc3.yaml',
      '--overwrite', '--tstart', '1836496300.0',
      '--tstop', '1836496310.0','--suffix',
      'grbdc3_first_ten_seconds']
cosi_threemlfit (argv=args)
#
args=['--config','prep_fit_grbdc3.yaml',
      '--overwrite', '--override',
      'cuts:kwargs:tstart=1836496368.1', 'cuts:kwargs:tstop=1836496388.1',
      '--suffix','grbdc3_last_twenty_seconds']
cosi_threemlfit (argv=args)




