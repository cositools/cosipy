import os
from cosipy import test_data
#
data_path = str(test_data.path)
os.system("cosi-threemlfit --config test_spec_grb.yaml")
assert os.path.exists(str(data_path+"/grb_test.fits"))
assert os.path.exists(str(data_path+"/fit_grb_test.pdf"))