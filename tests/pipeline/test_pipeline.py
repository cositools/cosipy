import os
os.system("cosi-threemlfit --config test_spec_grb.yaml")
assert os.path.exists("../../cosipy/test_data/grb_test.fits")
assert os.path.exists("../../cosipy/test_data/fit_grb_test.pdf")