import os
from cosipy import test_data
#
data_path=str(test_data.path)
config_path=os.path.join(data_path,"test_spec_grb.yaml")
def test_run_task():
    os.system(str("cosi-threemlfit --config "+ config_path))
