import os
from cosipy import test_data

data_path=str(test_data.path)
config_path=os.path.join(data_path,"test_pipeline.yaml")

def test_run_task(tmp_path):
    tmpdir = tmp_path.as_posix()
    os.system(str("cosi-bindata --config "+ config_path + " -o " + tmpdir + " --overwrite"))
    os.system(str("cosi-threemlfit --config "+ config_path + " -o " + tmpdir + " --overwrite"))
