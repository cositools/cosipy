import os
from cosipy import test_data
from cosipy.pipeline.task.task import cosi_bindata,cosi_threemlfit

data_path=str(test_data.path)
config_path=os.path.join(data_path,"test_pipeline.yaml")

def test_run_task(tmp_path):
    tmpdir = tmp_path.as_posix()
    args = ['--config', config_path, '--overwrite', '--output-dir', tmpdir]
    cosi_bindata(argv=args)
    cosi_threemlfit(argv=args)
    #
    #os.system(str("cosi-bindata --config "+ config_path + " -o " + tmpdir + " --overwrite"))
    #os.system(str("cosi-threemlfit --config "+ config_path + " -o " + tmpdir + " --overwrite"))

test_run_task(test_data.path)