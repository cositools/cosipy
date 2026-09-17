import os
from cosipy import test_data
from cosipy.pipeline.task.task import cosi_bindata, cosi_threemlfit,cosi_tsdetect
from pathlib import Path
import pytest

data_path = Path(test_data.path)
config_path = os.path.join(data_path, "test_pipeline.yaml")


def test_run_task(tmp_path):
    # Usiamo tmp_path che pytest ci fornisce automaticamente
    tmpdir = tmp_path.as_posix()
    print(tmpdir)
    args = ['--config', config_path, '--overwrite', '--output-dir', tmpdir]

    # Esecuzione
    cosi_bindata(argv=args)
    print(tmpdir)
    assert Path(tmp_path/ "bin.yaml").exists()
    assert Path(tmp_path / "tsel_binned_data_local.hdf5").exists()
    #
    #cosi_threemlfit(argv=args)
    #assert Path(tmp_path/ "results.hdf").exists()
    #assert Path(tmp_path / "raw_spectrum.pdf").exists()
    #
    cosi_tsdetect(argv=args)
    assert Path(tmp_path / "raw_ts.png").exists()


#test_run_task(tmp_path)