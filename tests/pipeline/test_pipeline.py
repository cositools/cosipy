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
    args = ['--config', config_path, '--overwrite', '--output-dir', tmpdir]

    # Esecuzione
    cosi_bindata(argv=args)
    cosi_threemlfit(argv=args)
    cosi_tsdetect(argv=args)

    # Output files
    expected_files = [
        tmp_path / "bin.yaml",
        tmp_path / "binned_data_local.hdf5",
        tmp_path / "results.hdf",
        tmp_path / "raw_spectrum.pdf"
        tmp_path / "raw_ts.png"
    ]

    for file_path in expected_files:
        assert file_path.exists()

test_run_task(tmp_path)