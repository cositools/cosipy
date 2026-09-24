import os
from cosipy import test_data
from cosipy.pipeline.task.task import cosi_bindata, cosi_threemlfit,cosi_tsdetect
from cosipy.pipeline.src.fitting import get_fit_fluxes
from pathlib import Path
from threeML import load_analysis_results_hdf
import pytest

data_path = Path(test_data.path)
config_path = os.path.join(data_path, "test_pipeline.yaml")

l_true=177.42
b_true=-9.41
fl_true=0.0785

def test_run_task(tmp_path,capsys):
    #
    # Usiamo tmp_path che pytest ci fornisce automaticamente
    tmpdir = tmp_path.as_posix()
    #
    #Execute cosi-bindata
    #
    args = ['--config', config_path, '--overwrite', '--output-dir', tmpdir]
    cosi_bindata(argv=args)
    #
    #Check if outputs (binned dataset & bin info yaml) exist
    #
    assert Path(tmp_path/ "bin.yaml").exists()
    assert Path(tmp_path / "tsel_binned_data_local.hdf5").exists()
    #
    #Execute cosi-tsdetect and capture the output
    #
    capsys.readouterr() #Erase the ouput
    cosi_tsdetect(argv=args)
    captured = capsys.readouterr()
    log_file = tmp_path / "cosi_tsdetect.txt"
    log_file.write_text(captured.out)
    #
    #Check if output (ts map pmd & fits map) exist
    #
    assert Path(log_file).exists()
    assert Path(tmp_path / "raw_ts.png").exists()
    assert Path(tmp_path / "raw_ts.fits").exists()
    #
    #Check the coordinate at maximum TS
    #
    ##########Read the output
    with open(log_file, 'r') as f:
        lines = f.readlines()
    line_coordinate = lines[1]
    l_measured = float(line_coordinate.split('l=')[1].split(',')[0])
    b_measured = float(line_coordinate.split('b=')[1].strip())
    line_pixel = lines[2]
    pixel_size = float(line_pixel.split(': ')[1].strip())
    #
    #######Check that measured coordinates are within l_true +/- pixel_size
    assert (l_true - pixel_size) <= l_measured <= (l_true + pixel_size)
    assert (b_true - pixel_size) <= b_measured <= (b_true + pixel_size)
    #
    #Execute cosi-tsdetect and capture the output
    #
    cosi_threemlfit(argv=args)
    #
    #Check if outputs (results h5 file and png plot) exist
    #
    assert Path(tmp_path/ "results.h5").exists()
    assert Path(tmp_path / "raw_spectrum.png").exists()
    #
    #Check the measured flux
    #
    ##########I used the saved results.h5 to check the measured flux
    ###########is within 3*errors the measured flux
    fit_results=load_analysis_results_hdf(Path(tmp_path/ "results.h5"))
    fl,em_fl,ep_fl=get_fit_fluxes(fit_results)
    assert (fl_true - 3*em_fl) <= fl <= (fl_true + 3*ep_fl)

