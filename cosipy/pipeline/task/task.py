import argparse, textwrap
import os
from yayc import Configurator
from threeML import Band
from cosipy import test_data
from cosipy.pipeline.src.io import *
from cosipy.pipeline.src.preprocessing import *
from cosipy.pipeline.src.fitting import *
from cosipy.pipeline.src.plotting import *
from cosipy.pipeline.src.fitting import MODEL_IDS
from astropy import units as u
from astropy.io.misc import yaml

from pathlib import Path

from astromodels.core.model_parser import ModelParser

def cosi_threemlfit(argv=None):
    # Parse arguments from commandline
    apar = argparse.ArgumentParser(
        usage=textwrap.dedent(
            """
            %(prog)s [--help] <command> [<args>] <filename> [<options>]
            """),
        description=textwrap.dedent(
            """
            Fits a source at l,b and optionally in a time window tstart-tstop using the given model.
            Data, response and orientation files paths in the config file should be relative to the config file.
            Outputs fit results in a fits file and a pdf plot of the fits.
            """),
        formatter_class=argparse.RawTextHelpFormatter)

    apar.add_argument('--config',
                      help="Path to .yaml file listing all the parameters.See example in test_data.",
                      required=True)
    apar.add_argument('-o','--output-dir',
                      help="Output directory. Current working directory by default")

    args = apar.parse_args(argv)

    # Config
    config = Configurator.open(args.config)

    # Default output
    odir = Path.cwd() if not args.output_dir else Path(args.output_dir)

    # Parse model
    model = ModelParser(model_dict = config['cosi_threemlfit:model']).get_model()

    #
    sou_data=config.get("sou_data")
    sou_yaml=config.get("sou_yaml")
    bk_data = config.get("bk_data")
    bk_yaml= config.get("bk_yaml")

    ori = config.get("ori")

    resp=config.get("resp")

    tstart=config.get("tstart")
    tstop=config.get("tstop")

    bkname=config.get("bkname")


    sou_data_path = config.config_path.parent/sou_data
    sou_yaml_path = config.config_path.parent/sou_yaml
    bk_data_path = config.config_path.parent/bk_data
    bk_yaml_path = config.config_path.parent/bk_yaml
    resp_path = config.config_path.parent/resp
    ori_path = config.config_path.parent/ori


    sou_binned_data = load_binned_data(sou_yaml_path, sou_data_path)
    bk_binned_data = load_binned_data(bk_yaml_path, bk_data_path)
    ori = load_ori(ori_path)

    if tstart is not None and tstop is not None:
        sou_sliced_data=tslice_binned_data(sou_binned_data, tstart, tstop)
        sou_binned_data=sou_sliced_data
        bk_sliced_data = tslice_binned_data(bk_binned_data, tstart - 100, tstop + 100)
        bk_binned_data=bk_sliced_data
        ori_sliced = tslice_ori(ori, tstart, tstop)
        ori=ori_sliced


    results, cts_exp = get_fit_results(sou_binned_data, bk_binned_data, resp_path, ori,
                                                   bkname, model)
    results.display()
    results.write_to(odir/"results.fits", overwrite=True)

    fitted_par_err = get_fit_par(results)
    for par_name,(par_median,par_err) in fitted_par_err.items():
        print(f"{par_name} = {par_median:.2e} +/- {par_err:.2e}")

    fl, el_fl, eh_fl = get_fit_fluxes(results)
    print("flux=%f +%f -%f" % (fl, el_fl, eh_fl))

    plot_fit(sou_binned_data, cts_exp, odir/"raw_spectrum.pdf")

if __name__ == "__main__":
    cosi_threemlfit()