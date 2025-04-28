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

    #
    sou_data=config.get("sou_data")
    sou_yaml=config.get("sou_yaml")
    bk_data = config.get("bk_data")
    bk_yaml= config.get("bk_yaml")
    ori =config.get("ori")
    resp=config.get("resp")
    l=config.get("l")
    b=config.get("b")
    tstart=config.get("tstart")
    tstop=config.get("tstop")
    souname=config.get("souname")
    bkname=config.get("bkname")
    model_id=config.get("model_id")
    par_streams=config.get("par_streams")
    par_minvalues=config.get("par_minvalues")
    par_maxvalues=config.get('par_maxvalues')

    sou_data_path = config.config_path.parent/sou_data
    sou_yaml_path = config.config_path.parent/sou_yaml
    bk_data_path = config.config_path.parent/bk_data
    bk_yaml_path = config.config_path.parent/bk_yaml
    resp_path = config.config_path.parent/resp
    ori_path = config.config_path.parent/ori


    sou_binned_data = load_binned_data(sou_yaml_path, sou_data_path)
    bk_binned_data = load_binned_data(bk_yaml_path, bk_data_path)
    ori = load_ori(ori_path)
            #
    if tstart is not None and tstop is not None:
        sou_sliced_data=tslice_binned_data(sou_binned_data, tstart, tstop)
        sou_binned_data=sou_sliced_data
        bk_sliced_data = tslice_binned_data(bk_binned_data, tstart - 100, tstop + 100)
        bk_binned_data=bk_sliced_data
        ori_sliced = tslice_ori(ori, tstart, tstop)
        ori=ori_sliced
            #
    pars = [yaml.load(p) for p in par_streams]
    spectrum=build_spectrum(MODEL_IDS[model_id], pars, par_minvalues, par_maxvalues)
    results, cts_exp = get_fit_results(sou_binned_data, bk_binned_data, resp_path, ori, l, b,
                                                   souname, bkname, spectrum)
    results.display()
    odir_souname = odir/souname
    results.write_to(odir_souname.with_suffix(".fits"), overwrite=True)
                #
    pars_sou, epars_sou, par_bk, epar_bk = get_fit_par(results, souname, bkname)
    fl, el_fl, eh_fl = get_fit_fluxes(results)
    print("flux=%f +%f -%f" % (fl, el_fl, eh_fl))
                #
    figname = odir_souname.with_suffix(".pdf")
    plot_fit(sou_binned_data, cts_exp, figname)
#


if __name__ == "__main__":
    cosi_threemlfit()