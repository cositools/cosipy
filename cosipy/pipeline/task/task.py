import argparse, textwrap

from yayc import Configurator

from cosipy.pipeline.src.io import *
from cosipy.pipeline.src.preprocessing import *
from cosipy.pipeline.src.fitting import *
from cosipy.pipeline.src.plotting import *

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
    apar.add_argument("--config_group", default='threemlfit',
                      help="Path withing the config file with the tutorials information")
    apar.add_argument("--override", nargs='*',
                      help="Override config parameters. e.g. \"section:param_int = 2\" \"section:param_string = b\"")
    apar.add_argument('-o','--output-dir',
                      help="Output directory. Current working directory by default")

    args = apar.parse_args(argv)

    # config file
    full_config = Configurator.open(args.config)
    config = Configurator(full_config[args.config_group])
    config.config_path = full_config.config_path
    if args.override is not None:
        config.override(*args.override)

    # Default output
    odir = Path.cwd() if not args.output_dir else Path(args.output_dir)

    # Parse model
    model = ModelParser(model_dict = config['model']).get_model()

    # Parse input files from config file
    sou_data_path = config.config_path.parent/config["data:args"][0]
    sou_yaml_path = config.config_path.parent / config["data:kwargs:input_yaml"]
    sou_binned_data = load_binned_data(sou_yaml_path, sou_data_path)

    bk_data_path = config.config_path.parent/config["background:args"][0]
    bk_yaml_path = config.config_path.parent/config["background:kwargs:input_yaml"]
    bk_binned_data = load_binned_data(bk_yaml_path, bk_data_path)

    resp_path = config.config_path.parent/config["response:args"][0]

    ori = load_ori(config.config_path.parent/config["sc_file"])

    # Slice time, if needed
    tstart = config.get("cuts:kwargs:tstart")
    tstop = config.get("cuts:kwargs:tstop")

    if tstart is not None and tstop is not None:
        sou_sliced_data=tslice_binned_data(sou_binned_data, tstart, tstop)
        sou_binned_data=sou_sliced_data
        bk_sliced_data = tslice_binned_data(bk_binned_data, tstart - 100, tstop + 100)
        bk_binned_data=bk_sliced_data
        ori_sliced = tslice_ori(ori, tstart, tstop)
        ori=ori_sliced

    # Calculation
    results, cts_exp = get_fit_results(sou_binned_data, bk_binned_data, resp_path, ori,
                                                   "cosi_bkg", model)

    # Results
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