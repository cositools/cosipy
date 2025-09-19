import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

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
            %(prog)s [--help] --config /path/to/config/file <command> [<options>]
            """),
        description=textwrap.dedent(
            """
            Fits a source at l,b and optionally in a time window tstart-tstop using the given model.
            Data, response and orientation files paths in the config file should be relative to the config file.
            Outputs fit results in a fits file and a pdf plot of the fits. The fitted parameter
            are also printed to stdout.
            """),
        formatter_class=argparse.RawTextHelpFormatter)

    apar.add_argument('--config',
                      help="Path to .yaml file listing all the parameters.See example in test_data.",
                      required=True)
    apar.add_argument("--config_group", default='threemlfit',
                      help="Path withing the config file with the tutorials information")
    apar.add_argument("--override", nargs='*',
                      help="Override config parameters. e.g. \"section:param_int = 2\" \"section:param_string = b\"")
    apar.add_argument("--tstart", type = float,
                      help="Start time of the signal (unix seconds)")
    apar.add_argument("--tstop", type=float,
                      help="Stop time of the signal (unix seconds)")
    apar.add_argument('-o','--output-dir',
                      help="Output directory. Current working directory by default")
    apar.add_argument('--suffix',
                      help="Optional suffix to be added in the names of the output files")
    apar.add_argument('--log-level', default='info',
                      help='Set the logging level (debug, info, warning, error, critical)')
    apar.add_argument('--overwrite', action='store_true', default=False,
                      help='Overwrite outputs. Otherwise, if a file with the same name already exists, it will throw an error.')

    args = apar.parse_args(argv)

    # Logger
    logger.setLevel(level=args.log_level.upper())

    # config file
    full_config = Configurator.open(args.config)
    config = Configurator(full_config[args.config_group])
    config.config_path = full_config.config_path

    # General overrides
    if args.override is not None:
        config.override(*args.override)

    # Other specific convenience overrides
    if args.tstart:
        config["cuts:kwargs:tstart"] = args.tstart

    if args.tstop:
        config["cuts:kwargs:tstop"] = args.tstop

    # Default output
    odir = Path.cwd() if not args.output_dir else Path(args.output_dir)
    result_name="results.fits" if not args.suffix else str("results_"+args.suffix+".fits")
    plot_name="raw_spectrum.pdf" if not args.suffix else str("raw_spectrum_"+args.suffix+".pdf")

    # Parse model
    model = ModelParser(model_dict = config['model']).get_model()

    # Parse input files from config file
    data_path = config.absolute_path(config["data:args"][0])
    yaml_path = config.absolute_path(config["data:kwargs:input_yaml"])
    binned_data = load_binned_data(yaml_path, data_path)

    bk_data_path = config.absolute_path(config["background:args"][0])
    bk_yaml_path = config.absolute_path(config["background:kwargs:input_yaml"])
    bk_binned_data = load_binned_data(bk_yaml_path, bk_data_path)

    resp_path = config.absolute_path(config["response:args"][0])

    ori = load_ori(config.absolute_path(config["sc_file"]))

    # Slice time, if needed
    tstart = config.get("cuts:kwargs:tstart")
    tstop = config.get("cuts:kwargs:tstop")

    if tstart is not None and tstop is not None:

        tstart = Time(tstart, format='unix')
        tstop = Time(tstop, format='unix')

        sliced_data=tslice_binned_data(binned_data, tstart, tstop)
        binned_data=sliced_data
        bk_sliced_data = tslice_binned_data(bk_binned_data, tstart - 100, tstop + 100)
        bk_binned_data=bk_sliced_data
        ori_sliced = ori.source_interval(tstart, tstop)
        ori=ori_sliced

    # Calculation
    results, cts_exp = get_fit_results(binned_data, bk_binned_data, resp_path, ori,
                                                   "cosi_bkg", model)

    # Results
    results.display()
    results.write_to(odir/result_name, overwrite=args.overwrite)

    print("Median and errors:")
    fitted_par_err = get_fit_par(results)
    for par_name,(par_median,par_err) in fitted_par_err.items():
        print(f"{par_name} = {par_median:.2e} +/- {par_err:.2e}")

    print("Total flux:")
    fl, el_fl, eh_fl = get_fit_fluxes(results)
    print("flux=%f +%f -%f" % (fl, el_fl, eh_fl))

    plot_filename = odir/plot_name
    if plot_filename.exists() and not args.overwrite:
        raise RuntimeError(f"{plot_filename} already exists. If you mean to replace it then use --overwrite.")

    plot_fit(binned_data, cts_exp, plot_filename)

if __name__ == "__main__":
    cosi_threemlfit()