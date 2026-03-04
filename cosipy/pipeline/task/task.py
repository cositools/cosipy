import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

import os
import subprocess

import argparse, textwrap

from yayc import Configurator
from cosipy import UnBinnedData

from cosipy.pipeline.src.io import *
from cosipy.pipeline.src.preprocessing import *
from cosipy.pipeline.src.fitting import *
from cosipy.pipeline.src.plotting import *

from pathlib import Path

from astromodels.core.model_parser import ModelParser


def cosi_bindata(argv=None):
    # Parse arguments from commandline
    apar = argparse.ArgumentParser(
        usage=textwrap.dedent(
            """
            %(prog)s [--help] --config /path/to/config/file <command> [<options>]
            """),
        description=textwrap.dedent(
            """
            Bins an unbinned dataset matching the given response matrix
            and within the time interval of the given orientation file.
            Uses the given time bin size (dt) and coordinate system (either "local" or "galactic").
            Optionally, applies a time selection tmin-tmax to the data before binning.
            Data, response and orientation files paths in the config file should be relative to the config file.
            Outputs the input_yaml describing the binning and the binned dataset.
            """),
        formatter_class=argparse.RawTextHelpFormatter)

    apar.add_argument('--config',
                      help="Path to .yaml file listing all the parameters.See example in test_data.",
                      required=True)
    apar.add_argument("--config_group", default='bindata',
                      help="Path within the config file with the tutorials information")
    apar.add_argument("--override", nargs='*',
                      help="Override config parameters. e.g. \"section:param_int = 2\" \"section:param_string = b\"")
    apar.add_argument("--tmin", type = float,
                      help="Start time of the data selection (unix seconds)")
    apar.add_argument("--tmax", type=float,
                      help="Stop time of the data selection  (unix seconds)")
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
    if args.tmin:
        config["tmin"] = args.tmin

    if args.tmax:
        config["tmax"] = args.tmax

    # Default output
    odir = Path.cwd() if not args.output_dir else Path(args.output_dir)
    yaml_name="bin.yaml" if not args.suffix else str("bin_"+args.suffix+".yaml")
    # Coordinate system
    psichi_coo = config.get("coo_sys")
    #
    bdata_name=str("binned_data_"+psichi_coo) if not args.suffix else str("binned_data_"+psichi_coo+"_"+args.suffix)


    # Parse input files from config file
    data_path=config.absolute_path(config["unbinned_data_file"])
    resp_path = config.absolute_path(config["response:args"][0])
    ori_path=config.absolute_path(config["sc_file"])

    # Time info
    ori = SpacecraftFile.open(ori_path)
    ori_time=ori.get_time()
    tmin=config.get("tmin")
    tmax=config.get("tmax")
    if  config.get("tmin")==None:
        tmin = np.min(ori_time).value
    if config.get("tmin")==None:
        tmax = np.max(ori_time).value
    dt=  config.get("dt")


    #Prepare the input ymal
    yaml_path = odir/yaml_name
    if yaml_path.exists() and not args.overwrite:
        raise RuntimeError(f"{yaml_path} already exists. If you mean to replace it then use --overwrite.")
    write_yaml(str(data_path),str(ori_path),str(resp_path),dt,tmin,tmax,str(yaml_path))

    #Apply optional time selection:
    if config.get("tmin") is not None and config.get("tmax") is not None:
        #
        logger.info("Applying time selection %f-%f to the unbinned data" % (tmin,tmax))
        #
        tseldata_name="tsel_unbinned_data" if not args.suffix else str("tsel_unbinned_data_"+args.suffix)
        tseldata_path=odir/tseldata_name
        tseldata=UnBinnedData(yaml_path)
        tseldata.select_data_time(unbinned_data=data_path,output_name=str(tseldata_path))
        #
        #Unzip:
        subprocess.run(["gunzip", str(tseldata_path)+".fits.gz"])
        tseldata_name=tseldata_name+".fits"
        tseldata_path=odir/tseldata_name
        data_path=tseldata_path
        #
        tselbdata_name = str("tsel_binned_data_"+psichi_coo) if not args.suffix else str("tsel_binned_data_"+psichi_coo+"_"+ args.suffix)
        bdata_name=tselbdata_name


    #Make the binned dataset
    bdata_path=odir/bdata_name
    if bdata_path.exists() and not args.overwrite:
        raise RuntimeError(f"{bdata_path} already exists. If you mean to replace it then use --overwrite.")
    get_binned_data(yaml_path,data_path,bdata_path, psichi_coo)
    #
    logger.info(str(" Binning configuration file " + str(yaml_path) + " is ready"))
    logger.info(str(" Binned data file "+str(bdata_path)+" is ready for analysis"))


if __name__ == "__main__":
    cosi_bindata()









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
                      help="Path within the config file with the tutorials information")
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
    results, cts_exp = get_fit_results(binned_data, bk_binned_data, resp_path, ori, "cosi_bkg", model)


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
