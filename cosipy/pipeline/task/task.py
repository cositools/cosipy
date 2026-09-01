import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

import subprocess
from astropy.time import Time
import numpy as np
import argparse, textwrap
from yayc import Configurator
from cosipy import UnBinnedData
from cosipy.pipeline.src.fitting import (
    get_fit_fluxes,
    get_fit_par,
    get_fit_results,
    get_ts_results,
)
from cosipy.pipeline.src.io import load_binned_data, load_ori, tslice_binned_data
from cosipy.pipeline.src.plotting import plot_fit
from cosipy.pipeline.src.preprocessing import get_binned_data, write_yaml

from pathlib import Path
from histpy import Histogram
from cosipy import FastTSMap,MOCTSMap

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

    #Set histogram sparse or dense.
    sparse = config.get('sparse')

    # Time info
    ori = SpacecraftHistory.open(ori_path)
    ori_time=ori.obstime
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
    write_yaml(str(data_path),str(ori_path),str(resp_path), dt,tmin,tmax,str(yaml_path))
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
        #Unzip and force overwrite
        subprocess.run(["gunzip", "-f", f"{tseldata_path}.fits.gz"])
        #
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
    get_binned_data(yaml_path,data_path,bdata_path, psichi_coo, sparse=sparse)
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
            Outputs fit results in a hdf file and a pdf plot of the fits. The fitted parameter
            are also printed to stdout. If the fit fails, the error message is printed to stdout, 
            an empty hdf is saved and no plot is produced
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
    plot_name="raw_spectrum.png" if not args.suffix else str("raw_spectrum_"+args.suffix+".png")

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
        tfrac = (tstop - tstart).to_value('s')*0.5

        sliced_data=tslice_binned_data(binned_data, tstart, tstop)
        binned_data=sliced_data
        bk_sliced_data = tslice_binned_data(bk_binned_data, tstart - tfrac, tstop + tfrac)
        bk_binned_data=bk_sliced_data

    tmin_sou = Time(binned_data.axes['Time'].edges.min(), format='unix')
    tmax_sou = Time(binned_data.axes['Time'].edges.max(), format='unix')
    tmin_bk = Time(bk_binned_data.axes['Time'].edges.min(), format='unix')
    tmax_bk = Time(bk_binned_data.axes['Time'].edges.max(), format='unix')
    ori_sliced_sou = ori.select_interval(tmin_sou, tmax_sou)
    ori_sliced_bk = ori.select_interval(tmin_bk, tmax_bk)


    # Calculation
    results, cts_exp = get_fit_results(binned_data, bk_binned_data, resp_path, ori_sliced_sou, ori_sliced_bk, model)

    # Results
    if results is not None: # FRANCESCO
        result_name = "results.h5" if not args.suffix else str("results_" + args.suffix + ".h5")
        results.display()
        results.write_to(odir/result_name, overwrite=args.overwrite, as_hdf=True)

        print("Median and errors:")
        fitted_par_err = get_fit_par(results)
        for par_name,(par_median,par_err) in fitted_par_err.items():
            print(f"{par_name} = {par_median:.2e} +/- {par_err:.2e}")

        print("Total flux:")
        fl, el_fl, eh_fl = get_fit_fluxes(results)
        print("flux=%f +%f -%f" % (fl, el_fl, eh_fl))
    else:
        x = np.array([1.0])
        y = np.array([0.0])
        yerr = np.array([1.0])
        plugin = XYLike("single_point", x, y, yerr)
        plugins = DataList(plugin)
        model = Model(
            PointSource(
            "src",
            0, 0,
            spectral_shape=Constant()
            )
        )
        model.src.spectrum.main.value = 0.0
        like2 = JointLikelihood(model, plugins, verbose=False)
        like2.fit()
        results2=like2.results
        result_name = "results_crash.h5" if not args.suffix else str("results_crash_" + args.suffix + ".h5")
        results2.write_to(odir/result_name,overwrite=args.overwrite,as_hdf=True)

#PLOT:
    plot_filename = odir/plot_name
    if plot_filename.exists() and not args.overwrite:
        raise RuntimeError(f"{plot_filename} already exists. If you mean to replace it then use --overwrite.")
    if cts_exp is not None:
        plot_fit(binned_data, cts_exp, plot_filename)

if __name__ == "__main__":
    cosi_threemlfit()


def  cosi_tsdetect(argv=None):
    # Parse arguments from commandline
    apar = argparse.ArgumentParser(
        usage=textwrap.dedent(
            """
            %(prog)s [--help] --config /path/to/config/file <command> [<options>]
            """),
        description=textwrap.dedent(
            """
            Performs Test Statistic (TS) map fitting, with optional time-windowing (tstart-tstop).
            Data, background, response, and orientation file paths in the configuration file 
            must be relative to the configuration file's location.
            Features:
            - Multiresolution: If enabled in the config, uses Multi-Order Coverage (MOC) maps.
            - Coordinate Systems: Supports Galactic or Local systems as specified by the user.
            - Resolution Control: The 'nside' parameter in the config sets either the fixed resolution or the maximum depth for MOC fits.   
            Outputs:
            - Printed to stdout: Maximum TS value, its Galactic coordinates, and the pixel linear size of the TS/MOC map.
            - Files: A PNG plot of the TS map is saved to the output directory. 
            """),
        formatter_class=argparse.RawTextHelpFormatter)

    apar.add_argument('--config',
                      help="Path to .yaml file listing all the parameters.See example in test_data.",
                      required=True)
    apar.add_argument("--config_group", default='tsdetect',
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
    plot_name="raw_ts.png" if not args.suffix else str("raw_ts_"+args.suffix+".png")
    map_name="raw_ts.fits" if not args.suffix else str("raw_ts_"+args.suffix+".fits")

    #Setup of the tsmap search
    coo_sys=config.get('coo_sys')
    nside_search=config.get('nside')
    multiresolution=config.get('multiresolution')
    energy_channels=config.get('energy_channels')
    cpu_cores=config.get('cpu_cores')
    moc_init_nside=config.get('moc_init_nside')
    moc_containment_strategy=config.get('moc_containment_strategy')

    # Parse template spectrum
    model = ModelParser(model_dict = config['model']).get_model()
    spectrum=model.template.spectrum.main.Powerlaw

    # Parse input files from config file
    data_path = config.absolute_path(config["data:args"][0])
    bk_data_path = config.absolute_path(config["background:args"][0])
    resp_path = config.absolute_path(config["response:args"][0])
    ori_full = load_ori(config.absolute_path(config["sc_file"]))

    #Open the data histogram
    data_full = Histogram.open(data_path)
    
    # Slice the data in time if needed:

    tstart = config.get("cuts:kwargs:tstart")
    tstop = config.get("cuts:kwargs:tstop")

    if tstart is not None and tstop is not None:

        tstart = Time(tstart, format='unix')
        tstop = Time(tstop, format='unix')

        sliced_data = tslice_binned_data(data_full, tstart, tstop)
        binned_data = sliced_data.project(['Em', 'Phi', 'PsiChi'])

    else:
        tstart=Time(np.min(data_full.axes['Time'].edges), format='unix')
        tstop=Time(np.max (data_full.axes['Time'].edges), format='unix')
        binned_data = data_full.project(['Em', 'Phi', 'PsiChi'])

    # Slice the ori file in the time interval of the data:
    ori_sliced = ori_full.select_interval(Time(tstart,format="unix"), Time(tstop, format="unix"))
    ori=ori_sliced

    # Prepare the background model.
    # TBD: use the estimated background here:

    delta = tstop - tstart
    delta = delta.to_value('s')
    #
    bkg_full=Histogram.open(bk_data_path)
    bk_tstart=np.min(bkg_full.axes['Time'].edges.value)
    bk_tstop=np.max(bkg_full.axes['Time'].edges.value)
    ori_bkg = ori_full.select_interval(Time(bk_tstart,format="unix"), Time(bk_tstop, format="unix"))
    bkg_full_livetime = ori_bkg.cumulative_livetime().to_value("s")
    #
    bkg_model = bkg_full.project(['Em', 'Phi', 'PsiChi'])
    bkg_model /= (bkg_full_livetime / delta)
    del bkg_full

    #Check for previous running
    plot_filename = odir / plot_name
    if plot_filename.exists() and not args.overwrite:
        raise RuntimeError(f"{plot_filename} already exists. If you mean to replace it then use --overwrite.")

    map_filename = odir / map_name
    if map_filename.exists() and not args.overwrite:
        raise RuntimeError(f"{map_filename} already exists. If you mean to replace it then use --overwrite.")
    # Calculation

    if multiresolution==False:

        ts = FastTSMap(data=binned_data, bkg_model=bkg_model, orientation=ori,
                   response_path=resp_path, cds_frame=coo_sys)

        ts_results = ts.fit(nside=nside_search, energy_channel=energy_channels,
                        spectrum=spectrum, cpu_cores=cpu_cores)
        max_ts,max_coo,max_l,max_b,pixel_mean_spacing=get_ts_results(map_filename, ts_results,multiresolution,nside=nside_search,overwrite_map=args.overwrite)
        print("Maximum TS= %f" % max_ts)
        print("Galactic coordinate at maximum TS: l=%f, b=%f" %(max_l, max_b))
        print("Linear Size of TS map pixel: %f" % (pixel_mean_spacing))

        # Results and plot

        ts.plot_ts(ts_results, skycoord=max_coo, save_dir=odir, save_plot=True, save_name=plot_name, mark_center=False)

    elif multiresolution==True:

        moc_ts = MOCTSMap(data = binned_data, bkg_model = bkg_model, response_path = resp_path, orientation = ori, cds_frame = coo_sys)

        strategy = MOCTSMap.PaddingStrategy(
            MOCTSMap.ContainmentStrategy(moc_containment_strategy)
        )

        moc_results, moc_uniq = moc_ts.fit(
            max_nside=nside_search,
            init_nside=moc_init_nside,
            energy_channel=energy_channels,
            spectrum=spectrum,
            cpu_cores=cpu_cores,
            strategy=strategy,
        )

        max_ts, max_coo, max_l, max_b, pixel_mean_spacing = get_ts_results(map_filename,moc_results,moc_uniq, multiresolution=multiresolution, overwrite_map=args.overwrite)
        print("Maximum TS= %f" % max_ts)
        print("Galactic coordinate at maximum TS: l=%f, b=%f" % (max_l, max_b))
        print("Linear Size of MOC map max_nside_pixel: %f" % (pixel_mean_spacing))

        # Results and plot
        moc_ts.plot_ts(moc_results, moc_uniq, skycoord=max_coo,save_dir=odir, save_plot=True, save_name=plot_name)

if __name__ == "__main__":
        cosi_tsdetect()
