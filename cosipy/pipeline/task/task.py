import argparse, textwrap
from yayc import Configurator
from threeML import Band
from cosipy.pipeline.src.io import *
from cosipy.pipeline.src.preprocessing import *
from cosipy.pipeline.src.fitting import *
from cosipy.pipeline.src.plotting import *

from cosipy.pipeline.src.fitting import MODEL_IDS

from astropy import units as u
from astropy.io.misc import yaml


def cosi_threemlfit(argv=None):
                #(sou_data_path,sou_yaml_path,bk_data_path,bk_yaml_path,ori_path,resp_path,odir,l,b,souname,bkname,spectrum):
    # Parse arguments from commandline
    apar = argparse.ArgumentParser(
        usage=textwrap.dedent(
            """
            %(prog)s [--help] <command> [<args>] <filename> [<options>]
            """),
        description=textwrap.dedent(
            """
            Quick view of the information contained in a response file

            %(prog)s --help
            %(prog)s dump header [FILENAME]
            %(prog)s dump aeff [FILENAME] --lon [LON] --lat [LAT]
            %(prog)s dump expectation [FILENAME] --config [CONFIG]
            %(prog)s plot aeff [FILENAME] --lon [LON] --lat [LAT]
            %(prog)s plot dispersion [FILENAME] --lon [LON] --lat [LAT]
            %(prog)s plot expectation [FILENAME] --lon [LON] --lat [LAT]

            Arguments:
            - header: Response header and axes information
            - aeff: Effective area
            - dispersion: Energy dispection matrix
            - expectation: Expected number of counts
            """),
        formatter_class=argparse.RawTextHelpFormatter)

    apar.add_argument('--config',
                      help="Latitude in sopacecraft coordinates. e.g. '10deg'")

    args = apar.parse_args(argv)

    # Config
    if args.config is None:
        config = Configurator()
    else:
        config = Configurator.open(args.config)
    print(config["sou_yaml_path"])
    #
    sou_data_path=config["sou_data_path"]
    sou_yaml_path=config["sou_yaml_path"]
    bk_data_path = config["bk_data_path"]
    bk_yaml_path = config["bk_yaml_path"]
    ori_path =config["ori_path"]
    resp_path=config["resp_path"]
    odir=config["odir"]
    l=config["l"]
    b=config["b"]
    tstart=config["tstart"]
    tstop=config["tstop"]
    souname=config["souname"]
    bkname=config["bkname"]
    model_id=config["model_id"]
    par_streams=config["par_streams"]
    par_minvalues=config["par_minvalues"]
    par_maxvalues=config.get('par_maxvalues')
    #par_maxvalues=config["par_maxvalues"]
    #
    grb_binned_data = load_binned_data(sou_yaml_path, sou_data_path)
    bk_binned_data = load_binned_data(bk_yaml_path, bk_data_path)
    ori = load_ori(ori_path)
                #
    bk_sliced_data = tslice_binned_data(bk_binned_data, tstart - 100, tstop + 100)
    ori_sliced = tslice_ori(ori, tstart, tstop)
                #
    pars = [yaml.load(p) for p in par_streams]
    spectrum=build_spectrum(MODEL_IDS[model_id], pars, par_minvalues, par_maxvalues)
    results, cts_exp = get_fit_results(grb_binned_data, bk_sliced_data, resp_path, ori_sliced, l, b,
                                                   souname, bkname, spectrum)
    results.display()
    results.write_to(str(odir + "/" + souname + ".fits"), overwrite=True)
                #
    pars_sou, epars_sou, par_bk, epar_bk = get_fit_par(results, souname, bkname)
    fl, el_fl, eh_fl = get_fit_fluxes(results)
    print("flux=%f +%f -%f" % (fl, el_fl, eh_fl))
                #
    figname = str(odir + "/fit_"+souname+".pdf")
    plot_fit(grb_binned_data, cts_exp, figname)
#


