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
    # Parse arguments from commandline
    apar = argparse.ArgumentParser(
        usage=textwrap.dedent(
            """
            %(prog)s [--help] <command> [<args>] <filename> [<options>]
            """),
        description=textwrap.dedent(
            """
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
    #
    sou_data_path=config.get("sou_data_path")
    sou_yaml_path=config.get("sou_yaml_path")
    bk_data_path = config.get("bk_data_path")
    bk_yaml_path = config.get("bk_yaml_path")
    ori_path =config.get("ori_path")
    resp_path=config.get("resp_path")
    odir=config.get("odir")
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
    #
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
    results.write_to(str(odir + "/" + souname + ".fits"), overwrite=True)
                #
    pars_sou, epars_sou, par_bk, epar_bk = get_fit_par(results, souname, bkname)
    fl, el_fl, eh_fl = get_fit_fluxes(results)
    print("flux=%f +%f -%f" % (fl, el_fl, eh_fl))
                #
    figname = str(odir + "/fit_"+souname+".pdf")
    plot_fit(sou_binned_data, cts_exp, figname)
#


