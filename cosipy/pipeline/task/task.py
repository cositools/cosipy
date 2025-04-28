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
            Data, response and orientation files should be in the same indir.
            Outputs fit results in a fits file and a pdf plot of the fits.
            """),
        formatter_class=argparse.RawTextHelpFormatter)

    apar.add_argument('--config',
                      help="Path to .yaml file listing all the parameters.See example in test_data.")
    apar.add_argument('--indir',
                      help="Optional. Path to a folder containing data, orientation and response files. Default is cosipy/test_data")
    apar.add_argument('--odir',
                      help="Optional. Path to a folder where to save outputs. Default is cosipy/test_data")
    apar.add_argument('--sou_data',
                      help="Required. Name of the file containing the source.")
    apar.add_argument('--sou_yaml',
                      help="Required. Name of the .yaml file containing the binning information of the source file")
    apar.add_argument('--bk_data',
                      help="Required. Name of the background file.")
    apar.add_argument('--bk_yaml',
                      help="Required. Name of the .yaml file containing the binning information of the bk file")
    apar.add_argument('--ori',
                      help="Required. Name of the orientation file.")
    apar.add_argument('--resp',
                      help="Required. Name of the response file.")
    apar.add_argument('--l',
                      help="Required. Galactic longitude of the source.")
    apar.add_argument('--b',
                      help="Required. Galactic latitude of the source.")
    apar.add_argument('--tstart',
                      help="Optional. Starting time of time window to be fitted.")
    apar.add_argument('--tstop',
                      help="Optional. Ending time of time window to be fitted.")
    apar.add_argument('--souname',
                      help="Required. Name of the source, to be used internally to threeml.")
    apar.add_argument('--bkname',
                      help="Required. Name of the background, to be used internally to threeml.")
    apar.add_argument('--model_id',
                      help="Required. ID of the model to be fitted. Dictionary of models in pipeline/fitting")
    apar.add_argument('--par_streams',
                      help="Required. List of start values of parameters as astropy quantity streams. Streams can be generated with the method yaml.dump. Order should be as in threeml.")
    apar.add_argument('--par_minvalues',
                      help="Optional. List of min values of the parameters.")
    apar.add_argument('--par_maxvalues',
                      help="Optional. List of max values of the parameters.")

    args = apar.parse_args(argv)

    # Config
    if args.config is None:
        config = Configurator()
    else:
        config = Configurator.open(args.config)
    #
    indir=config.get("indir")
    odir=config.get("odir")
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
    #
    if indir is None:
        print(indir)
        indir=str(test_data.path)
    if odir is None:
        print(odir)
        odir=str(test_data.path)
    #
    print(indir)
    print(odir)
    #
    sou_data_path=os.path.join(indir,sou_data)
    sou_yaml_path=os.path.join(indir,sou_yaml)
    bk_data_path=os.path.join(indir,bk_data)
    bk_yaml_path=os.path.join(indir,bk_yaml)
    resp_path=os.path.join(indir,resp)
    ori_path=os.path.join(indir,ori)
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


if __name__ == "__main__":
    cosi_threemlfit()