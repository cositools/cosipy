import textwrap
from pathlib import Path

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)

from yayc import Configurator

from astropy import units as u

import argparse

try:
    from mpi4py import MPI
    mpi4py_imported = True
except ModuleNotFoundError as e:
    mpi4py_imported = False
    mpi4py_imported_error = e

from histpy import Histogram

from cosipy.response import FullDetectorResponse
from cosipy.image_deconvolution import ImageDeconvolution, DataIF_Parallel, DataIF_COSI_DC2, ParallelImageDeconvolution

# Define MPI variables
MASTER = 0                      # Indicates master process
DRM_DIR = Path('/Users/penguin/Documents/Grad School/Research/COSI/COSIpy/docs/tutorials/data')
DATA_DIR = Path('/Users/penguin/Documents/Grad School/Research/COSI/COSIpy/docs/tutorials/image_deconvolution/511keV/GalacticCDS')

def main():
    args = argparse.ArgumentParser(
        usage=textwrap.dedent(
            """
            %(prog)s [--help] --config <file> --data <file> --bkg <file> --response <file> [<options>]
            """),
        description=textwrap.dedent(
            """
            Main script to create a parallel execution-compatible
            dataset using DataIF_Parallel and call ParallelImageDeconvolution
            
            NOTE: Currently limited to DC2 data and PSR only.
            """),
        formatter_class=argparse.RawTextHelpFormatter)

    args.add_argument("--force-mpi", action="store_true", default = False,
                      help="This will force MPI parallelization. Used to testing and debuggin."
                           "Running 'mpiexec -n 1 %(prog)s' is otherwise indistinguishable from "
                           "not using mpiexec or mpi4py not being installed")
    args.add_argument("--plot", action="store_true", default=False,
                      help="Plot the resulting image")
    args.add_argument("--config",
                      help="Path to config file")
    args.add_argument("--bkg",
                      help="Path to binned bkg file")
    args.add_argument("--data",
                      help="Path to binned data file")
    args.add_argument("--response",
                      help="Path to PSR file")
    args.add_argument('--log-level', default='info',
                      help='Set the logging level (debug, info, warning, error, critical)')

    args = args.parse_args()

    # Logger
    logger.setLevel(level=args.log_level.upper())

    # Get inputs
    data_file = Path(args.data)
    bkg_file = Path(args.bkg)
    response_file = Path(args.response)
    config_file = Path(args.config)

    # Check if we running in parallel using MPI or not
    if mpi4py_imported:

        comm = MPI.COMM_WORLD
        mpi_parallel = args.force_mpi or (comm.Get_size() > 1)

    else:

        if args.force_mpi:
            raise RuntimeError(f"Can't run with MPI. Import error: {mpi4py_imported_error}")

        mpi_parallel = False

    # Create dataset

    if mpi_parallel:

        logger.info(f"Running in parallel using MPI with {comm.Get_size()} sub-processes")

        dataset = DataIF_Parallel(name = '511keV',
                                  event_filename = data_file,
                                  bkg_filename = bkg_file,
                                  bkg_norm_label = 'albedo',
                                  drm_filename = response_file,
                                  comm = comm)     # Convert dataset to a list of datasets before passing to RichardsonLucy class

        # Create image deconvolution object
        image_deconvolution = ParallelImageDeconvolution(comm)

    else:

        logger.info(f"Running with no MPI parallelization")

        bkg = Histogram.open(bkg_file)
        event = Histogram.open(data_file)
        image_response = Histogram.open(response_file)
        dataset = DataIF_COSI_DC2.load(name = "511keV",             # Create a dataset compatible with ImageDeconvolution: name (unique identifier), event data, background model, response, coordinate system conversion matrix (if detector response is not in galactic coordinates)
                                       event_binned_data = event.project(['Em', 'Phi', 'PsiChi']),
                                       dict_bkg_binned_data = {"albedo": bkg.project(['Em', 'Phi', 'PsiChi'])},
                                       rsp = image_response)

        # Create image deconvolution object
        image_deconvolution = ImageDeconvolution()

    # set data_interface to image_deconvolution
    image_deconvolution.set_dataset([dataset])

    # set a parameter file for the image deconvolution
    image_deconvolution.read_parameterfile(config_file)

    # Initialize model
    image_deconvolution.initialize()

    # Execute deconvolution
    image_deconvolution.run_deconvolution()

    # Results
    if not mpi_parallel or comm.Get_rank() == 0:
        print(image_deconvolution.results)

        if args.plot:

            from mhealpy import HealpixMap
            import matplotlib.pyplot as plt

            def plot_reconstructed_image(result, source_position=None):  # source_position should be (l,b) in degrees
                iteration = result['iteration']
                image = result['model']

                for energy_index in range(image.axes['Ei'].nbins):
                    map_healpxmap = HealpixMap(data=image[:, energy_index], unit=image.unit)

                    _, ax = map_healpxmap.plot('mollview')

                    _.colorbar.set_label(str(image.unit))

                    if source_position is not None:
                        ax.scatter(source_position[0] * u.deg, source_position[1] * u.deg, transform=ax.get_transform('world'),
                                   color='red')

                    plt.title(
                        label=f"iteration = {iteration}, energy_index = {energy_index} ({image.axes['Ei'].bounds[energy_index][0]}-{image.axes['Ei'].bounds[energy_index][1]})")

            plot_reconstructed_image(image_deconvolution.results[-1])

            plt.show()

    # MPI Shutdown
    if mpi_parallel:
        MPI.Finalize()

if __name__ == "__main__":
    main()