from pathlib import Path

import logging
logger = logging.getLogger(__name__)

from mpi4py import MPI
from histpy import Histogram

from cosipy.response import FullDetectorResponse
from cosipy.image_deconvolution import ImageDeconvolution, DataIF_Parallel, DataIF_COSI_DC2

# Define MPI variables
MASTER = 0                      # Indicates master process
DRM_DIR = Path('/Users/penguin/Documents/Grad School/Research/COSI/COSIpy/docs/tutorials/data')
DATA_DIR = Path('/Users/penguin/Documents/Grad School/Research/COSI/COSIpy/docs/tutorials/image_deconvolution/511keV/GalacticCDS')

def main():
    '''
    Main script to create a parallel execution-compatible
    dataset using DataIF_Parallel and call ImageDeconvolution
    '''

    # Set up MPI
    comm = MPI.COMM_WORLD

    # Create dataset
    dataset = DataIF_Parallel(name = '511keV',
                                 event_filename = '511keV_dc2_galactic_event.hdf5',
                                 bkg_filename = '511keV_dc2_galactic_bkg.hdf5',
                                 drm_filename = 'psr_gal_511_DC2.h5',
                                 comm = comm)     # Convert dataset to a list of datasets before passing to RichardsonLucy class
    
    # bkg = Histogram.open(DATA_DIR / '511keV_dc2_galactic_bkg.hdf5')
    # event = Histogram.open(DATA_DIR / '511keV_dc2_galactic_event.hdf5')
    # image_response = Histogram.open(DRM_DIR / 'psr_gal_511_DC2.h5')
    # dataset = DataIF_COSI_DC2.load(name = "511keV",             # Create a dataset compatible with ImageDeconvolution: name (unique identifier), event data, background model, response, coordinate system conversion matrix (if detector response is not in galactic coordinates)
    #                                event_binned_data = event.project(['Em', 'Phi', 'PsiChi']),
    #                                dict_bkg_binned_data = {"total": bkg.project(['Em', 'Phi', 'PsiChi'])},
    #                                rsp = image_response)

    # Create image deconvolution object
    image_deconvolution = ImageDeconvolution()

    # set data_interface to image_deconvolution
    image_deconvolution.set_dataset([dataset])

    # set a parameter file for the image deconvolution
    parameter_filepath = DATA_DIR / 'imagedeconvolution_parfile_gal_511keV.yml'
    image_deconvolution.read_parameterfile(parameter_filepath)

    parallel_computation = True
    if comm.Get_rank() == MASTER:
        master_node = True
    else:
        master_node = False

    # Initialize model
    image_deconvolution.initialize(parallel_computation = parallel_computation,
                                   master_node = master_node)

    # Execute deconvolution
    image_deconvolution.run_deconvolution()

    # MPI Shutdown
    MPI.Finalize()

if __name__ == "__main__":
    main()