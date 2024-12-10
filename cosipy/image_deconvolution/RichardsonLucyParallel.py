import os
import subprocess
import logging
logger = logging.getLogger(__name__)

# Import third party libraries
import numpy as np
import h5py
# from histpy import Histogram

from .deconvolution_algorithm_base import DeconvolutionAlgorithmBase
from .RichardsonLucy import RichardsonLucy

class RichardsonLucyParallel(RichardsonLucy):
    """
    NOTE: Comments copied from RichardsonLucy.py
    A class for a parallel implementation of the Richardson-
    Lucy algorithm. 
    
    An example of parameter is as follows.

    iteration_max: 100
    minimum_flux:
        value: 0.0
        unit: "cm-2 s-1 sr-1"
    background_normalization_optimization: True 
    """

    def __init__(self, initial_model, dataset, mask, parameter):
        RichardsonLucy.__init__(self, initial_model, dataset, mask, parameter)

        # Specific to parallel implementation
        self.numproc = parameter.get('numproc', 1)
        self.iteration_max = parameter.get('iteration_max', 10)
        self.base_dir = os.getcwd()
        self.data_dir = parameter.get('data_directory', './data')     # NOTE: Data should ideally be present in disk scratch space.
        self.config_file = 'imagedeconvolution_parfile_gal_511keV.yml' # XXX: CWD / config_path

        for dataset in self.dataset:                    # NOTE: There may be multiple datasets to analyze, each with its own response file. 
            image_response = dataset._image_response    # TODO: Ideate on how they should be handled. We currently ignore everything beyond the first dataset.
            self.numrows = np.product(image_response.contents.shape[-3:])   # Em, Phi, PsiChi. NOTE: Change the "-3" if more general model space definitions are expected
            self.numcols = np.product(image_response.contents.shape[:-3])   # Remaining columns
            break       # NOTE: Ignoring everything beyond the first dataset

    def initialization(self):
        """
        initialization before running the image deconvolution
        """

        # Run parent class initialization
        RichardsonLucy.initialization(self)             # Useful outputs: model, summed_exposure_map (required to calculate delta_model), response_weighting_filter, mask (required to calculate alpha) will be saved
                                                        # Discarded outputs: iteration_count, expectation_list will be not be saved
                                                        # self.results will be saved to disk at every iteration. No variable will be returned.

        # Flatten and write dense bkg and events to scratch space. 
        self.write_intermediate_files()

    def write_intermediate_files(self):
        # Event
        event = self.dataset[0].event.contents.flatten()
        np.savetxt(self.base_dir + '/event.csv', event)

        # Background
        bg = np.zeros(len(event))
        bg_models = self.dataset[0]._bkg_models
        for key in bg_models:
            bg += bg_models[key].contents.flatten()
        np.savetxt(self.base_dir + '/bg.csv', bg)

        # Response matrix
        image_response = self.dataset[0]._image_response
        new_shape = (self.numrows, self.numcols)
        ndim = image_response.contents.ndim
        with h5py.File(self.base_dir + '/response_matrix.h5', 'w') as output_file:
            dset1 = output_file.create_dataset('response_matrix', data=np.transpose(image_response.contents, 
                                                                                    np.take(np.arange(ndim), 
                                                                                            range(ndim-3, 2*ndim-3), 
                                                                                            mode='wrap')
                                                                                    ).reshape(new_shape))  # NOTE: Change the "ndim-3" if more general model space definitions are expected
            logger.info(f'Shape of response matrix {dset1.shape}')
            dset2 = output_file.create_dataset('response_vector', data=np.sum(dset1, axis=0))
            logger.info(f'Shape of response vector summed along axis=0 {dset2.shape}')

    def iteration(self):
        """
        Performs all iterations of image deconvolution.
        NOTE: Overrides implementation in deconvolution_algorithm_base.py and invokes an external script
        """
        
        # All arguments must be passed as type=str. Explicitly type cast boolean and number to string.
        FILE_DIR = os.path.dirname(os.path.abspath(__file__))               # Path to directory containing RichardsonLucyParallel.py
        logger.info(f"Subprocess call to run RLparallelscript.py at '{FILE_DIR}'")

        # RLparallelscript.py will be installed in the same directory as RichardsonLucyParallel.py
        stdout = subprocess.run(args=["mpiexec", "-n", str(self.numproc), 
                             "python", FILE_DIR + "/RLparallelscript.py", 
                             "--numrows", str(self.numrows),
                             "--numcols", str(self.numcols),
                             "--base_dir", str(self.base_dir),
                             "--config_file", str(self.config_file)
                             ], env=os.environ)
        logger.info(stdout)

        # RLparallelscript already contains check_stopping_criteria and iteration_max break condition. 
        # NOTE: RichardsonLucy.py currently does not support a sophisticated break condition. 
        return True

    def finalization(self):
        """
        finalization after running the image deconvolution
        """
        # NOTE: RLparallel algorithm saves every iteration's result by default
        # No need to call finalization function implemented in RichardsonLucy

        # Delete intermediate files from scratch space
        self.remove_intermediate_files_from_disk()

    def remove_intermediate_files_from_disk(self):
        # Ensure that the number of deletions corresponds to the 
        # number of file creations in write_... function
        os.remove(self.base_dir + '/event.csv')
        os.remove(self.base_dir + '/bg.csv')
        os.remove(self.base_dir + '/response_matrix.h5')

    # The following will be performed in RLparallelscript.py
    def pre_processing(self):
        pass
    def Estep(self):
        pass
    def Mstep(self):
        pass
    def post_processing(self):
        pass
    def check_stopping_criteria(self):
        pass
    def register_result(self):
        pass