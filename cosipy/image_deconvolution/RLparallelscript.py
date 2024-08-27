import os
from pathlib import Path
import logging
import argparse

# logging setup
logger = logging.getLogger(__name__)

# argparse setup
parser = argparse.ArgumentParser()
parser.add_argument("--numrows", type=int, dest='numrows', help="Number of rows in the response matrix")
parser.add_argument("--numcols", type=int, dest='numcols', help="Number of columns in the response matrix")
parser.add_argument("--base_dir", type=str, dest='base_dir', help="Current working directory and where configuration file is assumed to lie")
parser.add_argument("--config_file", type=str, dest='config_file', help="Name of configuration file (assumed to lie in CWD)")
args = parser.parse_args()

# Import third party libraries
import numpy as np
from mpi4py import MPI
import h5py
from yayc import Configurator

# Load configuration file
config = Configurator.open(f'{args.base_dir}/{args.config_file}')

# Number of elements in data space (ROWS) and model space (COLS)
NUMROWS = args.numrows        # TODO: Ideally, for row-major form to exploit caching, NUMROWS must be smaller than NUMCOLS
NUMCOLS = args.numcols

# Define MPI and iteration misc variables
MASTER = 0                      # Indicates master process
MAXITER = config.get('deconvolution:parameter:iteration_max', 10)

# FILE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = Path(args.base_dir)
RESULTS_DIR = BASE_DIR / config.get('deconvolution:parameter:save_results_directory', './results')

'''
Response matrix
'''
def load_response_matrix(comm, start_row, end_row, filename='response_matrix.h5'):
    with h5py.File(BASE_DIR / filename, "r", driver="mpio", comm=comm) as f1:
        dataset = f1["response_matrix"]
        R = dataset[start_row:end_row, :]
    return R

'''
Response matrix transpose
'''
def load_response_matrix_transpose(comm, start_col, end_col, filename='response_matrix.h5'):
    with h5py.File(BASE_DIR / filename, "r", driver="mpio", comm=comm) as f1:
        dataset = f1["response_matrix"]
        RT = dataset[:, start_col:end_col]
    return RT

'''
Response matrix summed along axis=i
'''
def load_axis0_summed_response_matrix(filename='response_matrix.h5'):
    with h5py.File(BASE_DIR / filename, "r") as f1:
        dataset = f1["response_vector"]
        Rj = dataset[:]
    return Rj

'''
Sky model
'''
def initial_sky_model(model_init_val=[1e-4]):
    M0 = np.ones(NUMCOLS, dtype=np.float64) * float(model_init_val[0])     # Initial guess according to image_deconvolution.py. TODO: Make this more general than element 0
    return M0

'''
Background model
'''
def load_bg_model(filename='bg.csv'):
    bg = np.loadtxt(filename)
    return bg

'''
Observed data
'''
def load_event_data(filename='event.csv'):
    event = np.loadtxt(filename)
    return event

def register_result(iter, M, delta):
    """
    The values below are stored at the end of each iteration.
    - iteration: iteration number
    - model: updated image
    - delta_model: delta map after M-step 
    - processed_delta_model: delta map after post-processing
    - alpha: acceleration parameter in RL algirithm
    - background_normalization: optimized background normalization
    - loglikelihood: log-likelihood
    """
    
    this_result = {"iteration": iter, 
                    "model": M, 
                    "delta_model": delta,
                    # "processed_delta_model": copy.deepcopy(self.processed_delta_model), TODO: The RL parallel implementation does not currently support smooth convergence through weighting, background normalization, or likelihood calculation
                    # "background_normalization": copy.deepcopy(self.dict_bkg_norm),
                    # "alpha": self.alpha, 
                    # "loglikelihood": copy.deepcopy(self.loglikelihood_list)
                    }

    # # show intermediate results
    # logger.info(f'  alpha: {this_result["alpha"]}')
    # logger.info(f'  background_normalization: {this_result["background_normalization"]}')
    # logger.info(f'  loglikelihood: {this_result["loglikelihood"]}')
    
    return this_result

def save_results(results):
    '''
    NOTE: Copied from RichardsonLucy.py
    '''
    logger.info('Saving results in {RESULTS_DIR}')
    # model
    for this_result in results:
        iteration_count = this_result["iteration"]
        # this_result["model"].write(f"{RESULTS_DIR}/model_itr{iteration_count}.hdf5", overwrite = True)    # TODO: numpy arrays do not support write_to_hdf5 as a method. Need to ensure rest of code is modified to support cosipy.image_deconvolution.allskyimage.AllSkyImageModel
        # this_result["delta_model"].write(f"{RESULTS_DIR}/delta_model_itr{iteration_count}.hdf5", overwrite = True)
        # this_result["processed_delta_model"].write(f"{RESULTS_DIR}/processed_delta_model_itr{iteration_count}.hdf5", overwrite = True)    TODO: processed_delta_model here is not different from delta_model
        np.savetxt(f'{RESULTS_DIR}/model_itr{iteration_count}.csv', this_result['model'], delimiter=',')
        np.savetxt(f'{RESULTS_DIR}/delta_model_itr{iteration_count}.csv', this_result['delta_model'], delimiter=',')

    # TODO: The following will be enabled once the respective calculations are incorporated
    # #fits
    # primary_hdu = fits.PrimaryHDU()

    # col_iteration = fits.Column(name='iteration', array=[float(result['iteration']) for result in self.results], format='K')
    # col_alpha = fits.Column(name='alpha', array=[float(result['alpha']) for result in self.results], format='D')
    # cols_bkg_norm = [fits.Column(name=key, array=[float(result['background_normalization'][key]) for result in self.results], format='D') 
    #                 for key in self.dict_bkg_norm.keys()]
    # cols_loglikelihood = [fits.Column(name=f"{self.dataset[i].name}", array=[float(result['loglikelihood'][i]) for result in self.results], format='D') 
    #                     for i in range(len(self.dataset))]

    # table_alpha = fits.BinTableHDU.from_columns([col_iteration, col_alpha])
    # table_alpha.name = "alpha"

    # table_bkg_norm = fits.BinTableHDU.from_columns([col_iteration] + cols_bkg_norm)
    # table_bkg_norm.name = "bkg_norm"

    # table_loglikelihood = fits.BinTableHDU.from_columns([col_iteration] + cols_loglikelihood)
    # table_loglikelihood.name = "loglikelihood"

    # hdul = fits.HDUList([primary_hdu, table_alpha, table_bkg_norm, table_loglikelihood])
    # hdul.writeto(f'{RESULTS_DIR}/results.fits',  overwrite=True)

def main():
    # Set up MPI
    comm = MPI.COMM_WORLD
    numtasks = comm.Get_size()
    taskid = comm.Get_rank()

    # Calculate the indices in Rij that the process has to parse. My hunch is that calculating these scalars individually will be faster than the MPI send broadcast overhead.
    averow = NUMROWS // numtasks
    extra_rows = NUMROWS % numtasks
    start_row = taskid * averow
    end_row = (taskid + 1) * averow if taskid < (numtasks - 1) else NUMROWS

    # Calculate the indices in Rji, i.e., Rij transpose, that the process has to parse.
    avecol = NUMCOLS // numtasks
    extra_cols = NUMCOLS % numtasks
    start_col = taskid * avecol
    end_col = (taskid + 1) * avecol if taskid < (numtasks - 1) else NUMCOLS

    # Initialise vectors required by all processes
    epsilon = np.zeros(NUMROWS)                 # All gatherv-ed. Explicit variable declaration.
    epsilon_fudge = 1e-12                       # To prevent divide-by-zero and underflow errors. Value taken from `almost_zero = 1e-12` in dataIF_COSI_DC2.py

    # Initialise epsilon_slice and C_slice. Explicit variable declarations. 
    epsilon_slice = np.zeros(end_row - start_row)
    C_slice = np.zeros(end_col - start_col)

    # Load R and RT into memory (single time if response matrix doesn't 
    # change with time)
    R = load_response_matrix(comm, start_row, end_row)
    RT = load_response_matrix_transpose(comm, start_col, end_col)

    # Loaded and broadcasted by master.
    M = np.empty(NUMCOLS, dtype=np.float64) 
    d = np.empty(NUMROWS, dtype=np.float64)  
    bg = np.zeros(NUMROWS)                  

# ****************************** MPI ******************************

# **************************** Part I *****************************

    '''*************** Master ***************'''

    if taskid == MASTER:

        # Pretty print definitions
        linebreak_stars = '**********************'
        linebreak_dashes = '----------------------'

        # Log input information (Only master node does this)
        save_results_flag = config.get('deconvolution:parameter:save_results', False)       # Extract from config file
        logger.info(linebreak_stars)
        logger.info(f'Number of elements in data space: {NUMROWS}')
        logger.info(f'Number of elements in model space: {NUMCOLS}')
        logger.info(f'Base directory: {BASE_DIR}')
        if save_results_flag == True:
            logger.info(f'Results directory (if save_results flag is set to True): {RESULTS_DIR}')
        logger.info(f'Configuration filename: {args.config_file}')
        logger.info(f'Master node: {MASTER}')
        logger.info(f'Maximum number of RL iterations: {MAXITER}')

        # Load Rj vector (response matrix summed along axis=i)
        Rj = load_axis0_summed_response_matrix()

        # Generate initial sky model from configuration file
        M = initial_sky_model(model_init_val=config.get('model_definition:initialization:parameter:value', [1e-4]))

        # Load event data and background model (intermediate files created in RichardsonLucyParallel.py)
        bg = load_bg_model()
        d = load_event_data()

        # Sanity check: print d
        print()
        print('Observed data-space d vector:')
        print(d)
        ## Pretty print
        print()
        print(linebreak_stars)

        # Initialise C vector. Only master requires full length. Explicit variable declaration.
        C = np.empty(NUMCOLS, dtype=np.float64)

        # Initialise update delta vector. Explicit variable declaration.
        delta = np.empty(NUMCOLS, dtype=np.float64)

        # Initialise list for results. See function register_result() for list elements. 
        results = []

    '''*************** Worker ***************'''

    if taskid > MASTER:
        # Only separate if... clause for NON-MASTER processes. 
        # Initialise C vector to None. Only master requires full length.
        C = None

    # Broadcast d vector
    comm.Bcast([d, MPI.DOUBLE], root=MASTER)

    # Scatter bg vector to epsilon_BG
    comm.Bcast([bg, MPI.DOUBLE], root=MASTER)
    # comm.Scatter(bg, [epsilon_BG, recvcounts, displacements, MPI.DOUBLE])

    # print(f"TaskID {taskid}, gathered broadcast")

    # Sanity check: print epsilon
    # if taskid == MASTER:
    #     print('epsilon_BG')
    #     print(bg)
    #     print()

# **************************** Part IIa *****************************

    '''***************** Begin Iterative Segment *****************'''
    # Set up initial values for iterating variables.
    # Exit if:
    ## 1. Max iterations are reached
    # for iter in tqdm(range(MAXITER)):
    for iter in range(MAXITER):

        '''*************** Master ***************'''
        if taskid == MASTER:
            # Pretty print - starting
            print(f"Starting iteration {iter + 1}")
            # logger.info(f"## Iteration {self.iteration_count}/{self.iteration_max} ##")
            # logger.info("<< E-step >>")


    # Calculate epsilon vector and all gatherv

        '''**************** All *****************'''

        '''Synchronization Barrier 1'''
        # Broadcast M vector
        comm.Bcast([M, MPI.DOUBLE], root=MASTER)

        # Calculate epsilon slice
        epsilon_BG = bg[start_row:end_row]             # TODO: Change the way epsilon_BG is loaded. Make it taskID dependent through MPI.Scatter for example. Use `recvcounts`
        epsilon_slice = np.dot(R, M) + epsilon_BG + epsilon_fudge   # TODO: For a more general implementation, see calc_expectation() in dataIF_COSI_DC2.py

        '''Synchronization Barrier 2'''
        # All vector gather epsilon slices
        recvcounts = [averow] * (numtasks-1) + [averow + extra_rows]
        displacements = np.arange(numtasks) * averow
        comm.Allgatherv(epsilon_slice, [epsilon, recvcounts, displacements, MPI.DOUBLE])

        # Sanity check: print epsilon
        # if taskid == MASTER:
        #     print('epsilon')
        #     print(epsilon)
        #     print(epsilon.min(), epsilon.max())
        #     print()

# **************************** Part IIb *****************************

    # Calculate C vector and gatherv
    
        '''**************** All *****************'''

        # Calculate C slice
        C_slice = np.dot(RT.T, d/epsilon)

        '''Synchronization Barrier 3'''
        # All vector gather C slices
        recvcounts = [avecol] * (numtasks-1) + [avecol + extra_cols]
        displacements = np.arange(numtasks) * avecol
        comm.Gatherv(C_slice, [C, recvcounts, displacements, MPI.DOUBLE], root=MASTER)

# **************************** Part IIc *****************************

    # Iterative update of model-space M vector

        if taskid == MASTER:

            # logger.info("<< M-step >>")

            # Sanity check: print C
            # print('C')
            # print(C)
            # print(C.min(), C.max())
            # print()

            delta = C / Rj - 1
            M = M + delta * M           # Allows for optimization features presented in Siegert et al. 2020

            # Sanity check: print M
            # print('M')
            # print(np.round(M, 5))
            # print(np.round(M.max(), 5))

            # Sanity check: print delta
            # print('delta')
            # print(delta)

            # Pretty print - completion
            print(f"Done")
            print(linebreak_dashes)

            # Save iteration
            if save_results_flag == True:
                results.append(register_result(iter, M, delta))
  
    '''****************** End Iterative Segment ******************'''

    # Print converged M
    if taskid == MASTER:
        # logger.info("<< Registering Result >>")
        print('Converged M vector:')
        print(np.round(M, 5))
        print(np.round(M.max(), 5))
        print(np.sum(M))
        print()

        if save_results_flag == True:
            save_results(results)

        # MAXITER
        if iter == (MAXITER - 1):
            print(f'Reached maximum iterations = {MAXITER}')
            print(linebreak_stars)
            print()

    # MPI Shutdown
    MPI.Finalize()
    
if __name__ == "__main__":
    main()
