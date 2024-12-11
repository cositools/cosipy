import sys
from tqdm import tqdm
import warnings
from pathlib import Path

import logging
logger = logging.getLogger(__name__)

import numpy as np
import astropy.units as u
from mpi4py import MPI
import h5py
from histpy import Histogram, Axes, Axis
from yayc import Configurator

from cosipy.response import FullDetectorResponse
from cosipy.image_deconvolution import ImageDeconvolutionDataInterfaceBase, RichardsonLucyWithParallel, AllSkyImageModel, ImageDeconvolution

# Define MPI variables
MASTER = 0                      # Indicates master process
NUMROWS = 3072
NUMCOLS = 3072
DRM_DIR = Path('/Users/penguin/Documents/Grad School/Research/COSI/COSIpy/docs/tutorials/data')
DATA_DIR = Path('/Users/penguin/Documents/Grad School/Research/COSI/COSIpy/docs/tutorials/image_deconvolution/511keV/GalacticCDS')

def main():
    '''
    ImageDeconvolution() script
    '''

    # Set up MPI
    comm = MPI.COMM_WORLD

    # Create dataset
    dataset = DataIFWithParallel(comm=comm)     # Convert to list of objects before passing to RichardsonLucy class

    # print(f'Initial Model: {dataset.model.axes.labels}')
    # print(f'Event Data: {dataset.event.axes.labels}')
    # print(f'Background Model keys: {dataset.keys_bkg_models()}')
    # print(f"Background Model: {dataset.bkg_model('total').axes.labels}")
    # print(f'Response Matrix: {dataset._image_response.axes.labels}')
    # print(f'Model Axes: {[axis.edges for axis in dataset.model_axes]}')
    # print(f'Data Axes: {[axis.edges for axis in dataset.data_axes]}')

    # Create image deconvolution object
    image_deconvolution = ImageDeconvolution()

    # set data_interface to image_deconvolution
    image_deconvolution.set_dataset([dataset])

    # set a parameter file for the image deconvolution
    parameter_filepath = DATA_DIR / 'imagedeconvolution_parfile_gal_511keV.yml'
    image_deconvolution.read_parameterfile(parameter_filepath)

    # Initialize model
    image_deconvolution.initialize()

    # Execute deconvolution
    image_deconvolution.run_deconvolution()

    # parameter = read_parameterfile()

    # initial_model = model_initialization(parameter)

    # if dataset.model_axes != initial_model.axes:
    #     raise ValueError("The model axes mismatches with the reponse in the dataset!")

    # deconvolution = register_deconvolution_algorithm(initial_model = initial_model,
    #                                  dataset = [dataset],
    #                                  parameter = parameter['deconvolution:parameter']
    #                                  )
    
    # run_deconvolution(deconvolution)

    # MPI Shutdown
    MPI.Finalize()

def read_parameterfile(parameter_filepath: str | Path = 'imagedeconvolution_parfile_gal_511keV.yml'):
    return Configurator.open(DATA_DIR / parameter_filepath)

def model_initialization(parameter):
    initial_model = AllSkyImageModel.instantiate_from_parameters(parameter['model_definition:property'])
    initial_model.set_values_from_parameters(parameter['model_definition:initialization'])
    return initial_model

def register_deconvolution_algorithm(initial_model, dataset, parameter):
    # Call RL
    deconvolution = RichardsonLucyWithParallel(initial_model = initial_model,
                                               dataset = dataset,
                                               mask = None,
                                               parameter = parameter)
    return deconvolution

def run_deconvolution(deconvolution):
    print("#### Image Deconvolution Starts ####")
    
    print(f"<< Initialization >>")
    deconvolution.initialization()
    
    stop_iteration = False
    for i in tqdm(range(deconvolution.iteration_max)):
        if stop_iteration:
            break
        stop_iteration = deconvolution.iteration()

    print(f"<< Finalization >>")
    deconvolution.finalization()

    print("#### Image Deconvolution Finished ####")

def load_response_matrix(comm, start_col, end_col, filename='response.h5'):
    '''
    Response matrix
    '''
    with h5py.File(DRM_DIR / filename, "r", driver="mpio", comm=comm) as f1:
        dataset = f1['hist/contents']
        R = dataset[1:-1, 1:-1, 1:-1, 1:-1, start_col+1:end_col+1]

        hist_group = f1['hist']
        if 'unit' in hist_group.attrs:
            unit = u.Unit(hist_group.attrs['unit'])

        # Axes
        axes_group = hist_group['axes']

        axes = []
        for axis in axes_group.values():
            label = axis.attrs['label']

            # Get class. Backwards compatible with version
            # with only Axis
            axis_cls = Axis

            if '__class__' in axis.attrs:
                class_module, class_name = axis.attrs['__class__']
                axis_cls = getattr(sys.modules[class_module], class_name)
                axis_tmp = axis_cls._open(axis)

            if label == 'PsiChi':
                axis_tmp = Axis(edges = axis_tmp.edges[start_col:end_col+1], 
                                label = axis_tmp.label, 
                                scale = axis_tmp._scale)

            axes += [axis_tmp]

    return Histogram(axes, contents = R, unit = unit)

def load_response_matrix_transpose(comm, start_row, end_row, filename='response.h5'):
    '''
    Response matrix tranpose
    '''
    with h5py.File(DRM_DIR / filename, "r", driver="mpio", comm=comm) as f1:
        dataset = f1['hist/contents']
        RT = dataset[start_row+1:end_row+1, 1:-1, 1:-1, 1:-1, 1:-1]

        hist_group = f1['hist']
        if 'unit' in hist_group.attrs:
            unit = u.Unit(hist_group.attrs['unit'])

        # Axes
        axes_group = hist_group['axes']

        axes = []
        for axis in axes_group.values():
            label = axis.attrs['label']

            # Get class. Backwards compatible with version
            # with only Axis
            axis_cls = Axis

            if '__class__' in axis.attrs:
                class_module, class_name = axis.attrs['__class__']
                axis_cls = getattr(sys.modules[class_module], class_name)
                axis_tmp = axis_cls._open(axis)

            if label == 'NuLambda':
                axis_tmp = Axis(edges = axis_tmp.edges[start_row:end_row+1], 
                                label = axis_tmp.label, 
                                scale = axis_tmp._scale)

            axes += [axis_tmp]

    return Histogram(axes, contents = RT, unit = unit)

class DataIFWithParallel(ImageDeconvolutionDataInterfaceBase):
    """
    A subclass of ImageDeconvolutionDataInterfaceBase for the COSI data challenge 2.
    """

    def __init__(self, name = None, comm = None):

        numtasks = comm.Get_size()
        taskid = comm.Get_rank()

        print(f'TaskID = {taskid}, Number of tasks = {numtasks}')

        ImageDeconvolutionDataInterfaceBase.__init__(self, name)

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

        # Load event_binned_data
        event = Histogram.open(DATA_DIR / "511keV_dc2_galactic_event.hdf5")
        self._event = event.project(['Em', 'Phi', 'PsiChi'])

        # Load dict_bg_binned_data
        bg = Histogram.open(DATA_DIR / "511keV_dc2_galactic_bkg.hdf5")
        self._bkg_models = {"total": bg.project(['Em', 'Phi', 'PsiChi'])}

        # Load response and response transpose
        self._image_response = load_response_matrix(comm, start_col, end_col, filename='psr_gal_511_DC2.h5')
        self._image_response_T = load_response_matrix_transpose(comm, start_row, end_row, filename='psr_gal_511_DC2.h5')

        # Set variable _model_axes
        # Derived from Parent class (ImageDeconvolutionDataInterfaceBase)
        axes = [self._image_response.axes['NuLambda'], self._image_response.axes['Ei']]
        axes[0].label = 'lb' 
        self._model_axes = Axes(axes)

        # Set variable _data_axes
        # Derived from Parent class
        axes = [self._image_response_T.axes['Em'], self._image_response_T.axes['Phi'], self._image_response_T.axes['PsiChi']]
        self._data_axes = Axes(axes)

    @classmethod
    def load(cls, name: str, event_binned_data: Histogram, dict_bkg_binned_data: dict, rsp, coordsys_conv_matrix = None, is_miniDC2_format: bool = False):
        """
        Load data

        Parameters
        ----------
        name : str
            The name of data
        event_binned_data : :py:class:`histpy.Histogram`
            Event histogram
        dict_bkg_binned_data : dict
            Background models as {background_model_name: :py:class:`histpy.Histogram`}
        rsp : :py:class:`histpy.Histogram` or :py:class:`cosipy.response.FullDetectorResponse`
            Response
        coordsys_conv_matrix : :py:class:`cosipy.image_deconvolution.CoordsysConversionMatrix`, default False
            Coordsys conversion matrix 
        is_miniDC2_format : bool, default False
            Whether the file format is for mini-DC2. It will be removed in the future.

        Returns
        -------
        :py:class:`cosipy.image_deconvolution.DataIF_COSI_DC2`
            An instance of DataIF_COSI_DC2 containing the input data set
        """

        new = cls(name)

        new._event = event_binned_data.to_dense()

        new._bkg_models = dict_bkg_binned_data

        for key in new._bkg_models:
            if new._bkg_models[key].is_sparse:
                new._bkg_models[key] = new._bkg_models[key].to_dense()

            new._summed_bkg_models[key] = np.sum(new._bkg_models[key])

        new._coordsys_conv_matrix = coordsys_conv_matrix

        new.is_miniDC2_format = is_miniDC2_format

        if isinstance(rsp, FullDetectorResponse):
            logger.info('Loading the response matrix onto your computer memory...')
            new._load_full_detector_response_on_memory(rsp, is_miniDC2_format)
            logger.info('Finished')
        elif isinstance(rsp, Histogram):
            new._image_response = rsp
        
        # We modify the axes in event, bkg_models, response. This is only for DC2.
        new._modify_axes()
        
        new._data_axes = new._event.axes
        
    def _calc_exposure_map(self):
        """
        Calculate exposure_map, which is an intermidiate matrix used in RL algorithm.
        """

        logger.info("Calculating an exposure map...")
        
        if self._coordsys_conv_matrix is None:
            self._exposure_map = Histogram(self._model_axes, unit = self._image_response.unit * u.sr)
            self._exposure_map[:] = np.sum(self._image_response.contents, axis = (2,3,4)) * self.model_axes['lb'].pixarea()
        else:
            self._exposure_map = Histogram(self._model_axes, unit = self._image_response.unit * self._coordsys_conv_matrix.unit * u.sr)
            self._exposure_map[:] = np.tensordot(np.sum(self._coordsys_conv_matrix, axis = (0)), 
                                                 np.sum(self._image_response, axis = (2,3,4)),
                                                 axes = ([1], [0]) ) * self._image_response.unit * self._coordsys_conv_matrix.unit * self.model_axes['lb'].pixarea()
            # [Time/ScAtt, lb, NuLambda] -> [lb, NuLambda]
            # [NuLambda, Ei, Em, Phi, PsiChi] -> [NuLambda, Ei]
            # [lb, NuLambda] x [NuLambda, Ei] -> [lb, Ei]

        logger.info("Finished...")

    def calc_expectation(self, model, dict_bkg_norm = None, almost_zero = 1e-12):
        """
        Calculate expected counts from a given model.

        Parameters
        ----------
        model : :py:class:`cosipy.image_deconvolution.AllSkyImageModel`
            Model map
        dict_bkg_norm : dict, default None
            background normalization for each background model, e.g, {'albedo': 0.95, 'activation': 1.05}
        almost_zero : float, default 1e-12
            In order to avoid zero components in extended count histogram, a tiny offset is introduced.
            It should be small enough not to effect statistics.

        Returns
        -------
        :py:class:`histpy.Histogram`
            Expected count histogram

        Notes
        -----
        This method should be implemented in a more general class, for example, extended source response class in the future.
        """
        # Currenly (2024-01-12) this method can work for both local coordinate CDS and in galactic coordinate CDS.
        # This is just because in DC2 the rotate response for galactic coordinate CDS does not have an axis for time/scatt binning.
        # However it is likely that it will have such an axis in the future in order to consider background variability depending on time and pointign direction etc.
        # Then, the implementation here will not work. Thus, keep in mind that we need to modify it once the response format is fixed.

        expectation = Histogram(self.data_axes)
        
        if self._coordsys_conv_matrix is None:
            expectation[:] = np.tensordot( model.contents, self._image_response.contents, axes = ([0,1],[0,1])) * model.axes['lb'].pixarea()
            # ['lb', 'Ei'] x [NuLambda(lb), Ei, Em, Phi, PsiChi] -> [Em, Phi, PsiChi]
        else:
            map_rotated = np.tensordot(self._coordsys_conv_matrix.contents, model.contents, axes = ([1], [0])) 
            # ['Time/ScAtt', 'lb', 'NuLambda'] x ['lb', 'Ei'] -> [Time/ScAtt, NuLambda, Ei]
            map_rotated *= self._coordsys_conv_matrix.unit * model.unit
            map_rotated *= model.axes['lb'].pixarea()
            # data.coordsys_conv_matrix.contents is sparse, so the unit should be restored.
            # the unit of map_rotated is 1/cm2 ( = s * 1/cm2/s/sr * sr)
            expectation[:] = np.tensordot( map_rotated, self._image_response.contents, axes = ([1,2], [0,1]))
            # [Time/ScAtt, NuLambda, Ei] x [NuLambda, Ei, Em, Phi, PsiChi] -> [Time/ScAtt, Em, Phi, PsiChi]

        if dict_bkg_norm is not None: 
            for key in self.keys_bkg_models():
                expectation += self.bkg_model(key) * dict_bkg_norm[key]
        expectation += almost_zero
        
        return expectation

    def calc_T_product(self, dataspace_histogram):
        """
        Calculate the product of the input histogram with the transonse matrix of the response function.
        Let R_{ij}, H_{i} be the response matrix and dataspace_histogram, respectively.
        Note that i is the index for the data space, and j is for the model space.
        In this method, \sum_{j} H{i} R_{ij}, namely, R^{T} H is calculated.

        Parameters
        ----------
        dataspace_histogram: :py:class:`histpy.Histogram`
            Its axes must be the same as self.data_axes

        Returns
        -------
        :py:class:`histpy.Histogram`
            The product with self.model_axes
        """
        # TODO: currently, dataspace_histogram is assumed to be a dense.

        hist_unit = self.exposure_map.unit
        if dataspace_histogram.unit is not None:
            hist_unit *= dataspace_histogram.unit

        hist = Histogram(self.model_axes, unit = hist_unit)

        if self._coordsys_conv_matrix is None:
            hist[:] = np.tensordot(dataspace_histogram.contents, self._image_response_T.contents, axes = ([0,1,2], [2,3,4])) * self.model_axes['lb'].pixarea()
            # [Em, Phi, PsiChi] x [NuLambda (lb), Ei, Em, Phi, PsiChi] -> [NuLambda (lb), Ei]
        else:
            _ = np.tensordot(dataspace_histogram.contents, self._image_response_T.contents, axes = ([1,2,3], [2,3,4])) 
            # [Time/ScAtt, Em, Phi, PsiChi] x [NuLambda, Ei, Em, Phi, PsiChi] -> [Time/ScAtt, NuLambda, Ei]

            hist[:] = np.tensordot(self._coordsys_conv_matrix.contents, _, axes = ([0,2], [0,1])) \
                        * _.unit * self._coordsys_conv_matrix.unit * self.model_axes['lb'].pixarea()
            # [Time/ScAtt, lb, NuLambda] x [Time/ScAtt, NuLambda, Ei] -> [lb, Ei]
            # note that coordsys_conv_matrix is sparse, so the unit should be recovered separately.

        return hist

    def calc_bkg_model_product(self, key, dataspace_histogram):
        """
        Calculate the product of the input histogram with the background model.
        Let B_{i}, H_{i} be the background model and dataspace_histogram, respectively.
        In this method, \sum_{i} B_{i} H_{i} is calculated.

        Parameters
        ----------
        key: str
            Background model name
        dataspace_histogram: :py:class:`histpy.Histogram`
            its axes must be the same as self.data_axes

        Returns
        -------
        flaot
        """
        # TODO: currently, dataspace_histogram is assumed to be a dense.

        if self._coordsys_conv_matrix is None:

            return np.tensordot(dataspace_histogram.contents, self.bkg_model(key).contents, axes = ([0,1,2], [0,1,2]))

        return np.tensordot(dataspace_histogram.contents, self.bkg_model(key).contents, axes = ([0,1,2,3], [0,1,2,3]))

    def calc_loglikelihood(self, expectation):
        """
        Calculate log-likelihood from given expected counts or model/expectation.

        Parameters
        ----------
        expectation : :py:class:`histpy.Histogram`
            Expected count histogram.

        Returns
        -------
        float
            Log-likelood
        """
        loglikelood = np.sum( self.event * np.log(expectation) ) - np.sum(expectation)

        return loglikelood

if __name__ == "__main__":
    main()