import sys
from pathlib import Path

import logging
logger = logging.getLogger(__name__)

import numpy as np
import astropy.units as u
import h5py
from histpy import Histogram, Axes, Axis, HealpixAxis

from cosipy.response import FullDetectorResponse
from cosipy.image_deconvolution import ImageDeconvolutionDataInterfaceBase

# Define npix in NuLambda and PsiChi
# TODO: information is contained in FullDetectorResponse 
# and will be supported at a later release
NUMROWS = 3072
NUMCOLS = 3072

# Define data paths
DRM_DIR = Path('/Users/penguin/Documents/Grad School/Research/COSI/COSIpy/docs/tutorials/data')
DATA_DIR = Path('/Users/penguin/Documents/Grad School/Research/COSI/COSIpy/docs/tutorials/image_deconvolution/511keV/GalacticCDS')

def load_response_matrix(comm, start_col, end_col, filename):
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
                axis_tmp = HealpixAxis(edges = axis_tmp.edges[start_col:end_col+1], 
                                       label = axis_tmp.label, 
                                       scale = axis_tmp._scale,
                                       coordsys = axis_tmp._coordsys,
                                       nside = axis_tmp.nside)

            axes += [axis_tmp]

    return Histogram(axes, contents = R, unit = unit)

def load_response_matrix_transpose(comm, start_row, end_row, filename):
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
                axis_tmp = HealpixAxis(edges = axis_tmp.edges[start_row:end_row+1], 
                                       label = axis_tmp.label, 
                                       scale = axis_tmp._scale,
                                       coordsys = axis_tmp._coordsys,
                                       nside = axis_tmp.nside)

            axes += [axis_tmp]

    return Histogram(axes, contents = RT, unit = unit)

class DataIFWithParallel(ImageDeconvolutionDataInterfaceBase):
    """
    A subclass of ImageDeconvolutionDataInterfaceBase for the COSI data challenge 2.
    """

    def __init__(self, event_filename, bkg_filename, drm_filename, name = None, comm = None):

        numtasks = comm.Get_size()
        taskid = comm.Get_rank()

        print(f'TaskID = {taskid}, Number of tasks = {numtasks}')

        ImageDeconvolutionDataInterfaceBase.__init__(self, name)

        # Calculate the indices in Rij that the process has to parse. My hunch is that calculating these scalars individually will be faster than the MPI send broadcast overhead.
        self.averow = NUMROWS // numtasks
        self.extra_rows = NUMROWS % numtasks
        self.start_row = taskid * self.averow
        self.end_row = (taskid + 1) * self.averow if taskid < (numtasks - 1) else NUMROWS

        # Calculate the indices in Rji, i.e., Rij transpose, that the process has to parse.
        self.avecol = NUMCOLS // numtasks
        self.extra_cols = NUMCOLS % numtasks
        self.start_col = taskid * self.avecol
        self.end_col = (taskid + 1) * self.avecol if taskid < (numtasks - 1) else NUMCOLS

        # Load event_binned_data
        event = Histogram.open(DATA_DIR / event_filename)
        self._event = event.project(['Em', 'Phi', 'PsiChi']).to_dense()

        # Load dict_bg_binned_data
        bkg = Histogram.open(DATA_DIR / bkg_filename)
        self._bkg_models = {"total": bkg.project(['Em', 'Phi', 'PsiChi']).to_dense()}

        # Load response and response transpose
        self._image_response = load_response_matrix(comm, self.start_col, self.end_col, filename=drm_filename)
        self._image_response_T = load_response_matrix_transpose(comm, self.start_row, self.end_row, filename=drm_filename)

        # Set variable _model_axes
        # Derived from Parent class (ImageDeconvolutionDataInterfaceBase)
        axes = [self._image_response.axes['NuLambda'], self._image_response.axes['Ei']]
        axes[0].label = 'lb' 
        self._model_axes = Axes(axes)
        ## Create model_axes_slice
        axes = []
        for axis in self.model_axes:
            if axis.label == 'lb':
                axes.append(HealpixAxis(edges = axis.edges[self.start_row:self.end_row+1], 
                                        label = axis.label, 
                                        scale = axis._scale,
                                        coordsys = axis._coordsys,
                                        nside = axis.nside))
            else:
                axes.append(axis)
        self._model_axes_slice = Axes(axes)

        # Set variable _data_axes
        # Derived from Parent class (ImageDeconvolutionDataInterfaceBase)
        self._data_axes = self.event.axes
        ## Create data_axes_slice
        axes = []
        for axis in self.data_axes:
            if axis.label == 'PsiChi':
                axes.append(HealpixAxis(edges = axis.edges[self.start_col:self.end_col+1], 
                                        label = axis.label, 
                                        scale = axis._scale,
                                        coordsys = axis._coordsys,
                                        nside = axis.nside))
            else:
                axes.append(axis)
        self._data_axes_slice = Axes(axes)
        
        ## Create bkg_model_slice Histogram
        self._bkg_models_slice = {}
        for key in self._bkg_models:
            # if self._bkg_models[key].is_sparse:
            #     self._bkg_models[key] = self._bkg_models[key].to_dense()
            bkg_model = self._bkg_models[key]
            self._summed_bkg_models[key] = np.sum(bkg_model)
            self._bkg_models_slice[key] = bkg_model.slice[:, :, self.start_col:self.end_col]
        
        # None if using Galactic CDS, required if using local CDS
        self._coordsys_conv_matrix = None 

        # Calculate exposure map
        self._calc_exposure_map()
        
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

        expectation = Histogram(self._data_axes_slice)
        
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
                expectation += self.bkg_model_slice(key) * dict_bkg_norm[key]

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

        hist = Histogram(self._model_axes_slice, unit = hist_unit)
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

    def bkg_model_slice(self, key):
        return self._bkg_models_slice[key]
