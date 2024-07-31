import numpy as np
from tqdm.autonotebook import tqdm
import astropy.units as u

import logging
logger = logging.getLogger(__name__)

import warnings

from histpy import Histogram, Axes

from cosipy.response import FullDetectorResponse

from .image_deconvolution_data_interface_base import ImageDeconvolutionDataInterfaceBase

class DataIF_COSI_DC2(ImageDeconvolutionDataInterfaceBase):
    """
    A subclass of ImageDeconvolutionDataInterfaceBase for the COSI data challenge 2.
    """

    def __init__(self, name = None):

        ImageDeconvolutionDataInterfaceBase.__init__(self, name)

        self._image_response = None # histpy.Histogram (dense)

        # None if using Galactic CDS, mandotary if using local CDS
        self._coordsys_conv_matrix = None 

        # optional
        self.is_miniDC2_format = False #should be removed in the future

    @classmethod
    def load(cls, name, event_binned_data, dict_bkg_binned_data, rsp, coordsys_conv_matrix = None, is_miniDC2_format = False):
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
        
        if new._coordsys_conv_matrix is None:
            axes = [new._image_response.axes['NuLambda'], new._image_response.axes['Ei']]
            axes[0].label = 'lb' 
            # The gamma-ray direction of pre-computed response in DC2 is in the galactic coordinate, not in the local coordinate.
            # Actually, it is labeled as 'NuLambda'. So I replace it with 'lb'.
            new._model_axes = Axes(axes)
        else:
            new._model_axes = Axes([new._coordsys_conv_matrix.axes['lb'], new._image_response.axes['Ei']])

        new._calc_exposure_map()

        return new

    def _modify_axes(self):
        """
        Modify the axes of data. This method will be removed in the future.
        """

        warnings.warn("Note that _modify_axes() in DataIF_COSI_DC2 was implemented for a temporary use. It will be removed in the future.", DeprecationWarning)

        if self._coordsys_conv_matrix is None:
            axis_name = ['Em', 'Phi', 'PsiChi']

        elif self._coordsys_conv_matrix.binning_method == 'Time':
            axis_name = ['Time', 'Em', 'Phi', 'PsiChi']

        elif self._coordsys_conv_matrix.binning_method == 'ScAtt':
            axis_name = ['ScAtt', 'Em', 'Phi', 'PsiChi']

        for name in axis_name:

            logger.info(f"... checking the axis {name} of the event and background files...")
            
            event_edges, event_unit = self._event.axes[name].edges, self._event.axes[name].unit

            for key in self._bkg_models:

                bkg_edges, bkg_unit = self._bkg_models[key].axes[name].edges, self._bkg_models[key].axes[name].unit

                if np.all(event_edges == bkg_edges):
                    logger.info(f"    --> pass (edges)") 
                else:
                    logger.error(f"Warning: the edges of the axis {name} are not consistent between the event and the background model {key}!")
                    logger.error(f"         event      : {event_edges}")
                    logger.error(f"         background : {bkg_edges}")
                    raise ValueError

        # check the axes of the event/response files. 
        # Note that currently (2023-08-29) no unit is stored in the binned data. So only the edges are compared. This should be modified in the future.

        axis_name = ['Em', 'Phi', 'PsiChi']
        
        for name in axis_name:

            logger.info(f"...checking the axis {name} of the event and response files...")

            event_edges, event_unit = self._event.axes[name].edges, self._event.axes[name].unit
            response_edges, response_unit = self._image_response.axes[name].edges, self._image_response.axes[name].unit
            
#            if type(response_edges) == u.quantity.Quantity and self.is_miniDC2_format == True:
            if event_unit is None and response_unit is not None and self.is_miniDC2_format == True: # this is only for the old data in the miniDC2 challenge. I will remove them in the near future (or in the final dataIF).
                response_edges = response_edges.value

            if np.all(event_edges == response_edges):
                logger.info(f"    --> pass (edges)") 
            else:
                logger.error(f"Warning: the edges of the axis {name} are not consistent between the event and background!")
                logger.error(f"        event      : {event_edges}")
                logger.error(f"        response : {response_edges}")
                raise ValueError

        if self._coordsys_conv_matrix is None:
            axes_cds = Axes([self._image_response.axes["Em"], \
                             self._image_response.axes["Phi"], \
                             self._image_response.axes["PsiChi"]])
        else:
            axes_cds = Axes([self._event.axes[0], \
                             self._image_response.axes["Em"], \
                             self._image_response.axes["Phi"], \
                             self._image_response.axes["PsiChi"]])
        
        self._event = Histogram(axes_cds, unit = self._event.unit, contents = self._event.contents)

        for key in self._bkg_models:
            bkg_model = self._bkg_models[key]
            self._bkg_models[key] = Histogram(axes_cds, unit = bkg_model.unit, contents = bkg_model.contents)
            del bkg_model

        logger.info(f"The axes in the event and background files are redefined. Now they are consistent with those of the response file.")

        return True

    def _load_full_detector_response_on_memory(self, full_detector_response, is_miniDC2_format):
        """
        Load a response file on the computer memory.
        """

        axes_image_response = [full_detector_response.axes["NuLambda"], full_detector_response.axes["Ei"],
                               full_detector_response.axes["Em"], full_detector_response.axes["Phi"], full_detector_response.axes["PsiChi"]]

        self._image_response = Histogram(axes_image_response, unit = full_detector_response.unit)

        nside = full_detector_response.axes["NuLambda"].nside
        npix = full_detector_response.axes["NuLambda"].npix 
    
        if is_miniDC2_format:
            for ipix in tqdm(range(npix)):
                self._image_response[ipix] = np.sum(full_detector_response[ipix].to_dense(), axis = (4,5)) #Ei, Em, Phi, ChiPsi
        else:
            contents = full_detector_response._file['DRM']['CONTENTS'][:]
            self._image_response[:] = contents * full_detector_response.unit

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
            hist[:] = np.tensordot(dataspace_histogram.contents, self._image_response.contents, axes = ([0,1,2], [2,3,4])) * self.model_axes['lb'].pixarea()
            # [Em, Phi, PsiChi] x [NuLambda (lb), Ei, Em, Phi, PsiChi] -> [NuLambda (lb), Ei]
        else:
            _ = np.tensordot(dataspace_histogram.contents, self._image_response.contents, axes = ([1,2,3], [2,3,4])) 
            # [Time/ScAtt, Em, Phi, PsiChi] x [NuLambda, Ei, Em, Phi, PsiChi] -> [Time/ScAtt, NuLambda, Ei]

            hist[:] = np.tensordot(self._coordsys_conv_matrix.contents, _, axes = ([0,2], [0,1])) \
                        * _.unit * self._coordsys_conv_matrix.unit * self.model_axes['lb'].pixarea()
            # [Time/ScAtt, lb, NuLambda] x [Time/ScAtt, NuLambda, Ei] -> [lb, Ei]
            # note that coordsys_conv_matrix is the sparse, so the unit should be recovered.

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
