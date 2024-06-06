import warnings
import numpy as np
from tqdm.autonotebook import tqdm
import astropy.units as u

from histpy import Histogram, Axes

from cosipy.response import FullDetectorResponse
from cosipy.data_io import BinnedData
from .coordsys_conversion_matrix import CoordsysConversionMatrix

class DataLoader(object):
    """
    A class to manage data for image analysis, 
    namely event data, background model, response, coordsys conversion matrix.
    Ideally, these data should be input directly to ImageDeconvolution class,
    but considering their data formats are not fixed, this class is introduced.
    The purpose of this class is to check the consistency between input data and calculate intermediate files etc.
    In the future, this class may be removed or hidden in ImageDeconvolution class.
    """

    def __init__(self):
        self.event_dense = None
        self.bkg_dense = None
        self.full_detector_response = None
        self.coordsys_conv_matrix = None
        self.mask = None

        self.is_miniDC2_format = False

        self.response_on_memory = False

        self.exposure_map = None

    @classmethod
    def load(cls, event_binned_data, bkg_binned_data, rsp, coordsys_conv_matrix, is_miniDC2_format = False):
        """
        Load data

        Parameters
        ----------
        event_binned_data : :py:class:`histpy.Histogram`
            Event histogram
        bkg_binned_data : :py:class:`histpy.Histogram`
            Background model
        rsp : :py:class:`histpy.Histogram` or :py:class:`cosipy.response.FullDetectorResponse`
            Response
        coordsys_conv_matrix : :py:class:`cosipy.image_deconvolution.CoordsysConversionMatrix`
            Coordsys conversion matrix 
        is_miniDC2_format : bool, default False
            Whether the file format is for mini-DC2. It will be removed in the future.

        Returns
        -------
        :py:class:`cosipy.image_deconvolution.DataLoader`
            DataLoader instance containing the input data set
        """

        new = cls()

        new.event_dense = event_binned_data.to_dense()

        new.bkg_dense = bkg_binned_data.to_dense()

        new.full_detector_response = rsp

        new.coordsys_conv_matrix = coordsys_conv_matrix

        new.is_miniDC2_format = is_miniDC2_format

        return new
    
    @classmethod
    def load_from_filepath(cls, event_hdf5_filepath = None, event_yaml_filepath = None, 
                           bkg_hdf5_filepath = None, bkg_yaml_filepath = None, 
                           rsp_filepath = None, ccm_filepath = None,
                           is_miniDC2_format = False):
        """
        Load data from file pathes

        Parameters
        ----------
        event_hdf5_filepath : str or None, default None
            File path of HDF5 file for event histogram.
        event_yaml_filepath : str or None, default None
            File path of yaml file to read the HDF5 file.
        bkg_hdf5_filepath : str or None, default None
            File path of HDF5 file for background model.
        bkg_yaml_filepath : str or None, default None
            File path of yaml file to read the HDF5 file.
        rsp_filepath : str or None, default None
            File path of the response matrix.
        ccm_filepath : str or None, default None
            File path of the coordsys conversion matrix.
        is_miniDC2_format : bool, default False
            Whether the file format is for mini-DC2. should be removed in the future.

        Returns
        -------
        :py:class:`cosipy.image_deconvolution.DataLoader`
            DataLoader instance containing the input data set
        """

        new = cls()

        new.set_event_from_filepath(event_hdf5_filepath, event_yaml_filepath)

        new.set_bkg_from_filepath(bkg_hdf5_filepath, bkg_yaml_filepath)
        
        new.set_rsp_from_filepath(rsp_filepath)

        new.set_ccm_from_filepath(ccm_filepath)

        new.is_miniDC2_format = is_miniDC2_format

        return new

    def set_event_from_filepath(self, hdf5_filepath, yaml_filepath):
        """
        Load event data from file pathes

        Parameters
        ----------
        hdf5_filepath : str
            File path of HDF5 file for event histogram.
        yaml_filepath : str
            File path of yaml file to read the HDF5 file.
        """

        self._event_hdf5_filepath = hdf5_filepath
        self._event_yaml_filepath = yaml_filepath

        print(f'... loading event from {hdf5_filepath} and {yaml_filepath}')

        event = BinnedData(self._event_yaml_filepath)
        event.load_binned_data_from_hdf5(self._event_hdf5_filepath)

        self.event_dense = event.binned_data.to_dense()

        print("... Done ...")

    def set_bkg_from_filepath(self, hdf5_filepath, yaml_filepath):
        """
        Load background model from file pathes

        Parameters
        ----------
        hdf5_filepath : str
            File path of HDF5 file for background model.
        yaml_filepath : str
            File path of yaml file to read the HDF5 file.
        """

        self._bkg_hdf5_filepath = hdf5_filepath
        self._bkg_yaml_filepath = yaml_filepath

        print(f'... loading background from {hdf5_filepath} and {yaml_filepath}')

        bkg = BinnedData(self._bkg_yaml_filepath)
        bkg.load_binned_data_from_hdf5(self._bkg_hdf5_filepath)

        self.bkg_dense = bkg.binned_data.to_dense()

        print("... Done ...")

    def set_rsp_from_filepath(self, filepath):
        """
        Load response matrix from file pathes

        Parameters
        ----------
        filepath : str
            File path of the response matrix.
        """

        self._rsp_filepath = filepath

        print(f'... loading full detector response from {filepath}')

        self.full_detector_response = FullDetectorResponse.open(self._rsp_filepath) 

        print("... Done ...")

    def set_ccm_from_filepath(self, filepath):
        """
        Load coordsys conversion matrix from file pathes

        Parameters
        ----------
        filepath : str
            File path of the coordsys conversion matrix.
        """

        self._ccm_filepath = filepath
        
        print(f'... loading coordsys conversion matrix from {filepath}')

        self.coordsys_conv_matrix = CoordsysConversionMatrix.open(self._ccm_filepath)

        print("... Done ...")

    def _check_file_registration(self):
        """
        Check whether files are loaded.

        Returns
        -------
        bool
            True if all required files are loaded.
        """

        print(f"... checking the file registration ...")

        if self.event_dense and self.bkg_dense \
        and self.full_detector_response and self.coordsys_conv_matrix:

            print(f"    --> pass")
            return True
        
        return False

    def _check_axis_consistency(self):
        """
        Check whether the axes of event/background/response are consistent with each other.

        Returns
        -------
        bool
            True if their axes are consistent.
        """
        
        print(f"... checking the axis consistency ...")

        # check the axes of the event/background files
        if self.coordsys_conv_matrix.binning_method == 'Time':
            axis_name = ['Time', 'Em', 'Phi', 'PsiChi']

        elif self.coordsys_conv_matrix.binning_method == 'ScAtt':
            axis_name = ['ScAtt', 'Em', 'Phi', 'PsiChi']

        for name in axis_name:
            if not self.event_dense.axes[name] == self.bkg_dense.axes[name]:
                print(f"Warning: the axis {name} is not consistent between the event and background!")
                return False

        # check the axes of the event/response files
        axis_name = ['Em', 'Phi', 'PsiChi']
        
        for name in axis_name:
            if not self.event_dense.axes[name] == self.full_detector_response.axes[name]:
                print(f"Warning: the axis {name} is not consistent between the event and response!")
                return False

        print(f"    --> pass")
        return True

    def _modify_axes(self):
        """
        Modify the axes of data. This method will be removed in the future.
        """

        warnings.warn("Note that _modify_axes() in DataLoader was implemented for a temporary use. It will be removed in the future.", FutureWarning)
        warnings.warn("Make sure to perform _modify_axes() only once after the data are loaded.")

        if self.coordsys_conv_matrix.binning_method == 'Time':
            axis_name = ['Time', 'Em', 'Phi', 'PsiChi']

        elif self.coordsys_conv_matrix.binning_method == 'ScAtt':
            axis_name = ['ScAtt', 'Em', 'Phi', 'PsiChi']

        for name in axis_name:

            print(f"... checking the axis {name} of the event and background files...")
            
            event_edges, event_unit = self.event_dense.axes[name].edges, self.event_dense.axes[name].unit
            bkg_edges, bkg_unit = self.bkg_dense.axes[name].edges, self.bkg_dense.axes[name].unit

            if np.all(event_edges == bkg_edges):
                print(f"    --> pass (edges)") 
            else:
                print(f"Warning: the edges of the axis {name} are not consistent between the event and background!")
                print(f"        event      : {event_edges}")
                print(f"        background : {bkg_edges}")
                return False

            if event_unit == bkg_unit:
                print(f"    --> pass (unit)") 
            else:
                print(f"Warning: the unit of the axis {name} are not consistent between the event and background!")
                print(f"        event      : {event_unit}")
                print(f"        background : {bkg_unit}")
                return False

        # check the axes of the event/response files. 
        # Note that currently (2023-08-29) no unit is stored in the binned data. So only the edges are compared. This should be modified in the future.

        axis_name = ['Em', 'Phi', 'PsiChi']
        
        for name in axis_name:

            print(f"...checking the axis {name} of the event and response files...")

            event_edges, event_unit = self.event_dense.axes[name].edges, self.event_dense.axes[name].unit
            response_edges, response_unit = self.full_detector_response.axes[name].edges, self.full_detector_response.axes[name].unit
            
            if type(response_edges) == u.quantity.Quantity and self.is_miniDC2_format == True:
                response_edges = response_edges.value

            if np.all(event_edges == response_edges):
                print(f"    --> pass (edges)") 
            else:
                print(f"Warning: the edges of the axis {name} are not consistent between the event and background!")
                print(f"        event      : {event_edges}")
                print(f"        response : {response_edges}")
                return False

        axes_cds = Axes([self.event_dense.axes[0], \
                         self.full_detector_response.axes["Em"], \
                         self.full_detector_response.axes["Phi"], \
                         self.full_detector_response.axes["PsiChi"]])
        
        self.event_dense = Histogram(axes_cds, unit = self.event_dense.unit, contents = self.event_dense.contents)

        self.bkg_dense = Histogram(axes_cds, unit = self.bkg_dense.unit, contents = self.bkg_dense.contents)

        print(f"The axes in the event and background files are redefined. Now they are consistent with those of the response file.")

    def load_full_detector_response_on_memory(self):
        """
        Load a response file on the computer memory.
        """

        axes_image_response = [self.full_detector_response.axes["NuLambda"], self.full_detector_response.axes["Ei"],
                               self.full_detector_response.axes["Em"], self.full_detector_response.axes["Phi"], self.full_detector_response.axes["PsiChi"]]

        self.image_response_dense = Histogram(axes_image_response, unit = self.full_detector_response.unit)

        nside = self.full_detector_response.axes["NuLambda"].nside
        npix = self.full_detector_response.axes["NuLambda"].npix 
    
        if self.is_miniDC2_format:
            for ipix in tqdm(range(npix)):
                self.image_response_dense[ipix] = np.sum(self.full_detector_response[ipix].to_dense(), axis = (4,5)) #Ei, Em, Phi, ChiPsi
        else:
            contents = self.full_detector_response._file['DRM']['CONTENTS'][:]
            self.image_response_dense[:] = contents * self.full_detector_response.unit

        self.response_on_memory = True

    def calc_exposure_map(self):
        """
        Calculate exposure_map, which is an intermidiate matrix used in RL algorithm.
        """

        print("... (DataLoader) calculating a projected image response ...")

        self.exposure_map = Histogram([ self.coordsys_conv_matrix.axes["lb"], self.full_detector_response.axes["Ei"] ],
                                                        unit = self.full_detector_response.unit * self.coordsys_conv_matrix.unit)

        if self.response_on_memory == False:
            self.load_full_detector_response_on_memory()

        self.exposure_map[:] = np.tensordot( np.sum(self.coordsys_conv_matrix, axis = (0)), 
                                                               np.sum(self.image_response_dense, axis = (2,3,4)),
                                                               axes = ([1], [0]) ) * self.full_detector_response.unit * self.coordsys_conv_matrix.unit
        # [Time/ScAtt, lb, NuLambda] -> [lb, NuLambda]
        # [NuLambda, Ei, Em, Phi, PsiChi] -> [NuLambda, Ei]
        # [lb, NuLambda] x [NuLambda, Ei] -> [lb, Ei]

        if np.any(self.exposure_map.contents == 0):
            print("... There are zero-exposure pixels. Preparing a mask to ignore them ...")
            self.mask = Histogram(self.exposure_map.axes, \
                                  contents = self.exposure_map.contents > 0)

    def calc_expectation(self, model_map, bkg_norm = 1.0, almost_zero = 1e-12):
        """
        Calculate expected counts from a given model map.

        Parameters
        ----------
        model_map : :py:class:`cosipy.image_deconvolution.ModelMap`
            Model map
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

        expectation = Histogram(self.event_dense.axes) 

        map_rotated = np.tensordot(self.coordsys_conv_matrix.contents, model_map.contents, axes = ([1], [0])) 
        # ['Time/ScAtt', 'lb', 'NuLambda'] x ['lb', 'Ei'] -> [Time/ScAtt, NuLambda, Ei]
        map_rotated *= self.coordsys_conv_matrix.unit * model_map.unit
        map_rotated *= model_map.axes['lb'].pixarea()
        # data.coordsys_conv_matrix.contents is sparse, so the unit should be restored.
        # the unit of map_rotated is 1/cm2 ( = s * 1/cm2/s/sr * sr)

        expectation[:] = np.tensordot( map_rotated, self.image_response_dense.contents, axes = ([1,2], [0,1]))
        expectation += self.bkg_dense * bkg_norm
        expectation += almost_zero
        
        return expectation

    def calc_T_product(self, dataspace_matrix):

        hist_unit = self.exposure_map.unit
        if dataspace_matrix.unit is not None:
            hist_unit *= dataspace_matrix.unit

        hist = Histogram(self.exposure_map.axes, \
                         unit = hist_unit)

        _ = np.tensordot(dataspace_matrix.contents, self.image_response_dense.contents, axes = ([1,2,3], [2,3,4])) 
            # [Time/ScAtt, Em, Phi, PsiChi] x [NuLambda, Ei, Em, Phi, PsiChi] -> [Time/ScAtt, NuLambda, Ei]

        hist[:] = np.tensordot(self.coordsys_conv_matrix.contents, _, axes = ([0,2], [0,1])) \
                             * _.unit * self.coordsys_conv_matrix.unit
            # [Time/ScAtt, lb, NuLambda] x [Time/ScAtt, NuLambda, Ei] -> [lb, Ei]
            # note that coordsys_conv_matrix is the sparse, so the unit should be recovered.

        return hist
