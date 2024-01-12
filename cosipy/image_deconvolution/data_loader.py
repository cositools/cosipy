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

        self.is_miniDC2_format = False

        self.response_on_memory = False

        self.image_response_dense_projected = None

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

    '''
    def _check_sc_orientation_coverage(self):

        init_time_orientation = self.orientation.get_time()[0]
        init_time_event = Time(self.event_dense.axes["Time"].edges[0], format = 'unix')

        if not init_time_orientation <= init_time_event:
            print(f"Warning: the orientation file does not cover the observation")
            print(f"         initial time of the orientation file = {init_time_orientation}")
            print(f"         initial time of the event file       = {init_time_event}")
            return False

        end_time_orientation = self.orientation.get_time()[-1]
        end_time_event = Time(self.event_dense.axes["Time"].edges[-1], format = 'unix')

        if not end_time_event <= end_time_orientation:
            print(f"Warning: the orientation file does not cover the observation")
            print(f"         the end time of the orientation file = {end_time_orientation}")
            print(f"         the end time of the event file       = {end_time_event}")
            return False

        return True
    '''

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

    ''' 
    def calc_coordsys_conv_matrix(self): 

        if not self._check_file_registration():
            print("Please load all files!")
            return 
    
        if not self._check_axis_consistency():
            print("Please the axes of the input files!")
            return 

#        if not self._check_sc_orientation_coverage():
#            print("Please the axes of the input files!")
#            return 

        print("... (DataLoader) calculating a coordinate conversion matrix...")
        
        # make an empty histogram for the response calculation
        axis_model_map = HealpixAxis(nside = self.full_detector_response.axes["NuLambda"].nside, 
                                     coordsys = "galactic", label = "lb")

        axis_coordsys_conv_matrix = [ axis_model_map, self.event_dense.axes["Time"], self.full_detector_response.axes["NuLambda"] ] #lb, Time, NuLambda

        self.coordsys_conv_matrix = Histogram(axis_coordsys_conv_matrix, unit = u.s, sparse = True)

        # calculate a dwell time map at each time bin and sky location
        nside = self.full_detector_response.axes["NuLambda"].nside
        npix = self.full_detector_response.axes["NuLambda"].npix 

        for ipix in tqdm(range(npix)):
            theta, phi = hp.pix2ang(nside, ipix)
            l, b = phi, np.pi/2 - theta

            pixel_coord = SkyCoord(l, b, unit = u.rad, frame = 'galactic')

            for i_time, [init_time, end_time] in enumerate(self.coordsys_conv_matrix.axes["Time"].bounds):
                init_time = Time(init_time, format = 'unix')
                end_time = Time(end_time, format = 'unix')
    
                filtered_orientation = self.orientation.source_interval(init_time, end_time)
                pixel_movement = filtered_orientation.get_target_in_sc_frame(target_name = f"pixel_{ipix}_{i_time}",
                                                                             target_coord = pixel_coord,
                                                                             quiet = True)

                time_diff = filtered_orientation.get_time_delta()

                dwell_time_map = filtered_orientation.get_dwell_map(response = self.full_detector_response.filename.resolve(),
                                                                    dts = time_diff,
                                                                    src_path = pixel_movement,
                                                                    quiet = True)

                self.coordsys_conv_matrix[ipix,i_time] = dwell_time_map.data * dwell_time_map.unit
                # (HealpixMap).data returns the numpy array without its unit.

        self.calc_image_response_projected()

    def save_coordsys_conv_matrix(self, filename = "coordsys_conv_matrix.hdf5"): 
        self.coordsys_conv_matrix.write(filename, overwrite = True)

    def load_coordsys_conv_matrix_from_filepath(self, filepath):

        if not self._check_file_registration():
            print("Please load all files!")
            return 
    
        if not self._check_axis_consistency():
            print("Please the axes of the input files!")
            return 

#        if not self._check_sc_orientation_coverage():
#            print("Please the axes of the input files!")
#            return 

        print("... (DataLoader) loading a coordinate conversion matrix...")

        self.coordsys_conv_matrix = Histogram.open(filepath)

        if not self.coordsys_conv_matrix.is_sparse:
            self.coordsys_conv_matrix = self.coordsys_conv_matrix.to_sparse()

        print(f"... checking the axes of the coordinate conversion matrix ...")

        if self.coordsys_conv_matrix.unit == u.s:
            print(f"    --> pass (unit)")
        else:
            print(f"Warning: the unit is wrong {self.coordsys_conv_matrix.unit}")
            return False

        axis_model_map = HealpixAxis(nside = self.full_detector_response.axes["NuLambda"].nside, 
                                     coordsys = "galactic", label = "lb")

        if self.coordsys_conv_matrix.axes['lb'] == axis_model_map:
            print(f"    --> pass (axis lb)")
        else:
            print(f"Warning: the axis of lb is inconsistent")
            return False

        if self.coordsys_conv_matrix.axes['Time'] == self.event_dense.axes["Time"]:
            print(f"    --> pass (axis Time)")
        else:
            print(f"Warning: the axis of Time is inconsistent")
            return False

        if self.coordsys_conv_matrix.axes['NuLambda'] == self.full_detector_response.axes['NuLambda']:
            print(f"    --> pass (axis NuLambda)")
        else:
            print(f"Warning: the axis of NuLambda is inconsistent")
            return False

        self.calc_image_response_projected()
    '''

    def calc_image_response_projected(self):
        """
        Calculate image_response_dense_projected, which is an intermidiate matrix used in RL algorithm.
        """

        print("... (DataLoader) calculating a projected image response ...")

        self.image_response_dense_projected = Histogram([ self.coordsys_conv_matrix.axes["lb"], self.full_detector_response.axes["Ei"] ],
                                                        unit = self.full_detector_response.unit * self.coordsys_conv_matrix.unit)

        if self.response_on_memory:

            self.image_response_dense_projected[:] = np.tensordot( np.sum(self.coordsys_conv_matrix, axis = (0)), 
                                                                   np.sum(self.image_response_dense, axis = (2,3,4)),
                                                                   axes = ([1], [0]) ) * self.full_detector_response.unit * self.coordsys_conv_matrix.unit
            # [Time/ScAtt, lb, NuLambda] -> [lb, NuLambda]
            # [NuLambda, Ei, Em, Phi, PsiChi] -> [NuLambda, Ei]
            # [lb, NuLambda] x [NuLambda, Ei] -> [lb, Ei]

        else:
            npix = self.full_detector_response.axes["NuLambda"].npix 

            for ipix in tqdm(range(npix)):
                if self.is_miniDC2_format:
                    full_detector_response_projected_Ei = np.sum(self.full_detector_response[ipix].to_dense(), axis = (1,2,3,4,5)) #Ei
                    # when np.sum is applied to a dense histogram, the unit is restored. when it is a sparse histogram, the unit is not restored. 
                else:
                    full_detector_response_projected_Ei = np.sum(self.full_detector_response[ipix].to_dense(), axis = (1,2,3)) #Ei
    
                coordsys_conv_matrix_projected_lb = np.sum(self.coordsys_conv_matrix[:,:,ipix], axis = (0)).todense() * self.coordsys_conv_matrix.unit #lb
    
                self.image_response_dense_projected += np.outer(coordsys_conv_matrix_projected_lb, full_detector_response_projected_Ei)
