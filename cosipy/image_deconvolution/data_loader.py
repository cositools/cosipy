import numpy as np
from tqdm.autonotebook import tqdm
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord
import healpy as hp

from histpy import Histogram, Axes, HealpixAxis
from mhealpy import HealpixMap

from cosipy.response import FullDetectorResponse
from scoords import SpacecraftFrame, Attitude
from cosipy.SpacecraftFile import SpacecraftFile
from cosipy.data_io import BinnedData

class DataLoader(object):

    def __init__(self):
        self.event_dense, self.event_sparse = None, None
        self.bkg_dense, self.bkg_sparse = None, None
        self.full_detector_response = None
        self.orientation = None
        self.coordsys_conv_matrix = None

        self.response_on_memory = False

    @classmethod
    def load(cls, event_binned_data, bkg_binned_data, rsp, sc_orientation):

        new = cls()

        new.event_dense = event_binned_data.to_dense()
        new.event_sparse = event_binned_data.to_sparse()

        new.bkg_dense = bkg_binned_data.to_dense()
        new.bkg_sparse = bkg_binned_data.to_sparse()

        new.full_detector_response = rsp

        new.orientation = sc_orientation

        return new
    
    @classmethod
    def load_from_filepath(cls, event_hdf5_filepath = None, event_yaml_filepath = None, 
                           bkg_hdf5_filepath = None, bkg_yaml_filepath = None, 
                           rsp_filepath = None, sc_orientation_filepath = None):

        new = cls()

        new.set_event_from_filepath(event_hdf5_filepath, event_yaml_filepath)

        new.set_bkg_from_filepath(bkg_hdf5_filepath, bkg_yaml_filepath)
        
        new.set_rsp_from_filepath(rsp_filepath)

        new.set_sc_orientation_from_filepath(sc_orientation_filepath)

        return new

    def set_event_from_filepath(self, hdf5_filepath, yaml_filepath):

        self._event_hdf5_filepath = hdf5_filepath
        self._event_yaml_filepath = yaml_filepath

        print(f'... loading event from {hdf5_filepath} and {yaml_filepath}')

        event = BinnedData(self._event_yaml_filepath)
        event.load_binned_data_from_hdf5(self._event_hdf5_filepath)

        self.event_dense = event.binned_data.to_dense()
        self.event_sparse = event.binned_data.to_sparse()

        print("... Done ...")

    def set_bkg_from_filepath(self, hdf5_filepath, yaml_filepath):

        self._bkg_hdf5_filepath = hdf5_filepath
        self._bkg_yaml_filepath = yaml_filepath

        print(f'... loading background from {hdf5_filepath} and {yaml_filepath}')

        bkg = BinnedData(self._bkg_yaml_filepath)
        bkg.load_binned_data_from_hdf5(self._bkg_hdf5_filepath)

        self.bkg_dense = bkg.binned_data.to_dense()
        self.bkg_sparse = bkg.binned_data.to_sparse()

        print("... Done ...")

    def set_rsp_from_filepath(self, filepath):

        self._rsp_filepath = filepath

        print(f'... loading full detector response from {filepath}')

        self.full_detector_response = FullDetectorResponse.open(self._rsp_filepath) 

        print("... Done ...")

    def set_sc_orientation_from_filepath(self, filepath):

        self._sc_orientation_filepath = filepath
        
        print(f'... loading orientation from {filepath}')

        self.orientation = SpacecraftFile.parse_from_file(self._sc_orientation_filepath)

        print("... Done ...")

    def _check_file_registration(self):

        print(f"... checking the file registration ...")

        if self.event_dense and self.event_sparse \
        and self.bkg_dense and self.bkg_sparse \
        and self.full_detector_response and self.orientation:

            print(f"    --> pass")
            return True
        
        return False

    def _check_axis_consistency(self):
        
        print(f"... checking the axis consistency ...")

        # check the axes of the event/background files
        axis_name = ['Time', 'Em', 'Phi', 'PsiChi'] # 'Time' should be changed if one uses the scat binning.

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

    def _modify_axes(self): # this is a tentetive function

        print(f"Note that this function is tentetive. It should be removed in the future!")
        print(f"Please run this function only once!")

        axis_name = ['Time', 'Em', 'Phi', 'PsiChi'] # 'Time' should be changed if one uses the scat binning.

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
            
            if type(response_edges) == u.quantity.Quantity:
                response_edges = response_edges.value

            if np.all(event_edges == response_edges):
                print(f"    --> pass (edges)") 
            else:
                print(f"Warning: the edges of the axis {name} are not consistent between the event and background!")
                print(f"        event      : {event_edges}")
                print(f"        background : {response_edges}")
                return False

        axes_cds = Axes([self.event_dense.axes["Time"], \
                         self.full_detector_response.axes["Em"], \
                         self.full_detector_response.axes["Phi"], \
                         self.full_detector_response.axes["PsiChi"]])
        
        self.event_dense = Histogram(axes_cds, unit = self.event_dense.unit, contents = self.event_dense.contents)
        self.event_sparse = self.event_dense.to_sparse()

        self.bkg_dense = Histogram(axes_cds, unit = self.bkg_dense.unit, contents = self.bkg_dense.contents)
        self.bkg_sparse = self.bkg_dense.to_sparse()

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

    def calc_image_response(self): 

        if not self._check_file_registration():
            print("Please load all files!")
            return 
    
        if not self._check_axis_consistency():
            print("Please the axes of the input files!")
            return 

#        if not self._check_sc_orientation_coverage():
#            print("Please the axes of the input files!")
#            return 

        print("... (DataLoader) calculating a flat source response at each sky location and each time bin ...")
        
        # make an empty histogram for the response calculation
        axis_model_map = HealpixAxis(nside = self.full_detector_response.axes["NuLambda"].nside, 
                                     coordsys = "galactic", label = "lb")

        axes_image_response = [axis_model_map, self.full_detector_response.axes["Ei"],
                               self.event_dense.axes["Time"], self.full_detector_response.axes["Em"], 
                               self.full_detector_response.axes["Phi"], self.full_detector_response.axes["PsiChi"]]

        self.image_response_dense = Histogram(axes_image_response, 
                                              unit = self.full_detector_response.unit * u.s, sparse = False)

        # calculate a dwell time map at each time bin and sky location

        nside = self.full_detector_response.axes["NuLambda"].nside
        npix = self.full_detector_response.axes["NuLambda"].npix 
        # they need to be the same as npix of the skymodel. Need to a functionality to check it in the future.

        for ipix in tqdm(range(npix)):
            theta, phi = hp.pix2ang(nside, ipix)
            l, b = phi, np.pi/2 - theta

            pixel_coord = SkyCoord(l, b, unit = u.rad, frame = 'galactic')
            pixel_obj = SpacecraftPositionAttitude.SourceSpacecraft(f"pixel_{ipix}", pixel_coord)

            for i_time, [init_time, end_time] in enumerate(self.image_response_dense.axes["Time"].bounds):
                init_time = Time(init_time, format = 'unix')
                end_time = Time(end_time, format = 'unix')
    
                filtered_orientation = self.orientation.source_interval(init_time, end_time)
                x,y,z = filtered_orientation.get_attitude().as_axes()
                pixel_movement = pixel_obj.sc_frame(x_pointings = x, z_pointings = z)

                time_diff = filtered_orientation.get_time_delta()
                time_diff = Time(0.5*(np.insert(time_diff.value, 0, 0) + np.append(time_diff.value, 0)), format = 'unix')

                dwell_time_map = pixel_obj.get_dwell_map(response = self.full_detector_response.filename,
                                                         dts = time_diff,
                                                         src_path = pixel_movement)

                point_source_rsp = self.full_detector_response.get_point_source_response(dwell_time_map).project(['Ei', 'Em', 'Phi', 'PsiChi']).todense()

                for i_Ei in range(self.image_response_dense.axes["Ei"].nbins):
                    self.image_response_dense[ipix, i_Ei:i_Ei+1, i_time:i_time+1] = point_source_rsp[i_Ei]

        print("... (DataLoader) calculating the projected response ...")
        self.image_response_dense_projected = self.image_response_dense.project("lb", "Ei")

        print("... (DataLoader) dense to sparse ...")
        self.image_response_sparse = self.image_response_dense.to_sparse()

        print("... (DataLoader) calculating the projected response (sparse) ...")
        self.image_response_sparse_projected = self.image_response_sparse.project("lb", "Ei")

    def load_full_detector_response_on_memory(self):

        axes_image_response = [self.full_detector_response.axes["NuLambda"], self.full_detector_response.axes["Ei"],
                               self.full_detector_response.axes["Em"], self.full_detector_response.axes["Phi"], self.full_detector_response.axes["PsiChi"]]

        self.image_response_dense = Histogram(axes_image_response, unit = self.full_detector_response.unit)

        nside = self.full_detector_response.axes["NuLambda"].nside
        npix = self.full_detector_response.axes["NuLambda"].npix 

        for ipix in tqdm(range(npix)):
            self.image_response_dense[ipix] = np.sum(self.full_detector_response[ipix].to_dense(), axis = (4,5)) #Ei, Em, Phi, ChiPsi

        self.response_on_memory = True

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

                dwell_time_map = filtered_orientation.get_dwell_map(response = self.full_detector_response.filename,
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

    def calc_image_response_projected(self):
        # calculate the image_response_dense_projected

        print("... (DataLoader) calculating a projected image response ...")

        self.image_response_dense_projected = Histogram([ self.coordsys_conv_matrix.axes["lb"], self.full_detector_response.axes["Ei"] ],
                                                        unit = self.full_detector_response.unit * self.coordsys_conv_matrix.unit)

        if self.response_on_memory:

            self.image_response_dense_projected[:] = np.tensordot( np.sum(self.coordsys_conv_matrix, axis = (1)), 
                                                                np.sum(self.image_response_dense, axis = (2,3,4)),
                                                                axes = ([1], [0]) ) * self.full_detector_response.unit * self.coordsys_conv_matrix.unit #lb, Ei

        else:
            npix = self.full_detector_response.axes["NuLambda"].npix 

            for ipix in tqdm(range(npix)):
                full_detector_response_projected_Ei = np.sum(self.full_detector_response[ipix].to_dense(), axis = (1,2,3,4,5)) #Ei
                # when np.sum is applied to a dense histogram, the unit is restored. when it is a sparse histogram, the unit is not restored. 
    
                coordsys_conv_matrix_projected_lb = np.sum(self.coordsys_conv_matrix[:,:,ipix], axis = (1)).todense() * self.coordsys_conv_matrix.unit #lb
    
                self.image_response_dense_projected += np.outer(coordsys_conv_matrix_projected_lb, full_detector_response_projected_Ei)
