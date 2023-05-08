import numpy as np
from tqdm.autonotebook import tqdm
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord
import healpy as hp

from histpy import Histogram, Axes, HealpixAxis
from mhealpy import HealpixMap

from cosipy.response import FullDetectorResponse
from cosipy.coordinates import SpacecraftFrame, Attitude
from cosipy.coordinates.orientation import Orientation_file
from cosipy.spacecraftpositionattitude import SpacecraftPositionAttitude
from cosipy.data_io import BinnedData

class DataLoader(object):

    def __init__(self, event_hdf5_filepath = None, event_yaml_filepath = None, 
                       bkg_hdf5_filepath = None, bkg_yaml_filepath = None, 
                       rsp_filepath = None, sc_orientation_filepath = None):

        self.event_dense, self.event_sparse = None, None
        self.bkg_dense, self.bkg_sparse = None, None
        self.full_detector_response = None
        self.orientation = None

        if event_hdf5_filepath and event_yaml_filepath:
            self.set_event(event_hdf5_filepath, event_yaml_filepath)

        if bkg_hdf5_filepath and bkg_yaml_filepath:
            self.set_bkg(bkg_hdf5_filepath, bkg_yaml_filepath)
        
        if rsp_filepath:
            self.set_rsp(rsp_filepath)

        if sc_orientation_filepath:
            self.set_sc_orientation(sc_orientation_filepath)

    def set_event(self, hdf5_filepath, yaml_filepath):

        self._event_hdf5_filepath = hdf5_filepath
        self._event_yaml_filepath = yaml_filepath

        print("... (DataLoader) loading event ...")

        event = BinnedData(self._event_yaml_filepath)
        event.load_binned_data_from_hdf5(self._event_hdf5_filepath)

        self.event_dense = event.binned_data.to_dense()
        self.event_sparse = event.binned_data.to_sparse()

        print("... Done ...")

    def set_bkg(self, hdf5_filepath, yaml_filepath):

        self._bkg_hdf5_filepath = hdf5_filepath
        self._bkg_yaml_filepath = yaml_filepath

        print("... (DataLoader) loading background ...")

        bkg = BinnedData(self._bkg_yaml_filepath)
        bkg.load_binned_data_from_hdf5(self._bkg_hdf5_filepath)

        self.bkg_dense = bkg.binned_data.to_dense()
        self.bkg_sparse = bkg.binned_data.to_sparse()

        print("... Done ...")

    def set_rsp(self, filepath):

        self._rsp_filepath = filepath

        print("... (DataLoader) loading full detector response ...")

        self.full_detector_response = FullDetectorResponse.open(self._rsp_filepath) 

        print("... Done ...")

    def set_sc_orientation(self, filepath):

        self._sc_orientation_filepath = filepath
        
        print("... (DataLoader) loading orientation file ...")

        self.orientation = Orientation_file(self._sc_orientation_filepath)

        print("... Done ...")

    def show_registered_files(self):
        print(f"event       : {self._event_hdf5_filepath}, {self._event_yaml_filepath}")
        print(f"background  : {self._bkg_hdf5_filepath}, {self._bkg_yaml_filepath}")
        print(f"response    : {self._rsp_filepath}")
        print(f"orientation : {self._sc_orientation_filepath}")

    def _check_file_registration(self):
        if self.event_dense and self.event_sparse \
        and self.bkg_dense and self.bkg_sparse \
        and self.full_detector_response and self.orientation:
            return True
        
        return False

    def _check_axis_consistency(self):
        # check the axes of the event/background files
        axis_name = ['Time', 'Em', 'Phi', 'PsiChi']

        for name in axis_name:
            if not self.event_dense.axes[name] == self.bkg_dense.axes[name]:
                print(f"Warning: the axis {name} is not consistent between the event and background!")
                return False

        # check the axes of the event/response files
        axis_name = ['Em', 'Phi', 'PsiChi']
        
        for name in axis_name:
            if not self.event_dense.axes[name] == self.full_detector_response.axes[name]:
                print(f"Warning: the axis {name} is not consistent with the event and response!")
                return False

        return True

    def _modify_axes(self): # this is a tentetive function
        self.event_dense.axes['Time']._unit = u.s

        axes_cds = Axes([self.event_dense.axes["Time"], \
                         self.full_detector_response.axes["Em"], \
                         self.full_detector_response.axes["Phi"], \
                         self.full_detector_response.axes["PsiChi"]])
        
        self.event_dense = Histogram(axes_cds, unit = self.event_dense.unit, contents = self.event_dense.contents)
        self.event_sparse = self.event_dense.to_sparse()

        self.bkg_dense = Histogram(axes_cds, unit = self.bkg_dense.unit, contents = self.bkg_dense.contents)
        self.bkg_sparse = self.bkg_dense.to_sparse()

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

    def calc_image_response(self): 

        self._modify_axes() # Here I just fix the current discrepancy between response and event/bkg files. I remove this line soon.

        if not self._check_file_registration():
            print("Please load all files!")
            return 
    
        if not self._check_axis_consistency():
            print("Please the axes of the input files!")
            return 

        if not self._check_sc_orientation_coverage():
            print("Please the axes of the input files!")
            return 

        print("... (DataLoader) calculating a response at each sky location / time bin ...")
        
        # make an empty histogram for the response calculation
        axis_model_map = HealpixAxis(nside = self.full_detector_response.axes["NuLambda"].nside, 
                                     coordsys = "galactic", label = "lb")

        axes_image_response = [axis_model_map, self.full_detector_response.axes["Ei"],
                               self.event_dense.axes["Time"], self.full_detector_response.axes["Em"], 
                               self.full_detector_response.axes["Phi"], self.full_detector_response.axes["PsiChi"]]

        self.image_response = Histogram(axes_image_response, 
                                        unit = self.response.unit * u.s, sparse = False)

        # calculate a dwell time map at each time bin and sky location

        nside = self.response.axes["lb"].nside
        npix = self.response.axes["lb"].npix 
        # they need to be the same as npix of the skymodel. Need to a functionality to check it in the future.

        sc_attitude = data_loader.orientation.get_attitude()
        sc_time = data_loader.orientation.get_time()
        
        for ipix in tqdm(range(npix)):
            theta, phi = hp.pix2ang(nside, ipix)
            ra, dec = phi, np.pi/2 - theta

            pixel_coord = SkyCoord(ra = ra * u.rad, dec = dec * u.rad, frame = 'galactic')
            obj = SpacecraftPositionAttitude.SourceSpacecraft(f"pixel_{ipix}", pixel_coord)

            for init_time, end_time in self.image_response["Time"].bounds:
                init_time = Time(init_time, format = 'unix')
                end_time = Time(end_time, format = 'unix')
    
                _ = (init_time <= sc_time) & (sc_time <= end_time)
    
#                filtered_sc_attitude = sc_attitude[_]
#                filtered_sc_time = sc_time[_] 
                
                x,y,z = self.orientation.get_attitude().as_axes()
                src_movement = obj.sc_frame(x_pointings = x[1:], z_pointings = z[1:])

                filtered_sc_time_delta = np.diff(filtered_sc_time)









            dwell_time_map = self._get_dwell_time_map(coord)

            response_single_pixel = self.response.get_point_source_response(dwell_time_map).project(["Ei", 'Em', 'Phi' ,'PsiChi']).todense()
            self.image_response_mul_time_dense[ipix] = response_single_pixel.contents

        print("... (DataIOdummy) calculating projected response ...")
        self.image_response_mul_time_dense_projected = self.image_response_mul_time_dense.project("NuLambda", "Ei")

        print("... (DataIOdummy) dense to sparse ...")
        self.image_response_mul_time = self.image_response_mul_time_dense.to_sparse()

        print("... (DataIOdummy) calculating projected response (sparse) ...")
        self.image_response_mul_time_projected = self.image_response_mul_time.project("NuLambda", "Ei")

    def _get_dwell_time_map(self, coord): #copied from COSILike.py
        """
        This will be eventually be provided by another module
        """
        
        # The dwell time map has the same pixelation (base) as the detector response.
        # We start with an empty map
        dwell_time_map = HealpixMap(base = self.response, 
                                    unit = u.s, 
                                    coordsys = SpacecraftFrame())

        # Get timestamps and attitude values
        timestamps, attitudes = zip(*self._sc_orientation)
            
        for attitude,duration in zip(attitudes[:-1], np.diff(timestamps)):

            local_coord = coord.transform_to(SpacecraftFrame(attitude = attitude))

            # Here we add duration in between timestamps using interpolations
            pixels, weights = dwell_time_map.get_interp_weights(local_coord)

            for p,w in zip(pixels, weights):
                dwell_time_map[p] += w*duration.to(u.s)
                
        return dwell_time_map
