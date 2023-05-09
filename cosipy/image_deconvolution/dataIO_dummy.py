import numpy as np
from tqdm.autonotebook import tqdm
import astropy.units as u
from astropy.coordinates import SkyCoord
import healpy as hp

from histpy import Histogram, Axes
from mhealpy import HealpixMap

from cosipy.response import FullDetectorResponse
from scoords import SpacecraftFrame, Attitude

class dataIO(object):
    def __init__(self):
        pass

    def __init__(self, event_filepath = None, bkg_filepath = None, response_filepath = None, sc_orientation = None):
        if event_filepath and bkg_filepath and response_filepath and sc_orientation:
            self.set_response_filepath(response_filepath)
            self.set_event_filepath(event_filepath)
            self.set_bkg_filepath(bkg_filepath)
            self.set_sc_orientation(sc_orientation)

            self.load_data()

    def set_response_filepath(self, filepath):
        self._response_filepath = filepath

    def set_event_filepath(self, filepath):
        self._event_filepath = filepath

    def set_bkg_filepath(self, filepath):
        self._bkg_filepath = filepath

    def set_sc_orientation(self, sc_orientation):
        self._sc_orientation = sc_orientation 

    def load_FullDetectorResponse(self):
        self.response = FullDetectorResponse.open(self._response_filepath) 

        self._axes_cds = Axes([self.response.axes["Em"], \
                               self.response.axes["Phi"], \
                               self.response.axes["PsiChi"]])

        self._axes_response = Axes([self.response.axes["NuLambda"], \
                                    self.response.axes["Ei"], \
                                    self.response.axes["Em"], \
                                    self.response.axes["Phi"], \
                                    self.response.axes["PsiChi"]])

    def load_event(self):
        event = Histogram.open(self._event_filepath)
        self.event_dense = Histogram(self._axes_cds, unit = event.unit, contents = np.array(event.to_dense()))
        self.event = self.event_dense.to_sparse()

    def load_bkg(self):
        bkg = Histogram.open(self._bkg_filepath)
        self.bkg_dense = Histogram(self._axes_cds, unit = bkg.unit, contents = np.array(bkg.to_dense()))
        self.bkg = self.bkg_dense.to_sparse()

    def load_data(self):
        print("... (DataIOdummy) loading FullDetectorResponse ...")
        self.load_FullDetectorResponse()

        print("... (DataIOdummy) loading event ...")
        self.load_event()

        print("... (DataIOdummy) loading background ...")
        self.load_bkg()

        print("... (DataIOdummy) calculating dwell time maps at each sky location ...")
        self.image_response_mul_time_dense = Histogram(self._axes_response, 
                                                       unit = self.response.unit * u.s, sparse = False)

        nside = self.response.axes["NuLambda"].nside # it needs to be the same as nside of the skymodel. Need to a functionality to check it.
        npix = self.response.axes["NuLambda"].npix # it needs to be the same as npix of the skymodel. Need to a functionality to check it.
        for ipix in tqdm(range(npix)):
            theta, phi = hp.pix2ang(nside, ipix)
            ra, dec = phi, np.pi/2 - theta

            coord = SkyCoord(ra = ra * u.rad, dec = dec * u.rad, 
                             frame = 'icrs', attitude = Attitude.identity())

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
