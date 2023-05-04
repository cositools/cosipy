import numpy as np
from tqdm.autonotebook import tqdm
import astropy.units as u

from histpy import Histogram, Axes
from cosipy.response import FullDetectorResponse

class dataIO(object):
    def __init__(self):
        pass

    def __init__(self, event_filepath = None, bkg_filepath = None, response_filepath = None, duration = None):
        if event_filepath and bkg_filepath and response_filepath and duration:
            self.set_response_filepath(response_filepath)
            self.set_event_filepath(event_filepath)
            self.set_bkg_filepath(bkg_filepath)
            self.set_duration(duration)
            self.load_data()

    def set_duration(self, duration):
        self._duration = duration

    def set_response_filepath(self, filepath):
        self._response_filepath = filepath

    def set_event_filepath(self, filepath):
        self._event_filepath = filepath

    def set_bkg_filepath(self, filepath):
        self._bkg_filepath = filepath

    def load_data(self):
        self.response = FullDetectorResponse.open(self._response_filepath) 

        axes = Axes([self.response.axes["Em"], \
                     self.response.axes["Phi"], \
                     self.response.axes["PsiChi"]])

        event = Histogram.open(self._event_filepath)
        self.event_dense = Histogram(axes, unit = event.unit, contents = np.array(event.to_dense()))
        self.event = self.event_dense.to_sparse()

        bkg = Histogram.open(self._bkg_filepath)
        self.bkg_dense = Histogram(axes, unit = bkg.unit, contents = np.array(bkg.to_dense()))
        self.bkg = self.bkg_dense.to_sparse()
        
        print("... (DataIOdummy) creating response ...")

        axes = Axes([self.response.axes["NuLambda"], \
                     self.response.axes["Ei"], \
                     self.response.axes["Em"], \
                     self.response.axes["Phi"], \
                     self.response.axes["PsiChi"]])

        self.image_response_mul_time_dense = Histogram(axes, unit = self.response.unit * u.s, sparse = False)

        npix = self.response.axes["PsiChi"].npix 
        for ipix in tqdm(range(npix)):
            response_single_pixel = self.response[ipix].project(["Ei", 'Em', 'Phi' ,'PsiChi']).todense()
            self.image_response_mul_time_dense[ipix] = response_single_pixel.contents * self._duration

        print("... (DataIOdummy) calculationg projected response ...")
        self.image_response_mul_time_dense_projected = self.image_response_mul_time_dense.project("NuLambda", "Ei")

        print("... (DataIOdummy) dense to sparse ...")
        self.image_response_mul_time = self.image_response_mul_time_dense.to_sparse()

        print("... (DataIOdummy) calculationg projected response (sparse) ...")
        self.image_response_mul_time_projected = self.image_response_mul_time.project("NuLambda", "Ei")
