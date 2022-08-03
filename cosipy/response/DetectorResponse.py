from histpy import Histogram, Axes, Axis

import astropy.units as u

from .SpectralResponse import SpectralResponse

class DetectorResponse(Histogram):

    def get_spectral_response(self):

        spec = self.project(['Ei','Em'])
        
        return SpectralResponse(spec.axes,
                                contents = spec.full_contents,
                                unit = self.unit)
        
