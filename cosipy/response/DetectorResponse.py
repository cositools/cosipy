from histpy import Histogram, Axes, Axis

import astropy.units as u

from .quantity_histogram import QuantityHistogram
from .SpectralResponse import SpectralResponse

class DetectorResponse(QuantityHistogram):

    def get_spectral_response(self):

        spec = self.project(['Ei','Em'])
        
        return SpectralResponse(spec.axes,
                                contents = spec.full_contents,
                                unit = self.unit)
        
