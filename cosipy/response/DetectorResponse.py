from histpy import Histogram, Axes, Axis

import astropy.units as u

from .quantity_histogram import QuantityHistogram

class DetectorResponse(QuantityHistogram):

    def get_spectral_response(self):

        spec = self.project(['Ei','Em'])
        
        return SpectralResponse(spec.axes,
                                contents = spec.full_contents,
                                unit = spec.unit)
        
