from histpy import Histogram, Axes, Axis

import astropy.units as u

class DetectorResponse(QuantityHistogram):

    _unit_base = u.cm*u.cm
    
    def get_spectral_response(self):

        spec = self.project(['Ei','Em'])
        
        return SpectralResponse(spec.axes,
                                contents = spec.full_contents,
                                unit = spec.unit)
        
