from histpy import Histogram, Axes, Axis

import astropy.units as u

from .quantity_histogram import QuantityHistogram

class PointSourceResponse(QuantityHistogram):

    _enforce_unit_type = u.cm*u.cm*u.s
    
    @property
    def photon_energy_axis(self):
        return self.axes['Ei']
        
    def get_expectation(self, spectrum):

        eaxis = self.photon_energy_axis
        
        flux = Quantity([spectrum.integral(lo_lim, hi_lim)
                         for lo_lim,hi_lim
                         in zip(eaxis.lower_bounds, eaxis.upper_bounds)])

        flux = self.expand_dims(flux.value, 'Ei') * flux.unit

        expectation = self * flux
        
        return expectation

    
