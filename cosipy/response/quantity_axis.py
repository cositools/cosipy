from histpy import Axis

import numpy as np

import astropy.units as u

class QuantityAxis(Axis):

    def __init__(self,
                 *args,
                 unit = None,
                 **kwargs):

        super().__init__(*args, **kwargs)

        self._unit = u.Unit(unit)

    @property
    def unit(self):
        return self._unit

    
        

