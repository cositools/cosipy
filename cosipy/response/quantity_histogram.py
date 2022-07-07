from histpy import Histogram

import astropy.units as u
from astropy.units import Quantity, UnitBase

import numpy as np

class QuantityHistogram(Histogram):

    def __init__(self,
                 *args,
                 unit = None,
                 **kwargs):

        super().__init__(*args, **kwargs)

        self._unit = unit

    @property
    def unit(self):
        return self._unit

    def _to_value_unit(self, other):
        """
        Separate between unit and value for binary operations

        Parameters
        ----------
        other : QuantityHistogram, Quantity, UnitBase, float, int, array
            The other operand

        Returns
        -------
        value : float, int, array
        unit : astropy.units.Unit
        """
        
        if isinstance(other, QuantityHistogram):
            other_unit = other.unit
            other_value = other # It will be handled as a regular Histogram
        elif isinstance(other, Quantity):
            other_unit = other.unit
            other_value = other.value
        elif isinstance(other, UnitBase):
            other_unit = other
            other_value = np.array(1)
        else:
            # float, int, array, list
            other_unit = u.dimensionless_unscaled
            other_value = np.array(other)

        return other_value, other_unit
            
    def _ioperation(self, other, operation):

        # Separate
        other_value, other_unit = self._to_value_unit(other)

        # Value operation
        super()._ioperation(other_value, operation)

        # Unit operation
        self._unit = operation(1*self.unit, 1*other_unit).unit
        
        return self

    def to(self, unit, equivalencies=[], copy=True):
        """
        Return a new QuantityHistogram object with the specified unit.

        Parameters
        ----------
        unit : unit-like
            Unit to convert to.

        equivalencies : list of tuple
            A list of equivalence pairs to try if the units are not
            directly convertible.  

        copy : bool, optional
            If `True` (default), then the value is copied.  Otherwise, a copy
            will only be made if necessary.
        """

        factor = self.unit.to(unit, equivalencies = equivalencies)

        if copy:

            new = self * factor

            new._unit = unit

            return new
            
        else:
        
            self *= factor

            self._unit = unit
            
            return self
            