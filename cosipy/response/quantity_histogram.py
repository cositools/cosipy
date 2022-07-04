

class QuantityHistogram(Histogram):

    # Enforce a physical dimension
    _unit_base = None
    
    def __init__(self,
                 *args,
                 unit = None,
                 **kwargs):

        super().__init__(*args, **kwargs)

        self.unit = unit

    @property
    def unit(self):
        return self._unit

    @unit.setter
    def unit(self, unit):
        if _unit_base is not None and not _unit_base.is_equivalent(unit):
            raise ValueError(f"{self.unit} is not a valid unit")

        self._unit = u.Unit(unit)

    def _unit_operation(self, other, operation):
        """
        Returns the unit of the result of an operation with a given object.

        Parameters
        ----------
        other : QuantityHistogram, Quantity, UnitBase, float, int, array
            The other operand
        operations : function
            Binary operation function
        """

        if isinstance(other, (QuantityHistogram, Quantity)):
            other_unit = other.unit
        elif isinstance(other, UnitBase):
            other_unit = other
        else:
            other_unit = u.dimensionless_unscaled

        return operation(1*self.unit, 1*other_unit).unit

    
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

        # Match units. self.unit doesn't change
        other_value *= other_unit.to(self.unit).value

        # Operation
        super()._ioperation(other, operation)

        return self

    def _operation(self, other, operation):

        # Separate
        other_value, other_unit = self._to_value_unit(other)

        # Value operation
        new = super()._operation(other_value, operation)

        # Unit operation
        # In this case change the base unit, since there is nothing to enforce (new object)
        new_unit = operation(1*self.unit, 1*other_unit).unit
        new._unit_base = new_unit
        new.unit = new_unit
        
        return new

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

            new.unit = unit

            return new
            
        else:
        
            self *= factor

            self.unit = unit
            
            return self
            
            
