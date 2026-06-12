"""
Utility functions for data interfaces.
"""
import numpy as np
import astropy.units as u


def tensordot_sparse(A, A_unit, B, axes):
    """
    Perform a tensordot operation on A and B. A is sparse
    and so does not carry a unit; rather it must be passed
    as a separate argument. B is a normal Quantity. Return
    a proper Quantity as the result.
    """
    dotprod = np.tensordot(A, B.value, axes=axes)
    return u.Quantity(dotprod, unit=A_unit * B.unit, copy=False)
