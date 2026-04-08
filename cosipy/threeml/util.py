from typing import Union
import numpy as np
import astropy.units as u

from astromodels.core.polarization import Polarization, LinearPolarization, StokesPolarization

def to_linear_polarization(polarization: Union[Polarization, None]):
    # FIXME: the logic of this code block should be moved to 3ML.
    #   We want to see if the source is polarized, and if so, confirm
    #   transform to linear polarization.
    #   https://github.com/threeML/astromodels/blob/master/astromodels/core/polarization.py

    if polarization is None:
        return LinearPolarization(0,0)

    if isinstance(polarization, LinearPolarization):
        return polarization

    if type(polarization) == Polarization:
        # FIXME: Polarization is the base class, but a 3ML source
        #   with no polarization default to the base class.
        #   The base class shouldn't be able to be instantiated,
        #   and we should have a NullPolarization subclass or None

        if not hasattr(polarization, '_polarization_type') or polarization._polarization_type != 'linear' or (hasattr(polarization, 'degree') and polarization.degree != 0):
            raise RuntimeError("FIXME: new 3ML likely broke some assumptions of this code.")

        return LinearPolarization(0,0)

    elif isinstance(polarization, StokesPolarization):

        Q = polarization.Q.value.k.value
        U = polarization.U.value.k.value

        pa = np.degrees(.5 * np.arctan2(U, Q))
        pa = pa % 360
        pa = np.where(pa > 180, 360 - pa, pa)
        pd = np.sqrt(Q**2 + U**2) * 100.

        return LinearPolarization(pd, pa)

    else:

        if isinstance(polarization, Polarization):
            raise TypeError(f"Fix me. I don't know how to handle this polarization type")
        else:
            raise TypeError(f"Polarization must be a Polarization subclass or None")