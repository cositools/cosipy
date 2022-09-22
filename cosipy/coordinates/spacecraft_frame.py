from astropy.coordinates import BaseCoordinateFrame, QuantityAttribute, TimeAttribute, EarthLocationAttribute
from astropy.coordinates.representation import SphericalRepresentation
from astropy.coordinates import frame_transform_graph, ICRS, DynamicMatrixTransform

from .attitude import AttitudeAttribute

import numpy as np

from abc import ABC, abstractmethod

class SpacecraftFrame(BaseCoordinateFrame):
    """
    Reference frame attached to the spacecraft.

    Parameters
    ----------
    lon : :py:class:`astropy.units.Quantity`
        Latitude
    lat : :py:class:`astropy.units.Quantity`
        Latitude
    attitude : :py:class:`.Attitude`, optional
        The orientation of the spacecraft with respect to an internatial frame.
    obtime : :py:class:`astropy.time.Time`, optional
        The time at which the observation was taken.
    location : :py:class:`astropy.coordinates.EarthLocation`, optional
        The location of the spacecraft on the Earth.
    """
    
    default_representation = SphericalRepresentation
    
    attitude = AttitudeAttribute(default = None)

    obstime = TimeAttribute(default=None)
    
    location = EarthLocationAttribute(default=None)
    
@frame_transform_graph.transform(DynamicMatrixTransform, SpacecraftFrame, ICRS)
def spacecraft_to_icrs(sc_coord, icrs_frame):
    
    if sc_coord.attitude is None:
        raise RuntimeError("Spacecraft coordinates need attitude to transform to ICRS")
        
    return sc_coord.attitude.transform_to('icrs').as_matrix()

@frame_transform_graph.transform(DynamicMatrixTransform, ICRS, SpacecraftFrame)
def spacecraft_to_icrs(icrs_coord, sc_frame):
    
    if sc_frame.attitude is None:
        raise RuntimeError("Spacecraft coordinates need attitude to transform from ICRS")
        
    return sc_frame.attitude.transform_to('icrs').inv().as_matrix()

