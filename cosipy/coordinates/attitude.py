from astropy.coordinates import (TimeAttribute, Attribute, EarthLocationAttribute,
                                 ICRS, CartesianRepresentation, SkyCoord)
import astropy.units as u

import numpy as np

from abc import ABC, abstractmethod

from scipy.spatial.transform import Rotation

class Attitude:

    def __init__(self, rot, frame = None):
        
        self._rot = rot

        if frame is None:
            self._frame = 'icrs'
        else:
            self._frame = frame
    
    @classmethod
    def from_quat(cls, quat, frame = None):

        return cls(Rotation.from_quat(quat), frame)

    @classmethod
    def from_matrix(cls, matrix, frame = None):

        return cls(Rotation.from_matrix(matrix), frame)
        
    @classmethod
    def from_rotvec(cls, rotvec, frame = None):

        return cls(Rotation.from_rotvec(rotvec.to_value(u.rad)), frame)

    def transform_to(self, frame):

        if self.frame == frame:
            return self
        
        # Each row of a rotation matrix is composed of the unit vector along
        # each axis on the new frame. We then convert each of this to the new frame,
        # resulting on a new rotation matrix
        
        old_rot = CartesianRepresentation(x = self._rot.as_matrix().transpose())
        
        new_rot = SkyCoord(old_rot, frame = self.frame).transform_to(frame)

        new_rot = new_rot.represent_as('cartesian').xyz.value.transpose()

        return self.from_matrix(new_rot, frame = frame)

    @property
    def frame(self):
        return self._frame
    
    def as_matrix(self):
        return self._rot.as_matrix()

    def as_rotvet(self):
        return self._rot.as_rotvec()*u.rad

    def as_quat(self):
        return self._rot.as_quat()
    
    @property
    def shape(self):
        return np.asarray(self._rot).shape

    def __getitem__(self, key):
        return self._rot[key]

    def __str__(self):
        return f"<{self._rot.as_matrix()}, frame = {self.frame}>"

    
class AttitudeAttribute(Attribute):
    """
    Interface for attitude (e.g. a quaternion) with astropy's custom frame
    """
    
    def convert_input(self, value):

        if value is None:
            return None, False
        
        if not isinstance(value, Attitude):
            raise ValueError("Attitude is not an instance of Attitude.")
            
        converted = True
                        
        return value,converted
