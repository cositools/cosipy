from astropy.coordinates import TimeAttribute, Attribute, EarthLocationAttribute
import astropy.units as u

import numpy as np

from abc import ABC, abstractmethod

class Attitude(ABC):
    
    @property
    @abstractmethod
    def rot_matrix(self):
        ...
        
    @property
    @abstractmethod
    def shape(self):
        ...        

class Quaternion(Attitude):
    """
    Holds one or more quaternions representing a rotation.
    
    The convention for a rotation around an unit vector :math:`(X,Y,Z)` by an 
    angle :math:`\theta` is as follows:
        
    .. math::
        q = (XS, YS, ZS, C)
        
    where :math:`C = \cos(\theta/2)` and :math:`S = \sin(\theta/2)`
    
    Args:
        quat (array): Shaped (4,N). Will be normalized
    """
    
    def __init__(self, quat):
        
        if np.isscalar(quat) or np.shape(quat)[0] != 4:
            raise ValueError("Wrong quaternion shape")

        # Standard format
        self._quat = np.array(quat, dtype = float)
        
        #Normalize
        self._quat /= np.sqrt(np.sum(self._quat*self._quat, axis = 0))

    @classmethod
    def from_v_theta(cls, v, theta):

        v = np.array(v, dtype = float)
        v /= np.sqrt(np.sum(v*v, axis=0))

        half_theta = theta.to_value(u.rad)/2
        
        s = np.sin(half_theta)
        c = np.cos(half_theta)

        return cls(np.append(s*v, c))
                
    @property
    def rot_matrix(self):
        """
        Corresponding rotation matrix. Shaped (3,3,N)
        """

        a = self._quat[3]
        b = self._quat[0]
        c = self._quat[1]
        d = self._quat[2]

        a2 = a*a
        b2 = b*b
        c2 = c*c
        d2 = d*d

        ab = a*b
        ac = a*c
        ad = a*d
        
        bc = b*c
        bd = b*d

        cd = c*d
        
        return np.array([[a2+b2-c2-d2,    2*(bc-ad),    2*(bd+ac)],
                         [  2*(bc+ad),  a2-b2+c2-d2,    2*(cd-ab)],
                         [  2*(bc-ac),    2*(cd+ab),  a2-b2-c2+d2]])

    @property
    def shape(self):
        return self.quat.shape[1:]

    def __array__(self):
        return self._quat

    def __str__(self):
        return str(self._quat)

    def __repr__(self):
        return repr(self._quat)
    
        
class AttitudeAttribute(Attribute):
    """
    Interface for attitude (e.g. a quaternion) with astropy's custom frame
    """
    
    def convert_input(self, value):

        if value is None:
            return None, False
        
        converted = False
        
        if not isinstance(value, Attitude):
            # Currently only a quaternion is supported
            
            value = Quaternion(value)
            
            converted = True
                        
        return value,converted
