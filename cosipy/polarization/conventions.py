import numpy as np
from astropy.coordinates import SkyCoord, Angle
import astropy.units as u
import inspect
from scoords import Attitude, SpacecraftFrame

# Base class for polarization conventions
class PolarizationConvention:

    def __init__(self):
        """
        Base class for polarization conventions
        """

    _registered_conventions = {}
        
    @classmethod
    def register(cls, name):

        name = name.lower()
        
        def _register(convention_class):
            cls._registered_conventions[name] = convention_class
            return convention_class 
        return _register

    @classmethod
    def get_convention(cls, name, *args, **kwargs):

        if inspect.isclass(name):
            if issubclass(name, PolarizationConvention):
                return name(*args, **kwargs)
            else:
                raise TypeError("Class must be subclass of PolarizationConvention")

        if isinstance(name, PolarizationConvention):
            return name

        if not isinstance(name, str):
            raise TypeError("Input must be str, or PolarizationConvention subclass or object")
        
        name = name.lower()

        try:
            return cls._registered_conventions[name](*args, **kwargs)
        except KeyError as e:
            raise Exception(f"No polarization convention by name '{name}'") from e

    @property
    def frame(self):
        """
        Astropy coordinate frame
        """
        return None
        
    def transform(self, pa_1, convention2, source_direction: SkyCoord):
        
        # Ensure pa_1 is an Angle object
        pa_1 = Angle(pa_1)

        # Get the projection vectors for the source direction in the current convention
        (px1, py1) = self.get_basis(source_direction)

        # Calculate the cosine and sine of the polarization angle
        cos_pa_1 = np.cos(pa_1.radian)
        sin_pa_1 = np.sin(pa_1.radian)

        # Calculate the polarization vector in the current convention
        pol_vec = px1 * cos_pa_1 + py1 * sin_pa_1

        # Get the projection vectors for the source direction in the new convention
        (px2, py2) = convention2.get_basis(source_direction)

        # Compute the dot products for the transformation
        a = np.sum(pol_vec * px2, axis=0)
        b = np.sum(pol_vec * py2, axis=0)

        # Calculate the new polarization angle in the new convention
        pa_2 = Angle(np.arctan2(b, a), unit=u.rad)
        return pa_2

    def get_basis(self, source_direction: SkyCoord):
        """
        Get the px,py unit vectors that define the polarization plane on 
        this convention. Polarization angle increments from px to py.

        Parameters
        ----------
        source_direction : SkyCoord
            The direction of the source
        """


# Orthographic projection convention
class OrthographicConvention(PolarizationConvention):

    def __init__(self,
                 ref_vector: SkyCoord = None,
                 clockwise: bool = False):
        """
        The local polarization x-axis points towards an arbitrary reference vector, 
        and the polarization angle increasing counter-clockwise when looking 
        at the source.
        
        Parameters
        ----------
        ref_vector : SkyCoord
            Set the reference vector, defaulting to celestial north if not provided 
            (IAU convention)
        clockwise : bool
            Direction of increasing PA, when looking at the source. Default is false 
            --i.e. counter-clockwise when looking outwards.
            
        """
        if ref_vector is None:
            self.ref_vector = SkyCoord(ra=0 * u.deg, dec=90 * u.deg, frame="icrs")
        else:
            self.ref_vector = ref_vector

        self._sign = 1 if clockwise else -1

    @property
    def frame(self):
        return self.ref_vector.frame
        
    def get_basis(self, source_direction: SkyCoord):
        # Extract Cartesian coordinates for the source direction.
        pz = self._sign * source_direction.transform_to(self.frame).cartesian.xyz

        # Broadcast reference vector
        ref = np.expand_dims(self.ref_vector.cartesian.xyz,
                             axis = tuple(np.arange(1,pz.ndim, dtype = int)))

        # Get py. Normalize because pz and ref dot not make 90deg angle
        py = np.cross(pz, ref, axisa = 0, axisb = 0, axisc = 0)
        py /= np.linalg.norm(py, axis = 0, keepdims = True)

        # Get px
        px = np.cross(py, pz, axisa = 0, axisb = 0, axisc = 0)
        
        # To SkyCoord
        px = SkyCoord(*px, representation_type='cartesian', frame = self.frame)
        py = SkyCoord(*py, representation_type='cartesian', frame = self.frame)
        
        return px, py

# https://lambda.gsfc.nasa.gov/product/about/pol_convention.html
# https://www.iau.org/static/resolutions/IAU1973_French.pdf
# The following resolution was adopted by Commissions 25 and 40: 'RESOLVED, that the frame of reference for the Stokes parameters is that of Right Ascension and Declination with the position angle ofelectric-vector maximum, e, starting from North and increasing through East.
@PolarizationConvention.register("IAU")
class IAUPolarizationConvention(OrthographicConvention):

    def __init__(self):
        super().__init__(ref_vector = SkyCoord(ra=0 * u.deg, dec=90 * u.deg,
                                               frame="icrs"),
                         clockwise = False)
    
    
# Stereographic projection convention
class StereographicConvention(PolarizationConvention):

    def __init__(self,
                 clockwise: bool = False,
                 attitude: Attitude = None):
        """
        Basis vector follow the steregraphic projection lines. Meant to describe
        polarization in spacecraft coordinate by minimizing the number of undefined location withing the field of view.
        
        Near the boresight --i.e. on axis, center of the FoV, north pole-- it is
        similar to 
        ``OrthographicConvention(ref_vector = SkyCoord(lon = 0*u.deg, lat = 0*u.deg, frame = SpacecraftFrame())``
        however, it has a single undefined point on the opposite end --i.e. south pole,
        back of the detector---
        
        
        Parameters
        ----------
        clockwise : bool
            Direction of increasing PA, when looking at the source. Default is false 
            --i.e. counter-clockwise when looking outwards.
        attitude : Attitude
            Spacecraft orientation
        """

        self._attitude = attitude
        
        self._sign = 1 if clockwise else -1

    @property 
    def frame(self):
        return SpacecraftFrame(attitude = self._attitude)
        
    def get_basis(self, source_direction: SkyCoord):
        # Extract Cartesian coordinates for the source direction
        x, y, z = source_direction.cartesian.xyz

        # Calculate the projection of the reference vector in stereographic coordinates
        px_x = 1 - (x**2 - y**2) / (z + 1) ** 2
        px_y = -2 * x * y / (z + 1) ** 2
        px_z = -2 * x / (z + 1)

        # Combine the components into the projection vector px
        px = np.array([px_x, px_y, px_z])

        # Normalize the projection vector
        norm = np.linalg.norm(px, axis=0)
        px /= norm

        # Calculate the perpendicular vector py using the cross product
        py = self._sign*np.cross([x, y, z], px, axis=0)

        # To SkyCoord
        px = SkyCoord(*px, representation_type='cartesian', frame = self.frame)
        py = SkyCoord(*py, representation_type='cartesian', frame = self.frame)
        
        return px, py
