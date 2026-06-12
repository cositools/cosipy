from typing import Union, Optional

import numpy as np
from astropy.coordinates import SkyCoord, Angle, BaseCoordinateFrame, frame_transform_graph, ICRS
import astropy.units as u
import inspect
from scoords import Attitude, SpacecraftFrame

# Base class for polarization conventions
class PolarizationConvention:

    _registered_conventions = {}

    registered_name = None # class has no registered name by default

    def __init__(self):
        """
        Base class for polarization conventions
        """

    @classmethod
    def register(cls, name):

        name = name.lower()

        def _register(convention_class):
            cls._registered_conventions[name] = convention_class
            convention_class.registered_name = name

            return convention_class
        return _register

    @classmethod
    def get_convention(cls, name, *args, **kwargs):
        """
        Create an object of a specified PolarizationConvention type
        using the provided *args/**kwargs.

        Parameters
        ----------
        name : either
           - string [name of a registered polarization convention]
           - subclass of PolarizationConvention
        *args, **kwargs
           Arguments used to create new convention object

        """
        if isinstance(name, str):
            name = name.lower()

            try:
                return cls._registered_conventions[name](*args, **kwargs)
            except KeyError as e:
                raise Exception(f"No polarization convention by name '{name}'") from e

        elif inspect.isclass(name) and issubclass(name, PolarizationConvention):
            return name(*args, **kwargs)

        else:
            raise TypeError("Input must be str or subclass of PolarizationConvention")

    @property
    def frame(self):
        """
        Astropy coordinate frame
        """
        return None

    def get_basis_local(self, source_vector: np.ndarray):
        """
        Get the px,py unit vectors that define the polarization plane on
        this convention, and in the convention's frame.

        Polarization angle increments from px to py.

        Parameters
        ----------
        source_vector: np.ndarray
            Unit cartesian vector. Shape (3,N)

        Returns
        -------
        px,py : np.ndarray
            Polarization angle increases from px to py. pz is always
            the opposite of the source direction --i.e. in the direction of the
            particle.
        """

    def get_basis(self, source_direction: SkyCoord):
        """
        Get the px,py unit vectors that define the polarization plane on
        this convention. Polarization angle increments from px to py.

        Parameters
        ----------
        source_direction : SkyCoord
            The direction of the source

        Return
        ------
        px,py : SkyCoord
            Polarization angle increases from px to py. pz is always
            the opposite of the source direction --i.e. in the direction of the
            particle.
        """

        # To the convention's frame
        source_vector = source_direction.transform_to(self.frame).cartesian.xyz.value

        # Bare basis
        px,py = self.get_basis_local(source_vector)

        # To SkyCoord
        px = SkyCoord(*px, representation_type='cartesian', frame=self.frame)
        py = SkyCoord(*py, representation_type='cartesian', frame=self.frame)

        return px, py



# Orthographic projection convention
class OrthographicConvention(PolarizationConvention):

    def __init__(self,
                 ref_vector: Optional[Union[SkyCoord, np.ndarray[float]]] = None,
                 frame:Optional[BaseCoordinateFrame] = None,
                 clockwise: bool = False):
        """
        The local polarization x-axis points towards an arbitrary reference vector,
        and the polarization angle increasing counter-clockwise when looking
        at the source.

        Parameters
        ----------
        ref_vector : Union[SkyCoord, np.ndarray[float]]
            Set the reference vector, defaulting to celestial north if not provided
            (IAU convention). Alternatively, pass the cartesian representation and set a frame.
        frame : BaseCoordinateFrame
            Only used if ref_vector is a bare cartesian vector. Default: ICRS
        clockwise : bool
            Direction of increasing PA, when looking at the source. Default is false
            --i.e. counter-clockwise when looking outwards.

        """

        if frame is None:
            frame = ICRS

        if ref_vector is None:
            self._ref_vector = np.asarray([0,0,1])
            self._frame = frame
        else:
            if isinstance(ref_vector, SkyCoord):
                self._ref_vector = ref_vector.cartesian.xyz
                self._frame = ref_vector.frame
            else:
                self._ref_vector = ref_vector
                self._frame = frame

        if isinstance(self._frame, str):
            self._frame = frame_transform_graph.lookup_name(self._frame)

        self._sign = 1 if clockwise else -1

    @property
    def ref_vector(self):
        return SkyCoord(*self._ref_vector, representation_type = 'cartesian', frame = self.frame)

    def __repr__(self):
        return f"<OrthographicConvention(starting from {self.ref_vector} and {'clockwise' if self.is_clockwise else 'counter-clockwise'} when looking at the source)>"

    @property
    def is_clockwise(self):
        """
        When looking at the source
        """
        return True if self._sign == 1 else False

    @property
    def frame(self):
        return self._frame

    def get_basis_local(self, source_vector: np.ndarray):
        # Extract Cartesian coordinates for the source direction.
        pz = self._sign * source_vector

        # Broadcast reference vector
        ref = np.expand_dims(self._ref_vector, axis = tuple(np.arange(1,pz.ndim, dtype = int)))

        # Get py. Normalize because pz and ref dot not make 90deg angle
        py = np.cross(pz, ref, axisa = 0, axisb = 0, axisc = 0)
        py /= np.linalg.norm(py, axis = 0, keepdims = True)

        # Get px
        px = np.cross(py, pz, axisa = 0, axisb = 0, axisc = 0)

        return px, py

class ConventionInSpacecraftFrameMixin:
    """
    Checks for a frame with attitude

    Sub-classes need _frame property, and be sub-classes of PolarizationConvention
    """

    def get_basis(self, source_direction: SkyCoord, attitude=None):
        """

        Parameters
        ----------
        source_direction
        attitude: This overrides the object frame!

        Returns
        -------

        """
        if self._frame is None and attitude is None:
            raise RuntimeError("You need to pass an attitude to convert between local and inertial coordinates")

        if attitude is not None:
            self._frame = SpacecraftFrame(attitude=attitude)

        return super().get_basis(source_direction)


    #https://github.com/zoglauer/megalib/blob/1eaad14c51ec52ad1cb2399a7357fe2ca1074f79/src/cosima/src/MCSource.cc#L3452
class MEGAlibRelative(ConventionInSpacecraftFrameMixin, OrthographicConvention):

    def __init__(self, axis, attitude = None):
        """
        Use a polarization vector which is created the following way:
        Create an initial polarization vector which is orthogonal on the
        initial flight direction vector of the particle and the given axis vector
        (e.g. x-axis for RelativeX). This is a simple crossproduct. Then rotate
        the polarization vector (right-hand-way) around the initial flight
        direction vector of the particle by the given rotation angle.
        """

        if not isinstance(axis, str):
            raise TypeError("Axis must be a string. 'x', 'y' or 'z'.")

        axis = axis.lower()

        match axis:
            case 'x':
                ref_vector = np.asarray([1,0,0])

            case 'y':
                ref_vector = np.asarray([0,1,0])

            case 'z':
                ref_vector = np.asarray([0,0,1])

            case _:
                raise ValueError("Axis must be 'x', 'y' or 'z'.")

        frame = SpacecraftFrame(attitude = attitude)

        super().__init__(ref_vector, frame = frame, clockwise = False)

    def get_basis_local(self, source_vector: np.ndarray):

        # The MEGAlib and orthographic definitions are pretty much the
        # same, but they differ on the order of the cross products

        # In MEGAlib definition
        # pz = -source_direction = particle_direction
        # px = particle_direction x ref_vector = pz x ref_vector
        # py = pz x px

        # In the base orthographic definition
        # pz = -source_direction = particle_direction
        # px = py x pz
        # py = source_direction x ref_vector = -pz x ref_vector

        # Therefore
        # px = py_base
        # py = -px_base

        # MEGAlib's PA is counter-clockwise when looking at the sourse

        # Flip px <-> py
        py,px = super().get_basis_local(source_vector)

        # Sign of px
        py = -py

        return px,py

@PolarizationConvention.register("RelativeX")
class MEGAlibRelativeX(MEGAlibRelative):

    def __init__(self, *args, **kwargs):
        super().__init__('x', *args, **kwargs)

@PolarizationConvention.register("RelativeY")
class MEGAlibRelativeY(MEGAlibRelative):

    def __init__(self, *args, **kwargs):
        super().__init__('y', *args, **kwargs)

@PolarizationConvention.register("RelativeZ")
class MEGAlibRelativeZ(MEGAlibRelative):

    def __init__(self, *args, **kwargs):
        super().__init__('z', *args, **kwargs)


# https://lambda.gsfc.nasa.gov/product/about/pol_convention.html
# https://www.iau.org/static/resolutions/IAU1973_French.pdf
@PolarizationConvention.register("IAU")
class IAUPolarizationConvention(OrthographicConvention):

    def __init__(self):
        """
        The following resolution was adopted by Commissions 25 and 40:
        'RESOLVED, that the frame of reference for the Stokes parameters
        is that of Right Ascension and Declination with the position
        angle of electric-vector maximum, e, starting from North and
        increasing through East.
        """
        super().__init__(ref_vector = [0,0,1],
                         frame="icrs",
                         clockwise = False)


# Stereographic projection convention
@PolarizationConvention.register("stereographic")
class StereographicConvention(ConventionInSpacecraftFrameMixin, PolarizationConvention):

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

        self._frame = SpacecraftFrame(attitude=attitude)

        self._sign = 1 if clockwise else -1

    @property
    def frame(self):
        return self._frame

    def get_basis_local(self, source_vector:np.ndarray[float]):
        """
        source_vector already in SC coordinates as a vector

        Parameters
        ----------
        source_vector: (3,N)

        Returns
        -------
        px,py: Basis vector. (2,N). Also in SC coordinates
        """

        x,y,z = source_vector

        if np.allclose(source_vector, [0,0,-1]):
            raise RuntimeError("StereographicConvention is undefined at the -z (lat = -90 deg)")

        # Calculate the projection of the reference vector in stereographic coordinates
        px_x = 1 - (x ** 2 - y ** 2) / (z + 1) ** 2
        px_y = -2 * x * y / (z + 1) ** 2
        px_z = -2 * x / (z + 1)

        # Combine the components into the projection vector px
        px = np.array([px_x, px_y, px_z])

        # Normalize the projection vector
        norm = np.linalg.norm(px, axis=0)
        px /= norm

        # Calculate the perpendicular vector py using the cross product
        py = self._sign * np.cross([x, y, z], px, axis=0)

        return px,py
