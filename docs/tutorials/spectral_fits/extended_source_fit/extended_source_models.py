from astromodels import *
import numpy as np
from astromodels.utils.angular_distance import angular_distance
from past.utils import old_div

class SpecFromDat(Function1D, metaclass=FunctionMeta):
        r"""
        description :
            A  spectrum loaded from a dat file
        latex : a spectrum loaded from a .dat file
        parameters :
            K :
                desc : Normalization
                initial value : 1.0
                is_normalization : True
                transformation : log10
                min : 1e-30
                max : 1e3
                delta : 0.1
        properties:
            dat:
                desc: the data file to load
                initial value: test.dat
                defer: True
        """
        
        #def __init__(
        #    self,
        #name: Optional[str] = None,
        #function_definition: Optional[str] = None,
        #parameters: Optional[Dict[str, Parameter]] = None,
        #properties: Optional[Dict[str, FunctionProperty]] = None,
        #):
            #self.dataFlux = np.genfromtxt(dat,comments = "#",usecols = (2),skip_footer=1,skip_header=5)
            #self.dataEn = np.genfromtxt(dat,comments = "#",usecols = (1),skip_footer=1,skip_header=5)
            #print (self.dataEn)
            #pass
            # could do something fancy here to check that dataEn is just incrermenting by 1 keV
            # and save time later
            # but do we need to be generalizable to a .dat that is not defined for every 1 keV??

        # NOTE: I would like to get something here in init, but everything i try keeps throwing errors
        # I will return to this
        # for now, we have to load the data every time we evaluate the function... 
        # but that's probably ok since this function doesn't really need to get called a lot of times
        # see below
            
        def _set_units(self, x_unit, y_unit):
            
            self.K.unit = y_unit

        def evaluate(self, x, K): #ultimately want to add K as a normalization, done once...
            
            #try: self.dataEn[0]
            #except: 
            self.dataFlux = np.genfromtxt(self.dat.value,comments = "#",usecols = (2),skip_footer=1,skip_header=5)
            self.dataEn = np.genfromtxt(self.dat.value,comments = "#",usecols = (1),skip_footer=1,skip_header=5)

            if self._x_unit != None: 
                self.dataEn *= self._x_unit
            
            return K* np.interp(x,self.dataEn,self.dataFlux,left=0,right=0)



class Wide_Asymm_Gaussian_on_sphere(Function2D, metaclass=FunctionMeta):
    r"""
    description :

        A bidimensional Gaussian function on a sphere (in spherical coordinates)

        see https://en.wikipedia.org/wiki/Gaussian_function#Two-dimensional_Gaussian_function

    parameters :

        lon0 :

            desc : Longitude of the center of the source
            initial value : 0.0
            min : 0.0
            max : 360.0

        lat0 :

            desc : Latitude of the center of the source
            initial value : 0.0
            min : -90.0
            max : 90.0

        a :

            desc : Standard deviation of the Gaussian distribution (major axis)
            initial value : 10
            min : 0
            max : 90

        e :

            desc : Excentricity of Gaussian ellipse
            initial value : 0.9
            min : 0
            max : 1

        theta :

            desc : inclination of major axis to a line of constant latitude
            initial value : 10.
            min : -90.0
            max : 90.0

    """
    def _set_units(self, x_unit, y_unit, z_unit):

        # lon0 and lat0 and a have most probably all units of degrees. However,
        # let's set them up here just to save for the possibility of using the
        # formula with other units (although it is probably never going to happen)

        self.lon0.unit = x_unit
        self.lat0.unit = y_unit
        self.a.unit = x_unit
        self.e.unit = u.dimensionless_unscaled
        self.theta.unit = u.degree

    def evaluate(self, x, y, lon0, lat0, a, e, theta):

        lon, lat = x, y

        b = a * np.sqrt(1.0 - e**2)

        dX = np.atleast_1d(angular_distance(lon0, lat0, lon, lat0))
        dY = np.atleast_1d(angular_distance(lon0, lat0, lon0, lat))

        dlon = lon - lon0
        if isinstance(dlon, u.Quantity):
            dlon = (dlon.to(u.degree)).value

        idx = np.logical_and(
            np.logical_or(dlon < 0, dlon > 180),
            np.logical_or(dlon > -180, dlon < -360),
        )
        dX[idx] = -dX[idx]

        idx = lat < lat0
        dY[idx] = -dY[idx]

        if isinstance(theta, u.Quantity):
            phi = (theta.to(u.degree)).value + 90.0
        else:
            phi = theta + 90.0

        cos2_phi = np.power(np.cos(phi * np.pi / 180.0), 2)
        sin2_phi = np.power(np.sin(phi * np.pi / 180.0), 2)

        sin_2phi = np.sin(2.0 * phi * np.pi / 180.0)

        A = old_div(cos2_phi, (2.0 * b**2)) + old_div(sin2_phi, (2.0 * a**2))
        
        B = old_div(-sin_2phi, (4.0 * b**2)) + old_div(sin_2phi, (4.0 * a**2))

        C = old_div(sin2_phi, (2.0 * b**2)) + old_div(cos2_phi, (2.0 * a**2))

        E = -A * np.power(dX, 2) + 2.0 * B * dX * dY - C * np.power(dY, 2)

        return np.power(old_div(180, np.pi), 2) * 1.0 / (2 * np.pi * a * b) * np.exp(E)

    def get_boundaries(self):

        # Truncate the gaussian at 2 times the max of sigma allowed

        min_lat = max(-90.0, self.lat0.value - 2 * self.a.max_value)
        max_lat = min(90.0, self.lat0.value + 2 * self.a.max_value)

        max_abs_lat = max(np.absolute(min_lat), np.absolute(max_lat))

        if (
            max_abs_lat > 89.0
            or 2 * self.a.max_value / np.cos(max_abs_lat * np.pi / 180.0) >= 180.0
        ):

            min_lon = 0.0
            max_lon = 360.0

        else:

            min_lon = self.lon0.value - 2 * self.a.max_value / np.cos(
                max_abs_lat * np.pi / 180.0
            )
            max_lon = self.lon0.value + 2 * self.a.max_value / np.cos(
                max_abs_lat * np.pi / 180.0
            )

            if min_lon < 0.0:

                min_lon = min_lon + 360.0

            elif max_lon > 360.0:

                max_lon = max_lon - 360.0

        return (min_lon, max_lon), (min_lat, max_lat)

    def get_total_spatial_integral(self, z=None):
        """
        Returns the total integral (for 2D functions) or the integral over the spatial components (for 3D functions).
        needs to be implemented in subclasses.

        :return: an array of values of the integral (same dimension as z).
        """

        if isinstance(z, u.Quantity):
            z = z.value
        return np.ones_like(z)        
        
