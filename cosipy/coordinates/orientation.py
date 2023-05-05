import numpy as np
from astropy.coordinates import SkyCoord, Galactic
from astropy import units as u
from astropy.time import Time
from scoords import Attitude

class Orientation_file:

    def __init__(self, file):

        #defines the file that needs to be parsed

        self._file = file
        self._load_time = np.loadtxt(self._file, usecols = 1, delimiter = ' ')
        self._x_direction = np.loadtxt(self._file, usecols = (2,3), delimiter = ' ')
        self._z_direction = np.loadtxt(self._file, usecols = (4,5), delimiter = ' ')


    def get_time(self):

        #returns an array of pointing times as a time object

        time = Time(self._load_time, format='unix')

        return time

    def get_attitude(self):

        #returns an attitude object from the scoords library for the spacecraft

        x_sky = SkyCoord(self._x_direction[:,1]*u.deg, self._x_direction[:,0]*u.deg, frame = 'galactic')
        z_sky = SkyCoord(self._z_direction[:,1]*u.deg, self._z_direction[:,0]*u.deg, frame = 'galactic')

        attitude = Attitude.from_axes(x = x_sky, z = z_sky)

        return attitude

    def get_time_delta(self):

        #returns an array of the time between observations as a time object
        time_delta = np.diff(self._load_time)
        time_delta = Time(time_delta, format='unix')

        return time_delta
        
