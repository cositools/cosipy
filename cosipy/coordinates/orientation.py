import numpy as np
from astropy.coordinates import SkyCoord, Galactic
from astropy import units as u
from astropy.time import Time
from scoords import Attitude

class Orientation_file:

    def __init__(self, time, axis_1, axis_2):

        #stores the parsed contents

        self._load_time = time
        self._x_direction = axis_1
        self._z_direction = axis_2

    @classmethod
    def parse_from_file(cls, file):

        #parses timestamps, axis positions from file and returns to __init__

        time_stamps = np.loadtxt(file, usecols = 1, delimiter = ' ', skiprows = 1)
        axis_1 = np.loadtxt(file, usecols = (2,3), delimiter = ' ', skiprows = 1)
        axis_2 = np.loadtxt(file, usecols = (4,5), delimiter = ' ', skiprows = 1)

        return cls(time_stamps, axis_1, axis_2)

    def get_time(self):

        #returns an array of pointing times as a time object

        self._time = Time(self._load_time, format='unix')

        return self._time

    def get_attitude(self):

        #returns an attitude object from the scoords library for the spacecraft

        x_sky = SkyCoord(self._x_direction[:,1]*u.deg, self._x_direction[:,0]*u.deg, frame = 'galactic')
        z_sky = SkyCoord(self._z_direction[:,1]*u.deg, self._z_direction[:,0]*u.deg, frame = 'galactic')

        self._attitude = Attitude.from_axes(x = x_sky, z = z_sky)

        return self._attitude

    def get_time_delta(self):

        #returns an array of the time between observations as a time object

        time_delta = np.diff(self._load_time)
        time_delta = Time(time_delta, format='unix')

        return time_delta

    def interpolate_direction(self, trigger, idx, direction):

        #linearly interpolates position at a given time between two timestamps

        new_direction_lat = np.interp(trigger.value, self._load_time[idx : idx + 2], direction[idx : idx + 2, 0])
        if (direction[idx, 1] > direction[idx + 1, 1]):
            new_direction_long = np.interp(trigger.value, self._load_time[idx : idx + 2], [direction[idx, 1], 360 + direction[idx + 1, 1]])
            new_direction_long = new_direction_long - 360
        else:
            new_direction_long = np.interp(trigger.value, self._load_time[idx : idx + 2], direction[idx : idx + 2, 1])

        return np.array([new_direction_lat, new_direction_long])

    def source_interval(self, start, stop):

        #returns the Orientation file class object for the source time
        
        if(start.format != 'unix' or stop.format != 'unix'):
            start = Time(start.unix, format='unix')
            stop = Time(stop.unix, format='unix')

        if(start > stop):
            raise ValueError("start time cannot be after stop time.")

        stop_idx = self._load_time.searchsorted(stop.value)

        if (start.value % 1 == 0):
            start_idx = self._load_time.searchsorted(start.value)
            new_times = self._load_time[start_idx : stop_idx + 1]
            new_x_direction = self._x_direction[start_idx : stop_idx + 1]
            new_z_direction = self._z_direction[start_idx : stop_idx + 1]
        else:
            start_idx = self._load_time.searchsorted(start.value) - 1

            x_direction_start = self.interpolate_direction(start, start_idx, self._x_direction)
            z_direction_start = self.interpolate_direction(start, start_idx, self._z_direction)

            new_times = self._load_time[start_idx + 1 : stop_idx + 1]
            new_times = np.insert(new_times, 0, start.value)

            new_x_direction = self._x_direction[start_idx + 1 : stop_idx + 1]
            new_x_direction = np.insert(new_x_direction, 0, x_direction_start, axis = 0)

            new_z_direction = self._z_direction[start_idx + 1 : stop_idx + 1]
            new_z_direction = np.insert(new_z_direction, 0, z_direction_start, axis = 0)


        if (stop.value % 1 != 0):
            stop_idx = self._load_time.searchsorted(stop.value) - 1

            x_direction_stop = self.interpolate_direction(stop, stop_idx, self._x_direction)
            z_direction_stop = self.interpolate_direction(stop, stop_idx, self._z_direction)

            new_times = np.delete(new_times, -1)
            new_times = np.append(new_times, stop.value)

            new_x_direction = new_x_direction[:-1]
            new_x_direction = np.append(new_x_direction, [x_direction_stop], axis = 0)

            new_z_direction = new_z_direction[:-1]
            new_z_direction = np.append(new_z_direction, [z_direction_stop], axis = 0)

        return Orientation_file(new_times, new_x_direction, new_z_direction)
