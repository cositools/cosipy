import numpy as np

import pathlib

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import astropy.units as u
from astropy.io import fits
from astropy.time import Time, TimeDelta
from astropy.coordinates import (
    SkyCoord,
    UnitSphericalRepresentation,
)

from mhealpy import HealpixMap

from scoords import Attitude, SpacecraftFrame

from histpy import Axis, HealpixAxis

from cosipy.response import FullDetectorResponse
from .scatt_map import SpacecraftAttitudeMap

import logging
logger = logging.getLogger(__name__)

class SpacecraftFile():

    def __init__(self, time,
                 x_pointings = None,
                 y_pointings = None,
                 z_pointings = None,
                 attitudes = None,
                 earth_zenith = None,
                 altitude = None,
                 livetime = None,
                 frame = "galactic"):

        """
        Handles the spacecraft orientation. Calculates the dwell time
        map and point source response over a certain orientation
        period.  Exports the point source response as RMF and ARF
        files that can be read by XSPEC.

        Input must contain either pointings on at least two axes or a
        set of Attitudes for each time point; at most one of these is
        permitted to avoid inconsistency.  All input pointings and
        Attitudes will be stored in the specified frame.

        If the input pointings for provided axes are not orthogonal
        directions, they are stored as-is, but the Attitude class will
        compute an orthogonal approximation to them that is used
        internally for all operations.

        Parameters
        ----------
        Time : astropy.time.Time
            The time stamps for each pointings. Note this is NOT the
            time duration.
        x_pointings : astropy.coordinates.SkyCoord, optional
            The pointings of the x axis of the local
            coordinate system attached to the spacecraft (the default
            is `None`, which implies no input for the x pointings).
        y_pointings : astropy.coordinates.SkyCoord, optional
            The pointings of the y axis of the local
            coordinate system attached to the spacecraft (the default
            is `None`, which implies no input for the y pointings).
        z_pointings : astropy.coordinates.SkyCoord, optional
            The pointings of the z axis of the local
            coordinate system attached to the spacecraft (the default
            is `None`, which implies no input for the z pointings).
        attitudes : array, optional
            Attitudes corresponding to the pointings at each time point.
          earth_zenith : astropy.coordinates.SkyCoord, optional
            The pointings of the Earth zenith (the default is `None`,
            which implies no input for the earth pointings).
        altitude : array, optional
            Altitude of the spacecraft in km.
        livetime : array, optional
            Time in seconds the instrument is live for the corresponding
            energy bin (using left endpoints so that the last entry in
            the ori file is 0).
        frame : str, optional
            Coordinate frame for stored pointing directions and
            Attitudes (default: "galactic")

        """

        # check if the inputs are valid

        # Time
        if isinstance(time, Time):
            self._time = time
            self._raw_time = self._time.to_value(format = "unix")
            self._raw_time_delta = np.diff(self._raw_time)
        else:
            raise TypeError("The time should be a astropy.time.Time object")

        # x pointings
        if not isinstance(x_pointings, (SkyCoord, type(None))):
            raise TypeError("The x_pointings should be a SkyCoord object or None!")

        # y pointings
        if not isinstance(y_pointings, (SkyCoord, type(None))):
            raise TypeError("The y_pointings should be a SkyCoord object or None!")

        # z pointings
        if not isinstance(z_pointings, (SkyCoord, type(None))):
            raise TypeError("The z_pointings should be a SkyCoord object or None!")

        # attitudes
        if not isinstance(attitudes, (Attitude, type(None))):
            raise TypeError("attitudes should be an Attitude object or None!")

        n_axes = sum(x is not None for x in (x_pointings, y_pointings, z_pointings))

        if attitudes is None:

            if n_axes < 2:
                raise ValueError("SpacecraftFile requires pointings for at least two axes")

            self.x_pointings = None if x_pointings is None else x_pointings.transform_to(frame)
            self.y_pointings = None if y_pointings is None else y_pointings.transform_to(frame)
            self.z_pointings = None if z_pointings is None else z_pointings.transform_to(frame)

            self._attitude = Attitude.from_axes(x = x_pointings,
                                                y = y_pointings,
                                                z = z_pointings,
                                                frame = frame)
        else:

            if n_axes > 0:
                raise ValueError("Cannot specify both attitudes and per-axis pointings")

            self._attitude = attitudes.transform_to(frame)

            pointings = self._attitude.as_axes()
            self.x_pointings = pointings[0]
            self.y_pointings = pointings[1]
            self.z_pointings = pointings[2]

        # earth pointings
        if isinstance(earth_zenith, SkyCoord):
            self.earth_zenith = earth_zenith.transform_to(frame)
        elif earth_zenith is not None:
            raise TypeError("The earth_zeniths should be a SkyCoord object or None!")

        # altitude
        if altitude is not None:
            self._altitude = np.array(altitude)

        # livetime
        if livetime is not None:
            self.livetime = np.array(livetime)

        self.frame = frame


    @classmethod
    def parse_from_file(cls, file, frame='galactic'):

        """
        Parses timestamps, axis positions from file and returns to __init__.

        Parameters
        ----------
        file : str
            The file path of the pointings.
        frame : str, optional
            Frame of returned SpacecraftFile object (default: "galactic",
            which matches how the data is stored)
        Returns
        -------
        cosipy.spacecraftfile.SpacecraftFile
            The SpacecraftFile object.
        """

        orientation_file = np.loadtxt(file,
                                      usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9),
                                      delimiter=' ', skiprows=1,
                                      comments=("#", "EN"))

        time_stamps = orientation_file[:, 0]
        axis_1 = orientation_file[:, [2, 1]]
        axis_2 = orientation_file[:, [4, 3]]
        axis_3 = orientation_file[:, [7, 6]]
        altitude = np.array(orientation_file[:, 5])

        # left end points, so remove last bin.
        livetime = np.array(orientation_file[:, 8])[:-1]

        time = Time(time_stamps, format = "unix")

        # pointings are assumd to be stored in the file in galactic
        # coordinates.
        xpointings = SkyCoord(l = axis_1[:,0], b = axis_1[:,1],
                              unit=u.deg, frame = "galactic")
        zpointings = SkyCoord(l = axis_2[:,0], b = axis_2[:,1],
                              unit=u.deg, frame = "galactic")
        earthpointings = SkyCoord(l = axis_3[:,0], b = axis_3[:,1],
                                  unit=u.deg, frame = "galactic")

        return cls(time,
                   x_pointings = xpointings,
                   z_pointings = zpointings,
                   earth_zenith = earthpointings,
                   altitude = altitude,
                   livetime = livetime,
                   frame = frame)

    def get_time(self):

        """
        Return the array of pointing times as a astropy.Time object.

        Returns
        -------
        astropy.time.Time
            The time stamps of the orientation.
        """

        return self._time

    def get_time_delta(self):

        """
        Return an array of the differences between neighbouring time points.

        Returns
        -------
        time_delta : astropy.time.TimeDelta
            The differences between the neighbouring time stamps.
        """

        time_delta = TimeDelta(self._raw_time_delta, format="sec")

        return time_delta

    def get_altitude(self):

        """
        Return the array of Earth altitude.

        Returns
        -------
        numpy array
            the Earth altitude.
        """

        return self._altitude

    def get_attitude(self):

        return self._attitude

    @staticmethod
    def _cart_to_polar(v):
        """
        Convert Cartesian 3D unit direction vectors to polar coordinates.

        Parameters
        ----------
        v : np.ndarray(float) [N x 3]
          array of N 3D unit vectors

        Returns
        -------
        lon, colat : np.ndarray(float) [N]
          longitude and co-latitude corresponding to v in radians

        """

        lon   = np.arctan2(v[:,1], v[:,0])
        colat = np.arccos(v[:,2])
        return (lon, colat)

    def source_interval(self, start, stop):

        """
        Return a new SpacecraftFile object including only attitude
        information from this object in the time range [start, stop].

        start and stop must be within the range of the full object's
        times; if they exceed this range, they are trimmed to it. If
        start and stop fall between times present in the original
        object, the attitudes and other position information at these
        times are interpolated.

        Parameters
        ----------
        start : astropy.time.Time
            The start time of the orientation period.
        stop : astropy.time.Time
            The end time of the orientation period.

        Returns
        -------
        cosipy.spacecraft.SpacecraftFile

        """

        def interp_scalar(t, x1, x2):
            """
            Interpolate two scalar quantities

            Parameters
            ----------
            t : float in [0, 1]
              interpolation fraction
            x1 : float
              1st value
            x2 : float
              2nd value

            Returns
            -------
            float: interpolated value

            """

            return (1 - t) * x1 + t * x2

        def interp_direction(t, d1, d2):
            """
            Compute a direction that linearly interpolates between
            directions d1 and d2 using SLERP.

            The two directions are assumed to have the same frame,
            which is also used for the interpolated result.

            Parameters
            ----------
            t : float in [0, 1]
              interpolation fraction
            d1 : SkyCoord
              1st direction
            d2 : ndarray
              2nd direction

            Returns
            -------
            SkyCoord: interpolated direction

            """

            if d1 == d2:
                return d1

            v1 = d1.cartesian.xyz.value
            v2 = d2.cartesian.xyz.value

            # angle between v1, v2
            theta = np.arccos(np.dot(v1, v2))

            # SLERP interpolated vector
            den = np.sin(theta)
            vi = (np.sin((1-t)*theta) * v1 + np.sin(t*theta) * v2) / den

            dvi = SkyCoord(*vi, representation_type='cartesian')

            # make output representation actually (unit) spherical
            usr = UnitSphericalRepresentation(lon=dvi.spherical.lon,
                                              lat=dvi.spherical.lat)

            di = SkyCoord(usr,
                          representation_type=UnitSphericalRepresentation,
                          frame=d1.frame)

            return di

        def interp_attitude(t, att1, att2):
            """
            Compute an Attitude that linearly interpolates between
            att1 and att2 using SLERP on their quaternion
            representations.

            The two Attitudes are assumed to have the same frame,
            which is also used for the interpolated result.

            Parameters
            ----------
            t : float in [0, 1]
              interpolation fraction
            att1 : Attitude
            att2 : Attitude

            Returns
            -------
            Attitude : interpolated attitude

            """

            if att1 == att2:
                return att1

            p1 = att1.as_quat()
            p2 = att2.as_quat()

            # angle between quaternions p1, p2 (xyzw order)
            theta = 2 * np.arccos(np.dot(p1, p2))

            # SLERP interpolated quaternion
            den = np.sin(theta)
            pi = (np.sin((1-t)*theta) * p1 + np.sin(t*theta) * p2)/den

            return Attitude.from_quat(pi, frame = att1.frame)


        # trim times to within range of input orientations
        start = max(start, self._time[0])
        stop  = min(stop,  self._time[-1])

        if start > stop:
            raise ValueError("start time cannot be after stop time.")

        start_time = start.to_value(format='unix')
        stop_time  = stop.to_value(format='unix')

        # Find smallest range of indices that contain range [start_time,
        # stop_ime]. Range will always have size >= 2 unless
        # start_time == stop_time and start_time falls exactly
        # on a time point.
        start_idx = self._raw_time.searchsorted(start_time, side='right') - 1
        stop_idx  = self._raw_time.searchsorted(stop_time,  side='left')

        new_raw_time     = self._raw_time[start_idx : stop_idx + 1]
        new_attitude     = self._attitude[start_idx : stop_idx + 1]
        new_earth_zenith = self.earth_zenith[start_idx : stop_idx + 1]
        new_altitude     = self._altitude[start_idx : stop_idx + 1]
        new_livetime     = self.livetime[start_idx : stop_idx]

        if start_time > self._raw_time[start_idx] or stop_time < self._raw_time[stop_idx]:

            # need to modify first and/or last entries -- make a copy
            new_raw_time     = new_raw_time.copy()
            new_attitude     = new_attitude.copy()
            new_earth_zenith = new_earth_zenith.copy()
            new_altitude     = new_altitude.copy()

            if start_time > self._raw_time[start_idx]:

                new_raw_time[0] = start_time

                start_frac = \
                    (start_time - self._raw_time[start_idx]) / \
                    (self._raw_time[start_idx + 1] - self._raw_time[start_idx])

                new_attitude[0] = interp_attitude(start_frac,
                                                  self._attitude[start_idx],
                                                  self._attitude[start_idx + 1])

                # inputs are SkyCoords; result should be too
                new_earth_zenith[0] = interp_direction(start_frac,
                                                       self.earth_zenith[start_idx],
                                                       self.earth_zenith[start_idx + 1])

                new_altitude[0] = interp_scalar(start_frac,
                                                self._altitude[start_idx],
                                                self._altitude[start_idx + 1])

            if stop_time < self._raw_time[stop_idx]:

                new_raw_time[-1] = stop_time

                stop_frac = \
                    (stop_time - self._raw_time[stop_idx - 1]) / \
                    (self._raw_time[stop_idx] - self._raw_time[stop_idx - 1])

                new_attitude[-1] = interp_attitude(stop_frac,
                                                   self._attitude[stop_idx - 1],
                                                   self._attitude[stop_idx])

                # inputs are SkyCoords; result should be too
                new_earth_zenith[-1] = interp_direction(stop_frac,
                                                        self.earth_zenith[stop_idx - 1],
                                                        self.earth_zenith[stop_idx])

                new_altitude[-1] = interp_scalar(stop_frac,
                                                 self._altitude[stop_idx - 1],
                                                 self._altitude[stop_idx])

            # SAA livetime
            new_livetime = new_livetime.copy()

            new_livetime[0] = \
                0 if self.livetime[start_idx] == 0 \
                else new_raw_time[1] - new_raw_time[0]

            new_livetime[-1] = \
                0 if self.livetime[stop_idx - 1] == 0 \
                else new_raw_time[-1] - new_raw_time[-2]

        new_time = Time(new_raw_time, format = "unix")

        return self.__class__(new_time,
                              attitudes = new_attitude,
                              earth_zenith = new_earth_zenith,
                              altitude = new_altitude,
                              livetime = new_livetime)

    def get_target_in_sc_frame(self, target_coord):

        """
        Convert a target coordinate in an inertial frame to the path of
        the source in the spacecraft frame.  The target coordinate may
        be provided either as a SkyCoord or as a Cartesian 3-vector,
        which determines the type of the output.

        Parameters
        ----------
        target_coord : astropy.coordinates.SkyCoord or Cartesian 3-vector
            The coordinates of the target object.
        Returns
        -------
        astropy.coordinates.SkyCoord or pair of np.ndarrays
            The target coordinates in the spacecraft frame.  If input
            was a SkyCoord, output is a vector SkyCoord; otherwise, it
            is a pair (longitude, co-latitude) in radians.

        """

        useSkyCoord = isinstance(target_coord, SkyCoord)

        if useSkyCoord:
            target_coord = target_coord.transform_to(self.frame)
            target_coord = target_coord.cartesian.xyz.value

        src_path_cartesian = np.dot(self._attitude.rot.inv().as_matrix(),
                                    target_coord)

        # convert to spherical lon, colat in radians
        lon, colat = self._cart_to_polar(src_path_cartesian)

        if useSkyCoord:
            # SpacecraftFrame takes lon, lat arguments
            src_path_skycoord = SkyCoord(lon = lon, lat = np.pi/2 - colat,
                                         unit = u.rad,
                                         frame = SpacecraftFrame())

            return src_path_skycoord
        else:
            # return raw longitude and co-latitude in radians
            return lon, colat


    @staticmethod
    def _sparse_sum_duplicates(indices, weights=None, dtype=None):
        """
        Given an array of indices, possibly with duplicates, and an
        optional array of weights per index (defaults to all ones if
        None), return a sorted array of the unique values in indices
        and, for each, the sum of the weights for each unique index.

        If input weights are provided, only unique indices with
        nonzero weights are returned.

        Parameters
        ----------
        indices : array of int
        weights : array of int or float type
        dtype : data type (optional)
           Type of returned weights.  If None, type is int if weights
           not given, or float64 if they are.

        Returns
        -------
          - unique_indices : array of int
             sorted unique indices in input
          - idx_weights : array of type as described above
             sum of weights for each unique index in input

        """

        if weights is None:
            unique_indices, idx_weights = np.unique(indices,
                                                    return_counts=True)
        else:
            sp_weights = np.bincount(indices, weights)
            unique_indices = np.flatnonzero(sp_weights)
            idx_weights = sp_weights[unique_indices]

        if dtype is not None:
            idx_weights = idx_weights.astype(dtype, copy=False)

        return unique_indices, idx_weights

    def get_exposure(self, base, theta, phi=None,
                     lonlat=False, interp=True):
        """
        Compute the set of exposed HEALPix pixels relative to a
        HealpixBase arising from a sequence of spacecraft-frame
        directions with durations as specified in this SpacecraftFile.

        If theta is a SkyCoord, it specifies the full direction.
        Else, theta and phi specify the direction as angles.  If
        lonlat = True, theta and phi are longitude and latitude in
        degrees; else, theta and phi are co-latitude and longitude in
        radians.

        Parameters
        ----------
        base : HealpixBase
           HEALPix grid used to discretize exposure
        theta : np.ndarray or SkyCoord
           if phi is None, a vector SkyCoord
           if phi is not none, a vector of angles
        colat: np.ndarray, optional
           a vector of angles
        interp : bool, optional
             If True, interpolate the weights onto the HEALPix grid;
             else, just map to nearest bin. (Default: interpolate)

        Returns
        -------
        pixels : np.ndarray (int)
          all HEALPix pixels in the grid with nonzero exposure time
        exposures: np.ndarray (float)
          exposure time for each pixel

        """

        duration = self._raw_time_delta

        if len(duration) + 1 != len(theta):
            raise ValueError("Source path must have length equal to # times in SpacecraftFile")

        # remove the last src location. Effectively a 0th-order
        # interpolation
        theta = theta[:-1]
        if phi is not None:
            phi = phi[:-1]

        if interp:
            pixels, weights = base.get_interp_weights(theta=theta,
                                                      phi=phi,
                                                      lonlat=lonlat)
            weighted_duration = weights * duration[None]
        else:
            # do not interpolate
            pixels = base.ang2pix(theta=theta,
                                  phi=phi,
                                  lonlat=lonlat)
            weighted_duration = duration

        unique_pixels, unique_weights = \
            self._sparse_sum_duplicates(pixels.ravel(),
                                        weighted_duration.ravel(),
                                        dtype=np.float32)

        return unique_pixels, unique_weights

    def get_dwell_map(self, base, src_path, interp = True):

        """
        Generate a dwell-time map from a source's time-weighted
        path in local coordinates.  Interpolate the path's time
        weights onto the HEALPix grid defined by an instrument
        response's NuLambda axis.

        Parameters
        ----------
        base : HealpixBase
            Definition of HEALPix grid for map
        src_path : astropy.coordinates.SkyCoord
            Movement of source in detector frame
        interp : bool, optional
             If True, interpolate the weights onto the HEALPix grid;
             else, just map to nearest bin. (Default: interpolate)

        Returns
        -------
        mhealpy.containers.healpix_map.HealpixMap
            The dwell time map.

        """

        # check if the target source path is astropy.Skycoord object
        if type(src_path) != SkyCoord:
            raise TypeError("The coordinates of the source movement in "
                            "the Spacecraft frame must be a SkyCoord object")

        pixels, weights = self.get_exposure(base, src_path, interp=interp)

        dwell_map = HealpixMap(base = base,
                               unit = u.second,
                               coordsys = SpacecraftFrame())

        map_data = dwell_map.data
        map_data[pixels] = weights

        return dwell_map


    def get_scatt_map(self,
                      nside,
                      target_coord = None,
                      earth_occ = True,
                      angle_nbins = None):

        """
        Bin the spacecraft attitude history into a list of discretized
        attitudes with associated time weights.  Discretization is
        performed on the rotation-vector representation of the
        attitude; the supplied nside parameter describes a HEALPix
        grid that discretizes the rotvec's direction, while a multiple
        of nside defines the number of bins to discretize its angle.

        If a target coordinate is provided and earth_occ is True,
        attitudes for which the view of the target is occluded by
        the earth are excluded.

        Parameters
        ----------
        nside : int
            The nside of the scatt map.
        target_coord : astropy.coordinates.SkyCoord, optional
            The coordinates of the target object.
        earth_occ : bool, optional
            Option to include Earth occultation in scatt map calculation.
            Default is True.
        angle_nbins : int (optional)
            Number of bins used for the rotvec's angle. If none
            specified, default is 8*nside

        Returns
        -------
        cosipy.spacecraftfile.scatt_map.SpacecraftAttitudeMap
            The spacecraft attitude map.

        """

        source = target_coord

        if earth_occ:

            # earth radius
            r_earth = 6378.0

            # Need a source location to compute earth occultation
            if source is None:
                raise ValueError("target_coord is needed when earth_occ is True")

            # calculate angle between source direction and Earth zenith
            # for each time stamp
            src_angle = source.separation(self.earth_zenith)

            # get max angle based on altitude
            max_angle = np.pi - np.arcsin(r_earth/(r_earth + self._altitude))

            # get pointings that are occluded by Earth
            is_occluded = src_angle.rad >= max_angle

            # zero out weights of time bins corresponding to occluded pointings
            time_weights = np.where(is_occluded[:-1], 0, self.livetime)

        else:
            source = None # w/o occultation, result is not dependent on source
            time_weights = self.livetime

        # Get orientations as rotation vectors (center dir, angle around center)

        rot_vecs   = self._attitude[:-1].as_rotvec()
        rot_angles = np.linalg.norm(rot_vecs, axis=-1)
        rot_dirs   = rot_vecs / rot_angles[:,None]

        # discretize rotvecs for input Attitudes

        dir_axis = HealpixAxis(nside=nside, coordsys=self.frame)

        if angle_nbins is None:
            angle_nbins = 8*nside

        angle_axis = Axis(np.linspace(0., 2*np.pi, num=angle_nbins+1), unit=u.rad)

        r_lon, r_colat = self._cart_to_polar(rot_dirs.value)

        dir_bins = dir_axis.find_bin(theta=r_colat,
                                     phi=r_lon)
        angle_bins = angle_axis.find_bin(rot_angles)

        # compute list of unique rotvec bins occurring in input,
        # along with mapping from time to rotvec bin
        shape = (dir_axis.nbins, angle_axis.nbins)

        att_bins = np.ravel_multi_index((dir_bins, angle_bins),
                                        shape)

        # compute an Attitude for each unique rotvec bin

        unique_atts, time_to_att_map = np.unique(att_bins,
                                                 return_inverse=True)
        (unique_dirs, unique_angles) = np.unravel_index(unique_atts,
                                                        shape)
        v = dir_axis.pix2vec(unique_dirs)

        binned_attitudes = Attitude.from_rotvec(np.column_stack(v) *
                                                angle_axis.centers[unique_angles][:,None],
                                                frame = self.frame)

        # sum weights for all attitudes mapping to each bin
        binned_weights = np.zeros(len(unique_atts))
        np.add.at(binned_weights, time_to_att_map, time_weights)

        # remove any attitudes with zero weight
        binned_attitudes = binned_attitudes[binned_weights > 0]
        binned_weights   = binned_weights[binned_weights > 0]

        return SpacecraftAttitudeMap(binned_attitudes,
                                     u.Quantity(binned_weights, unit=u.s, copy=False),
                                     source = source)


    def get_psr_rsp(self, response_file, dwell_map, dts = None, pa_convention = None):

        """
        Generates the point source response based on the response file and dwell time map.
        dts is used to find the exposure time for this observation.

        Parameters
        ----------
        :response_file : str or pathlib.Path
            The response file for the observation
        dwell_map : HealpixMap object or str.pathlib.Path
            The time dwell map for the source, or the name of a file
            from which to load it
        dts : numpy.ndarray, optional
           The elapsed time for each pointing. It must has the same size
           as the pointings. If you have saved this array, you can pass
           it using this parameter (the defaul is `None`, which implies
           that the `dts` will be read from the instance).
        pa_convention : str, optional
           Polarization convention of response ('RelativeX',
           'RelativeY', or 'RelativeZ')

        Returns
        -------
        Ei_edges : numpy.ndarray
            The edges of the incident energy.
        Ei_lo : numpy.ndarray
            The lower edges of the incident energy.
        Ei_hi : numpy.ndarray
            The upper edges of the incident energy.
        Em_edges : numpy.ndarray
            The edges of the measured energy.
        Em_lo : numpy.ndarray
            The lower edges of the measured energy.
        Em_hi : numpy.ndarray
            The upper edges of the measured energy.
        areas : numpy.ndarray
            The effective area of each energy bin.
        matrix : numpy.ndarray
            The energy dispersion matrix.
        pa_convention : str, optional
             Polarization convention of response ('RelativeX', 'RelativeY', or 'RelativeZ')

        """

        if isinstance(dwell_map, (str, pathlib.Path)):
            dwell_map = HealpixMap.read_map(dwell_map)

        if dts is None:
            dts = self.get_time_delta()
        else:
            dts = TimeDelta(dts*u.second)

        with FullDetectorResponse.open(response_file, pa_convention=pa_convention) as response:

            # get point source response
            psr = response.get_point_source_response(dwell_map)

            Ei_edges = np.array(response.axes['Ei'].edges)
            self.Ei_lo = np.float32(Ei_edges[:-1])  # use float32 to match the requirement of the data type
            self.Ei_hi = np.float32(Ei_edges[1:])

            Em_edges = np.array(response.axes['Em'].edges)
            self.Em_lo = np.float32(Em_edges[:-1])
            self.Em_hi = np.float32(Em_edges[1:])

         # get the effective area and matrix
        logger.info("Getting the effective area ...")
        self.areas = np.float32(np.array(psr.project('Ei').to_dense().contents))/dts.to_value(u.second).sum()
        spectral_response = np.float32(np.array(psr.project(['Ei','Em']).to_dense().contents))
        self.matrix = np.float32(np.zeros((self.Ei_lo.size, self.Em_lo.size))) # initialize matrix

        logger.info("Getting the energy redistribution matrix ...")
        for i in range(self.Ei_lo.size):
            new_raw = spectral_response[i,:] / spectral_response[i,:].sum()
            self.matrix[i,:] = new_raw
        self.matrix = self.matrix.T

        return Ei_edges, self.Ei_lo, self.Ei_hi, Em_edges, self.Em_lo, self.Em_hi, self.areas, self.matrix


    def get_arf(self, out_name):

        """
        Converts the point source response to an arf file that can be read by XSPEC.

        Parameters
        ----------
        out_name: str
            The name of the arf file to save.

        """

        self.out_name = out_name

        # blow write the arf file
        copyright_string="  FITS (Flexible Image Transport System) format is defined in 'Astronomy and Astrophysics', volume 376, page 359; bibcode: 2001A&A...376..359H "

        ## Create PrimaryHDU
        primaryhdu = fits.PrimaryHDU() # create an empty primary HDU
        primaryhdu.header["BITPIX"] = -32 # since it's an empty HDU, I can just change the data type by resetting the BIPTIX value
        primaryhdu.header["COMMENT"] = copyright_string # add comments
        primaryhdu.header # print headers and their values

        col1_energ_lo = fits.Column(name="ENERG_LO", format="E", unit = "keV", array=self.Em_lo)
        col2_energ_hi = fits.Column(name="ENERG_HI", format="E", unit = "keV", array=self.Em_hi)
        col3_specresp = fits.Column(name="SPECRESP", format="E", unit = "cm**2", array=self.areas)
        cols = fits.ColDefs([col1_energ_lo, col2_energ_hi, col3_specresp]) # create a ColDefs (column-definitions) object for all columns
        specresp_bintablehdu = fits.BinTableHDU.from_columns(cols) # create a binary table HDU object

        specresp_bintablehdu.header.comments["TTYPE1"] = "label for field   1"
        specresp_bintablehdu.header.comments["TFORM1"] = "data format of field: 4-byte REAL"
        specresp_bintablehdu.header.comments["TUNIT1"] = "physical unit of field"
        specresp_bintablehdu.header.comments["TTYPE2"] = "label for field   2"
        specresp_bintablehdu.header.comments["TFORM2"] = "data format of field: 4-byte REAL"
        specresp_bintablehdu.header.comments["TUNIT2"] = "physical unit of field"
        specresp_bintablehdu.header.comments["TTYPE3"] = "label for field   3"
        specresp_bintablehdu.header.comments["TFORM3"] = "data format of field: 4-byte REAL"
        specresp_bintablehdu.header.comments["TUNIT3"] = "physical unit of field"

        specresp_bintablehdu.header["EXTNAME"]  = ("SPECRESP","name of this binary table extension")
        specresp_bintablehdu.header["TELESCOP"] = ("COSI","mission/satellite name")
        specresp_bintablehdu.header["INSTRUME"] = ("COSI","instrument/detector name")
        specresp_bintablehdu.header["FILTER"]   = ("NONE","filter in use")
        specresp_bintablehdu.header["HDUCLAS1"] = ("RESPONSE","dataset relates to spectral response")
        specresp_bintablehdu.header["HDUCLAS2"] = ("SPECRESP","extension contains an ARF")
        specresp_bintablehdu.header["HDUVERS"]  = ("1.1.0","version of format")

        new_arfhdus = fits.HDUList([primaryhdu, specresp_bintablehdu])
        new_arfhdus.writeto(f'{out_name}.arf', overwrite=True)


    def get_rmf(self, out_name):

        """
        Converts the point source response to an rmf file that can be read by XSPEC.

        Parameters
        ----------
        out_name: str
            The name of the arf file to save.
        """

        self.out_name = out_name

        # blow write the arf file
        copyright_string="  FITS (Flexible Image Transport System) format is defined in 'Astronomy and Astrophysics', volume 376, page 359; bibcode: 2001A&A...376..359H "

        ## Create PrimaryHDU
        primaryhdu = fits.PrimaryHDU() # create an empty primary HDU
        primaryhdu.header["BITPIX"] = -32 # since it's an empty HDU, I can just change the data type by resetting the BIPTIX value
        primaryhdu.header["COMMENT"] = copyright_string # add comments
        primaryhdu.header # print headers and their values

        ## Create binary table HDU for MATRIX
        ### prepare colums
        energ_lo = []
        energ_hi = []
        n_grp = []
        f_chan = []
        n_chan = []
        matrix = []
        for i in range(len(self.Ei_lo)):
            energ_lo_temp = np.float32(self.Em_lo[i])
            energ_hi_temp = np.float32(self.Ei_hi[i])

            if self.matrix[:,i].sum() != 0:
                nz_matrix_idx = np.nonzero(self.matrix[:,i])[0] # non-zero index for the matrix
                subsets = np.split(nz_matrix_idx, np.where(np.diff(nz_matrix_idx) != 1)[0]+1)
                n_grp_temp = np.int16(len(subsets))
                f_chan_temp = []
                n_chan_temp = []
                matrix_temp = []
                for m in range(n_grp_temp):
                    f_chan_temp += [subsets[m][0]]
                    n_chan_temp += [len(subsets[m])]
                for m in nz_matrix_idx:
                    matrix_temp += [self.matrix[:,i][m]]
                f_chan_temp = np.int16(np.array(f_chan_temp))
                n_chan_temp = np.int16(np.array(n_chan_temp))
                matrix_temp = np.float32(np.array(matrix_temp))
            else:
                n_grp_temp = np.int16(0)
                f_chan_temp = np.int16(np.array([0]))
                n_chan_temp = np.int16(np.array([0]))
                matrix_temp = np.float32(np.array([0]))

            energ_lo.append(energ_lo_temp)
            energ_hi.append(energ_hi_temp)
            n_grp.append(n_grp_temp)
            f_chan.append(f_chan_temp)
            n_chan.append(n_chan_temp)
            matrix.append(matrix_temp)

        col1_energ_lo = fits.Column(name="ENERG_LO", format="E",unit = "keV", array=energ_lo)
        col2_energ_hi = fits.Column(name="ENERG_HI", format="E",unit = "keV", array=energ_hi)
        col3_n_grp = fits.Column(name="N_GRP", format="I", array=n_grp)
        col4_f_chan = fits.Column(name="F_CHAN", format="PI(54)", array=f_chan)
        col5_n_chan = fits.Column(name="N_CHAN", format="PI(54)", array=n_chan)
        col6_n_chan = fits.Column(name="MATRIX", format="PE(161)", array=matrix)
        cols = fits.ColDefs([col1_energ_lo, col2_energ_hi, col3_n_grp, col4_f_chan, col5_n_chan, col6_n_chan]) # create a ColDefs (column-definitions) object for all columns
        matrix_bintablehdu = fits.BinTableHDU.from_columns(cols) # create a binary table HDU object

        matrix_bintablehdu.header.comments["TTYPE1"] = "label for field   1 "
        matrix_bintablehdu.header.comments["TFORM1"] = "data format of field: 4-byte REAL"
        matrix_bintablehdu.header.comments["TUNIT1"] = "physical unit of field"
        matrix_bintablehdu.header.comments["TTYPE2"] = "label for field   2"
        matrix_bintablehdu.header.comments["TFORM2"] = "data format of field: 4-byte REAL"
        matrix_bintablehdu.header.comments["TUNIT2"] = "physical unit of field"
        matrix_bintablehdu.header.comments["TTYPE3"] = "label for field   3 "
        matrix_bintablehdu.header.comments["TFORM3"] = "data format of field: 2-byte INTEGER"
        matrix_bintablehdu.header.comments["TTYPE4"] = "label for field   4"
        matrix_bintablehdu.header.comments["TFORM4"] = "data format of field: variable length array"
        matrix_bintablehdu.header.comments["TTYPE5"] = "label for field   5"
        matrix_bintablehdu.header.comments["TFORM5"] = "data format of field: variable length array"
        matrix_bintablehdu.header.comments["TTYPE6"] = "label for field   6"
        matrix_bintablehdu.header.comments["TFORM6"] = "data format of field: variable length array"

        matrix_bintablehdu.header["EXTNAME" ] = ("MATRIX","name of this binary table extension")
        matrix_bintablehdu.header["TELESCOP"] = ("COSI","mission/satellite name")
        matrix_bintablehdu.header["INSTRUME"] = ("COSI","instrument/detector name")
        matrix_bintablehdu.header["FILTER"]   = ("NONE","filter in use")
        matrix_bintablehdu.header["CHANTYPE"] = ("PI","total number of detector channels")
        matrix_bintablehdu.header["DETCHANS"] = (len(self.Em_lo),"total number of detector channels")
        matrix_bintablehdu.header["HDUCLASS"] = ("OGIP","format conforms to OGIP standard")
        matrix_bintablehdu.header["HDUCLAS1"] = ("RESPONSE","dataset relates to spectral response")
        matrix_bintablehdu.header["HDUCLAS2"] = ("RSP_MATRIX","dataset is a spectral response matrix")
        matrix_bintablehdu.header["HDUVERS"]  = ("1.3.0","version of format")
        matrix_bintablehdu.header["TLMIN4"]   = (0,"minimum value legally allowed in column 4")

        ## Create binary table HDU for EBOUNDS
        channels = np.arange(len(self.Em_lo), dtype=np.int16)
        e_min = np.float32(self.Em_lo)
        e_max = np.float32(self.Em_hi)

        col1_channels = fits.Column(name="CHANNEL", format="I", array=channels)
        col2_e_min = fits.Column(name="E_MIN", format="E",unit="keV", array=e_min)
        col3_e_max = fits.Column(name="E_MAX", format="E",unit="keV", array=e_max)
        cols = fits.ColDefs([col1_channels, col2_e_min, col3_e_max])
        ebounds_bintablehdu = fits.BinTableHDU.from_columns(cols)

        ebounds_bintablehdu.header.comments["TTYPE1"] = "label for field   1"
        ebounds_bintablehdu.header.comments["TFORM1"] = "data format of field: 2-byte INTEGER"
        ebounds_bintablehdu.header.comments["TTYPE2"] = "label for field   2"
        ebounds_bintablehdu.header.comments["TFORM2"] = "data format of field: 4-byte REAL"
        ebounds_bintablehdu.header.comments["TUNIT2"] = "physical unit of field"
        ebounds_bintablehdu.header.comments["TTYPE3"] = "label for field   3"
        ebounds_bintablehdu.header.comments["TFORM3"] = "data format of field: 4-byte REAL"
        ebounds_bintablehdu.header.comments["TUNIT3"] = "physical unit of field"

        ebounds_bintablehdu.header["EXTNAME"]  = ("EBOUNDS","name of this binary table extension")
        ebounds_bintablehdu.header["TELESCOP"] = ("COSI","mission/satellite")
        ebounds_bintablehdu.header["INSTRUME"] = ("COSI","nstrument/detector name")
        ebounds_bintablehdu.header["FILTER"]   = ("NONE","filter in use")
        ebounds_bintablehdu.header["CHANTYPE"] = ("PI","channel type (PHA or PI)")
        ebounds_bintablehdu.header["DETCHANS"] = (len(self.Em_lo),"total number of detector channels")
        ebounds_bintablehdu.header["HDUCLASS"] = ("OGIP","format conforms to OGIP standard")
        ebounds_bintablehdu.header["HDUCLAS1"] = ("RESPONSE","dataset relates to spectral response")
        ebounds_bintablehdu.header["HDUCLAS2"] = ("EBOUNDS","dataset is a spectral response matrix")
        ebounds_bintablehdu.header["HDUVERS"]  = ("1.2.0","version of format")

        new_rmfhdus = fits.HDUList([primaryhdu, matrix_bintablehdu,ebounds_bintablehdu])
        new_rmfhdus.writeto(f'{out_name}.rmf', overwrite=True)


    def get_pha(self, src_counts, errors, rmf_file = None, arf_file = None, bkg_file = None, exposure_time = None, dts = None, telescope="COSI", instrument="COSI"):

        """
        Generate the pha file that can be read by XSPEC. This file stores the counts info of the source.

        Parameters
        ----------
        src_counts : numpy.ndarray
            The counts in each energy band. If you have src_counts with unit counts/kev/s, you must convert it to counts by multiplying it with exposure time and the energy band width.
        errors : numpy.ndarray
            The error for counts. It has the same unit requirement as src_counts.
        rmf_file : str, optional
            The rmf file name to be written into the pha file (the default is `None`, which implies that it uses the rmf file generate by function `get_rmf`)
        arf_file : str, optional
            The arf file name to be written into the pha file (the default is `None`, which implies that it uses the arf file generate by function `get_arf`)
        bkg_file : str, optional
            The background file name (the default is `None`, which implied the `src_counts` is source counts only).
        exposure_time : float, optional
            The exposure time for this source observation (the default is `None`, which implied that the exposure time will be calculated by `dts`).
        dts : numpy.ndarray, optional
            It's used to calculate the exposure time. It has the same effect as `exposure_time`. If both `exposure_time` and `dts` are given, `dts` will write over the exposure_time (the default is `None`, which implies that the `dts` will be read from the instance).
        telescope : str, optional
            The name of the telecope (the default is "COSI").
        instrument : str, optional
            The instrument name (the default is "COSI").
        """

        if rmf_file is None:
            rmf_file = f'{self.out_name}.rmf'

        if arf_file is None:
            arf_file = f'{self.out_name}.arf'

        if dts is not None:
            dts = self.__str_or_array(dts) # FIXME: function does not exist???
            exposure_time = dts.sum()

        channel_number = len(src_counts)

        # define other hardcoded inputs
        copyright_string="  FITS (Flexible Image Transport System) format is defined in 'Astronomy and Astrophysics', volume 376, page 359; bibcode: 2001A&A...376..359H "
        channels = np.arange(channel_number)

        # Create PrimaryHDU
        primaryhdu = fits.PrimaryHDU() # create an empty primary HDU
        primaryhdu.header["BITPIX"] = -32 # since it's an empty HDU, I can just change the data type by resetting the BIPTIX value
        primaryhdu.header["COMMENT"] = copyright_string # add comments
        primaryhdu.header["TELESCOP"] = telescope # add telescope keyword valie
        primaryhdu.header["INSTRUME"] = instrument # add instrument keyword valie
        primaryhdu.header # print headers and their values

        # Create binary table HDU
        a1 = np.array(channels,dtype="int32") # I guess I need to convert the dtype to match the format J
        a2 = np.array(src_counts,dtype="int64")  # int32 is not enough for counts
        a3 = np.array(errors,dtype="int64") # int32 is not enough for errors
        col1 = fits.Column(name="CHANNEL", format="J", array=a1)
        col2 = fits.Column(name="COUNTS", format="K", array=a2,unit="count")
        col3 = fits.Column(name="STAT_ERR", format="K", array=a3,unit="count")
        cols = fits.ColDefs([col1, col2, col3]) # create a ColDefs (column-definitions) object for all columns
        bintablehdu = fits.BinTableHDU.from_columns(cols) # create a binary table HDU object

        #add other BinTableHDU hear keywords,their values, and comments
        bintablehdu.header.comments["TTYPE1"] = "label for field 1"
        bintablehdu.header.comments["TFORM1"] = "data format of field: 32-bit integer"
        bintablehdu.header.comments["TTYPE2"] = "label for field 2"
        bintablehdu.header.comments["TFORM2"] = "data format of field: 32-bit integer"
        bintablehdu.header.comments["TUNIT2"] = "physical unit of field 2"


        bintablehdu.header["EXTNAME"] = ("SPECTRUM","name of this binary table extension")
        bintablehdu.header["TELESCOP"] = (telescope,"telescope/mission name")
        bintablehdu.header["INSTRUME"] = (instrument,"instrument/detector name")
        bintablehdu.header["FILTER"] = ("NONE","filter type if any")
        bintablehdu.header["EXPOSURE"] = (exposure_time,"integration time in seconds")
        bintablehdu.header["BACKFILE"] = (bkg_file,"background filename")
        bintablehdu.header["BACKSCAL"] = (1,"background scaling factor")
        bintablehdu.header["CORRFILE"] = ("NONE","associated correction filename")
        bintablehdu.header["CORRSCAL"] = (1,"correction file scaling factor")
        bintablehdu.header["CORRSCAL"] = (1,"correction file scaling factor")
        bintablehdu.header["RESPFILE"] = (rmf_file,"associated rmf filename")
        bintablehdu.header["ANCRFILE"] = (arf_file,"associated arf filename")
        bintablehdu.header["AREASCAL"] = (1,"area scaling factor")
        bintablehdu.header["STAT_ERR"] = (0,"statistical error specified if any")
        bintablehdu.header["SYS_ERR"] = (0,"systematic error specified if any")
        bintablehdu.header["GROUPING"] = (0,"grouping of the data has been defined if any")
        bintablehdu.header["QUALITY"] = (0,"data quality information specified")
        bintablehdu.header["HDUCLASS"] = ("OGIP","format conforms to OGIP standard")
        bintablehdu.header["HDUCLAS1"] = ("SPECTRUM","PHA dataset")
        bintablehdu.header["HDUVERS"] = ("1.2.1","version of format")
        bintablehdu.header["POISSERR"] = (False,"Poissonian errors to be assumed, T as True")
        bintablehdu.header["CHANTYPE"] = ("PI","channel type (PHA or PI)")
        bintablehdu.header["DETCHANS"] = (channel_number,"total number of detector channels")

        new_phahdus = fits.HDUList([primaryhdu, bintablehdu])
        new_phahdus.writeto(f'{self.out_name}.pha', overwrite=True)


    def plot_arf(self, file_name = None, save_name = None, dpi = 300):

        """
        Read the arf fits file, plot and save it.

        Parameters
        ----------
        file_name: str, optional
            The directory if the arf fits file (the default is `None`, which implies the file name will be read from the instance).
        save_name: str, optional
            The name of the saved image of effective area (the default is `None`, which implies the file name will be read from the instance).
        dpi: int, optional
            The dpi of the saved image (the default is 300).
        """

        if file_name is None:
            file_name = f'{self.out_name}.arf'

        if save_name is None:
            save_name = self.out_name

        arf = fits.open(file_name) # read file

        # SPECRESP HDU
        self.specresp_hdu = arf["SPECRESP"]

        self.areas = np.array(self.specresp_hdu.data["SPECRESP"])
        self.Em_lo = np.array(self.specresp_hdu.data["ENERG_LO"])
        self.Em_hi = np.array(self.specresp_hdu.data["ENERG_HI"])

        E_center = (self.Em_lo+self.Em_hi)/2
        E_edges = np.append(self.Em_lo,self.Em_hi[-1])

        fig, ax = plt.subplots()
        ax.hist(E_center,E_edges,weights=self.areas,histtype='step')

        ax.set_title("Effective area")
        ax.set_xlabel("Energy[$keV$]")
        ax.set_ylabel(r"Effective area [$cm^2$]")
        ax.set_xscale("log")
        fig.savefig(f"Effective_area_for_{save_name}.png", bbox_inches = "tight", pad_inches=0.1, dpi=dpi)


    def plot_rmf(self, file_name = None, save_name = None, dpi = 300):

        """
        Read the rmf fits file, plot and save it.

        Parameters
        ----------
        file_name: str, optional
            The directory if the arf fits file (the default is `None`, which implies the file name will be read from the instance).
        save_name: str, optional
            The name of the saved image of effective area (the default is `None`, which implies the file name will be read from the instance).
        dpi: int, optional
            The dpi of the saved image (the default is 300).
        """

        if file_name is None:
            file_name = f'{self.out_name}.rmf'

        if save_name is None:
            save_name = self.out_name

        # Read rmf file
        rmf = fits.open(file_name) # read file

        # Read the ENOUNDS information
        ebounds_ext = rmf["EBOUNDS"]
        channel_low = ebounds_ext.data["E_MIN"] # energy bin lower edges for channels (channels are just incident energy bins)
        channel_high = ebounds_ext.data["E_MAX"] # energy bin higher edges for channels (channels are just incident energy bins)

        # Read the MATRIX extension
        matrix_ext = rmf['MATRIX']
        #logger.info(repr(matrix_hdu.header[:60]))
        energy_low = matrix_ext.data["ENERG_LO"] # energy bin lower edges for measured energies
        energy_high = matrix_ext.data["ENERG_HI"] # energy bin higher edges for measured energies
        data = matrix_ext.data

        # Create a 2-d numpy array and store probability data into the redistribution matrix
        rmf_matrix = np.zeros((len(energy_low),len(channel_low))) # create an empty matrix
        for i in range(data.shape[0]): # i is the measured energy index, examine the matrix_ext.data rows by rows
            if data[i][5].sum() == 0: # if the sum of probabilities is zero, then skip since there is no data at all
                pass
            else:
                #measured_energy_index = np.argwhere(energy_low == data[157][0])[0][0]
                f_chan = data[i][3] # get the starting channel of each subsets
                n_chann = data[i][4] # get the number of channels in each subsets
                matrix = data[i][5] # get the probabilities of this row (incident energy)
                indices = []
                for k in f_chan:
                    channels = 0
                    channels = np.arange(k,k + n_chann[np.argwhere(f_chan == k)][0][0]).tolist() # generate the cha
                    indices += channels # fappend the channels togeter
                indices = np.array(indices)
                for m in indices:
                    rmf_matrix[i][m] = matrix[np.argwhere(indices == m)[0][0]] # write the probabilities into the empty matrix


        # plot the redistribution matrix
        xcenter = np.divide(energy_low+energy_high,2)
        x_center_coords = np.repeat(xcenter, 10)
        y_center_coords = np.tile(xcenter, 10)
        energy_all_edges = np.append(energy_low,energy_high[-1])
        #bin_edges = np.array([incident_energy_bins,incident_energy_bins]) # doesn't work
        bin_edges = np.vstack((energy_all_edges, energy_all_edges))
        #logger.info(bin_edges)

        self.probability = []
        for i in range(10):
            for j in range(10):
                self.probability.append(rmf_matrix[i][j])
        #logger.info(type(probability))

        plt.hist2d(x=x_center_coords,y=y_center_coords,weights=self.probability,bins=bin_edges, norm=LogNorm())
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("Incident energy [$keV$]")
        plt.ylabel("Measured energy [$keV$]")
        plt.title("Redistribution matrix")
        #plt.xlim([70,10000])
        #plt.ylim([70,10000])
        plt.colorbar(norm=LogNorm())
        plt.savefig(f"Redistribution_matrix_for_{save_name}.png", bbox_inches = "tight", pad_inches=0.1, dpi=dpi)
