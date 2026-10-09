import json
from pathlib import Path
from typing import Union, Optional

import numpy as np

from astropy.time import Time
from astropy.coordinates import (
    SkyCoord,
    EarthLocation,
    GCRS,
    ITRS,
    Galactic,
    CartesianRepresentation
)
from astropy.units import Quantity
import astropy.units as u
from astropy.table import QTable
from astropy.io import fits

from mhealpy import HealpixBase, HealpixMap

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.projections.geo import GeoAxes

from histpy import Axis, HealpixAxis

from scoords import Attitude, SpacecraftFrame

from .scatt_map import SpacecraftAttitudeMap
from cosipy.event_selection import GoodTimeInterval

import logging
logger = logging.getLogger(__name__)


__all__ = ["SpacecraftHistory"]

class SpacecraftHistory:

    # change version number if FITS on-disk format changes
    supported_file_versions = ["20260130"]

    # radius of earth
    _r_earth = 6378.0
    _far_dist = 1*u.Mpc

    def __init__(self,
                 obstime: Time,
                 attitude: Attitude,
                 location: GCRS,
                 livetime: Quantity = None,
                 earth_location: EarthLocation = None):
        """Handles the spacecraft orientation. Calculates the dwell time
        map and point source response over a certain orientation
        period.

        Parameters
        ----------
        obstime:
            The obstime stamps for each pointing. Note this is NOT
            the obstime duration, see "livetime".
        attitude:
            Spacecraft orientation with respect to an inertial system.
        location:
            Location of the spacecraft at each timestamp in
            Earth-centered inertial (ECI) coordinates.
        livetime:
            Time the instrument was live for the corresponding obstime
            bin. Should have one less element than the number of
            timestamps. If not provided, assume that the instrument
            was fully on without interruptions.
        earth_location:
            Geographic location of the spacecraft at each timestamp.
            Computed from 'location' when first needed if not provided.

        """

        self._obstime = obstime

        if obstime.size < 2:
            raise ValueError("SpacecraftHistory needs at least two timestamps.")

        self._t0 = self._obstime[0]
        self._obstime_dt_jd = self._get_dt_jd(obstime)

        self._time_axis = Axis(self._obstime_dt_jd, copy = False,
                               label= 'obstime_dt_jd')

        if livetime is None:
            livetime = time_axis.widths.to(u.s)

        self._livetime = livetime

        if not (location.shape == () or location.shape == obstime.shape):
            raise ValueError(f"'location' must be a scalar or have the same length as the timestamps ({obstime.shape}), but it has shape ({location.shape})")

        if not (attitude.shape == () or attitude.shape == obstime.shape):
            raise ValueError(f"'attitude' must be a scalar or have the same length as the timestamps ({obstime.shape}), but it has shape ({attitude.shape})")

        self._attitude = attitude

        self._gcrs = location

        if not (earth_location is None or earth_location.shape == obstime.shape):
            raise ValueError(f"'earth_location' must have the same length as the timestamps ({obstime.shape}), but it has shape ({earth_location.shape})")

        self._earth_location = earth_location

        self._cache_earth_occ = False

    def _get_dt_jd(self, time: Time):
        time = time - self._t0
        return time.jd1 + time.jd2

    @property
    def nintervals(self):
        return len(self._livetime)

    @property
    def intervals_duration(self):
        return Quantity(self._time_axis.widths, u.day, copy = False)

    @property
    def intervals_tstart(self):
        return self._obstime[:-1]

    @property
    def intervals_tstop(self):
        return self._obstime[1:]

    @property
    def tstart(self):
        return self._obstime[0]

    @property
    def tstop(self):
        return self._obstime[-1]

    @property
    def npoints(self):
        return self._obstime.size

    @property
    def obstime(self):
        return self._obstime

    @property
    def livetime(self):
        return self._livetime

    @property
    def attitude(self):
        return self._attitude

    @property
    def location(self) -> GCRS:
        return self._gcrs

    @property
    def altitude(self) -> Quantity:
        """
        Altitude with respect to Earth's surface
        """
        _, _, altitude = self._gcrs_to_earth_zenith_altitude(self._gcrs)
        return altitude

    @property
    def earth_zenith(self) -> SkyCoord:
        """
        Galactic pointing of the Earth's zenith at the location of the SC
        """
        lon,  lat, _ = self._gcrs_to_earth_zenith_altitude(self._gcrs)
        return SkyCoord(lon, lat, frame=Galactic(), copy=False)

    @property
    def earth_location(self) -> EarthLocation:
        """
        Geographic location of the SC at each timestamp.

        Computed the first time it is needed and cached, unless it was
        provided on initialization.
        """
        if self._earth_location is None:
            self._earth_location = self._compute_earth_location()

        return self._earth_location

    def _compute_earth_location(self) -> EarthLocation:
        """
        The full GCRS to ITRS transformation is slow, so it is only
        computed once per day, at the reference times closest to each
        timestamp. In between, only the change in the Earth rotation
        angle is applied. The error from the precession and nutation
        of the pole within half a day is at the arcsec level.
        """
        dt = (self._obstime - self.tstart).to_value(u.day)
        days = np.round(dt).astype(int)
        ref_time = self.tstart + np.arange(days[-1] + 1) * u.day

        # rotation[:, :, day] is the GCRS to ITRS rotation matrix at ref_time[day]
        basis = GCRS(CartesianRepresentation(np.eye(3)[:, :, None] * u.km), obstime=ref_time)
        rotation = basis.transform_to(ITRS(obstime=ref_time)).cartesian.xyz.value

        xyz = self._gcrs.cartesian.xyz.to_value(u.km).reshape(3, -1)
        xyz = np.broadcast_to(xyz, (3, self.npoints))

        # Timestamps are sorted, so each day is a contiguous block
        edges = np.searchsorted(days, np.arange(ref_time.size + 1))
        xyz_ref = np.empty(xyz.shape)
        for day, (start, stop) in enumerate(zip(edges[:-1], edges[1:])):
            xyz_ref[:, start:stop] = rotation[:, :, day] @ xyz[:, start:stop]

        # Earth rotation angle rate, in rad/day (IERS 2010 Conventions)
        delta_era = 2 * np.pi * 1.00273781191135448 * (dt - days)

        x = np.cos(delta_era) * xyz_ref[0] + np.sin(delta_era) * xyz_ref[1]
        y = -np.sin(delta_era) * xyz_ref[0] + np.cos(delta_era) * xyz_ref[1]

        return EarthLocation.from_geocentric(x, y, xyz_ref[2], unit=u.km)

    @property
    def rocking_angle(self) -> Quantity:
        """
        Angle between the SC z-axis and the Earth's zenith at the SC
        location, measured on the plane containing the zenith and the
        orbit normal. Positive values point towards the orbit normal
        (north for a prograde orbit).
        """
        zenith = self._gcrs.cartesian.xyz.value
        velocity = np.gradient(zenith, self._obstime_dt_jd, axis=1,
                               edge_order=min(2, self.npoints - 1))
        normal = np.cross(zenith, velocity, axis=0)

        zaxis = self._attitude.as_axes()[2].transform_to(GCRS())
        zaxis = zaxis.cartesian.xyz.value

        z_up = np.sum(zaxis * zenith, axis=0) / np.linalg.norm(zenith, axis=0)
        z_north = np.sum(zaxis * normal, axis=0) / np.linalg.norm(normal, axis=0)

        return np.rad2deg(np.arctan2(z_north, z_up)) * u.deg

    @property
    def roll_angle(self) -> Quantity:
        """
        Angle of the SC x-axis around the z-axis, measured from the
        direction of motion projected onto the plane perpendicular to
        the z-axis. Positive values are counter-clockwise when looking
        from the tip of the z-axis.
        """
        velocity = np.gradient(self._gcrs.cartesian.xyz.value, self._obstime_dt_jd, axis=1,
                               edge_order=min(2, self.npoints - 1))

        xaxis, _, zaxis = (axis.transform_to(GCRS()).cartesian.xyz.value
                           for axis in self._attitude.as_axes())

        x_along = np.sum(xaxis * velocity, axis=0)
        x_across = np.sum(np.cross(velocity, xaxis, axis=0) * zaxis, axis=0)

        return np.rad2deg(np.arctan2(x_across, x_along)) * u.deg

    @staticmethod
    def _default_file_version(file_version = None):
        if file_version is None:
            file_version = SpacecraftHistory.supported_file_versions[-1]
        elif file_version not in SpacecraftHistory.supported_file_versions:
            raise RuntimeError(f"File version not {file_version} not in {SpacecraftHistory.supported_file_versions}")

        return file_version

    def write_fits(self, filename, overwrite=False, compress=False,
                   file_version=None):
        """
        Write the contents of this object as a FITS file for later
        retrieval. We use Astropy QTable functionality to create the
        FITS file, so that every column is stored with its unit.

        If compression is requested, the resulting file will have the
        name {filename}.gz; the ".gz" should not be specfied as part
        of the input.  (If it is, the file will be compressed
        regardless of the setting of the compress flag.)

        Parameters
        ----------
        filename : str or pathlib Path
            The file path to the FITS file
        overwrite : bool, optional
            Overwrite the named file if it exists (default False)
        compress : bool, optional
            GZip-compress the FITS file (default False)
        file_version: str
            Defaults to latest file version. See
            supported_file_versions for options.

        """

        t = QTable()

        # Time objects do not carry units.  We store the time as
        # seconds in case someone reads it without converting to Time().
        t['TimeStamp'] = Quantity(self.obstime.unix, unit=u.s, copy=False)

        # Make sure the pointings are written in galactic coordinates,
        # however they are stored internally.  The units of the
        # lon/lat angles are preserved in the file, but they are saved
        # as simple Quantities rather than more complex
        # Angle/Latitude/Longitude objects.

        xc, yc, zc = self.attitude.transform_to('galactic').as_axes()

        xp = np.column_stack((xc.l, xc.b))
        t['XPointings'] = Quantity(xp, copy=False)

        zp = np.column_stack((zc.l, zc.b))
        t['ZPointings'] = Quantity(zp, copy=False)

        lon, lat, altitude = self._gcrs_to_earth_zenith_altitude(self._gcrs)

        ez_gal = np.column_stack((lon, lat))

        t['EarthZenith'] = Quantity(ez_gal, copy=False)
        t['Altitude'] = Quantity(altitude, copy=False)

        # add dummy to make sure livetime array length matches
        # other array lengths for writing
        t['LiveTime'] = np.append(self.livetime, 0*self.livetime.unit)

        # ensure the VERSION card ends up in the table's hdu header
        file_version = self._default_file_version(file_version)

        t.meta["VERSION"] = file_version

        filename = Path(filename)

        if compress and filename.suffix != ".gz":
            # FITS table writer will automatically compress if
            # file name ends with .gz
            filename = filename.parent / (filename.name + ".gz")

        if filename.exists() and not overwrite:
            raise RuntimeError(f"Not overwriting existing file '{filename}'")
        else:
            # must convert filanme Path to string, or astropy FITS I/O
            # does not honor overwrite flag!
            t.write(str(filename), format='fits', overwrite=True)

            # reopen the FITS file to add the VERSION card
            # to the PRIMARY header in hdu 0 as well. HEASARC
            # requires the version in all hdus in the file.
            with fits.open(filename, mode="update") as hdul:
                hdul[0].header["VERSION"] = file_version

    @classmethod
    def open(cls, filename, tstart:Time = None, tstop:Time = None) -> "SpacecraftHistory":

        """Parses timestamps, axis positions from file and returns to
        __init__.

        Parameters
        ----------
        filename : str
            The file path of the pointings.
        tstart:
            Start reading the file from an interval *including* this
            time. Use select_interval() to cut the SC file at exactly
            this time.
        tstop:
            Stop reading the file at an interval *including* this
            time. Use select_interval() to cut the SC file at exactly
            this time.

        Returns
        -------
        cosipy.spacecraftfile.SpacecraftHistory
            The SpacecraftHistory object.

        """

        filename = Path(filename)

        if filename.suffix == ".fits" or filename.suffixes[-2:] == [".fits", ".gz"]:
            return cls._open_fits(filename, tstart, tstop)
        elif filename.suffix == ".ori":
            return cls._open_ori(filename, tstart, tstop)
        else:
            raise ValueError("Unsupported file format. Only .ori and .fits/.fits.gz extensions are supported.")

    @staticmethod
    def _find_time_index(time:Time, tstart:Time, tstop:Time):

        t0 = time if time.isscalar else time[0]

        time = (time - t0).jd

        if tstart is not None:
            tstart = (tstart - t0).jd
            start_idx = np.searchsorted(time, tstart, side='right')
        else:
            start_idx = 0

        if tstop is not None:
            tstop = (tstop - t0).jd
            stop_idx = np.searchsorted(time, tstop, side='right')
        else:
            stop_idx = time.size

        # Include the full bin, but prevent an index error
        start_idx = max(0, start_idx - 1)
        stop_idx  = min(time.size, stop_idx + 1)

        return start_idx, stop_idx

    @classmethod
    def _open_fits(cls, filename, tstart:Time = None, tstop:Time = None) -> "SpacecraftHistory":
        """
        Read orientation data from a FITS file and construct a
        SpacecraftFile object.  The FITS file is assumed to contain an
        Astropy QTable produced by the write_fits() method.  Astropy
        supports .fits.gz natively, so this function can read either
        compressed or uncompressed FITS.

        Parameters
        ----------
        filename : str
            The file path to the FITS file

        Returns
        -------
        cosipy.spacecraftfile.SpacecraftFile
            The SpacecraftFile object

        """

        t = QTable.read(filename)

        #Convert EVERY column to native byte order ---
        for colname in t.colnames:
            col = t[colname]
            t[colname] = np.astype(col, col.dtype.newbyteorder('='), copy=False)
        
            
        
        # make sure we have version info, and that we support this version
        if "VERSION" not in t.meta:
            raise ValueError("FITS orientation file has no version info")
        elif t.meta["VERSION"] not in cls.supported_file_versions:
            raise ValueError(f"FITS orientation file has version {t.meta['VERSION']} "
                             f"that is not supported {cls.supported_file_versions}")

        time_stamps = Time(t['TimeStamp'], format = "unix")

        if tstart is not None or tstop is not None:
            # Cut early to skip some conversions later on

            start_idx, stop_idx = cls._find_time_index(time_stamps, tstart, tstop)

            time_stamps = time_stamps[start_idx:stop_idx]
            t = t[start_idx:stop_idx]

        # pointings are assumed to be stored in the file in
        # galactic # coordinates.
        xp = t['XPointings']
        xpointings = SkyCoord(l = xp[:,0], b = xp[:,1],
                              frame = Galactic(),
                              copy=False)
        zp = t['ZPointings']
        zpointings = SkyCoord(l = zp[:,0], b = zp[:,1],
                              frame = Galactic(),
                              copy=False)

        attitude = Attitude.from_axes(x = xpointings, z = zpointings,
                                      frame = Galactic())

        ez = t['EarthZenith']
        earth_lon = ez[:,0]
        earth_lat = ez[:,1]
        altitude = t['Altitude']

        gcrs = cls._earth_zenith_altitude_to_gcrs(earth_lon,
                                                  earth_lat,
                                                  altitude)

        # left end points, so remove last bin.
        livetime = t['LiveTime'][:-1]

        return cls(time_stamps, attitude, gcrs, livetime)

    @classmethod
    def _open_ori(cls, file, tstart:Time = None, tstop:Time = None) -> "SpacecraftHistory":
        """Parses an .ori txt file with MEGAlib formatting.

        # Columns
        # 0: Always "OG" (orbital geometry)
        # 1: obstime: timestamp in unix seconds
        # 2: lat_x: galactic latitude of SC x-axis (deg)
        # 3: lon_x: galactic longitude of SC x-axis (deg)
        # 4: lat_z galactic latitude of SC z-axis (deg)
        # 5: lon_z: galactic longitude of SC y-axis (deg)
        # 6: altitude: altitude above from Earth's ellipsoid (km)
        # 7: Earth_lat: galactic latitude of the direction the Earth's
        #    zenith is pointing to at the SC location (deg)
        # 8: Earth_lon: galactic longitude of the direction the
        #    Earth's zenith is pointing to at the SC location (deg)
        # 9: livetime (previously called SAA): accumulated uptime up
        #    to the following entry (seconds)

        Parameters
        ----------
        file:
            Path to .ori file
        tstart: Time
            start time to extract from file
        tstop: Time
            end time to extract from file
        Returns
        -------
        cosipy.spacecraftfile.spacecraft_file
            The SpacecraftHistory object.

        """

        import pandas as pd

        # First and last line are read only by MEGAlib e.g.
        # Type OrientationsGalactic
        # ...
        # EN

        # Check that the EN line is present (and throw an error if
        # not), lest we accidentally drop the last line of orientation
        # data!
        with open(file, "rb") as ori_file:
            ori_file.seek(-4, 2)
            s = ori_file.read(4)

            # allow for possibility of missing EOL at end of file
            if s[-1] == ord('\n'):
                s = s[:-1]

            if s != b'\nEN':
                raise ValueError(".ori file must end with final line 'EN'!")

        # now read the full file, skipping the first and last line.
        # Use [:-1] instead of skipfooter=1; the latter causes a
        # fallback to the slow Python CSV reading engine instead of
        # faster C engine.

        df = pd.read_csv(file, sep=r"\s+", skiprows=1,
                         usecols=tuple(range(1,10)),
                         header = None, comment = '#')
        vals = df.values[:-1].transpose()

        # assign units to read values
        time_stamps = Time(vals[0], format="unix", copy=False)
        lat_x       = Quantity(vals[1], unit=u.deg, copy=False)
        lon_x       = Quantity(vals[2], unit=u.deg, copy=False)
        lat_z       = Quantity(vals[3], unit=u.deg, copy=False)
        lon_z       = Quantity(vals[4], unit=u.deg, copy=False)
        altitude    = Quantity(vals[5], unit=u.km, copy=False)
        earth_lat   = Quantity(vals[6], unit=u.deg, copy=False)
        earth_lon   = Quantity(vals[7], unit=u.deg, copy=False)
        livetime    = Quantity(vals[8], unit=u.s, copy=False)

        if tstart is not None or tstop is not None:
            # Cut early to skip some conversions later on

            start_idx, stop_idx = cls._find_time_index(time_stamps, tstart, tstop)

            time_stamps = time_stamps[start_idx:stop_idx]
            lat_x = lat_x[start_idx:stop_idx]
            lon_x = lon_x[start_idx:stop_idx]
            lat_z = lat_z[start_idx:stop_idx]
            lon_z = lon_z[start_idx:stop_idx]
            altitude = altitude[start_idx:stop_idx]
            earth_lat = earth_lat[start_idx:stop_idx]
            earth_lon = earth_lon[start_idx:stop_idx]
            livetime = livetime[start_idx:stop_idx]

        xpointings = SkyCoord(l = lon_x, b = lat_x,
                              frame = Galactic(),
                              copy = False)
        zpointings = SkyCoord(l = lon_z, b = lat_z,
                              frame = Galactic(),
                              copy = False)

        attitude = Attitude.from_axes(x = xpointings, z = zpointings,
                                      frame = Galactic())

        gcrs = cls._earth_zenith_altitude_to_gcrs(earth_lon,
                                                  earth_lat,
                                                  altitude)

        # The last element is 0.
        livetime = livetime[:-1]

        return cls(time_stamps, attitude, gcrs, livetime)

    @staticmethod
    def _earth_zenith_altitude_to_gcrs(earth_lon: Quantity,
                                       earth_lat: Quantity,
                                       altitude: Quantity) -> GCRS:
        """
        Convert galactic latitude and longitude plus altitude w/r to
        earth to a standard GCRS coordinate.

        Parameters
        ----------
        earth_lon: Quantity
           galactic longitude of the direction the Earth's zenith is
           pointing to at the SC location (deg)
        earth_lat: Quantity
           galactic latitude of the direction the Earth's zenith is
           pointing to at the SC location (deg)
        altitude: Quantity
            altitude above from Earth's ellipsoid (km)

        Returns
        -------
        GCRS location

        """

        # Currently, the orbit information is in a weird format.  The
        # altitude is specified with respect to the Earth's surface,
        # like you would specify it in a geodetic format, while the
        # lon/lat is specified in J2000, like you would in ECI.
        # Eventually everything should be in ECI (GCRS in astropy for
        # all practical purposes), but for now let's do the
        # conversion.
        #
        # 1. Get the direction in galactic
        # 2. Transform to GCRS, which uses RA/Dec (ICRS-like).  This
        #    is represented in the unit sphere
        # 3. Add the altitude by transforming to EarthLocation.
        #    Should take care of the non-spherical Earth
        # 4. Go back GCRS, now with the correct distance (from the
        #    Earth's center)

        # Make the distance very far away such that the parallax
        # between earth-centered and barycenter doesn't matter
        zenith_gal = SkyCoord(l = earth_lon, b = earth_lat,
                              distance = SpacecraftHistory._far_dist,
                              frame = Galactic(),
                              copy = False)
        gcrs = zenith_gal.transform_to('gcrs')

        # Use EarthLocation to transform from altitude to the distance
        # from the earth-center given the correct Earth ellipsoid
        itrs = gcrs.transform_to(ITRS(obstime='J2000'))
        earth_loc = itrs.earth_location.geodetic
        earth_loc = EarthLocation.from_geodetic(earth_loc.lon, earth_loc.lat,
                                                height = altitude)

        # Combine RA/Dec from far field, with distance from geodetic
        gcrs2 = GCRS(ra=gcrs.ra, dec=gcrs.dec,
                     distance=earth_loc.itrs.cartesian.norm(),
                     copy=False)
        return gcrs2

    @staticmethod
    def _gcrs_to_earth_zenith_altitude(gcrs : GCRS) -> (Quantity, Quantity, Quantity):
        """
        Extract a galactic-frame earth pointing and altitude from a GCRS
        coordinate.

        Parameters
        ----------
        GCRS location

        Returns
        -------
        earth_lon: Quantity
           galactic longitude of the direction the Earth's zenith is
           pointing to at the SC location (deg)
        earth_lat: Quantity
           galactic latitude of the direction the Earth's zenith is
           pointing to at the SC location (deg)
        altitude: Quantity
            altitude above from Earth's ellipsoid (km)

        """

        # Make it far field o get galactic coordinates
        gcrs_far = GCRS(ra = gcrs.ra, dec = gcrs.dec,
                        distance = SpacecraftHistory._far_dist,
                        copy = False)
        zenith_gal = gcrs_far.transform_to(Galactic())

        # Get the distance from the center of the Earth to the
        # ellipsoid at this specific direction
        itrs = gcrs_far.transform_to(ITRS(obstime='J2000'))
        earth_loc = itrs.earth_location.geodetic
        earth_loc = EarthLocation.from_geodetic(earth_loc.lon, earth_loc.lat,
                                                height = 0*u.km)
        altitude = (gcrs.distance - \
                    earth_loc.itrs.cartesian.norm()).to(u.km, copy=False)

        lon = zenith_gal.l.to(u.deg, copy=False)
        lat = zenith_gal.b.to(u.deg, copy=False)
        return lon, lat, altitude

    @staticmethod
    def _interp_location(t, d1, d2):
        """
        Compute a direction that linearly interpolates between
        directions d1 and d2 using SLERP.

        The two directions are assumed to have the same frame,
        which is also used for the interpolated result.

        Parameters
        ----------
        t : float in [0, 1]
          interpolation fraction
        d1 : GCRS
          1st direction
        d2 : GCRS
          2nd direction

        Returns
        -------
        SkyCoord: interpolated direction

        """

        # NB: do NOT compare GCRS objects d1 and d2 directly for
        # Astropy performance reasons -- convert them to vectors
        # first!

        v1 = d1.cartesian.xyz
        v2 = d2.cartesian.xyz

        if np.all(v1 == v2):
            return d1

        # Keep the same representation, so they can be concatenated
        vi = CartesianRepresentation(SpacecraftHistory._slerp(t, v1, v2), copy=False)
        return GCRS(vi.represent_as(d1.data.__class__), copy=False)

    @staticmethod
    def _interp_earth_location(t, l1, l2):
        """
        Same as _interp_location, but for EarthLocation
        """

        v1 = Quantity(l1.geocentric)
        v2 = Quantity(l2.geocentric)

        if np.all(v1 == v2):
            return l1

        return EarthLocation.from_geocentric(*SpacecraftHistory._slerp(t, v1, v2))

    @staticmethod
    def _slerp(t, v1, v2):
        """
        SLERP between Cartesian vectors v1 and v2, with shape (3, ...)
        """

        # angle between v1, v2
        norm = np.linalg.norm(v1, axis=0) * np.linalg.norm(v2, axis=0)
        theta = np.arccos(np.einsum('i...,i...->...', v1, v2)/norm)

        den = np.sin(theta)
        return (np.sin((1 - t) * theta) * v1 + np.sin(t * theta) * v2) / den

    @staticmethod
    def _interp_attitude(t, att1, att2):
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
        theta = 2 * np.arccos(np.einsum('i...,i...->...',
                                        p1.transpose(),
                                        p2.transpose()))

        # Makes it work with scalars or any input shape
        t = t[..., np.newaxis]
        theta = theta[..., np.newaxis]

        # SLERP interpolated quaternion
        den = np.sin(theta)
        pi = (np.sin((1 - t) * theta) * p1 + np.sin(t * theta) * p2) / den

        return Attitude.from_quat(pi, frame=att1.frame)

    def interp_attitude(self, time) -> Attitude:
        points, weights = self.interp_weights(time)

        return self.__class__._interp_attitude(weights[1],
                                               self._attitude[points[0]],
                                               self._attitude[points[1]])

    def interp_location(self, time) -> GCRS:
        points, weights = self.interp_weights(time)

        return self.__class__._interp_location(weights[1],
                                               self._gcrs[points[0]],
                                               self._gcrs[points[1]])

    def _cumulative_livetime(self, points, weights) -> u.Quantity:
        cum_livetime_discrete = np.append(0 * self._livetime.unit,
                                          np.cumsum(self._livetime))

        up_to_tstart = cum_livetime_discrete[points[0]]

        within_bin = self.livetime[points[0]] * weights[1]

        cum_livetime = up_to_tstart + within_bin

        return cum_livetime

    def cumulative_livetime(self, time: Optional[Time] = None) -> u.Quantity:
        """
        Get the cumulative live obstime up to this obstime.

        The live obstime in between the internal timestamp is assumed
        constant.

        All by edfault

        Parameters
        ----------
        time:
            Timestamps

        Returns
        -------
        Cummulative live obstime, with units.

        """

        if time is None:
            # All
            return np.sum(self.livetime)

        points, weights = self.interp_weights(time)

        return self._cumulative_livetime(points, weights)

    def interp_weights(self, times: Time):
        times = self._get_dt_jd(times)
        return self._time_axis.interp_weights_edges(times)

    def interp(self, times: Time) -> 'SpacecraftHistory':

        """
        Linearly interpolates attitude and position at a given obstime

        Parameters
        ----------
        times:
            Timestamps to interpolate

        Returns
        -------

        A new SpacecraftHistory object interpolated at these location

        """

        if times.size < 2:
            raise ValueError("We need at least two obstime stamps. See also interp_attitude and inter_location")

        points, weights = self.interp_weights(times)

        interp_attitude = self._interp_attitude(weights[1],
                                                self._attitude[points[0]],
                                                self._attitude[points[1]])
        interp_location = self._interp_location(weights[1],
                                                self._gcrs[points[0]],
                                                self._gcrs[points[1]])

        cum_livetime = self._cumulative_livetime(points, weights)
        diff_livetime = cum_livetime[1:] - cum_livetime[:-1]

        interp_earth_location = None
        if self._earth_location is not None:
            interp_earth_location = self._interp_earth_location(weights[1],
                                                                self._earth_location[points[0]],
                                                                self._earth_location[points[1]])

        return self.__class__(times, interp_attitude, interp_location,
                              diff_livetime, interp_earth_location)

    def select_interval(self, start:Time = None, stop:Time = None) -> "SpacecraftHistory":
        """
        Returns the SpacecraftHistory file class object for the source
        interval.

        Parameters
        ----------
        start : astropy.time.Time
            The start obstime of the orientation period. Start of
            history by default.
        stop : astropy.time.Time
            The end obstime of the orientation period. End of history
            by default.

        Returns
        -------
        cosipy.spacecraft.SpacecraftHistory

        """

        from scipy.spatial.transform import Rotation as R

        if start is None:
            start = self.tstart

        if stop is None:
            stop = self.tstop

        if start < self.tstart or stop > self.tstop:
            raise ValueError(f"Input range ({start}-{stop}) is outside the SC history ({self.tstart}-{self.tstop})")

        points, weights = self.interp_weights(Time((start, stop)))
        start_points = points[:,0]
        stop_points  = points[:,1]
        start_weights = weights[:,0]
        stop_weights  = weights[:,1]

        # Center values
        new_obstime = self.obstime[start_points[1]:stop_points[1]]
        new_attitude = [ self._attitude[start_points[1]:stop_points[1]].rot ]
        new_location = self._gcrs[start_points[1]:stop_points[1]]
        new_livetime = self.livetime[start_points[1]:stop_points[0]]

        # Only carried over if already available
        earth_location = self._earth_location
        if earth_location is not None:
            new_earth_location = earth_location[start_points[1]:stop_points[1]]

        # Left edge
        # new_obstime.size can be zero if the requested interval fell
        # completely an existing interval
        if new_obstime.size == 0 or new_obstime[0] != start:
            # Left edge might be included already

            new_obstime = Time(np.append(start.jd1, new_obstime.jd1),
                               np.append(start.jd2, new_obstime.jd2),
                               format = 'jd')

            start_attitude = self._interp_attitude(start_weights[1],
                                                   self._attitude[start_points[0]],
                                                   self._attitude[start_points[1]])

            new_attitude.insert(0, start_attitude.rot)

            start_location = self._interp_location(start_weights[1],
                                                   self._gcrs[start_points[0]],
                                                   self._gcrs[start_points[1]])[None]

            new_location = np.concatenate((start_location, new_location))

            if earth_location is not None:
                start_earth_location = self._interp_earth_location(start_weights[1],
                                                                   earth_location[start_points[0]],
                                                                   earth_location[start_points[1]])
                new_earth_location = np.concatenate((start_earth_location[None], new_earth_location))

            first_livetime = self.livetime[start_points[0]] * start_weights[0]
            new_livetime = np.append(first_livetime, new_livetime)

        # Right edge
        # It's never included, since stop <=
        # self.obstime[stop_points[1]], and the selection above
        # excludes stop_points[1]
        new_obstime = Time(np.append(new_obstime.jd1, stop.jd1),
                           np.append(new_obstime.jd2, stop.jd2),
                           format='jd')

        stop_attitude = self._interp_attitude(stop_weights[1],
                                              self._attitude[stop_points[0]],
                                              self._attitude[stop_points[1]])
        new_attitude.append(stop_attitude.rot)

        new_attitude = Attitude(R.concatenate(new_attitude),
                                frame=self._attitude.frame)

        stop_location = self._interp_location(stop_weights[1],
                                              self._gcrs[stop_points[0]],
                                              self._gcrs[stop_points[1]])[None]

        new_location = np.concatenate((new_location, stop_location))

        if earth_location is not None:
            stop_earth_location = self._interp_earth_location(stop_weights[1],
                                                              earth_location[stop_points[0]],
                                                              earth_location[stop_points[1]])
            new_earth_location = np.concatenate((new_earth_location, stop_earth_location[None]))
        else:
            new_earth_location = None

        if np.all(start_points == stop_points):
            # This can only happen if the requested interval fell
            # completely an existing interval
            new_livetime[0] -= self.livetime[stop_points[0]]*stop_weights[0]
        else:
            last_livetime = self.livetime[stop_points[0]]*stop_weights[1]
            new_livetime = np.append(new_livetime, last_livetime)

        # We used the internal jd1 and jd2 values, which might have
        # changed the format.  Bring it back
        new_obstime.format = self.obstime.format

        new_history = self.__class__(new_obstime, new_attitude, new_location,
                                     new_livetime, new_earth_location)

        # make sure new object uses same earth occ caching as old object
        new_history.cache_earth_occ = self.cache_earth_occ

        return new_history

    def apply_gti(self, gti: GoodTimeInterval) -> "SpacecraftHistory":
        """
        Returns the SpacecraftHistory file class object by masking
        livetimes outside the good time interval.

        Parameters
        ----------
        gti: cosipy.event_selection.GoodTimeInterval

        Returns
        -------
        cosipy.spacecraft.SpacecraftHistory

        """
        new_obstime = None
        new_attitude = None
        new_location = None
        new_livetime = None
        new_earth_location = None

        for i, (start, stop) in enumerate(gti):
            _sph = self.select_interval(start, stop)

            _obstime = _sph.obstime
            _attitude = _sph._attitude.as_matrix()
            _location = _sph._gcrs
            _livetime = _sph.livetime
            _earth_location = _sph._earth_location

            if i == 0:
                new_obstime = _obstime
                new_attitude = _attitude
                new_location = _location
                new_livetime = _livetime
                new_earth_location = _earth_location
            else:
                new_obstime = Time(np.append(new_obstime.jd1, _obstime.jd1),
                                   np.append(new_obstime.jd2, _obstime.jd2),
                                   format='jd')
                new_attitude = np.append(new_attitude, _attitude, axis = 0)
                new_location = np.concatenate((new_location, _location))
                new_livetime = np.append(new_livetime, 0 * new_livetime.unit) # assign livetime of zero between GTIs
                new_livetime = np.append(new_livetime, _livetime)
                if new_earth_location is not None:
                    new_earth_location = np.concatenate((new_earth_location, _earth_location))

        # finalizing
        new_attitude = Attitude.from_matrix(new_attitude, frame=self._attitude.frame)
        new_obstime.format = self.obstime.format

        new_history = self.__class__(new_obstime, new_attitude, new_location,
                                     new_livetime, new_earth_location)

        # make sure new object uses same earth occ caching as old object
        new_history.cache_earth_occ = self.cache_earth_occ

        return new_history

    def get_earth_occ(self, target_coord: SkyCoord):
        """
        For each time point, determine whether a given source would be
        occluded by the earth.

        We can cache some source-independent parts of the computation
        to speed up repeated calls to this function.  Use the class
        property cache_earth_occ to control whether caching is
        enabled.

        Parameters
        ----------
        target_coord: SkyCoord
          source direction

        Returns
        -------
        array of bool, True for each time point where source is
        occluded
        """

        source_gcrs = target_coord.transform_to(self.location)
        source_gcrs_vec = source_gcrs.cartesian.xyz.value

        return self._get_earth_occ(source_gcrs_vec)

    def _get_earth_occ(self, src_vec):
        """
        For each time point, determine whether a given source would be
        occluded by the earth.

        We can cache some source-independent parts of the computation
        to speed up repeated calls to this function.  Use the class
        property cache_earth_occ to control whether caching is
        enabled.

        Parameters
        ----------
        src_vec : 3D Cartesian vector
          source direction, assumed to be in GCRS frame

        Returns
        -------
        array of bool, True for each time point where source is
        occluded

        """

        if hasattr(self, "_min_angle_cos"):
            # values for occultation testing have been cached
            min_angle_cos = self._min_angle_cos
            ez_cart = self._ez_cart
        else:
            dist_earth_center = self._gcrs.spherical.distance.km
            if np.any(dist_earth_center < self._r_earth):
                logger.warning("computed altitude for orientation is negative!"
                               "clamping to 0")
                dist_earth_center = np.max(dist_earth_center, self._r_earth)

            # sine of angle between lines through satellite (1) normal
            # to earth and (2) tangent to earth
            sin_earth_angle = self._r_earth / dist_earth_center

            # cosine of maximum unoccluded angle for source w/r to
            # satellite's earth zenith; that is,
            #
            #   cos(pi - asin(sin_earth_angle))
            #
            # Simplify this calculation to avoid arcsin/cos.
            min_angle_cos = -np.sqrt(1 - sin_earth_angle**2)
            ez_cart = self._gcrs.cartesian.xyz.value
            ez_cart /= np.linalg.norm(ez_cart, axis=0)

            if self._cache_earth_occ:
                # cache intermediates in case we need to use them
                # repeatedly.
                self._min_angle_cos = min_angle_cos
                self._ez_cart = ez_cart

        # get time points at which source is occluded by Earth.
        is_occluded = (src_vec @ ez_cart <= min_angle_cos)

        return is_occluded

    """
    Caching for source-independent parts of earth occultation
    calculation, to make it go faster for each new source.
    """

    @property
    def cache_earth_occ(self):
        return self._cache_earth_occ

    @cache_earth_occ.setter
    def cache_earth_occ(self, value):
        if value is False:
            # delete any cached data if present
            if hasattr(self, "_min_angle_cos"):
                del self._min_angle_cos
                del self._ez_cart

        self._cache_earth_occ = value

    def get_source_visibility(self,
                              source:Optional[Union[SkyCoord,
                                                    np.ndarray]] = None) -> Quantity:
        """
        Get the source's visibility to the detector in all time bins.
        Visibility is determined by the spacecraft's livetime (for,
        e.g., SAA passage) and, if requested, by earth occultation of
        the source.

        Parameters
        ----------
        source : SkyCoord or Cartesian 3-vector (ndarray), optional
            Location of the source. If 3-vector, assumed to be in
            GCRS. If not None, returned visibility will account
            for occultation of the source. Otherwise, the full
        livetime is returned.

        Returns
        -------
        A Quantity array with visible time in each bin.

        """

        livetime = self.livetime

        if source is not None:
            if isinstance(source, SkyCoord):
                source = source.transform_to(self._gcrs)
                source = source.cartesian.xyz.value

            # get pointings that are occluded by Earth
            is_occluded = self._get_earth_occ(source)

            # zero out weights of time bins corresponding to occluded
            # pointings.  Assume occlusion at start of bin holds for
            # entire bin.
            lt = np.where(is_occluded[:-1], 0, livetime.value)
            return Quantity(lt, unit=livetime.unit, copy=False)
        else:
            return livetime

    def get_target_in_sc_frame(self, source: SkyCoord) -> SkyCoord:
        """
        Convert a source coordinate to the path of the source in the spacecraft
        local frame.

        Parameters
        ----------
        source:
            The coordinates of the source
        Returns
        -------
        The source path in the spacecraft frame. Each entry corresponds to a timestamp ib obstime
        """

        source_attframe = source.transform_to(self.attitude.frame)
        source_attframe_vec = source_attframe.cartesian.xyz.value

        lon_sc, colat_sc = self._get_target_in_sc_frame(source_attframe_vec)

        return SkyCoord(lon = lon_sc, lat = np.pi/2 - colat_sc, unit = 'rad', frame = SpacecraftFrame())

    def _get_target_in_sc_frame(self, source: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Convert a source coordinate in the inertial frame of the
        SpacecraftHistory to the path of the source in the spacecraft
        local frame.

        Parameters
        ----------
        source : Cartesian 3-vector
            The coordinates of the source
        Returns
        -------
        pair of np.ndarrays
            The source path in the spacecraft frame, as a pair of
            vectors (longitude, co-latitude) in radians

        """

        src_path_cartesian = np.dot(self._attitude.rot.inv().as_matrix(),
                                    source)

        # convert to spherical lon, colat in radians
        lon, colat = self._cart_to_polar(src_path_cartesian)

        # return raw longitude and co-latitude in radians
        return lon, colat

    def get_exposure(self, source:Union[SkyCoord, np.ndarray],
                     base:HealpixBase,
                     interp:Optional[bool] = True,
                     earth_occ:Optional[bool] = True,
                     dtype = np.float64) -> (np.ndarray, np.ndarray):
        """
        Compute the set of exposed HEALPix pixels relative to a
        HealpixBase for an fixed inertial-frame source that is
        present while the spacecraft orientation changes over time.
        Compute exposure weights based on the time that the
        spacecraft spends in each attitude.

        Parameters
        ----------
        source : 3-vector or SkyCoord
           The location of the source; if a 3-vector, it is given
           in the SpacecraftHistory's coordinate frame.
        base : HealpixBase
           HEALPix grid used to discretize exposure
        interp : bool, optional
           If True, interpolate the weights onto the HEALPix grid;
           else, just map to nearest bin. (Default: interpolate)
        earth_occ : bool, optional
           If True, exposure includes only times that the source
           was not occluded by earth. (Default: True)
        dtype : numpy datatype, optional
           Type of returned exposure weights (default: double)

        Returns
        -------
        pixels : np.ndarray (int)
          all HEALPix pixels in the grid with nonzero exposure weight
        exposures: Quantity array (dtype)
          exposure weight for each pixel, in units of spacecraft
          obstime

        """

        if isinstance(source, SkyCoord):
            source = source.transform_to(self._attitude.frame)
            source = source.cartesian.xyz.value

        duration = self.get_source_visibility(source if earth_occ else None)

        # Get source path
        phi, theta = self._get_target_in_sc_frame(source)

        # Remove the last source location (effectively a 0th-order
        # interpolation)
        theta = theta[:-1]
        phi = phi[:-1]

        if interp:
            pixels, weights = base.get_interp_weights(theta=theta,
                                                      phi=phi,
                                                      lonlat=False)
            weighted_duration = weights * duration[None]
        else:
            # do not interpolate
            pixels = base.ang2pix(theta=theta,
                                  phi=phi,
                                  lonlat=False)
            weighted_duration = duration

        unique_pixels, unique_weights = \
            self._sparse_sum_duplicates(pixels.ravel(),
                                        weighted_duration.ravel(),
                                        dtype=dtype)

        return unique_pixels, unique_weights

    def get_dwell_map(self, target_coord:SkyCoord,
                      base:HealpixBase = None,
                      interp:Optional[bool] = True,
                      earth_occ:Optional[bool] = False,
                      nside:Optional[int] = None,
                      scheme:Optional[str] = 'ring') -> HealpixMap:

        """
        Generates the dwell obstime map for a target coordinte. The map
        properties must be specified by setting one of base or
        nside/scheme.  If both are specified, base takes precedence.

        Parameters
        ----------
        target_coord:
            Target coordinate
        base: HealpixBase, optional
            HealpixBase defining the grid. If not specified,
            use nside and scheme instead.
        interp : bool, optional
            If true, interpolate weights onto HEALPix grid;
            else, just map to nearest bin. (Default: interpolate)
        earth_occ : bool, optional
           If True, exposure includes only times that the source
           was not occluded by earth. (Default: False)
        nside: int, optional
            Healpix NSIDE for map
        scheme: int, optional
            Healpix scheme for map

        Returns
        -------
        mhealpy.containers.healpix_map.HealpixMap
            The dwell obstime map.

        """

        if base is None:
            base = HealpixBase(nside=nside, scheme=scheme,
                               coordsys=SpacecraftFrame())

        pixels, weights = self.get_exposure(target_coord, base,
                                            interp=interp,
                                            earth_occ=earth_occ)

        dwell_map = HealpixMap(base = base,
                               unit = weights.unit,
                               coordsys = SpacecraftFrame())

        map_data = dwell_map.data
        map_data[pixels] = weights

        return dwell_map

    def get_scatt_map(self,
                      nside:int,
                      target_coord:Optional[SkyCoord] = None,
                      earth_occ:Optional[bool] = True,
                      angle_nbins:Optional[int] = None) -> SpacecraftAttitudeMap:

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
        angle_nbins : int, ptional
            Number of bins used for the rotvec's angle. If none
            specified, default is 8*nside

        Returns
        -------
        cosipy.spacecraftfile.scatt_map.SpacecraftAttitudeMap
            The spacecraft attitude map.

        """

        source = target_coord

        # compute time that the source is visible per time bin
        duration = self.get_source_visibility(source if earth_occ else None)

        # Convert attitudes from points to time bins.  We use the
        # attitude at the start of the bin as the representative value
        # for the whole bin.
        attitude = self._attitude[:-1]

        # remove time bins in which the source was invisible
        attitude = attitude[duration > 0.]
        duration = duration[duration > 0.]

        # Get orientations as rotation vectors (center dir, angle
        # around center)
        rot_vecs   = attitude.as_rotvec()
        rot_angles = np.linalg.norm(rot_vecs, axis=-1)
        rot_dirs   = rot_vecs / rot_angles[:,None]

        # discretize rotvecs for input Attitudes
        dir_axis = HealpixAxis(nside=nside, coordsys=self._attitude.frame)

        if angle_nbins is None:
            angle_nbins = 8*nside

        angle_axis = Axis(np.linspace(0., 2*np.pi, num=angle_nbins+1),
                          unit=u.rad)

        r_lon, r_colat = self._cart_to_polar(rot_dirs.value)

        dir_bins = dir_axis.find_bin(theta=r_colat,
                                     phi=r_lon)
        angle_bins = angle_axis.find_bin(rot_angles)

        # compute list of unique rotvec bins occurring in input,
        # along with mapping from time to rotvec bin
        shape = (dir_axis.nbins, angle_axis.nbins)

        att_bins = np.ravel_multi_index((dir_bins, angle_bins),
                                        shape)

        # compute the set of unique attitude bins, along with
        # the total weight of each, eliminating any bins
        # with zero weight
        unique_bins, duration = \
            self._sparse_sum_duplicates(att_bins, duration)

        # construct discretized attitudes from the representative
        # rotation vector for each unique bin
        (unique_dirs, unique_angles) = np.unravel_index(unique_bins,
                                                        shape)
        v = dir_axis.pix2vec(unique_dirs)

        angle_centers = angle_axis.centers
        binned_attitudes = \
            Attitude.from_rotvec(np.column_stack(v) *
                                 angle_centers[unique_angles][:,None],
                                 frame = self._attitude.frame)

        return SpacecraftAttitudeMap(binned_attitudes, duration,
                                     source = source if earth_occ else None)

    @staticmethod
    def _sparse_sum_duplicates(indices:np.ndarray,
                               weights:Optional[np.ndarray] = None,
                               dtype:Optional[np.dtype] = None) -> (np.ndarray, np.ndarray):
        """
        Given an array of indices, possibly with duplicates, and an
        optional array of weights per index (defaults to all ones if
        None), return a sorted array of the unique values in indices
        and, for each, the sum of the weights for each unique index.

        If input weights are provided, only unique indices with
        nonzero weights are returned.

        If weights is a Quantity, the unique weights will have the
        same unit.

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

        if len(indices) == 0:
            # cannot pass through bincount -- Numpy gets
            # output type of weights wrong for empty input
            return indices.copy(), None if weights is None else weights.copy()

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
    
    def update_ephemeris(self, ephemeris, intervals):
        """
        Adjust the mission livetime in-place based on a pulsar phase selection.

        Parameters
        ----------
        ephemeris : PhaseEphemeris
            An object adhering to the PhaseEphemeris Protocol.
        intervals : list of tuple of float
            A list of (start_phase, stop_phase) intervals.
        """
        exposed_durations = ephemeris.get_duty_cycle(
            self.intervals_tstart, 
            self.intervals_tstop, 
            intervals
        )
        
        bin_durations = (self.intervals_tstop - self.intervals_tstart).to(u.s)
        
        fraction = np.zeros_like(bin_durations.value)
        valid_bins = bin_durations > 0
        fraction[valid_bins] = (exposed_durations[valid_bins] / bin_durations[valid_bins]).decompose().value
        
        if hasattr(self, '_livetime_hist') and self._livetime_hist is not None:
            # Legacy handling if a histpy object was manually injected
            self._livetime_hist.contents[:] = self._livetime_hist.contents * fraction
        else:
            # Standard cosipy SpacecraftHistory storage natively uses _livetime
            self._livetime = self.livetime * fraction

    def plot_orbit(self, ax=None, saa=True):
        """
        Plot the SC ground track on top of the Earth's coastlines,
        colored by the roll angle. Timestamps followed by zero livetime
        (e.g. SAA passages) are not drawn.

        The roll angle (see ``roll_angle``) is the rotation of the SC
        around its z-axis: the angle between the SC x-axis and the
        direction of motion of the SC, after projecting the latter onto
        the plane perpendicular to the z-axis. It increases
        counter-clockwise when looking from the tip of the z-axis
        towards the SC. A roll angle of 0 means the x-axis points
        forward, along the orbit. It is unwrapped for plotting, so it
        can go beyond the [-180, 180] deg range instead of jumping by
        360 deg.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. A new figure with a Mollweide projection
            is created by default.
        saa : bool or tuple of array-like, optional
            SAA region to draw. If True (default), the contour where
            the magnetic field strength at the Earth's surface is
            28000 nT, according to IGRF-14 at epoch 2028.0. Pass a
            (longitude, latitude) pair to draw a custom polygon, with
            Quantities with angular units or floats in degrees. None or
            False to not draw it.

        Returns
        -------
        matplotlib.axes.Axes
        """

        if ax is None:
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(projection='mollweide')

        # Geographic projections use radians
        to_ax = np.deg2rad if isinstance(ax, GeoAxes) else np.asarray

        for lon, lat in _coastlines():
            ax.plot(to_ax(lon), to_ax(lat), color='0.5', lw=0.5)

        if saa is True:
            saa = _saa_contour()

        if saa is not None and saa is not False:
            lon, lat = (Quantity(c, u.deg).value for c in saa)
            ax.fill(to_ax(lon), to_ax(lat), facecolor='none', edgecolor='C0',
                    hatch='xx', label='SAA')
            ax.legend(loc='lower right')

        location = self.earth_location
        lon = location.lon.wrap_at(180 * u.deg).deg
        lat = location.lat.deg
        roll = np.unwrap(self.roll_angle.to_value(u.deg), period=360)

        # Each timestamp is drawn if the following interval had livetime
        live = np.append(self.livetime.value > 0, False)

        orbit = ax.scatter(to_ax(lon[live]), to_ax(lat[live]), c=roll[live],
                           s=2, lw=0, rasterized=True)
        ax.figure.colorbar(orbit, ax=ax, orientation='horizontal',
                           shrink=0.6, label='Roll angle (deg)')

        ax.grid(True, ls=':', color='0.8')
        ax.set_xlabel('Longitude (deg)')
        ax.set_ylabel('Latitude (deg)')

        return ax

    def plot_pointing(self, time: Time, fov: Quantity = 60 * u.deg,
                      ax=None, nside=128):
        """
        Plot the SC z-axis, field of view and the region occulted by
        the Earth at a given time, in galactic coordinates.

        Parameters
        ----------
        time : Time
            Scalar time within the history.
        fov : Quantity
            Radius of the field of view around the z-axis.
        ax : mhealpy 'mollview' axes, optional
            A new figure is created by default.
        nside : int
            Resolution of the HEALPix maps used to draw the regions.

        Returns
        -------
        astropy.visualization.wcsaxes.WCSAxes
        """

        if ax is None:
            fig = plt.figure(figsize=(10, 5))
            ax = fig.add_subplot(projection='mollview', coord='G')

        zaxis = self.interp_attitude(time).as_axes()[2].transform_to(Galactic())

        location = self.interp_location(time)
        nadir = GCRS(ra=location.ra + 180 * u.deg, dec=-location.dec)
        nadir = nadir.transform_to(Galactic())
        earth_radius = np.arcsin(self._r_earth * u.km / location.distance)

        for center, radius, color in [(nadir, earth_radius, 'C0'),
                                      (zaxis, fov, 'C1')]:
            disc = HealpixMap(nside=nside, coordsys='galactic',
                              data=np.full(12 * nside ** 2, np.nan))
            disc[disc.query_disc(center.cartesian.xyz.value,
                                 radius.to_value(u.rad))] = 1
            disc.plot(ax, cbar=False, cmap=ListedColormap([color]), alpha=0.4)

        ax.scatter(zaxis.l.deg, zaxis.b.deg, transform=ax.get_transform('galactic'),
                   marker='*', s=150, color='C1', edgecolor='k')

        ax.legend(handles=[Patch(color='C1', alpha=0.4, label='Field of view'),
                           Patch(color='C0', alpha=0.4, label='Earth')],
                  loc='lower right')
        ax.coords.grid(True, color='0.8', ls=':')
        ax.set_title(time.isot)

        return ax

    def plot_rocking_angle(self, ax=None):
        """
        Plot the rocking angle (see ``rocking_angle``) as a function of
        time.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. A new figure is created by default.

        Returns
        -------
        matplotlib.axes.Axes
        """

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))

        ax.plot(self.obstime.to_value('datetime64'),
                self.rocking_angle.to_value(u.deg))

        ax.set_xlabel('Time (UTC)')
        ax.set_ylabel('Rocking angle (deg)')

        return ax


def _coastlines():
    """
    Coastlines from Natural Earth (public domain), 1:110m scale.

    Returns
    -------
    list of (lon, lat) arrays, in degrees
    """

    path = Path(__file__).parent / 'data' / 'ne_110m_coastline.geojson'

    with open(path) as f:
        features = json.load(f)['features']

    return [np.transpose(f['geometry']['coordinates']) for f in features]


def _saa_contour():
    """
    Contour of the South Atlantic Anomaly where the magnetic field
    strength at the Earth's surface is 28000 nT, according to
    IGRF-14 at epoch 2028.0. Generated with::

        lon, lat = np.meshgrid(np.arange(-180, 180.01, 0.25), np.arange(-89, 89.01, 0.25))
        B = np.linalg.norm(ppigrf.igrf(lon, lat, 0, datetime(2028, 1, 1)), axis=0)[0]
        contour = plt.contour(lon, lat, B, levels=[28000]).get_paths()[0].vertices[::4]

    Returns
    -------
    (lon, lat) arrays, in degrees
    """

    path = Path(__file__).parent / 'data' / 'saa_igrf14_2028.geojson'

    with open(path) as f:
        polygon = json.load(f)['features'][0]['geometry']['coordinates'][0]

    return np.transpose(polygon)
