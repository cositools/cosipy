import numpy as np
import astropy.units as u

from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.io import fits

from histpy import HealpixAxis

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    # Guard preventing circulat import
    from cosipy import SpacecraftHistory

import logging
logger = logging.getLogger(__name__)


class GoodTimeInterval():

    def __init__(self, tstart_list, tstop_list):
        """
        Initialize GTI object.

        Parameters
        ----------
        tstart_list : astropy.time.Time (array)
            Start times of GTI intervals
        tstop_list : astropy.time.Time (array)
            Stop times of GTI intervals
        """
        # Check that starts and stops are scalar
        if tstart_list.isscalar == True:
            tstart_list = Time([tstart_list])
        if tstop_list.isscalar == True:
            tstop_list = Time([tstop_list])

        # Check that starts and stops have the same length
        if len(tstart_list) != len(tstop_list):
            raise ValueError(f"Length mismatch between starts ({len(tstart_list)}) and stops ({len(tstop_list)})")

        self._tstart_list = tstart_list
        self._tstop_list = tstop_list

        # Sort by start time
        self._sort()

    @property
    def tstart_list(self):
        return self._tstart_list

    @property
    def tstop_list(self):
        return self._tstop_list

    def __len__(self):
        """Return the number of GTI intervals."""
        return len(self._tstart_list)

    def __getitem__(self, index):
        """
        Get GTI interval(s) by index.

        Parameters
        ----------
        index : int, slice, or array-like
            Index, slice, or boolean/integer array to retrieve

        Returns
        -------
        tuple of (Time, Time)
            (tstart_list, tstop_list) for the indexed interval(s)
        """
        return self._tstart_list[index], self._tstop_list[index]

    def __iter__(self):
        """
        Iterate over GTI intervals.

        Yields
        ------
        tuple of (Time, Time)
            Each (start, stop) pair
        """
        for start, stop in zip(self._tstart_list, self._tstop_list):
            yield start, stop

    def _sort(self):
        """
        Sort GTI by start time in ascending order.

        Modifies the GTI in place.
        Stops are sorted according to the start time order.
        """
        sort_idx = np.argsort(self._tstart_list)
        self._tstart_list = self._tstart_list[sort_idx]
        self._tstop_list = self._tstop_list[sort_idx]

    def contains(self, times, time_format='unix', time_scale='utc'):
        """
        Check which times fall inside this GTI.

        Parameters
        ----------
        times : astropy.time.Time or array-like
            Times to test. Numeric inputs are interpreted with
            ``time_format`` and ``time_scale``.
        time_format : str, optional
            Astropy time format for numeric inputs. Default is 'unix'.
        time_scale : str, optional
            Astropy time scale for numeric inputs. Default is 'utc'.

        Returns
        -------
        numpy.ndarray or numpy.bool_
            Boolean mask with the same shape as ``times``. Intervals
            use the half-open convention ``start <= time < stop``.
        """

        input_is_time = isinstance(times, Time)
        if input_is_time:
            time = times
        else:
            time = Time(times, format=time_format, scale=time_scale)

        result_shape = np.shape(time)
        if len(self) == 0:
            return np.zeros(result_shape, dtype=bool)

        flat_time = Time(np.ravel(time.jd1), np.ravel(time.jd2), format='jd',
                         scale=time.scale)

        t0 = self._tstart_list[0]
        relative_time = (flat_time - t0).jd
        relative_starts = (self._tstart_list - t0).jd
        relative_stops = (self._tstop_list - t0).jd

        # Support overlapping intervals without requiring GTIs to be
        # normalized by tracking the latest stop seen at each start.
        cumulative_stops = np.maximum.accumulate(relative_stops)

        interval_idx = np.searchsorted(relative_starts, relative_time,
                                       side='right') - 1
        mask = np.zeros(relative_time.shape, dtype=bool)
        valid_interval = interval_idx >= 0
        mask[valid_interval] = (
            relative_time[valid_interval]
            < cumulative_stops[interval_idx[valid_interval]]
        )

        return mask.reshape(result_shape)

    def save_as_fits(self, filename, overwrite=False, output_format='unix'):
        """
        Save GTI data to a FITS file.

        Parameters
        ----------
        filename : str
            Output FITS filename
        overwrite : bool, optional
            If True, overwrite existing file (default: False)
        output_format : str, optional
            Time format for output (e.g., 'unix', 'mjd'). Default: 'unix'
        """
        # Get values in the specified output format using getattr
        if not hasattr(self._tstart_list, output_format):
            raise ValueError(f"Unsupported output format: {output_format}")

        start_times = getattr(self._tstart_list, output_format)
        stop_times = getattr(self._tstop_list, output_format)

        # Use the scale from the stored Time objects
        output_scale = self._tstart_list.scale

        output_unit = 's'
        if output_format in ['jd', 'mjd']:
            output_unit = 'd'

        # Create primary HDU
        primary_hdu = fits.PrimaryHDU()

        # Define table columns
        col1 = fits.Column(name='TSTART', format='D', unit=output_unit, array=start_times)
        col2 = fits.Column(name='TSTOP', format='D', unit=output_unit, array=stop_times)

        # Create table HDU
        table_hdu = fits.BinTableHDU.from_columns([col1, col2])
        table_hdu.header['EXTNAME'] = 'GTI'
        table_hdu.header['TIMESYS'] = output_scale.upper()
        table_hdu.header['TIMEUNIT'] = output_unit
        table_hdu.header['HIERARCH TIMEFORMAT'] = output_format

        # Create HDUList and write to FITS file
        hdul = fits.HDUList([primary_hdu, table_hdu])
        hdul.writeto(filename, overwrite=overwrite)

    @classmethod
    def from_fits(cls, filename):
        """
        Load GTI from a FITS file.

        Reads time format and scale from FITS header.

        Parameters
        ----------
        filename : str
            Input FITS filename

        Returns
        -------
        GoodTimeIntervals
            GTI object
        """
        infile = fits.open(filename)

        # Search for GTI extension
        gti_hdu = None
        for hdu in infile:
            if isinstance(hdu, fits.BinTableHDU) and hdu.name in ['GTI']:
                gti_hdu = hdu
                break

        if gti_hdu is None:
            infile.close()
            logger.error("GTI table not found in FITS file")

        # Read time system/format from header
        time_scale = gti_hdu.header.get('TIMESYS').lower()
        time_format = gti_hdu.header.get('TIMEFORMAT').lower()

        # Read start and stop times as arrays
        tstart_list = Time(gti_hdu.data['TSTART'], format=time_format, scale=time_scale)
        tstop_list = Time(gti_hdu.data['TSTOP'], format=time_format, scale=time_scale)

        infile.close()
        return cls(tstart_list, tstop_list)

    @classmethod
    def from_pointing_cut(cls,
                          target_coord : SkyCoord,
                          sc_history : 'SpacecraftHistory',
                          max_offaxis : u.Quantity,
                          earth_occ : bool = False):
        """
        Build a GTI where a fixed sky position is within a maximum off-axis angle of the spacecraft boresight.

        Parameters
        ----------
        target_coord : astropy.coordinates.SkyCoord
            Fixed target (source) position.
        sc_history : cosipy.spacecraftfile.SpacecraftHistory
            Spacecraft pointinghistory to evaluate (.ori file).
        max_offaxis : astropy.units.Quantity
            Maximum allowed off-axis angle (FOV).
        earth_occ : bool, optional
            If True, exclude time bins in which the target is occulted
            by the Earth. Default is False.

        Returns
        -------
        GoodTimeInterval
            GTI containing time ranges that satisfy the pointing cut.
        """

        source_sc = sc_history.get_target_in_sc_frame(target_coord)

        colatitude = np.pi/2 - source_sc.lat.to_value(u.rad)

        in_fov = colatitude[:-1] <= max_offaxis.to_value(u.rad)

        if earth_occ:
            occulted = sc_history.get_earth_occ(target_coord)
            in_fov = in_fov & (~occulted[:-1])

        if not np.any(in_fov):
            empty_time = Time([],
                              format=sc_history.intervals_tstart.format,
                              scale=sc_history.intervals_tstart.scale)
            return cls(empty_time, empty_time.copy())

        edges = np.flatnonzero(
            np.diff(np.concatenate(([False], in_fov, [False])))
        )
        start_idx = edges[::2]
        stop_idx = edges[1::2] - 1

        return cls(sc_history.intervals_tstart[start_idx],
                   sc_history.intervals_tstop[stop_idx])

    @classmethod
    def from_region_cut(cls,
                        region: 'Histogram',
                        sc_history: 'SpacecraftHistory',
                        earth_occ: bool = False,
                        earth_occ_mode: str = 'all'):
        """
        Build a GTI where the spacecraft z-pointing is within a given sky region.
 
        Parameters
        ----------
        region : histpy.Histogram
            1-D boolean Histogram with a single HealpixAxis defining the on-region.
            Pixels whose value is True are considered "on-region".
        sc_history : cosipy.spacecraftfile.SpacecraftHistory
            Spacecraft pointinghistory to evaluate (.ori file).
        earth_occ : bool, optional
            If True, exclude time bins in which the target is occulted
            by the Earth. Default is False.
        earth_occ_mode : {'all', 'any'}, optional
            Controls how Earth occultation is evaluated across on-region pixels.
            'all' : *all* on-region pixels must be unoccluded (stricter).
            'any' : *at least one* on-region pixel must be unoccluded (looser).
            Only used when earth_occ=True. Default is 'all'.
 
        Returns
        -------
        GoodTimeInterval
            GTI containing time ranges where the z-pointing condition is satisfied.
        """
 
        if earth_occ and earth_occ_mode not in ('all', 'any'):
            raise ValueError(f"earth_occ_mode must be 'all' or 'any', got '{earth_occ_mode}'")

        if region.ndim != 1 or not isinstance(region.axes[0], HealpixAxis):
            raise ValueError(
                f"region must be a 1D histogram with a HealpixAxis, "
                f"got ndim={region.ndim}"
            )

        region_coordsys = region.axes[0].coordsys
        if region_coordsys is None or region_coordsys.name != 'galactic':
            raise ValueError(
                f"region must be in galactic coordinates, "
                f"got '{getattr(region_coordsys, 'name', region_coordsys)}'"
            )
 
        _, _, z_gal = sc_history.attitude[:-1].transform_to('galactic').as_axes()
 
        z_l = z_gal.l.deg
        z_b = z_gal.b.deg
 
        z_hp_idx = region.axis.ang2pix(z_l, z_b, lonlat=True)
 
        # Bool mask: is the z-pointing pixel inside the on-region?
        in_region = np.asarray(region[z_hp_idx], dtype=bool)
 
        if earth_occ:
            on_pixels = np.flatnonzero(np.asarray(region[:], dtype=bool))
            on_region_coords = region.axis.pix2skycoord(on_pixels)  # shape (n_on_pix,)
 
            occulted = np.array([sc_history.get_earth_occ(coord)
                                 for coord in on_region_coords])
 
            # Align with time bins: use left-edge
            occulted_bins = occulted[:, :-1]
 
            if earth_occ_mode == 'all':
                earth_ok = np.all(~occulted_bins, axis=0)
            else:
                earth_ok = np.any(~occulted_bins, axis=0)
 
            in_region = in_region & earth_ok
 
        if not np.any(in_region):
            empty_time = Time([],
                              format=sc_history.intervals_tstart.format,
                              scale=sc_history.intervals_tstart.scale)
            return cls(empty_time, empty_time.copy())
 
        edges = np.flatnonzero(
            np.diff(np.concatenate(([False], in_region, [False])))
        )
        start_idx = edges[::2]
        stop_idx  = edges[1::2] - 1
 
        return cls(sc_history.intervals_tstart[start_idx],
                   sc_history.intervals_tstop[stop_idx])

    @classmethod
    def intersection(cls, *gti_list):
        """
        Find the intersection of multiple GTI objects.

        Returns a new GTI object containing only time intervals that overlap
        in all input GTI objects.

        Assumes all GTI objects are sorted by start time (guaranteed by _sort() in __init__).

        Parameters
        ----------
        *gti_list : GoodTimeInterval
            Variable number of GTI objects to intersect

        Returns
        -------
        GoodTimeInterval
            New GTI object with intersected intervals

        Examples
        --------
        >>> gti1 = GoodTimeInterval(tstart1, tstop1)
        >>> gti2 = GoodTimeInterval(tstart2, tstop2)
        >>> gti3 = GoodTimeInterval(tstart3, tstop3)
        >>> intersected = GoodTimeInterval.intersection(gti1, gti2, gti3)
        """
        if len(gti_list) == 0:
            raise ValueError("At least one GTI object is required")

        if len(gti_list) == 1:
            # Return a copy of the single GTI
            gti = gti_list[0]
            return cls(gti.tstart_list.copy(), gti.tstop_list.copy())

        # Start with intervals from the first GTI
        current_starts = list(gti_list[0].tstart_list)
        current_stops = list(gti_list[0].tstop_list)

        # Iteratively intersect with each subsequent GTI
        for gti in gti_list[1:]:
            new_starts = []
            new_stops = []

            i = 0  # Index for current intervals
            j = 0  # Index for gti intervals

            # Two-pointer approach for sorted intervals
            while i < len(current_starts) and j < len(gti):
                start1, stop1 = current_starts[i], current_stops[i]
                start2, stop2 = gti[j]

                # Check if intervals overlap
                if start1 < stop2 and start2 < stop1:
                    # Calculate intersection
                    max_start = max(start1, start2)
                    min_stop = min(stop1, stop2)

                    if max_start < min_stop:
                        new_starts.append(max_start)
                        new_stops.append(min_stop)

                # Move the pointer for the interval that ends first
                if stop1 <= stop2:
                    i += 1
                else:
                    j += 1

            # Update current intervals for next iteration
            current_starts = new_starts
            current_stops = new_stops

            # If no overlaps found, we can stop early
            if len(current_starts) == 0:
                break

        # Handle case with no overlapping intervals
        if len(current_starts) == 0:
            # Return empty GTI with appropriate time format
            empty_time = Time([], format=gti_list[0].tstart_list.format,
                             scale=gti_list[0].tstart_list.scale)
            return cls(empty_time, empty_time.copy())

        return cls(Time(current_starts), Time(current_stops))
