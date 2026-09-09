from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, Tuple

import numpy as np
from astropy import units as u
from astropy.coordinates import spherical_to_cartesian, UnitSphericalRepresentation
from astropy.io.fits import update
from astropy.units import Quantity
from histpy import Histogram, HealpixAxis, Axis
from mhealpy.plot.axes import HealpyAxes

from cosipy.interfaces import EventDataInterface
from cosipy.interfaces.data_interface import EmCDSEventDataInSCFrameInterface
from cosipy.interfaces.event import EmCDSEventInSCFrameInterface
from cosipy.interfaces.instrument_response_interface import FarFieldSpectralInstrumentResponseFunctionInterface
from cosipy.interfaces.photon_parameters import PhotonWithDirectionAndEnergyInSCFrameInterface, PhotonListInterface, \
    PhotonListWithDirectionInSCFrameInterface, PhotonListWithDirectionAndEnergyInSCFrameInterface

import h5py as h5

from scoords import SpacecraftFrame

from cosipy.polarization import PolarizationAxis
from cosipy.response.relative_coordinates import RelativeCDSCoordinates
from cosipy.util.iterables import asarray


class IRFRelativeHistUnpolarized(FarFieldSpectralInstrumentResponseFunctionInterface):
    """
    Histogram-based far-field instrument response function parametrized in
    coordinates relative to the photon and linearly interpolated.

    The response is stored as a 6-dimensional :class:`histpy.Histogram`
    with axes, in order:

    - ``NuLambda`` (:class:`histpy.HealpixAxis`): incoming photon
      direction in the spacecraft frame.
    - ``Ei`` (:class:`histpy.Axis`, units of energy): true incident
      photon energy.
    - ``Epsilon`` (:class:`histpy.Axis`, unitless): fractional energy
      deviation ``(Em - Ei) / Ei`` between the measured energy ``Em``
      and the true energy ``Ei``.
    - ``Phi`` (:class:`histpy.Axis`, units of angle): Compton
      kinematics-derived scattering angle.
    - ``Theta`` (:class:`histpy.Axis`, units of angle): offset of the
      geometric scattering angle (NuLambda to PsiChi) relative to the kinematic one
      (``Theta = phi_geo - phi_kin``).
    - ``Zeta`` (:class:`cosipy.polarization.PolarizationAxis`):
      azimuthal angle of the scattered photon around the source
      direction, referenced to the polarization convention stored in
      the axis.

    On construction the input histogram is validated, its axis units
    are standardized (energies to keV, angles to radians, contents to
    cm^2) and then divided by the phase-space volume of each bin to
    produce a differential effective area with implicit units of
    ``cm^2 / sr / rad / keV``. The polarization convention carried by
    the ``Zeta`` axis is preserved as :attr:`_pol_convention` so it
    can be re-applied when translating event directions into the
    relative coordinate system.

    This class is labelled "Unpolarized" because it does not model a
    dependence of the response on the incident photon polarization
    angle; the ``Zeta`` axis is used purely as the scattered-photon
    azimuth, not as a polarization-input degree of freedom.

    Parameters
    ----------
    irf : histpy.Histogram
        A 6D histogram with the axes described above and contents in
        units equivalent to area (``cm^2``).
    aeff : histpy.Histogram, optional
        A separate 2D histogram, with axes ``['NuLambda', 'Ei']`` and
        contents in units equivalent to area (``cm^2``), used as the
        total effective area instead of projecting it out of ``irf``.
        Its ``NuLambda``/``Ei`` binning does not need to match that of
        ``irf`` -- e.g. it can use a finer grid, since the total
        effective area is typically cheaper to compute/store at higher
        resolution than the full differential response. If not
        provided (the default), the total effective area is obtained
        by projecting ``irf`` onto its own ``NuLambda``/``Ei`` axes, as
        before.
    copy : bool, optional
        If True (default) the input histogram(s) are copied before
        their axes and contents are modified in place. Set to False to
        avoid the copy when the caller no longer needs the original(s).
    nthreads : int, optional
        Number of worker threads used to parallelize
        :meth:`effective_area_cm2`/:meth:`differential_effective_area_cm2`
        over large photon/event lists. Defaults to ``1`` (no thread pool,
        no behavior change from before this parameter existed). The
        underlying ``histpy.Histogram.interp()`` call is vectorized numpy
        (fancy indexing + reduction) that releases the GIL for large
        inputs, so values > 1 can give a real speedup on multi-core
        machines -- but only once there's enough work per thread to be
        worth the fan-out/synchronization cost, see
        ``npoints_parallel_thresh``.
    npoints_parallel_thresh : int, optional
        Minimum number of points per thread for parallelization to be
        used. Below ``nthreads * npoints_parallel_thresh`` points,
        :meth:`effective_area_cm2`/:meth:`differential_effective_area_cm2`
        call ``interp()`` directly, single-threaded, regardless of
        ``nthreads`` -- fanning small workloads out across threads costs
        more than it saves. Defaults to ``1000``; only matters when
        ``nthreads`` > 1.
    """

    event_data_type = EmCDSEventDataInSCFrameInterface
    photon_list_type = PhotonListWithDirectionAndEnergyInSCFrameInterface

    def __init__(self,
                 irf: Histogram,
                 aeff: Histogram = None,
                 copy = True,
                 nthreads = 1,
                 npoints_parallel_thresh = 1000):
        """
        Validate the input histogram(s), standardize their axis units,
        and pre-compute the total and differential effective area used
        at evaluation time.

        See the class docstring for a description of the expected axes
        and units.

        Parameters
        ----------
        irf : histpy.Histogram
            Input response histogram.
        aeff : histpy.Histogram, optional
            Optional separate total effective area histogram. See the
            class docstring.
        copy : bool, optional
            Whether to copy ``irf``/``aeff`` before modifying them.
        nthreads : int, optional
            Number of worker threads for parallel interpolation. See the
            class docstring.
        npoints_parallel_thresh : int, optional
            Minimum points per thread before parallelizing. See the
            class docstring.

        Raises
        ------
        ValueError
            If the histogram contents are not area-equivalent, the
            axis labels do not match the expected sequence, or an
            axis has an unexpected type or units.
        """

        if copy:
            irf = irf.copy()

        # Checks
        if not irf.unit.is_equivalent('cm^2'):
            raise ValueError("IRF contents are expected to have units of area.")

        axes = irf.axes

        if not np.array_equal(axes.labels, ['NuLambda', 'Ei', 'Epsilon', 'Phi', 'Theta', 'Zeta']):
            raise ValueError("IRF axes label must be ['NuLambda', 'Ei', 'Epsilon', 'Phi', 'Theta', 'Zeta']")

        if not isinstance(axes['NuLambda'], HealpixAxis):
            raise ValueError("IRF NuLambda axis is expected to be of HealpixAxis type")

        if axes['Ei'].unit is None or not axes['Ei'].unit.is_equivalent('keV'):
            raise ValueError("Ei axis is expected to have units of energy.")

        if axes['Epsilon'].unit is not None and not axes['Epsilon'].unit.is_equivalent(''):
            raise ValueError("Epsilon axis is expected to be unitless")

        if axes['Phi'].unit is None or not axes['Phi'].unit.is_equivalent('deg'):
            raise ValueError("Phi axis is expected to have units of angle.")

        if axes['Theta'].unit is None or not axes['Theta'].unit.is_equivalent('deg'):
            raise ValueError("Theta axis is expected to have units of angle.")

        if not isinstance(axes['Zeta'], PolarizationAxis):
            raise ValueError("IRF Zeta axis is expected to be of PolarizationAxis type")

        if not isinstance(axes['Zeta'].convention.frame, SpacecraftFrame):
            raise ValueError("IRF Zeta axis polarization convention must be defined in the spacecraft "
                              "frame (e.g. MEGAlibRelativeX/Y/Z or StereographicConvention).")

        # Events are evaluated with Zeta wrapped into [0, 360) deg (see
        # _differential_effective_area_cm2), so the axis must span exactly
        # that range for the wrapped values to land on a valid bin.
        zeta_edges_deg = axes['Zeta'].edges.angle.to_value(u.deg)
        if not (np.isclose(zeta_edges_deg[0], 0) and np.isclose(zeta_edges_deg[-1], 360)):
            raise ValueError("IRF Zeta axis is expected to span the full [0, 360) deg range, got "
                              f"[{zeta_edges_deg[0]}, {zeta_edges_deg[-1]}] deg")

        # Standardize units
        axes['Ei'] = axes['Ei'].to(u.keV, copy = False).to(None, update = False, copy = False)
        axes['Epsilon'] = axes['Epsilon'].to(None, update = False, copy = False)
        axes['Phi'] = axes['Phi'].to(u.rad, copy = False).to(None, update = False, copy = False)
        axes['Theta'] = axes['Theta'].to(u.rad, copy = False).to(None, update = False, copy = False)
        self._pol_convention = axes['Zeta'].convention
        axes['Zeta'] = Axis(axes['Zeta'].edges.angle.to(u.rad).value, label = 'Zeta')

        irf = irf.to(u.cm * u.cm, copy=False).to(None, copy=False, update=False) # To cm2 and remove units

        # Get the total effective area
        if aeff is not None:
            self._tot_aeff = self._standardize_aeff(aeff, copy) # cm^2
        else:
            self._tot_aeff = irf.project('NuLambda','Ei') # cm^2

        # Phase space
        # Final content units will be cm^2/sr/rad/keV
        phi_edges_mesh, arm_edges_mesh, az_edges_mesh = np.meshgrid(axes['Phi'].edges,
                                                                    axes['Theta'].edges,
                                                                    axes['Zeta'].edges, indexing='ij')

        phase_space_cds = RelativeCDSCoordinates.get_relative_cds_phase_space(phi_edges_mesh[:-1, :-1, :-1],
                                                                              phi_edges_mesh[1:, :-1, :-1],
                                                                              arm_edges_mesh[:-1, :-1, :-1],
                                                                              arm_edges_mesh[:-1, 1:, :-1],
                                                                              az_edges_mesh[:-1, :-1, :-1],
                                                                              az_edges_mesh[:-1, :-1, 1:])

        ei_centers_mesh, epsilon_widths_mesh = np.meshgrid(axes['Ei'].centers,
                                                      axes['Epsilon'].widths,
                                                      indexing='ij')

        phase_space_em = ei_centers_mesh * epsilon_widths_mesh

        irf /= axes.expand_dims(phase_space_cds, axes.label_to_index(['Phi', 'Theta', 'Zeta']))
        irf /= axes.expand_dims(phase_space_em, axes.label_to_index(['Ei', 'Epsilon']))

        # Bins in the unphysical region of the CDS reparametrization (Phi +
        # Theta outside [0, pi]) have zero phase space and zero contents,
        # so the divisions above produce 0/0 = NaN there. Replace with the
        # physically correct value of zero differential effective area, so
        # these bins don't poison interpolation for nearby physical events.
        irf[:] = np.nan_to_num(irf.contents, nan = 0.0)

        self._diff_aeff = irf

        # Extra params
        self._nthreads = nthreads
        self._npoints_parallel_thresh = npoints_parallel_thresh
        self._executor = ThreadPoolExecutor(max_workers=nthreads) if nthreads > 1 else None

    def _parallel_interp(self, hist, build_args, raw_arrays):
        """
        Interpolate ``hist`` at the points described by ``raw_arrays``,
        splitting the work across :attr:`_nthreads` worker threads when
        there's enough of it to be worth it.

        ``histpy.Histogram.interp()``'s bottleneck (fancy-indexing the
        contents array, then a stack/reduction, done once per axis-bin
        combination -- see ``Histogram._interp_multilinear``) is
        vectorized numpy that releases the GIL for large inputs, so
        splitting the input points into chunks and interpolating each in
        its own thread gives real wall-clock speedup once there are
        enough points per thread. Below that, the fixed cost of fanning
        work out across threads and gathering results back dominates and
        makes it slower than calling ``interp()`` directly -- hence the
        ``_npoints_parallel_thresh``-based threshold below.

        Parameters
        ----------
        hist : histpy.Histogram
            The histogram to interpolate (``self._tot_aeff`` or
            ``self._diff_aeff``).
        build_args : callable
            Takes the (possibly chunked) raw arrays and returns the
            positional arguments ``hist.interp()`` expects. This is
            where e.g. a raw (lon, lat) pair gets wrapped into an
            ``astropy.coordinates.UnitSphericalRepresentation`` for the
            ``NuLambda`` axis -- cheaply, and once per chunk rather than
            once for the whole input.
        raw_arrays : tuple of numpy.ndarray
            The flat arrays to (maybe) split across threads. All must
            have the same length.

        Returns
        -------
        numpy.ndarray
            The concatenated interpolated values, in the original point
            order.
        """

        n = len(raw_arrays[0])

        if self._executor is None or n < self._nthreads * self._npoints_parallel_thresh:
            return hist.interp(*build_args(*raw_arrays))

        nchunks = min(self._nthreads, max(1, n // self._npoints_parallel_thresh))
        chunks = zip(*(np.array_split(a, nchunks) for a in raw_arrays))
        futures = [self._executor.submit(lambda c=c: hist.interp(*build_args(*c)))
                   for c in chunks]

        return np.concatenate([f.result() for f in futures])

    @staticmethod
    def _standardize_aeff(aeff: Histogram, copy: bool) -> Histogram:
        """
        Validate a standalone total-effective-area histogram (the
        ``aeff`` constructor argument) and standardize its axis units,
        the same way ``irf``'s ``NuLambda``/``Ei`` axes and contents
        are standardized in :meth:`__init__`.

        Parameters
        ----------
        aeff : histpy.Histogram
            2D histogram with axes ``['NuLambda', 'Ei']`` and contents
            in units equivalent to area.
        copy : bool
            Whether to copy ``aeff`` before modifying it.

        Returns
        -------
        histpy.Histogram
            ``aeff`` with its ``Ei`` axis in keV and its contents in
            cm^2, both unitless (implicit units), ready to be
            interpolated on directly.

        Raises
        ------
        ValueError
            If the histogram contents are not area-equivalent, the
            axis labels are not ``['NuLambda', 'Ei']``, or an axis has
            an unexpected type or units.
        """

        if copy:
            aeff = aeff.copy()

        if not aeff.unit.is_equivalent('cm^2'):
            raise ValueError("aeff contents are expected to have units of area.")

        axes = aeff.axes

        if not np.array_equal(axes.labels, ['NuLambda', 'Ei']):
            raise ValueError("aeff axes label must be ['NuLambda', 'Ei']")

        if not isinstance(axes['NuLambda'], HealpixAxis):
            raise ValueError("aeff NuLambda axis is expected to be of HealpixAxis type")

        if axes['Ei'].unit is None or not axes['Ei'].unit.is_equivalent('keV'):
            raise ValueError("aeff Ei axis is expected to have units of energy.")

        axes['Ei'] = axes['Ei'].to(u.keV, copy = False).to(None, update = False, copy = False)

        return aeff.to(u.cm * u.cm, copy=False).to(None, copy=False, update=False)

    @classmethod
    def from_h5(cls, filename, *args, **kwargs):
        """
        Construct an :class:`IRFRelativeHistUnpolarized` from an HDF5
        file that stores the response histogram under the group
        ``"IRF"``.

        If the file also has a group named ``"AEFF"``, it is read as
        well and passed to :meth:`__init__` as the ``aeff`` argument
        (see the class docstring), unless ``aeff`` was already provided
        via ``*args``/``**kwargs``, in which case that value takes
        precedence and the file is not checked for an ``"AEFF"``
        group.

        Parameters
        ----------
        filename : str or path-like
            Path to the HDF5 file containing the response histogram
            (and, optionally, the separate total effective area
            histogram).
        *args, **kwargs
            Extra arguments forwarded verbatim to
            :meth:`__init__` (e.g. ``aeff``, ``copy``, ``nthreads`` or
            ``npoints_parallel_thresh``).

        Returns
        -------
        IRFRelativeHistUnpolarized
            Initialized instance with the histogram(s) loaded from
            disk.
        """

        if 'aeff' not in kwargs and len(args) == 0:
            with h5.File(filename, 'r') as f:
                if 'AEFF' in f:
                    kwargs['aeff'] = Histogram.open(filename, 'AEFF')

        return cls(Histogram.open(filename, "IRF"), *args, **kwargs)

    def _effective_area_cm2(self, photons: PhotonListWithDirectionAndEnergyInSCFrameInterface) -> Iterable[float]:
        """
        Total effective area, in cm^2, for each incident photon.

        Interpolates the ``NuLambda``/``Ei`` projection of the full
        response at the direction and energy of each photon in the
        list. See :meth:`_parallel_interp` for how this is (optionally)
        parallelized over threads.

        Parameters
        ----------
        photons : PhotonListWithDirectionAndEnergyInSCFrameInterface
            Photons to evaluate the effective area on.

        Returns
        -------
        Iterable[float]
            One effective-area value per photon, in cm^2.
        """

        photon_lon_rad, photon_lat_rad, photon_energy_keV = self._photon_list_to_raw_values(photons)

        def build_args(lon_rad, lat_rad, energy_keV):
            photon_dir = UnitSphericalRepresentation(lon=Quantity(lon_rad, 'rad', copy=False),
                                                     lat=Quantity(lat_rad, 'rad', copy=False))
            return photon_dir, energy_keV

        return self._parallel_interp(self._tot_aeff, build_args,
                                     (photon_lon_rad, photon_lat_rad, photon_energy_keV))

    @staticmethod
    def _photon_list_to_raw_values(photons:PhotonListWithDirectionAndEnergyInSCFrameInterface):
        """
        Extract the raw arrays required to evaluate the response from
        a photon list.

        Returned as plain arrays, rather than a prebuilt
        ``UnitSphericalRepresentation``, so callers can split them into
        chunks (e.g. for :meth:`_parallel_interp`) with ``np.array_split``
        before building coordinate objects per chunk.

        Parameters
        ----------
        photons : PhotonListWithDirectionAndEnergyInSCFrameInterface
            Photon list providing spacecraft-frame directions and
            energies.

        Returns
        -------
        photon_lon_rad : numpy.ndarray
            Photon longitudes in the spacecraft frame, in radians.
        photon_lat_rad : numpy.ndarray
            Photon latitudes in the spacecraft frame, in radians.
        photon_energy_keV : numpy.ndarray
            Photon energies in keV, as a float array.
        """

        photon_lon_rad = asarray(photons.direction_lon_rad_sc, float)
        # Clip away floating-point overshoot past the poles (e.g. from
        # float32 downcasting upstream), which Latitude validates strictly
        # against.
        photon_lat_rad = np.clip(asarray(photons.direction_lat_rad_sc, float), -np.pi / 2, np.pi / 2)

        photon_energy_keV = asarray(photons.energy_keV, float)

        return photon_lon_rad, photon_lat_rad, photon_energy_keV

    def _differential_effective_area_cm2(self, photons:PhotonListWithDirectionAndEnergyInSCFrameInterface, events: EmCDSEventDataInSCFrameInterface) -> Iterable[float]:
        """
        Differential effective area, in cm^2 per unit phase-space
        for each (photon, event) pair.

        For each pair the incident photon direction and energy are
        combined with the event's scattered direction, kinematic
        scattering angle and measured energy to build the six
        relative-coordinate arguments consumed by the underlying
        differential-area histogram:

        - ``NuLambda`` : photon direction in the spacecraft frame,
        - ``Ei`` : true photon energy in keV,
        - ``Epsilon`` : fractional energy deviation
          ``(Em - Ei) / Ei``,
        - ``Phi`` : kinematics-derived scattering angle in radians,
        - ``Theta`` : difference between the geometric and kinematic
          scattering angles in radians,
        - ``Zeta`` : azimuthal scattered-photon angle around the
          source direction, in radians, expressed in the stored
          polarization convention.

        Parameters
        ----------
        photons : PhotonListWithDirectionAndEnergyInSCFrameInterface
            True photon directions and energies, one per event.
        events : EmCDSEventDataInSCFrameInterface
            Reconstructed events providing the scattered direction,
            kinematics-derived scattering angle, and measured energy.

        Returns
        -------
        Iterable[float]
            Differential effective area interpolated at each
            (photon, event) pair, in ``cm^2 / sr / rad / keV``.
        """

        photon_lon_rad, photon_lat_rad, photon_energy_keV = self._photon_list_to_raw_values(photons)

        photon_dir = UnitSphericalRepresentation(lon=Quantity(photon_lon_rad, 'rad', copy=False),
                                                 lat=Quantity(photon_lat_rad, 'rad', copy=False))

        psichi_lon_rad = asarray(events.scattered_lon_rad_sc, float)

        # Clip away values outside the range due to floating-point errors since
        # UnitSphericalRepresentation validates strictly
        psichi_lat_rad = np.clip(asarray(events.scattered_lat_rad_sc, float), -np.pi / 2, np.pi / 2)

        psichi_dir = UnitSphericalRepresentation(lon = Quantity(psichi_lon_rad, 'rad', copy = False),
                                                 lat = Quantity(psichi_lat_rad, 'rad', copy = False))

        phi_kin_rad = asarray(events.scattering_angle_rad, float)
        measured_energy_keV = asarray(events.energy_keV, float)

        # Convert to relative coordinates
        epsilon = (measured_energy_keV - photon_energy_keV)/photon_energy_keV

        relcoords = RelativeCDSCoordinates(photon_dir.to_cartesian().xyz, pol_convention=self._pol_convention)
        phi_geo, zeta = relcoords.to_relative(psichi_dir.to_cartesian().xyz)

        phi_geo_rad = phi_geo.to_value(u.rad)

        # RelativeCDSCoordinates.to_relative() returns az in (-pi, pi], but
        # the Zeta axis spans [0, 2*pi) and, once converted to a plain
        # (non-circular) Axis in __init__, clamps rather than wraps
        # out-of-range values. Wrap into [0, 2*pi) so negative az values
        # land on their correct bin instead of being clamped to Zeta = 0.
        zeta_rad = zeta.to_value(u.rad) % (2 * np.pi)

        theta_rad = phi_geo_rad - phi_kin_rad

        # The relative-coordinate math above is already fully vectorized
        # over the whole input and cheap relative to the interpolation
        # below, so only the interp() call itself is (optionally)
        # parallelized -- see _parallel_interp.
        def build_args(lon_rad, lat_rad, energy_keV, eps, phi, theta, zeta_r):
            photon_dir_chunk = UnitSphericalRepresentation(lon=Quantity(lon_rad, 'rad', copy=False),
                                                            lat=Quantity(lat_rad, 'rad', copy=False))
            return photon_dir_chunk, energy_keV, eps, phi, theta, zeta_r

        return self._parallel_interp(self._diff_aeff, build_args,
                                     (photon_lon_rad, photon_lat_rad, photon_energy_keV,
                                      epsilon, phi_kin_rad, theta_rad, zeta_rad))


    def _random_events(self, photons: PhotonListWithDirectionInSCFrameInterface) -> EventDataInterface:
        """
        Not implemented yet; provided to satisfy the
        :class:`FarFieldSpectralInstrumentResponseFunctionInterface`
        contract.
        """
        raise NotImplementedError("random_events not implemented yet.")

