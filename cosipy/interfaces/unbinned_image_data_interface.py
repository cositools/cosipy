import logging

import healpy as hp
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from histpy import Axes, Axis, HealpixAxis, Histogram

from cosipy.image_deconvolution.constants import NUMERICAL_ZERO
from cosipy.image_deconvolution.data_interfaces.image_deconvolution_data_interface_base import (
    ImageDeconvolutionDataInterfaceBase,
)
from cosipy.response.photon_types import PhotonListWithDirectionAndEnergyInSCFrame
from cosipy.util.iterables import asarray

logger = logging.getLogger(__name__)

# Unit of the model map, as in AllSkyImageModel.
FLUX_UNIT = u.Unit("1/(cm2 s sr)")


class UnbinnedImageDataInterfaceMono(ImageDeconvolutionDataInterfaceBase):
    """
    Event-by-event data interface for Richardson-Lucy imaging of a gamma-ray
    line.

    Instead of binning the events into a CDS histogram, the response is
    evaluated at each event's own measured coordinates.  The IRF is evaluated
    at a single incident energy, which is exact only for a monochromatic
    source -- hence "Mono".  A continuum band needs the IRF integrated over Ei
    instead, which belongs in a separate interface.

    The likelihood
    --------------
    The data is the list of N = ``n_events`` observed events, each a point
    x_i in the data space (measured energy, Compton data space, time).  A model
    map F predicts an intensity lambda(x) over that space, and its integral

        Lambda = \\int \\lambda(x) dx

    is the number of events the model expects -- a property of the model, not
    of the data.  The unbinned extended log-likelihood is

        ln L = \\sum_i ln \\lambda(x_i) - Lambda.

    For pixel j and event i,

        \\lambda_j(x_i) = F_j dOmega A_eff(j, t_i) P(x_i | j, t_i) e(t_i) V_j(t_i)

    where e is the live fraction and V_j is 0 while the Earth blocks pixel j.
    Lambda integrates the same quantity, so the exposure carries the same
    livetime and Earth occultation as the per-event densities.  Leave either
    out of one side only and the flux scale is biased.

    Carrying Lambda
    ---------------
    Lambda cannot be recovered from the per-event densities, because the
    events are not a quadrature rule for the data space -- summing
    ``expectation`` gives a sum of densities, not an integral.  A binned
    interface gets it for free from ``np.sum(expectation)``.

    The base class hands ``calc_log_likelihood`` only an expectation
    histogram, and the accelerators combine expectations linearly
    (``before + a * (after - before)``), so Lambda has to travel inside the
    expectation.  The data axis therefore has ``n_events + 1`` bins: one per
    event, plus a trailing bin that holds Lambda.  That last bin is model
    bookkeeping, not an event.  It survives everything the algorithms do to an
    expectation because Lambda is linear in the model and in each background
    normalization.

    ``self._event`` holds 1 in every event bin and 0 in the Lambda bin.  The RL
    M-step forms ``event / expectation``, so the ratio is 0 there and the bin
    drops out of every response and background product.

    Scaling
    -------
    The response matrix is dense with shape ``(12 * nside**2, n_events)``, and
    is held in memory for the lifetime of the object.  Choose ``nside`` with
    that in mind.
    """

    def __init__(
        self,
        irf,
        events,
        nside,
        sc_history,
        energy,
        energy_edges,
        exposure_map=None,
        response_matrix=None,
        background_models=None,
        name="UnbinnedImageDataInterfaceMono",
    ):
        """
        Parameters
        ----------
        irf : :py:class:`cosipy.interfaces.FarFieldInstrumentResponseFunctionInterface`
            Instrument response.  Only ``differential_effective_area_cm2`` and
            ``effective_area_cm2`` are used, so any implementation of the
            far-field IRF interface works.
        events : :py:class:`cosipy.interfaces.EmCDSEventDataInSCFrameInterface`
            Time-tagged event list in the spacecraft frame.
        nside : int
            HEALPix nside of the galactic model map (RING scheme).
        sc_history : :py:class:`cosipy.spacecraftfile.SpacecraftHistory`
            Attitude, location and livetime.  Its attitude may be in any
            inertial frame.  Used for the attitude, live fraction and Earth
            occultation at each event's timestamp, and to integrate the
            exposure map when one is not supplied.
        energy : :py:class:`astropy.units.Quantity` or float
            Incident energy of the line (keV if a float).  The IRF is evaluated
            here.
        energy_edges : array-like
            Edges in keV of the model's single incident-energy bin.  They only
            label the model map, which must match the image model's axes, and
            must contain ``energy``.
        exposure_map : array-like, optional
            Pre-computed per-pixel exposure of shape ``(npix,)`` in cm^2 s sr
            (effective area x visible livetime x pixel solid angle).  Computed
            from ``sc_history`` when omitted.
        response_matrix : array-like, optional
            Pre-computed sky response of shape ``(npix, n_events)``.  Building
            it costs one IRF evaluation per pixel over the whole event list, so
            supplying a cached array turns a multi-minute step into a file
            read.  Only its shape is checked: a matrix is valid solely for the
            exact ``nside``, event list, ``energy``, ``sc_history`` window and
            IRF it was built from, and a mismatched one produces wrong fluxes
            rather than an error.  Key any cache on all of those.  Computed
            from ``irf`` when omitted.
        background_models : dict, optional
            Maps a background model name to its per-event contribution.  Each
            value is either an object implementing
            :py:class:`cosipy.interfaces.ExpectationDensityInterface` -- i.e.
            providing ``expectation_density()`` and ``expected_counts()``, as
            :py:class:`FreeNormBackgroundInterpolatedDensityTimeTagEmCDS` and
            :py:class:`FreeNormNFUnbinnedBackground` both do -- or a
            ``(per_event_density, total_expected_counts)`` pair.

            Each model is captured once, at the rate it holds when it is
            registered; the background objects keep that rate in Hz.  The RL
            background normalization is a dimensionless multiplier of it, so
            1.0 reproduces the registered rate.  Use ``background_rate`` to
            convert a fitted normalization back to Hz.
        name : str, optional
            Name of this dataset.
        """
        super().__init__(name)

        self._irf = irf
        self._events = events
        self._sc_history = sc_history

        self._nside = nside
        self._npix = hp.nside2npix(nside)
        self._pixarea = hp.nside2pixarea(nside, degrees=False)  # sr

        self._n_events = events.nevents

        # --- Incident energy ---
        self._energy_keV = u.Quantity(energy, u.keV).to_value(u.keV)
        energy_edges_keV = u.Quantity(energy_edges, u.keV).to_value(u.keV)
        if energy_edges_keV.shape != (2,):
            raise ValueError(
                "energy_edges must hold the two edges of a single bin, got "
                f"{energy_edges_keV.shape[0] - 1} bins"
            )
        if not energy_edges_keV[0] <= self._energy_keV <= energy_edges_keV[1]:
            raise ValueError(
                f"energy {self._energy_keV} keV is outside energy_edges "
                f"{energy_edges_keV.tolist()} keV"
            )

        # --- Axes ---
        self._model_axes = Axes(
            [
                HealpixAxis(nside=nside, scheme="ring", coordsys="galactic", label="lb"),
                Axis(edges=u.Quantity(energy_edges_keV, u.keV), label="Ei", scale="log"),
            ]
        )
        # n_events event bins plus the trailing Lambda bin.
        self._data_axes = Axes(
            [Axis(np.arange(self._n_events + 2), label="event")]
        )
        self._i_norm = self._n_events

        self._event = self._dataspace_histogram(
            density=np.ones(self._n_events), total=0.0
        )

        # --- Pixel directions ---
        # The attitude rotates the spacecraft frame into its own inertial frame,
        # which need not be galactic, so the pixels are expressed in that frame.
        # Earth occultation is tested in GCRS, whose axes match ICRS for
        # far-field directions.
        theta, phi = hp.pix2ang(nside, np.arange(self._npix))
        pixels = SkyCoord(phi, np.pi / 2.0 - theta, unit=u.rad, frame="galactic")
        self._att_vecs = pixels.transform_to(sc_history.attitude.frame).cartesian.xyz.value.T
        self._gcrs_vecs = pixels.transform_to("icrs").cartesian.xyz.value.T

        # --- Spacecraft state at each event ---
        event_times = Time(events.jd1, events.jd2, format="jd")
        attitude = sc_history.interp_attitude(event_times)
        self._rot_to_sc = attitude.rot.inv().as_matrix()  # (n_events, 3, 3)

        # Live fraction of the interval each event falls in.  The exposure
        # integrates livetime, so the per-event density needs the same factor.
        interval_live_fraction = (
            sc_history.livetime / sc_history.intervals_duration
        ).to_value(u.dimensionless_unscaled)
        event_unix = event_times.utc.unix
        interval = np.searchsorted(sc_history.obstime.utc.unix, event_unix, side="right") - 1
        interval = np.clip(interval, 0, len(interval_live_fraction) - 1)
        self._event_live_fraction = interval_live_fraction[interval]

        # Spacecraft location at each distinct event time, for Earth
        # occultation.  SpacecraftHistory needs strictly increasing timestamps,
        # hence the unique.
        unique_unix, self._event_time_index = np.unique(event_unix, return_inverse=True)
        if len(unique_unix) < 2:
            raise ValueError("events must span at least two distinct timestamps")
        self._sc_at_events = sc_history.interp(Time(unique_unix, format="unix", scale="utc"))
        self._sc_at_events.cache_earth_occ = True

        # --- Exposure map ---
        if exposure_map is None:
            self._exposure_map = None  # integrated from sc_history on first use
        else:
            exposure = np.asarray(exposure_map, dtype=float)
            if exposure.shape != (self._npix,):
                raise ValueError(
                    f"exposure_map must have shape ({self._npix},), got {exposure.shape}"
                )
            self._exposure_map = Histogram(
                self._model_axes, contents=exposure.reshape(self._npix, 1)
            )

        # --- Background models ---
        for key, model in (background_models or {}).items():
            self.set_background_model(key, model)

        # --- Response matrix ---
        if response_matrix is None:
            self._response_matrix = None  # built on first use
        else:
            matrix = np.asarray(response_matrix, dtype=float)
            if matrix.shape != (self._npix, self._n_events):
                raise ValueError(
                    f"response_matrix must have shape ({self._npix}, "
                    f"{self._n_events}), got {matrix.shape}"
                )
            self._response_matrix = matrix

    # ------------------------------------------------------------------
    # Data-space helpers
    # ------------------------------------------------------------------

    def _dataspace_histogram(self, density, total):
        """Assemble a data-space histogram from per-event densities and Lambda."""

        contents = np.empty(self._n_events + 1, dtype=float)
        contents[: self._n_events] = density
        contents[self._i_norm] = total

        return Histogram(self._data_axes, contents=contents, copy_contents=False)

    @staticmethod
    def _contents(dataspace_histogram):
        """Per-bin values of a data-space histogram, as a plain float array."""

        if isinstance(dataspace_histogram, Histogram):
            contents = dataspace_histogram.contents
        else:
            contents = dataspace_histogram

        if hasattr(contents, "value"):
            contents = contents.value

        return np.asarray(contents, dtype=float).ravel()

    def _coerce_model(self, model):
        """Model map as a flat array of ``npix`` fluxes in ``FLUX_UNIT``.

        Plain arrays are taken to be in ``FLUX_UNIT`` already.
        """

        contents = model.contents if isinstance(model, Histogram) else model

        if isinstance(contents, u.Quantity):
            contents = contents.to_value(FLUX_UNIT)

        values = np.asarray(contents, dtype=float)
        if isinstance(model, Histogram):
            values = values[:, 0]
        values = values.ravel()

        if values.shape != (self._npix,):
            raise ValueError(
                f"model must have {self._npix} pixels, got {values.shape}"
            )

        return values

    # ------------------------------------------------------------------
    # Response and exposure
    # ------------------------------------------------------------------

    @property
    def response_matrix(self):
        """Sky response of shape ``(npix, n_events)``.

        Element ``(j, i)`` is A_eff(j, t_i) P(event_i | photon from pixel j)
        e(t_i) V_j(t_i), in cm^2 per unit data-space volume.  Times a flux in
        ``FLUX_UNIT`` and the pixel solid angle, it gives counts per second per
        unit data-space volume.
        """

        if self._response_matrix is None:
            self._response_matrix = self._build_response_matrix()

        return self._response_matrix

    def _build_response_matrix(self):

        logger.info(
            "Building the sky response for %d pixels x %d events (%.2f GB)...",
            self._npix,
            self._n_events,
            self._npix * self._n_events * 8 / 1024**3,
        )

        matrix = np.zeros((self._npix, self._n_events), dtype=float)

        for j, (v_att, v_gcrs) in enumerate(zip(self._att_vecs, self._gcrs_vecs)):
            photons = self._photons_from_direction(self._rot_to_sc @ v_att)

            occluded = self._sc_at_events._get_earth_occ(v_gcrs)[self._event_time_index]

            matrix[j, :] = asarray(
                self._irf.differential_effective_area_cm2(photons, self._events),
                np.float64,
            )
            matrix[j, :] *= self._event_live_fraction
            matrix[j, occluded] = 0.0

            if (j + 1) % 100 == 0 or (j + 1) == self._npix:
                logger.info("... %.1f%% complete.", 100.0 * (j + 1) / self._npix)

        return matrix

    def _photons_from_direction(self, v_sc):
        """Photon list arriving from the given directions in the spacecraft frame.

        ``v_sc`` is an array of unit vectors of shape ``(n, 3)``.  The
        (phi, arm, azimuth) transformation is the IRF's business, so only the
        source direction is passed here.
        """

        lon = np.arctan2(v_sc[:, 1], v_sc[:, 0])
        lat = np.pi / 2.0 - np.arccos(np.clip(v_sc[:, 2], -1.0, 1.0))

        energy = np.full(len(v_sc), self._energy_keV, dtype=float)

        return PhotonListWithDirectionAndEnergyInSCFrame(lon, lat, energy)

    @property
    def exposure_map(self):
        """Per-pixel exposure in cm^2 s sr, with ``self.model_axes``."""

        if self._exposure_map is None:
            self._exposure_map = self._build_exposure_map()

        return self._exposure_map

    def _exposure_per_pixel(self):
        contents = self.exposure_map.contents
        if hasattr(contents, "value"):
            contents = contents.value

        return np.asarray(contents, dtype=float)[:, 0]

    def _build_exposure_map(self, n_time_samples=5000):
        """Integrate the effective area over the spacecraft history.

        Each interval contributes its livetime, or nothing while the Earth
        blocks the pixel.  The pixel solid angle is folded in so that the RL
        M-step denominator matches the ``pixarea`` factor applied in
        ``calc_source_expectation`` and ``calc_T_product`` -- model fluxes are
        per steradian.  This is the same convention as
        ``DataIF_COSI_DC2._calc_exposure_map``.

        Parameters
        ----------
        n_time_samples : int, optional
            Maximum number of attitudes to sample from the history.

        Returns
        -------
        :py:class:`histpy.Histogram`
        """

        logger.info("Integrating the exposure map over the spacecraft history...")

        sc = self._sc_history
        n_intervals = sc.nintervals

        # Each interval is represented by the attitude at its start.
        rot = sc.attitude.rot.inv()

        if n_intervals > n_time_samples:
            # Split the history into contiguous chunks and weight each sampled
            # attitude by the summed visible livetime of its chunk.  This keeps
            # dead time and occultation (an SAA pass, a source setting) exact
            # instead of smearing them uniformly over the orbit.
            edges = np.linspace(0, n_intervals, n_time_samples + 1, dtype=int)
            chunk_starts = edges[:-1]
            # Index the rotation before expanding it: three months of 1 s bins is
            # ~8e6 intervals, and as_matrix() over all of them costs ~0.5 GB only
            # to discard all but n_time_samples of the rows.
            rot_intervals = rot[(edges[:-1] + edges[1:]) // 2].as_matrix()
        else:
            chunk_starts = None
            rot_intervals = rot.as_matrix()[:-1]

        n_t = len(rot_intervals)

        # v_sc[t, j] = rot_intervals[t] @ att_vecs[j]
        v_sc = np.einsum("tik,jk->tji", rot_intervals, self._att_vecs)
        photons = self._photons_from_direction(v_sc.reshape(n_t * self._npix, 3))

        aeff = asarray(self._irf.effective_area_cm2(photons), np.float64)
        aeff = aeff.reshape(n_t, self._npix)

        # Occultation depends on the pixel, so the visible livetime is computed
        # one pixel at a time rather than as an (npix, n_intervals) array.
        exposure = np.empty(self._npix, dtype=float)
        cache_earth_occ = sc.cache_earth_occ
        sc.cache_earth_occ = True
        try:
            for j, v_gcrs in enumerate(self._gcrs_vecs):
                visible = sc.get_source_visibility(v_gcrs).to_value(u.s)
                if chunk_starts is not None:
                    visible = np.add.reduceat(visible, chunk_starts)
                exposure[j] = aeff[:, j] @ visible  # cm^2 s
        finally:
            sc.cache_earth_occ = cache_earth_occ

        exposure *= self._pixarea  # cm^2 s sr

        return Histogram(
            self._model_axes,
            contents=exposure.reshape(self._npix, 1),
            copy_contents=False,
        )

    # ------------------------------------------------------------------
    # Background models
    # ------------------------------------------------------------------

    def set_background_model(self, key, model):
        """Register (or replace) a background model.

        Parameters
        ----------
        key : str
        model : :py:class:`cosipy.interfaces.ExpectationDensityInterface` or tuple
            Either an object providing ``expectation_density()`` and
            ``expected_counts()``, or a ``(per_event_density, total)`` pair.
        """

        if hasattr(model, "expectation_density"):
            density = asarray(model.expectation_density(), np.float64)
            total = float(model.expected_counts())
        else:
            density, total = model
            density = np.asarray(density, dtype=float)
            total = float(total)

        if density.shape != (self._n_events,):
            raise ValueError(
                f"background model '{key}' must have shape ({self._n_events},), "
                f"got {density.shape}"
            )

        self._bkg_models[key] = self._dataspace_histogram(density, total)
        self._summed_bkg_models[key] = total

    def background_rate(self, key, norm=1.0):
        """Rate of a background model at a given RL normalization.

        Parameters
        ----------
        key : str
            Background model name
        norm : float, optional
            RL background normalization, e.g. from
            ``results[-1]["background_normalization"][key]``.

        Returns
        -------
        :py:class:`astropy.units.Quantity`
            Expected background counts per second of livetime, in Hz.
        """

        livetime = self._sc_history.livetime.sum()

        return (norm * self.summed_bkg_model(key) / livetime).to(u.Hz)

    # ------------------------------------------------------------------
    # Interface methods
    # ------------------------------------------------------------------

    def calc_source_expectation(self, model):
        """
        Calculate expected counts from a given model.

        Parameters
        ----------
        model : :py:class:`cosipy.image_deconvolution.AllSkyImageModel`
            Model map

        Returns
        -------
        :py:class:`histpy.Histogram`
            Per-event intensity densities, plus the total expected source
            counts in the Lambda bin.
        """

        flux = self._coerce_model(model)

        density = self.response_matrix.T @ flux * self._pixarea
        density = np.maximum(density, NUMERICAL_ZERO)

        # The integral factorizes: P(x | pixel j) is normalized over the data
        # space, so \int \lambda dx is just exposure . flux.  Summing the
        # per-event densities would not give this.
        total = float(np.dot(self._exposure_per_pixel(), flux))

        return self._dataspace_histogram(density, total)

    def calc_bkg_expectation(self, dict_bkg_norm=None):
        """
        Calculate expected counts from a given set of background normalizations.

        Parameters
        ----------
        dict_bkg_norm : dict, default None
            background normalization for each background model, e.g, {'albedo': 0.95, 'activation': 1.05}

        Returns
        -------
        :py:class:`histpy.Histogram`
        """

        contents = np.zeros(self._n_events + 1, dtype=float)

        for key in self.keys_bkg_models():
            norm = 1.0 if dict_bkg_norm is None else dict_bkg_norm.get(key, 1.0)
            contents += norm * self.bkg_model(key).contents

        return Histogram(self._data_axes, contents=contents, copy_contents=False)

    def calc_T_product(self, dataspace_histogram):
        """
        Calculate the product of the input histogram with the transpose matrix of the response function.

        Parameters
        ----------
        dataspace_histogram: :py:class:`histpy.Histogram`
            Its axes must be the same as self.data_axes

        Returns
        -------
        :py:class:`histpy.Histogram`
            The product with self.model_axes
        """

        values = self._contents(dataspace_histogram)[: self._n_events]

        tprod = self.response_matrix @ values * self._pixarea

        return Histogram(
            self._model_axes,
            contents=tprod.reshape(self._npix, 1),
            copy_contents=False,
        )

    def calc_bkg_model_product(self, key, dataspace_histogram):
        """
        Calculate the product of the input histogram with the background model.

        Parameters
        ----------
        key: str
            Background model name
        dataspace_histogram: :py:class:`histpy.Histogram`
            its axes must be the same as self.data_axes

        Returns
        -------
        float
        """

        values = self._contents(dataspace_histogram)[: self._n_events]
        density = self.bkg_model(key).contents[: self._n_events]

        return float(np.dot(density, values))

    def calc_log_likelihood(self, expectation):
        """
        Calculate the unbinned extended log-likelihood,

            ln L = \\sum_i ln \\lambda(x_i) - Lambda.

        Parameters
        ----------
        expectation : :py:class:`histpy.Histogram`
            Expected count histogram, as returned by ``calc_expectation``.  Its
            Lambda bin supplies the expected total.

        Returns
        -------
        float
            Log-likelihood
        """

        contents = self._contents(expectation)

        density = np.maximum(contents[: self._n_events], NUMERICAL_ZERO)
        expected_total = contents[self._i_norm]

        return float(np.sum(np.log(density)) - expected_total)
