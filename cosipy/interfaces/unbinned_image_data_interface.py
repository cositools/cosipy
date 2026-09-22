import logging

import healpy as hp
import numpy as np
from astropy import units as u
from astropy.time import Time
from histpy import Axes, Axis, HealpixAxis, Histogram

from cosipy.image_deconvolution.constants import NUMERICAL_ZERO
from cosipy.image_deconvolution.data_interfaces.image_deconvolution_data_interface_base import (
    ImageDeconvolutionDataInterfaceBase,
)
from cosipy.response.photon_types import PhotonListWithDirectionAndEnergyInSCFrame
from cosipy.util.iterables import asarray

logger = logging.getLogger(__name__)


class UnbinnedImageDataInterface(ImageDeconvolutionDataInterfaceBase):
    """
    Event-by-event data interface for Richardson-Lucy image deconvolution.

    Instead of binning the events into a CDS histogram, the response is
    evaluated at each event's own measured coordinates. 

    The data space
    --------------
    The data axis has ``n_events + 1`` bins.  The first ``n_events`` hold one
    event each.  The last one is a normalization bin: it carries the integral
    of the intensity over the whole data space,

        N_tot = \\int \\lambda(x) dx.

    That bin is what makes this interface self-contained.  The unbinned
    extended log-likelihood is

        ln L = \\sum_i ln \\lambda(x_i) - N_tot,

    and N_tot cannot be recovered from the per-event densities, because the
    events are not a quadrature rule for the data space -- summing
    ``expectation`` gives a sum of densities, not an integral.  A binned
    interface gets it for free from ``np.sum(expectation)``.  Carrying it in a
    dedicated bin keeps every expectation histogram self-describing, so
    ``calc_log_likelihood(expectation)`` keeps the base-class signature and no
    caller has to hand the model back to the interface.

    The bookkeeping survives everything the algorithms do to an expectation,
    because both the source and background terms are linear:  source and
    background expectations are added element-wise (so their normalization bins
    add to the total), and a background normalization scales its model (so it
    scales that model's contribution to the total).  An expectation computed
    several iterations ago still carries its own correct N_tot, which is what
    the accelerators need when they compare a trial step against the
    unaccelerated one.

    ``self._event`` holds 1 in every event bin and 0 in the normalization bin.
    The RL M-step forms ``event / expectation``, so the ratio is 0 there and
    the normalization bin drops out of every response and background product.

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
        energy_edges,
        exposure_map=None,
        response_matrix=None,
        background_models=None,
        name="UnbinnedImageDataInterface",
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
            Used to interpolate the attitude at each event's timestamp, and to
            integrate the exposure map when one is not supplied.
        energy_edges : array-like
            Edges in keV of the model's incident-energy axis.  This discretizes
            the *sky model*; each event's measured energy enters the likelihood
            unbinned.  The IRF is evaluated at the arithmetic midpoint of each
            bin.  Only a single bin is supported for now (see below).
        exposure_map : array-like, optional
            Pre-computed per-pixel exposure of shape ``(npix,)`` in cm^2 s sr
            (effective area x livetime x pixel solid angle).  Computed from
            ``sc_history`` when omitted.
        response_matrix : array-like, optional
            Pre-computed sky response of shape ``(npix, n_events)``.  Building
            it costs one IRF evaluation per pixel over the whole event list, so
            supplying a cached array turns a multi-minute step into a file
            read.  Only its shape is checked: a matrix is valid solely for the
            exact ``nside``, event list, ``energy_edges``, ``sc_history``
            window and IRF it was built from, and a mismatched one produces
            wrong fluxes rather than an error.  Key any cache on all of those.
            Computed from ``irf`` when omitted.
        background_models : dict, optional
            Maps a background model name to its per-event contribution.  Each
            value is either an object implementing
            :py:class:`cosipy.interfaces.ExpectationDensityInterface` -- i.e.
            providing ``expectation_density()`` and ``expected_counts()``, as
            :py:class:`FreeNormBackgroundInterpolatedDensityTimeTagEmCDS` and
            :py:class:`FreeNormNFUnbinnedBackground` both do -- or a
            ``(per_event_density, total_expected_counts)`` pair.

            The densities are kept at their absolute scale, so a normalization
            of 1.0 means the model-predicted counts.  That matches the cosipy
            convention and works both with ``RichardsonLucyBasic``, where the
            norm stays at 1.0, and with ``RichardsonLucy``, where it is fitted
            in a range around 1.
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

        # --- Model incident-energy axis ---
        energy_edges_keV = np.asarray(energy_edges, dtype=float)
        energy_mids_keV = 0.5 * (energy_edges_keV[:-1] + energy_edges_keV[1:])
        if len(energy_mids_keV) != 1:
            raise NotImplementedError(
                "UnbinnedImageDataInterface currently supports a single model "
                f"incident-energy bin (got {len(energy_mids_keV)}). The response "
                "matrix has no incident-energy axis."
            )
        # One entry per event so the photon list lines up with the event list.
        # All entries are the same in the single-bin case.
        self._incident_energy_keV = np.full(
            self._n_events, energy_mids_keV[0], dtype=float
        )

        # --- Axes ---
        self._model_axes = Axes(
            [
                HealpixAxis(nside=nside, scheme="ring", coordsys="galactic", label="lb"),
                Axis(edges=u.Quantity(energy_edges_keV, u.keV), label="Ei", scale="log"),
            ]
        )
        # n_events event bins plus the trailing normalization bin.
        self._data_axes = Axes(
            [Axis(np.arange(self._n_events + 2), label="event")]
        )
        self._i_norm = self._n_events

        self._event = self._dataspace_histogram(
            density=np.ones(self._n_events), total=0.0
        )

        # --- Per-event rotation matrices, galactic -> spacecraft frame ---
        event_times = Time(events.jd1, events.jd2, format="jd")
        attitude = sc_history.interp_attitude(event_times)
        self._rot_gal_to_sc = attitude.rot.inv().as_matrix()  # (n_events, 3, 3)

        # All HEALPix pixel directions in galactic coordinates.
        theta, phi = hp.pix2ang(nside, np.arange(self._npix))
        self._gal_vecs = hp.ang2vec(theta, phi)  # (npix, 3)

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
        """Assemble a data-space histogram from per-event densities and their integral."""

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
        """Model map as a flat array of ``npix`` fluxes."""

        if isinstance(model, Histogram):
            contents = model.contents
            if hasattr(contents, "value"):
                contents = contents.value
            values = np.asarray(contents, dtype=float)[:, 0]
        else:
            values = np.asarray(model, dtype=float).ravel()

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

        Element ``(j, i)`` is A_eff(j, t_i) P(event_i | photon from pixel j),
        a density per unit data-space volume.
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

        for j, v_gal in enumerate(self._gal_vecs):
            photons = self._photons_from_direction(self._rot_gal_to_sc @ v_gal)

            matrix[j, :] = asarray(
                self._irf.differential_effective_area_cm2(photons, self._events),
                np.float64,
            )

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

        energy = self._incident_energy_keV
        if len(v_sc) != len(energy):
            energy = np.full(len(v_sc), energy[0], dtype=float)

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

        The pixel solid angle is folded in so that the RL M-step denominator
        matches the ``pixarea`` factor applied in ``calc_source_expectation``
        and ``calc_T_product`` -- model fluxes are per steradian.  This is the
        same convention as ``DataIF_COSI_DC2._calc_exposure_map``.

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
        livetime = sc.livetime.to(u.s).value  # (n_intervals,)
        n_intervals = len(livetime)

        # Each interval is represented by the attitude at its start.
        rot = sc.attitude.rot.inv()

        if n_intervals > n_time_samples:
            # Split the history into contiguous chunks and weight each sampled
            # attitude by the summed livetime of its chunk.  This preserves the
            # total livetime exactly and keeps dead-time structure (an SAA pass,
            # say) from being smeared uniformly over the orbit.
            edges = np.linspace(0, n_intervals, n_time_samples + 1, dtype=int)
            cumulative = np.concatenate(([0.0], np.cumsum(livetime)))
            livetime = cumulative[edges[1:]] - cumulative[edges[:-1]]
            # Index the rotation before expanding it: three months of 1 s bins is
            # ~8e6 intervals, and as_matrix() over all of them costs ~0.5 GB only
            # to discard all but n_time_samples of the rows.
            rot_intervals = rot[(edges[:-1] + edges[1:]) // 2].as_matrix()
        else:
            rot_intervals = rot.as_matrix()[:-1]

        n_t = len(livetime)

        # v_sc[t, j] = rot_intervals[t] @ gal_vecs[j]
        v_sc = np.einsum("tik,jk->tji", rot_intervals, self._gal_vecs)
        photons = self._photons_from_direction(v_sc.reshape(n_t * self._npix, 3))

        aeff = asarray(self._irf.effective_area_cm2(photons), np.float64)
        aeff = aeff.reshape(n_t, self._npix)

        exposure = (aeff * livetime[:, np.newaxis]).sum(axis=0)  # cm^2 s
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
            counts in the normalization bin.
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

            ln L = \\sum_i ln \\lambda(x_i) - N_tot.

        Parameters
        ----------
        expectation : :py:class:`histpy.Histogram`
            Expected count histogram, as returned by ``calc_expectation``.  Its
            normalization bin supplies N_tot.

        Returns
        -------
        float
            Log-likelihood
        """

        contents = self._contents(expectation)

        density = np.maximum(contents[: self._n_events], NUMERICAL_ZERO)
        n_tot = contents[self._i_norm]

        return float(np.sum(np.log(density)) - n_tot)
