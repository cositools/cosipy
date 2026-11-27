from copy import copy as shallow_copy, deepcopy

from astromodels.sources import Source, ExtendedSource

import numpy as np
import astropy.units as u

from histpy import Axes, Histogram
from scoords import SpacecraftFrame

from cosipy.data_io import SegmentedEmCDSBinnedData
from cosipy.threeml.util import to_linear_polarization

from cosipy.interfaces import BinnedThreeMLSourceResponseInterface

import logging
logger = logging.getLogger(__name__)


__all__ = ["BinnedThreeMLSegmentedExtendedSourceResponse"]


class BinnedThreeMLSegmentedExtendedSourceResponse(BinnedThreeMLSourceResponseInterface):
    """
    Segmented COSI 3ML plugin for extended/diffuse sources.

    Unlike :class:`~cosipy.response.threeml_extended_source_response.BinnedThreeMLExtendedSourceResponse`,
    which folds a sky model through a precomputed, time-integrated
    Galactic-frame :class:`~cosipy.response.ExtendedSourceResponse`
    file, this class builds the expectation directly from a *local*
    (:class:`~cosipy.response.FullDetectorResponse`) detector response
    and a :class:`~cosipy.image_deconvolution.CoordsysConversionMatrix`
    (CCM), which carries the per-segment (e.g. 'Time' or 'ScAtt')
    sky-pixel -> local-pixel dwell-time map. This lets the expectation
    have an extra segment axis, so a spectral fit can account for how
    the exposure to an extended/diffuse source changes with time or
    spacecraft attitude.

    This mirrors the forward-folding done for image deconvolution (see
    ``DataIF_COSI_DC2.calc_source_expectation``), specialized to a
    fixed (astromodels) sky model instead of a free-form image, and
    exposed through the :class:`~cosipy.interfaces.BinnedThreeMLSourceResponseInterface`
    used by the spectral-fit likelihood.

    Notes
    -----
    * `data`'s PsiChi axis must be local (:class:`~scoords.SpacecraftFrame`),
      matching `full_detector_response`'s own PsiChi axis -- the same
      convention used by the ScAtt/Time image-deconvolution tutorials.
    * Polarized sources are not currently supported by this class.
    * For an energy-independent (2D) spatial morphology, the (expensive)
      sky convolution is cached and only the (cheap) spectral refold is
      repeated as spectral parameters change during a fit -- the same
      strategy used by `BinnedThreeMLExtendedSourceResponse`. That cache
      has shape (segment, Ei, Em, Phi, PsiChi): for very fine segment
      binning combined with a fine detector-response binning, this can
      use a non-trivial amount of memory.
    * For a fully energy-dependent (3D) spatial morphology (including
      GALPROP-style templates), there is currently no such caching: the
      full sky+energy convolution is repeated whenever any source
      parameter changes.
    """

    def __init__(self,
                 data: SegmentedEmCDSBinnedData,
                 full_detector_response,
                 coordsys_conv_matrix):
        """
        Parameters
        ----------
        data:
            Segmented binned data (segment axis + 'Em', 'Phi', 'PsiChi',
            with 'PsiChi' in the local/SpacecraftFrame).
        full_detector_response:
            :class:`~cosipy.response.FullDetectorResponse` (local frame).
        coordsys_conv_matrix:
            :class:`~cosipy.image_deconvolution.CoordsysConversionMatrix`
            built from the *same* exposure table (e.g.
            :class:`~cosipy.image_deconvolution.SpacecraftAttitudeExposureTable`
            or :class:`~cosipy.image_deconvolution.TimeBinnedExposureTable`)
            used to bin `data`, so that the segment axis of both line up.
        """

        if not isinstance(data, SegmentedEmCDSBinnedData):
            raise TypeError(f"Wrong data type '{type(data)}', expected {SegmentedEmCDSBinnedData}.")

        if not isinstance(data.axes['PsiChi'].coordsys, SpacecraftFrame):
            raise ValueError("This implementation requires a local (SpacecraftFrame) "
                             "PsiChi axis; use the galactic-frame "
                             "ExtendedSourceResponse workflow instead for a "
                             "non-segmented, inertial-frame fit.")

        segment_label = data.segment_label

        if coordsys_conv_matrix.binning_method != segment_label:
            raise ValueError(f"data's segment axis ('{segment_label}') doesn't match "
                             f"coordsys_conv_matrix.binning_method "
                             f"('{coordsys_conv_matrix.binning_method}').")

        if coordsys_conv_matrix.axes[segment_label].nbins != data.axes[segment_label].nbins:
            raise ValueError(f"data's '{segment_label}' axis has "
                             f"{data.axes[segment_label].nbins} bins, but "
                             f"coordsys_conv_matrix has "
                             f"{coordsys_conv_matrix.axes[segment_label].nbins}.")

        if 'Pol' in full_detector_response.axes.labels:
            raise NotImplementedError("Polarized segmented extended-source response "
                                      "is not currently supported.")

        # Interface inputs
        self._source = None

        # Other implementation inputs
        self._data = data
        self._fdr = full_detector_response
        self._ccm = coordsys_conv_matrix

        # Lazily-loaded dense local-frame detector response
        # (NuLambda, Ei, Em, Phi, PsiChi)
        self._dr = None

        # Cache
        # Prevent unnecessary calculations and new memory allocations
        # See this issue for the caveats of comparing models
        # https://github.com/threeML/threeML/issues/645
        self._last_convolved_source_dict = None
        self._expectation = None

        # The expensive sky convolution for an energy-independent (2D)
        # spatial morphology is cached separately, since it's
        # independent of all spectral parameters.
        self._last_spatial_model_dict = None
        self._spatial_response = None

    @property
    def axes(self) -> Axes:
        return self._data.axes

    def clear_cache(self):

        self._last_convolved_source_dict = None
        self._expectation = None

        self._last_spatial_model_dict = None
        self._spatial_response = None

    def copy(self) -> "BinnedThreeMLSegmentedExtendedSourceResponse":
        """
        Safe copy to use for multiple sources
        Returns
        -------
        A copy than can be used safely to convolve another source
        """
        new = shallow_copy(self)
        new.clear_cache()
        return new

    def set_source(self, source: Source):
        """
        The source is passed as a reference and its parameters can
        change. Remember to check if it changed since the last time
        the user called expectation.

        """
        if not isinstance(source, ExtendedSource):
            raise TypeError("I only know how to handle extended sources!")

        polarization = to_linear_polarization(source.spectrum.main.polarization)

        if polarization.degree.value != 0:
            raise RuntimeError("This response can't handle a polarized source.")

        self._source = source

    def _compute_spatial_response(self, source: ExtendedSource) -> Histogram:
        """
        Convolve a 2D (energy-independent) spatial morphology with the
        CCM and the local detector response, while retaining Ei, so
        the (expensive) result can be cached and cheaply re-folded with
        different spectra.

        Returns
        -------
        :py:class:`histpy.Histogram`
            Axes: (<segment>, Ei, Em, Phi, PsiChi).
        """

        from cosipy.util import tensordot_sparse

        lb_axis = self._ccm.axes['lb']

        l, b = lb_axis.pix2ang(np.arange(lb_axis.npix), lonlat=True)

        # Same spatial normalization used by get_integrated_extended_model()
        normalized_map = source.spatial_shape(l, b) / u.sr

        lb_index = self._ccm.axes.label_to_index('lb')

        # (<segment>, lb, NuLambda) x (lb,) -> (<segment>, NuLambda)
        sky_dwell = tensordot_sparse(self._ccm.contents, self._ccm.unit,
                                     normalized_map,
                                     axes=(lb_index, 0))

        sky_dwell = sky_dwell * lb_axis.pixarea()

        dr_axes = self._dr.axes
        nulambda_index = dr_axes.label_to_index('NuLambda')

        # (<segment>, NuLambda) x (NuLambda, Ei, Em, Phi, PsiChi)
        #   -> (<segment>, Ei, Em, Phi, PsiChi)
        contents = np.tensordot(sky_dwell.value, np.asarray(self._dr.contents),
                                axes=(1, nulambda_index))

        remaining_labels = [lbl for lbl in dr_axes.labels if lbl != 'NuLambda']
        segment_axis = self._ccm.axes[self._ccm.binning_method]

        result_axes = Axes([segment_axis] + [dr_axes[lbl] for lbl in remaining_labels],
                           copy_axes=False)

        return Histogram(result_axes, contents=contents,
                         unit=sky_dwell.unit * self._dr.unit,
                         copy_contents=False)

    def _fold_spectrum(self, source: ExtendedSource) -> Histogram:
        """
        Fold a spectrum through the cached spatial response.

        Returns
        -------
        :py:class:`histpy.Histogram`
            Axes: (<segment>, Em, Phi, PsiChi).
        """

        from .functions import get_integrated_spectral_model

        energy_axis = self._spatial_response.axes['Ei']

        integrated_flux = get_integrated_spectral_model(spectrum=source.spectrum.main.shape,
                                                         energy_axis=energy_axis)

        ei_index = self._spatial_response.axes.label_to_index('Ei')

        contents = np.tensordot(integrated_flux.contents, self._spatial_response.contents,
                                axes=(0, ei_index))

        remaining_labels = [lbl for lbl in self._spatial_response.axes.labels if lbl != 'Ei']
        result_axes = Axes([self._spatial_response.axes[lbl] for lbl in remaining_labels],
                           copy_axes=False)

        return Histogram(result_axes, contents=contents,
                         unit=self._spatial_response.unit * integrated_flux.unit,
                         copy_contents=False)

    def _compute_generic_expectation(self, source: ExtendedSource) -> Histogram:
        """
        Full sky+energy convolution for a (possibly energy-dependent)
        spatial morphology, repeated in full whenever the source
        changes -- no spatial/spectral caching.

        Returns
        -------
        :py:class:`histpy.Histogram`
            Axes: (<segment>, Em, Phi, PsiChi).
        """

        from astromodels import Function3D
        from .functions import get_integrated_extended_model
        from .functions_3d import get_integrated_extended_model_3d
        from cosipy.util import tensordot_sparse

        lb_axis = self._ccm.axes['lb']
        energy_axis = self._dr.axes['Ei']

        if isinstance(source.spatial_shape, Function3D):
            allsky_image_model = get_integrated_extended_model_3d(source,
                                                                   image_axis=lb_axis,
                                                                   energy_axis=energy_axis)
        else:
            allsky_image_model = get_integrated_extended_model(source,
                                                                image_axis=lb_axis,
                                                                energy_axis=energy_axis)

        lb_index = self._ccm.axes.label_to_index('lb')

        # (<segment>, lb, NuLambda) x (lb, Ei) -> (<segment>, NuLambda, Ei)
        map_rotated = tensordot_sparse(self._ccm.contents, self._ccm.unit,
                                       allsky_image_model.contents,
                                       axes=(lb_index, 0))

        map_rotated = map_rotated * lb_axis.pixarea()

        dr_axes = self._dr.axes
        nulambda_index = dr_axes.label_to_index('NuLambda')
        ei_index = dr_axes.label_to_index('Ei')

        # map_rotated axes: (<segment>, NuLambda, Ei)
        # (<segment>, NuLambda, Ei) x (NuLambda, Ei, Em, Phi, PsiChi)
        #   -> (<segment>, Em, Phi, PsiChi)
        contents = np.tensordot(map_rotated.value, np.asarray(self._dr.contents),
                                axes=((1, 2), (nulambda_index, ei_index)))

        remaining_labels = [lbl for lbl in dr_axes.labels if lbl not in ('NuLambda', 'Ei')]
        segment_axis = self._ccm.axes[self._ccm.binning_method]

        result_axes = Axes([segment_axis] + [dr_axes[lbl] for lbl in remaining_labels],
                           copy_axes=False)

        return Histogram(result_axes, contents=contents,
                         unit=map_rotated.unit * self._dr.unit,
                         copy_contents=False)

    def expectation(self, copy: bool = True) -> Histogram:

        if self._data is None or self._source is None:
            raise RuntimeError("Call set_source() first.")

        source_dict = self._source.to_dict()

        # Use cached expectation if nothing has changed
        if self._expectation is not None and self._last_convolved_source_dict == source_dict:
            if copy:
                return self._expectation.copy()
            else:
                return self._expectation

        if self._dr is None:
            logger.info("... Loading local detector response ...")
            self._dr = self._fdr.to_dr()
            logger.info("--> done")

        if self._source.spatial_shape.n_dim == 2:

            spatial_model_dict = self._source.spatial_shape.to_dict()

            if self._spatial_response is None or self._last_spatial_model_dict != spatial_model_dict:

                logger.info("... Convolving spatial morphology with segmented "
                            "extended-source response ...")

                self._spatial_response = self._compute_spatial_response(self._source)
                self._last_spatial_model_dict = deepcopy(spatial_model_dict)

                logger.info("--> spatial response convolution done")

            self._expectation = self._fold_spectrum(self._source)

        else:

            self._expectation = self._compute_generic_expectation(self._source)

        # Check if axes match
        if self._data.axes != self._expectation.axes:
            raise ValueError("Currently, the expectation axes must exactly match the "
                             "detector response measurement axes")

        self._last_convolved_source_dict = source_dict

        if copy:
            return self._expectation.copy()
        else:
            return self._expectation
