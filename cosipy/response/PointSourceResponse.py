from histpy import Histogram
from astropy.coordinates import SkyCoord
from astropy.units import Quantity

from cosipy.polarization.polarization_axis import PolarizationAxis
from cosipy.threeml.util import to_linear_polarization
from mhealpy import HealpixMap
from cosipy.interfaces import BinnedInstrumentResponseInterface, BinnedDataInterface
from histpy import Histogram, Axis, Axes 

import numpy as np
import astropy.units as u
from scoords import Attitude, SpacecraftFrame

from .functions import get_integrated_spectral_model

import logging

from cosipy.spacecraftfile import SpacecraftAttitudeMap
from ..data_io import EmCDSBinnedData, SegmentedEmCDSBinnedData

logger = logging.getLogger(__name__)

class PointSourceResponse(Histogram):
    """
    Handles the multi-dimensional matrix that describes the expected
    response of the instrument for a particular point in the sky.

    Parameters
    ----------
    axes : :py:class:`histpy.Axes`
        Binning information for each variable. The following labels are expected:\n
        - ``Ei``: Real energy
        - ``Em``: Measured energy. Optional
        - ``Phi``: Compton angle. Optional.
        - ``PsiChi``:  Location in the Compton Data Space (HEALPix pixel). Optional.
        - ``SigmaTau``: Electron recoil angle (HEALPix pixel). Optional.
        - ``Dist``: Distance from first interaction. Optional.
    contents : array, :py:class:`astropy.units.Quantity` or :py:class:`sparse.SparseArray`
        Array containing the differential effective area convolved with wht source exposure.
    unit : :py:class:`astropy.units.Unit`, optional
        Physical units, if not specified as part of ``contents``. Units of ``area*time``
        are expected.
    """

    @property
    def photon_energy_axis(self):
        """
        Real energy bins (``Ei``).

        Returns
        -------
        :py:class:`histpy.Axes`
        """

        return self.axes['Ei']

    @property
    def measurement_axes(self):
        # All axes except the true-energy (and, if present, polarization)
        # axes. Normally this is just ('Em', 'Phi', 'PsiChi'), but a
        # segmented response (see from_exposure_table()) also carries an
        # extra segment axis (e.g. 'Time', 'ScAtt', 'Depth') that should
        # pass through get_expectation() untouched.
        excluded = {'Ei', 'Pol'}
        labels = [l for l in self.axes.labels if l not in excluded]
        return self.axes[labels]

    def get_expectation(self, spectrum, polarization=None, flux=None):
        """
        Convolve the response with a spectral (and optionally, polarization) hypothesis to obtain the expected
        excess counts from the source.

        Parameters
        ----------
        spectrum : :py:class:`threeML.Model`
            Spectral hypothesis.
        polarization : 'astromodels.core.polarization.LinearPolarization', optional
            Polarization angle and degree. The angle is assumed to have same convention as point source response.
        flux : 1D Histogram, optional
            Pre-computed integrated flux of spectrum for each bin on Ei axis

        Returns
        -------
        :py:class:`histpy.Histogram`
             Histogram with the expected counts on each analysis bin
        """

        polarization = to_linear_polarization(polarization)

        if 'Pol' in self.axes.labels:

            pol_axis = self.axes['Pol']

            polarization_angle = polarization.angle.value
            polarization_level = polarization.degree.value / 100.

            if polarization_angle == 180.:
                polarization_angle = 0.

            # unpolarized weights
            weights = np.full(pol_axis.nbins, (1. - polarization_level) / pol_axis.nbins)

            # add polarized weights
            if polarization_level != 0:
                polarization_bin_index = pol_axis.find_bin(polarization_angle * u.deg)
                weights[polarization_bin_index] += polarization_level

            weights *= self.axes['Pol'].nbins

            contents = np.tensordot(weights, self.contents, axes=(0, self.axes.label_to_index('Pol')))

        else:

            if polarization.degree.value != 0:
                raise RuntimeError(
                    "Response must have polarization angle axis to include polarization in point source response")

            contents = self.contents


        if flux is None:
            energy_axis = self.photon_energy_axis
            flux = get_integrated_spectral_model(spectrum, energy_axis)

        expectation = np.tensordot(contents, flux.contents, axes=(0, 0))

        # if self is sparse, expectation will be a SparseArray with
        # no units, so set the result's unit explicitly
        hist = Histogram(self.measurement_axes, contents = expectation,
                         unit = self.unit * flux.unit,
                         copy_contents = False)

        if not hist.unit == u.dimensionless_unscaled:
            raise RuntimeError(f"Expectation should be dimensionless, but has units of {(hist.unit)}.")

        return hist

    @classmethod
    def from_dwell_time_map(cls,
                            data:BinnedDataInterface,
                            response: BinnedInstrumentResponseInterface,
                            exposure_map: HealpixMap,
                            energy_axis: Axis,
                            polarization_axis: PolarizationAxis = None
                            ):

        axes = [energy_axis]

        polarization_centers = None
        if polarization_axis is not None:
            axes += [polarization_axis]
            polarization_centers = polarization_axis.centers

        axes += list(data.axes)

        psr = PointSourceResponse(axes, unit=u.cm * u.cm * u.s)

        for p in range(exposure_map.npix):

            coord = exposure_map.pix2skycoord(p)

            if exposure_map[p] != 0:
                psr += response.differential_effective_area(coord, energy_axis.centers, polarization_centers) * exposure_map[p]

        return psr

    @classmethod
    def from_scatt_map(cls,
                        coord: SkyCoord,
                        data:BinnedDataInterface,
                        response: BinnedInstrumentResponseInterface,
                        scatt_map: SpacecraftAttitudeMap,
                        energy_axis: Axis,
                        polarization_axis: PolarizationAxis = None
                        ):
        """

        Parameters
        ----------
        measured_axes
        response
        scatt_map
        energy_axis
        polarization_axis

        Returns
        -------

        """

        if not isinstance(data, EmCDSBinnedData):
            raise TypeError(f"Wrong data type '{type(data)}', expected {EmCDSBinnedData}.")

        axes = [energy_axis]

        if polarization_axis is not None:
            axes += [polarization_axis]

        axes += list(data.axes)
        axes = Axes(axes)

        psr = Quantity(np.zeros(shape=axes.shape), unit = u.cm * u.cm * u.s)

        for att, exposure in zip(scatt_map.attitudes, scatt_map.weights):

            response.differential_effective_area(coord,
                                                 energy_axis.centers,
                                                 None if polarization_axis is None else polarization_axis,
                                                 attitude = att,
                                                 weight=exposure,
                                                 out=psr,
                                                 add_inplace=True)

        return PointSourceResponse(axes, contents = psr)

    @classmethod
    def from_exposure_table(cls,
                            coord: SkyCoord,
                            data: SegmentedEmCDSBinnedData,
                            response: BinnedInstrumentResponseInterface,
                            exposure_table,
                            energy_axis: Axis,
                            polarization_axis: PolarizationAxis = None
                            ):
        """
        Build a *segmented* point-source response: one slice per row of
        an exposure table (e.g. a time or spacecraft-attitude binning),
        instead of a single exposure-integrated response.

        This is the segmented analog of :meth:`from_scatt_map`, used to
        let a spectral fit account for how the response changes with
        time/attitude. Each row of `exposure_table` fills its own slice
        along the segment axis of `data.axes` (the extra axis besides
        'Em', 'Phi' and 'PsiChi'), weighted by that row's livetime and
        using its time-averaged attitude.

        Parameters
        ----------
        coord:
            Source direction. If `data.axes['PsiChi']` is in an
            inertial frame (e.g. galactic), this must be given in that
            frame. If it's in a local (:class:`~scoords.SpacecraftFrame`)
            frame, `coord` must be given in an inertial frame, and it
            will internally be rotated into the spacecraft frame using
            each row's attitude.
        data:
            Segmented binned data (segment axis + 'Em', 'Phi', 'PsiChi').
            The number of bins of the segment axis must match the
            number of rows of `exposure_table`.
        response:
            Instrument response built using the *non-segmented*
            'Em'/'Phi'/'PsiChi' binning, e.g.
            ``BinnedInstrumentResponse(dr, data.to_emcds())``.
        exposure_table:
            Object with one row per segment bin (in the same order as
            the segment axis), exposing `zpointing_averaged` and
            `xpointing_averaged` (galactic [l, b] in degrees) and
            `total_livetime` (seconds) -- e.g. a
            :class:`cosipy.image_deconvolution.SpacecraftAttitudeExposureTable`
            or :class:`cosipy.image_deconvolution.TimeBinnedExposureTable`
            (or anything exposing the same columns via `itertuples()`).
        energy_axis:
            Desired binning of the true photon energy (Ei).
        polarization_axis:
            Desired binning of the photon polarization angle (Pol).

        Returns
        -------
        :py:class:`PointSourceResponse`
            Its axes are [Ei, (Pol,) <segment>, Em, Phi, PsiChi].
        """

        if not isinstance(data, SegmentedEmCDSBinnedData):
            raise TypeError(f"Wrong data type '{type(data)}', expected {SegmentedEmCDSBinnedData}.")

        segment_label = data.segment_label
        n_segments = data.axes[segment_label].nbins

        if n_segments != len(exposure_table):
            raise ValueError(f"The '{segment_label}' axis has {n_segments} bins, "
                             f"but exposure_table has {len(exposure_table)} rows.")

        axes = [energy_axis]

        polarization_centers = None
        if polarization_axis is not None:
            axes += [polarization_axis]
            polarization_centers = polarization_axis.centers

        axes += list(data.axes)
        axes = Axes(axes, copy_axes=False)

        segment_axis_index = axes.label_to_index(segment_label)

        psr = Quantity(np.zeros(shape=axes.shape), unit=u.cm * u.cm * u.s)

        is_local = isinstance(data.axes['PsiChi'].coordsys, SpacecraftFrame)

        for i, row in enumerate(exposure_table.itertuples()):

            z = SkyCoord(*row.zpointing_averaged, frame='galactic', unit=u.deg)
            x = SkyCoord(*row.xpointing_averaged, frame='galactic', unit=u.deg)
            attitude = Attitude.from_axes(x=x, z=z, frame='galactic')

            livetime = Quantity(row.total_livetime, u.s)

            out_index = [slice(None)] * psr.ndim
            out_index[segment_axis_index] = i
            out = psr[tuple(out_index)]

            if is_local:
                # PsiChi is attitude-invariant in the local frame: rotate
                # the (inertial) source direction into the spacecraft
                # frame for this bin's averaged attitude instead.
                local_coord = coord.transform_to(SpacecraftFrame(attitude=attitude))

                response.differential_effective_area(local_coord,
                                                     energy_axis.centers,
                                                     polarization_centers,
                                                     weight=livetime,
                                                     out=out,
                                                     add_inplace=True)
            else:
                response.differential_effective_area(coord,
                                                     energy_axis.centers,
                                                     polarization_centers,
                                                     attitude=attitude,
                                                     weight=livetime,
                                                     out=out,
                                                     add_inplace=True)

        return PointSourceResponse(axes, contents=psr)

