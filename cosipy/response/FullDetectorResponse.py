from pathlib import Path

import numpy as np

import h5py as h5
import hdf5plugin

from astropy.units import Quantity
import astropy.units as u
from astropy.coordinates import SkyCoord

from scoords import SpacecraftFrame, Attitude

from mhealpy import HealpixBase

from histpy import Axes, HealpixAxis

from .DetectorResponse import DetectorResponse
from .PointSourceResponse import PointSourceResponse
from .ExtendedSourceResponse import ExtendedSourceResponse

import logging
logger = logging.getLogger(__name__)


class FullDetectorResponse(HealpixBase):
    """
    Handles the multi-dimensional matrix that describes the
    full all-sky response of the instrument.

    You can access the :py:class:`DetectorResponse` at a given pixel
    using the ``[]`` operator. Alternatively you can obtain the
    interpolated reponse using :py:func:`get_interp_response`.

    """

    # supported HDF5 response version
    rsp_version = 2

    def __init__(self, *args, **kwargs):
        # Overload parent init. Called in class methods.
        pass

    @classmethod
    def open(cls, filename, dtype=None, pa_convention=None):

        """
        Open a detector response file.

        Parameters
        ----------
        filename : str, :py:class:`~pathlib.Path`
             Path to the response file (.h5 or .rsp.gz)

        dtype : numpy dtype or None
             Dtype of values to be returned when accessing response
             contents. If None, use the type stored in the file

        pa_convention : str, optional
            Polarization convention of response ('RelativeX', 'RelativeY', or 'RelativeZ')
        """

        filename = Path(filename)

        if filename.suffix == ".h5":
            return cls._open_h5(filename, dtype, pa_convention)
        else:
            raise ValueError(
                "Unsupported file format. Only .h5 and .rsp.gz extensions are supported.")

    @classmethod
    def _open_h5(cls, filename, dtype=None, pa_convention=None):
        """
         Open a detector response h5 file.

         Parameters
         ----------
         filename : str, :py:class:`~pathlib.Path`
             Path to HDF5 file

        dtype : numpy dtype or None
             Dtype of values to be returned when accessing response
             contents. If None, use the type stored in the file
             (specifically, the type of EFF_AREA)

         pa_convention : str, optional
             Polarization convention of response ('RelativeX', 'RelativeY', or 'RelativeZ')
         """
        new = cls(filename)

        new._file = h5.File(filename, mode='r')

        new._drm = new._file['DRM']

        # verify response format version
        rsp_version = new._drm.attrs.get('VERSION', default=1)
        if rsp_version != cls.rsp_version:
            raise RuntimeError(f"Response format is version {rsp_version}; we require version {cls.rsp_version}")

        new._axes = Axes.open(new._drm["AXES"])

        if new._axes[0].label != "NuLambda":
            raise RuntimeError("Full detector response must have NuLambda as its first dimension")

        # axes minus NuLambda -- used for getting pixel slices for PSRs
        new._rest_axes = new._axes[1:]

        new._unit = u.Unit(new._drm.attrs['UNIT'])

        # effective area for counts
        ea = np.array(new._drm["EFF_AREA"])

        # eff_area type determines return type of __getitem__
        if dtype is not None:
            ea = ea.astype(dtype, copy=False)
        new._eff_area = ea

        # Init HealpixMap (local coordinates, main axis)
        HealpixBase.__init__(new,
                             base=new.axes['NuLambda'],
                             coordsys=SpacecraftFrame())

        new.pa_convention = pa_convention
        if 'Pol' in new._axes.labels and pa_convention not in ('RelativeX', 'RelativeY', 'RelativeZ'):
            raise RuntimeError("Polarization angle convention of response "
                               "('RelativeX', 'RelativeY', or 'RelativeZ') must be provided")

        return new

    @property
    def ndim(self):
        """
        Dimensionality of detector response matrix.

        Returns
        -------
        int
        """

        return self._axes.ndim

    @property
    def shape(self):
        """
        Shape of detector response matrix.

        Returns
        -------
        tuple of axis sizes
        """

        return self._axes.shape

    @property
    def axes(self):
        """
        List of axes.

        Returns
        -------
        :py:class:`histpy.Axes`
        """
        return self._axes

    @property
    def dtype(self):
        """
        Data type returned by to_dt() and slicing

        Returns
        -------
        :py:class:`numpy.dtype`
        """

        return self._eff_area.dtype

    @property
    def unit(self):
        """
        Physical unit of the contents of the detector reponse.

        Returns
        -------
        :py:class:`astropy.units.Unit`
        """

        return self._unit

    @property
    def eff_area(self):
        """
        Effective area of bins with each Ei.

        Returns
        -------
        :py:class:`np.ndarray`
        """

        return self._eff_area

    @property
    def counts(self):
        """
        Raw counts array on disk.

        Returns
        -------
        :py:class:`h5py.dataset`
        """

        return self._drm['COUNTS']

    @property
    def headers(self):
        """
        Headers from original .rsp file

        Returns
        -------
        dict mapping header tags (e.g. "SP") to contents
        """

        # extract the headers in the order that they were written
        hdr_attrs = self._drm["HEADERS"].attrs
        hdr_ids = list(hdr_attrs.keys())
        hdr_order = self._drm.attrs["HEADER_ORDER"]
        hdrs = { hdr_ids[idx] : hdr_attrs[hdr_ids[idx]] for idx in hdr_order }

        return hdrs

    def get_pixel(self, pix, weight=None):
        """
        Extract the portion of the response corresponding to a
        single source sky pixel on the NuLambda axis, optionally
        weighting the result by a given weight.

        Specifying the weight as an argument lets us apply it to the
        eff_area, rather than to the entire slice of counts, for
        greater efficiency.

        Parameters
        ----------
        pix : integer
           pixel index to extract
        weight : optional float or Quantity
           weight to apply to the response slice

        Returns
        -------
        A DetectorResponse containing the specified part of the full
        response.

        """

        data = self._get_pixel_raw(pix, weight)

        unit = self.unit
        if isinstance(weight, Quantity):
            unit *= weight.unit

        return DetectorResponse(self._rest_axes,
                                contents = data,
                                unit = unit,
                                copy_contents = False)

    def _get_pixel_raw(self, pix, weight=None):
        """
        The guts of get_pixel() -- actually loads the data from the HDF
        dataset and performs the multiply.

        Parameters
        ----------
        as for get_pixel()

        Returns
        -------
        data : ndarray of float
           the PSR data

        """

        counts = self._drm['COUNTS'][pix]

        w = self._eff_area

        if weight is not None:
            if isinstance(weight, Quantity):
                w = w * weight.value # don't modify eff_area in place
            else:
                w = w * weight

        data = counts * self._rest_axes.expand_dims(w, self._rest_axes.label_to_index("Ei"))

        return data

    def __getitem__(self, pix):

        if not isinstance(pix, (int, np.integer)):
            raise IndexError("Pixel index must be an integer")

        return self.get_pixel(pix)

    def to_dr(self):
        """
        Load the full response in memory.

        Returns
        -------
        a DetectorResponse containing the full response.

        """

        counts = np.array(self._drm['COUNTS'])

        data = counts * self._axes.expand_dims(self._eff_area,
                                               self._axes.label_to_index("Ei"))

        return DetectorResponse(self._axes,
                                contents = data,
                                unit = self._unit,
                                copy_contents = False)

    def close(self):
        """
        Close the HDF5 file containing the response
        """

        self._file.close()

    def __enter__(self):
        """
        Start a context manager
        """

        return self

    def __exit__(self, type, value, traceback):
        """
        Exit a context manager
        """

        self.close()

    @property
    def filename(self):
        """
        Path to on-disk file containing DetectorResponse

        Returns
        -------
        :py:class:`~pathlib.Path`
        """

        return Path(self._file.filename)

    def get_interp_response(self, coord):
        """
        Get the bilinearly interpolated response at a given coordinate location.

        Parameters
        ----------
        coord : :py:class:`astropy.coordinates.SkyCoord`
            Coordinate in the :py:class:`.SpacecraftFrame`

        Returns
        -------
        :py:class:`DetectorResponse`
        """

        pixels, weights = self.get_interp_weights(coord)

        dr = np.zeros(self._rest_axes.shape)

        for p, w in zip(pixels, weights):
            dr_p = self._get_pixel_raw(p, weight=w)
            dr += dr_p

        return DetectorResponse(self._rest_axes,
                                contents = dr,
                                unit = self.unit,
                                copy_contents = False)


    def get_point_source_response(self,
                                  exposure_map = None,
                                  coord = None,
                                  scatt_map = None,
                                  earth_occ = True):
        """
        Convolve this response with exposure for a point source at a
        given sky location.

        Input must provide one of:
          * an exposure map (dwell-time map) of time intervals to
            instrument-frame pixels
          * an inertial-frame sky coordinate for the source, plus a
            spacecraft attitude map describing the exposure of the
            spacecraft to the source over time.

        If a scatt_map is used, it is important to know whether the
        weighting of each time bin accounts for earth occultation.  If
        so, then the scatt_map depends on the source coordinate, and
        the same (single) coordinate must be passed as an argument.
        If earth occultation was not considered, the scatt_map is
        independent of the source coordinate and may be used to
        compute PSRs for many source coordinates at once; however, not
        considering earth occultation results in a non-physical
        result.

        Parameters
        ----------
        exposure_map : :py:class:`mhealpy.HealpixMap`
            Effective time spent by the source at each pixel location
            in spacecraft coordinates
        coord : :py:class:`astropy.coordinates.SkyCoord`
            Source coordinate(s) for which we want to generate a PSR
        scatt_map : :py:class:`SpacecraftAttitudeMap`
            Spacecraft attitude map used to calculate source path over
            time in spacecraft coordinates
        earth_occ : bool, optional
            Option to include Earth occultation in the response
            (scatt_map mode only). If True (the default), only one
            source coordinate may be provided, which must be the same
            as the one used to create the scatt_map.

        Returns
        -------
        :py:class:`PointSourceResponse` or tuple of same
            Inertial-frame point-source response for each source
            coordinate; tuple if more than one coordinate provided

        """

        # TODO: deprecate exposure_map in favor of source + scatt map
        # for both local and inertial coords

        if exposure_map is not None:

            if not self.conformable(exposure_map):
                raise ValueError(
                    "Exposure map has a different grid than the detector response")

            psr = np.zeros(self._rest_axes.shape)

            for p in np.nonzero(exposure_map)[0]:
                psr_p = self._get_pixel_raw(p, weight=exposure_map[p])
                psr += psr_p

            return PointSourceResponse(self._rest_axes,
                                       contents = psr,
                                       unit = u.cm*u.cm*u.s,
                                       copy_contents = False)

        else:

            def rotate_coords(c, rot):
                """
                Apply a rotation matrix to one or more 3D directions
                represented as Cartesian 3-vectors.  Return rotated directions
                in polar form as a pair (co-latitude, longitude) in
                radians.

                """
                c_local = rot @ c

                c_x, c_y, c_z = c_local

                theta = np.arctan2(c_y, c_x)
                phi = np.arccos(c_z)

                return (phi, theta)

            source = coord

            if source is None or scatt_map is None:
                raise ValueError("Provide either exposure map or source + scatt_map")

            if isinstance(source.frame, SpacecraftFrame):
                raise ValueError("scatt_map is not supported for source in local coordinate frame")

            if earth_occ and source.size > 1:
                raise ValueError("For Earth occultation, only one source coordinate "
                                 "(the one used to create the scatt map) is allowed")

            has_pol = ('Pol' in self._axes.labels and source.frame != 'spacecraftframe')
            if has_pol and source.size > 1:
                raise ValueError("For polarization, only a single source coordinate is supported")

            source = np.atleast_1d(source)

            psr_axes = self._rest_axes

            # directions corresponding to center of each HEALPix
            # pixel on PsiChi axis in source's frame
            sf_psichi_axis = psr_axes['PsiChi'].copy()
            sf_psichi_axis.coordsys = source.frame
            sf_psichi_dirs = sf_psichi_axis.pix2skycoord(np.arange(sf_psichi_axis.nbins))

            if has_pol:

                from cosipy.polarization.polarization_angle import PolarizationAngle
                from cosipy.polarization.conventions import IAUPolarizationConvention

                # angles in IAU convention's frame corresponding to
                # each bin on Pol axis in source's frame
                pol_convention = IAUPolarizationConvention()
                iau_pol_angles = PolarizationAngle(psr_axes['Pol'].centers,
                                                   source,
                                                   convention = pol_convention)

            # output PSR accumulators
            sf_psrs = tuple( np.zeros(psr_axes.shape, dtype=self.dtype)
                             for i in range(source.size) )

            if scatt_map.contents.nnz > 0:
                dirs_x, dirs_y = scatt_map.contents.coords
                attitudes = Attitude.from_axes(x = scatt_map.axes['x'].pix2skycoord(dirs_x),
                                               y = scatt_map.axes['y'].pix2skycoord(dirs_y),
                                               frame = 'icrs')

                # rotation from source frame to local spacecraft
                # frame in ICRS coordinate system
                rots = attitudes.rot.inv().as_matrix()

                # compute cartesian forms of source and PsiChi pixel dirs,
                # using ICRS coord system to match Attitudes that will be
                # used to rotate them
                src_cart = source.transform_to('icrs').cartesian.xyz.value
                sf_psichi_dirs_cart = sf_psichi_dirs.transform_to('icrs').cartesian.xyz.value

            else:
                # scatt_map is empty
                attitudes = []
                rots = []

            for att, rot, exposure in zip(attitudes, rots, scatt_map.contents.data):

                # rotate source dir(s) from source frame to local
                # spacecraft frame
                loc_src_colat, loc_src_lon = rotate_coords(src_cart, rot)

                # map each source dir in local spacecraft frame to its nearest
                # HEALPix pixel. TODO: this could be interpolated to map
                # each dir to multiple pixels + weights
                loc_src_pixels = self._axes['NuLambda'].find_bin(theta = loc_src_colat,
                                                                 phi   = loc_src_lon)

                # rotate PsiChi pixel dirs from source frame into local
                # spacecraft frame
                loc_psichi_colat, loc_psichi_lon = rotate_coords(sf_psichi_dirs_cart, rot)

                # map each local-frame PsiChi pixel dir to its nearest HEALPix
                # pixel. TODO: this could be interpolated to map each dir to
                # multiple pixels + weights
                loc_psichi_pixels = sf_psichi_axis.find_bin(theta = loc_psichi_colat,
                                                            phi   = loc_psichi_lon)

                if has_pol:

                    # rotate each bin's polarization angle from IAU to
                    # local convention
                    loc_pol_angles = iau_pol_angles.transform_to(self.pa_convention, att)

                    # wrap 180-degree polarization angles to keep them
                    # within bin range
                    la = loc_pol_angles.angle
                    la = np.where(la.deg == 180., 0. * u.deg , la)

                    # map each local-convention Pol bin angle to
                    # nearest bin (TODO: this could also be
                    # interpolated)
                    loc_pol_bins = psr_axes['Pol'].find_bin(la)

                    self._add_rot_psrs_pol(psr_axes, exposure,
                                           loc_psichi_pixels, loc_pol_bins,
                                           loc_src_pixels, sf_psrs)
                else:
                    self._add_rot_psrs(psr_axes, exposure,
                                       loc_psichi_pixels,
                                       loc_src_pixels, sf_psrs)

            # output PSRs for each source dir are in source frame
            psr_axes.set('PsiChi', sf_psichi_axis)

            results = tuple( PointSourceResponse(psr_axes,
                                                 contents = sf_psr,
                                                 unit = self._unit * scatt_map.unit,
                                                 copy_contents = False)
                             for sf_psr in sf_psrs )

            return results[0] if len(results) == 1 else results


    def _add_rot_psrs(self, axes, exposure,
                      loc_psichi_pixels,
                      loc_src_pixels, sf_psrs):
        """
        For each pixel p in loc_src_pixels, rotate the local-frame PSR
        for a source at p into the source's frame and add it to the
        corresponding accumulator in the list sf_psrs.

        Parameters
        ----------
        axes : Axes
            axes of PSR
        exposure : float
            exposure weighting for current local frame
        loc_psichi_pixels : ndarray
            local-frame pixel corresponding to each source-frame
            pixel on PSiChi axis
        loc_src_pixels : ndarray
            local-frame pixels for one or more source dirs
        sf_psrs : ndarray [output parameter]
            source-frame PSRs that accumulate rotated PSR weights
            for each local-frame source dir

        """

        aid_psichi = axes.label_to_index('PsiChi')

        for sf_psr, p in zip(sf_psrs, loc_src_pixels):

            # retrieve local-frame PSR for source pixel,
            # weighted by exposure of local frame
            loc_psr = self._get_pixel_raw(p, weight=exposure)

            # rotate local PSR into source frame and add to accumulator
            sf_psr += loc_psr.take(loc_psichi_pixels, axis=aid_psichi)


    def _add_rot_psrs_pol(self, axes, exposure,
                          loc_psichi_pixels, loc_pol_bins,
                          loc_src_pixels, sf_psrs):
        """
        For each pixel p in loc_src_pixels, rotate the local-frame
        polarization PSR for a source at p into the source's frame and
        add it to the corresponding accumulator in the list sf_psrs.
        We must rotate both the PsiChi axis and the polarization angle
        axis.

        Parameters
        ----------
        axes : Axes
            axes of PSR
        exposure : float
            exposure weighting for current local frame
        loc_psichi_pixels : ndarray
            local-frame pixel corresponding to each source-frame
            pixel on PSiChi axis
        loc_pol_bins : ndarray
            local-frame polarization bin corresponding to each
            source-frame bin on Pol axis
        loc_src_pixels : ndarray
            local-frame source pixels
        sf_psrs : ndarray [output parameter]
            source-frame PSRs that accumulate rotated PSR weights
            for each local-frame source pixel

        """

        aid_psichi = axes.label_to_index('PsiChi')
        aid_pol = axes.label_to_index('Pol')

        for sf_psr, p in zip(sf_psrs, loc_src_pixels):

            # retrieve local-frame PSR for this pixel,
            # weighted by exposure time
            psr_loc = self._get_pixel_raw(p, weight=exposure)

            sf_psr += psr_loc.take(loc_psichi_pixels,
                                   axis=aid_psichi).take(loc_pol_bins,
                                                         axis=aid_pol)


    def _setup_esr_params(self, coordsys, nside_image, nside_scatt_map):
        """
        Validate coordinate system and setup NSIDE parameters for extended
        source response generation.

        Parameters
        ----------
        coordsys : str
            Coordinate system to be used (currently only 'galactic'
            is supported)
        nside_image : int or None
            NSIDE parameter for the image reconstruction.
            If None, uses the full detector response's NSIDE.
        nside_scatt_map : int or None
            NSIDE parameter for scatt map generation.
            If None, uses the full detector response's NSIDE.

        Returns
        -------
        tuple
            (nside_image, nside_scatt_map) : validated/defaulted parameters

        """

        if coordsys != 'galactic':
            raise ValueError(f'Coordsys {coordsys} is not currently supported')

        if nside_image is None:
            nside_image = self.nside

        if nside_scatt_map is None:
            nside_scatt_map = self.nside

        return nside_image, nside_scatt_map


    def _get_psr_for_image_pixel(self, ipix, hpbase, orientation, nside_scatt_map, earth_occ = True):
        """
        Generate a PSR for one pixel of a sky map whose resolution and
        coordinate frame is specified by the HealpixBase object
        hpbase.  Use the scatt_map method to compute exposure,
        constructing a scatt_map of specified resolution from the
        supplied orientation data.

        Parameters
        ----------
        ipix: int
            HEALPix pixel index
        hpbase : HealpixBase
            HEALPixBase object describing size and frame of the map
            containing the pixel for which we're building the PSR
        orientation : cosipy.spacecraftfile.SpacecraftFile
            Spacecraft attitude information
        nside_scatt_map : int
            NSIDE parameter for scatt map generation.
            If None, uses the detector response's NSIDE.
        earth_occ : bool, default True
            Whether to include Earth occultation in the response

        Returns
        -------
        :py:class:`PointSourceResponse`
            Point source response for the specified pixel

        """

        coord = hpbase.pix2skycoord(ipix)

        scatt_map = orientation.get_scatt_map(nside = nside_scatt_map,
                                              target_coord = coord,
                                              scheme = 'ring',
                                              coordsys = hpbase.coordsys,
                                              earth_occ = earth_occ)

        psr = self.get_point_source_response(coord = coord,
                                             scatt_map = scatt_map,
                                             earth_occ = earth_occ)

        return psr


    def get_point_source_response_per_image_pixel(self, ipix, orientation, coordsys = 'galactic',
                                                  nside_image = None, nside_scatt_map = None,
                                                  earth_occ = True):
        """
        Generate a PSR for one pixel of a sky map whose resolution and
        coordinate frame are specified explicitly, or taken from the FDR
        if not given.  Use the scatt_map method to compute exposure,
        constructing a scatt_map of specified resolution from the
        supplied orientation data.

        Parameters
        ----------
        ipix_image : int
            HEALPix pixel index
        orientation : cosipy.spacecraftfile.SpacecraftFile
            Spacecraft attitude information
        coordsys : str, default 'galactic'
            Coordinate system (currently only 'galactic' is supported)
        nside_image : int, optional
            NSIDE parameter for image reconstruction.
            If None, uses the detector response's NSIDE.
        nside_scatt_map : int, optional
            NSIDE parameter for scatt map generation.
            If None, uses the detector response's NSIDE.
        earth_occ : bool, default True
            Whether to include Earth occultation in the response

        Returns
        -------
        :py:class:`PointSourceResponse`
            Point source response for the specified pixel
        """
        nside_image, nside_scatt_map = self._setup_esr_params(coordsys, nside_image, nside_scatt_map)

        hpbase = HealpixBase(nside = nside_image, coordsys = coordsys, scheme='ring')

        return self._get_psr_for_image_pixel(ipix, hpbase, orientation, nside_scatt_map, earth_occ)


    def get_extended_source_response(self, orientation, coordsys = 'galactic',
                                     nside_image = None, nside_scatt_map = None,
                                     earth_occ = True):
        """
        Generate an extended source response by combining PSRs for all
        pixels of a sky map whose resolution and coordinate frame are
        specified explicitly, or taken from the FDR if not given.  Use
        the scatt_map method to compute exposure, constructing a
        scatt_map of specified resolution from the supplied
        orientation data.

        Parameters
        ----------
        orientation : cosipy.spacecraftfile.SpacecraftFile
            Spacecraft attitude information
        coordsys : str, default 'galactic'
            Coordinate system (currently only 'galactic' is supported)
        nside_image : int, optional
            NSIDE parameter for output extended response
            If None, uses the detector response's NSIDE.
        nside_scatt_map : int, optional
            NSIDE parameter for scatt map generation.
            If None, uses the detector response's NSIDE.
        earth_occ : bool, default True
            Whether to include Earth occultation in the response

        Returns
        -------
        :py:class:`ExtendedSourceResponse`
            Extended source response covering all pixels in a map
            of resolution nside_image

        """

        from tqdm.autonotebook import tqdm

        nside_image, nside_scatt_map = self._setup_esr_params(coordsys, nside_image, nside_scatt_map)

        # This axis label should be 'lb' in the future
        pixel_axis = HealpixAxis(nside = nside_image, coordsys = coordsys, scheme='ring', label = 'NuLambda')

        esr = np.empty((pixel_axis.nbins,) + self._rest_axes.shape, dtype=self.dtype)

        for ipix in tqdm(range(pixel_axis.npix)):
            psr = self._get_psr_for_image_pixel(ipix, pixel_axis,
                                                orientation, nside_scatt_map,
                                                earth_occ)
            esr[ipix] = psr.contents.value

        axes = Axes([pixel_axis] + list(psr.axes), copy_axes=False)
        return ExtendedSourceResponse(axes, contents = esr,
                                      unit = psr.unit,
                                      copy_contents = False)


    @staticmethod
    def merge_psr_to_extended_source_response(basepath, coordsys = 'galactic', nside_image = None):
        """
        Combine a set of PSRs stored in files into one extended source
        response.  Infer the size of the ESR's sky map from the number of
        files to be read, and the pixel ID for each file by parsing its name.

        The names of the component PSR files are assumed to be of the
        form `basepath` + index + file_extension.  For example, with
        basepath='histograms/hist_', filenames are expected to be
        'histograms/hist_<int>.<ext>' (for example,
        'histograms/hist_00001.h5').

        Note that nside_image, if specified, must correspond to the number
        of pixels read; otherwise, an exception is raised. The individual
        PSRS must have the same axes (including the coordinate system for
        the PsiChi dimension) and must have compatible units.

        Parameters
        ----------
        basepath : str
            Base filename pattern for point source response files
        coordsys : str, default 'galactic'
            Coordinate system (currently only 'galactic' is supported)
        nside_image : int, optional
            NSIDE parameter for image reconstruction (must match
            number of PSRs to combine) If None, infer from number of
            PSRS to combine

        Returns
        -------
        :py:class:`ExtendedSourceResponse`
            Combined extended source response. The unit will be that
            of the PSR with the lexicographically least filename.

        """

        import mhealpy as hp

        if coordsys != 'galactic':
            raise ValueError("Only galactic coordinates are supported for output ESR")

        # get list of PSRs to merge
        basepath = Path(basepath)
        basename = basepath.name
        psr_files = sorted(basepath.parent.glob(basename + "*"))

        npix = len(psr_files)
        if npix == 0:
            raise FileNotFoundError(f"No files found matching pattern {basename}*")
        else:
            try:
                nside = hp.npix2nside(npix)
            except:
                raise RuntimeError(f"Number of input PSRs {npix} does not correspond to a valid nside")

            if nside_image is not None and nside_image != nside:
                raise RuntimeError(f"Number of input PSRs {npix} does not correspond to nside {nside}")

        # All component PSRs must have same axes and unit; get them here
        psr = PointSourceResponse.open(psr_files[0])
        psr_axes = psr.axes
        psr_unit = psr.unit

        esr = np.empty((npix,) + psr_axes.shape, dtype = psr.dtype)
        for psr_file in psr_files:

            ipix = int(psr_file.stem[len(basename):])

            psr = PointSourceResponse.open(psr_file)

            # make sure all PSRs have same axes (includng coord frame) and unit
            if psr.axes != psr_axes:
                raise ValueError(f"Axes of PSR '{psr_file}' do not match those of other PSRs")
            elif not psr.unit.is_equivalent(psr_unit):
                raise ValueError(f"Unit of PSR '{psr_file}' is not compatible with unit of other PSRs")

            esr[ipix] = psr.contents.to_value(psr_unit) # make sure units of all PSRs conform

        # This axis label should be 'lb' in the future
        pixel_axis = HealpixAxis(nside = nside, coordsys = coordsys, scheme='ring', label = 'NuLambda')

        axes = Axes([pixel_axis] + list(psr_axes), copy_axes=False)
        return ExtendedSourceResponse(axes, contents = esr,
                                      unit = psr_unit,
                                      copy_contents = False)


    def __str__(self):
        return f"{self.__class__.__name__}(filename = '{self.filename.resolve()}')"

    def __repr__(self):

        output = (f"FILENAME: '{self.filename.resolve()}'\n"
                  f"AXES:\n")

        for naxis, axis in enumerate(self._axes):

            if naxis == 0:
                description = "Location of the simulated source in the spacecraft coordinates"
            else:
                description = self._drm['AXIS_DESCRIPTIONS'].attrs[axis.label]

            output += (f"  {axis.label}:\n"
                       f"    DESCRIPTION: '{description}'\n")

            if isinstance(axis, HealpixAxis):
                output += (f"    TYPE: 'healpix'\n"
                           f"    NPIX: {axis.npix}\n"
                           f"    NSIDE: {axis.nside}\n"
                           f"    SCHEME: '{axis.scheme}'\n")
            else:
                output += (f"    TYPE: '{axis.axis_scale}'\n"
                           f"    UNIT: '{axis.unit}'\n"
                           f"    NBINS: {axis.nbins}\n"
                           f"    EDGES: [{', '.join([str(e) for e in axis.edges])}]\n")

        return output

    def _repr_pretty_(self, p, cycle):

        if cycle:
            p.text(str(self))
        else:
            p.text(repr(self))


def cosi_response(argv=None):
    """
    Print the content of a detector response to stdout.
    """
    import argparse
    import textwrap
    from yayc import Configurator
    import matplotlib.pyplot as plt
    from mhealpy import HealpixMap
    from astropy.coordinates import SkyCoord

    # Parse arguments from commandline
    apar = argparse.ArgumentParser(
        usage=textwrap.dedent(
            """
            %(prog)s [--help] <command> [<args>] <filename> [<options>]
            """),
        description=textwrap.dedent(
            """
            Quick view of the information contained in a response file

            %(prog)s --help
            %(prog)s dump header [FILENAME]
            %(prog)s dump aeff [FILENAME] --lon [LON] --lat [LAT]
            %(prog)s dump expectation [FILENAME] --config [CONFIG]
            %(prog)s plot aeff [FILENAME] --lon [LON] --lat [LAT]
            %(prog)s plot dispersion [FILENAME] --lon [LON] --lat [LAT]
            %(prog)s plot expectation [FILENAME] --lon [LON] --lat [LAT]

            Arguments:
            - header: Response header and axes information
            - aeff: Effective area
            - dispersion: Energy dispection matrix
            - expectation: Expected number of counts
            """),
        formatter_class=argparse.RawTextHelpFormatter)

    apar.add_argument('command',
                      help=argparse.SUPPRESS)
    apar.add_argument('args', nargs='*',
                      help=argparse.SUPPRESS)
    apar.add_argument('filename',
                      help="Path to instrument response")
    apar.add_argument('--lon',
                      help="Longitude in sopacecraft coordinates. e.g. '11deg'")
    apar.add_argument('--lat',
                      help="Latitude in sopacecraft coordinates. e.g. '10deg'")
    apar.add_argument('--output', '-o',
                      help="Save output to file. Default: stdout")
    apar.add_argument('--config', '-c',
                      help="Path to config file describing exposure and source charateristics.")
    apar.add_argument('--config-override', dest='override',
                      help="Override option in config file")

    args = apar.parse_args(argv)

    # Config
    if args.config is None:
        config = Configurator()
    else:
        config = Configurator.open(args.config)

        if args.override is not None:
            config.override(args.override)

    # Get info
    with FullDetectorResponse.open(args.filename) as response:

        # Commands and functions
        def get_drm():

            lat = Quantity(args.lat)
            lon = Quantity(args.lon)

            loc = SkyCoord(lon=lon, lat=lat, frame=SpacecraftFrame())

            return response.get_interp_response(loc)

        def get_expectation():

            from astromodels.core.model_parser import ModelParser
            from astropy.time import Time

            # Exposure map
            exposure_map = HealpixMap(base=response,
                                      unit=u.s,
                                      coordsys=SpacecraftFrame())

            ti = Time(config['exposure:time_i'])
            tf = Time(config['exposure:time_f'])
            dt = (tf-ti).to(u.s)

            exposure_map[:4] = dt/4

            logger.warning(f"Spacecraft file not yet implemented, faking source on "
                           f"axis from {ti} to {tf} ({dt:.2f})")

            # Point source response
            psr = response.get_point_source_response(exposure_map)

            # Spectrum
            model = ModelParser(model_dict=config['sources']).get_model()

            spectrum = model.point_sources['source'].components['main'].shape
            logger.info(f"Using spectrum:\n {spectrum}")

            # Expectation
            expectation = psr.get_expectation(spectrum).project('Em')

            return expectation

        def command_dump():

            if len(args.args) != 1:
                apar.error("Command 'dump' takes a single argument")

            option = args.args[0]

            if option == 'header':

                result = repr(response)

            elif option == 'aeff':

                drm = get_drm()

                aeff = drm.get_spectral_response().get_effective_area()

                result = "#Energy[keV]     Aeff[cm2]\n"

                for e, a in zip(aeff.axis.centers, aeff):
                    # IMC: fix this latter when histpy has units
                    result += f"{e.to_value(u.keV):>12.2e}  {a.to_value(u.cm*u.cm):>12.2e}\n"

            elif option == 'expectation':

                expectation = get_expectation()

                result = "#Energy_min[keV]   Energy_max[keV]  Expected_counts\n"

                for emin, emax, ex in zip(expectation.axis.lower_bounds,
                                          expectation.axis.upper_bounds,
                                          expectation):
                    # IMC: fix this latter when histpy has units
                    result += (f"{emin.to_value(u.keV):>16.2e}  "
                               f"{emax.to_value(u.keV):>16.2e}  "
                               f"{ex:>15.2e}\n")

            else:

                apar.error(f"Argument '{option}' not valid for 'dump' command")

            if args.output is None:
                logger.info(result)
            else:
                logger.info(f"Saving result to {Path(args.output).resolve()}")
                f = open(args.output, 'a')
                f.write(result)
                f.close()

        def command_plot():

            if len(args.args) != 1:
                apar.error("Command 'plot' takes a single argument")

            option = args.args[0]

            if option == 'aeff':

                drm = get_drm()

                drm.get_spectral_response().get_effective_area().plot(errorbars=False)

            elif option == 'dispersion':

                drm = get_drm()

                drm.get_spectral_response().get_dispersion_matrix().plot()

            elif option == 'expectation':

                expectation = get_expectation().plot(errorbars=False)

            else:

                apar.error(f"Argument '{option}' not valid for 'plot' command")

            if args.output is None:
                plt.show()
            else:
                logger.info(f"Saving plot to {Path(args.output).resolve()}")
                plt.savefig(args.output)

        # Run
        if args.command == 'plot':
            command_plot()
        elif args.command == 'dump':
            command_dump()
        else:
            apar.error(f"Command '{args.command}' unknown")
