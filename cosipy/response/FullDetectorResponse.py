import glob
from pathlib import Path

from tqdm.autonotebook import tqdm

import numpy as np

import h5py as h5
import hdf5plugin

from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy.units import Quantity
import astropy.units as u

from scoords import SpacecraftFrame, Attitude

import mhealpy as hp
from mhealpy import HealpixBase, HealpixMap

from histpy import Histogram, Axes, Axis, HealpixAxis

from astromodels.core.model_parser import ModelParser

from .RspConverter import RspConverter
from .PointSourceResponse import PointSourceResponse
from .DetectorResponse import DetectorResponse
from .ExtendedSourceResponse import ExtendedSourceResponse

import logging
logger = logging.getLogger(__name__)


class FullDetectorResponse(HealpixBase):
    """
    Handles the multi-dimensional matrix that describes the
    full all-sky response of the instrument.

    You can access the :py:class:`DetectorResponse` at a given pixel using the ``[]``
    operator. Alternatively you can obtain the interpolated reponse using
    :py:func:`get_interp_response`.
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
            raise RuntimeError("Polarization angle convention of response ('RelativeX', 'RelativeY', or 'RelativeZ') must be provided")

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
        --------
        :py:class:`np.ndarray`
        """

        return self._eff_area

    def __getitem__(self, pix):

        if not isinstance(pix, (int, np.integer)):
            raise IndexError("Pixel index must be an integer")

        rest_axes = self._axes[1:]

        counts = self._drm['COUNTS'][pix]

        data = counts * rest_axes.expand_dims(self._eff_area,
                                              rest_axes.label_to_index("Ei"))

        return DetectorResponse(rest_axes,
                                contents=data,
                                unit=self.unit,
                                copy_contents=False)

    def to_histogram(self):
        """
        Return a Histogram containing the response, with matching
        axes and units.

        """
        counts = np.array(self._drm['COUNTS'])

        contents = counts * self._axes.expand_dims(self._eff_area,
                                               self._axes.label_to_index("Ei"))

        return Histogram(self._axes,
                         contents = contents,
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

        dr = DetectorResponse(self._axes[1:],
                              unit=self.unit)


        for p, w in zip(pixels, weights):

            dr += self[p]*w

        return dr

    def get_point_source_response(self,
                                  exposure_map = None,
                                  coord = None,
                                  scatt_map = None,
                                  Earth_occ = True):
        """
        Convolve the all-sky detector response with exposure for a source at a given
        sky location.

        Provide either a exposure map (aka dweel time map) or a combination of a
        sky coordinate and a spacecraft attitude map.

        Parameters
        ----------
        exposure_map : :py:class:`mhealpy.HealpixMap`
            Effective time spent by the source at each pixel location in spacecraft coordinates
        coord : :py:class:`astropy.coordinates.SkyCoord`
            Source coordinate
        scatt_map : :py:class:`SpacecraftAttitudeMap`
            Spacecraft attitude map
        Earth_occ : bool, optional
            Option to include Earth occultation in the respeonce.
            Default is True, in which case you can only pass one
            coord, which must be the same as was used for the scatt map.

        Returns
        -------
        :py:class:`PointSourceResponse`
        """

        # TODO: deprecate exposure_map in favor of coords + scatt map for both local
        # and interntial coords

        if Earth_occ == True:
            if coord != None:
                if coord.size > 1:
                    raise ValueError("For Earth occultation you must use the same coordinate as was used for the scatt map!")

        if exposure_map is not None:
            if not self.conformable(exposure_map):
                raise ValueError(
                    "Exposure map has a different grid than the detector response")

            psr = PointSourceResponse(self._axes[1:],
                                      unit=u.cm*u.cm*u.s)

            for p in np.nonzero(exposure_map)[0]:
                psr += self[p] * exposure_map[p]

            return psr

        else:

            # Rotate to inertial coordinates

            if coord is None or scatt_map is None:
                raise ValueError("Provide either exposure map or coord + scatt_map")

            if isinstance(coord.frame, SpacecraftFrame):
                raise ValueError("Local coordinate + scatt_map not currently supported")

            axis = "PsiChi"

            coords_axis = Axis(np.arange(coord.size+1), label = 'coords')
            axes = Axes([coords_axis] + list(self._axes[1:])) # copies all Axis objects
            axes["PsiChi"].coordsys = coord.frame # OK because not shared with any other Axes yet

            psrs = Histogram(axes, unit = self.unit * scatt_map.unit)

            for i,(pixels, exposure) in \
                enumerate(zip(scatt_map.contents.coords.transpose(),
                              scatt_map.contents.data * scatt_map.unit)):

                att = Attitude.from_axes(x = scatt_map.axes['x'].pix2skycoord(pixels[0]),
                                         y = scatt_map.axes['y'].pix2skycoord(pixels[1]))

                coord.attitude = att

                #TODO: Change this to interpolation
                loc_nulambda_pixels = np.array(self._axes['NuLambda'].find_bin(coord),
                                               ndmin = 1)

                dr_pix = Histogram.concatenate(coords_axis, [self[i] for i in loc_nulambda_pixels])

                dr_pix.axes['PsiChi'].coordsys = SpacecraftFrame(attitude = att)

                self._sum_rot_hist(dr_pix, psrs, exposure, coord, self.pa_convention)

            # Convert to tuple of PSRs for each bin of coords axis
            psr = tuple(PointSourceResponse(psrs.axes[1:],
                                            contents = data,
                                            unit = psrs.unit,
                                            copy_contents = False)
                        for data in psrs.contents)

            if coord.size == 1:
                return psr[0]
            else:
                return psr

    def _setup_extended_source_response_params(self, coordsys, nside_image, nside_scatt_map):
        """
        Validate coordinate system and setup NSIDE parameters for extended source response generation.

        Parameters
        ----------
        coordsys : str
            Coordinate system to be used (currently only 'galactic' is supported)
        nside_image : int or None
            NSIDE parameter for the image reconstruction.
            If None, uses the full detector response's NSIDE.
        nside_scatt_map : int or None
            NSIDE parameter for scatt map generation.
            If None, uses the full detector response's NSIDE.

        Returns
        -------
        tuple
            (coordsys, nside_image, nside_scatt_map) : validated parameters
        """
        if coordsys != 'galactic':
            raise ValueError(f'The coordsys {coordsys} not currently supported')

        if nside_image is None:
            nside_image = self.nside

        if nside_scatt_map is None:
            nside_scatt_map = self.nside

        return coordsys, nside_image, nside_scatt_map

    def get_point_source_response_per_image_pixel(self, ipix_image, orientation, coordsys = 'galactic',
                                                  nside_image = None, nside_scatt_map = None, Earth_occ = True):
        """
        Generate point source response for a specific HEALPix pixel by convolving
        the all-sky detector response with exposure.

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
        Earth_occ : bool, default True
            Whether to include Earth occultation in the response

        Returns
        -------
        :py:class:`PointSourceResponse`
            Point source response for the specified pixel
        """
        coordsys, nside_image, nside_scatt_map = self._setup_extended_source_response_params(coordsys, nside_image, nside_scatt_map)

        image_axes = HealpixAxis(nside = nside_image, coordsys = coordsys, scheme='ring', label = 'NuLambda') # The label should be 'lb' in the future

        coord = image_axes.pix2skycoord(ipix_image)

        scatt_map = orientation.get_scatt_map(nside = nside_scatt_map,
                                              target_coord = coord,
                                              scheme='ring',
                                              coordsys=coordsys,
                                              earth_occ=Earth_occ)

        psr = self.get_point_source_response(coord = coord, scatt_map = scatt_map, Earth_occ = Earth_occ)

        return psr

    def get_extended_source_response(self, orientation, coordsys = 'galactic', nside_image = None, nside_scatt_map = None, Earth_occ = True):
        """
        Generate extended source response by convolving the all-sky detector
        response with exposure over the entire sky.

        Parameters
        ----------
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
        Earth_occ : bool, default True
            Whether to include Earth occultation in the response

        Returns
        -------
        :py:class:`ExtendedSourceResponse`
            Extended source response covering the entire sky
        """
        coordsys, nside_image, nside_scatt_map = self._setup_extended_source_response_params(coordsys, nside_image, nside_scatt_map)

        axes = [HealpixAxis(nside = nside_image, coordsys = coordsys, scheme='ring', label = 'NuLambda')] # The label should be 'lb' in the future
        axes += list(self._axes[1:])
        axes[-1].coordsys = coordsys

        extended_source_response = ExtendedSourceResponse(axes, unit = u.Unit("cm2 s"))

        for ipix in tqdm(range(hp.nside2npix(nside_image))):

            psr = self.get_point_source_response_per_image_pixel(ipix, orientation, coordsys = coordsys,
                                                                 nside_image = nside_image, nside_scatt_map = nside_scatt_map, Earth_occ = Earth_occ)

            extended_source_response[ipix] = psr.contents

        return extended_source_response

    def merge_psr_to_extended_source_response(self, basename, coordsys = 'galactic', nside_image = None):
        """
        Create extended source response by merging multiple point source responses.

        Reads point source response files matching the pattern `basename` + index + file_extension.
        For example, with basename='histograms/hist_', filenames are expected to be like 'histograms/hist_00001.hdf5'.

        Parameters
        ----------
        basename : str
            Base filename pattern for point source response files
        coordsys : str, default 'galactic'
            Coordinate system (currently only 'galactic' is supported)
        nside_image : int, optional
            NSIDE parameter for image reconstruction.
            If None, uses the detector response's NSIDE.

        Returns
        -------
        :py:class:`ExtendedSourceResponse`
            Combined extended source response
        """
        coordsys, nside_image, _ = self._setup_extended_source_response_params(coordsys, nside_image, None)

        psr_files = glob.glob(basename + "*")

        if not psr_files:
            raise FileNotFoundError(f"No files found matching pattern {basename}*")

        axes = [HealpixAxis(nside = nside_image, coordsys = coordsys, scheme='ring', label = 'NuLambda')] # The label should be 'lb' in the future
        axes += list(self._axes[1:])
        axes[-1].coordsys = coordsys

        extended_source_response = ExtendedSourceResponse(axes, unit = u.Unit("cm2 s"))

        filled_pixels = []

        for filename in psr_files:

            ipix = int(filename[len(basename):].split(".")[0])

            psr = Histogram.open(filename)

            extended_source_response[ipix] = psr.contents

            filled_pixels.append(ipix)

        expected_pixels = set(range(extended_source_response.axes[0].npix))
        if set(filled_pixels) != expected_pixels:
            raise ValueError(f"Missing pixels in the response files. Expected {extended_source_response.axes[0].npix} pixels, got {len(filled_pixels)} pixels")

        return extended_source_response

    @staticmethod
    def _sum_rot_hist(h, h_new, exposure, coord, pa_convention, axis = "PsiChi"):
        """
        Rotate a histogram with HealpixAxis h into the grid of h_new, and sum
        it up with the weight of exposure.

        Meant to rotate the PsiChi of a CDS from local to galactic
        """

        axis_id = h.axes.label_to_index(axis)

        old_axis = h.axes[axis_id]
        new_axis = h_new.axes[axis_id]

        # Convolve
        # TODO: Change this to interpolation (pixels + weights)
        old_pixels = old_axis.find_bin(new_axis.pix2skycoord(np.arange(new_axis.nbins)))

        if 'Pol' in h.axes.labels and h_new.axes[axis].coordsys.name != 'spacecraftframe':

            if coord.size > 1:
                raise ValueError("For polarization, only a single source coordinate is supported")

            from cosipy.polarization.polarization_angle import PolarizationAngle
            from cosipy.polarization.conventions import IAUPolarizationConvention

            pol_axis_id = h.axes.label_to_index('Pol')

            old_pol_axis = h.axes[pol_axis_id]
            new_pol_axis = h_new.axes[pol_axis_id]

            old_pol_indices = []
            for i in range(h_new.axes['Pol'].nbins):

                pa = PolarizationAngle(h_new.axes['Pol'].centers.to_value(u.deg)[i] * u.deg, coord.transform_to('icrs'), convention=IAUPolarizationConvention())
                pa_old = pa.transform_to(pa_convention, attitude=coord.attitude)

                if pa_old.angle.deg == 180.:
                    pa_old = PolarizationAngle(0. * u.deg, coord, convention=IAUPolarizationConvention())

                old_pol_indices.append(old_pol_axis.find_bin(pa_old.angle))

            old_pol_indices = np.array(old_pol_indices)

        # NOTE: there are some pixels that are duplicated, since the center 2 pixels
        # of the original grid can land within the boundaries of a single pixel
        # of the target grid. The following commented code fixes this, but it's slow, and
        # the effect is very small, so maybe not worth it
        # nulambda_npix = h.axes['NuLamnda'].nbins
        # new_norm = np.zeros(shape = nulambda_npix)
        # for p in old_pixels:
        #     h_slice = h[{axis:p}]
        #     norm_rot += np.sum(h_slice, axis = tuple(np.arange(1, h_slice.ndim)))
        # old_norm = np.sum(h, axis = tuple(np.arange(1, h.ndim)))
        # norm_corr = h.expand_dims(norm / norm_rot, "NuLambda")

        for old_pix,new_pix in zip(old_pixels,range(new_axis.npix)):

            #h_new[{axis:new_pix}] += exposure * h[{axis: old_pix}] # * norm_corr
            # The following code does the same than the code above, but is faster

            if not 'Pol' in h.axes.labels:

                old_index = (slice(None),)*axis_id + (old_pix,)
                new_index = (slice(None),)*axis_id + (new_pix,)

                h_new[new_index] += exposure * h[old_index] # * norm_corr

            else:

                for old_pol_bin,new_pol_bin in zip(old_pol_indices,range(new_pol_axis.nbins)):

                    if pol_axis_id < axis_id:

                        old_index = (slice(None),)*pol_axis_id + (old_pol_bin,) + (slice(None),)*(axis_id-pol_axis_id-1) + (old_pix,)
                        new_index = (slice(None),)*pol_axis_id + (new_pol_bin,) + (slice(None),)*(axis_id-pol_axis_id-1) + (new_pix,)

                    else:

                        old_index = (slice(None),)*axis_id + (old_pix,) + (slice(None),)*(pol_axis_id-axis_id-1) + (old_pol_bin,)
                        new_index = (slice(None),)*axis_id + (new_pix,) + (slice(None),)*(pol_axis_id-axis_id-1) + (new_pol_bin,)

                    h_new[new_index] += exposure * h[old_index] # * norm_corr


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
