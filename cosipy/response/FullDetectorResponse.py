import logging
logger = logging.getLogger(__name__)

import argparse
import textwrap

import h5py as h5

from histpy import Histogram, Axes, Axis, HealpixAxis

from scoords import SpacecraftFrame
from cosipy.config import Configurator

from mhealpy import HealpixBase, HealpixMap
import mhealpy as hp

import numpy as np

from pathlib import Path

from sparse import COO

import astropy.units as u
from astropy.units import Quantity
from astropy.coordinates import SkyCoord
from astropy.time import Time

import matplotlib.pyplot as plt

from astromodels.core.model_parser import ModelParser

from .DetectorResponse import DetectorResponse
from .PointSourceResponse import PointSourceResponse

class FullDetectorResponse(HealpixBase):
    """
    Handles the multi-dimensional matrix that describes the
    full all-sky response of the instrument.

    You can access the :py:class:`DetectorResponse` at a given pixel using the ``[]``
    operator. Alternatively you can obtain the interpolated reponse using
    :py:func:`get_interp_response`.    
    """
    
    def __init__(self, *args, **kwargs):
        # Overload parent init. Called in class methods.
        pass
        
    @classmethod
    def open(cls, filename):
        """
        Open a detector response file.

        Parameters
        ----------
        filename : str, :py:class:`~pathlib.Path`
            Path to HDF5 file
        """

        new = cls(filename)

        new._file = h5.File(filename, mode = 'r')
        
        new._drm = new._file['DRM']

        new._unit = u.Unit(new._drm.attrs['UNIT'])

        # Axes
        axes = []
        
        for axis_label in new._drm["AXES"]:
            
            axis = new._drm['AXES'][axis_label]

            axis_type = axis.attrs['TYPE']

            if axis_type == 'healpix':

                axes += [HealpixAxis(edges = np.array(axis),
                                     nside = axis.attrs['NSIDE'],
                                     label = axis_label,
                                     scheme = axis.attrs['SCHEME'],
                                     coordsys = SpacecraftFrame())]

            else:
                axes += [Axis(np.array(axis) * u.Unit(axis.attrs['UNIT']),
                              scale = axis_type,
                              label = axis_label)]

        new._axes = Axes(axes)

        # Init HealpixMap (local coordinates, main axis)
        HealpixBase.__init__(new,
                             base = new.axes['NuLambda'],
                             coordsys = SpacecraftFrame())

        return new
        
    @property
    def ndim(self):
        """
        Dimensionality of detector response matrix.

        Returns
        -------
        int
        """
        
        return self.axes.ndim

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
        
    def __getitem__(self, pix):

        if not isinstance(pix, (int, np.integer)) or pix < 0 or not pix < self.npix:
            raise IndexError("Pixel number out of range, or not an integer")
        
        coords = np.reshape(self._file['DRM']['BIN_NUMBERS'][pix], (self.ndim-1,-1))
        data = np.array(self._file['DRM']['CONTENTS'][pix])

        return DetectorResponse(self.axes[1:],
                                contents = COO(coords = coords,
                                               data = data,
                                               shape = tuple(self.axes.nbins[1:])),
                                unit = self.unit)
    
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

        dr = DetectorResponse(self.axes[1:],
                              sparse = True,
                              unit = self.unit)
        
        for p,w in zip(pixels, weights):

            dr += self[p]*w

        return dr
    
    def get_point_source_response(self, exposure_map):
        """
        Convolve the all-sky detector response with exposure for a source at a given
        sky location.
        
        Parameters
        ----------
        exposure_map : :py:class:`mhealpy.HealpixMap`
            Effective time spent by the source at each pixel location in spacecraft coordinates

        Returns
        -------
        :py:class:`PointSourceResponse`
        """

        if not self.conformable(exposure_map):
            raise ValueError("Exposure map has a different grid than the detector response")
            
        psr = PointSourceResponse(self.axes[1:],
                                  sparse = True,
                                  unit = u.cm*u.cm*u.s)
        
        for p in range(self.npix):

            if exposure_map[p] != 0:
                psr += self[p]*exposure_map[p]
            
        return psr

    def __str__(self):
        return f"{self.__class__.__name__}(filename = '{self.filename.resolve()}')"

    def __repr__(self):
        
        output = (f"FILENAME: '{self.filename.resolve()}'\n"
                  f"AXES:\n")
        
        for naxis, axis in enumerate(self.axes):

            if naxis == 0:
                description = "Location of the simulated source in the spacecraft coordinates"
            else:
                description = self._drm['AXES'][axis.label].attrs['DESCRIPTION']
                
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
    
def cosi_response(argv = None):
    """
    Print the content of a detector response to stdout.
    """

    # Parse arguments from commandline
    apar = argparse.ArgumentParser(
        usage = textwrap.dedent(
            """
            %(prog)s [--help] <command> [<args>] <filename> [<options>]
            """),
        description = textwrap.dedent(
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
    apar.add_argument('args', nargs = '*',
                      help=argparse.SUPPRESS)
    apar.add_argument('filename', 
                      help="Path to instrument response")
    apar.add_argument('--lon',
                      help = "Longitude in sopacecraft coordinates. e.g. '11deg'")
    apar.add_argument('--lat',
                      help = "Latitude in sopacecraft coordinates. e.g. '10deg'")
    apar.add_argument('--output','-o',
                      help="Save output to file. Default: stdout")
    apar.add_argument('--config','-c',
                      help="Path to config file describing exposure and source charateristics.")
    apar.add_argument('--config-override', dest = 'override', 
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
            
            loc = SkyCoord(lon = lon, lat = lat, frame = SpacecraftFrame())
            
            return response.get_interp_response(loc)

        def get_expectation():

            # Exposure map
            exposure_map = HealpixMap(base = response, 
                                      unit = u.s, 
                                      coordsys = SpacecraftFrame())

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
            
            for src_name, src in model.point_sources.items():
                for comp_name, component in src.components.items():
                    logger.info(f"Using spectrum:\n {component.shape}")
            
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

                for e,a in zip(aeff.axis.centers, aeff):
                    # IMC: fix this latter when histpy has units
                    result += f"{e.to_value(u.keV):>12.2e}  {a.to_value(u.cm*u.cm):>12.2e}\n"

            elif option == 'expectation':

                expectation = get_expectation()
                
                result = "#Energy_min[keV]   Energy_max[keV]  Expected_counts\n"

                for emin,emax,ex in zip(expectation.axis.lower_bounds,
                                        expectation.axis.upper_bounds,
                                        expectation):
                    # IMC: fix this latter when histpy has units
                    result += (f"{emin.to_value(u.keV):>16.2e}  "
                               f"{emax.to_value(u.keV):>16.2e}  "
                               f"{ex:>15.2e}\n")
                    
            else:
                    
                apar.error(f"Argument '{option}' not valid for 'dump' command")

            if args.output is None:
                print(result)
            else:
                logger.info(f"Saving result to {Path(args.output).resolve()}")
                f = open(args.output,'a')
                f.write(result)
                f.close()
            
                
        def command_plot():

            if len(args.args) != 1:
                apar.error("Command 'plot' takes a single argument")
                
            option = args.args[0]

            if option == 'aeff':
                
                drm = get_drm()
            
                drm.get_spectral_response().get_effective_area().plot(errorbars = False)

            elif option == 'dispersion':

                drm = get_drm()
                
                drm.get_spectral_response().get_dispersion_matrix().plot() 

            elif option == 'expectation':

                expectation = get_expectation().plot(errorbars = False)
                
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
        
        

            
