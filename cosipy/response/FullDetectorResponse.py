import argparse
import textwrap

import h5py as h5

from histpy import Histogram, Axes, Axis

from cosipy.coordinates import SpacecraftFrame

from mhealpy import HealpixBase
import mhealpy as hp

import numpy as np

from pathlib import Path

from sparse import COO

import astropy.units as u

from .DetectorResponse import DetectorResponse
from .healpix_axis import HealpixAxis
from .quantity_axis import QuantityAxis
from .PointSourceResponse import PointSourceResponse

class FullDetectorResponse(HealpixBase):
    """
    FullDetectorResponse handles the multi-dimensional matrix that describes the
    full all-sky response of the instrument.

    Parameters
    ----------
    filename : str, Path, optional
        Path to file
    """
    
    def __init__(self, filename = None, *args, **kwargs):

        if filename is not None:
            self.open(filename, *args, **kwargs)

    def open(self, filename, *args, **kwargs):
        """
        Open a detector response file.

        Parameters
        ----------
        filename : str, Path
            Path to HDF5 file
        """

        # Open HDF5
        self._file = h5.File(filename, *args, **kwargs)

        self._drm = self._file['DRM']

        # Init HealpixMap (local coordinates, main axis)
        super().__init__(nside = self._drm.attrs["NSIDE"],
                         scheme = self._drm.attrs["SCHEME"],
                         coordsys = SpacecraftFrame())

        self._unit = u.Unit(self._drm.attrs['UNIT'])
        
        # The rest of the axes
        axes = []

        for axis_label in self._drm["AXES"]:
            
            axis = self._drm['AXES'][axis_label]

            axis_type = axis.attrs['TYPE']

            if axis_type == 'healpix':

                axes += [HealpixAxis(edges = np.array(axis),
                                     nside = axis.attrs['NSIDE'],
                                     label = axis_label,
                                     scheme = axis.attrs['SCHEME'],
                                     coordsys = SpacecraftFrame())]

            else:
                axes += [QuantityAxis(np.array(axis),
                                      scale = axis_type,
                                      label = axis_label,
                                      unit = axis.attrs['UNIT'])]

        self._axes = Axes(axes)

    @property
    def ndim(self):

        return self._axes.ndim+1

    @property
    def axes(self):
        return self._axes

    @property
    def unit(self):
        return self._unit
        
    def __getitem__(self, pix):

        if not isinstance(pix, (int, np.integer)) or pix < 0 or not pix < self.npix:
            raise IndexError("Pixel number out of range, or not an integer")
        
        coords = np.reshape(self._file['DRM']['BIN_NUMBERS'][pix], (self.ndim - 1,-1))
        data = np.array(self._file['DRM']['CONTENTS'][pix])

        return DetectorResponse(self.axes,
                                contents = COO(coords = coords,
                                               data = data,
                                               shape = tuple(self.axes.nbins)),
                                unit = self.unit)
    
    def close(self):
        """
        Close the HDF5 file containing the DetectorResponse
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
        
        Return:
            Path
        """
        
        return Path(self._file.filename)

    def get_interp_response(self, coord):

        pixels, weights = self.get_interp_weights(coord)

        dr = DetectorResponse(self.axes,
                              sparse = True,
                              unit = self.unit)
        
        for p,w in zip(pixels, weights):

            dr += self[p]*w

        return dr
    
    def get_point_source_response(self, exposure_map):
        """

        exposure_map : HealpixMap
            Effective time spent by the source at each location in spacecraft coordinate
        """

        if not self.conformable(exposure_map):
            raise ValueError("Exposure map has a different grid than the detector response")
            
        psr = PointSourceResponse(self.axes,
                                  sparse = True,
                                  unit = u.cm*u.cm*u.s)
        
        for p in range(self.npix):

            if exposure_map[p] != 0:
                psr += self[p]*exposure_map[p]
            
        return psr

    def __str__(self):
        return f"{self.__class__.__name__}(filename = '{self.filename.resolve()}')"

    def __repr__(self):
        return str(self)

    def describe(self):
        
        output = (f"FILENAME: '{self.filename.resolve()}'\n"
                  f"NPIX: {self.npix}\n"
                  f"NSIDE: {self.nside}\n"
                  f"SCHEME: '{self.scheme}'\n"
                  f"AXES:\n")
        
        for axis in self.axes:

            output += (f"  {axis.label}:\n"
                       f"    DESCRIPTION: '{self._drm['AXES'][axis.label].attrs['DESCRIPTION']}'\n")
                
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
            p.text(self.describe())
    
    def dump(self):
        """
        Print the content of the response to stdout.
        """

        print(f"Filename: {self._filename}. No contents for now!")

def cosi_rsp_dump(argv = None):
    """
    Print the content of a detector response to stdout.
    """

    # Parse arguments from commandline
    aPar = argparse.ArgumentParser(
        usage = ("%(prog)s filename "
                 "[--help] [options]"),
        description = textwrap.dedent(
            """
            Dump DR contents
            """),
        formatter_class=argparse.RawTextHelpFormatter)

    aPar.add_argument('filename',
                      help="Path to instrument response")
    
    args = aPar.parse_args(argv)

    # Init and dump 
    dr =  FullDetectorResponse(args.filename)
  
    dr.dump()
        
        
        
