import argparse
import textwrap

import h5py as h5

from histpy import Histogram, Axes, Axis

from cosipy.coords import SpacecraftFrame

from mhealpy import HealpixBase
import mhealpy as hp

import numpy as np

from pathlib import Path

from sparse import COO

from .DetectorResponseDirection import DetectorResponseDirection
from .healpix_axis import HealpixAxis

class DetectorResponse(HealpixBase):
    """
    DetectorResponse handles the multi-dimensional matrix that describes the
    full response of the instruments.

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
                         scheme = self._drm.attrs["SCHEME"])

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
                axes += [Axis(np.array(axis),
                              scale = axis_type,
                              label = axis_label)]

        self._axes = Axes(axes)


    @property
    def ndim(self):

        return self._axes.ndim+1
        
    def __getitem__(self, pix):

        if not isinstance(pix, (int, np.integer)) or pix < 0 or not pix < self.npix:
            raise IndexError("Pixel number out of range, or not an integer")
        
        coords = np.reshape(self._file['DRM']['BIN_NUMBERS'][pix], (self.ndim - 1,-1))
        data = np.array(self._file['DRM']['CONTENTS'][pix])

        return DetectorResponseDirection(self._axes, COO(coords = coords,
                                                         data = data,
                                                         shape = tuple(self._axes.nbins)))
        
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

    def get_interp_matrix(self, coord):

        pixels, weights = self.get_interp_weights(self, coord)

        
    
    def get_directional_response(self, coord, interp = True):
        """
        Get the 
        """
        
        
    def get_point_source_expectation(self, orientation):
        pass

    
    
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
    dr =  DetectorResponse(args.filename)
  
    dr.dump()
        
        
        
