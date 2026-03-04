from pathlib import Path

import numpy as np

import astropy.units as u
from astropy.coordinates import SkyCoord

from mhealpy import HealpixBase

from histpy import Axes

from .DetectorResponse import DetectorResponse
from .PointSourceResponse import PointSourceResponse

class GalacticResponse(HealpixBase):
    """
    Handles the multi-dimensional matrix that describes an
    all-sky response of the instrument in the galactic frame.

    FIXME: this class and FullDetectorResponse should be made children
    of a common base class to remove reundant code.  Other than the
    on-disk format differences, the main difference between this class
    and FullDetectorResponse is that the counts on disk have already
    been pre-multiplied by the effective area.

    """

    def __init__(self, *args, **kwargs):
        # Overload parent init. Called in class methods.
        pass

    @classmethod
    def open(cls, filename, dtype=None):
        """
        Load a galactic-frame response from a specified path.  The
        response is stored as a standard Histogram; we selectively
        load enough information to retrieve slices for individual
        source directions from the file as needed, rather than loading
        the whole response.

        Parameters
        ----------
        filename : string or Path
          file containing response
        dtype : numpy dtype or None
          Dtype of values to be returned when accessing response
          contents. If None, use the type stored in the file

        """

        import h5py as h5

        filename = Path(filename)

        new = cls(filename)

        new._file = h5.File(filename, mode='r')

        new._axes = Axes.open(new._file['hist/axes'])

        if new._axes[0].label != "NuLambda":
            raise RuntimeError("Galactic response must have NuLambda as its first dimension")

        # axes minus NuLambda -- used for getting pixel slices for PSRs
        new._rest_axes = new._axes[1:]

        new._unit = u.Unit(new._file['hist'].attrs['unit'])

        # Init HealpixMap (galactic coordinates, main axis)
        HealpixBase.__init__(new,
                             base=new._axes['NuLambda'],
                             coordsys="galactic")

        new._contents = new._file['hist/contents']

        # dummy effective area correction; "counts" in Histogram
        # have already had the effective area correction applied.
        dtype = new._contents.dtype if dtype is None else dtype
        new._eff_area = np.ones(new._axes["Ei"].nbins, dtype=dtype)

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
    def eff_area_correction(self):
        """
        Effective area correction for bins with each Ei.

        Returns
        -------
        :py:class:`np.ndarray`
        """

        return self._eff_area

    def __getitem__(self, pix):
        """
        Extract the portion of the response corresponding to a
        single source sky pixel on the NuLambda axis.

        Parameters
        ----------
        pix : integer
           pixel index to extract

        Returns
        -------
        A DetectorResponse containing the specified part of the full
        response.

        """

        if not isinstance(pix, (int, np.integer)):
            raise IndexError("Pixel index must be an integer")

        data = self._contents[pix].astype(self.dtype, copy=False)

        return DetectorResponse(self._rest_axes,
                                contents = data,
                                unit = self._unit,
                                copy_contents = False)

    def to_dr(self):
        """
        Load the full response in memory.

        Returns
        -------
        a DetectorResponse containing the full response.

        """

        data = np.array(self._contents, dtype=self.dtype)

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

    def get_counts(self, pix, em_slice=None):
        """
        Get count data for a given NuLambda pixel from the underlying
        HDF5 file.  Optionally return only a given slice along the Em
        axis.  Note that the counts have already been multiplied by
        the effective area.

        Parameters
        ----------
        pix : int
          NuLambda pixel to read
        em_slice: Slice, optional
          slice of the Em axis to return; None means return all

        Returns
        -------
        raw response values for specified pixel/slice
        """

        em_dim = self._rest_axes.label_to_index("Em")

        all_slice = slice(None)
        if em_slice is None:
            em_slice = all_slice

        idx  = [pix]
        idx += [all_slice] * em_dim
        idx += [em_slice]
        idx += [all_slice] * (self._rest_axes.ndim - em_dim - 1)
        idx = tuple(idx)

        return self._contents[idx].astype(self.dtype, copy=False)

    def get_point_source_response(self, source):

        """
        Get point source response (psr) corresponding to a
        given source direction in the galactic frame.

        Parameters
        ----------
        source : astropy.coordinates.SkyCoord
            Source direction in galactic frame.

        Returns
        -------
        psr : histpy.Histogram
            Point source response for source direction.

        """

        pix = self.ang2pix(source)
        data = self.get_counts(pix)

        return PointSourceResponse(self._rest_axes,
                                   contents = data,
                                   unit = self._unit,
                                   copy_contents = False)
