import numpy as np
from .conventions import PolarizationConvention
from astropy import units as u

from .polarization_angle import PolarizationAngle

from histpy import Axis


class PolarizationAxis(Axis):
    """
    Defines a polarization axis compatible with PolarizationAngle.

    Parameters:
    edges (array-like):
        Bin edges. Can be a Quantity array or PolarizationAngle
    convention : PolarizationConvention
        Convention defining the polarization basis in
        the polarization plane (for which the source direction is normal).
        Overrides the convention of "edges", if a PolarizationAngle object
        was provided
    label (str): Label for axis. If edges is an Axis object, this will
        override its label
    unit  (unit-like): Unit for axis (will override unit of edges)
    copy (bool): True if edge array should be distinct from passed-in
                 edges; if False, will use same edge array if possible
    *args, **kwargs
        Passed to convention class.
    """

    def __init__(self,
                 edges,
                 convention = 'iau',
                 label = None,
                 unit = None,
                 copy=True):

        if isinstance(edges, PolarizationAngle):
            convention = edges.convention if convention is None else convention
            edges = edges.angle

        self._convention = PolarizationConvention.get_convention(convention)

        super().__init__(edges, label = label, scale='linear', unit=unit, copy=copy)

        if self.unit is None:
            raise ValueError("PolarizationAxis needs edges with units")

    @property
    def convention(self):
        return self._convention

    def _copy(self, edges=None, copy_edges=True):
        """Make a deep copy of a HealpixAxis, optionally
        replacing edge array. (The superclass's _copy
        method handles edge replacement.)
        """

        new = super()._copy(edges, copy_edges)

        # self._convention is not copied. It's safe to share it.

        return new

    def _standardize_value(self, value):
        if isinstance(value, PolarizationAngle):
            # Transform to axis' convention
            return value.transform_to(self.convention).angle
        else:
            return value

    def find_bin(self, value, right = False):
        return super().find_bin(self._standardize_value(value), right = right)

    def interp_weights(self, values):
        return super().interp_weights(self._standardize_value(values))

    def interp_weights_edges(self, values):
        return super().interp_weights_edges(self._standardize_value(values))

    @property
    def lower_bounds(self):
        return PolarizationAngle(super().lower_bounds, convention=self.convention)

    @property
    def upper_bounds(self):
        return PolarizationAngle(super().upper_bounds, convention=self.convention)

    @property
    def bounds(self):
        return PolarizationAngle(super().bounds, convention=self.convention)

    @property
    def lo_lim(self):
        return PolarizationAngle(super().lo_lim, convention=self.convention)

    @property
    def hi_lim(self):
        return PolarizationAngle(super().hi_lim, convention=self.convention)

    @property
    def edges(self):
        return PolarizationAngle(super().edges, convention=self.convention)

    @property
    def centers(self):
        return PolarizationAngle(super().centers, convention=self.convention)

    def _write_metadata(self, axis_set):
        """
        Save extra metadata to existing dataset
        """

        super()._write_metadata(axis_set)

        convention = PolarizationConvention.get_convention_registered_name(self._convention)

        if convention is None:
            raise RuntimeError(f"Only PolarizationAxis object with a registered named convention "
                               "can be saved disk")

        axis_set.attrs['convention'] = convention

    @classmethod
    def _open(cls, dataset):
        """
        Create Axis from HDF5 dataset
        Written as a virtual constructor so that
        subclasses may override
        """


        edges = np.asarray(dataset)

        metadata = cls._open_metadata(dataset)

        new = cls.__new__(cls)
        PolarizationAxis.__init__(new,
                                  edges = edges,
                                  unit = metadata['unit'],
                                  convention = metadata['convention'],
                                  label = metadata['label'],
                                  copy = False)

        return new

    @classmethod
    def _open_metadata(cls, dataset):
        """
        Returns unit, label and scale as a dictionary
        """

        metadata = super()._open_metadata(dataset)

        metadata['convention'] = dataset['convention']

        return metadata





