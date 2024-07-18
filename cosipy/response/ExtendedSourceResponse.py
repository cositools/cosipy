from histpy import Histogram, Axes, Axis
import h5py as h5
import numpy as np
import sys
import astropy.units as u 

from .functions import get_integrated_extended_model

class ExtendedSourceResponse(object):
    """
    A class to represent and manipulate extended source response data.

    This class provides methods to load data from HDF5 files, access contents,
    units, and axes information, and calculate expectations based on sky models.

    Attributes
    ----------
    _contents : astropy.units.Quantity
        The contents of the extended source response as a Quantity array
        (numpy array with astropy units).
    _unit : astropy.units.Unit
        The unit of the contents.
    _axes : Axes
        The axes object representing the dimensions of the data.

    Methods
    -------
    open(filename, name='hist')
        Load data from an HDF5 file.
    get_expectation(allsky_image_model)
        Calculate expectation based on an all-sky image model.
    get_expectation_from_astromodel(source)
        Calculate expectation from an astronomical model source.
    """

    def __init__(self, contents = None, unit = None, axes = None):
        """
        Initialize an ExtendedSourceResponse object.
        """
        self._contents = contents
        self._unit = unit
        self._axes = axes

    @property
    def contents(self):
        """
        Get the contents of the extended source response.

        Returns
        -------
        astropy.units.Quantity
            The contents of the extended source response as a Quantity array
            (numpy array with astropy units).
        """
        return self._contents

    @property
    def unit(self):
        """
        Get the unit of the contents.

        Returns
        -------
        astropy.units.Unit
            The unit of the contents.
        """
        return self._unit

    @property
    def axes(self):
        """
        Get the axes object.

        Returns
        -------
        Axes
            The axes object representing the dimensions of the data.
        """
        return self._axes

    @classmethod
    def open(cls, filename, name='hist'):
        """
        Load data from an HDF5 file.

        Parameters
        ----------
        filename : str
            The path to the HDF5 file.
        name : str, optional
            The name of the histogram group in the HDF5 file (default is 'hist').

        Returns
        -------
        ExtendedSourceResponse
            A new instance of ExtendedSourceResponse with loaded data.

        Raises
        ------
        ValueError
            If the shape of the contents does not match the axes.
        """
        new = cls()
        
        with h5.File(filename, 'r') as f:
            hist_group = f[name]
            
            # load axes
            axes_group = hist_group['axes']
            
            axes = []
            for axis in axes_group.values():
                if '__class__' in axis.attrs:
                    class_module, class_name = axis.attrs['__class__']
                    axis_cls = getattr(sys.modules[class_module], class_name)
                axes += [axis_cls._open(axis)]
            
            new._axes = Axes(axes)
            
            # load unit
            if 'unit' in hist_group.attrs:
                new._unit = u.Unit(hist_group.attrs['unit'])
                
            # load contents
            contents = np.zeros(new.axes.nbins)
            if np.all(new.axes.nbins == hist_group['contents'].shape):
                contents = hist_group['contents'][:]
            elif np.all(new.axes.nbins + 2 == hist_group['contents'].shape):
                contents = hist_group['contents'][tuple(slice(1, -1) for _ in range(len(new.axes)))]
            else:
                raise ValueError
                
            new._contents = contents * new.unit
            
        return new

    def get_expectation(self, allsky_image_model):
        """
        Calculate expectation based on an all-sky image model.

        Parameters
        ----------
        allsky_image_model : Histogram 
            The all-sky image model to use for calculation.

        Returns
        -------
        Histogram
            A histogram representing the calculated expectation.
        """
        return Histogram(edges=self.axes[2:],
                         contents=np.tensordot(allsky_image_model.contents, self.contents, axes=([0,1], [0,1])) * self.axes[0].pixarea())

    def get_expectation_from_astromodel(self, source):
        """
        Calculate expectation from an astromodels extended source model.

        This method creates an AllSkyImageModel based on the current axes configuration,
        sets its values from the provided astromodels extended source model, and then
        calculates the expectation using the get_expectation method.

        Parameters
        ----------
        source : astromodels.ExtendedSource
            An astromodels extended source model object. This model represents
            the spatial and spectral distribution of an extended astronomical source.

        Returns
        -------
        Histogram
            A histogram representing the calculated expectation based on the
            provided extended source model.
        """

        allsky_image_model = get_integrated_extended_model(source, image_axis = self.axes[0], energy_axis = self.axes[1])

        return self.get_expectation(allsky_image_model)
