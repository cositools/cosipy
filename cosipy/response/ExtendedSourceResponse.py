from histpy import Histogram, Axes, Axis
import numpy as np
import astropy.units as u
import gc

from .functions import get_integrated_extended_model

class ExtendedSourceResponse(Histogram):
    """
    A class to represent and manipulate extended source response data.

    This class provides methods to load data from HDF5 files, access contents,
    units, and axes information, and calculate expectations based on sky models.

    Methods
    -------
    get_expectation(allsky_image_model)
        Calculate expectation based on an all-sky image model.
    get_expectation_from_astromodel(source)
        Calculate expectation from an astronomical model source.

    Notes
    -----
    Currently, the axes of the response must be ['NuLambda', 'Ei', 'Em', 'Phi', 'PsiChi'].
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize an ExtendedSourceResponse object.
        """
        # Not to track the under/overflow bins
        kwargs['track_overflow'] = False

        super().__init__(*args, **kwargs)
        
        if not np.all(self.axes.labels == ['NuLambda', 'Ei', 'Em', 'Phi', 'PsiChi']):
            # 'NuLambda' should be 'lb' if it is in the gal. coordinates?
            raise ValueError(f"The input axes {self.axes.labels} is not supported by ExtendedSourceResponse class.")

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
        hist = super().open(filename, name)

        axes = hist.axes
        contents = hist[:]
        sumw2 = hist.sumw2 
        unit = hist.unit
        track_overflow = False
        
        new = cls(axes, contents = contents,
                        sumw2 = sumw2,
                        unit = unit,
                        track_overflow = track_overflow)

        if new.is_sparse:
            new = new.to_dense()
        
        del hist
        gc.collect()

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
        if self.axes[0].label == allsky_image_model.axes[0].label \
            and self.axes[1].label == allsky_image_model.axes[1].label \
            and np.all(self.axes[0].edges == allsky_image_model.axes[0].edges) \
            and np.all(self.axes[1].edges == allsky_image_model.axes[1].edges) \
            and allsky_image_model.unit == u.Unit('1/(s*cm*cm*sr)'):
            
            contents = np.tensordot(allsky_image_model.contents, self.contents, axes=([0,1], [0,1]))
            contents *= self.axes[0].pixarea()

            return Histogram(edges=self.axes[2:], contents=contents)
        
        else:
            raise ValueError(f"The input allskymodel mismatches with the extended source response.")

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
