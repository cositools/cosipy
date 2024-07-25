from histpy import Histogram
from pathlib import Path
from cosipy.response import FullDetectorResponse
import h5py as h5
from histpy import Histogram, Axis, Axes
from cosipy.response import PointSourceResponse
import sys
from mhealpy import HealpixMap

class SourceInjector():
    
    def __init__(self, response_path, response_frame = "local"):

        """
        `SourceInjector` convolve response, source model(s) and orientation to produce a mocked simulated data. The data can be saved for data anlysis with cosipy.

        Parameters
        ----------
        response : str or pathlib.Path
            The path to the response file
        response_frame : str, optional
            The frame of the Compton data space (CDS) of the response. It only accepts `local` or "galactic". (the default is `local`, which means the CDS is in the local detector frame.
        """

        self.response_path =  response_path

        if response_frame == "local" or response_frame == "galactic":

            self.response_frame = response_frame

        else:
            raise ValueError("The response frame can only be `local` or `galactic`!")


    @staticmethod
    def get_psr_in_galactic(coordinate, response_path, spectrum):
        
        """
        Get the point source response (psr) in galactic. Please be aware that you must use a galactic response!
        To do: to make the weight parameter not hardcoded
        
        Parameters
        ----------
        coordinate : astropy.coordinates.SkyCoord
            The coordinate.
        response_path : str or path.lib.Path
            The path to the response.
        spectrum : astromodels.functions
            The spectrum of the source to be placed at the hypothesis coordinate.
        
        Returns
        -------
        psr : histpy.Histogram
            The point source response of the spectrum at the hypothesis coordinate.
        """
        
        # Open the response
        # Notes from Israel: Inside it contains a single histogram with all the regular axes for a Compton Data Space (CDS) analysis, in galactic coordinates. Since there is no class yet to handle it, this is how to read in the HDF5 manually.
        
        with h5.File(response_path) as f:

            axes_group = f['hist/axes']
            axes = []
            for axis in axes_group.values():
                # Get class. Backwards compatible with version
                # with only Axis
                axis_cls = Axis
                if '__class__' in axis.attrs:
                    class_module, class_name = axis.attrs['__class__']
                    axis_cls = getattr(sys.modules[class_module], class_name)
                axes += [axis_cls._open(axis)]
        axes = Axes(axes)
        
        # get the pixel number of the hypothesis coordinate
        map_temp = HealpixMap(base = axes[0])
        coordinate_pix_number = map_temp.ang2pix(coordinate)
        
        # get the expectation for the hypothesis coordinate (a point source)
        with h5.File(response_path) as f:
            pix = coordinate_pix_number
            psr = PointSourceResponse(axes[1:], f['hist/contents'][pix+1], unit = f['hist'].attrs['unit'])
                
        return psr


    def inject_point_source(self, spectrum, coordinate, orientation = None, source_name = "point_source",
                            make_spectrum_plot = False, data_save_path = None, project_axes = None):

        """
        Get the expected counts for a point source.

        Parameters
        ----------
        spectrum : astromodels.functions
            The spectrum model defined from `astromodels`.
        coordinate : astropy.coordinates.SkyCoord
            The coordinate of the point source.
        orientation : cosipy.spacecraftfile.SpacecraftFile, option
            The orientation of the telescope during the mock simulation. This is needed when using a detector response. (the default is `None`, which means a galactic response is used.
        source_name : str, optional
            The name of the source (the default is `point_source`).
        make_spectrum_plot : bool, optional
            Set `True` to make the plot of the injected spectrum.
        data_save_path : str or pathlib.Path, optional
            The path to save the injected data to a `.h5` file. This should include the file name. (the default is `None`, which means the injected data won't be saved.
        project_axes : list, optional
            The axes to project before saving the data file (the default is `None`, which means the data won't be projected).
            
        Returns
        -------
        histpy.Histogram
            The `Histogram object of the injected spectrum.`
        """
        
        
        # get the point source response in local frame
        if self.response_frame == "local":

            if orientation == None:
                raise TypeError("The when the data are binned in local frame, orientation must be provided to compute the expected counts.")
                
            # get the dwell time map
            coord_in_sc_frame = orientation.get_target_in_sc_frame(target_name = source_name, 
                                                                   target_coord = coordinate, 
                                                                   quiet = True)
            
            # get the dwell time map in the detector frame
            dwell_time_map = orientation.get_dwell_map(response = self.response_path)

            with FullDetectorResponse.open(self.response_path) as response:
                psr = response.get_point_source_response(dwell_time_map)

        # get the point source response in galactic frame
        elif self.response_frame == "galactic":

            psr = SourceInjector.get_psr_in_galactic(coordinate = coordinate, response_path = self.response_path, spectrum = spectrum)

        injected = psr.get_expectation(spectrum)

        if project_axes is not None:
            injected = injected.project(project_axes)

        if make_spectrum_plot is True:
            ax, plot = injected.project("Em").draw(label = "Injected point source", color = "green")
            ax.legend()
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylabel("Counts")

        if data_save_path is not None:
            injected.write(data_save_path)

        return injected

        


