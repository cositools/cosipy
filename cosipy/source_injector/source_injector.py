from histpy import Histogram
from pathlib import Path
from cosipy.response import FullDetectorResponse
import h5py as h5
from histpy import Histogram, Axis, Axes
from cosipy.response import PointSourceResponse, ExtendedSourceResponse
import sys
from mhealpy import HealpixMap
import copy

class SourceInjector():
    
    def __init__(self, response_path, response_frame = "spacecraftframe"):

        """
        `SourceInjector` convolve response, source model(s) and orientation to produce a mocked simulated data. The data can be saved for data anlysis with cosipy.

        Parameters
        ----------
        response : str or pathlib.Path
            The path to the response file
        response_frame : str, optional
            The frame of the Compton data space (CDS) of the response. It only accepts `spacecraftframe` or "galactic". (the default is `spacecraftframe`, which means the CDS is in the detector frame.)
        """

        self.response_path =  response_path
        
        if response_frame == "spacecraftframe" or response_frame == "galactic":
            
            self.response_frame = response_frame
        
        else:
            raise ValueError("The response frame can only be `spacecraftframe` or `galactic`!")



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
        if self.response_frame == "spacecraftframe":

            if orientation == None:
                raise TypeError("The when the data are binned in spacecraftframe frame, orientation must be provided to compute the expected counts.")

            with FullDetectorResponse.open(self.response_path) as response:
                
                scatt_map = orientation.get_scatt_map(coordinate, response.nside*2, coordsys = 'galactic', earth_occ = True)
                
                psr = response.get_point_source_response(coord=coordinate, scatt_map=scatt_map)
                
        # get the point source response in galactic frame
        elif self.response_frame == "galactic":

            psr = SourceInjector.get_psr_in_galactic(coordinate = coordinate, response_path = self.response_path, spectrum = spectrum)

        injected = psr.get_expectation(spectrum)
        # setting the Em and Ei scale to linear to match the simulated data
        # The linear scale of Em is the default for COSI data
        injected.axes["Em"].axis_scale = "linear"

        if project_axes is not None:
            injected = injected.project(project_axes)

        if make_spectrum_plot is True:
            ax, plot = injected.project("Em").draw(label = "Injected point source", color = "green")
            ax.legend(fontsize=12, loc="upper right", frameon=True)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Em [keV]", fontsize=14, fontweight="bold")
            ax.set_ylabel("Counts", fontsize=14, fontweight="bold")

        if data_save_path is not None:
            injected.write(data_save_path)

        return injected

    @staticmethod
    def get_esr(source_model, response_path):
        """
        Get the extended source response from the response file.

        Parameters
        ----------
        source_model : astromodels.ExtendedSource
            The model representing the extended source.
        response_path : str or pathlib.Path
            The path to the response file.

        Returns
        -------
        esr : histpy.Histogram
            The extended source response object.
        """
        try:
            return ExtendedSourceResponse.open(response_path)
        except Exception as e:
            raise RuntimeError(f"Error loading Extended Source Response: {e}")

    def inject_extended_source(
        self,
        source_model,
        source_name="extended_source",
        data_save_path=None,
        project_axes=None,
        make_spectrum_plot=False,
    ):
        """
        Get the expected counts for an extended source.

        Parameters
        ----------
        source_model : astromodels.ExtendedSource
            The all sky model defined from an astromodels extended source model.
        source_name : str, optional
            The name of the source (the default is `extended_source`).
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

        esr = self.get_esr(source_model, self.response_path)
        injected = esr.get_expectation_from_astromodel(source_model)

        if project_axes is not None:
            injected = injected.project(project_axes)

        if make_spectrum_plot:
            ax, plot = injected.project("Em").draw(label=f"Injected {source_name}", color="blue", linewidth=2)
            ax.legend(fontsize=12, loc="upper right", frameon=True)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Em [keV]", fontsize=14, fontweight="bold")
            ax.set_ylabel("Counts", fontsize=14, fontweight="bold")

        if data_save_path is not None:
            injected.write(data_save_path)

        return injected
    
    def inject_model(self, model, orientation = None, make_spectrum_plot = False, data_save_path = None, project_axes = None):
        
        if self.response_frame == "spacecraftframe":
            if orientation == None:
                raise TypeError("The when the data are binned in spacecraftframe frame, orientation must be provided to compute the expected counts.")
                
        self.components = {}
        
        # first inject point sources
        point_sources = model.point_sources
        
        # iterate through all point sources
        for name, source in point_sources.items():
            
            injected = self.inject_point_source(spectrum = source.spectrum.main.shape, coordinate = source.position.sky_coord,
                                                orientation = orientation, source_name = name)
            
            injected.axes["Em"].axis_scale = "log"  # set to log scale manually. This inconsistency is from the detector response module
            
            self.components[name] = injected
            
        # second inject extended sources
        extended_sources = model.extended_sources
        
        # iterate through all extended sources
        for name, source in extended_sources.items():
            
            injected = self.inject_extended_source(source_model = source, source_name = name)
            self.components[name] = injected
            
        
        if len(self.components) == 1:
            
            # if there is only one component, the injected all is just the only component
            injected_all = list(self.components.values())[0]
            
            if data_save_path is not None:
                injected_all.write(data_save_path)
                
        elif len(self.components) > 1:
            
            injected_list = list(self.components.values())
            
            injected_all = copy.deepcopy(injected_list[0])
            
            # add the rest of the injected sources
            for i in injected_list[1:]:
                injected_all += i
                
        if make_spectrum_plot:
            ax, plot = injected_all.project("Em").draw(color="green", linewidth=2)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Em [keV]", fontsize=14, fontweight="bold")
            ax.set_ylabel("Counts", fontsize=14, fontweight="bold")
                
            return injected_all
        