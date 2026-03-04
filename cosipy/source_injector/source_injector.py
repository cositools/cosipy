import numpy as np
from cosipy.response import (
    GalacticResponse,
    FullDetectorResponse,
    PointSourceResponse,
    ExtendedSourceResponse
)
import logging
logger = logging.getLogger(__name__)

class SourceInjector():

    def __init__(self, response_path, response_frame="spacecraftframe", pa_convention=None):
        """
        `SourceInjector` convolve response, source model(s) and
        orientation to produce a mocked simulated data. The data can
        be saved for data anlysis with cosipy.

        Parameters
        ----------
        response : str or pathlib.Path
            The path to the response file
        response_frame : str, optional
            The frame of the Compton data space (CDS) of the
            response. It only accepts `spacecraftframe` or
            "galactic". (the default is `spacecraftframe`, which means
            the CDS is in the detector frame.)
        pa_convention : str, optional
            Polarization angle convention for polarization-enabled
            detector-frame responses. Must be one of ('RelativeX',
            'RelativeY', 'RelativeZ') when the response includes a
            `Pol` axis and `response_frame="spacecraftframe"`.

        """

        self.response_path = response_path

        if response_frame == "spacecraftframe" or response_frame == "galactic":
            self.response_frame = response_frame
        else:
            raise ValueError("The response frame can only be `spacecraftframe` or `galactic`!")

        self.pa_convention = pa_convention

    @staticmethod
    def get_psr_in_galactic(coordinate, response_path):
        """
        Given a galactic-frame response, return the PSR corresponding
        to the source pixel that contains a given coordinate.

        Parameters
        ----------
        coordinate : astropy.coordinates.SkyCoord
            The coordinate.
        response_path : str or path.lib.Path
            The path to the response.

        Returns
        -------
        psr : histpy.Histogram
            The point source response of the spectrum at the
            coordinate.

        """

        rsp = GalacticResponse.open(response_path)

        return rsp.get_point_source_response(coordinate)


    def inject_point_source(self, spectrum, coordinate,
                            orientation=None,
                            source_name="point_source",
                            make_spectrum_plot=False,
                            make_PsiChi_plot=False,
                            data_save_path=None,
                            project_axes=None,
                            polarization=None):
        """
        Get the expected counts for a point source.

        Parameters
        ----------
        spectrum : astromodels.functions
            The spectrum model defined from `astromodels`.
        coordinate : astropy.coordinates.SkyCoord
            The coordinate of the point source.
        orientation : cosipy.spacecraftfile.SpacecraftFile, optional
            The orientation of the telescope during the mock
            simulation. This is needed when using a detector
            response. (the default is `None`, which means a galactic
            response is used.
        source_name : str, optional
            The name of the source (the default is `point_source`).
        make_spectrum_plot : bool, optional
            Set `True` to make the plot of the injected spectrum.
        make_PsiChi_plot : bool, optional
            Set `True` to make the plot of the PsiChi map (galactic).
        data_save_path : str or pathlib.Path, optional
            The path to save the injected data to a `.h5` file. This
            should include the file name. (the default is `None`,
            which means the injected data won't be saved.
        project_axes : list, optional
            The axes to project before saving the data file (the
            default is `None`, which means the data won't be
            projected).
        polarization : astromodels.core.polarization.LinearPolarization, optional
            The polarization model (angle and degree). The angle is
            assumed to have the same convention as the point source
            response. If the response does not include a `Pol` axis,
            the injector will fall back to an unpolarized expectation.

        Returns
        -------
        histpy.Histogram
            The `Histogram object of the injected source.`

        """

        if self.response_frame == "spacecraftframe":

            # get the point source response in local frame
            if orientation is None:
                raise TypeError("When the data are binned in "
                                "spacecraftframe frame, "
                                "orientation must "
                                "be provided to compute the "
                                "expected counts.")

            # If the response includes a polarization axis, FullDetectorResponse requires an explicit polarization-angle convention.
            if self.pa_convention is None:
                response = FullDetectorResponse.open(self.response_path)
            else:
                response = FullDetectorResponse.open(self.response_path, pa_convention=self.pa_convention)

            with response as response:

                scatt_map = orientation.get_scatt_map(response.nside * 2,
                                                      target_coord=coordinate,
                                                      earth_occ=True)

                psr = response.get_point_source_response(coord=coordinate,
                                                         scatt_map=scatt_map)

        else:  # self.response_frame == "galactic":

            # get the point source response in galactic frame
            psr = SourceInjector.get_psr_in_galactic(coordinate=coordinate,
                                                     response_path=self.response_path)

        # Convolve response with spectrum (If the response does not include a polarization axis (`Pol`), fall back to unpolarized expectation.)
        polarization_to_use = polarization
        if polarization is not None and 'Pol' not in psr.axes.labels:
            raise RuntimeError(
                "Polarization was specified, but the response has no 'Pol' axis. "
                "Use a polarization-capable response to inject a polarized source."
            )



        injected = psr.get_expectation(spectrum, polarization=polarization_to_use)

        # Set the Em (and Ei) scale to linear to match the simulated
        # data. The linear scale of Em is the default for COSI data.
        # Because Histograms can share Axis objects, we must copy the
        # Axis before modifying it and then replace it in the
        # Histogram's Axes object.

        em_axis = injected.axes["Em"].copy()
        em_axis.axis_scale = "linear"
        injected.axes.set("Em", em_axis, copy=False)

        if project_axes is not None:
            injected = injected.project(project_axes)

        if make_spectrum_plot:
            ax, plot = injected.project("Em").draw(label="Injected point source",
                                                   color="green")
            ax.legend(fontsize=12, loc="upper right", frameon=True)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Em [keV]", fontsize=14, fontweight="bold")
            ax.set_ylabel("Counts", fontsize=14, fontweight="bold")

        if make_PsiChi_plot:
            plot, ax = injected.project('PsiChi').plot(coord='G',
                                                       ax_kw={'coord': 'G'})
            ax.get_figure().set_figwidth(4)
            ax.get_figure().set_figheight(3)

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


    def inject_extended_source(self, source_model,
                               source_name="extended_source",
                               make_spectrum_plot=False,
                               make_PsiChi_plot=False,
                               data_save_path=None,
                               project_axes=None):
        """
        Get the expected counts for an extended source.

        Parameters
        ----------
        source_model : astromodels.ExtendedSource
            The all sky model defined from an astromodels extended
            source model.
        source_name : str, optional
            The name of the source (the default is `extended_source`).
        make_spectrum_plot : bool, optional
            Set `True` to make the plot of the injected spectrum.
        make_PsiChi_plot : bool, optional
            Set `True` to make the plot of the PsiChi map (galactic).
        data_save_path : str or pathlib.Path, optional
            The path to save the injected data to a `.h5` file. This
            should include the file name. (the default is `None`,
            which means the injected data won't be saved.
        project_axes : list, optional
            The axes to project before saving the data file (the
            default is `None`, which means the data won't be
            projected).

        Returns
        -------
        histpy.Histogram
            The `Histogram object of the injected source.`

        """

        esr = self.get_esr(source_model, self.response_path)
        injected = esr.get_expectation_from_astromodel(source_model)

        if project_axes is not None:
            injected = injected.project(project_axes)

        if make_spectrum_plot:
            ax, plot = injected.project("Em").draw(label=f"Injected {source_name}",
                                                   color="blue", linewidth=2)
            ax.legend(fontsize=12, loc="upper right", frameon=True)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Em [keV]", fontsize=14, fontweight="bold")
            ax.set_ylabel("Counts", fontsize=14, fontweight="bold")

        if make_PsiChi_plot:
            plot, ax = injected.project('PsiChi').plot(coord='G',
                                                       ax_kw={'coord': 'G'})
            ax.get_figure().set_figwidth(4)
            ax.get_figure().set_figheight(3)

        if data_save_path is not None:
            injected.write(data_save_path)

        return injected


    def inject_model(self, model,
                     orientation=None,
                     make_spectrum_plot=False,
                     make_PsiChi_plot=False,
                     data_save_path=None,
                     project_axes=None,
                     polarization=None,
                     fluctuate=True):
        """
        Build an injected source by combining all the sources in a
        model.  Each injected source is stored by name in a dictionary
        self.components.  Return the expected counts for all injected
        sources combined.

        Note that this method only makes sense if all the models use
        the same underlying response.  It is not possible to combine,
        e.g., a point source using an instrument-frame response with
        an extended source using a galactic response.

        Parameters
        ----------
        model : astromodels.Model
            The all-sky model defined from an astromodels
            source model containing one or more components.
        make_spectrum_plot : bool, optional
            Set `True` to make the plot of the injected spectrum.
        make_PsiChi_plot : bool, optional
            Set `True` to make the plot of the PsiChi map (galactic).
        data_save_path : str or pathlib.Path, optional
            The path to save the injected data to a `.h5` file. This
            should include the file name. (the default is `None`,
            which means the injected data won't be saved.
        project_axes : list, optional
            The axes to project before saving the data file (the
            default is `None`, which means the data won't be
            projected).
        polarization : astromodels.core.polarization.LinearPolarization, optional
            A single polarization hypothesis applied to all injected
            point sources. If a given point source response does not
            include a `Pol` axis, the injector will fall back to an
            unpolarized expectation for that source.
        fluctuate : bool,optional
            Add poisson fluctuations on the injected source. 
            The default value is set to True.

        Returns
        -------
        histpy.Histogram
            The `Histogram object of the combined injected sources.`

        """

        if self.response_frame == "spacecraftframe":
            # get the point source response in local frame
            if orientation is None:
                raise TypeError("When the data are binned in "
                                "spacecraftframe frame, "
                                "orientation must "
                                "be provided to compute the "
                                "expected counts.")

        self.components = {}

        # first, inject point sources
        point_sources = model.point_sources

        # iterate through all point sources
        for name, source in point_sources.items():

            injected = self.inject_point_source(spectrum=source.spectrum.main.shape,
                                                coordinate=source.position.sky_coord,
                                                orientation=orientation,
                                                source_name=name,
                                                project_axes=project_axes,
                                                polarization=polarization)

            # set to log scale manually. This inconsistency is from
            # the detector response module
            em_axis = injected.axes["Em"].copy()
            em_axis.axis_scale = "log"
            injected.axes.set("Em", em_axis, copy=False)

            self.components[name] = injected

        # second, inject extended sources
        extended_sources = model.extended_sources

        # iterate through all extended sources
        for name, source in extended_sources.items():

            injected = self.inject_extended_source(source_model=source,
                                                   source_name=name,
                                                   project_axes=project_axes)
            self.components[name] = injected

        # combine all the sources
        injected_all = None
        for component in self.components.values():

            if injected_all is None:
                injected_all = component.copy()
            else:
                injected_all += component

        if fluctuate :
            injected_all[:] = np.random.poisson(injected_all)
                         
        if data_save_path is not None:
            injected_all.write(data_save_path)

        if make_spectrum_plot:
            ax, plot = injected_all.project("Em").draw(color="green",
                                                       linewidth=2)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Em [keV]", fontsize=14, fontweight="bold")
            ax.set_ylabel("Counts", fontsize=14, fontweight="bold")

        if make_PsiChi_plot:
            plot, ax = injected_all.project('PsiChi').plot(coord='G',
                                                           ax_kw={'coord': 'G'})
            ax.get_figure().set_figwidth(4)
            ax.get_figure().set_figheight(3)
            
        return injected_all
