import numpy as np

import astropy.units as u

from astromodels import Parameter

from threeML import PluginPrototype

from cosipy.response import (
    FullDetectorResponse,
    ExtendedSourceResponse
)
from cosipy.polarization.conventions import (
    IAUPolarizationConvention,
    MEGAlibRelativeX,
    MEGAlibRelativeY,
    MEGAlibRelativeZ
)

import logging
logger = logging.getLogger(__name__)

class COSILike(PluginPrototype):
    """COSI 3ML plugin.

    Parameters
    ----------
    name : str
        Plugin name e.g. "cosi". Needs to have a distinct name with
        respect to other plugins in the same analysis
    dr : str
        Path to full detector response
    data : histpy.Histogram
        Binned data. Note: Eventually this should be a cosipy data
        class
    bkg : histpy.Histogram
        Binned background model. Note: Eventually this should be a
        cosipy data class
    sc_orientation : cosipy.spacecraftfile.SpacecraftFile
        Contains the information of the orientation: timestamps
        (astropy.Time) and attitudes (scoord.Attitude) that describe
        the spacecraft for the duration of the data included in the
        analysis
    nuisance_param : astromodels.core.parameter.Parameter, optional
        Background parameter
    coordsys : str, optional
        Coordinate system (name of an inertial frame, such as
        'galactic', or 'spacecraftframe') to perform fit in, which
        should match coordinate system of data and background. This
        only needs to be specified if the binned data and background
        do not have a coordinate system attached to them
    precomputed_psr_file : str, optional
        Full path to precomputed point source response in Galactic
        coordinates
    earth_occ : bool, optional
        Option to include Earth occultation in fit (default is True).
    response_pa_convention : str, optional
        Polarization reference convention of response ('RelativeX',
        'RelativeY', or 'RelativeZ'). Required if response contains
        polarization angle axis

    """
    def __init__(self, name, dr, data, bkg, sc_orientation,
                 nuisance_param = None,
                 coordsys = None,
                 precomputed_psr_file = None,
                 earth_occ=True,
                 response_pa_convention=None,
                 **kwargs):

        def prepare_binned(data, cds_order):
            """Given a Histogram describing binned data/model, convert it to a
            bare array.  Make sure it is dense, lacks units, and
            contains only CDS axes in the order specified.

            """

            # we *must* project before densifying, as some
            # data sets are too big to densify otherwise
            if tuple(data.axes.labels) != tuple(cds_order):
                data = data.project(cds_order)

            data = data.todense() if data.is_sparse else data

            data = data.contents

            if isinstance(data, u.Quantity):
                data = data.value

            return data

        # create the hash for the nuisance parameters. We have none for now.
        self._nuisance_parameters = {}

        # call the prototype constructor. Boilerplate.
        super(COSILike, self).__init__(name, self._nuisance_parameters)

        # User inputs needed to compute the likelihood
        self._name = name
        self._dr = FullDetectorResponse.open(dr, pa_convention=response_pa_convention)
        self._sc_orientation = sc_orientation
        self.earth_occ = earth_occ

        # Option to use precomputed point source response.
        # Note: this still needs to be implemented in a
        # consistent way for point srcs and extended srcs.
        self.precomputed_psr_file = precomputed_psr_file
        if self.precomputed_psr_file is not None:
            logger.info("... loading the pre-computed image response ...")
            self.image_response = ExtendedSourceResponse.open(self.precomputed_psr_file)
            logger.info("--> done")

        cds_order = tuple(self._dr.axes.labels[-3:])
        if not all(ax in ("Em", "Phi", "PsiChi") for ax in cds_order):
            raise ValueError("Response CDS axes must be Em/Phi/PsiChi")

        self._data = prepare_binned(data, cds_order)
        self._bkg  = prepare_binned(bkg, cds_order)

        try:
            data_frame = data.axes["PsiChi"].coordsys.name
            bkg_frame  = bkg.axes["PsiChi"].coordsys.name
            if data_frame != bkg_frame:
                raise RuntimeError(f"Data is binned in {data_frame}, while background is binned in {bkg_frame}. Coordinate systems must match." )
            else:
                self._coordsys = data_frame
        except:
            if coordsys is None:
                raise RuntimeError("There is no coordinate system attached to the binned data. One must be provided by specifying either 'spacecraftframe' or an inertial frame for the coordsys argument.")
            else:
                self._coordsys = coordsys

        # Set to fit nuisance parameter if given by user
        if nuisance_param is None:
            self.set_inner_minimization(False)
        elif isinstance(nuisance_param, Parameter):
            self.set_inner_minimization(True)
            self._bkg_par = nuisance_param
            self._bkg_par.free = self._fit_nuisance_params
            self._nuisance_parameters[self._bkg_par.name] = self._bkg_par
        else:
            raise RuntimeError("Nuisance parameter must be astromodels.core.parameter.Parameter object")

        if 'Pol' in self._dr.axes.labels:
            self._response_pa_convention = response_pa_convention
            if self._coordsys == 'spacecraftframe':
                att = self._sc_orientation.get_attitude()[0]
                if self._response_pa_convention == 'RelativeX':
                    self._pa_convention = MEGAlibRelativeX(attitude=att)
                elif self._response_pa_convention == 'RelativeY':
                    self._pa_convention = MEGAlibRelativeY(attitude=att)
                elif self._response_pa_convention == 'RelativeZ':
                    self._pa_convention = MEGAlibRelativeZ(attitude=att)
                else:
                    raise RuntimeError("Response convention must be 'RelativeX', 'RelativeY', or 'RelativeZ'")
            elif self._coordsys == 'galactic':
                self._pa_convention = IAUPolarizationConvention()
            else:
                raise RuntimeError("Unknown coordinate system")
        else:
            self._response_pa_convention = None

        # Temporary fix to only print log-likelihood warning once max per fit
        self._printed_warning = False

        self._model = None

    def set_model(self, model):
        """
        Set the model to be used in the joint minimization and discard
        any cached information from prior models.  This function
        *must* be called at least once before computing the log
        likelihood with get_log_like(), as well as any time the model
        structure (as opposed to its free parameters' values) changes.
        It is called once, automatically when constructing the
        JointLikelihood object.

        """

        point_sources = model.point_sources
        extended_sources = model.extended_sources

        # cached per-source information
        self._expected_counts = {} # used externally

        # used only for point sources internally
        self._source_location = {}
        self._psr = {}

        for name in point_sources:
            self._source_location[name] = None
            self._psr[name] = None
            self._expected_counts[name] = None

        for name in extended_sources:
             self._expected_counts[name] = None

        self._model = model

    def compute_expectation(self, model):
        """
        Compute the total expected counts of the model

        Parameters
        ----------
        model : astromodels.core.model.Model
            Any model supported by astromodels

        Returns
        -------
        signal : total expected counts
        """

        def get_point_source_coords(source):
            """
            Extract the sky position of a point source.  Get the coordinate
            values in a generic way so that we aren't dependent on
            whether the position was stored in galactic or ICRS.
            """
            pos = source.position
            return tuple(p.value for p in pos.parameters.values())

        signal = None

        # Get expectation for extended sources
        for name, source in model.extended_sources.items():
            # Set spectrum
            # Note: the spectral parameters are updated internally by 3ML
            # during the likelihood scan.

            # Get expectation using precomputed psr in Galactic coordinates
            total_expectation = \
                self.image_response.get_expectation_from_astromodel(source)

            # Save expected counts for each source,
            # in order to enable easy plotting after likelihood scan
            self._expected_counts[name] = total_expectation

            # extract expectation from source as raw numpy array
            if total_expectation.is_sparse:
                total_expectation = total_expectation.contents.todense()
            elif total_expectation.unit is not None:
                total_expectation = total_expectation.contents.value
            else:
                total_expectation = total_expectation.contents

            # Add source to signal
            if signal is None:
                signal = total_expectation.copy()
            else:
                signal += total_expectation

        # Get expectation for point sources
        for name, source in model.point_sources.items():
            src_coords = get_point_source_coords(source)

            if src_coords != self._source_location[name]: # source loc changed
                logger.info(f"... Re-calculating the point source response of {name} ...")

                self._source_location[name] = src_coords

                coord = source.position.sky_coord

                if self._coordsys == 'spacecraftframe':
                    dwell_time_map = self._get_dwell_time_map(coord)
                    self._psr[name] = self._dr.get_point_source_response(exposure_map=dwell_time_map)
                else:
                    coord = coord.transform_to(self._coordsys)
                    scatt_map = self._get_scatt_map(coord)
                    self._psr[name] = self._dr.get_point_source_response(coord=coord, scatt_map=scatt_map)

                logger.info(f"--> done (source name : {name})")

            # Convolve with spectrum
            # See also the Detector Response and Source Injector tutorials
            if hasattr(source.spectrum, 'main'):

                spectrum = source.spectrum.main.shape
                total_expectation = self._psr[name].get_expectation(spectrum)

            else:

                component_counter = 0

                for item in source.spectrum.to_dict():

                    spectrum = getattr(source.spectrum, item).shape

                    if not 'Pol' in self._dr.axes.labels:
                        this_expectation = self._psr[name].get_expectation(spectrum)
                    else:
                        if self._coordsys == 'spacecraftframe':
                            this_expectation = self._psr[name].get_expectation(spectrum, source.components[item].polarization)
                        elif self._coordsys == 'galactic':
                            this_expectation = self._psr[name].get_expectation(spectrum, source.components[item].polarization)
                        else:
                            raise RuntimeError("Unknown coordinate system")

                    if component_counter == 0:
                        total_expectation = this_expectation
                    else:
                        total_expectation += this_expectation

                    component_counter += 1

            # Save expected counts for each source,
            # in order to enable easy plotting after likelihood scan
            self._expected_counts[name] = total_expectation

            # extract expectation from source as raw numpy array
            if total_expectation.is_sparse:
                total_expectation = total_expectation.contents.todense()
            elif total_expectation.unit is not None:
                total_expectation = total_expectation.contents.value
            else:
                total_expectation = total_expectation.contents

            # Add source to signal
            if signal is None:
                signal = total_expectation.copy()
            else:
                signal += total_expectation

        return signal

    def get_log_like(self):
        """
        Calculate the log-likelihood.

        Returns
        ----------
        log_like : float
            Value of the log-likelihood
        """

        if self._model is None:
            raise ValueError("Must set model before computing likelihood!")

        expectation = self.compute_expectation(self._model)

        if self._fit_nuisance_params:
            # Compute expectation including free background parameter
            nv = self._nuisance_parameters[self._bkg_par.name].value
            expectation += nv * self._bkg
        else:
            # Compute expectation without background parameter
            expectation += self._bkg

        # avoid -infinite log-likelihood (occurs when expected counts
        # = 0 but data != 0)
        expectation += 1e-12

        if not self._printed_warning:
            # This 1e-12 should be defined as a parameter in the near
            # future (HY)
            logger.warning("Adding 1e-12 to each bin of the expectation to avoid log-likelihood = -inf.")
            self._printed_warning = True

        # Compute the log-likelihood:
        log_like = np.nansum(self._data * np.log(expectation) - expectation)

        return log_like

    def inner_fit(self):
        """
        Required for 3ML fit.

        In theory, this function is called to optimize the nuisance
        parameters given fixed values of the model parameters.  But
        in fact, it is called on every iteration, so the nuisance
        params are treated as "just another parameter" to optimize.
        """

        return self.get_log_like()

    def _get_dwell_time_map(self, coord):
        """
        Get the dwell time map of the source in the inertial (spacecraft)
        frame.

        Parameters
        ----------
        coord : astropy.coordinates.SkyCoord
            Coordinates of the target source

        Returns
        -------
        dwell_time_map : mhealpy.containers.healpix_map.HealpixMap
            Dwell time map

        """

        src_path = self._sc_orientation.get_target_in_sc_frame(coord)
        dwell_time_map = \
            self._sc_orientation.get_dwell_map(base = self._dr,
                                               src_path = src_path)

        return dwell_time_map

    def _get_scatt_map(self, coord):
        """
        Get the spacecraft attitude map of the source in the inertial
        (spacecraft) frame.

        Parameters
        ----------
        coord : astropy.coordinates.SkyCoord
            The coordinates of the target object.

        Returns
        -------
        scatt_map : cosipy.spacecraftfile.scatt_map.SpacecraftAttitudeMap

        """

        scatt_map = \
            self._sc_orientation.get_scatt_map(nside = self._dr.nside * 2,
                                               target_coord = coord,
                                               earth_occ = self.earth_occ)

        return scatt_map

    def set_inner_minimization(self, flag: bool):
        """
        Turn on the minimization of the internal COSI (nuisance) parameters.

        Parameters
        ----------
        flag : bool
            Turns on and off the minimization of the internal parameters
        """

        self._fit_nuisance_params: bool = bool(flag)

        for parameter in self._nuisance_parameters:
            self._nuisance_parameters[parameter].free = self._fit_nuisance_params
