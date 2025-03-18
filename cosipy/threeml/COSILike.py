from threeML import PluginPrototype
from threeML.minimizer import minimization
from threeML.config.config import threeML_config
from threeML.exceptions.custom_exceptions import FitFailed
from astromodels import Parameter

from cosipy.response.FullDetectorResponse import FullDetectorResponse
from cosipy.response.ExtendedSourceResponse import ExtendedSourceResponse
from cosipy.image_deconvolution import AllSkyImageModel
from cosipy.polarization.polarization_angle import PolarizationAngle
from cosipy.polarization.conventions import IAUPolarizationConvention, MEGAlibRelativeX, MEGAlibRelativeY, MEGAlibRelativeZ

from scoords import SpacecraftFrame, Attitude

from mhealpy import HealpixMap

from cosipy.response import PointSourceResponse, DetectorResponse
from histpy import Histogram
import h5py as h5
from histpy import Axis, Axes
import sys

import astropy.units as u
import astropy.coordinates as coords

from sparse import COO

import numpy as np

from scipy.special import factorial

import collections

import copy

import logging
logger = logging.getLogger(__name__)

import inspect

class COSILike(PluginPrototype):
    """
    COSI 3ML plugin.

    Parameters
    ----------
    name : str
        Plugin name e.g. "cosi". Needs to have a distinct name with respect to other plugins in the same analysis
    dr : str
        Path to full detector response
    data : histpy.Histogram
        Binned data. Note: Eventually this should be a cosipy data class
    bkg : histpy.Histogram
        Binned background model. Note: Eventually this should be a cosipy data class
    sc_orientation : cosipy.spacecraftfile.SpacecraftFile
        Contains the information of the orientation: timestamps (astropy.Time) and attitudes (scoord.Attitude) that describe
        the spacecraft for the duration of the data included in the analysis
    nuisance_param : astromodels.core.parameter.Parameter, optional
        Background parameter
    coordsys : str, optional
        Coordinate system ('galactic' or 'spacecraftframe') to perform fit in, which should match coordinate system of data 
        and background. This only needs to be specified if the binned data and background do not have a coordinate system 
        attached to them
    precomputed_psr_file : str, optional
        Full path to precomputed point source response in Galactic coordinates
    earth_occ : bool, optional
        Option to include Earth occultation in fit (default is True).
    response_pa_convention : str, optional
        Polarization reference convention of response ('RelativeX', 'RelativeY', or 'RelativeZ'). Required if response contains polarization angle axis
    """
    def __init__(self, name, dr, data, bkg, sc_orientation, 
                 nuisance_param=None, coordsys=None, precomputed_psr_file=None, earth_occ=True, response_pa_convention=None, **kwargs):
        
        # create the hash for the nuisance parameters. We have none for now.
        self._nuisance_parameters = collections.OrderedDict()

        # call the prototype constructor. Boilerplate.
        super(COSILike, self).__init__(name, self._nuisance_parameters)

        # User inputs needed to compute the likelihood
        self._name = name
        self._rsp_path = dr
        self._dr = FullDetectorResponse.open(dr)
        self._data = data
        self._bkg = bkg
        self._sc_orientation = sc_orientation
        self.earth_occ = earth_occ
        
        try:
            if data.axes["PsiChi"].coordsys.name != bkg.axes["PsiChi"].coordsys.name:
                raise RuntimeError("Data is binned in " + data.axes["PsiChi"].coordsys.name + " and background is binned in " 
                                   + bkg.axes["PsiChi"].coordsys.name + ". They should be binned in the same coordinate system.")
            else:
                self._coordsys = data.axes["PsiChi"].coordsys.name
        except:
            if coordsys == None:
                raise RuntimeError(f"There is no coordinate system attached to the binned data. One must be provided by " 
                                   f"specifiying coordsys='galactic' or 'spacecraftframe'")
            else:
                self._coordsys = coordsys
            
        # Place-holder for cached data.
        self._model = None
        self._source = None
        self._psr = None
        self._signal = None
        self._expected_counts = None 

        # Set to fit nuisance parameter if given by user
        if nuisance_param == None:
            self.set_inner_minimization(False)
        elif isinstance(nuisance_param, Parameter):
            self.set_inner_minimization(True)
            self._bkg_par = nuisance_param
            self._nuisance_parameters[self._bkg_par.name] = self._bkg_par
            self._nuisance_parameters[self._bkg_par.name].free = self._fit_nuisance_params
        else:
            raise RuntimeError("Nuisance parameter must be astromodels.core.parameter.Parameter object")
        
        # Option to use precomputed point source response.
        # Note: this still needs to be implemented in a 
        # consistent way for point srcs and extended srcs. 
        self.precomputed_psr_file = precomputed_psr_file
        if self.precomputed_psr_file != None:
            logger.info("... loading the pre-computed image response ...")
            self.image_response = ExtendedSourceResponse.open(self.precomputed_psr_file)
            logger.info("--> done")

        if 'Pol' in self._dr.axes.labels:
            self._response_pa_convention = response_pa_convention
            if self._coordsys == 'spacecraftframe':
                if self._response_pa_convention == 'RelativeX':
                    self._pa_convention = MEGAlibRelativeX(attitude=self._sc_orientation.get_attitude()[0])
                elif self._response_pa_convention == 'RelativeY':
                    self._pa_convention = MEGAlibRelativeY(attitude=self._sc_orientation.get_attitude()[0])
                elif self._response_pa_convention == 'RelativeZ':
                    self._pa_convention = MEGAlibRelativeZ(attitude=self._sc_orientation.get_attitude()[0])
                else:
                    raise RuntimeError("Response convention must be 'RelativeX', 'RelativeY', or 'RelativeZ'")
            elif self._coordsys == 'galactic':
                self._pa_convention = IAUPolarizationConvention()
            else:
                raise RuntimeError("Unknown coordinate system")
        
    def set_model(self, model):
        """
        Set the model to be used in the joint minimization.
        
        Parameters
        ----------
        model : astromodels.core.model.Model
            Any model supported by astromodels
        """
        
        # Temporary fix to only print log-likelihood warning once max per fit
        if inspect.stack()[1][3] == '_assign_model_to_data':
            self._printed_warning = False
    
        # Get point sources and extended sources from model: 
        point_sources = model.point_sources
        extended_sources = model.extended_sources
        
        # Source counter for models with multiple sources:
        self.src_counter = 0
        
        # Get expectation for extended sources:
        
        # Save expected counts for each source,
        # in order to enable easy plotting after likelihood scan:
        if self._expected_counts == None:
            self._expected_counts = {}

        for name,source in extended_sources.items():

            # Set spectrum:
            # Note: the spectral parameters are updated internally by 3ML
            # during the likelihood scan. 

            # Get expectation using precomputed psr in Galactic coordinates:
            total_expectation = self.image_response.get_expectation_from_astromodel(source)

            # Save expected counts for source:
            self._expected_counts[name] = copy.deepcopy(total_expectation)

            # Need to check if self._signal type is dense (i.e. 'Quantity') or sparse (i.e. 'COO').
            if type(total_expectation.contents) == u.quantity.Quantity:
                total_expectation = total_expectation.contents.value
            elif type(total_expectation.contents) == COO:
                total_expectation = total_expectation.contents.todense() 
            else:
                raise RuntimeError("Expectation is an unknown object")

            # Add source to signal and update source counter:
            if self.src_counter == 0:
                self._signal = total_expectation
            if self.src_counter != 0:
                self._signal += total_expectation
            self.src_counter += 1

        # Initialization
        # probably it is better that this part be outside of COSILike (HY).
        if len(point_sources) != 0:
        
            if self._psr is None or len(point_sources) != len(self._psr):

                logger.info("... Calculating point source responses ...")

                self._psr = {}
                self._source_location = {} # Should the poition information be in the point source response? (HY)

                for name, source in point_sources.items():
                    coord = source.position.sky_coord
                
                    self._source_location[name] = copy.deepcopy(coord) # to avoid same memory issue

                    if self._coordsys == 'spacecraftframe':
                        dwell_time_map = self._get_dwell_time_map(coord)
                        self._psr[name] = self._dr.get_point_source_response(exposure_map=dwell_time_map)
                    elif self._coordsys == 'galactic':
                        scatt_map = self._get_scatt_map(coord)
                        self._psr[name] = self._dr.get_point_source_response(coord=coord, scatt_map=scatt_map)
                    else:
                        raise RuntimeError("Unknown coordinate system")

                    logger.info(f"--> done (source name : {name})")

                logger.info(f"--> all done")
        
        # check if the source location is updated or not
        for name, source in point_sources.items():

            if source.position.sky_coord != self._source_location[name]:
                logger.info(f"... Re-calculating the point source response of {name} ...")
                coord = source.position.sky_coord

                self._source_location[name] = copy.deepcopy(coord) # to avoid same memory issue
                
                if self._coordsys == 'spacecraftframe':
                    dwell_time_map = self._get_dwell_time_map(coord)
                    self._psr[name] = self._dr.get_point_source_response(exposure_map=dwell_time_map)
                elif self._coordsys == 'galactic':
                    scatt_map = self._get_scatt_map(coord)
                    self._psr[name] = self._dr.get_point_source_response(coord=coord, scatt_map=scatt_map)
                else:
                    raise RuntimeError("Unknown coordinate system")

                logger.info(f"--> done (source name : {name})")

        # Get expectation for point sources:
        for name,source in point_sources.items():

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
                        polarization_level = source.components['grb'].polarization.degree.value / 100.
                        polarization_angle = PolarizationAngle(coords.Angle(source.components['grb'].polarization.angle.value, unit=u.deg), source.position.sky_coord, convention=self._pa_convention)
                        if self._coordsys == 'spacecraftframe':
                            this_expectation = self._psr[name].get_expectation(spectrum, polarization_level, polarization_angle)
                        elif self._coordsys == 'galactic':
                            scatt_map = self._get_scatt_map(source.position.sky_coord)
                            this_expectation = self._psr[name].get_expectation(spectrum, polarization_level, polarization_angle, scatt_map, self._response_pa_convention)
                        else:
                            raise RuntimeError("Unknown coordinate system")

                    if component_counter == 0:
                        total_expectation = this_expectation
                    else:
                        total_expectation += this_expectation
                    
                    component_counter += 1

            # Save expected counts for source:
            self._expected_counts[name] = copy.deepcopy(total_expectation)
         
            # Need to check if self._signal type is dense (i.e. 'Quantity') or sparse (i.e. 'COO').
            if type(total_expectation.contents) == u.quantity.Quantity:
                total_expectation = total_expectation.project(['Em', 'Phi', 'PsiChi']).contents.value
            elif type(total_expectation.contents) == COO:
                total_expectation = total_expectation.project(['Em', 'Phi', 'PsiChi']).contents.todense() 
            else:
                raise RuntimeError("Expectation is an unknown object")

            # Add source to signal and update source counter:
            if self.src_counter == 0:
                self._signal = total_expectation
            if self.src_counter != 0:
                self._signal += total_expectation
            self.src_counter += 1

        # Cache
        self._model = model

    def get_log_like(self):
        """
        Calculate the log-likelihood.
        
        Returns
        ----------
        log_like : float
            Value of the log-likelihood
        """
        
        # Recompute the expectation if any parameter in the model changed
        if self._model is None:
            log.error("You need to set the model first")
       
        # Set model:
        self.set_model(self._model)
        
        # Compute expectation including free background parameter:
        if self._fit_nuisance_params: 
            if type(self._bkg.contents) == COO:
                expectation = self._signal + self._nuisance_parameters[self._bkg_par.name].value * self._bkg.contents.todense()
            else:
                expectation = self._signal + self._nuisance_parameters[self._bkg_par.name].value * self._bkg.contents
        
        # Compute expectation without background parameter:
        else: 
            if type(self._bkg.contents) == COO:
                expectation = self._signal + self._bkg.contents.todense()
            else:
                expectation = self._signal + self._bkg.contents

        expectation += 1e-12 # to avoid -infinite log-likelihood (occurs when expected counts = 0 but data != 0)
        if not self._printed_warning:
            logger.warning("Adding 1e-12 to each bin of the expectation to avoid log-likelihood = -inf.")
            self._printed_warning = True
        # This 1e-12 should be defined as a parameter in the near future (HY)
        
        # Convert data into an arrary:
        data = self._data.contents
        
        # Compute the log-likelihood:
        log_like = np.nansum(data*np.log(expectation) - expectation)
        
        return log_like
    
    def inner_fit(self):
        """
        Required for 3ML fit.
        """
        
        return self.get_log_like()
    
    def _get_dwell_time_map(self, coord):
        """
        Get the dwell time map of the source in the inertial (spacecraft) frame.
        
        Parameters
        ----------
        coord : astropy.coordinates.SkyCoord
            Coordinates of the target source
        
        Returns
        -------
        dwell_time_map : mhealpy.containers.healpix_map.HealpixMap
            Dwell time map
        """
        
        self._sc_orientation.get_target_in_sc_frame(target_name = self._name, target_coord = coord)
        dwell_time_map = self._sc_orientation.get_dwell_map(response = self._rsp_path)
        
        return dwell_time_map
    
    def _get_scatt_map(self, coord):
        """
        Get the spacecraft attitude map of the source in the inertial (spacecraft) frame.
        
        Parameters
        ----------
        coord : astropy.coordinates.SkyCoord
            The coordinates of the target object.

        Returns
        -------
        scatt_map : cosipy.spacecraftfile.scatt_map.SpacecraftAttitudeMap
        """
        
        scatt_map = self._sc_orientation.get_scatt_map(coord, nside = self._dr.nside * 2, \
                coordsys = 'galactic', earth_occ = self.earth_occ)
        
        return scatt_map
    
    def set_inner_minimization(self, flag: bool):
        """
        Turn on the minimization of the internal COSI (nuisance) parameters.
        
        Parameters
        ----------
        flag : bool
            Turns on and off the minimization  of the internal parameters
        """
        
        self._fit_nuisance_params: bool = bool(flag)

        for parameter in self._nuisance_parameters:
            self._nuisance_parameters[parameter].free = self._fit_nuisance_params
