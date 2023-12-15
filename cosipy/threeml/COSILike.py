from threeML import PluginPrototype
from threeML.minimizer import minimization
from threeML.config.config import threeML_config
from threeML.exceptions.custom_exceptions import FitFailed
from astromodels import Parameter

from cosipy.response.FullDetectorResponse import FullDetectorResponse
from cosipy.image_deconvolution import ModelMap

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

class COSILike(PluginPrototype):
    
    def __init__(self, name, dr, data, bkg, sc_orientation, 
            nuisance_param=None, coordsys=None, precomputed_psr_file=None, **kwargs):
        
        """
        COSI 3ML plugin
        
        Parameters
        ----------
        name : str
            Plugin name e.g. "cosi". Needs to have a distinct name with respect to other plugins in the same analysis
        dr : Path
            Path to full detector response
        data: histpy.Histogram
            Binned data. Note: Eventually this should be a cosipy data class
        bkg: histpy.Histogram
            Binned background model. Note: Eventually this should be a cosipy data class
        sc_orientation: cosipy.spacecraftfile.SpacecraftFile
            Contains the information of the orientation: timestamps (astropy.Time) and attitudes (scoord.Attitude) that describe
            the spacecraft for the duration of the data included in the analysis
        nuisance_param: astromodels.core.parameter.Parameter
            Background parameter (optional)
        coordsys: str
            Coordinate system ('galactic' or 'spacecraftframe') to perform fit in, which should match coordinate system of data 
            and background. This only needs to be specified if the binned data and background do not have a coordinate system 
            attached to them
        precomputed_psr_file: str
            Full path to precomputed point source response in Galactic coordinates (optional). 
        """
        
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

            print("... loading the pre-computed image response ...")
            self.image_response = DetectorResponse.open(self.precomputed_psr_file)
            # in the near future, we will implement ExtendedSourceResponse class, which should be used here.
            # probably, it is better to move this loading part outside of this class. Then, we don't have to load the response everytime we start the fitting.
            print("--> done")
        
    def set_model(self, model):
        
        """
        Set the model to be used in the joint minimization.
        
        Parameters
        ----------
        model: astromodels.core.model.Model; any model supported by astromodels
        """
    
        # Get point sources and extended sources from model: 
        point_sources = model.point_sources
        extended_sources = model.extended_sources
        
        # Source counter for models with multiple sources:
        self.src_counter = 0
        
       # Get expectation for extended sources:
        for name,source in extended_sources.items():

            # Set spectrum:
            # Note: the spectral parameters are updated internally by 3ML
            # during the likelihood scan. 

            model_map = ModelMap(nside = self.image_response.axes['NuLambda'].nside, 
                                energy_edges = self.image_response.axes['Ei'].edges,
                                coordsys = 'galactic',
                                label_image = 'NuLambda', # I think the label should be something like 'lb' to distiguish the photon direction in the local/galactic coordinates 
                                label_energy = 'Ei')

            model_map.set_values_from_extendedmodel(source)

            # Get expectation using precomputed psr in Galactic coordinates:
            total_expectation = Histogram(edges = self.image_response.axes[2:],
                                          contents = np.tensordot(model_map.contents, self.image_response.contents, axes = ([0,1], [0,1])) * model_map.axes['NuLambda'].pixarea())
            # ['NuLambda', 'Ei'] x ['NuLambda', 'Ei', 'Em', 'Phi', 'PsiChi'] => 'Em', 'Phi', 'PsiChi']
            # this part should be modified with the future ExtendedSourceResponse class like
            # total_expectation = self.image_response.get_expectation(model_map)

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

        # Get expectation for point sources:
        for name,source in point_sources.items():

            if self._source is None:
                self._source = copy.deepcopy(source) # to avoid same memory issue
                     
            # Compute point source response for source position
            # See also the Detector Response and Source Injector tutorials
            if self._psr is None:
            
                coord = self._source.position.sky_coord
                
                if self._coordsys == 'spacecraftframe':
                    dwell_time_map = self._get_dwell_time_map(coord)
                    self._psr = self._dr.get_point_source_response(exposure_map=dwell_time_map)
                elif self._coordsys == 'galactic':
                    scatt_map = self._get_scatt_map()
                    self._psr = self._dr.get_point_source_response(coord=coord, scatt_map=scatt_map)
                else:
                    raise RuntimeError("Unknown coordinate system")
                
            elif (source.position.sky_coord != self._source.position.sky_coord):
                
                coord = source.position.sky_coord
                
                if self._coordsys == 'spacecraftframe':
                    dwell_time_map = self._get_dwell_time_map(coord)
                    self._psr = self._dr.get_point_source_response(exposure_map=dwell_time_map)
                elif self._coordsys == 'galactic':
                    scatt_map = self._get_scatt_map()
                    self._psr = self._dr.get_point_source_response(coord=coord, scatt_map=scatt_map)
                else:
                    raise RuntimeError("Unknown coordinate system")
                
            # Caching source to self._source after position judgment
            if self._source is not None:
                self._source = copy.deepcopy(source)

            # Convolve with spectrum
            # See also the Detector Response and Source Injector tutorials
            spectrum = source.spectrum.main.shape
                
            total_expectation = self._psr.get_expectation(spectrum).project(['Em', 'Phi', 'PsiChi'])
         
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

        # Cache
        self._model = model

    def get_log_like(self):
        
        """
        Return the value of the log-likelihood.
        
        Returns
        ----------
        log_like: float
        """
        
        # Recompute the expectation if any parameter in the model changed
        if self._model is None:
            log.error("You need to set the model first")
       
        # Set model:
        self.set_model(self._model)
        
        # Compute expectation including free background parameter:
        if self._fit_nuisance_params: 
            expectation = self._signal + self._nuisance_parameters[self._bkg_par.name].value * self._bkg.contents.todense()
        
        # Compute expectation without background parameter:
        else: 
            expectation = self._signal + self._bkg.contents.todense()
        
        # Convert data into an arrary:
        data = self._data.contents
        
        # Compute the log-likelihood:
        log_like = np.nansum(data*np.log(expectation) - expectation)
        
        # Need to mask zero-values pixels if obtaining infinite likelihood.
        # Note: the mask function gives errors sometimes. This is a bug that needs to be fixed. 
        if log_like == -np.inf:
            logger.warning(f"There are bins in which the total expected counts = 0 but data != 0, making log-likelihood = -inf. "
                           f"Masking these bins.")
            log_like = np.nansum(np.ma.masked_invalid(data*np.log(expectation) - expectation))
        
        return log_like
    
    def inner_fit(self):
        
        """
        This fits nuisance parameters.
        """
        
        return self.get_log_like()
    
    def _get_dwell_time_map(self, coord):
        
        """
        Get the dwell time map of the source in the spacecraft frame.
        
        Parameters
        ----------
        coord: astropy.coordinates.SkyCoord; the coordinate of the target source.
        
        Returns
        -------
        dwell_time_map: mhealpy.containers.healpix_map.HealpixMap
        """
        
        self._sc_orientation.get_target_in_sc_frame(target_name = self._name, target_coord = coord)
        dwell_time_map = self._sc_orientation.get_dwell_map(response = self._rsp_path)
        
        return dwell_time_map
    
    def _get_scatt_map(self):
        
        """
        Get the spacecraft attitude map of the source in the inertial (spacecraft) frame.
        
        Parameters
        ----------
        nside: int; resolution of scatt map
        coordsys: BaseFrameRepresentation or str; coordinate system of map
        
        Returns
        -------
        scatt_map: cosipy.spacecraftfile.scatt_map.SpacecraftAttitudeMap
        """
        
        scatt_map = self._sc_orientation.get_scatt_map(nside = self._dr.nside * 2, coordsys = 'galactic')
        
        return scatt_map
    
    def set_inner_minimization(self, flag: bool):
       
        """
        Turn on the minimization of the internal COSI parameters.
        
        Parameters
        ----------
        flag: bool; turns on and off the minimization  of the internal parameters
        """
        
        self._fit_nuisance_params: bool = bool(flag)

        for parameter in self._nuisance_parameters:
            self._nuisance_parameters[parameter].free = self._fit_nuisance_params
