from threeML import PluginPrototype
from threeML.minimizer import minimization
from threeML.config.config import threeML_config
from threeML.exceptions.custom_exceptions import FitFailed
from astromodels import Parameter

from cosipy.response.FullDetectorResponse import FullDetectorResponse

from scoords import SpacecraftFrame, Attitude

from mhealpy import HealpixMap

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
    
    def __init__(self, name, dr, data, bkg, sc_orientation, nuisance_param=None, coordsys=None, **kwargs):
        
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
            It contains the information of the orientation: timestamps (astropy.Time) and attitudes (scoord.Attitude) that describe
            the spacecraft for the duration of the data included in the analysis.
            orientation module
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
                logger.warning("Data is binned in " + data.axes["PsiChi"].coordsys.name + " and background is binned in " 
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
        
    def set_model(self, model):
        
        """
        Set the model to be used in the joint minimization.
        
        Parameters
        ----------
        model: astromodels.core.model.Model; any model supported by astromodels
        """
        
        # Check for limitations
        if len(model.extended_sources) != 0 or len(model.particle_sources):
            raise RuntimeError("Only point source models are supported")
        
        sources = model.point_sources
        
        if len(sources) != 1:
            raise RuntimeError("Only one for now")
        
        # Get expectation
        for name,source in sources.items():

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
                
            self._signal = self._psr.get_expectation(spectrum).project(['Em', 'Phi', 'PsiChi'])
            
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
        
        self.set_model(self._model)
        
        if type(self._signal.contents) == u.quantity.Quantity:
            signal = self._signal.contents.value
        elif type(self._signal.contents) == COO:
            signal = self._signal.contents.todense()
        else:
            raise RuntimeError("Expectation is an unknown object")
            
        if self._fit_nuisance_params: # Compute expectation including free background parameter
            expectation = signal + self._nuisance_parameters[self._bkg_par.name].value * self._bkg.contents.todense()
        else: # Compute expectation without background parameter
            expectation = signal + self._bkg.contents.todense()
        
        data = self._data.contents # Into an array
        
        # Compute the log-likelihood
        
        log_like = np.nansum(data*np.log(expectation) - expectation)
        
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
