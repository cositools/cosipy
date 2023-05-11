from threeML import PluginPrototype
from threeML.minimizer import minimization
from threeML.config.config import threeML_config
from threeML.exceptions.custom_exceptions import FitFailed
from astromodels import Parameter

from cosipy.response.FullDetectorResponse import FullDetectorResponse
from cosipy.coordinates.orientation import Orientation_file

from scoords import SpacecraftFrame, Attitude

from mhealpy import HealpixMap

import astropy.units as u

import numpy as np

from scipy.special import factorial

import collections

import copy

class COSILike(PluginPrototype):
    def __init__(self, name, dr, data, bkg, sc_orientation, nuisance_param=None, **kwargs):
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
        sc_orientation: array
            Pair of timestamps (astropy.Time) and attitudes (scoord.Attitude) that describe
            the orientation of the spacecraft for the duration of the data included in
            the analysis. Note: this will eventually be handled by the SC location and
            orientation module
        """
        
        # create the hash for the nuisance parameters. We have none for now.
        self._nuisance_parameters = collections.OrderedDict()

        # call the prototype constructor. Boilerplate.
        super(COSILike, self).__init__(name, self._nuisance_parameters)

        # User inputs needed to compute the likelihood
        self._name = name
        self._dr = FullDetectorResponse.open(dr)
        self._data = data
        self._bkg = bkg
        self._sc_orientation = sc_orientation
    
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
        
        Parameters:
            model: LikelihoodModel
                Any model supported by astromodel. However, this simple plugin only support single 
                point-sources with a power law spectrum
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
            
                dwell_time_map = self._get_dwell_time_map(coord)
            
                self._psr = self._dr.get_point_source_response(dwell_time_map)
                
            elif (source.position.sky_coord != self._source.position.sky_coord):
                
                coord = source.position.sky_coord
                
                dwell_time_map = self._get_dwell_time_map(coord)
                
                self._psr = self._dr.get_point_source_response(dwell_time_map)
                
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
        Return the value of the log-likelihood
        """
        
        # Recompute the expectation if any parameter in the model changed
        if self._model is None:
            log.error("You need to set the model first")
        
        self.set_model(self._model)
        
        if self._fit_nuisance_params: # Compute expectation including free background parameter
            expectation = self._signal.contents + self._nuisance_parameters[self._bkg_par.name].value * self._bkg.contents
        else: # Compute expectation without background parameter
            expectation = self._signal.contents.todense() + self._bkg.contents.todense()
        
        data = self._data.contents # Into an array
        
        # Compute the log-likelihood from the equations above
        #log_like = np.sum(np.log(np.power(expectation, data) * 
        #                         np.exp(-expectation) / 
        #                         factorial(data)))
        
        log_like = np.nansum(data*np.log(expectation) - expectation)
        
        return log_like
    
    def inner_fit(self):
        """
        This fits nuisance parameters.
        """
        
        return self.get_log_like()
    
    def _get_dwell_time_map(self, coord):
        """
        This will be eventually be provided by another module
        """
        
        # The dwell time map has the same pixelation (base) as the detector response.
        # We start with an empty map
        dwell_time_map = HealpixMap(base = self._dr, 
                                    unit = u.s, 
                                    coordsys = SpacecraftFrame())

        # Get timestamps and attitude values
        timestamps = self._sc_orientation.get_time()
        attitudes = self._sc_orientation.get_attitude().as_matrix()

        for attitude,duration in zip(attitudes[:-1], np.diff(timestamps.unix)):
            
            local_coord = coord.transform_to(SpacecraftFrame(attitude = Attitude.from_matrix(attitude)))
            
            # Here we add duration in between timestamps using interpolations
            pixels, weights = dwell_time_map.get_interp_weights(local_coord)

            for p,w in zip(pixels, weights):
                dwell_time_map[p] += w*(duration*u.s)
                
        return dwell_time_map
 
    def set_inner_minimization(self, flag: bool):
        """
        Turn on the minimization of the internal COSI parameters
        :param flag: turning on and off the minimization  of the internal parameters
        :type flag: bool
        :returns:
        """
        self._fit_nuisance_params: bool = bool(flag)

        for parameter in self._nuisance_parameters:
            self._nuisance_parameters[parameter].free = self._fit_nuisance_params
