from threeML import PluginPrototype

from cosipy.response.FullDetectorResponse import FullDetectorResponse
from cosipy.coordinates import SpacecraftFrame

from mhealpy import HealpixMap

import astropy.units as u

import numpy as np

from scipy.special import factorial

import collections

class COSILike(PluginPrototype):
    def __init__(self, name, dr, data, bkg, sc_orientation, **kwargs):
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
        nuisance_parameters = collections.OrderedDict()

        # call the prototype constructor. Boilerplate.
        super(COSILike, self).__init__(name, nuisance_parameters)

        # User inputs needed to compute the likelihood
        self._dr = FullDetectorResponse.open(dr)
        self._data = data
        self._bkg = bkg
        self._sc_orientation = sc_orientation
    
        # Place-holder for cached data.
        self._model = None
        self._source = None
        self._psr = None
        self._signal = None
        
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
                self._source = source
                     
            # Compute point source response for source position
            # See also the Detector Response and Source Injector tutorials
            if self._psr is None:
            
                coord = self._source.position.sky_coord
            
                dwell_time_map = self._get_dwell_time_map(coord)
            
                self._psr = self._dr.get_point_source_response(dwell_time_map)
            
            elif source.position != self._source.position:
                
                raise RuntimeError("No change in position for now")

            # Convolve with spectrum
            # See also the Detector Response and Source Injector tutorials
            spectrum = source.spectrum.main.shape
                
            self._signal = self._psr.get_expectation(spectrum).project(['Em', 'Phi', 'PsiChi'])
            
        # Cache
        self._model = model

    def get_log_like(self):

        # Recompute the expectation if any parameter in the model changed
        if self._model is None:
            log.error("You need to set the model first")
        
        self.set_model(self._model)
        
        # Compute "lambda" in the equations above
        expectation = self._signal.contents + self._bkg.contents
        
        data = self._data.contents # Into an array
        
        # Compute the log-likelihood from the equations above
        log_like = np.sum(np.log(np.power(expectation, data) * 
                             np.exp(-expectation) / 
                             factorial(data)))
        
        return log_like

    def inner_fit(self):
        """
        This fits nuisance parameters, but we have none for now.
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
        timestamps, attitudes = zip(*self._sc_orientation)
            
        for attitude,duration in zip(attitudes[:-1], np.diff(timestamps)):

            local_coord = coord.transform_to(SpacecraftFrame(attitude = attitude))

            # Here we add duration in between timestamps using interpolations
            pixels, weights = dwell_time_map.get_interp_weights(local_coord)

            for p,w in zip(pixels, weights):
                dwell_time_map[p] += w*duration.to(u.s)
                
        return dwell_time_map