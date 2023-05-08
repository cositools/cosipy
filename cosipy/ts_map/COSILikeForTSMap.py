from cosipy.threeml.COSILike import COSILike

import copy


class COSILikeForTSMap(COSILike):
    
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
        for name, source in sources.items():

            if self._source is None:
                self._source = copy.deepcopy(source) # to avoid same memory issue
                     
            # Compute point source response for source position
            # See also the Detector Response and Source Injector tutorials
            if self._psr is None:
            
                coord = self._source.position.sky_coord
            
                dwell_time_map = self._get_dwell_time_map(coord)
            
                self._psr = self._dr.get_point_source_response(dwell_time_map)
                
            elif (source.position.ra._internal_value != self._source.position.ra._internal_value) or\
            (source.position.dec._internal_value != self._source.position.dec._internal_value):
                
                print('position change!')
                
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
        self._model = model # They share the same memory!
        