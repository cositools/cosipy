import numpy as np

import logging
logger = logging.getLogger(__name__)

class ResponseWeightingFilter:
    """
    Response weighting filter.
    
    This class calculates and stores a response weighting filter
    based on the exposure map. The filter renormalizes the delta map
    so that pixels with large exposure times have more feedback.
    """
    
    def __init__(self, exposure_map, index=0.5):
        """
        Initialize response weighting filter.
        
        Parameters
        ----------
        exposure_map : Histogram
            Summed exposure map
        index : float, optional
            Response weighting index (default: 0.5)
        """

        self.exposure_map = exposure_map
        self.index = index
        self.filter = self._calculate_filter()
        
        logger.info(f"[Response weighting filter created (index={index})]")
    
    def _calculate_filter(self):
        """
        Calculate the filter array.
        """

        max_exposure = np.max(self.exposure_map.contents)
        if max_exposure == 0:
            logger.warning("Maximum exposure is zero")
            return np.ones_like(self.exposure_map.contents)
        
        return (self.exposure_map.contents / max_exposure)**self.index
    
    def apply(self, delta_model):
        """
        Apply filter to delta model.
        
        Parameters
        ----------
        delta_model : Histogram
            Delta model to be weighted
        
        Returns
        -------
        Histogram
            Weighted delta model (delta_model * filter)
        """

        return delta_model * self.filter
