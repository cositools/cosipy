from histpy import Axis

from mhealpy import HealpixBase

import numpy as np

from astropy.coordinates import SkyCoord, BaseRepresentation

class HealpixAxis(Axis, HealpixBase):

    def __init__(self,
                 nside,
                 edges = None,
                 scheme = 'ring',
                 coordsys = None,
                 label = None,
                 *args, **kwargs):

        HealpixBase.__init__(self,
                             nside = nside,
                             scheme = scheme,
                             coordsys = coordsys)

        if edges is None:
            # Default to full map
            edges = np.arange(self.npix + 1)
        
        super().__init__(edges,
                         label = label)
        
    def _sanitize_edges(self, edges):

        edges = super()._sanitize_edges(edges)

        # Check it corresponds to pixels
        if edges.dtype.kind not in 'ui':
            raise ValueError("HeapixAxis needs integer edges")

        if edges[0] < 0 or edges[-1] > self.npix+1:
            raise ValueError("Edges must be within 0 and the total number of pixels")

        return edges
        
    def __eq__(self, other):
        return self.conformable(other) and super().__eq__(other)

    def __getitem__(self, key):

        base_axis = super().__getitem__(key)

        return HealpixAxis(edges = base_axis,
                           nside = self.nside,
                           scheme = self.scheme,
                           coordsys = coordsys)

    def find_bin(self, value):

        if isinstance(value, (SkyCoord, BaseRepresentation)):
            # Transform first from coordinates to pixel
            value = self.ang2pix(value)

        return super().find_bin(value)

    def interp_weights(self, value):

        if isinstance(value, (SkyCoord, BaseRepresentation)):

            pixels, weights = self.get_interp_weights(value)

            return self.find_bin(pixels), weights

        else:
        
            return super().interp_weights(value)
    
    def _operation(self, key):
        raise AttributeError("HealpixAxis doesn't support operations")

    
    
    

    



