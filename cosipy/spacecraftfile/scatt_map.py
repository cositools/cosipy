from histpy import Histogram, HealpixAxis

import astropy.units as u

class SpacecraftAttitudeMap(Histogram):

    def __init__(self,
                 nside,
                 scheme = 'ring',
                 coordsys = 'galactic',
                 labels = ['x', 'y']
                 ):
        """
        Bin the spacecraft attitude history into a 4D histogram that contains 
        the accumulated time the axes of the spacecraft where looking at a 
        given direction. 

        Same arguments as an HealpixAxis
        """

        super().__init__([HealpixAxis(nside = nside,
                                      scheme = scheme,
                                      coordsys = coordsys,
                                      label = labels[0]), 
                          HealpixAxis(nside = nside,
                                      scheme = scheme,
                                      coordsys = coordsys,
                                      label = labels[1])],
                         sparse = True,
                         unit = u.s)    

        
