import healpy as hp
import numpy as np

from .prior_base import PriorBase

from .allskyimage import AllSkyImageModel

class PriorTSV(PriorBase):

    usable_model_classes = [AllSkyImageModel]

    def __init__(self, coefficient, model):

        super().__init__(coefficient, model)

        if self.model_class == AllSkyImageModel:

            nside = model.axes['lb'].nside
            npix  = model.axes['lb'].npix
            nest = False if model.axes['lb'].scheme == 'RING' else True

            theta, phi = hp.pix2ang(nside = nside, ipix = np.arange(npix), nest = nest)

            self.neighbour_pixel_index = hp.get_all_neighbours(nside = nside, theta = theta, phi = phi, nest = nest) # Its shape is (8, num. of pixels)

            self.num_neighbour_pixels = np.sum(self.neighbour_pixel_index >= 0, axis = 0) # Its shape is (num. of pixels)
            
            # replace -1 with its pixel index
            for idx, ipixel in np.argwhere(self.neighbour_pixel_index == -1):
                self.neighbour_pixel_index[idx, ipixel] = ipixel

    def log_prior(self, model):

        if self.model_class == AllSkyImageModel:

            diff = (model[:] - model[self.neighbour_pixel_index]).value
            # Its shape is (8, num. of pixels, num. of energies)
            diff /= self.num_neighbour_pixels.reshape((1,-1,1))

            return -1.0 * self.coefficient * np.sum(diff**2)
    
    def grad_log_prior(self, model): 

        if self.model_class == AllSkyImageModel:

            diff = (model[:] - model[self.neighbour_pixel_index]).value
            diff /= self.num_neighbour_pixels.reshape((1,-1,1))

            return -1.0 * self.coefficient * 4 * np.sum(diff, axis = 0) / model.unit
