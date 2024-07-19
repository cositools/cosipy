import healpy as hp
import numpy as np

from .prior_base import PriorBase

from .allskyimage import AllSkyImageModel

class PriorTSV(PriorBase):

    allowerd_model_class = [AllSkyImageModel]

    def __init__(self, coefficient, model):

        super().__init__(coefficient, model)

        if self.model_class == AllSkyImageModel:

            nside = model.axes['lb'].nside
            npix  = model.axes['lb'].npix
            nest = False if model.axes['lb'].scheme == 'RING' else True

            theta, phi = hp.pix2ang(nside = nside, ipix = np.arange(npix), nest = nest)

            self.neighbour_pixel_index = hp.get_all_neighbours(nside = nside, theta = theta, phi = phi, nest = nest)

    def log_prior(self, model):

        if self.model_class == AllSkyImageModel:

            diff = (model[:] - model[self.neighbour_pixel_index]).value
            diff[np.isnan(diff)] = 0

            return self.coefficient * np.sum(diff**2, axis = 0)
    
    def grad_log_prior(self, model): 

        if self.model_class == AllSkyImageModel:

            diff = (model[:] - model[self.neighbour_pixel_index]).value
            diff[np.isnan(diff)] = 0

            return self.coefficient * 4 * np.sum(diff, axis = 0) / model.unit
