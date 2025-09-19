import pytest

import astropy.units as u
import numpy as np
import healpy as hp

from cosipy.image_deconvolution.prior_tsv import PriorTSV
from cosipy.image_deconvolution import AllSkyImageModel

def test_PriorTSV():

    coefficient = 1.0
    
    nside = 1
    allskyimage_model = AllSkyImageModel(nside = nside, 
                                         energy_edges = np.array([500.0, 510.0]) * u.keV)
    allskyimage_model[:,0] = np.arange(hp.nside2npix(nside)) * allskyimage_model.unit

    prior_tsv = PriorTSV(coefficient, allskyimage_model)
    
    assert np.isclose(prior_tsv.log_prior(allskyimage_model), -1176.0)

    grad_log_prior_correct = np.array([[  92.],
                                       [  76.],
                                       [  60.],
                                       [  28.],
                                       [  40.],
                                       [  -8.],
                                       [  -8.],
                                       [ -24.],
                                       [ -36.],
                                       [ -52.],
                                       [ -68.],
                                       [-100.]]) * u.Unit('cm2 s sr')

    assert np.allclose(prior_tsv.grad_log_prior(allskyimage_model), grad_log_prior_correct)
