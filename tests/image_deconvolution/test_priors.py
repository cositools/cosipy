import pytest

import astropy.units as u
import numpy as np
import healpy as hp

from cosipy.image_deconvolution.algorithms.prior_tsv import PriorTSV
from cosipy.image_deconvolution.algorithms.prior_entropy import PriorEntropy
from cosipy.image_deconvolution import AllSkyImageModel

def test_PriorTSV():

    parameter = {'coefficient': 1.0}
    
    nside = 1
    allskyimage_model = AllSkyImageModel(nside = nside, 
                                         energy_edges = np.array([500.0, 510.0]) * u.keV)
    allskyimage_model[:,0] = np.arange(hp.nside2npix(nside)) * allskyimage_model.unit

    prior_tsv = PriorTSV(parameter, allskyimage_model)
    
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

def test_PriorEntropy():

    parameter = {'coefficient': 1.0, 'reference_map': {'value': 1.0, 'unit': 'cm-2 s-1 sr-1'}}
    
    nside = 1
    allskyimage_model = AllSkyImageModel(nside = nside, 
                                         energy_edges = np.array([500.0, 510.0]) * u.keV)
    allskyimage_model[:,0] = (np.arange(hp.nside2npix(nside)) + 1 ) * allskyimage_model.unit

    prior_entropy = PriorEntropy(parameter, allskyimage_model)
    
    assert np.isclose(prior_entropy.log_prior(allskyimage_model), -80.27855835017301)

    grad_log_prior_correct = np.array([[-0.        ],
                                       [-0.69314718],
                                       [-1.09861229],
                                       [-1.38629436],
                                       [-1.60943791],
                                       [-1.79175947],
                                       [-1.94591015],
                                       [-2.07944154],
                                       [-2.19722458],
                                       [-2.30258509],
                                       [-2.39789527],
                                       [-2.48490665]]) * u.Unit('cm2 s sr')

    assert np.allclose(prior_entropy.grad_log_prior(allskyimage_model), grad_log_prior_correct)
