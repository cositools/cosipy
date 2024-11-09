from pathlib import Path
import itertools

import numpy as np
import astropy.units as u
from astropy.units import Quantity
from astropy.coordinates import (UnitSphericalRepresentation, SkyCoord,
                                 BaseRepresentation, CartesianRepresentation,
                                 BaseCoordinateFrame)

from histpy import Histogram, Axis, Axes, HealpixAxis
import mhealpy as hp
from mhealpy import HealpixBase, HealpixMap
from scoords import SpacecraftFrame, Attitude

class ListModeResponse(Histogram):
    """
    Handles nonlinear parametrizations of detector response
    and supports extensions of list mode analysis
    """

    def __init__(self, *args, **kwargs):
        # Overload parent init. Called in class methods.
        super().__init__(*args, **kwargs)
        self.mapping = {'Ei': 'Ei', 'Em': 'eps', 'Phi': 'Phi', 'PsiChi': 'PsiChi'}   # key_target : label

    def _get_all_interp_weights(self, target: dict):

        indices = []
        weights = []

        for key in target:
            label = self.mapping[key]
            axis = self.axes[label]
            axis_scale = axis._scale
            axis_type = str(type(axis)).split('.')[-1].strip("'>")      # XXX: Could probably be simplified using `isinstance()`

            # Scale
            if axis_scale in ['linear', 'log']:   # To ensure nonlinear binning parametrizations are converted to a linear scale.

                # Axis Type
                if axis_type in ['Axis', 'HealpixAxis']:
                    if key == label:    # If key and label are the same, then there was no reparametrization along this axis
                        idx, w = axis.interp_weights(target[key])
                    else:
                        centers = self.transform_eps_to_Em(axis.centers, target['Ei'])      # Transform coordinates to more physical units      # TODO: Generalize this
                        absdiff = np.abs(centers - target[key])                             # Calculate absolute difference to given target
                        idx = np.argpartition(absdiff, (1,2))[:2]                               # Find indices corresponding to two smallest absdiff
                        w = 1 - np.partition(absdiff, (1,2))[:2] / (centers[1] - centers[0])    # Calculate weights corresponding to two smallest absdiff
                else:
                    raise ValueError(f'Axis type: {axis_type} is not supported')
                
            # elif axis_scale == 'nonlinear':   
            #     pass

            else:
                raise ValueError(f'{axis_scale} binning / scale scheme is not supported')
            
            indices.append(idx)
            weights.append(w)
        
        return (indices, weights)
    
    def transform_eps_to_Em(self, eps, Ei0):
        return (eps + 1) * Ei0

    def transform_Em_to_eps(self, Em, Ei0):
        return Em/Ei0 - 1

    def get_nearest_neighbors(self, target: dict, indices=None):
        if indices is not None:
            neighbors = {}
            for idx, key in zip(indices, target):
                label = self.mapping[key]
                neighbors[label] = self.axes[label].centers[idx]
            return neighbors
        
        else:
            target = dict(sorted(target.items()))
            indices, _ = self._get_all_interp_weights(target)
            return self.get_nearest_neighbors(target, indices)

    def get_interp_response(self, target: dict):
        """
        Currently only supports nonlinear spectral responses (
        and for a particular parametrization)
        TODO: In the future, this will also support nonlinear / 
        piecewise-linear directional responses.
        """

        target = dict(sorted(target.items()))       # Sort dictionary by key (XXX: assuming response matrix also sorts in the same way)
        indices, weights = self._get_all_interp_weights(target)
        perm_indices = list(itertools.product(*indices))
        perm_weights = list(itertools.product(*weights))

        if len(target) == len(self.axes):
            interpolated_response_value = 0
            for idx, w in zip(perm_indices, perm_weights):
                interpolated_response_value += np.prod(w) * self.contents[idx]

        else:
            interpolated_response_value = np.zeros(len(self.axes['Ei']) - 1) * self.contents.unit     # XXX: Assuming all measured variables require interpolation
            for idx, w in zip(perm_indices, perm_weights):
                i = (Ellipsis,) + idx   # XXX: Assuming 'Ei' is the first index
                interpolated_response_value += np.prod(w) * self.contents[i]

        self.neighbors = self.get_nearest_neighbors(target, indices)
        
        return interpolated_response_value