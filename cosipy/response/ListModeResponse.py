from pathlib import Path
import itertools
from copy import deepcopy

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

    def __init__(self, **kwargs):
        # Overload parent init. Called in class methods.
        super().__init__(**kwargs)
        
        self.mapping = {'Ei': 'Ei', 'Em': 'eps', 'Phi': 'Phi', 'PsiChi': 'PsiChi'}   # key_target : label

        self._spec = None
        self._aeff = None

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
    
    def get_spectral_response(self, copy = True):
        """
        Reduced detector response, projected along the real and measured energy axes only.
        The Compton Data Space axes are not included.

        Parameters
        ----------
        copy : bool
            If true, a copy of the cached spectral response will be returned.
        
        Returns
        -------
        :py:class:`DetectorResponse`
        """

        # Cache the spectral response
        if self._spec is None:
            spec = self.project(['Ei','Em'])
            self._spec = ListModeResponse(spec.axes,
                                          contents = spec.contents,
                                          unit = spec.unit)

        if copy:
            return deepcopy(self._spec)
        else:
            return self._spec
        
    def get_effective_area(self, energy = None, copy = True):
        """
        Compute the effective area at a given energy. If no energy is specified, the
        output is a histogram for the effective area at each energy bin.

        Parameters
        ----------
        energy : optional, :py:class:`astropy.units.Quantity`
            Energy/energies at which to interpolate the linearly effective area
        copy : bool
            If true, a copy of the cached effective will be returned.
        
        Returns
        -------
        :py:class:`astropy.units.Quantity` or :py:class:`histpy.Histogram`
        """
        
        if self._aeff is None:
            self._aeff = self.get_spectral_response(copy = False).project('Ei').to_dense()

        if energy is None:
            if copy:
                return deepcopy(self._aeff)
            else:
                return self._aeff
        else:
            return self._aeff.interp(energy)

    def get_dispersion_matrix(self):
        """
        Compute the energy dispersion matrix, also known as migration matrix. This holds the
        probability of an event with real energy ``Ei`` to be reconstructed with an measured
        energy ``Em``.

        Returns
        -------
        :py:class:`histpy.Histogram`
        """
        
        # Get spectral response and effective area normalization
        spec = self.get_spectral_response(copy = False)
        norm = self.get_effective_area().contents

        # "Broadcast" such that it has the compatible dimensions with the 2D matrix
        norm = spec.expand_dims(norm, 'Ei')
        
        # Normalize column-by-column
        return (spec / norm)    # XXX: Runtime warning needs to be dealt with in histpy. 
                                # Overflow and underflow bins functionality should be removed.

    @property
    def photon_energy_axis(self):
        """
        Real energy bins (``Ei``).

        Returns
        -------
        :py:class:`histpy.Axes`
        """
        
        return self.axes['Ei']

    
    @property
    def measured_energy_axis(self):
        """
        Measured energy bins (``Em``).

        Returns
        -------
        :py:class:`histpy.Axes`        
        """
        
        return self.axes['Em']