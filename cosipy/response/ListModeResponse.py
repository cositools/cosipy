from pathlib import Path
import itertools

import numpy as np
import astropy.units as u
from astropy.units import Quantity

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

    def _get_nearest_neighbors(self, centers, target: dict):
        """
        Given n-dimensional axes, identify the indices of the nearest neighbors.
        Ensures there are at least 2 dimensions.
        """
        if len(centers) < 2:
            raise ValueError("At least 2 dimensions are required")
        
        if len(centers) != len(target):
            raise ValueError("Dimensions of centers and target must be equal")

        indices = []
        for dim_centers, key_target in zip(centers, target):
            dim_target = target[key_target]
            dim_index = np.sort(np.argpartition(np.abs(dim_centers - dim_target), 1)[:2]).tolist()
            indices.append(dim_index)

        # for i, dim_centers in enumerate(centers):
        #     print(f"Dimension {i} centers: {dim_centers}")

        # for i, dim_indices in enumerate(indices):
        #     print(f"Dimension {i} indices: {dim_indices}")

        return indices
    
    def transform_eps_to_Em(self, eps, Ei0):
        return (eps + 1) * Ei0

    def transform_Em_to_eps(self, Em, Ei0):
        return Em/Ei0 - 1

    def _create_nd_array(self):
        shape = tuple([2] * self.ndim)
        array = np.zeros(2**self.ndim).reshape(shape)
        return array

    def get_interp_response(self, target: dict):
        """
        Currently only supports nonlinear spectral responses (
        and for a particular parametrization)
        TODO: In the future, this will also support nonlinear / 
        piecewise-linear directional responses.
        """

        centers = []
        for axis in self.axes.labels:
            if axis == 'eps':
                Em_centers = self.transform_eps_to_Em(self.axes[axis].centers, target['Ei'])
                centers.append(Em_centers)
            else:
                centers.append(self.axes[axis].centers)

        # Ei_centers = self.axes['Ei'].centers
        # eps_centers = self.axes['eps'].centers    # TODO: Does this make sense? As eps is nonlinearly binned

        indices = self._get_nearest_neighbors(centers, target)

        xindex = indices[0]
        yindex = indices[1]

        x1, x2 = self.axes['Ei'].centers[xindex]
        y1, y2 = Em_centers[yindex]
        xdist = x2 - x1
        ydist = y2 - y1

        fQ00 = self.contents[xindex[0], yindex[0]]
        fQ01 = self.contents[xindex[0], yindex[1]]
        fQ10 = self.contents[xindex[1], yindex[0]]
        fQ11 = self.contents[xindex[1], yindex[1]]

        tx = (target['Ei'] - x1) / xdist if xdist != 0 else 0
        ty = (target['Em'] - y1) / ydist if ydist != 0 else 0

        interpolated_response_value = (fQ00 * (1 - tx) * (1 - ty) + 
                            fQ10 * tx * (1 - ty) + 
                            fQ01 * (1 - tx) * ty + 
                            fQ11 * tx * ty)
        
        print(f'Bilinear interpolated value: {interpolated_response_value}')
        
        # neighbors = []
        # dists = []
        # for i in range(self.ndim):
        #     neighbors.append(centers[i][indices[i]])
        #     dists.append(np.diff(neighbors[-1]))

        # Initialize neighbors and dists
        neighbors = [centers[i][indices[i]] for i in range(self.ndim)]
        dists = [np.diff(neighbors[i]) for i in range(self.ndim)]

        # Assign to self.neighbors
        self.neighbors = neighbors

        # Convert indices to a numpy array
        indices = np.array(indices)

        # Initialize fQ with zeros
        fQ = np.zeros(2 ** self.ndim) * self.contents.unit

        # Generate permutations and fill fQ
        permutations = list(itertools.product(*indices))
        for j, perm in enumerate(permutations):
            fQ[j] = self.contents[perm]
        # fQ = np.array([self.contents[perm] for perm in permutations])

        # Reshape fQ
        fQ = fQ.reshape([2] * self.ndim)
        
        t = np.where(dists == 0, 0, [(target[key] - neighbors[i][0]) / dists[i] for i, key in enumerate(target)])

        # Compute the interpolated response value for multidimensional interpolation
        # TODO: May / may not break for higher dimensions
        interpolated_response_value = 0
        fQ_flat = fQ.flatten('F')[::-1]

        for idx in range(2**self.ndim):
            weight = np.prod([1 - t[dim] if (idx >> dim) & 1 else t[dim] for dim in range(self.ndim)])
            interpolated_response_value += fQ_flat[idx] * weight

        print(f'Multidimensional interpolated value: {interpolated_response_value}')

        # eps_centers = self.axes['eps'].centers
        # print(x1, y1, eps_centers[yindex[0]])
        # print(x1, y2, eps_centers[yindex[1]])
        # print(x2, y1, eps_centers[yindex[0]])
        # print(x2, y2, eps_centers[yindex[1]])
        # print(fQ00, fQ01, fQ10, fQ11)
        # print(fQ)
        # print(xdist, ydist)
        # print(dists)

        return interpolated_response_value