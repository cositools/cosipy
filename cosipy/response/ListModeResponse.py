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
    
    def _get_all_interp_weights(self, target: dict):

        indices = []
        weights = []

        for label in self.axes.labels:
            axis = self.axes[label]
            axis_scale = axis._scale
            axis_type = str(type(axis)).split('.')[-1].strip("'>")      # XXX: Could probably be simplified using `isinstance()`

            # Scale
            if axis_scale in ['linear', 'log']:   # To ensure nonlinear binning parametrizations are converted to a linear scale.

                # Axis Type
                if axis_type in ['Axis', 'HealpixAxis']:
                    idx, w = axis.interp_weights(target[label])
                else:
                    raise ValueError(f'Axis type: {axis_type} is not supported')
                
            elif axis_scale == 'nonlinear':   
                pass

            else:
                raise ValueError(f'Scale: {axis_scale} is not supported')
            
            indices.append(idx)
            weights.append(w)
        
        return (indices, weights)
    
    def transform_eps_to_Em(self, eps, Ei0):
        # return (eps + 1) * Ei0
        return eps

    def transform_Em_to_eps(self, Em, Ei0):
        # return Em/Ei0 - 1
        return Em

    def _create_nd_array(self):
        shape = tuple([2] * self.ndim)
        array = np.zeros(2**self.ndim).reshape(shape)
        return array
    
    # def _standarize_theta_phi_lonlat(self, theta, phi, lonlat):

    #     if isinstance(theta, (SkyCoord, BaseRepresentation)):
    #         # Support astropy
            
    #         if isinstance(theta, SkyCoord):

    #             if self.coordsys is None:
    #                 raise ValueError("Undefined coordinate system")
                
    #             theta = theta.transform_to(self.coordsys)
        
    #         coord = theta.represent_as(UnitSphericalRepresentation)

    #         theta,phi = coord.lon.deg, coord.lat.deg

    #         lonlat = True

    #     return theta,phi,lonlat
    
    # def get_interp_weights(self, theta, phi = None, lonlat = False):
    #     """
    #     Return the 4 closest pixels on the two rings above and below the 
    #     location and corresponding weights. Weights are provided for bilinear 
    #     interpolation along latitude and longitude

    #     Args:
    #         theta (float or array): Zenith angle (rad)
    #         phi (float or array): Azimuth angle (rad)
 
    #     Return:
    #         tuple: (pixels, weights), each with of (4,) if the input is scalar,
    #             if (4,N) where N is size of
    #             theta and phi. For MOC maps, these pixel numbers might repeate.
    #     """

    #     theta, phi, lonlat = self._standarize_theta_phi_lonlat(theta, phi, lonlat)

    #     pixels,weights = hp.get_interp_weights(self.nside, theta, phi,
    #                                            nest = self.is_nested,
    #                                            lonlat = lonlat)

    #     if self.is_moc:
    #         pixels = self.nest2pix(pixels)

    #     return (pixels, weights)

    def get_neighbors(self, indices):
        return [axis.centers[idx] for idx, axis in zip(indices, self.axes)]

    def get_interp_response(self, target: dict):
        """
        Currently only supports nonlinear spectral responses (
        and for a particular parametrization)
        TODO: In the future, this will also support nonlinear / 
        piecewise-linear directional responses.
        XXX: To get the correct interpolated response, ensure 
        all scales passed into this function are in linear scale
        """

        # centers = []
        # axis_types = []
        # for axis in self.axes.labels:
        #     scale = self.axes[axis]._scale
        #     print(scale)
        #     axis_type = str(type(self.axes[axis])).split('.')[-1].strip("'>")
        #     print(axis_type)

        #     # Scale
        #     if scale in ['linear', 'log']:   # To ensure nonlinear binning parametrizations are converted to a linear scale.

        #         # Axis Type
        #         if axis_type == 'HealpixAxis':
        #             centers.append(self.axes[axis].centers) # TODO: HealpixAxis type? Will this be different?
        #         elif axis_type == 'Axis':
        #             centers.append(self.axes[axis].centers)
        #         else:
        #             raise ValueError(f'Axis type: {axis_type} is not supported')
                
        #     elif scale == 'nonlinear':    # XXX: For now, eps_to_Em is the only nonlinear transformation that has been implemented
        #         # This "nonlinear" transformation is still represented on a linear scale. So need to find a different attribute to use for `if <> == 'nonlinear'` comparison
        #         this_edges = self.transform_eps_to_Em(self.axes[axis].edges, target['Ei'])
        #         this_centers = (this_edges[:-1] + this_edges[1:]) / 2
        #         centers.append(this_centers)

            # elif scale == 'log':

            #     this_centers = np.log2(self.axes[axis].centers.value)       # I chose log2 instead of ln as the former was used in `histpy.axis` too. Also see https://stackoverflow.com/questions/33809789/why-are-log2-and-log1p-so-much-faster-than-log-and-log10-in-numpy
            #     centers.append(this_centers)

        #     else:
        #         raise ValueError(f'Scale: {scale} is not supported')

        # indices = self._get_nearest_neighbors(centers, target)
        indices, weights = self._get_all_interp_weights(target)
        perm_indices = list(itertools.product(*indices))
        perm_weights = list(itertools.product(*weights))
        interpolated_response_value = 0
        for idx, w in zip(perm_indices, perm_weights):
            interpolated_response_value += np.prod(w) * self.contents[idx]

        self.neighbors = self.get_neighbors(indices)
        
        return interpolated_response_value

        # interpolated_response_value = 0
        # for i, axis in enumerate(self.axes):
        #     for j, idx in enumerate(indices[i]):
        #         print(axis.centers[idx])
        #         print(weights[i][j])
        #         interpolated_response_value += weights[i][j] * axis.centers[idx]

        # return interpolated_response_value

        # # Initialize neighbors and dists
        # neighbors = [centers[i][indices[i]] for i in range(self.ndim)]
        # dists = [np.diff(neighbors[i]) for i in range(self.ndim)]       # Only linear dimensions should be used to calculate distance measures. 

        # # Assign to self.neighbors
        # self.neighbors = neighbors

        # # Convert indices to a numpy array
        # indices = np.array(indices)

        # # Initialize fQ with zeros
        # fQ = np.zeros(2 ** self.ndim) * self.contents.unit

        # # Generate permutations and fill fQ
        # permutations = list(itertools.product(*indices))
        # for j, perm in enumerate(permutations):
        #     fQ[j] = self.contents[perm]
        # # fQ = np.array([self.contents[perm] for perm in permutations])

        # # Reshape fQ
        # fQ = fQ.reshape([2] * self.ndim)
        
        # t = np.where(dists == 0, 0, [(target[key] - neighbors[i][0]) / dists[i] for i, key in enumerate(target)])

        # # Compute the bilinearly interpolated response value for multidimensional interpolation
        # interpolated_response_value = 0
        # fQ_flat = fQ.flatten('F')[::-1]

        # for idx in range(2**self.ndim):
        #     weight = np.prod([1 - t[dim] if (idx >> dim) & 1 else t[dim] for dim in range(self.ndim)])
        #     interpolated_response_value += fQ_flat[idx] * weight

        # print(f'Multidimensional interpolated value: {interpolated_response_value}')

        # return interpolated_response_value